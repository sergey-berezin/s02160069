using System;
using System.Threading;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.IO;
using System.Linq;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.PixelFormats;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxImageProcessor
{
    public abstract class WorkerResult { public string ThreadName { get; set; } }

    public abstract class WorkerFinished : WorkerResult { }
    public class WorkerSucceeded : WorkerFinished { }
    public class WorkerInterrupted : WorkerFinished { }
    public class AllWorkersFinished : WorkerFinished { }

    public class RecognitionResult : WorkerResult
    {
        public string FileName { get; private set; }
        public double[] Probs { get; private set; }
        public RecognitionResult(string threadName, string fileName, double[] probs)
        {
            if (probs.Length != 10)
                throw new ArgumentException("probs length must be equal to 10");
            ThreadName = threadName;
            FileName = fileName;
            Probs = probs;
        }
    }

    public class DirectoryProcessor : IDisposable
    {
        private bool disposedValue;
        private InferenceSession session;

        private ConcurrentQueue<string> filenames = new ConcurrentQueue<string>();
        private ConcurrentQueue<WorkerResult> results = new ConcurrentQueue<WorkerResult>();
        private ManualResetEvent stopper = new ManualResetEvent(true);
        private AutoResetEvent outputMutex = new AutoResetEvent(false);

        private Action<WorkerResult> callback;

        public DirectoryProcessor(string[] files, Action<WorkerResult> callback)
        {
            session = new InferenceSession("mnist-8.onnx");
            Array.ForEach(files, p => filenames.Enqueue(p));
            this.callback = callback;
            var o = new Thread(OutputProc);
            o.Name = "Output thread";
            o.Start();
            for (int i = 0; i < Environment.ProcessorCount; i++)
            {
                var t = new Thread(WorkerProc);
                t.Name = $"Worker thread {i}";
                t.Start();
            }
        }

        public void Stop() => stopper.Reset();

        private static double[] Softmax(IEnumerable<float> z)
        {
            var exps = z.Select(System.Convert.ToDouble).Select(Math.Exp);
            var sum = exps.Sum();
            return exps.Select(i => i / sum).ToArray();
        }

        private static Tensor<float> LoadImage(string path)
        {
            using var img = Image.Load<Rgb24>(path);
            img.Mutate(arg => arg
                .Grayscale()
                .Resize(new ResizeOptions { Size = img.Size(), Mode = ResizeMode.Min })
                .Resize(new ResizeOptions { Size = img.Size(), Mode = ResizeMode.Crop })
            );

            Tensor<float> t = new DenseTensor<float>(new[] { 1, 1, 28, 28 });
            for (int y = 0; y < img.Height; y++)
            {
                var span = img.GetPixelRowSpan(y);
                for (int x = 0; x < img.Width; x++)
                {
                    t[0, 0, y, x] = span[x].R / 255.0f;
                }
            }

            return t;
        }

        private void WorkerProc()
        {
            var threadName = Thread.CurrentThread.Name;
            var inputName = session.InputMetadata.Keys.First();
            while (true)
            {
                if (!stopper.WaitOne(0))
                {
                    results.Enqueue(new WorkerInterrupted { ThreadName = threadName });
                    outputMutex.Set();
                    break;
                }

                if (!filenames.TryDequeue(out string path))
                {
                    results.Enqueue(new WorkerSucceeded { ThreadName = threadName });
                    outputMutex.Set();
                    break;
                }

                var onnxValue = NamedOnnxValue.CreateFromTensor(inputName, LoadImage(path));
                using var res = session.Run(new List<NamedOnnxValue> { onnxValue });
                var output = Softmax(res.First().AsEnumerable<float>());

                results.Enqueue(new RecognitionResult(threadName, path, output));
                outputMutex.Set();
            }
        }

        private void OutputProc()
        {
            var threadsEnded = 0;
            while (true)
            {
                if (threadsEnded == Environment.ProcessorCount) { break; }
                if (!results.TryDequeue(out WorkerResult info))
                {
                    outputMutex.WaitOne();
                    continue;
                }
                if (info is WorkerFinished) { ++threadsEnded; }
                callback(info);
            }
            stopper.Reset();
            callback(new AllWorkersFinished { ThreadName = Thread.CurrentThread.Name });
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    session.Dispose();
                    stopper.Dispose();
                    outputMutex.Dispose();
                }
                disposedValue = true;
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
