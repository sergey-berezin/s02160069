using System;
using System.IO;
using System.Linq;

using OnnxImageProcessor;

namespace _01_threads
{
    class Program
    {
        static void Callback(WorkerResult info)
        {
            switch (info)
            {
                case WorkerSucceeded r:
                    Console.WriteLine($"Succeeded {r.ThreadName}");
                    break;
                case WorkerInterrupted r:
                    Console.WriteLine($"Interrupted {r.ThreadName}");
                    break;
                case AllWorkersFinished r:
                    Console.WriteLine($"All workers finished from {r.ThreadName}");
                    break;
                case RecognitionResult r:
                    var table = r.Probs.Select((p, i) => ($"   {i}  ", $"{p,6:P1}"));
                    var rowOne = string.Join(' ', table.Select(t => t.Item1));
                    var rowTwo = string.Join(' ', table.Select(t => t.Item2));
                    var msg = "\n" +
                        $"Recognition from {r.ThreadName}\n" +
                        $"File {r.FileName}\n" +
                        $"{rowOne}\n" +
                        $"{rowTwo}\n" +
                    "";
                    Console.WriteLine(msg);
                    break;
            }
        }

        static void Main(string[] args)
        {
            var path = (args.Length == 0) ? "sample" : args[0];
            var files = Directory.GetFiles(path);
            Console.WriteLine("Directory scan done");
            Console.WriteLine("Press any key to start model");
            Console.WriteLine("Press any key again to stop worker threads");
            Console.ReadKey(true);
            using var p = new DirectoryProcessor(files, Callback);
            Console.ReadKey(true);
            p.Stop();
        }
    }
}
