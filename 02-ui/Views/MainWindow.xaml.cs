using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace _02_ui.Views
{
    public class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void InitializeComponent()
        {
            AvaloniaXamlLoader.Load(this);
        }
    }
}
