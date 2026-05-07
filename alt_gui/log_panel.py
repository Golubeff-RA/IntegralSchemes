"""Панель логов с перенаправлением stdout"""

import sys
from tkinter import LabelFrame, Text


class LogPanel(LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="Console Log", bg='#2c3e50', fg='white', font=('Arial', 10, 'bold'))
        self.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.text = Text(self, bg='#1e1e1e', fg='#00ff00', font=('Courier', 8), height=8)
        self.text.pack(fill='both', expand=True, padx=5, pady=5)
        
        self._redirect_stdout()
    
    def _redirect_stdout(self):
        class Redirect:
            def __init__(self, widget):
                self.widget = widget
            def write(self, text):
                self.widget.insert('end', text)
                self.widget.see('end')
            def flush(self):
                pass
        
        sys.stdout = Redirect(self.text)
        print("=== Graph Partitioning Visualizer ===\n")