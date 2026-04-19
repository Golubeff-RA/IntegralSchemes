# run_gui.py
#!/usr/bin/env python3
"""
Запуск GUI приложения для визуализации разбиения графов
"""

import sys
from pathlib import Path

# Добавляем путь к корню проекта
sys.path.append(str(Path(__file__).parent))

from visualization.main_window import GraphPartitioningGUI


def main():
    """Запуск GUI"""
    app = GraphPartitioningGUI()
    app.run()


if __name__ == "__main__":
    main()