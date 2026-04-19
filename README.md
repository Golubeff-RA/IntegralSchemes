# Graph Partitioning - Многоуровневое разбиение графов

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 О проекте

Проект реализует алгоритмы разбиения графов для задачи декомпозиции логических схем. Основное внимание уделяется **многоуровневому разбиению** (multilevel partitioning) для работы с большими графами (до 10⁵ вершин и 5×10⁵ рёбер).

### Цели работы

1. **Освоить технику многоуровневого разбиения** для работы с большими графами
2. **Сравнить качество и скорость** с классическим алгоритмом Кернигана-Лина (KL)
3. **Создать гибкую систему** для генерации тестовых графов и анализа результатов

### Установка зависимостей

```bash
pip install -r requirements.txt
```

```
graph_partitioning/
│
├── core/                    # Базовые структуры данных
│   ├── graph.py            # Класс графа (списки смежности, веса)
│   ├── partition.py        # Класс разбиения (принадлежность вершин)
│   └── coarse_graph.py     # Грубый граф для многоуровневого алгоритма
│
├── algorithms/              # Алгоритмы разбиения
│   ├── kernighan_lin.py    # Базовый KL (эталон)
│   └── multilevel/         # Многоуровневый алгоритм
│       ├── coarsener.py    # Стягивание графа
│       ├── initial_partitioner.py # Разбиение грубого графа
│       ├── uncoarsener.py  # Проекция и улучшение
│       └── multilevel_partitioner.py
│
├── data/                    # Генерация и загрузка данных
│   ├── generators/         # Генераторы графов
│   │   ├── cluster_generator.py    # Кластерная структура
│   │   └── barabasi_albert.py      # Масштабно-инвариантные графы
│   └── formats/            # Чтение/запись графов
│
├── metrics/                 # Оценка качества
│   ├── partition_metrics.py    # Cut size, баланс, плотность
│   ├── performance_tracker.py  # Время, память
│   └── comparison.py           # Сравнение алгоритмов
│
├── visualization/           # Визуализация
│   ├── graph_canvas.py      # Force-directed layout
│   ├── coarsening_viewer.py # Этапы стягивания
│   └── metrics_panel.py     # Отображение метрик
│
├── tests/                   # Модульные тесты
│   ├── test_generators.py   # Проверка генераторов
│   ├── test_kl.py          # Проверка KL
│   └── test_multilevel.py   # Проверка многоуровневого алгоритма
│
├── experiments/             # Эксперименты
│   ├ ### TODO ###
│
├── main.py               # Графический интерфейс
├── test.py               # Консольная версия
└── requirements.txt         # Зависимости
```