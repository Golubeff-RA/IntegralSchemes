"""
Отслеживание производительности алгоритмов: время, память, итерации
Добавлено отслеживание времени по этапам
"""

import time
import psutil
import os
import gc
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StageTiming:
    """Время выполнения отдельного этапа"""
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class PerformanceRecord:
    """Запись о производительности одного запуска"""
    algorithm_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    memory_peak_mb: float = 0.0
    iterations: int = 0
    stage_timings: List[StageTiming] = field(default_factory=list)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def elapsed_time(self) -> float:
        """Время выполнения в секундах"""
        return self.end_time - self.start_time
    
    @property
    def memory_used_mb(self) -> float:
        """Использованная память в МБ"""
        return max(0, self.memory_after_mb - self.memory_before_mb)
    
    def get_stage_time(self, stage_name: str) -> float:
        """Получение времени выполнения этапа"""
        for stage in self.stage_timings:
            if stage.name == stage_name:
                return stage.duration
        return 0.0
    
    def get_stage_times_dict(self) -> Dict[str, float]:
        """Получение словаря времен по этапам"""
        return {stage.name: stage.duration for stage in self.stage_timings}
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        result = {
            'algorithm': self.algorithm_name,
            'time_seconds': self.elapsed_time,
            'memory_before_mb': self.memory_before_mb,
            'memory_after_mb': self.memory_after_mb,
            'memory_used_mb': self.memory_used_mb,
            'memory_peak_mb': self.memory_peak_mb,
            'iterations': self.iterations,
            **self.get_stage_times_dict()
        }
        # Добавляем дополнительные метрики
        result.update(self.additional_metrics)
        return result


class PerformanceTracker:
    """
    Отслеживание производительности алгоритмов
    
    Поддерживает:
    - Общее время выполнения
    - Время по этапам (coarsening, initial_partition, uncoarsening)
    - Использование памяти
    - Количество итераций
    """
    
    def __init__(self):
        self.records: List[PerformanceRecord] = []
        self._current_record: Optional[PerformanceRecord] = None
        self._current_stage: Optional[StageTiming] = None
        self._process = psutil.Process(os.getpid())
        self._peak_memory = 0.0
    
    def start(self, algorithm_name: str) -> None:
        """
        Начать измерение производительности
        
        Args:
            algorithm_name: имя алгоритма
        """
        gc.collect()
        
        self._current_record = PerformanceRecord(
            algorithm_name=algorithm_name,
            start_time=time.time(),
            memory_before_mb=self._get_memory_usage()
        )
        self._peak_memory = self._current_record.memory_before_mb
    
    def start_stage(self, stage_name: str) -> None:
        """
        Начать измерение этапа алгоритма
        
        Args:
            stage_name: имя этапа (например, 'coarsening', 'initial_partition', 'uncoarsening')
        """
        if self._current_record is None:
            raise RuntimeError("No measurement in progress. Call start() first.")
        
        self._current_stage = StageTiming(
            name=stage_name,
            start_time=time.time()
        )
    
    def end_stage(self) -> None:
        """Завершить измерение текущего этапа"""
        if self._current_stage is None:
            raise RuntimeError("No stage in progress. Call start_stage() first.")
        
        self._current_stage.end_time = time.time()
        if self._current_record is not None:
            self._current_record.stage_timings.append(self._current_stage)
        self._current_stage = None
    
    def end(self, iterations: int = 0, **additional_metrics) -> PerformanceRecord:
        """
        Завершить измерение производительности
        
        Args:
            iterations: количество итераций алгоритма
            **additional_metrics: дополнительные метрики
        """
        if self._current_record is None:
            raise RuntimeError("No measurement in progress. Call start() first.")
        
        # Завершаем текущий этап, если он есть
        if self._current_stage is not None:
            self.end_stage()
        
        self._current_record.end_time = time.time()
        self._current_record.memory_after_mb = self._get_memory_usage()
        self._current_record.memory_peak_mb = self._peak_memory
        self._current_record.iterations = iterations
        self._current_record.additional_metrics = additional_metrics
        
        self.records.append(self._current_record)
        result = self._current_record
        self._current_record = None
        
        return result
    
    @contextmanager
    def measure(self, algorithm_name: str, **kwargs):
        """
        Контекстный менеджер для измерения производительности
        """
        self.start(algorithm_name)
        try:
            yield self._current_record
        finally:
            self.end(**kwargs)
    
    @contextmanager
    def stage(self, stage_name: str):
        """
        Контекстный менеджер для измерения этапа
        """
        self.start_stage(stage_name)
        try:
            yield
        finally:
            self.end_stage()
    
    def _get_memory_usage(self) -> float:
        """Получение текущего использования памяти в МБ"""
        try:
            memory_info = self._process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            if memory_mb > self._peak_memory:
                self._peak_memory = memory_mb
            
            return memory_mb
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики по всем записям"""
        if not self.records:
            return {}
        
        try:
            import numpy as np
        except ImportError:
            np = None
        
        times = [r.elapsed_time for r in self.records]
        memories = [r.memory_used_mb for r in self.records]
        
        # Собираем времена по этапам
        stage_names = set()
        for r in self.records:
            stage_names.update(r.get_stage_times_dict().keys())
        
        stage_times = {stage: [] for stage in stage_names}
        for r in self.records:
            for stage, duration in r.get_stage_times_dict().items():
                stage_times[stage].append(duration)
        
        stats = {
            'total_runs': len(self.records),
            'total_time_seconds': sum(times),
            'avg_time_seconds': np.mean(times) if np else sum(times) / len(times),
            'min_time_seconds': min(times),
            'max_time_seconds': max(times),
            'total_memory_mb': sum(memories),
            'avg_memory_mb': np.mean(memories) if np else sum(memories) / len(memories),
            'min_memory_mb': min(memories),
            'max_memory_mb': max(memories),
            'stage_times': {stage: {
                'avg': np.mean(times_stage) if np else sum(times_stage) / len(times_stage),
                'total': sum(times_stage),
                'min': min(times_stage),
                'max': max(times_stage)
            } for stage, times_stage in stage_times.items() if times_stage},
            'records': [r.to_dict() for r in self.records]
        }
        
        if np:
            stats['std_time_seconds'] = np.std(times)
            stats['std_memory_mb'] = np.std(memories)
        
        return stats
    
    def get_record_by_algorithm(self, algorithm_name: str) -> List[PerformanceRecord]:
        """Получение записей для конкретного алгоритма"""
        return [r for r in self.records if r.algorithm_name == algorithm_name]
    
    def print_summary(self):
        """Вывод сводки производительности с временем по этапам"""
        stats = self.get_stats()
        
        if not stats:
            print("No performance data available")
            return
        
        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"Total runs: {stats['total_runs']}")
        
        print(f"\n⏱️  TIME METRICS:")
        print(f"  Total: {stats['total_time_seconds']:.3f}s")
        print(f"  Average: {stats['avg_time_seconds']:.4f}s")
        print(f"  Min: {stats['min_time_seconds']:.4f}s")
        print(f"  Max: {stats['max_time_seconds']:.4f}s")
        print(f"  Std: {stats.get('std_time_seconds', 0):.4f}s")
        
        print(f"\n📊 STAGE TIMES:")
        for stage, times in stats.get('stage_times', {}).items():
            print(f"  {stage}:")
            print(f"    Average: {times['avg']:.4f}s")
            print(f"    Total: {times['total']:.4f}s")
            print(f"    Range: [{times['min']:.4f}s, {times['max']:.4f}s]")
        
        print(f"\n💾 MEMORY METRICS:")
        print(f"  Total: {stats['total_memory_mb']:.2f} MB")
        print(f"  Average: {stats['avg_memory_mb']:.2f} MB")
        print(f"  Min: {stats['min_memory_mb']:.2f} MB")
        print(f"  Max: {stats['max_memory_mb']:.2f} MB")
        
        print("\n📈 PER ALGORITHM:")
        algorithms = set(r.algorithm_name for r in self.records)
        for algo in algorithms:
            algo_records = self.get_record_by_algorithm(algo)
            if algo_records:
                avg_time = sum(r.elapsed_time for r in algo_records) / len(algo_records)
                avg_mem = sum(r.memory_used_mb for r in algo_records) / len(algo_records)
                print(f"  {algo}: avg time={avg_time:.4f}s, avg mem={avg_mem:.2f}MB")
        
        print("=" * 60)
    
    def reset(self):
        """Сброс всех записей"""
        self.records.clear()
        self._current_record = None
        self._current_stage = None
        self._peak_memory = 0.0
    
    def export_to_csv(self, filename: str):
        """Экспорт данных в CSV файл"""
        import csv
        
        if not self.records:
            print("No data to export")
            return
        
        all_fields = set()
        for record in self.records:
            all_fields.update(record.to_dict().keys())
        
        fieldnames = sorted(all_fields)
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in self.records:
                row = record.to_dict()
                writer.writerow(row)
        
        print(f"Performance data exported to {filename}")
    
    def export_to_json(self, filename: str):
        """Экспорт данных в JSON файл"""
        import json
        
        data = {
            'records': [r.to_dict() for r in self.records],
            'summary': self.get_stats()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Performance data exported to {filename}")