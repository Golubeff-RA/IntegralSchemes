#!/usr/bin/env python3
"""
Генерация расширенного HTML‑отчёта по результатам сравнения алгоритмов разбиения.

Отчёт включает три раздела:
1. Cut size – сравнение cut weight (KL vs Multilevel)
2. Execution time – сравнение времени работы
3. Balance quality – сравнение баланса частей

Запуск:
    python experiments/generate_advanced_report.py --input comparison_results.csv --output report.html
"""

import argparse
import base64
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Настройки стиля
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 120
plt.rcParams['font.size'] = 10


def read_data(csv_path: str) -> pd.DataFrame:
    """Читает CSV и добавляет колонки с типом графа и размером."""
    df = pd.read_csv(csv_path)

    def parse_graph_info(name):
        parts = name.replace('\\', '/').split('/')
        if len(parts) >= 2:
            graph_type = parts[0]
            size_str = parts[1].split('_')[-1].split('.')[0]
            try:
                size = int(size_str)
            except ValueError:
                size = 0
        else:
            graph_type = 'unknown'
            size = 0
        return graph_type, size

    df[['graph_type', 'size']] = df['graph'].apply(
        lambda x: pd.Series(parse_graph_info(x))
    )
    df = df.sort_values('size').reset_index(drop=True)
    return df


def fig_to_base64(fig) -> str:
    """Конвертирует фигуру matplotlib в строку base64 для встраивания в HTML."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64


def plot_cut_comparison(df: pd.DataFrame) -> str:
    """График сравнения cut weight."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: линии по размеру
    ax = axes[0]
    ax.plot(df['size'], df['kl_cut'], 'o-', color='#e74c3c', label='KL', linewidth=1.5, markersize=6)
    ax.plot(df['size'], df['ml_cut'], 's-', color='#2ecc71', label='Multilevel', linewidth=1.5, markersize=6)
    ax.set_xlabel('Number of vertices')
    ax.set_ylabel('Cut weight')
    ax.set_title('Cut weight vs size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: improvement по типам графов (boxplot)
    ax = axes[1]
    order = sorted(df['graph_type'].unique())
    data = [df[df['graph_type'] == t]['improvement_%'].values for t in order]
    bp = ax.boxplot(data, labels=order, patch_artist=True,
                    boxprops=dict(facecolor='#3498db', alpha=0.6))
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Graph type')
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Cut improvement over KL')
    plt.xticks(rotation=45)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_time_comparison(df: pd.DataFrame) -> str:
    """Графики времени выполнения."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: логарифмический масштаб
    ax = axes[0]
    ax.loglog(df['size'], df['kl_time'], 'o-', color='#e74c3c', label='KL', linewidth=1.5, markersize=6)
    ax.loglog(df['size'], df['ml_time'], 's-', color='#2ecc71', label='Multilevel', linewidth=1.5, markersize=6)
    ax.set_xlabel('Number of vertices')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Runtime (log‑log scale)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Right: speedup boxplot по типам
    ax = axes[1]
    order = sorted(df['graph_type'].unique())
    data = [df[df['graph_type'] == t]['speedup'].values for t in order]
    bp = ax.boxplot(data, labels=order, patch_artist=True,
                    boxprops=dict(facecolor='#9b59b6', alpha=0.6))
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Graph type')
    ax.set_ylabel('Speedup (KL time / ML time)')
    ax.set_title('Speedup over KL')
    plt.xticks(rotation=45)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig_to_base64(fig)


def plot_balance_comparison(df: pd.DataFrame) -> str:
    """Графики качества балансировки."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: сравнение баланса (линии)
    ax = axes[0]
    ax.plot(df['size'], df['kl_balance'], 'o-', color='#e74c3c', label='KL', linewidth=1.5, markersize=6)
    ax.plot(df['size'], df['ml_balance'], 's-', color='#2ecc71', label='Multilevel', linewidth=1.5, markersize=6)
    ax.set_xlabel('Number of vertices')
    ax.set_ylabel('Balance quality (1 = perfect)')
    ax.set_title('Balance vs size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: разница балансов (ML - KL)
    ax = axes[1]
    balance_diff = df['ml_balance'] - df['kl_balance']
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in balance_diff]
    ax.bar(range(len(df)), balance_diff, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['graph_type'].astype(str) + '_' + df['size'].astype(str), rotation=90, fontsize=8)
    ax.set_ylabel('Balance difference (ML - KL)')
    ax.set_title('Balance quality improvement')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig_to_base64(fig)


def generate_summary_cards(df: pd.DataFrame) -> str:
    """Генерирует HTML-карточки с общей статистикой."""
    avg_improvement = df['improvement_%'].mean()
    median_improvement = df['improvement_%'].median()
    wins = (df['improvement_%'] > 0).sum()
    total = len(df)
    avg_speedup = df['speedup'].mean()
    avg_balance_diff = (df['ml_balance'] - df['kl_balance']).mean()
    best_cut = df.loc[df['improvement_%'].idxmax()] if total > 0 else None

    cards = f"""
    <div class="summary">
        <div class="card">
            <div class="card-value">{total}</div>
            <div class="card-label">Graphs tested</div>
        </div>
        <div class="card">
            <div class="card-value">{avg_improvement:.1f}%</div>
            <div class="card-label">Avg. cut improvement (ML over KL)</div>
        </div>
        <div class="card">
            <div class="card-value">{avg_speedup:.2f}x</div>
            <div class="card-label">Avg. speedup</div>
        </div>
        <div class="card">
            <div class="card-value">{wins}/{total}</div>
            <div class="card-label">Multilevel wins (cut)</div>
        </div>
        <div class="card">
            <div class="card-value">{avg_balance_diff:+.3f}</div>
            <div class="card-label">Avg. balance change (ML-KL)</div>
        </div>
    </div>
    """
    if best_cut is not None:
        cards += f"""
        <div class="card" style="background:#2c3e50; color:white;">
            <div class="card-value">{best_cut['graph']}</div>
            <div class="card-label">Best improvement: {best_cut['improvement_%']:.1f}%</div>
        </div>
        """
    return cards


def generate_table(df: pd.DataFrame, metric: str) -> str:
    """Генерирует HTML-таблицу для заданной метрики (cut, time, balance)."""
    if metric == 'cut':
        display = df[['graph', 'vertices', 'kl_cut', 'ml_cut', 'improvement_%']].copy()
        display['improvement_%'] = display['improvement_%'].map(lambda x: f"{x:+.1f}%")
        display.columns = ['Graph', 'Vertices', 'KL cut', 'ML cut', 'Improvement']
    elif metric == 'time':
        display = df[['graph', 'vertices', 'kl_time', 'ml_time', 'speedup']].copy()
        display['speedup'] = display['speedup'].map(lambda x: f"{x:.2f}x")
        display['kl_time'] = display['kl_time'].map(lambda x: f"{x:.4f}s")
        display['ml_time'] = display['ml_time'].map(lambda x: f"{x:.4f}s")
        display.columns = ['Graph', 'Vertices', 'KL time', 'ML time', 'Speedup']
    elif metric == 'balance':
        display = df[['graph', 'vertices', 'kl_balance', 'ml_balance']].copy()
        display['kl_balance'] = display['kl_balance'].map(lambda x: f"{x:.4f}")
        display['ml_balance'] = display['ml_balance'].map(lambda x: f"{x:.4f}")
        display.columns = ['Graph', 'Vertices', 'KL balance', 'ML balance']
    else:
        return ""

    return display.to_html(index=False, classes='dataframe', float_format='%.4f')


def generate_html_report(df: pd.DataFrame, output_file: Path):
    """Формирует полный HTML-отчёт."""
    cut_img = plot_cut_comparison(df)
    time_img = plot_time_comparison(df)
    balance_img = plot_balance_comparison(df)
    summary_cards = generate_summary_cards(df)
    cut_table = generate_table(df, 'cut')
    time_table = generate_table(df, 'time')
    balance_table = generate_table(df, 'balance')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Advanced Algorithm Comparison Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .summary {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: #ecf0f1;
            border-radius: 8px;
            padding: 15px;
            min-width: 150px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .card-value {{
            font-size: 28px;
            font-weight: bold;
            color: #2980b9;
        }}
        .card-label {{
            font-size: 14px;
            color: #7f8c8d;
        }}
        .section {{
            margin-top: 40px;
            border-top: 2px solid #bdc3c7;
            padding-top: 20px;
        }}
        .graphics {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
            margin-bottom: 30px;
        }}
        .figure {{
            flex: 1;
            min-width: 400px;
            text-align: center;
        }}
        .figure img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
            padding: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 13px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: right;
        }}
        th {{
            background-color: #34495e;
            color: white;
            text-align: center;
        }}
        td:first-child, th:first-child {{
            text-align: left;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        footer {{
            margin-top: 30px;
            text-align: center;
            font-size: 12px;
            color: #95a5a6;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 Advanced Algorithm Comparison Report</h1>
    <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    {summary_cards}
    
    <div class="section">
        <h2>✂️ Cut size comparison</h2>
        <div class="graphics">
            <div class="figure"><img src="data:image/png;base64,{cut_img}" alt="Cut comparison"></div>
        </div>
        {cut_table}
    </div>
    
    <div class="section">
        <h2>⏱️ Execution time comparison</h2>
        <div class="graphics">
            <div class="figure"><img src="data:image/png;base64,{time_img}" alt="Time comparison"></div>
        </div>
        {time_table}
    </div>
    
    <div class="section">
        <h2>⚖️ Balance quality comparison</h2>
        <div class="graphics">
            <div class="figure"><img src="data:image/png;base64,{balance_img}" alt="Balance comparison"></div>
        </div>
        {balance_table}
    </div>
    
    <footer>
        Report generated by <code>generate_advanced_report.py</code>
    </footer>
</div>
</body>
</html>"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Report saved to {output_file.resolve()}")


def main():
    parser = argparse.ArgumentParser(description='Generate advanced HTML report from CSV.')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', default='advanced_report.html', help='Output HTML file')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    df = read_data(str(input_path))
    print(f"Loaded {len(df)} rows from {input_path}")

    output_path = Path(args.output)
    generate_html_report(df, output_path)


if __name__ == "__main__":
    main()