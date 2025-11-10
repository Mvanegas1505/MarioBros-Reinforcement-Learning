"""
Script para graficar el progreso del entrenamiento a partir de los logs
generados por `stable_baselines3.common.monitor` (directorio `logs/`).

Uso: python plot_progress.py --log-dir ./logs/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy


def plot_results(log_dir: str = './logs/'):
    # Convertir la ruta relativa a absoluta
    log_dir = os.path.abspath(log_dir)
    print(f"Buscando logs en: {log_dir}")
    
    if not os.path.exists(log_dir):
        print(f"El directorio de logs no existe: {log_dir}")
        return

    # Buscar el archivo monitor.csv específicamente
    monitor_file = os.path.join(log_dir, 'monitor.csv')
    print(f"Buscando archivo: {monitor_file}")
    if os.path.exists(monitor_file):
        print(f"Encontrado monitor.csv")
        results = load_results(os.path.dirname(monitor_file))
    else:
        print(f"No se encontró monitor.csv, buscando en subdirectorios...")
        results = load_results(log_dir)
    
    if results.size == 0:
        print(f"No se encontraron resultados en {log_dir}")
        return

    x, y = ts2xy(results, 'timesteps')
    # y contiene recompensa por episodio
    if len(x) == 0:
        print("No hay datos para graficar.")
        return

    # Suavizado simple (media móvil)
    window = min(100, max(1, len(y)//10))
    y_smooth = np.convolve(y, np.ones(window)/window, mode='valid')
    x_smooth = x[len(x)-len(y_smooth):]

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(x, y, color='lightgray', alpha=0.6, label='Recompensa por episodio')
    ax.plot(x_smooth, y_smooth, color='dodgerblue', linewidth=2.2, label=f'Recompensa suavizada (w={window})')

    ax.set_title('Progreso del Entrenamiento - Recompensa por Episodio', fontsize=18)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Recompensa por episodio')
    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.4)

    out_path = os.path.join(log_dir, 'training_rewards.png')
    fig.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Gráfico guardado en: {out_path}")
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Graficar progreso de entrenamiento desde logs de Monitor')
    parser.add_argument('--log-dir', type=str, default='./logs/', help='Directorio donde se guardan los logs (por defecto ./logs/)')
    args = parser.parse_args()
    plot_results(args.log_dir)
