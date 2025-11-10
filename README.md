# 🎮 Super Mario Bros con Aprendizaje por Refuerzo

Este proyecto implementa un agente de Aprendizaje por Refuerzo capaz de jugar Super Mario Bros usando PPO Recurrente (Proximal Policy Optimization) con memoria LSTM. El proyecto utiliza `stable-baselines3`, `gymnasium` y `gym-super-mario-bros`.

## 📚 Índice
- [Algoritmo PPO](#-algoritmo-ppo)
- [Características](#-características)
- [Instalación](#-instalación)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Uso](#-uso)
- [Monitoreo y Métricas](#-monitoreo-y-métricas)
- [Visualización](#-visualización)
- [FAQ y Troubleshooting](#-faq-y-troubleshooting)

## 🧠 Algoritmo PPO

PPO (Proximal Policy Optimization) es uno de los algoritmos más exitosos en RL por su estabilidad y rendimiento:

### Conceptos Clave
1. **Policy (Política)**: Red neuronal que decide qué acción tomar
2. **Value Function**: Red que estima la recompensa futura esperada
3. **Advantage**: Diferencia entre recompensa real y esperada

### Funcionamiento
1. **Recolección de Experiencia**:
   - El agente juega varios episodios
   - Guarda estados, acciones y recompensas

2. **Actualización de Política**:
   - Calcula ventajas (qué tan buenas fueron las acciones)
   - Actualiza la política gradualmente para mejorar acciones ventajosas
   - Limita cambios grandes para mantener estabilidad

3. **Memoria LSTM**:
   - Permite recordar información pasada
   - Útil para timing de saltos y patrones de niveles

## 🎮 Características

### Sistema de Aprendizaje
- **Algoritmo**: RecurrentPPO con memoria LSTM para patrones temporales
- **Arquitectura**: CNN + LSTM (procesa frames y recuerda contexto)
- **Observaciones**: 
  - Frames en escala de grises 84x84
  - Stack de 4 frames consecutivos
  - Reducción de dimensionalidad para eficiencia

### Entrenamiento Paralelo
- **Multi-entorno**: Entrenamiento en paralelo (8-16 instancias)
- **Checkpoints**: Guardado cada 100,000 steps
- **Evaluación**: Sistema independiente para tracking de progreso

### Sistema de Recompensas Optimizado (`wrappers.py`)

#### 🎯 Sistema de Recompensas por Progreso
- **Progreso Normal** (Δx > 0):
  - `reward = Δx * PROGRESS_SCALE`
  - Ejemplo: Avanzar 10 pixels = +10 (con PROGRESS_SCALE=1.0)

- **Récords Históricos**:
  - `reward = Δx * PROGRESS_SCALE * 2.0` cuando supera máximo histórico
  - Doble recompensa por explorar nuevo territorio
  - Ejemplo: Nuevo récord de 10 pixels = +20

#### 💀 Sistema de Penalización Adaptativo
- **Muerte**:
  - Penalización base: -50 puntos (DEATH_PENALTY)
  - Se reduce según el progreso logrado
  - Fórmula: `penalty = DEATH_PENALTY * (1.0 - progress_ratio)`
  - Ejemplo: Si llegó al 80% del máximo → -10 puntos
  - Máxima reducción: 90% de la penalización base

#### 🏆 Bonus y Anti-Estancamiento
- **Completar Nivel**: +500 puntos
- **Sistema Anti-Estancamiento**:
  - Cuenta pasos sin progreso
  - Después de 100 pasos: -0.1 por step
  - Se resetea al progresar

#### 🧮 Diseño y Justificación
- **Enfoque en Progreso**:
  ```python
  # Sistema base
  if nuevo_x > max_historico:
      reward = Δx * 2.0  # Doble recompensa por récord
  elif Δx > 0:
      reward = Δx       # Recompensa normal por avance
  
  # Penalizaciones
  if estancado > 100_steps:
      reward -= 0.1     # Anti-estancamiento
  
  if muerte:
      penalty = -50 * (1 - progress_ratio)  # Penalización adaptativa
  
  if nivel_completado:
      reward += 500     # Gran bonus final
  ```
  
- **Características Clave**:
  1. Recompensas inmediatas por progreso
  2. Penalizaciones proporcionadas
  3. Incentivos claros y consistentes
  4. Balance entre velocidad y exploración

- **Beneficios**:
  - Aprendizaje más rápido y estable
  - Menos distracciones (enemigos/powerups)
  - Comportamiento dirigido a objetivos (Pasarse el nivel)
  - Evita mínimos locales

## 💿 Instalación

1. **Crear Entorno Virtual**:
```powershell
# Crear virtualenv
python -m venv .venv

# Activar (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activar (Windows CMD)
.\.venv\Scripts\activate.bat

# Activar (Linux/macOS)
source .venv/bin/activate
```

2. **Actualizar pip e Instalar Dependencias**:
```powershell
# Actualizar pip
python -m pip install --upgrade pip

# Instalar requisitos
pip install -r requirements.txt
```

## 📁 Estructura del Proyecto

### Archivos Principales
- `train_agent.py`: Entrenamiento del agente
- `visualize_agent.py`: Visualización del agente entrenado
- `plot_progress.py`: Generación de gráficas de progreso
- `wrappers.py`: Sistema de recompensas y procesamiento

### Directorios
- `logs/`: Métricas y modelos
  - `best_model/`: Mejor modelo según evaluaciones
  - `results/`: Métricas de evaluación
  - `RecurrentPPO_*/`: Logs de cada sesión de entrenamiento
  - `monitor.csv`: Registro detallado de episodios

- `checkpoints/`: Modelos guardados periódicamente
  - `mario_model_100000_steps.zip`: Checkpoint cada 100k steps
  - Útiles para retomar entrenamiento o visualizar progreso

## 🚀 Uso

### Entrenamiento Básico
```powershell
# Entrenamiento básico (8 entornos, 100k steps)
python train_agent.py

# Entrenamiento optimizado (16 entornos, 3M steps)
python train_agent.py --timesteps 3000000 --n-envs 16

# Continuar desde checkpoint
python train_agent.py --resume-from ./checkpoints/mario_model_100000_steps.zip
```

### Visualización del Agente
```powershell
# Visualización básica
python visualize_agent.py --model ./checkpoints/mario_model_100000_steps.zip

# Visualización HD (4x, mejor calidad)
python visualize_agent.py --model ./checkpoints/mario_model_100000_steps.zip --scale 4 --interp lanczos

# Con vista de preprocesamiento
python visualize_agent.py --model ./checkpoints/mario_model_100000_steps.zip --show-grayscale
```

### Análisis de Progreso
```powershell
# Generar gráficas de entrenamiento
python plot_progress.py
```

## 📊 Monitoreo y Métricas

### TensorBoard (`logs/RecurrentPPO_*/`)
```powershell
# Iniciar TensorBoard
tensorboard --logdir=logs
# Abrir http://localhost:6006
```

Los directorios `RecurrentPPO_1/`, `RecurrentPPO_2/`, etc. representan diferentes sesiones de entrenamiento y contienen:

#### 📊 Gráficas de Evaluación (eval/)
- **mean_ep_length**: Duración promedio de episodios
  - Eje Y: Número de pasos por episodio
  - Mayor valor = El agente sobrevive más tiempo
  - Ideal: Aumento estable en el tiempo

- **mean_reward**: Recompensa promedio por episodio
  - Eje Y: Puntos totales por episodio
  - Indica qué tan bien juega el agente
  - Ideal: Tendencia creciente

#### 📈 Gráficas de Rollout (rollout/)
- **ep_len_mean**: Longitud media de episodios durante entrenamiento
  - Similar a eval/mean_ep_length pero en entrenamiento
  - Más variable por exploración

- **ep_rew_mean**: Recompensa media durante entrenamiento
  - Similar a eval/mean_reward pero en entrenamiento
  - Más ruidosa por naturaleza exploratoria

#### 🔄 Interpretación
- **Smoothed**: Línea suavizada para tendencia general
- **Value**: Valores individuales (más volátiles)
- **Step**: Paso de entrenamiento (eje X)

#### 📉 Otras Métricas
- **Policy loss**: Pérdida de la política (qué tan buenas son las decisiones)
- **Value loss**: Pérdida de la función de valor (qué tan bien predice recompensas)
- **Entropy**: Balance exploración/explotación
- **Learning rate**: Tasa de aprendizaje actual
- **Explanation**: 
  - Líneas ascendentes en recompensas = Mejora del agente
  - Picos y valles = Normal en exploración
  - Estabilización = Convergencia del aprendizaje

### Monitor CSV (`logs/monitor.csv`)
Registro detallado por episodio:
- **Recompensa**: Total por episodio
- **Length**: Duración del episodio
- **Time**: Timestamp para análisis temporal

### Checkpoints (`checkpoints/`)
Modelos guardados periódicamente:
- `mario_model_100000_steps.zip`: Estado a los 100k steps
- `mario_model_200000_steps.zip`: Estado a los 200k steps
- etc.

### Mejor Modelo (`logs/best_model/`)
- Guarda el modelo con mejor rendimiento en evaluaciones
- Se actualiza cuando hay mejoras significativas
- Evaluación basada en 5 episodios completos

### Métricas de Evaluación (`logs/results/`)
- Evaluaciones periódicas del agente
- Tracking de mejora en el tiempo
- Comparación entre checkpoints

## 🎥 Visualización

### Controles
- `q`: Salir de la visualización
- `p`: Pausar/Reanudar
- `Ctrl+C`: Interrumpir ejecución

### Opciones de Calidad
```powershell
# Calidad básica (rápida)
--scale 2 --interp nearest

# Calidad media (balance)
--scale 3 --interp linear

# Alta calidad (más lento)
--scale 4 --interp lanczos
```

### Diagnóstico
```powershell
# Ver frames procesados
--show-grayscale

# Ver recompensas en tiempo real
# Se muestra en la terminal
```

## ❓ FAQ y Troubleshooting

### Entrenamiento
- **P**: ¿Cuánto tiempo de entrenamiento necesito?
  - **R**: Mínimo 1M steps, recomendado 3M para buen rendimiento

- **P**: ¿Cuántos entornos paralelos usar?
  - **R**: 8-16 para balance entre velocidad y estabilidad

### Visualización
- **P**: Ventana negra/no aparece
  - **R**: Alt+Tab para buscar ventana
  - **R**: Verificar ruta del modelo

- **P**: Rendimiento lento
  - **R**: Reducir scale o usar interp=nearest
  - **R**: Desactivar show-grayscale

### Ver métricas con TensorBoard

```powershell
# Con el venv activo
tensorboard --logdir .\logs\
# Abre http://localhost:6006
```

Los directorios `RecurrentPPO_1/`, `RecurrentPPO_2/`, etc. en `logs/` representan sesiones individuales de entrenamiento. Se crea un nuevo directorio en estas situaciones:
- Al iniciar un entrenamiento desde cero
- Al continuar un entrenamiento desde un checkpoint existente
- El número incrementa automáticamente (1, 2, 3...) para mantener el histórico

Esto permite:
- Comparar diferentes sesiones de entrenamiento
- Analizar efectos de cambios en hiperparámetros
- Detectar problemas de convergencia
- Mantener registro de experimentos

---

## Cómo visualizar (ver jugar al agente)

Usa un checkpoint o el modelo final.

```powershell
# Activar entorno (si hace falta)
& C:\venv\mariorl\Scripts\Activate.ps1

# Visualizar con un checkpoint concreto
python .\visualize_agent.py --model .\checkpoints\mario_model_2500000_steps.zip --env SuperMarioBros-v0

# O con el modelo final
python .\visualize_agent.py --model .\mario_bros_model_final.zip --env SuperMarioBros-v0
```

> La ventana de juego (Pyglet) puede abrirse detrás de otras ventanas; usa Alt+Tab si no la ves.

---

## Detalles técnicos

- Se usa `gymnasium` como API principal y compatibilidad con `gym` para ciertos wrappers (p.ej. `JoypadSpace`).
- Preprocesamiento propio (`PreprocessFrame`):
  - Convierte a escala de grises (OpenCV)
  - Redimensiona a 84x84
  - Añade eje de canal y se apilan 4 frames (`VecFrameStack`) con orden de canales "last".
- `VecMonitor` registra estadísticas para gráficos y TensorBoard.

### Compatibilidad y versiones clave
- `gymnasium` < 1.3 (SB3 2.7 es compatible)
- `gym==0.26.2` para wrappers legacy necesarios por `nes-py`
- `numpy < 2.0` para evitar incompatibilidades de `gym`/`nes-py` (np.bool8, overflows)
- PyTorch CPU por defecto (no requiere GPU)

---

## Licencia
Este proyecto es exclusivamente para fines académicos/educativos.


## INTEGRANTES

Lucas Higuita Bedoya, Daniel Arcila Salazar, Martin Vanegas Ospina
