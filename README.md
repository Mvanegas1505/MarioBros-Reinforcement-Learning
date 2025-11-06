# Super Mario RL (Stable-Baselines3 + Gymnasium)

Aprendizaje por Refuerzo (PPO) para jugar Super Mario Bros usando `gym-super-mario-bros`, `nes-py`, `gymnasium` y `stable-baselines3`.

- Agente: PPO (SB3)
- Entorno: SuperMarioBros-v3 (gym-super-mario-bros)
- Observaciones: escala de grises 84x84, apilado de 4 frames
- Registro: TensorBoard + gráfico de recompensas en `logs/`

> Nota: Verás avisos de "Gym has been unmaintained". Es normal porque `gym-super-mario-bros`/`nes-py` usan wrappers de Gym. El proyecto ya está adaptado para usar Gymnasium con compatibilidad.

---

## Requisitos
- Windows 10/11
- Python 3.11 (recomendado)
- PowerShell

Sugerido: usar un entorno virtual en una ruta corta para evitar problemas de Long Paths en Windows.

---

## Instalación rápida (PowerShell)

1) Crear y activar un entorno virtual corto (recomendado):

```powershell
# Crea venv en ruta corta (evita Long Paths)
py -3.11 -m venv C:\venv\mariorl

# Activa
& C:\venv\mariorl\Scripts\Activate.ps1
```

2) Instalar dependencias:

```powershell
pip install --no-cache-dir -r .\requirements.txt
```

Si PyTorch fallara al instalarse en otra máquina Windows, usa la rueda CPU oficial:

```powershell
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
```

> Si PowerShell bloquea la activación del venv, ejecuta una vez:
>
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

## Cómo entrenar

```powershell
# Desde la carpeta del proyecto, con el venv activo
python .\train_agent.py --train
```

- Checkpoints en `train/` (p.ej. `best_model_10000.zip`).
- Modelo final: `mario_bros_model_final.zip`.
- Gráfico de recompensa promedio: `logs/training_rewards.png`.

### Ver métricas con TensorBoard

```powershell
# Con el venv activo
tensorboard --logdir .\logs\
# Abre http://localhost:6006
```

---

## Cómo visualizar (ver jugar al agente)

Usa un checkpoint o el modelo final.

```powershell
# Activar entorno (si hace falta)
& C:\venv\mariorl\Scripts\Activate.ps1

# Visualizar con un checkpoint concreto
python .\train_agent.py --visualize .\train\best_model_10000.zip

# O con el modelo final
python .\train_agent.py --visualize .\mario_bros_model_final.zip
```

> La ventana de juego (Pyglet) puede abrirse detrás de otras ventanas; usa Alt+Tab si no la ves.

---

## Estructura del proyecto

```
train_agent.py         # Script principal: crea entorno, entrena (PPO), visualiza
requirements.txt       # Dependencias fijadas para reproducibilidad
.gitignore             # Ignora venvs, logs, modelos, caches
logs/                  # Logs de entrenamiento y TensorBoard
train/                 # Checkpoints (best_model_*.zip) y modelos
README.md              # Este documento
```

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

## Problemas comunes (FAQ)

- "Trying to log data to tensorboard but tensorboard is not installed":
  - Solución: `pip install tensorboard` (ya está en requirements.txt).

- Error de Long Paths instalando torch en Windows:
  - Solución 1 (recomendada): usar venv en `C:\venv\...` como arriba.
  - Solución 2: habilitar Long Paths (requiere admin) y reiniciar sesión:
    ```powershell
    New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -PropertyType DWord -Value 1 -Force
    ```

- PowerShell bloquea `Activate.ps1`:
  - `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` y volver a activar venv.

- Avisos de Gym desactualizado:
  - Son esperados por la cadena de wrappers de `nes-py`/`gym-super-mario-bros`. El proyecto ya usa Gymnasium y compatibilidad.

---

## Créditos
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)
- [nes-py](https://github.com/Kautenja/nes-py)

---

## Licencia
Este proyecto es exclusivamente para fines académicos/educativos.
