"""
Entrenamiento Avanzado Multi-Entorno para Super Mario Bros RL

Este módulo implementa un sistema de entrenamiento paralelo para un agente de RL en
Super Mario Bros usando Stable-Baselines3. El sistema está optimizado para aprendizaje
eficiente con las siguientes características clave:

Características Principales:
1. Entrenamiento Paralelo:
   - Múltiples instancias del juego en paralelo (--n-envs)
   - Sincronización eficiente entre entornos
   - Aprovechamiento óptimo de CPU multi-núcleo

2. Sistema de Aprendizaje:
   - Algoritmo: RecurrentPPO con memoria LSTM
   - Observaciones: Frames 84x84 en escala de grises
   - Apilado de frames: 4 frames consecutivos
   - Sistema de recompensas enfocado en progreso X

3. Monitoreo y Checkpoints:
   - Guardado automático cada 100,000 steps
   - Evaluación periódica del mejor modelo
   - Integración con TensorBoard
   - Métricas detalladas de entrenamiento

Uso:
    # Entrenamiento básico (8 entornos, 100k steps)
    python train_agent.py

    # Entrenamiento optimizado (16 entornos, 3M steps)
    python train_agent.py --timesteps 3000000 --n-envs 16

    # Continuar entrenamiento desde checkpoint
    python train_agent.py --resume-from ./checkpoints/mario_model_100000_steps.zip

Archivos Relacionados:
    - visualize_agent.py: Para ver el agente en acción
    - plot_progress.py: Para graficar métricas de entrenamiento
    - wrappers.py: Sistema personalizado de recompensas
"""

import os
import argparse
import numpy as np
import cv2

import gymnasium as gym
import gym as gym_legacy  # para wrappers legacy (nes_py)
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from wrappers import DistanceRewardWrapper


# ------------------- Compatibilidad con NumPy 2.0 -------------------
def _patch_nes_py_numpy2_overflow():
    """Corrige problemas de overflow en nes-py con NumPy 2.0+
    
    NumPy 2.0 introdujo cambios en el manejo de enteros que causan
    overflow en el emulador NES. Esta función aplica parches para:
    1. Cálculo correcto de tamaños ROM
    2. Prevención de overflow en direcciones de memoria
    3. Conversión segura de tipos numéricos
    
    Nota: Este parche es temporal hasta que nes-py se actualice oficialmente.
    """
    try:
        from nes_py._rom import ROM as _ROM
        # Corregir cálculos de tamaño de ROM
        _ROM.prg_rom_size = property(lambda self: int(16 * int(self.header[4])))
        _ROM.chr_rom_size = property(lambda self: int(8 * int(self.header[5])))
        # Corregir cálculos de offset de memoria
        _ROM.prg_rom_stop = property(
            lambda self: int(self.prg_rom_start + int(self.prg_rom_size) * (2 ** 10))
        )
        _ROM.chr_rom_stop = property(
            lambda self: int(self.chr_rom_start + int(self.chr_rom_size) * (2 ** 10))
        )
    except Exception:
        pass  # Silencioso si el módulo no está disponible


def _patch_smb_numpy2_overflow():
    """Corrige el cálculo de posición X en gym-super-mario-bros para NumPy 2.0+
    
    El cálculo de la posición X del jugador usa operaciones que pueden causar
    overflow en NumPy 2.0+. Este parche:
    1. Asegura conversiones explícitas a int
    2. Previene overflow en multiplicaciones
    3. Mantiene precisión en el cálculo de posición
    
    Detalles Técnicos:
    - La posición X se almacena en dos bytes: 0x6D (high) y 0x86 (low)
    - Cálculo: high_byte * 256 + low_byte
    - Rango válido: 0-8447 (límite del nivel más largo)
    """
    try:
        from gym_super_mario_bros.smb_env import SuperMarioBrosEnv as _SMB

        def _x_position(self):
            # Obtener bytes de posición de la RAM del emulador
            ram = getattr(self, 'ram')
            # Conversión segura y cálculo de posición
            # high_byte * 256 + low_byte (usando 0x100 = 256)
            return int(int(ram[0x6D]) * 0x100 + int(ram[0x86]))

        try:
            # Reemplazar el property existente con la versión corregida
            setattr(_SMB, '_x_position', property(_x_position))
        except Exception:
            pass  # Silencioso si la propiedad no puede ser reemplazada
    except Exception:
        pass  # Silencioso si el módulo no está disponible


_patch_nes_py_numpy2_overflow()
_patch_smb_numpy2_overflow()


# ------------------- Wrappers de Preprocesamiento -------------------
class PreprocessFrame(gym_legacy.ObservationWrapper):
    """Optimiza las observaciones del juego para el aprendizaje.
    
    Este wrapper realiza tres transformaciones críticas:
    1. Conversión a escala de grises (reduce dimensionalidad)
    2. Redimensionamiento a 84x84 (tamaño estándar para CNNs)
    3. Normalización del formato (uint8, single channel)
    
    Args:
        env: Entorno base de gym
        new_shape: Tupla (height, width, channels). Default (84, 84, 1)
        
    El formato 84x84 es un estándar de facto en RL visual porque:
    - Es suficientemente grande para mantener detalles importantes
    - Es suficientemente pequeño para procesar eficientemente
    - Es cuadrado, lo que ayuda con las convoluciones
    """
    def __init__(self, env, new_shape=(84, 84, 1)):
        super(PreprocessFrame, self).__init__(env)
        self.new_shape = new_shape
        # Definir espacio de observación: imágenes uint8 [0-255]
        self.observation_space = gym_legacy.spaces.Box(
            low=0, high=255, shape=self.new_shape, dtype=np.uint8
        )

    def observation(self, obs):
        """Procesa un frame del juego.
        
        Pasos:
        1. RGB a escala de grises (reduce memoria y enfoca en features importantes)
        2. Resize a 84x84 usando INTER_AREA (mejor para downsampling)
        3. Añadir dimensión de canal (necesaria para CNNs)
        
        Args:
            obs: Frame RGB original del juego
            
        Returns:
            ndarray: Frame procesado en formato (84, 84, 1)
        """
        # RGB a escala de grises
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize con interpolación de área (mejor para reducción)
        obs = cv2.resize(obs, (self.new_shape[1], self.new_shape[0]), 
                        interpolation=cv2.INTER_AREA)
        # Añadir dimensión de canal
        obs = obs[:, :, np.newaxis]
        return obs


class SeedCompatWrapper(gym_legacy.Wrapper):
    """Asegura compatibilidad entre versiones nuevas y antiguas de Gym.
    
    Este wrapper es necesario porque:
    1. gym-super-mario-bros usa una versión antigua de Gym
    2. Las nuevas versiones de Gym requieren seed y options en reset()
    3. Las versiones antiguas no aceptan estos parámetros
    
    El wrapper intercepta las llamadas a reset() y:
    - Intenta usar la nueva API (seed, options)
    - Si falla, recurre a la API antigua
    - Mantiene consistencia entre versiones
    
    Este wrapper es temporal hasta que gym-super-mario-bros
    se actualice para usar la nueva API de Gymnasium.
    """
    def reset(self, seed=None, options=None):
        """Intenta reset() con nueva API, recurre a antigua si falla.
        
        Args:
            seed: Semilla para reproducibilidad (nuevo Gym)
            options: Opciones adicionales de reset (nuevo Gym)
            
        Returns:
            tuple: (observación, info) del reset
        """
        try:
            # Intentar API nueva (Gymnasium)
            return self.env.reset(seed=seed, options=options)
        except TypeError:
            # Recurrir a API antigua (Gym)
            return self.env.reset()


class ProgressiveRewardWrapper(gym_legacy.Wrapper):
    """
    Un wrapper de recompensa progresivo que incentiva el avance,
    penaliza el retroceso y la muerte, y da recompensas secundarias
    solo cuando el agente progresa.
    """
    def __init__(self, env):
        super(ProgressiveRewardWrapper, self).__init__(env)
        self.current_x = 0
        self.current_score = 0
        self.current_coins = 0

    def reset(self, **kwargs):
        self.current_x = 0
        self.current_score = 0
        self.current_coins = 0
        obs, info = self.env.reset(**kwargs)
        self.current_x = info.get('x_pos', 0)
        self.current_score = info.get('score', 0)
        self.current_coins = info.get('coins', 0)
        return obs, info

    def step(self, action):
        obs, original_reward, done, truncated, info = self.env.step(action)
        
        new_reward = 0

        # 1. Recompensa/Penalización primaria: Movimiento en el eje X
        x_pos_change = info['x_pos'] - self.current_x
        
        if x_pos_change > 0:  # Avanzando hacia la derecha
            new_reward += x_pos_change  # Recompensa por progreso

            # Recompensas secundarias solo si avanza
            coins_reward = (info['coins'] - self.current_coins) * 10
            new_reward += max(0, coins_reward)
            
            score_reward = (info['score'] - self.current_score) * 0.01
            new_reward += max(0, score_reward)

        elif x_pos_change < 0:  # Retrocediendo (moviéndose a la izquierda)
            new_reward += x_pos_change * 2  # Penalización amplificada por retroceder

        # Actualizar estado actual
        self.current_x = info['x_pos']
        self.current_coins = info['coins']
        self.current_score = info['score']

        # 2. Penalización por tiempo para incentivar la velocidad
        new_reward -= 0.1

        # 3. Penalización por muerte
        if done and not info.get('flag_get', False):
            new_reward = -20

        # 4. Bonus por completar el nivel
        if info.get('flag_get', False):
            new_reward += 300

        if done or truncated:
            print(f"Episode terminated. Done: {done}, Truncated: {truncated}")

        return obs, new_reward, done, truncated, info



def create_env(env_id, render_mode=None):
    """Crea y configura un entorno optimizado de Super Mario Bros para RL.
    
    Esta función construye un entorno completo aplicando una serie de wrappers
    en un orden específico para optimizar el aprendizaje:

    1. Entorno Base:
       - Crea el entorno de Mario Bros
       - Aplica compatibilidad con versiones nuevas de Gym
    
    2. Controles:
       - JoypadSpace: Mapea acciones complejas a botones NES
       - COMPLEX_MOVEMENT: Permite movimientos combinados
    
    3. Sistema de Recompensas:
       - DistanceRewardWrapper: Recompensas basadas en progreso X
       - Penalizaciones dinámicas por muerte/estancamiento
    
    4. Procesamiento de Observaciones:
       - Conversión a escala de grises
       - Redimensionamiento a 84x84
       - Normalización de valores
    
    5. Límites de Episodio:
       - Máximo 2000 steps por episodio
       - Previene episodios infinitos
    
    Args:
        env_id: ID del entorno (ej: 'SuperMarioBros-v3')
        render_mode: Modo de renderizado para visualización
    
    Returns:
        gym.Env: Entorno configurado y listo para entrenamiento
    
    Ejemplo:
        >>> env = create_env('SuperMarioBros-v3')
        >>> obs, info = env.reset()
        >>> for _ in range(2000):
        ...     action = env.action_space.sample()
        ...     obs, reward, done, truncated, info = env.step(action)
        ...     if done or truncated:
        ...         break
    """
    # 1. Crear entorno base con compatibilidad
    if render_mode:
        env = gym_legacy.make(env_id, render_mode=render_mode, apply_api_compatibility=True)
    else:
        env = gym_legacy.make(env_id, apply_api_compatibility=True)

    # 2. Configurar espacio de acciones complejas
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    
    # 3. Aplicar sistema de recompensas personalizado
    env = DistanceRewardWrapper(env)
    
    # 4. Asegurar compatibilidad entre versiones de Gym
    env = SeedCompatWrapper(env)
    
    # 5. Preprocesar frames para CNN
    env = PreprocessFrame(env, (84, 84, 1))
    
    # 6. Limitar duración de episodios
    env = gym_legacy.wrappers.TimeLimit(env, max_episode_steps=2000)
    
    return env





# ------------------- Sistema de Entrenamiento Multi-Entorno -------------------
def train_agent(env_id='SuperMarioBros-v3', n_envs=8, timesteps=100_000, 
                log_dir='./logs/', resume_from=None):
    """Implementa un sistema avanzado de entrenamiento paralelo para Mario Bros.
    
    Esta función configura y ejecuta el entrenamiento completo del agente usando
    múltiples instancias del juego en paralelo para maximizar la eficiencia del
    aprendizaje.

    Características Principales:
    1. Entrenamiento Paralelo:
       - Múltiples instancias simultáneas del juego
       - Aprovechamiento de múltiples núcleos CPU
       - Sincronización eficiente de experiencias

    2. Procesamiento de Observaciones:
       - Stack de 4 frames consecutivos
       - Formato optimizado para CNN+LSTM
       - Monitoreo de métricas en tiempo real

    3. Sistema de Checkpoints:
       - Guardado periódico de modelos
       - Evaluación continua de rendimiento
       - Posibilidad de reanudar entrenamiento

    Args:
        env_id: ID del entorno (default: 'SuperMarioBros-v3')
        n_envs: Número de entornos paralelos (default: 8)
        timesteps: Total de steps de entrenamiento (default: 100,000)
        log_dir: Directorio para logs/checkpoints (default: './logs/')
        resume_from: Path a modelo para continuar entrenamiento (optional)

    Returns:
        RecurrentPPO: Modelo entrenado
        str: Ruta al modelo final guardado

    Ejemplo:
        >>> model = train_agent(n_envs=16, timesteps=3_000_000)
        >>> model.save("mario_final.zip")
    """
    # Crear directorio de logs si no existe
    os.makedirs(log_dir, exist_ok=True)

    # 1. Configuración de Entornos Paralelos
    # Crear n_envs instancias independientes
    env_fns = [lambda eid=env_id: create_env(eid) for _ in range(n_envs)]
    # Inicializar entornos en procesos separados
    env = SubprocVecEnv(env_fns, start_method='spawn')
    # Apilar 4 frames para contexto temporal
    env = VecFrameStack(env, 4, channels_order='last')
    # Activar monitoreo de métricas
    env = VecMonitor(env, log_dir)

    # 2. Sistema de Evaluación y Monitoreo
    
    # 2.1 Evaluador Independiente ("El Juez")
    # Crear entorno de evaluación aislado para métricas consistentes
    eval_env = DummyVecEnv([lambda: create_env(env_id)])
    eval_env = VecFrameStack(eval_env, 4, channels_order='last')
    
    # Configurar frecuencia de evaluación (~25,000 steps)
    eval_freq = max(1, int(25000 / n_envs))
    
    # Configurar callback de evaluación y guardado automático
    eval_callback = EvalCallback(
        eval_env,                               # Entorno de evaluación
        best_model_save_path=os.path.join(log_dir, 'best_model'),  # Consistente con la estructura
        log_path=os.path.join(log_dir, 'results'),  # Directorio de métricas
        eval_freq=eval_freq,                   # Frecuencia de evaluación
        deterministic=True,                    # Evaluación determinista
        render=False,                          # Sin renderizado
        n_eval_episodes=5,                     # Más episodios para evaluación robusta
        verbose=1                              # Mostrar progreso
    )

    # 2.2 Sistema de Checkpoints para Visualización
    # Configurar frecuencia (~100,000 steps)
    checkpoint_freq = max(1, int(100000 / n_envs))
    print(f"Se guardará un checkpoint cada {checkpoint_freq * n_envs} timesteps totales")
    
    # Configurar callback de checkpoints periódicos
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,             # Frecuencia de guardado
        save_path='./checkpoints/',            # Directorio de checkpoints
        name_prefix='mario_model',             # Prefijo del archivo
        verbose=1                              # Mostrar progreso
    )

    # 3. Sistema de Monitoreo Completo
    callback_list = [eval_callback, checkpoint_callback]

    # 4. Inicialización del Modelo
    if resume_from and os.path.exists(resume_from):
        # 4.1 Cargar modelo existente para continuar entrenamiento
        print(f"Reanudando entrenamiento desde: {resume_from}")
        model = RecurrentPPO.load(
            resume_from,                    # Ruta al modelo guardado
            env=env,                        # Entorno actualizado
            tensorboard_log=log_dir         # Directorio de logs
        )
        reset_num_timesteps = False        # Continuar conteo de steps
    else:
        # 4.2 Crear nuevo modelo desde cero
        if resume_from:
            print(f"No se encontró el modelo: {resume_from}. Creando un modelo nuevo.")
        else:
            print("Iniciando un nuevo entrenamiento.")
            
        model = RecurrentPPO(
            "CnnLstmPolicy",               # CNN + LSTM para procesamiento temporal
            env,                           # Entorno vectorizado
            verbose=1,                     # Mostrar progreso
            tensorboard_log=log_dir,       # Directorio de métricas
            learning_rate=2.5e-4,          # Tasa de aprendizaje optimizada
            n_steps=512                    # Steps por actualización
        )
        reset_num_timesteps = True         # Iniciar conteo desde cero

    # 5. Entrenamiento
    model.learn(
        total_timesteps=timesteps,         # Total de steps a entrenar
        callback=callback_list,            # Sistema de monitoreo
        reset_num_timesteps=reset_num_timesteps  # Control de conteo
    )

    # 6. Guardado Final
    final_path = os.path.join(log_dir, 'mario_bros_model_final')
    model.save(final_path)
    print(f"Modelo final (último timestep) guardado en: {final_path}.zip")


# ------------------- CLI y Punto de Entrada Principal -------------------
if __name__ == '__main__':
    # 1. Configuración del CLI
    parser = argparse.ArgumentParser(
        description='Sistema de Entrenamiento Multi-Entorno para Super Mario Bros',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 2. Definición de Argumentos
    parser.add_argument(
        '--env', 
        type=str, 
        default='SuperMarioBros-v3',
        help='ID del entorno de Gym (ej: SuperMarioBros-v3, SuperMarioBrosRandomStages-v3)'
    )
    parser.add_argument(
        '--n-envs', 
        type=int, 
        default=8,
        help='Número de entornos paralelos para entrenamiento'
    )
    parser.add_argument(
        '--timesteps', 
        type=int, 
        default=100000,
        help='Total de timesteps para entrenar (pasos de simulación)'
    )
    parser.add_argument(
        '--log-dir', 
        type=str, 
        default='./logs/',
        help='Directorio para logs, métricas y checkpoints'
    )
    parser.add_argument(
        '--resume-from', 
        type=str, 
        default=None,
        help='Path al modelo para continuar entrenamiento (ej: ./logs/best_model.zip)'
    )

    # 3. Procesamiento de Argumentos e Inicio de Entrenamiento
    args = parser.parse_args()
    train_agent(
        env_id=args.env,           # Entorno a utilizar
        n_envs=args.n_envs,        # Entornos paralelos
        timesteps=args.timesteps,  # Total de steps
        log_dir=args.log_dir,      # Directorio de logs
        resume_from=args.resume_from  # Modelo para continuar (opcional)
    )

