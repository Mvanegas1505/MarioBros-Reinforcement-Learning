"""
Sistema Avanzado de Visualización para Agentes RL en Super Mario Bros

Este módulo implementa un visualizador de alta calidad para agentes entrenados
en Super Mario Bros, con las siguientes características clave:

Características Principales:
1. Visualización de Alta Calidad:
   - Escalado configurable de resolución
   - Múltiples modos de interpolación
   - Vista opcional de frames procesados
   - Soporte para 60 FPS

2. Controles Interactivos:
   - Pausa/Continuar (tecla 'p')
   - Salida suave (tecla 'q')
   - Escalado en tiempo real

3. Compatibilidad Universal:
   - Soporte para modelos RecurrentPPO
   - Compatible con checkpoints y modelos finales
   - Adaptación automática a diferentes versiones del juego

4. Diagnóstico y Monitoreo:
   - Visualización de recompensas acumuladas
   - Tracking de finalización de episodios
   - Métricas de rendimiento en tiempo real

Uso:
    # Visualización básica (3x, interpolación nearest)
    python visualize_agent.py --model ./checkpoints/mario_model_100000.zip

    # Visualización HD (4x, interpolación lanczos)
    python visualize_agent.py --model ./train/best_model.zip --scale 4 --interp lanczos

    # Modo debug con vista de frames procesados
    python visualize_agent.py --model ./train/final_model.zip --show-grayscale

Archivos Relacionados:
    - train_agent.py: Para entrenar nuevos agentes
    - plot_progress.py: Para analizar métricas de entrenamiento
    - wrappers.py: Sistema de recompensas y preprocesamiento
"""

import os
import time
import numpy as np
import cv2

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


# ------------------- Wrappers de Preprocesamiento -------------------
class PreprocessFrame(gym.ObservationWrapper):
    """Preprocesa los frames del juego para visualización y compatibilidad.
    
    Este wrapper mantiene la consistencia con el formato usado durante
    el entrenamiento, asegurando que:
    1. Las dimensiones sean 84x84 (estándar CNN)
    2. La imagen esté en escala de grises
    3. El formato sea compatible con el modelo
    
    Args:
        env: Entorno base de gym
        new_shape: Tupla (height, width, channels). Default (84, 84, 1)
        
    Nota: Es crítico usar exactamente el mismo preprocesamiento que en
    train_agent.py para asegurar el comportamiento correcto del modelo.
    """
    def __init__(self, env, new_shape=(84, 84, 1)):
        super(PreprocessFrame, self).__init__(env)
        self.new_shape = new_shape
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255, 
            shape=self.new_shape, 
            dtype=np.uint8
        )

    def observation(self, obs):
        """Procesa un frame RGB a escala de grises 84x84.
        
        Args:
            obs: Frame RGB original del juego
            
        Returns:
            ndarray: Frame procesado (84, 84, 1)
        """
        # RGB a escala de grises
        processed = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize a 84x84 usando interpolación de área
        processed = cv2.resize(
            processed, 
            (self.new_shape[1], self.new_shape[0]), 
            interpolation=cv2.INTER_AREA
        )
        # Añadir dimensión de canal
        return processed[:, :, np.newaxis]


class SeedCompatWrapper(gym.Wrapper):
    """Asegura compatibilidad entre APIs antiguas y nuevas de Gym.
    
    Este wrapper es necesario porque:
    1. gym-super-mario-bros usa una versión antigua de Gym
    2. Las nuevas versiones de Gym requieren seed y options en reset()
    3. Debemos mantener compatibilidad entre versiones
    
    La función reset() intenta:
    1. Usar la nueva API con seed y options
    2. Si falla, recurre a la API antigua
    3. Asegura funcionamiento en ambos casos
    """
    def reset(self, seed=None, options=None):
        """Maneja reset() de forma compatible entre versiones.
        
        Args:
            seed: Semilla para reproducibilidad
            options: Opciones adicionales de reset
            
        Returns:
            tuple: (observación, info) del reset
        """
        try:
            return self.env.reset(seed=seed, options=options)
        except TypeError:
            return self.env.reset()


# ------------------- Sistema de Recompensas -------------------
class XPosRewardWrapper(gym.Wrapper):
    """Implementa un sistema de recompensas simplificado para visualización.
    
    Este wrapper mantiene una versión simplificada del sistema de recompensas
    usado en entrenamiento, enfocándose solo en:
    1. Progreso horizontal (avance en eje X)
    2. Penalización por muerte
    3. Sin recompensas secundarias
    
    La simplificación ayuda a:
    - Entender el comportamiento base del agente
    - Visualizar la estrategia fundamental
    - Analizar la política aprendida
    
    Nota: Este wrapper es una versión reducida del usado en entrenamiento,
    manteniendo solo las recompensas esenciales para visualización.
    """
    def __init__(self, env):
        """Inicializa el wrapper con tracking de posición X.
        
        Args:
            env: Entorno base de gym
        """
        super(XPosRewardWrapper, self).__init__(env)
        self.current_x = 0

    def reset(self, **kwargs):
        """Reinicia el tracking de posición X.
        
        Args:
            **kwargs: Argumentos pasados a env.reset()
            
        Returns:
            tuple: (observación, info) del reset
        """
        self.current_x = 0
        obs, info = self.env.reset(**kwargs)
        self.current_x = info.get('x_pos', 0)
        return obs, info

    def step(self, action):
        """Calcula recompensas basadas en progreso horizontal.
        
        Args:
            action: Acción a ejecutar en el ambiente
            
        Returns:
            tuple: (obs, reward, done, truncated, info)
            
        La recompensa se calcula como:
        1. Avance en X: recompensa positiva
        2. Muerte: penalización fija de -15
        3. Sin recompensa por retroceso
        """
        obs, _, done, truncated, info = self.env.step(action)
        
        # Recompensa por progreso en X
        x_pos_reward = max(0, info['x_pos'] - self.current_x)
        self.current_x = info['x_pos']
        new_reward = x_pos_reward

        # Penalización por muerte (excepto si completa el nivel)
        if done and not info.get('flag_get', False):
            new_reward = -15

        return obs, new_reward, done, truncated, info



# ------------------- Funciones de Configuración y Visualización -------------------
def create_env(env_id='SuperMarioBros-v0', render_mode=None):
    """Crea y configura un entorno de Mario Bros optimizado para visualización.
    
    Esta función aplica una serie de wrappers en orden específico para:
    1. Asegurar compatibilidad con el modelo entrenado
    2. Mantener consistencia en el preprocesamiento
    3. Optimizar la calidad de renderizado
    
    Args:
        env_id: ID del entorno (ej: 'SuperMarioBros-v0')
        render_mode: Modo de renderizado ('rgb_array' recomendado)
            None: Sin renderizado
            'rgb_array': Frames de alta calidad para OpenCV
            'human': Visualización directa (no recomendado)
            
    Returns:
        gym.Env: Entorno configurado y listo para visualización
        
    Nota: render_mode='rgb_array' es crucial para obtener frames
    de alta calidad que podemos procesar con OpenCV.
    """
    # 1. Crear entorno base con compatibilidad
    env = gym.make(env_id, 
                  render_mode=render_mode, 
                  apply_api_compatibility=True)
    
    # 2. Configurar espacio de acciones complejas
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    
    # 3. Aplicar sistema de recompensas simplificado
    env = XPosRewardWrapper(env)
    
    # 4. Asegurar compatibilidad entre versiones
    env = SeedCompatWrapper(env)
    
    # 5. Preprocesar frames para CNN
    env = PreprocessFrame(env, (84, 84, 1))
    
    return env


def visualize(model_path: str, 
              env_id: str = 'SuperMarioBros-v3', 
              scale: int = 3, 
              interp: str = 'nearest', 
              show_grayscale: bool = False):
    """Visualiza un agente entrenado con opciones avanzadas de renderizado.
    
    Esta función implementa un sistema de visualización de alta calidad con:
    1. Múltiples opciones de escalado e interpolación
    2. Soporte para diagnóstico visual (frames procesados)
    3. Controles interactivos (pausa, salida)
    4. Métricas en tiempo real
    
    Args:
        model_path: Ruta al archivo .zip del modelo
        env_id: ID del entorno (default: 'SuperMarioBros-v3')
        scale: Factor de escalado para la ventana (default: 3)
        interp: Método de interpolación (default: 'nearest')
            - 'nearest': Más rápido, estilo pixelado
            - 'linear': Balance velocidad/calidad
            - 'cubic'/'lanczos': Máxima calidad
        show_grayscale: Muestra frames procesados (default: False)
    
    Controles:
        - 'q': Salir de la visualización
        - 'p': Pausar/Reanudar
        - Ctrl+C: Interrumpir ejecución
    """
    # 1. Validación y Carga del Modelo
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en: {model_path}")
        return

    # Cargar modelo entrenado
    print(f"Cargando modelo desde: {model_path}")
    model = RecurrentPPO.load(model_path)

    # 2. Configuración del Entorno
    # Crear entorno con renderizado de alta calidad
    env = create_env(env_id, render_mode='rgb_array')
    env.reset()  # Reset inicial requerido
    
    # Vectorizar entorno y apilar frames
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')

    # Reset del entorno vectorizado
    obs = env.reset()
    total_reward = 0.0

    # 3. Configuración de Visualización
    # Validar y configurar escalado
    display_scale = max(1, int(scale))
    
    # Mapeo de métodos de interpolación
    interp_map = {
        'nearest': cv2.INTER_NEAREST,  # Más rápido, estilo pixel art
        'area': cv2.INTER_AREA,        # Bueno para downscaling
        'linear': cv2.INTER_LINEAR,    # Balance velocidad/calidad
        'cubic': cv2.INTER_CUBIC,      # Alta calidad
        'lanczos': cv2.INTER_LANCZOS4, # Máxima calidad
    }
    interpolation = interp_map.get(interp.lower(), cv2.INTER_NEAREST)
    
    # Configurar ventana
    window_name = 'Mario Bros RL - [q]Salir [p]Pausa'
    try:
        # 4. Bucle Principal de Visualización
        while True:
            # 4.1 Predicción y Ejecución
            action, _ = model.predict(obs, deterministic=True)
            try:
                # API nueva (con truncated)
                obs, rewards, dones, truncateds, infos = env.step(action)
            except ValueError:
                # API antigua (sin truncated)
                obs, rewards, dones, infos = env.step(action)
                truncateds = [False]
            
            # Actualizar recompensa acumulada
            total_reward += float(rewards[0])

            # 4.2 Extracción de Frame RGB
            # Acceder al entorno base para frames RGB
            base_env = env.envs[0]
            
            # Encontrar fuente de frames en color
            unwrap = base_env
            chain = []
            while hasattr(unwrap, 'env'):
                chain.append(type(unwrap).__name__)
                if isinstance(unwrap, PreprocessFrame):
                    color_source = unwrap.env
                unwrap = unwrap.env
            
            # Usar última fuente si no se encuentra PreprocessFrame
            color_source = locals().get('color_source', unwrap)
            
            # 4.3 Renderizado de Alta Calidad
            frame = None
            try:
                # Intentar obtener frame RGB
                frame = color_source.render(mode='rgb_array')
            except TypeError:
                try:
                    frame = color_source.render('rgb_array')
                except Exception:
                    frame = None

            # 4.4 Procesamiento y Visualización
            if frame is not None:
                if show_grayscale:
                    # Preparar vista de diagnóstico
                    proc_obs = obs[0, :, :, -1] if obs.ndim == 5 else obs[0, :, :, 0]
                    # Escalar frame procesado
                    proc_up = cv2.resize(
                        proc_obs, 
                        (frame.shape[1], frame.shape[0]), 
                        interpolation=cv2.INTER_NEAREST
                    )
                    # Convertir a RGB para concatenación
                    proc_up = cv2.cvtColor(proc_up, cv2.COLOR_GRAY2RGB)
                    # Concatenar frames lado a lado
                    frame_to_show = np.concatenate([frame, proc_up], axis=1)
                else:
                    frame_to_show = frame

                # Aplicar escalado con interpolación configurada
                h, w = frame_to_show.shape[:2]
                scaled = cv2.resize(
                    frame_to_show, 
                    (w * display_scale, h * display_scale), 
                    interpolation=interpolation
                )
                
                # Mostrar frame y procesar input
                cv2.imshow(
                    window_name, 
                    cv2.cvtColor(scaled, cv2.COLOR_RGB2BGR)
                )
                
                # 4.5 Control de Usuario
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print('Visualización terminada por usuario (q)')
                    break
                elif key == ord('p'):
                    print('Pausa - Presiona cualquier tecla para continuar')
                    cv2.waitKey(0)

            # 4.6 Control de FPS y Reset
            time.sleep(0.016)  # ~60 FPS
            
            # Reset al terminar episodio
            if bool(dones[0]) or bool(truncateds[0]):
                print(f"Episodio completado - Recompensa total: {total_reward:.2f}")
                total_reward = 0.0
                obs = env.reset()
                
    except KeyboardInterrupt:
        print("\nVisualización interrumpida por usuario (Ctrl+C)")
        
    finally:
        # 5. Limpieza y Cierre
        print("\nCerrando visualizador...")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        env.close()


# ------------------- Interfaz de Línea de Comandos -------------------
if __name__ == '__main__':
    import argparse

    # 1. Configuración del Parser
    parser = argparse.ArgumentParser(
        description='Visualizador Avanzado de Agentes RL en Super Mario Bros',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 2. Definición de Argumentos
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Ruta al modelo entrenado (ej: ./checkpoints/mario_model_100000.zip)'
    )
    
    parser.add_argument(
        '--env',
        type=str,
        default='SuperMarioBros-v3',
        help='ID del entorno de Gym a utilizar'
    )
    
    parser.add_argument(
        '--scale',
        type=int,
        default=3,
        help='Factor de escalado para la ventana (1-6)'
    )
    
    parser.add_argument(
        '--interp',
        type=str,
        default='nearest',
        choices=['nearest', 'area', 'linear', 'cubic', 'lanczos'],
        help='Método de interpolación para el escalado'
    )
    
    parser.add_argument(
        '--show-grayscale',
        action='store_true',
        help='Muestra frame procesado 84x84 junto al original'
    )

    # 3. Procesamiento y Ejecución
    args = parser.parse_args()
    
    # Validar escalado
    args.scale = max(1, min(6, args.scale))
    
    # Iniciar visualización
    visualize(
        model_path=args.model,
        env_id=args.env,
        scale=args.scale,
        interp=args.interp,
        show_grayscale=args.show_grayscale
    )
