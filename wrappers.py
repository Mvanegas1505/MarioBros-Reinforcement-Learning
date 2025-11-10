"""
Wrappers personalizados para el entrenamiento de Mario Bros con enfoque en progreso.

Este módulo contiene wrappers especializados para el entrenamiento de agentes RL
en Super Mario Bros, con énfasis en aprendizaje progresivo y evitación de mínimos locales.
"""

import gym as gym_legacy
import numpy as np


class DistanceRewardWrapper(gym_legacy.Wrapper):
    """Wrapper especializado que premia ÚNICAMENTE el progreso en el eje X.
    
    Este wrapper implementa un sistema de recompensas enfocado exclusivamente en 
    el avance horizontal, ignorando distractores como monedas o puntos. El sistema
    usa recompensas dinámicas que se ajustan según el rendimiento histórico del agente.
    
    Características principales:
    1. Recompensa base por movimiento hacia la derecha (PROGRESS_SCALE = 1.0)
    2. Recompensa doble por superar máximos históricos (2.0 * PROGRESS_SCALE)
    3. Penalización gradual por estancamiento (-0.1 por paso después de 100 pasos)
    4. Penalización por muerte que se reduce según el progreso logrado (hasta -50.0)
    5. Gran bonus por completar el nivel (+500.0)
    
    Principios de diseño:
    1. La única métrica de éxito es la distancia máxima alcanzada en X
    2. Morir tiene penalización fuerte pero proporcional al progreso
    3. Estancarse tiene penalización gradual para evitar mínimos locales
    4. No hay recompensa por distractores (monedas/enemigos/puntos)
    """
    def __init__(self, env):
        """Inicializa el wrapper con configuración optimizada para aprendizaje progresivo.
        
        Args:
            env: Entorno base de gym-super-mario-bros
        
        Atributos configurados:
            current_x (int): Posición actual en X
            max_x_ever (int): Máxima posición X alcanzada históricamente
            max_x_episode (int): Máxima posición X en el episodio actual
            steps_without_progress (int): Contador de pasos sin avance
            stagnation_threshold (int): Umbral para considerar estancamiento
            PROGRESS_SCALE (float): Factor de escala para recompensas de progreso
            DEATH_PENALTY (float): Penalización base por muerte
            STAGNATION_PENALTY (float): Penalización por estancamiento
        """
        super(DistanceRewardWrapper, self).__init__(env)
        # Tracking de posición en X (actual e histórica)
        self.current_x = 0
        self.max_x_ever = 0      # Máximo histórico (persiste entre episodios)
        self.max_x_episode = 0   # Máximo del episodio actual (reset cada episodio)
        
        # Sistema de detección y penalización de estancamiento
        self.steps_without_progress = 0    # Contador de pasos sin avance
        self.stagnation_threshold = 100    # Umbral antes de penalizar estancamiento
        
        # Factores de recompensa/penalización (ajustados empíricamente)
        self.PROGRESS_SCALE = 1.0      # Recompensa base por unidad de avance
        self.DEATH_PENALTY = -50.0     # Penalización máxima por muerte
        self.STAGNATION_PENALTY = -0.1 # Penalización gradual por estancamiento
        
    def reset(self, **kwargs):
        """Reinicia el entorno y las métricas de seguimiento por episodio.
        
        Reinicia las variables específicas del episodio pero mantiene el máximo
        histórico (max_x_ever) para mantener el contexto de progreso global.
        
        Args:
            **kwargs: Argumentos adicionales pasados al reset del entorno base.
            
        Returns:
            tuple: (observación, info) del entorno reseteado.
        """
        # Reset de variables específicas del episodio
        self.current_x = 0
        self.max_x_episode = 0
        self.steps_without_progress = 0
        
        # Reset del entorno base
        obs, info = self.env.reset(**kwargs)
        self.current_x = info.get('x_pos', 0)
        self.max_x_episode = self.current_x
        return obs, info
        
    def step(self, action):
        """Ejecuta una acción y calcula la recompensa basada en progreso.
        
        Este método implementa la lógica central del sistema de recompensas:
        1. Recompensa base por avance en X
        2. Recompensa doble por superar máximos históricos
        3. Penalización gradual por estancamiento
        4. Penalización adaptativa por muerte
        5. Bonus por completar nivel
        
        Args:
            action: Acción a ejecutar en el entorno.
            
        Returns:
            tuple: (observación, recompensa, done, truncated, info)
        """
        # Ejecutar acción en el entorno base
        obs, _, done, truncated, info = self.env.step(action)
        
        # Obtener nueva posición X
        new_x = info.get('x_pos', self.current_x)
        reward = 0.0
        
        # 1. Sistema de Recompensas por Progreso
        x_progress = new_x - self.current_x
        if x_progress > 0:  # Si hay avance hacia la derecha
            # Bonus especial por superar récords históricos
            if new_x > self.max_x_ever:
                # Doble recompensa por superar máximos históricos
                reward += x_progress * self.PROGRESS_SCALE * 2.0
                self.max_x_ever = new_x
            else:
                # Recompensa base por progreso normal
                reward += x_progress * self.PROGRESS_SCALE
                
            # Resetear sistema anti-estancamiento
            self.steps_without_progress = 0
            
            # Actualizar máximo del episodio actual
            if new_x > self.max_x_episode:
                self.max_x_episode = new_x
        else:  # Si no hay avance o hay retroceso
            # Sistema de detección de estancamiento
            self.steps_without_progress += 1
            
            # Aplicar penalización gradual si supera umbral
            if self.steps_without_progress > self.stagnation_threshold:
                reward += self.STAGNATION_PENALTY  # Penalización suave
        
        # 2. Sistema de Penalización por Muerte Adaptativo
        if done and not info.get('flag_get', False):
            # Calcular ratio de progreso respecto al máximo histórico
            progress_ratio = self.max_x_episode / max(1, self.max_x_ever)
            
            # Penalización reducida si logró buen progreso
            # Máxima reducción: 90% de la penalización base
            # Ejemplo: Si llegó al 80% del máximo histórico:
            # death_penalty = -50 * (1 - 0.8) = -10
            death_penalty = self.DEATH_PENALTY * (1.0 - min(0.9, progress_ratio))
            reward += death_penalty
        
        # 3. Bonus por Completar Nivel
        if done and info.get('flag_get', False):
            reward += 500.0  # Recompensa significativa por victoria
        
        # Actualizar estado y retornar
        self.current_x = new_x
        return obs, reward, done, truncated, info