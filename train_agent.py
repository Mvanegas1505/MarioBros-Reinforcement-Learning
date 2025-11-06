import gymnasium as gym
import gym as gym_legacy  # legacy Gym for wrappers compatible with nes_py/JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Workaround for nes_py overflow with NumPy 2.0: ensure ROM size math uses Python ints
def _patch_nes_py_numpy2_overflow():
    try:
        from nes_py._rom import ROM as _ROM
        # Cast header bytes to int to avoid uint8 overflow on multiplications
        _ROM.prg_rom_size = property(lambda self: int(16 * int(self.header[4])))
        _ROM.chr_rom_size = property(lambda self: int(8 * int(self.header[5])))
        _ROM.prg_rom_stop = property(
            lambda self: int(self.prg_rom_start + int(self.prg_rom_size) * (2 ** 10))
        )
        _ROM.chr_rom_stop = property(
            lambda self: int(self.chr_rom_start + int(self.chr_rom_size) * (2 ** 10))
        )
    except Exception:
        # If patch fails, proceed; env creation may raise with NumPy 2 + old nes_py
        pass

_patch_nes_py_numpy2_overflow()

# Workaround for gym_super_mario_bros with NumPy 2.0: cast RAM bytes to int in position math
def _patch_smb_numpy2_overflow():
    try:
        from gym_super_mario_bros.smb_env import SuperMarioBrosEnv as _SMB

        def _x_position(self):
            ram = getattr(self, 'ram')
            return int(int(ram[0x6D]) * 0x100 + int(ram[0x86]))

        # Replace property if it exists
        try:
            setattr(_SMB, '_x_position', property(_x_position))
        except Exception:
            pass
    except Exception:
        pass

_patch_smb_numpy2_overflow()

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True

class PreprocessFrame(gym_legacy.ObservationWrapper):
    def __init__(self, env, new_shape):
        super(PreprocessFrame, self).__init__(env)
        self.new_shape = new_shape
        # Use legacy gym spaces to match legacy wrapper chain
        self.observation_space = gym_legacy.spaces.Box(low=0, high=255, shape=self.new_shape, dtype=np.uint8)

    def observation(self, obs):
        # Grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize
        obs = cv2.resize(obs, (self.new_shape[1], self.new_shape[0]), interpolation=cv2.INTER_AREA)
        # Add channel dimension
        obs = obs[:, :, np.newaxis]
        return obs

class SeedCompatWrapper(gym_legacy.Wrapper):
    """Compatibility wrapper to accept Gymnasium-style reset(seed, options)
    when wrapping older Gym wrappers like nes_py.JoypadSpace that don't accept kwargs.
    """
    def reset(self, seed=None, options=None):
        try:
            return self.env.reset(seed=seed, options=options)
        except TypeError:
            return self.env.reset()

def create_env(env_id, render_mode=None):
    if render_mode:
        env = gym_super_mario_bros.make(env_id, render_mode=render_mode, apply_api_compatibility=True)
    else:
        env = gym_super_mario_bros.make(env_id, apply_api_compatibility=True)
    
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SeedCompatWrapper(env)
    env = PreprocessFrame(env, (84, 84, 1))
    return env

def plot_results(log_dir):
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    if len(x) > 0:
        y_smooth = np.convolve(y, np.ones(100)/100, mode='valid')
        x_smooth = x[len(x)-len(y_smooth):]

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(x_smooth, y_smooth, color='dodgerblue', linewidth=2.5, label='Recompensa Promedio (Suavizada)')
        
        ax.set_title('Progreso del Entrenamiento del Agente de Mario', fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('Timesteps', fontsize=16, labelpad=15)
        ax.set_ylabel('Recompensa Promedio por Episodio', fontsize=16, labelpad=15)
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        fig.tight_layout()
        plt.savefig(os.path.join(log_dir, 'training_rewards.png'), dpi=300)
        print(f"Gráfico de recompensas guardado en {os.path.join(log_dir, 'training_rewards.png')}")
        plt.show()

def train_agent():
    LOG_DIR = './logs/'
    CHECKPOINT_DIR = './train/'
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    env = DummyVecEnv([lambda: create_env('SuperMarioBros-v3')])
    env = VecFrameStack(env, 4, channels_order='last')
    env = VecMonitor(env, LOG_DIR)

    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)
    
    TIMESTEPS = 100000
    model.learn(total_timesteps=TIMESTEPS, callback=callback)
    
    model.save('mario_bros_model_final')
    print("Modelo final guardado como mario_bros_model_final.zip")
    
    plot_results(LOG_DIR)

def visualize_agent(model_path):
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en la ruta: {model_path}")
        print("Asegúrate de entrenar un modelo primero con --train")
        return

    model = PPO.load(model_path)
    # Crear entorno con render para visualización. create_env ya aplica PreprocessFrame.
    env = create_env('SuperMarioBros-v0', render_mode='human')
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')

    obs = env.reset()
    total_reward = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        # DummyVecEnv devuelve arrays de tamaño n_envs; aquí n_envs=1
        total_reward += float(rewards[0])
        env.render()
        if bool(dones[0]):
            print(f"Episodio terminado. Recompensa total: {total_reward}")
            total_reward = 0.0
            obs = env.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar o visualizar un agente de RL para Super Mario Bros.')
    parser.add_argument('--train', action='store_true', help='Entrenar un nuevo agente.')
    parser.add_argument('--visualize', type=str, metavar='MODEL_PATH', help='Visualizar un agente entrenado. Proporciona la ruta al modelo (.zip).')

    args = parser.parse_args()

    if args.train:
        train_agent()
    elif args.visualize:
        visualize_agent(args.visualize)
    else:
        print("Por favor, especifica una acción: --train para entrenar o --visualize RUTA_DEL_MODELO para visualizar.")

