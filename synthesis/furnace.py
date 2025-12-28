import gymnasium as gym # type: ignore
from gymnasium import spaces # type: ignore
import numpy as np # type: ignore

class VirtualFurnaceEnv(gym.Env):
    """
    Virtual Furnace v5.0 (The Expert)
    Tweaks:
    1. Lower Start Range (Forces efficient ramping)
    2. Impurity Threshold (Allows risky behavior)
    """
    def __init__(self):
        super(VirtualFurnaceEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), 
            high=np.array([1, 1, 1, 1]), 
            dtype=np.float32
        )
        
        # --- PHYSICS (Standard) ---
        self.Ea_form_R = 13200.0   
        self.A_form = 50000.0      
        self.Ea_deg_R = 27600.0
        self.A_deg = 4.0e10        
        self.dt = 1.0  
        self.max_time = 180

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # TWEAK 1: Shift range down to [300, 700]
        # This forces the agent to practice "The Climb" from cold temperatures,
        # ensuring it works when we test it at real Room Temp (300K).
        self.temp = float(np.random.uniform(300, 700))
        
        self.state = np.array([1.0, 0.0, 0.0]) 
        self.time_step = 0
        return self._get_obs(), {}

    def step(self, action):
        if action == 0: self.temp -= 10
        elif action == 2: self.temp += 10
        self.temp = np.clip(self.temp, 300, 1400)
        
        # Physics Engine
        T = self.temp
        k_form = self.A_form * np.exp(-self.Ea_form_R / T)
        k_deg = self.A_deg * np.exp(-self.Ea_deg_R / T)
        
        precursor, target, impurity = self.state
        moles_forming = k_form * precursor * self.dt
        moles_burning = k_deg * target * self.dt
        
        self.state[0] -= moles_forming
        self.state[1] += (moles_forming - moles_burning)
        self.state[2] += moles_burning
        self.state = np.clip(self.state, 0, 1)
        
        self.time_step += 1
        done = self.time_step >= self.max_time
        
        # --- REWARD TUNING ---
        delta_yield = (moles_forming - moles_burning)
        reward = delta_yield * 2000.0 
        
        # TWEAK 2: The "Forgiveness" Threshold
        # Only punish if impurity is > 3%. 
        # This gives the agent confidence to surf the "Heat Wave."
        if self.state[2] > 0.03:
             reward -= (self.state[2] * 20.0)
        
        if done:
            reward += self.state[1] * 20.0
            
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # Normalize everything to 0.0 - 1.0 range
        return np.array([
            self.temp / 1500.0, 
            self.state[1], 
            self.state[2], 
            self.time_step / self.max_time
        ], dtype=np.float32)