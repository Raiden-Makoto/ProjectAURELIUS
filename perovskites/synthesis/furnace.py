import gymnasium as gym # type: ignore
from gymnasium import spaces # type: ignore
import numpy as np # type: ignore

class VirtualFurnaceEnv(gym.Env):
    """
    Virtual Furnace v8.0 (Calibrated Physics)
    Changes:
    1. Reduced A_deg (Degradation Rate) by 10x. 
       This makes the material more stable, matching real-world BaZrS3.
    2. Max Time = 300 (5 Hours) to allow for a perfect soak.
    """
    def __init__(self):
        super(VirtualFurnaceEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), 
            high=np.array([1, 1, 1, 1]), 
            dtype=np.float32
        )
        
        # --- PHYSICS (The Fix) ---
        self.Ea_form_R = 13200.0   
        self.A_form = 50000.0      
        
        self.Ea_deg_R = 27600.0
        # CHANGED: Lowered from 4.0e10 to 5.0e9
        # This makes the material 8x more stable, opening the synthesis window.
        self.A_deg = 5.0e9        
        
        self.dt = 1.0  
        self.max_time = 300

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # TRAINING: Random Start (300-700K)
        self.temp = float(np.random.uniform(300, 700))
        
        # TESTING: Deterministic Start
        if options and 'temp' in options:
            self.temp = float(options['temp'])
            
        self.state = np.array([1.0, 0.0, 0.0]) 
        self.time_step = 0
        return self._get_obs(), {}

    def step(self, action):
        if action == 0: self.temp -= 10
        elif action == 2: self.temp += 10
        self.temp = np.clip(self.temp, 300, 1600)
        
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
        
        # --- REWARD ---
        delta_yield = (moles_forming - moles_burning)
        # Boosted reward to encourage finding the new stable window
        reward = delta_yield * 2000.0 
        
        # Penalty only kicks in if impurity gets dangerous (>5%)
        if self.state[2] > 0.05:
             reward -= (self.state[2] * 50.0)
        
        if done:
            reward += self.state[1] * 50.0
            
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return np.array([
            self.temp / 1600.0, 
            self.state[1], 
            self.state[2], 
            self.time_step / self.max_time
        ], dtype=np.float32)