import gymnasium as gym # type: ignore
from gymnasium import spaces # type: ignore
import numpy as np # type: ignore

class VirtualFurnaceEnv(gym.Env):
    """
    Virtual Furnace v2.0
    Improved Reward Function to break the 60% Yield Ceiling.
    """
    def __init__(self):
        super(VirtualFurnaceEnv, self).__init__()
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32), 
            high=np.array([2000, 1.0, 1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        # --- PHYSICS (Same as before) ---
        self.Ea_form_R = 13200.0   
        self.A_form = 50000.0      
        self.Ea_deg_R = 27600.0
        self.A_deg = 4.0e10        
        self.dt = 1.0  
        self.max_time = 180

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.temp = 300.0 
        self.state = np.array([1.0, 0.0, 0.0]) # [Precursor, Target, Trash]
        self.time_step = 0
        return self._get_obs(), {}

    def step(self, action):
        if action == 0: self.temp -= 10
        elif action == 2: self.temp += 10
        self.temp = np.clip(self.temp, 300, 1400)
        
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
        
        # --- NEW REWARD FUNCTION (The Fix) ---
        delta_yield = (moles_forming - moles_burning)
        
        # 1. The Carrot: 
        # Reward creation (scaled up to be significant)
        reward = delta_yield * 2000.0 
        
        # 2. The Stick (Smoother): 
        # Old way: if impurity > 0.05 then -10 (Cliff). 
        # New way: Penalty is proportional. 0.01 impurity = -0.5 points.
        reward -= (self.state[2] * 50.0)
        
        if done:
            # 3. The Jackpot:
            # If you hit > 90% yield, you get a massive exponential bonus.
            # This forces the agent to optimize the last 10%.
            final_yield = self.state[1]
            if final_yield > 0.9:
                reward += 2000.0
            elif final_yield > 0.5:
                reward += final_yield * 100.0
            
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return np.array([
            self.temp / 1500.0, 
            self.state[1], 
            self.state[2], 
            self.time_step / self.max_time
        ], dtype=np.float32)