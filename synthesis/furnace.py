import gymnasium as gym # type: ignore
from gymnasium import spaces # type: ignore
import numpy as np # type: ignore

class VirtualFurnaceEnv(gym.Env):
    """
    Simulation of CaGeTe3 Synthesis.
    FIXES:
    1. Accelerated Physics (Simulating 48h reaction in 3h steps)
    2. Dense Rewards (Giving partial credit so the agent learns faster)
    """
    def __init__(self):
        super(VirtualFurnaceEnv, self).__init__()
        
        # ACTIONS: 0=Cool, 1=Hold, 2=Heat
        self.action_space = spaces.Discrete(3)
        
        # OBS: [Temp, Yield, Impurity, Time_Left]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32), 
            high=np.array([2000, 1.0, 1.0, 1.0], dtype=np.float32), 
            dtype=np.float32
        )
        
        # --- PHYSICS ENGINE (Accelerated) ---
        # We scaled up the 'A' (frequency) factors by 20x.
        # This allows us to see "2 days of chemistry" in just "180 minutes of simulation."
        
        # Formation: Starts ~600°C
        self.Ea_form_R = 13200.0   
        self.A_form = 50000.0      # Increased from 2,500 to 50,000
        
        # Decomposition: Starts ~1000°C
        self.Ea_deg_R = 27600.0
        self.A_deg = 4.0e10        # Increased from 2e9 to 4e10
        
        self.dt = 1.0  # 1 minute per step
        self.max_time = 180 # 3 hour simulation limit

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.temp = 300.0 # Start Cold
        self.state = np.array([1.0, 0.0, 0.0]) # [Precursor, Target, Trash]
        self.time_step = 0
        return self._get_obs(), {}

    def step(self, action):
        # 1. Control
        if action == 0: self.temp -= 10
        elif action == 2: self.temp += 10
        self.temp = np.clip(self.temp, 300, 1400)
        
        # 2. Kinetics
        T = self.temp
        k_form = self.A_form * np.exp(-self.Ea_form_R / T)
        k_deg = self.A_deg * np.exp(-self.Ea_deg_R / T)
        
        # 3. Reaction
        precursor, target, impurity = self.state
        moles_forming = k_form * precursor * self.dt
        moles_burning = k_deg * target * self.dt
        
        self.state[0] -= moles_forming
        self.state[1] += (moles_forming - moles_burning)
        self.state[2] += moles_burning
        self.state = np.clip(self.state, 0, 1)
        
        # 4. Check Done
        self.time_step += 1
        done = self.time_step >= self.max_time
        
        # --- CRITICAL FIX: DENSE REWARD ---
        # We give a small reward for EVERY step where the yield increases.
        # This is the "Breadcrumb" trail the agent follows to find the heat.
        
        delta_yield = (moles_forming - moles_burning)
        step_reward = delta_yield * 1000.0 # Instant feedback
        
        # Penalty for existing impurity
        step_penalty = 0
        if self.state[2] > 0.05: step_penalty = 10.0
        
        reward = step_reward - step_penalty
        
        # Bonus at the finish line
        if done:
            reward += self.state[1] * 50
            
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return np.array([
            self.temp / 1500.0, 
            self.state[1], 
            self.state[2], 
            self.time_step / self.max_time
        ], dtype=np.float32)