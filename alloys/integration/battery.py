import gymnasium as gym # type: ignore
from gymnasium import spaces # type: ignore
import numpy as np # type: ignore

class BatteryInterfaceEnv(gym.Env):
    """
    Battery Interface vFinal (Fear-Free)
    - Lowered Barrier Penalty to prevent paralysis.
    - High Reward to encourage activity.
    """
    def __init__(self):
        super(BatteryInterfaceEnv, self).__init__()
        
        self.action_space = spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32)
        
        # Standard observation
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([1000, 1000, 5000]), 
            dtype=np.float32
        )
        
        self.base_sei_growth_rate = 0.05    
        self.resistance_per_nm = 0.5        
        self.max_steps = 200 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sei_thickness = 2.0 
        self.charge_stored = 0.0
        self.time_step = 0
        return self._get_obs(), {}

    def step(self, action):
        # 1. INPUT
        J = float(np.clip(action[0], 0.0, 1.0))
        
        # 2. PHYSICS
        # Fast formation logic (High sensitivity)
        growth_factor = 1.0 + (J * 20.0) 
        passivation_damping = np.exp(-self.sei_thickness / 5.0)
        growth = self.base_sei_growth_rate * growth_factor * passivation_damping
        self.sei_thickness += growth
        resistance = self.sei_thickness * self.resistance_per_nm
        
        # 3. REWARD (High Incentive)
        # We boost the base reward to make the agent WANT to work.
        reward = J * 50.0  
        
        # 4. COSTS
        
        # Cost A: Physics (Joule Heating)
        joule_heating = (J ** 2) * resistance
        reward -= (joule_heating * 1.0)
        
        # Cost B: The Constraint (The Gentle Wall)
        if self.sei_thickness < 8.0:
            if J > 0.3:
                excess = J - 0.3
                # WAS: 500.0 (Death Sentence)
                # NOW: 50.0 (Traffic Ticket)
                # At J=0.5: Excess=0.2. Penalty = 0.04 * 50 = 2.0.
                # Reward = 25.0. Net = +23.0.
                # It is STILL PROFITABLE to speed, but LESS profitable than being perfect.
                reward -= (excess ** 2) * 50.0
        
        # 5. SAFETY LIMITS
        done = False
        if self.sei_thickness > 50.0: 
            reward -= 100.0; done = True
        elif self.time_step >= self.max_steps:
            done = True
            if J > 0.2: reward -= 200.0

        self.charge_stored += J * 1.0
        self.time_step += 1
        
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return np.array([self.sei_thickness, self.sei_thickness * self.resistance_per_nm, self.charge_stored], dtype=np.float32)