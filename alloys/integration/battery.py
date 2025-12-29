import gymnasium as gym # type: ignore
from gymnasium import spaces # type: ignore
import numpy as np # type: ignore

class BatteryInterfaceEnv(gym.Env):
    """
    Battery Interface v2.0 (Continuous Control)
    - Action: Float [0.0, 1.0] mA/cm²
    - Result: Perfectly smooth current profiles.
    """
    def __init__(self):
        super(BatteryInterfaceEnv, self).__init__()
        
        # CONTINUOUS ACTION SPACE
        # The agent can choose ANY number between 0 and 1.
        self.action_space = spaces.Box(
            low=np.array([0.0]), 
            high=np.array([1.0]), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([1000, 1000, 5000]), 
            dtype=np.float32
        )
        
        self.critical_current_density = 0.8 
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
        # 1. The Agent's Intent
        requested_J = float(np.clip(action[0], 0.0, 1.0))
        
        # 2. The Physics Governor (Hard Constraint)
        # If SEI < 8nm, the hardware PHYSICALLY restricts current to 0.3.
        # The agent can "ask" for 1.0, but it only "gets" 0.3.
        if self.sei_thickness < 8.0:
            actual_J = min(requested_J, 0.3)
        else:
            actual_J = requested_J
            
        # 3. Physics (Uses actual_J)
        # Tunneling Physics (Exponential Decay)
        growth_factor = 1.0 + (actual_J * 20.0) 
        passivation_damping = np.exp(-self.sei_thickness / 5.0)
        growth = self.base_sei_growth_rate * growth_factor * passivation_damping
        self.sei_thickness += growth
        
        resistance = self.sei_thickness * self.resistance_per_nm
        
        # 4. Failures (Uses actual_J)
        dendrite_fail = False
        if actual_J > self.critical_current_density:
             # Aligned thresholds
            risk_prob = 0.05 if self.sei_thickness < 8.0 else 0.001
            if np.random.random() < risk_prob: dendrite_fail = True
            
        choke_fail = self.sei_thickness > 50.0
        
        # 5. Reward (The Reality Check)
        # You get paid for what you DID (actual_J), not what you WANTED (requested_J).
        reward = actual_J * 15.0  
        
        # Joule Heating
        joule_heating = (actual_J ** 2) * resistance
        reward -= (joule_heating * 0.5)
        
        # 6. The "Wasted Effort" Penalty
        # If you asked for 0.8 but got clamped to 0.3, you are wasting control effort.
        # This teaches the agent to just output 0.3 efficiently.
        overshoot = requested_J - actual_J
        if overshoot > 0:
            reward -= (overshoot * 1.0) # Small nudge to be precise

        # Standard Failure Penalties
        done = False
        if dendrite_fail: reward -= 500.0; done = True
        elif choke_fail: reward -= 100.0; done = True
        elif self.time_step >= self.max_steps:
            done = True
            if actual_J > 0.2: reward -= 200.0

        self.charge_stored += actual_J * 1.0
        self.time_step += 1
        
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return np.array([self.sei_thickness, self.sei_thickness * self.resistance_per_nm, self.charge_stored], dtype=np.float32)