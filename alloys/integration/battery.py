import gymnasium as gym # type: ignore
from gymnasium import spaces # type: ignore
import numpy as np # type: ignore

class BatteryInterfaceEnv(gym.Env):
    """
    Battery Interface v1.2 (Corrected Physics)
    Fix: Dendrite risk is now tied to SEI thickness.
    - Thin SEI (<10nm) + High Current = High Risk (Dendrites)
    - Thick SEI (>10nm) + High Current = Safe (Operational Mode)
    """
    def __init__(self):
        super(BatteryInterfaceEnv, self).__init__()
        self.action_space = spaces.Discrete(3) # 0=Rest, 1=Slow, 2=Fast
        
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
        # 1. Action Map
        if action == 0: J = 0.0
        elif action == 1: J = 0.1
        elif action == 2: J = 1.0 # Fast Charge
        
        # Physics: SEI Growth
        growth_factor = 1.0 + (J * 5.0) 
        growth = self.base_sei_growth_rate * growth_factor
        passivation_damping = 1.0 / (self.sei_thickness * 0.1)
        self.sei_thickness += (growth * passivation_damping)
        
        resistance = self.sei_thickness * self.resistance_per_nm
        
        # --- PHYSICS FIX: DYNAMIC DENDRITE RISK ---
        dendrite_fail = False
        
        if J > self.critical_current_density:
            # Risk depends on SEI protection
            if self.sei_thickness < 10.0:
                # Thin SEI: High Risk (5% per step)
                # This scares the agent away from fast charging too early.
                risk_prob = 0.05 
            else:
                # Thick SEI: Low Risk (0.1% per step)
                # This gives the agent confidence to ramp up later.
                risk_prob = 0.001
                
            if np.random.random() < risk_prob: 
                dendrite_fail = True
        
        choke_fail = self.sei_thickness > 50.0
        
        # --- REWARD TUNING ---
        # We boost the throughput reward to make the risk worthwhile
        reward = J * 20.0            
        reward -= resistance * 0.5   
        
        done = False
        if dendrite_fail:
            reward -= 500.0 
            done = True
        elif choke_fail:
            reward -= 100.0 
            done = True
        elif self.time_step >= self.max_steps:
            done = True
            # End-of-Life Constraint
            if J > 0.2: reward -= 200.0
            if self.sei_thickness > 20.0: reward -= 50.0

        # Break-In Constraint (Explicit Helper)
        # Still punish fast charging on a fresh battery to guide the learning
        if self.sei_thickness < 8.0 and J > 0.5:
            reward -= 200.0

        self.charge_stored += J * 1.0
        self.time_step += 1
        
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return np.array([
            self.sei_thickness,
            self.sei_thickness * self.resistance_per_nm,
            self.charge_stored
        ], dtype=np.float32)