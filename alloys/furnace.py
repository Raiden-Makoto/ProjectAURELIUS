import gymnasium as gym # type: ignore
from gymnasium import spaces # type: ignore
import numpy as np # type: ignore

class AlloyFurnaceEnv(gym.Env):
    """
    Virtual Furnace v11.0 (Beta-Phase Specialist)
    Target: Metastable Beta-Li3PS4
    
    Physics Profile:
    - Formation Window: 140°C - 200°C (413K - 473K).
    - Danger Zone: > 250°C (523K) triggers rapid Gamma decay.
    - Strategy: "Pulse & Quench" (Heat precisely, then cool fast).
    """
    def __init__(self):
        super(AlloyFurnaceEnv, self).__init__()
        
        # ACTIONS: 0=Cool (-5K), 1=Hold, 2=Heat (+5K)
        # We use smaller temp steps (5K) because solvents are sensitive.
        self.action_space = spaces.Discrete(3)
        
        # OBS: [Temp, Beta_Yield, Gamma_Waste, Time_Left]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), 
            high=np.array([1, 1, 1, 1]), 
            dtype=np.float32
        )
        
        # --- PHYSICS ENGINE (Calibrated to User Literature) ---
        
        # 1. Formation (Precursor -> Beta)
        # Low Activation Energy (Ea) because it's solvent-assisted.
        # Reaction kicks in around 400K.
        self.Ea_form_R = 4000.0   
        self.A_form = 500.0      
        
        # 2. Transformation (Beta -> Gamma)
        # This is the "Trap". High barrier, but A_trans is high enough
        # that once you cross 523K, the Beta phase collapses quickly.
        self.Ea_trans_R = 9000.0
        self.A_trans = 5.0e5        
        
        self.dt = 1.0  
        self.max_time = 300 # 5 Hours total budget

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start Cold (Room Temp 300K)
        # We allow a slight random variance to robustify the agent.
        self.temp = float(np.random.uniform(295, 305))
        
        if options and 'temp' in options:
            self.temp = float(options['temp'])
            
        # State: [Precursor, Beta(Target), Gamma(Waste)]
        self.state = np.array([1.0, 0.0, 0.0]) 
        self.time_step = 0
        return self._get_obs(), {}

    def step(self, action):
        # 1. Control System
        if action == 0: self.temp -= 5 
        elif action == 2: self.temp += 5 
        
        # Safety Limits: 300K to 600K (27°C to 327°C)
        self.temp = np.clip(self.temp, 300, 600)
        
        # 2. Arrhenius Kinetics
        T = self.temp
        k_form = self.A_form * np.exp(-self.Ea_form_R / T)
        k_trans = self.A_trans * np.exp(-self.Ea_trans_R / T)
        
        prec, beta, gamma = self.state
        
        # Calculate Flows
        moles_forming = k_form * prec * self.dt       # Precursor -> Beta
        moles_decaying = k_trans * beta * self.dt     # Beta -> Gamma (Overcooking)
        
        # Update Mass Balance
        self.state[0] -= moles_forming
        self.state[1] += (moles_forming - moles_decaying)
        self.state[2] += moles_decaying
        
        # Physics Check (Mass conservation)
        self.state = np.clip(self.state, 0, 1)
        
        self.time_step += 1
        done = self.time_step >= self.max_time
        
        # --- 3. THE "DELTA JUDGE" REWARD SYSTEM ---
        
        # Reward A: Growth
        # You get paid for every milligram of NET Beta created.
        net_beta_change = (moles_forming - moles_decaying)
        reward = net_beta_change * 2000.0
        
        # Reward B: The "Pain" Signal
        # If ANY Beta is decaying into Gamma, we penalize heavily.
        # This teaches the agent to quench immediately if the stable phase appears.
        if moles_decaying > 0:
            reward -= (moles_decaying * 5000.0)
            
        # Reward C: Completion Bonus
        if done:
            reward += self.state[1] * 50.0
            
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # Normalized observation vector
        return np.array([
            self.temp / 600.0,      # Temp normalized to max range
            self.state[1],          # Current Yield
            self.state[2],          # Current Waste
            self.time_step / self.max_time
        ], dtype=np.float32)