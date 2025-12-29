import numpy as np # type: ignore
from sklearn.gaussian_process import GaussianProcessRegressor # type: ignore
from sklearn.gaussian_process.kernels import Matern # type: ignore
from scipy.stats import norm # type: ignore
import warnings

warnings.filterwarnings("ignore")

class RealPhysicsOptimizer:
    def __init__(self):
        # Physical Constants (Literature Values)
        # Radii in picometers (Shannon Radii, VI-coord)
        self.R_S  = 184.0 # Sulfur (Host)
        self.R_Cl = 181.0 
        self.R_Br = 196.0 
        self.R_I  = 220.0 
        
        # Pauling Electronegativity
        self.X_S  = 2.58
        self.X_Cl = 3.16
        self.X_Br = 2.96
        self.X_I  = 2.66
        
        # Base Stability of pure beta-Li3PS4 (vs Li/Li+)
        self.base_voltage = 2.3 

    def run_experiment(self, composition):
        """
        Calculates Stability Window based on Electronegativity and Strain Energy.
        Input: [x_Cl, x_Br, x_I]
        """
        x_Cl, x_Br, x_I = composition
        total_doping = np.sum(composition)
        
        # 1. SOLUBILITY CHECK (Hard Limit)
        if total_doping > 1.0: return 0.0
        if total_doping < 0.01: return self.base_voltage # Pure material
        
        # 2. VOLTAGE BOOST (Electronegativity Theory)
        # Higher average electronegativity = Higher Oxidation Potential
        # We assume linear mixing of anion character.
        # Delta X relative to Sulfur
        d_Cl = self.X_Cl - self.X_S # 0.58 (Strong boost)
        d_Br = self.X_Br - self.X_S # 0.38 (Medium boost)
        d_I  = self.X_I  - self.X_S # 0.08 (Weak boost)
        
        # Voltage Gain = Coefficient * Concentration * Delta_X
        # (Coefficient 1.5 is a fitted parameter for sulfides)
        voltage_gain = 1.5 * (x_Cl * d_Cl + x_Br * d_Br + x_I * d_I)
        
        # 3. LATTICE STRAIN PENALTY (Vegard's Law deviation)
        # Strain Energy E ~ k * (delta_r)^2
        # Cl is roughly 0 strain (181 vs 184)
        # I is HUGE strain (220 vs 184)
        
        strain_Cl = (self.R_Cl - self.R_S)**2 # (3)^2 = 9
        strain_Br = (self.R_Br - self.R_S)**2 # (12)^2 = 144
        strain_I  = (self.R_I  - self.R_S)**2 # (36)^2 = 1296 (Massive!)
        
        # Total Strain Energy
        total_strain = (x_Cl * strain_Cl) + (x_Br * strain_Br) + (x_I * strain_I)
        
        # Critical Strain Limit: 
        # If strain > 300 (arbitrary unit ~ equivalent to 0.25 mol of Iodine), crystal degrades.
        strain_penalty = 0.0
        if total_strain > 300.0:
            # Structure collapses, voltage crashes
            strain_penalty = 2.0 
        elif total_strain > 100.0:
             # Mild distortion, slight voltage loss
            strain_penalty = 0.001 * total_strain
            
        # 4. FINAL CALCULATION
        final_voltage = self.base_voltage + voltage_gain - strain_penalty
        
        # Add slight experimental noise
        return final_voltage + np.random.normal(0, 0.02)

    def expected_improvement(self, X, model, y_best, xi=0.01):
        mu, sigma = model.predict(X, return_std=True)
        with np.errstate(divide='warn'):
            imp = mu - y_best - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei

    def optimize(self, iterations=20):
        print(f"{'Iter':<5} | {'Cl':<6} {'Br':<6} {'I':<6} | {'Voltage':<10} | {'Physics Note'}")
        print("-" * 65)
        
        # Init random valid points
        X_sample = []
        while len(X_sample) < 3:
            pt = np.random.uniform(0, 0.4, 3) 
            if np.sum(pt) <= 1.0: X_sample.append(pt)
        X_sample = np.array(X_sample)
        Y_sample = np.array([self.run_experiment(x) for x in X_sample])
        
        # GP Loop
        kernel = Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
        
        # Grid Search
        candidate_pool = np.random.uniform(0, 1.0, (2000, 3))
        
        for i in range(iterations):
            gp.fit(X_sample, Y_sample)
            y_best = np.max(Y_sample)
            
            ei = self.expected_improvement(candidate_pool, gp, y_best)
            best_cand_idx = np.argmax(ei)
            next_x = candidate_pool[best_cand_idx]
            
            next_y = self.run_experiment(next_x)
            
            # Physics Diagnostics
            strain_I = (self.R_I - self.R_S)**2
            current_strain = next_x[0]*9 + next_x[1]*144 + next_x[2]*strain_I
            
            note = "Stable"
            if np.sum(next_x) > 1.0: note = "Insoluble"
            elif current_strain > 300: note = "High Strain"
            
            print(f"{i+1:<5} | {next_x[0]:.2f}   {next_x[1]:.2f}   {next_x[2]:.2f}   | {next_y:<10.4f} | {note}")
            
            X_sample = np.vstack((X_sample, next_x))
            Y_sample = np.append(Y_sample, next_y)
            
        best_idx = np.argmax(Y_sample)
        best_x = X_sample[best_idx]
        print("-" * 65)
        print("OPTIMAL COMPOSITION DISCOVERED:")
        print(f"Cl: {best_x[0]:.3f} | Br: {best_x[1]:.3f} | I: {best_x[2]:.3f}")
        print(f"Max Voltage: {Y_sample[best_idx]:.4f} V")
        
        return best_x

# Run Real Physics
optimizer = RealPhysicsOptimizer()
optimizer.optimize(iterations=100)