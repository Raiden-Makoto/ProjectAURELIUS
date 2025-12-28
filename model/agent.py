import warnings
warnings.filterwarnings('ignore')

import pandas as pd # type: ignore
import numpy as np # type: ignore
import pickle # type: ignore   
import random # type: ignore
import time # type: ignore
import os # type: ignore
# Disable multiprocessing for matminer to avoid spawn issues on macOS
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
from matminer.featurizers.composition import ElementProperty # type: ignore
from matminer.featurizers.conversions import StrToComposition # type: ignore

# Get script directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir
model_dir = os.path.join(project_root, "model")

# --- CONFIGURATION ---
# The Periodic Table of "Allowed Moves" (Chalcogenide Focused)
ACTION_SPACE = {
    'A_SITE': ['Ba', 'Sr', 'Ca', 'Eu', 'Rb', 'Cs', 'K', 'Na', 'La', 'Y'], 
    'B_SITE': ['Hf', 'Zr', 'Ti', 'Sn', 'Zn', 'Mg', 'Mn', 'Ge', 'Pb'], # Added Ge, Pb (as reference)    
    'X_SITE': ['S', 'Se', 'Te', 'Br', 'Cl', 'I']                   
}

# Lazy loading of judge to avoid multiple loads in multiprocessing
_JUDGE = None

def _load_judge():
    """Load judge model lazily, only once."""
    global _JUDGE
    if _JUDGE is None:
        judge_path = os.path.join(model_dir, "judge_stability.pkl")
        with open(judge_path, "rb") as f:
            _JUDGE = pickle.load(f)
    return _JUDGE

def get_stability(formula):
    """
    Consults the AI Oracle with ROBUST column matching.
    """
    # 1. Create a dataframe for the single formula
    df_single = pd.DataFrame({"formula": [formula]})
    
    # 2. Featurize (Chemistry -> Numbers)
    # Create featurizers fresh each time to avoid multiprocessing issues
    str_to_comp = StrToComposition()
    # Force single-threaded by setting internal attribute if possible
    if hasattr(str_to_comp, '_n_jobs'):
        str_to_comp._n_jobs = 1
    df_single = str_to_comp.featurize_dataframe(df_single, "formula")
    
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    if hasattr(ep_feat, '_n_jobs'):
        ep_feat._n_jobs = 1
    X_single = ep_feat.featurize_dataframe(df_single, col_id="composition", ignore_errors=True)
    
    # 3. Clean (Keep only numbers)
    X_single = X_single.select_dtypes(include=[np.number])
    
    # --- CRITICAL FIX: ALIGN COLUMNS ---
    # Load judge lazily
    judge = _load_judge()
    
    # We need to make sure X_single has EXACTLY the same columns as the trained model.
    # Check if the model is a VotingRegressor or a single model
    if hasattr(judge, 'feature_names_in_'):
        expected_cols = judge.feature_names_in_
    elif hasattr(judge, 'estimators_'):
        # If it's an ensemble, grab feature names from the first sub-model
        expected_cols = judge.estimators_[0].feature_names_in_
    else:
        # Fallback: If model doesn't track features (older sklearn), just proceed
        expected_cols = X_single.columns 

    # Reindex forces the columns to match. 
    # Missing cols become 0. Extra cols are dropped.
    X_aligned = X_single.reindex(columns=expected_cols, fill_value=0)
    
    # 4. Predict
    return judge.predict(X_aligned)[0]

class PerovskiteWalker:
    def __init__(self, start_formula):
        self.start_formula = start_formula
        self.current_formula = start_formula
        self.current_stability = get_stability(start_formula)
        self.best_formula = start_formula
        self.best_stability = self.current_stability
        self.history = [] # To save the path
    
    def parse_formula(self, formula):
        import re
        # Heuristic parser for ABX3
        elements = re.findall(r'([A-Z][a-z]*)', formula)
        return elements

    def mutate(self):
        # 1. Parse
        elements = self.parse_formula(self.current_formula)
        if len(elements) < 3: return self.current_formula
        
        # 2. Pick a site to mutate
        site = random.choice(['A', 'B', 'X'])
        new_elements = elements.copy()
        
        # 3. Swap element
        if site == 'A': new_elements[0] = random.choice(ACTION_SPACE['A_SITE'])
        elif site == 'B': new_elements[1] = random.choice(ACTION_SPACE['B_SITE'])
        elif site == 'X': new_elements[2] = random.choice(ACTION_SPACE['X_SITE'])
            
        # 4. Construct
        return f"{new_elements[0]}{new_elements[1]}{new_elements[2]}3"

    def walk(self, steps=200):
        print(f"🚀 LAUNCHING AGENT from {self.start_formula} (Stability: {self.current_stability:.3f} eV)")
        
        for i in range(steps):
            # Propose Mutation
            candidate = self.mutate()
            score = get_stability(candidate)
            
            # Acceptance Probability (Metropolis-like)
            # We accept better moves always.
            # We accept worse moves sometimes to escape local traps.
            diff = score - self.current_stability
            prob = np.exp(-diff / 0.05) # Temperature = 0.05 eV
            
            if diff < 0 or random.random() < prob:
                # Move Accepted
                self.current_formula = candidate
                self.current_stability = score
                
                # Check if it's a new Champion
                if score < self.best_stability:
                    self.best_stability = score
                    self.best_formula = candidate
                    print(f"Step {i:03}: 🌟 NEW CHAMPION: {candidate} (e_hull: {score:.4f})")
            
            # Log Data
            self.history.append({
                'step': i,
                'formula': candidate,
                'score': score,
                'accepted': (self.current_formula == candidate)
            })

        print(f"\n🏁 MISSION COMPLETE.")
        print(f"Top Discovery: {self.best_formula}")
        return pd.DataFrame(self.history)

if __name__ == '__main__':
    # Load judge once at startup
    print("Loading The Judge...")
    _load_judge()
    
    # --- RUN THE EXPERIMENT ---
    # We run 3 walkers starting from different "Actionable" seeds
    seeds = ["BaHfS3", "SrHfS3", "EuTiS3"]
    all_results = []

    for seed in seeds:
        agent = PerovskiteWalker(seed)
        df_history = agent.walk(steps=100)
        df_history['seed'] = seed
        all_results.append(df_history)

    # Save Results
    final_df = pd.concat(all_results)
    final_df.to_csv(os.path.join(project_root, "data", "discovery_log.csv"), index=False)
    print(f"\nData saved to '{os.path.join(project_root, 'data', 'discovery_log.csv')}'")