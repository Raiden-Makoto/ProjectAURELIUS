import warnings
warnings.filterwarnings('ignore')

import pandas as pd # type: ignore
import numpy as np # type: ignore
from matminer.featurizers.composition import ElementProperty # type: ignore
from matminer.featurizers.conversions import StrToComposition # type: ignore
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor # type: ignore
from sklearn.model_selection import train_test_split, KFold, cross_val_score # type: ignore
from sklearn.metrics import r2_score, mean_absolute_error, confusion_matrix, classification_report # type: ignore
import pickle # type: ignore
import os # type: ignore

# Get script directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

if __name__ == '__main__':
    # 1. Load your local dataset
    csv_path = os.path.join(project_root, "data", "perovskite_metadata.csv")
    df = pd.read_csv(csv_path)
    print(f"Original Dataset: {len(df)} materials")

    # --- STEP 1: DOMAIN FILTERING (The Specialist) ---
    # We remove Oxides because they confuse the model for Chalcogenide predictions.
    # We keep materials containing S, Se, Te (Chalcogens) or F, Cl, Br, I (Halogens)
    targets = ['S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I']
    pattern = '|'.join(targets)
    df_specialist = df[df['formula'].str.contains(pattern) & ~df['formula'].str.contains('O')].copy()

    print(f"Specialist Dataset (No Oxides): {len(df_specialist)} materials")

    # --- STEP 2: DEDUPLICATION ---
    # Keep only the most stable entry for each formula
    df_clean = df_specialist.sort_values('e_hull', ascending=True).drop_duplicates(subset='formula', keep='first')
    print(f"Final Clean Training Set: {len(df_clean)} unique formulas")

    # 2. Featurize
    print("Featurizing...")
    df_clean = StrToComposition().featurize_dataframe(df_clean, "formula")
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    # IonProperty is great, but can be noisy. Let's trust Magpie for now if Ion fails.
    X = ep_feat.featurize_dataframe(df_clean, col_id="composition")

    # Drop metadata
    X = X.select_dtypes(include=[np.number]).drop(columns=["e_hull", "formation_energy", "band_gap"], errors='ignore')

    # 3. Define Targets
    y_stability = df_clean['e_hull']    # Stability Target
    y_bandgap = df_clean['band_gap']    # Electronic Target

    # 4. Train/Test Split for Stability Model
    X_train_stab, X_test_stab, y_train_stab, y_test_stab = train_test_split(X, y_stability, test_size=0.1, random_state=42)

    # 5. Train The Specialist
    # HistGradientBoosting is generally the SOTA for tabular data like this
    model_stability = HistGradientBoostingRegressor(
        learning_rate=0.05, 
        max_iter=1000, 
        max_depth=15, 
        l2_regularization=0.1,
        random_state=42
    )

    print("Training Judge 4.0 (Specialist)...")
    model_stability.fit(X_train_stab, y_train_stab)

    # 6. Evaluate Stability Model (The Real Test)
    y_pred_stab = model_stability.predict(X_test_stab)
    r2_stab = r2_score(y_test_stab, y_pred_stab)
    mae_stab = mean_absolute_error(y_test_stab, y_pred_stab)

    print(f"\n--- JUDGE 4.0 RESULTS ---")
    print(f"R2 Score: {r2_stab:.4f}")
    print(f"MAE: {mae_stab:.4f} eV/atom")

    # --- CRITICAL: CAN IT FIND STABLE MATERIALS? ---
    # Even if R2 is low, if it correctly identifies 'Stable' vs 'Unstable', it works.
    # Let's check Classification Accuracy.
    # Definition: Stable if e_hull <= 0.05 eV
    true_stable = y_test_stab <= 0.05
    pred_stable = y_pred_stab <= 0.05

    print("\n--- DISCOVERY ACCURACY ---")
    print(classification_report(true_stable, pred_stable, target_names=["Unstable", "Stable"]))

    # Critical Check: Is the error low enough?
    # For stability, we care about error < 0.05 eV (the thermal limit).
    if mae_stab < 0.05:
        print(">> STATUS: GREEN. Model is accurate enough for discovery.")
    else:
        print(">> STATUS: YELLOW. Caution advised on close calls.")

    # 7. Train Bandgap Model
    print("\nTraining Bandgap Model...")
    reg_bandgap = RandomForestRegressor(n_estimators=100, random_state=67)
    reg_bandgap.fit(X, y_bandgap)

    # 8. Save the brains (using original file names)
    stability_path = os.path.join(script_dir, "judge_stability.pkl")
    bandgap_path = os.path.join(script_dir, "judge_bandgap.pkl")
    with open(stability_path, "wb") as f:
        pickle.dump(model_stability, f)
    with open(bandgap_path, "wb") as f:
        pickle.dump(reg_bandgap, f)

    print("\nSUCCESS: 'The Judge' is trained and saved.")