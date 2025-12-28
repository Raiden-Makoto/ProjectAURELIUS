import pandas as pd # type: ignore
from matminer.featurizers.composition import ElementProperty # type: ignore
from matminer.featurizers.conversions import StrToComposition # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import pickle # type: ignore
import os # type: ignore

# Get script directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

if __name__ == '__main__':
    # 1. Load your local dataset
    csv_path = os.path.join(project_root, "data", "perovskite_metadata.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} materials for training.")

    # 2. Convert formula strings to Composition objects
    # This allows matminer to understand "BaHfS3" is Barium + Hafnium + Sulfur
    print("Converting formulas...")
    df = StrToComposition().featurize_dataframe(df, "formula")

    # 3. Featurize: Turn chemistry into numbers
    # We use 'MagpieData' which describes atoms by electronegativity, radii, etc.
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    print("Featurizing data (this may take a minute)...")
    X = ep_feat.featurize_dataframe(df, col_id="composition") 
    # Drop non-numeric columns to get our Feature Matrix
    columns_to_drop = ["material_id", "formula", "band_gap", "e_hull", "formation_energy", "composition"]
    X = X.drop(columns=[col for col in columns_to_drop if col in X.columns])

    # 4. Define Targets
    y_stability = df['e_hull']    # Stability Target
    y_bandgap = df['band_gap']    # Electronic Target

    # 5. Train "The Judge" (Two separate models)
    print("Training Stability Model (The Hull Predictor)...")
    reg_stability = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_stability.fit(X, y_stability)

    print("Training Bandgap Model...")
    reg_bandgap = RandomForestRegressor(n_estimators=100, random_state=67)
    reg_bandgap.fit(X, y_bandgap)

    # 6. Sanity Check
    # Let's see how well it learned on a test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_stability, test_size=0.1)
    test_score = reg_stability.score(X_test, y_test)
    print(f"Stability Model R2 Score: {test_score:.3f} (1.0 is perfect)")

    # 7. Save the brains
    stability_path = os.path.join(script_dir, "judge_stability.pkl")
    bandgap_path = os.path.join(script_dir, "judge_bandgap.pkl")
    with open(stability_path, "wb") as f:
        pickle.dump(reg_stability, f)
    with open(bandgap_path, "wb") as f:
        pickle.dump(reg_bandgap, f)

    print("\nSUCCESS: 'The Judge' is trained and saved.")