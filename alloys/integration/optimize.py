import sys
import os
# Add the integration directory to path so imports work from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from battery import BatteryInterfaceEnv
from stable_baselines3 import PPO # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore

# 1. Init Environment
env = BatteryInterfaceEnv()

# 2. Train Agent
print("🔋 Starting Interface Stabilization Training...")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
print("✅ Training Complete.")

# 3. Test the "Formation Protocol"
obs, _ = env.reset()
done = False
history_current = []
history_sei = []
history_charge = []

print("\nRunning Battery Diagnostic Cycle...")
while not done:
    action, _ = model.predict(obs)
    
    # --- CHANGE THIS PART ---
    # Discrete was: J = action
    # Continuous is:
    J = action[0] # Extract float from array
    
    obs, _, done, _, _ = env.step(action)
    
    # Recording for plot
    history_current.append(J)
    history_sei.append(obs[0]) # Thickness
    history_charge.append(obs[2]) # Capacity

# 4. Visualization
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Time (min)')
ax1.set_ylabel('Current Density (mA/cm²)', color=color)
ax1.plot(history_current, color=color, label='Charging Protocol')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('SEI Thickness (nm)', color=color)
ax2.plot(history_sei, color=color, linestyle='--', label='SEI Growth')
ax2.tick_params(axis='y', labelcolor=color)
ax2.axhline(y=50, color='grey', linestyle=':', label='Failure Limit')

plt.title(f"Agent-Optimized Formation Cycle (Total Charge: {history_charge[-1]:.1f} mAh)")
fig.tight_layout()

# Save to file (in the integration directory)
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fcycle.png")
plt.savefig(output_path)
print(f"Graph saved to '{output_path}'")