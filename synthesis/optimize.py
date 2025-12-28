import sys
import os
# Add the synthesis directory to path so imports work from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
from stable_baselines3 import PPO # type: ignore
from furnace import VirtualFurnaceEnv # Imports your physics simulator

# 1. SETUP THE LAB
# We initialize the environment with the scientific values you added
env = VirtualFurnaceEnv()

# 2. HIRE THE OPERATOR (The Agent)
# We use PPO (Proximal Policy Optimization), a standard robust RL algorithm.
model = PPO("MlpPolicy", env, verbose=1)

print("👨‍🔬 TRAINING STARTED: The agent is learning thermal kinetics...")
print("    (This will take about 30 seconds on a laptop)")

# 3. TRAINING LOOP
# The agent tries 150,000 minutes of experiment time to find the pattern.
model.learn(total_timesteps=150000)
print("✅ TRAINING COMPLETE.")

# 4. THE FINAL EXAM
# We reset the furnace and let the trained agent run one perfect cycle.
obs, _ = env.reset()
done = False
path_temp = []
path_yield = []
path_impurity = []

print("\nRunning the Optimized Protocol...")
while not done:
    # The agent looks at the temp and decides: Heat? Cool? Hold?
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    
    # We log the data to plot it
    # Note: obs[0] is normalized, so we un-normalize it for the plot
    # The env clips at 1400K, but let's check the env max to be safe
    current_temp = obs[0] * 1500.0 
    
    path_temp.append(current_temp) 
    path_yield.append(obs[1])
    path_impurity.append(obs[2])

# 5. VISUALIZATION
# Plotting the "Master Recipe"
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Temperature (Red Line)
color = 'tab:red'
ax1.set_xlabel('Time (minutes)')
ax1.set_ylabel('Furnace Temp (K)', color=color)
ax1.plot(path_temp, color=color, linewidth=2, label="Temperature Profile")
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 1500)
ax1.grid(True, alpha=0.3)

# Plot Yield (Blue Line)
ax2 = ax1.twinx() 
color = 'tab:blue'
ax2.set_ylabel('Phase Fraction', color=color)
ax2.plot(path_yield, color=color, linewidth=2, label="CaGeTe3 Yield")
ax2.plot(path_impurity, color='black', linewidth=1, linestyle="--", label="Impurity (Degradation)")
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 1.1)

# Add Legend and Save
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

plt.title(f"RL-Optimized Synthesis for CaGeTe3\nFinal Yield: {path_yield[-1]*100:.1f}%")
plt.tight_layout()

# Save to file (in the synthesis directory)
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recipe.png")
plt.savefig(output_path)
print(f"💾 Recipe graph saved to '{output_path}'")
print(f"Final Yield Achieved: {path_yield[-1]*100:.2f}%")