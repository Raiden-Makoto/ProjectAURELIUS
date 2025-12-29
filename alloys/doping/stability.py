import sys
import os
from pathlib import Path

# Add the directory containing this file to sys.path to allow imports
# This ensures the script can be run from any directory
_current_dir = Path(__file__).parent.absolute()
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from dopant import RealPhysicsOptimizer

class MaterialsValidator:
    def __init__(self):
        # Ionic Properties (Charge, Radius in pm)
        self.ions = {
            'Li': {'charge': +1, 'r': 76},
            'P':  {'charge': +5, 'r': 38},
            'S':  {'charge': -2, 'r': 184},
            'Cl': {'charge': -1, 'r': 181},
            'Br': {'charge': -1, 'r': 196},
            'I':  {'charge': -1, 'r': 220}
        }
        # Host: Beta-Li3PS4
        self.host_sites = {'Li': 3, 'P': 1, 'S': 4}

    def validate(self, x_Cl, x_Br, x_I):
        print(f"\nVALIDATING COMPOSITION: Li3 P S(4-x) [Cl{x_Cl} Br{x_Br} I{x_I}]")
        print("-" * 60)

        total_dopant = x_Cl + x_Br + x_I
        
        # 1. CHARGE BALANCE CHECK (Aliovalent Substitution)
        # We are replacing S(2-) with Halogen(1-).
        # Charge mismatch per dopant atom = +1.
        # Total excess positive charge = Total Dopant Amount.
        excess_charge = total_dopant * 1.0
        
        print(f"\tAliovalent Mismatch: +{excess_charge:.2f}")
        
        # Compensation Mechanism: Lithium Vacancies (Li -> V_Li)
        # Removing 1 Li (+1) compensates for 1 Halogen substitution.
        li_remaining = 3.0 - excess_charge
        
        print(f"\tCompensation: Creating {excess_charge:.2f} Lithium Vacancies")
        print(f"\tNew Formula: Li_{li_remaining:.2f} P S_{4-total_dopant:.2f} X_{total_dopant:.2f}")

        # 2. STABILITY CHECKS
        is_stable = True
        reasons = []

        # Check A: Lithium Depletion Limit
        # If Li < 2.0, conductivity usually collapses (not enough carriers).
        if li_remaining < 2.0:
            is_stable = False
            reasons.append(f"CRITICAL: Lithium content too low ({li_remaining:.2f}). Lattice will collapse.")
        
        # Check B: Lattice Strain (Vegard's Law approximation)
        # Calculate average anion radius
        r_S = self.ions['S']['r']
        avg_r_dopant = (x_Cl * self.ions['Cl']['r'] + 
                        x_Br * self.ions['Br']['r'] + 
                        x_I  * self.ions['I']['r']) / total_dopant if total_dopant > 0 else r_S
        
        strain_percent = ((avg_r_dopant - r_S) / r_S) * 100
        print(f"\tLattice Strain: {strain_percent:+.2f}%")

        if abs(strain_percent) > 5.0:
            is_stable = False
            reasons.append(f"CRITICAL: Lattice strain too high ({strain_percent:.2f}%). Phase separation likely.")

        # Check C: The "Argyrodite Zone"
        # If doping is around 0.5 - 0.8, it might form a stable Argyrodite phase (Li6PS5Cl type)
        # This is actually GOOD, but it changes the phase.
        phase_type = "Beta-Li3PS4 (Doped)"
        if 0.5 <= total_dopant <= 1.2:
            phase_type = "Argyrodite-like (High Stability Candidate)"

        # 3. VERDICT
        print("-" * 60)
        if is_stable:
            print(f"RESULT: STABLE MATERIAL ({phase_type})")
            print("\tStructure can accommodate these defects.")
        else:
            print("RESULT: UNSTABLE")
            for r in reasons:
                print(f"\t- {r}")

if __name__ == "__main__":
    optimizer = RealPhysicsOptimizer()
    best_x = optimizer.optimize(iterations=100)
    validator = MaterialsValidator()
    validator.validate(*best_x)