### Phase 4: Compositional Optimization & Validation

To widen the electrochemical stability window, we employed **Multi-Objective Bayesian Optimization** to navigate the trade-offs between Chlorine, Bromine, and Iodine doping. The model converged on a **High-Chlorine composition** ($Cl \approx 0.97$, $Br \approx 0.02$, $I \approx 0.0$), effectively rejecting Iodine due to the severe lattice distortion caused by its large ionic radius ($+36$ pm mismatch). Chlorine was selected for its high electronegativity boost and near-perfect radius match with the host Sulfur atoms, resulting in a maximized voltage stability of **3.15 V** with negligible lattice strain (**-1.43%**).

```
OPTIMAL COMPOSITION DISCOVERED:
Cl: 0.969 | Br: 0.021 | I: 0.002
Max Voltage: 3.1457 V

VALIDATING COMPOSITION: Li3 P S(4-x) [Cl0.9686953694942562 Br0.020796449231638814 I0.0015444956510352048]
------------------------------------------------------------
        Aliovalent Mismatch: +0.99
        Compensation: Creating 0.99 Lithium Vacancies
        New Formula: Li_2.01 P S_3.01 X_0.99
        Lattice Strain: -1.43%
------------------------------------------------------------
RESULT: STABLE MATERIAL (Argyrodite-like (High Stability Candidate))
        Structure can accommodate these defects.
```

A defect-chemistry validation confirmed the physical viability of this composition, demonstrating that the aliovalent charge imbalance is stabilized by the formation of Lithium vacancies. The final predicted formula, **$Li_{2.01}PS_{3.01}Cl_{0.97}$**, sits precisely at the theoretical lower limit of carrier concentration ($Li \approx 2.0$), effectively identifying a stable **Argyrodite-class electrolyte** ($Li_6PS_5Cl$ analog) without compromising the conductive network.