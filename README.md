# ProjectAURELIUS

**Autonomous Rare-earth Exploration & Learning regarding Ionic Unit Structures**

An AI-driven materials science platform that autonomously discovers and optimizes synthesis protocols for advanced solid-state materials using reinforcement learning and Bayesian optimization.

---

## Overview

ProjectAURELIUS consists of two specialized modules that demonstrate end-to-end autonomous materials discovery and synthesis optimization:

### **Perovskites Module** (`perovskites/`)
- **Discovery Agent**: Evolutionary search identifies novel stable perovskite candidates
- **Synthesis Agent**: RL-optimized furnace protocols achieve **87.6% yield** for CaGeTe₃
- [Detailed Documentation →](perovskites/readme.md)

### **Alloys Module** (`alloys/`)
- **Synthesis Optimization**: RL agent discovers dynamic thermal profiles for β-Li₃PS₄ synthesis (**83.7% purity**)
- **Compositional Optimization**: Bayesian optimization finds optimal halide doping compositions
- **Battery Integration**: High-throughput charging strategies for solid-state electrolytes
- [Detailed Documentation →](alloys/readme.md)

---

## Key Features

- **Autonomous Discovery**: AI agents explore chemical space without human intervention
- **Physics-Informed Models**: Calibrated to literature values for realistic synthesis simulation
- **Multi-Objective Optimization**: Handles complex trade-offs (stability vs. conductivity, purity vs. yield)
- **Defect Chemistry Validation**: Validates physical viability of discovered compositions

---

## Project Structure

```
ProjectAURELIUS/
├── perovskites/          # Perovskite discovery & synthesis
│   ├── model/            # Stability prediction models
│   └── synthesis/        # RL furnace optimization
├── alloys/               # Solid-state electrolyte optimization
│   ├── doping/           # Compositional optimization
│   └── integration/      # Battery formation cycles
└── requirements.txt      # Python dependencies
```
