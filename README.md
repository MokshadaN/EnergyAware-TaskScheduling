# Energy-Aware Cloud Task Scheduling with Deep Q-Networks (DQN)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37%2B-FF4B4B.svg)](https://streamlit.io/)
[![Vercel](https://img.shields.io/badge/Vercel-Static%20Deployment-000000.svg)](https://vercel.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

An enhanced **Double Deep Q-Network (Double-DQN)** implementation with **Prioritized Experience Replay (PER)** and **Two-Stage Reward Processing** for energy-aware task scheduling in heterogeneous cloud computing environments.

This repository implements, evaluates, and extends the multi-objective reinforcement learning approach introduced by **Janjani et al. (IEEE TCSS 2025)**, analyzing how task allocation policies optimize power consumption, SLA deadline compliance, response time, and virtual machine (VM) utilization.

---

## 📌 Research Basis

> **Reference Paper:**  
> Harshal Janjani, Tanmay Agarwal, M. P. Gopinath, Vimoh Sharma, and S. P. Raja.  
> *“Designing Energy-Aware Scheduling and Task Allocation Algorithms for Online Reinforcement Learning Applications in Cloud Environments.”*  
> **IEEE Transactions on Computational Social Systems**, 12(3), 1218-1232, 2025.  
> DOI: [10.1109/TCSS.2024.3508089](https://doi.org/10.1109/TCSS.2024.3508089)

---

## 💡 Problem Statement

Cloud computing tasks arrive dynamically with heterogeneous computing requirements (MIPS) and strict deadline constraints. Simultaneously, Virtual Machines (VMs) in a cloud data center vary in processing capacity, current utilization, and power consumption rates ($\text{kW}/\text{MIPS}$).

Conventional static or heuristic placement policies (such as First-Come First-Served or Round Robin):
1. Create utilization hotspots on high-capacity servers.
2. Under-utilize energy-efficient VM capacity.
3. Increase overall data-center energy expenditure and deadline violation rates.

**Objective:** Learn an optimal online task assignment policy $\pi: \mathcal{S} \rightarrow \mathcal{A}$ that minimizes total energy consumption $E_{\text{total}}$ and Average Task Loss Probability (ATLP) while maximizing deadline satisfaction ($T_{\text{resp}} \le d_i$) and cluster load balance.

---

## 🚀 Key Features & RL Architecture

- **Heterogeneous Cloud Environment:** Simulates multi-tier virtual machines (high-, medium-, and low-capacity) with distinct power consumption rates ($\eta_j$).
- **Compact State Vector ($S_t \in \mathbb{R}^6$):** Captures system uptime, weighted CPU utilization, memory utilization, disk I/O, RAM state, and cluster load balance factor.
- **Double DQN Target Selection:** Eliminates Q-value overestimation bias using decoupled action selection and target evaluation:
  $$Y_t^{\text{DoubleQ}} = R_{t+1} + \gamma Q\left(S_{t+1}, \underset{a}{\operatorname{argmax}} Q(S_{t+1}, a; \theta); \theta^-\right)$$
- **Prioritized Experience Replay (PER):** Samples transitions proportional to TD-error magnitude ($P(i) \propto |\delta_i|^\alpha$), accelerating policy convergence.
- **Soft Target Updates ($\tau = 0.01$):** Ensures stable policy learning via Polyak averaging:
  $$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$$
- **Two-Stage Reward Processing:** 
  - *Stage 1 ($R_1$):* Resource utilization and memory/disk overflow penalty.
  - *Stage 2 ($R_2$):* Balance indicator penalizing variance between CPU and RAM loads.
- **Robust Loss Function:** Uses Huber loss (Smooth L1) with gradient clipping ($\text{max\_norm} = 1.0$).

---

## 📊 Evaluation & Benchmark Results

Synthetic workloads were evaluated across 50, 100, 200, 500, and 1000 tasks allocated over 10-12 heterogeneous virtual machines under reproducible random seeds ($S=42$).

### Energy Consumption Comparison (at 1000 Tasks)

| Policy | Scheduling Principle | Energy (kWh) @ 1000 Tasks | Difference vs DQN |
| :--- | :--- | :---: | :---: |
| **FCFS** | First-Come First-Served (Arrival Order) | 113.90 kWh | +24.5% higher |
| **EDF** | Earliest Deadline First | 131.25 kWh | +34.5% higher |
| **Round Robin** | Uniform Task Rotation | 120.56 kWh | +28.7% higher |
| **MOABCQ** | Multi-Objective ABC Q-learning Comparator | 114.76 kWh | +33.4% higher |
| **TaskSchedulingDQN** | **Enhanced Double DQN Scheduler** | **86.00 kWh** | **Reference (Lowest)** |

> **Key Observation:** At 50 tasks, DQN consumed 64.24 kWh due to initial $\epsilon$-greedy exploration. From 100 tasks onward, as Q-values converged, DQN consistently achieved the lowest energy consumption across all benchmarked policies.

---

## 🌐 Interactive Web Presentation & Streamlit Dashboard

### Vercel Static Presentation Site (`index.html`)
The Vercel static presentation site includes:
- Research basis and IEEE publication citation details.
- Interactive multi-policy browser simulator (FCFS, EDF, Round Robin, MOABCQ, TaskSchedulingDQN).
- Real-time SVG cumulative energy traces and VM utilization distribution heatmaps.
- Mathematical formulations and system architecture breakdown.
- Gallery of stored repository experiment figures from `results/`.

### Streamlit Web Dashboard (`app.py`)
Run the interactive Streamlit dashboard locally for scenario modeling:
```bash
streamlit run app.py
```

---

## 💻 Local Setup & Execution

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/MokshadaN/EnergyAware-TaskScheduling.git
cd EnergyAware-TaskScheduling

python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Main Benchmark & Training Loop
```bash
python main.py
```
This executes the 300-iteration training loop, evaluates baselines across workload scales (50 to 1000 tasks), and saves performance dashboards to `results/`.

### 4. Run Hyperparameter Tuning
```bash
python hyperparameter_tuning.py
```

---

## 📁 Repository Structure

```text
EnergyAware-TaskScheduling/
├── agents/
│   ├── __init__.py
│   ├── dqn.py                # Deep Q-Network model architecture (PyTorch)
│   └── scheduler.py          # SystemScheduler with PER and Double DQN logic
├── core/
│   ├── __init__.py
│   ├── environment.py        # CloudEnvironment simulation engine
│   └── models.py             # Task and Resource dataclass models
├── utils/
│   ├── __init__.py
│   ├── helpers.py            # Random seed configuration and plotting wrappers
│   ├── metrics.py            # ATLP and evaluation metric functions
│   └── visualizer.py         # Multi-panel matplotlib & seaborn visualizer
├── results/                  # Stored model weights (.pth) and benchmark figures (.png)
│   ├── best_scheduler_model.pth
│   ├── dashboard.png
│   ├── baseline_comparison_detailed.png
│   └── training_results.png
├── docs/                     # Detailed project report
│   └── EnergyAware_TaskScheduling_Report.docx
├── assets/                   # Web site CSS styles and JS simulation engine
│   ├── styles.css
│   └── app.js
├── index.html                # Vercel-ready interactive web presentation
├── vercel.json               # Vercel deployment configuration
├── app.py                    # Streamlit interactive cloud scheduler dashboard
├── main.py                   # Main training, evaluation, and plotting script
├── hyperparameter_tuning.py  # Grid-search hyperparameter tuner module
├── requirements.txt          # Python dependency specifications
├── LICENSE                   # Apache-2.0 License
└── README.md                 # Project documentation
```

---

## 🚀 Deployment Instructions

### Deploy Web Presentation to Vercel
1. Push repository changes to GitHub.
2. Import repository into [Vercel](https://vercel.com/new).
3. Vercel automatically detects `index.html` and `vercel.json`.
4. Click **Deploy** (No environment variables required).

---

## 📄 License

This project is licensed under the **Apache-2.0 License** - see the [LICENSE](LICENSE) file for details.
