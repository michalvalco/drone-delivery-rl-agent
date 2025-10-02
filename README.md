# 🚁 Drone Delivery Router - Reinforcement Learning Project

**Author:** Michal Valčo (Tool used: Claude.ai/Sonnet4.5) 
**Course:** AI Agenti (AI Agents) - Online Course  
**Lesson:** 10 - RL Agent – Praktický Projekt (Practical Project)  
**Date:** September 2025

---

## 📋 Project Overview

This project implements a Deep Q-Network (DQN) agent that learns to operate autonomous delivery drones in a grid-based environment. The agent must master multiple concurrent objectives:

- **Package Delivery**: Pick up and deliver packages to their destinations
- **Battery Management**: Monitor and maintain sufficient battery charge
- **Route Optimization**: Find efficient paths between locations
- **Priority Handling**: Process packages based on priority levels
- **Resource Utilization**: Use charging stations strategically

The environment is built using OpenAI Gymnasium standards and implements a custom reward structure that encourages strategic decision-making rather than simple shortest-path solutions.

### 🎯 Learning Objectives

This project demonstrates:
1. **Environment Design**: Creating custom RL environments from scratch
2. **DQN Implementation**: Deep Q-Learning with experience replay and target networks
3. **Multi-Objective Optimization**: Balancing competing goals (speed, battery, completeness)
4. **Real-World Constraints**: Working with limited resources (battery) and time limits
5. **Software Engineering**: Clean architecture, modular code, proper documentation

---

## 🏗️ Project Architecture

```
drone-delivery-rl/
│
├── environment/              # Custom Gymnasium environment
│   ├── __init__.py
│   ├── drone_env.py         # Main environment logic
│   └── entities.py          # Package and ChargingStation classes
│
├── agent/                   # DQN agent implementation
│   ├── __init__.py
│   ├── dqn_agent.py        # Agent with integrated replay buffer
│   └── replay_buffer.py    # Replay buffer (standalone reference)
│
├── utils/                   # Utilities for logging and visualization
│   ├── __init__.py
│   ├── logger.py           # Training progress logger
│   └── visualization.py    # Plotting and metrics visualization
│
├── train.py                 # Training script
├── evaluate.py             # Evaluation script
├── test.py                 # Unit tests (pytest)
│
├── models/                  # Saved model checkpoints
├── logs/                    # Training logs (CSV format)
├── results/                 # Plots and analysis outputs
│
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
└── _Quick Start Guide.md  # Quick reference
```

---

## 🎮 Environment Specifications

### State Space (12 dimensions)

The observation is a 12-dimensional vector, normalized to [0, 1]:

| Index | Feature | Description |
|-------|---------|-------------|
| 0-1 | Drone Position | Normalized (x, y) coordinates |
| 2 | Battery Level | Current battery charge [0, 1] |
| 3 | Carrying Package | Binary: 1 if carrying, 0 otherwise |
| 4 | Package Priority | Priority level of target package [0, 1] |
| 5-6 | Target Location | Next pickup or delivery location |
| 7-8 | Nearest Charger | Location of closest charging station |
| 9 | Distance to Target | Normalized Euclidean distance |
| 10 | Distance to Charger | Normalized distance to nearest station |
| 11 | Time Remaining | Fraction of episode time left |

### Action Space (5 discrete actions)

| Action | Description | Battery Cost |
|--------|-------------|--------------|
| 0 | Move North (↑) | -0.02 |
| 1 | Move South (↓) | -0.02 |
| 2 | Move East (→) | -0.02 |
| 3 | Move West (←) | -0.02 |
| 4 | Interact | Varies (context-sensitive) |

**Action 4 (Interact)** behavior depends on position:
- At **package location** (not carrying): Pick up package (+10 reward)
- At **destination** (carrying package): Deliver package (+100×priority reward)
- At **charging station**: Charge battery (+5 reward, +10% battery)
- **Invalid location**: Penalty (-5 reward)

### Reward Structure

| Event | Reward | Notes |
|-------|--------|-------|
| Successful Delivery | +100×priority | Scaled by package priority (1-3) |
| Package Pickup | +10 | Encourages starting deliveries |
| Charging | +5 | Small incentive for battery management |
| Movement | -1 | Cost per step to encourage efficiency |
| Invalid Action | -5 | Penalty for attempting invalid interactions |
| Battery Depleted | -50 | Terminal failure state |
| Timeout (incomplete) | -30 | Penalty for running out of time |

### Episode Termination

**Success (terminated=True)**:
- All packages delivered

**Failure (terminated=True)**:
- Battery reaches 0%

**Truncation (truncated=True)**:
- Maximum steps (200) reached

---

## 🧠 DQN Algorithm Details

### Network Architecture

```
Input (12) → Dense(128, ReLU) → Dense(128, ReLU) → Output(5)
```

- **Input Layer**: 12 state features
- **Hidden Layers**: 2 layers of 128 neurons each with ReLU activation
- **Output Layer**: 5 Q-values (one per action)
- **Total Parameters**: ~17,000

### Algorithm Components

#### 1. Experience Replay
- **Buffer Size**: 10,000 transitions
- **Sampling**: Uniform random sampling
- **Purpose**: Break correlation between consecutive experiences

#### 2. Target Network
- **Update Frequency**: Every 10 episodes
- **Method**: Hard copy of online network weights
- **Purpose**: Stabilize training by providing consistent targets

#### 3. ε-Greedy Exploration
- **Initial ε**: 1.0 (100% random)
- **Final ε**: 0.01 (1% random)
- **Decay**: Multiplicative decay per episode (0.995)
- **Formula**: ε_new = max(ε_min, ε × 0.995)

#### 4. Training Details
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 64
- **Discount Factor (γ)**: 0.99
- **Gradient Clipping**: Max norm of 1.0

### Training Process

1. **Initialization**: Random weights, full replay buffer capacity
2. **Episode Loop**:
   - Reset environment
   - While not done:
     - Select action (ε-greedy)
     - Execute action, observe reward and next state
     - Store transition in replay buffer
     - Sample mini-batch and perform gradient descent
3. **End of Episode**:
   - Update target network (if scheduled)
   - Decay exploration rate
   - Log metrics

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/michalvalco/drone-delivery-rl.git
cd drone-delivery-rl
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Dependencies

```
gymnasium>=0.29.0
numpy>=1.24.0
torch>=2.0.0
pygame>=2.5.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

---

## 📖 Usage Guide

### Training the Agent

**Basic training:**
```bash
python train.py
```

**Custom training configuration:**
```bash
python train.py --episodes 2000 --lr 0.0005 --save-freq 500 --experiment my_run
```

**All training options:**
```bash
python train.py --help
```

Key parameters:
- `--episodes`: Number of training episodes (default: 1000)
- `--grid-size`: Environment grid size (default: 20)
- `--packages`: Number of packages per episode (default: 3)
- `--stations`: Number of charging stations (default: 4)
- `--lr`: Learning rate (default: 0.001)
- `--gamma`: Discount factor (default: 0.99)
- `--epsilon-decay`: Exploration decay rate (default: 0.995)
- `--save-freq`: Model checkpoint frequency (default: 100 episodes)
- `--render-freq`: Visualization frequency (default: 0=disabled)
- `--experiment`: Name for the training run

**Expected Training Time:**
- CPU: 30-60 minutes for 1000 episodes
- GPU: 10-20 minutes for 1000 episodes

### Evaluating the Trained Agent

**Basic evaluation:**
```bash
python evaluate.py --model models/dqn_agent_final.pth
```

**Extended evaluation:**
```bash
python evaluate.py --model models/dqn_agent_final.pth --episodes 50 --detailed
```

**Evaluation without rendering:**
```bash
python evaluate.py --model models/dqn_agent_final.pth --no-render --episodes 100
```

### Running Tests

```bash
pytest test.py -v
```

Or with coverage:
```bash
pytest test.py -v --cov=. --cov-report=html
```

---

## 📊 Expected Results

### Training Performance

After 1000-2000 episodes of training:

| Metric | Value |
|--------|-------|
| **Success Rate** | 70-85% |
| **Average Reward** | 150-220 |
| **Average Deliveries** | 2.1-2.5 / 3 |
| **Average Steps** | 80-120 |
| **Final ε** | 0.01 |

### Learning Phases

1. **Episodes 0-300**: Random exploration
   - Success rate: <10%
   - Agent learns basic navigation and battery mechanics

2. **Episodes 300-700**: Strategy development
   - Success rate: 20-50%
   - Agent discovers pickup-delivery-charge cycle
   - Learns to prioritize high-priority packages

3. **Episodes 700-1000+**: Policy refinement
   - Success rate: 70-85%
   - Near-optimal routes
   - Proactive battery management
   - Consistent multi-package deliveries

### Visualization Outputs

Training generates:
- `results/training_metrics.png`: Rewards, losses, deliveries over time
- `logs/experiment_[timestamp].csv`: Detailed episode-by-episode data
- `logs/experiment_[timestamp]_hyperparameters.txt`: Configuration used

---

## 🐛 Troubleshooting

### Common Issues

**1. Import Error: "No module named 'agents'"**
- Fixed in this version - imports use correct `agent` directory

**2. Pygame display not working**
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install python3-pygame

# Set display for WSL
export DISPLAY=:0
```

**3. CUDA out of memory**
```bash
# Force CPU usage
python train.py --device cpu
```

**4. Slow training**
- Reduce `--episodes` or `--grid-size`
- Disable rendering: `--render-freq 0`
- Use GPU if available

---

## 🔬 Experimental Extensions (Optional)

Potential improvements for Phase 2:

1. **Algorithm Enhancements**
   - Prioritized Experience Replay
   - Dueling DQN architecture
   - Multi-step returns (n-step DQN)
   - Noisy Networks for exploration

2. **Environment Complexity**
   - Dynamic obstacles (buildings, no-fly zones)
   - Weather effects on battery consumption
   - Multiple drones (multi-agent RL)
   - Real-world drone physics

3. **Practical Applications**
   - Integration with actual drone APIs
   - Real map data (OpenStreetMap)
   - Customer time windows
   - Vehicle capacity constraints

---

## 📝 Academic Context

This project was developed as the practical assignment for **Lesson 10: RL Agent – Praktický Projekt** in the AI Agents online course (September 2025).

### Key Learning Outcomes

✓ Understanding reinforcement learning fundamentals  
✓ Implementing value-based methods (Q-learning, DQN)  
✓ Designing reward functions for complex behaviors  
✓ Managing exploration vs. exploitation trade-offs  
✓ Applying deep learning to decision-making problems  
✓ Engineering RL systems with proper software practices

### Alignment with Course Objectives

The project follows the course progression:
- Lessons 1-4: AI APIs, databases, workflows → Used for environment design
- Lessons 5-8: Agent frameworks, LangChain, AutoGen → Influenced architecture
- **Lesson 9**: Intro to RL → Direct foundation for this project
- **Lesson 10**: Practical RL Project → This implementation

---

## 📄 License

This project is released under the MIT License for educational purposes.

---

## 🎓 Author

**Prof. Michal Valčo**  
University Professor & Researcher  
Specializations: AI Ethics, Theology, Industry 4.0, Philosophy

*This project demonstrates the intersection of autonomous systems, resource optimization, and ethical AI design - reflecting real-world challenges in drone delivery logistics.*

---

## 🙏 Acknowledgments

- Course: **AI Agenti** - September 2025 cohort
- Framework: OpenAI Gymnasium
- Deep Learning: PyTorch
- Visualization: Matplotlib, Pygame

---

**Happy Training! 🚁📦**