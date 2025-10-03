# 🚁 Drone Delivery Router - Reinforcement Learning Project (FIXED)

**Author:** Michal Valčo (Tool used: Claude.ai/Sonnet4.5)  
**Course:** AI Agenti (AI Agents) - Online Course  
**Lesson:** 10 - RL Agent – Praktický Projekt (Practical Project)  
**Date:** September-October 2025  
**Status:** ✅ ALL CRITICAL FIXES APPLIED

---

## 🔧 What Was Fixed

This version includes **all critical fixes** from ChatReport 01:

### ✅ Fix 1: Fixed Environment Across Episodes
- **Implementation**: `fixed_env=True` with cached layout generation
- **Impact**: Consistent map enables stable learning (no chasing noise)
- **Code**: `_cached_layout` stores initial environment configuration

### ✅ Fix 2: Carrying Logic & Delivery Transitions
- **Implementation**: Proper state machine with `current_package` tracking
- **Impact**: Agent can now learn pickup → carry → deliver workflow
- **Code**: Blue drone (empty), Purple drone (carrying package)

### ✅ Fix 3: Recharge as Part of Policy
- **Implementation**: Strategic charging rewards
  - Base charging: +5 reward
  - **Low battery charging (<30%)**: +15 reward (strategic bonus)
  - Battery depletion: -50 penalty
- **Impact**: Agent learns proactive battery management

### ✅ Fix 4: Warehouse Workflow (NEW - CRITICAL)
- **Implementation**: 
  - 4×4 warehouse zone at grid origin (0,0) → (4,4)
  - **All packages spawn IN warehouse**
  - **Destinations spawn OUTSIDE warehouse** (minimum 5 cells away)
  - Warehouse contains at least one charging station
  - Drone starts at warehouse center
- **Impact**: Realistic logistics hub operation

### ✅ Fix 5: Priority-Based Package Selection (NEW - CRITICAL)
- **Implementation**: `_get_highest_priority_package()` method
  - Selects highest priority (3 > 2 > 1) first
  - Tie-breaker: closest to warehouse
- **Impact**: Agent learns to prioritize urgent deliveries

### ✅ Fix 6: Reward Structure Correction (NEW - CRITICAL)
- **Old (WRONG)**: `100 + (priority × 50)` → 150/200/250
- **New (CORRECT)**: `100 × priority` → 100/200/300
- **Impact**: Stronger learning signal for high-priority deliveries

### ✅ Fix 7: Reduced Battery Consumption
- **Change**: 0.02 → 0.01 per movement step
- **Max Steps**: 200 → 250
- **Impact**: Agent has more time to complete missions

### ✅ Fix 8: Visual Legend
- Warehouse zone highlighted in gray
- Priority-colored packages and destinations
- Blue drone (empty) vs. Purple drone (carrying)
- Complete legend panel showing all elements

---

## 📋 Project Overview

This project implements a Deep Q-Network (DQN) agent that learns to operate autonomous delivery drones in a grid-based environment. The agent must master multiple concurrent objectives:

- **Warehouse Operations**: Pick up packages from centralized hub
- **Package Delivery**: Deliver to destinations with priority awareness
- **Battery Management**: Monitor and maintain sufficient battery charge
- **Route Optimization**: Find efficient paths between locations
- **Priority Handling**: Process urgent packages first
- **Resource Utilization**: Use charging stations strategically

The environment implements a realistic logistics workflow where the drone operates from a warehouse hub, similar to real-world drone delivery systems (Amazon Prime Air, Wing, Zipline).

---

## 🏗️ Project Architecture

```
drone-delivery-rl/
│
├── environment/              # Custom Gymnasium environment
│   ├── __init__.py
│   ├── drone_env.py         # ✅ FIXED - Main environment with warehouse
│   └── entities.py          # Package and ChargingStation classes
│
├── agent/                   # DQN agent implementation
│   ├── __init__.py
│   ├── dqn_agent.py        # Agent with integrated replay buffer
│   └── replay_buffer.py    # Replay buffer (standalone reference)
│
├── utils/                   # Utilities
│   ├── __init__.py
│   ├── logger.py           # Training progress logger
│   └── visualization.py    # Plotting and metrics
│
├── train.py                 # Training script
├── evaluate.py             # Evaluation script
├── diagnose_env.py         # ✅ UPDATED - Comprehensive diagnostics
│
├── models/                  # Saved model checkpoints
├── logs/                    # Training logs (CSV format)
├── results/                 # Plots and analysis outputs
│
├── README.md               # This file (updated)
├── requirements.txt        # Python dependencies
└── _Quick Start Guide.md  # Quick reference
```

---

## 🎮 Environment Specifications

### Warehouse Zone

- **Location**: Bottom-left corner (0,0) to (3,3)
- **Purpose**: Central hub for all drone operations
- **Contains**:
  - All package spawn points
  - Primary charging station
  - Drone starting position (center at 2,2)

### State Space (12 dimensions)

The observation is a 12-dimensional vector, normalized to [0, 1]:

| Index | Feature | Description |
|-------|---------|-------------|
| 0-1 | Drone Position | Normalized (x, y) coordinates |
| 2 | Battery Level | Current battery charge [0, 1] |
| 3 | Carrying Package | Binary: 1 if carrying, 0 otherwise |
| 4 | Package Priority | **Priority of HIGHEST PRIORITY** undelivered package [0, 1] |
| 5-6 | Target Location | Next pickup (if empty) or delivery location (if carrying) |
| 7-8 | Nearest Charger | Location of closest charging station |
| 9 | Distance to Target | Normalized Euclidean distance |
| 10 | Distance to Charger | Normalized distance to nearest station |
| 11 | Time Remaining | Fraction of episode time left |

### Action Space (5 discrete actions)

| Action | Description | Battery Cost |
|--------|-------------|--------------|
| 0 | Move North (↑) | -0.01 |
| 1 | Move South (↓) | -0.01 |
| 2 | Move East (→) | -0.01 |
| 3 | Move West (←) | -0.01 |
| 4 | Interact | Context-sensitive |

**Action 4 (Interact)** behavior depends on position:
- At **package location in warehouse** (not carrying): Pick up **highest priority** package (+10 reward)
- At **destination** (carrying package): Deliver package (+100×priority reward)
- At **charging station**: Charge battery (+5 or +15 reward, +10% battery)
- **Invalid location**: Penalty (-5 reward)

### Reward Structure (CORRECTED)

| Event | Reward | Notes |
|-------|--------|-------|
| Successful Delivery | **+100×priority** | Priority 1: +100, Priority 2: +200, Priority 3: +300 |
| Package Pickup | +10 | Encourages starting deliveries |
| Charging (Normal) | +5 | Small incentive |
| **Charging (Strategic)** | **+15** | **When battery < 30% (NEW!)** |
| Movement | -1 | Cost per step (reduced from -1 in earlier version) |
| Invalid Action | -5 | Penalty for attempting invalid interactions |
| Battery Depleted | -50 | Terminal failure state |
| Timeout (incomplete) | -30 | Penalty for running out of time |
| All Delivered | +100 | Bonus for mission completion |

### Episode Termination

**Success (terminated=True)**:
- All packages delivered

**Failure (terminated=True)**:
- Battery reaches 0%

**Truncation (truncated=True)**:
- Maximum steps (250) reached

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

### Training Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Learning Rate | 0.001 | Gradient descent step size |
| Discount Factor (γ) | 0.99 | Future reward importance |
| Initial ε | 1.0 | Start with full exploration |
| Final ε | 0.01 | Minimal exploration |
| ε Decay | 0.995 | Gradual shift to exploitation |
| Batch Size | 64 | Training mini-batch |
| Replay Buffer | 10,000 | Experience storage |
| Target Update | Every 10 episodes | Stabilization frequency |

---

## 🚀 Installation & Usage

### Prerequisites

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 1: Validate Environment (CRITICAL)

**Before training**, run diagnostics to ensure all fixes are working:

```bash
python diagnose_env.py
```

**Expected output:**
```
✅ PASS - Fixed Map
✅ PASS - Warehouse
✅ PASS - Priority
✅ PASS - Transitions
✅ PASS - Battery
✅ PASS - Rewards
✅ PASS - Solvable

🎉 ALL TESTS PASSED! Environment is ready for training.
```

If any tests fail, review the error messages before proceeding.

### Step 2: Train the Agent

**Basic training:**
```bash
python train.py --episodes 1000
```

**Extended training (recommended):**
```bash
python train.py --episodes 2000 --save-freq 200 --experiment lesson10_final
```

**Training with visualization (slower):**
```bash
python train.py --episodes 500 --render-freq 50
```

### Step 3: Evaluate Performance

```bash
python evaluate.py --model models/dqn_agent_final.pth --episodes 50
```

**Expected results after 1000-2000 episodes:**
- Success Rate: 40-70%
- Average Deliveries: 2.0-2.5 / 3
- Average Reward: 150-250

---

## 📊 Expected Training Phases

### Phase 1: Episodes 0-300 (Random Exploration)
- Success rate: <10%
- Agent discovers basic mechanics (movement, battery, charging)
- High exploration (ε ≈ 1.0 → 0.7)

### Phase 2: Episodes 300-700 (Strategy Development)
- Success rate: 20-50%
- Agent learns warehouse → pickup → deliver → charge cycle
- Begins prioritizing high-value packages
- Moderate exploration (ε ≈ 0.7 → 0.3)

### Phase 3: Episodes 700-1000+ (Policy Refinement)
- Success rate: 50-70%
- Near-optimal routes
- Proactive battery management
- Consistent multi-package deliveries
- Low exploration (ε ≈ 0.3 → 0.01)

---

## 🔍 Troubleshooting

### Issue: "0% success rate after 200 episodes"
**Solution**: Ensure you're using the FIXED `drone_env.py` with:
- Warehouse workflow
- Priority-based selection
- Correct reward structure (100×priority)

### Issue: "Agent keeps running out of battery"
**Solution**: 
- Verify battery consumption is 0.01 (not 0.02)
- Check strategic charging reward (+15 when <30%)
- Ensure max_steps = 250

### Issue: "Packages not being picked up"
**Solution**:
- Verify all packages spawn in warehouse zone
- Check drone starts at warehouse center
- Confirm interact action (4) works at package positions

### Issue: "Random agent gets 0 rewards"
**Solution**: Run `diagnose_env.py` - if solvability test fails, there's a bug in the environment.

---

## 📝 Key Differences from Original Version

| Aspect | Original | Fixed |
|--------|----------|-------|
| Package Spawn | Random locations | Warehouse zone only |
| Package Selection | First available | Highest priority |
| Delivery Reward | 100 + 50×priority | **100×priority** |
| Battery Consumption | 0.02/step | 0.01/step |
| Max Steps | 200 | 250 |
| Strategic Charging | Not implemented | +15 when <30% battery |
| Warehouse Zone | Not defined | 4×4 area at origin |
| Drone Start | Random | Warehouse center |

---

## 🎓 Academic Context

This project was developed as the practical assignment for **Lesson 10: RL Agent – Praktický Projekt** in the AI Agents online course (September 2025).

### Learning Outcomes Demonstrated

✓ Environment design with realistic constraints  
✓ DQN implementation with experience replay  
✓ Reward shaping for complex multi-objective tasks  
✓ Priority-based decision making  
✓ Resource management under uncertainty  
✓ Software engineering best practices in RL

---

## 📄 License

This project is released under the MIT License for educational purposes.

---

## 🙏 Acknowledgments

- Course: **AI Agenti** - September 2025 cohort
- Framework: OpenAI Gymnasium
- Deep Learning: PyTorch
- Visualization: Matplotlib, Pygame
- AI Assistant: Claude.ai (Anthropic) for debugging and fixes

---

**Project Status**: ✅ **READY FOR SUBMISSION**  
All critical fixes from ChatReport 01 have been implemented and validated.

**Happy Training! 🚁📦**