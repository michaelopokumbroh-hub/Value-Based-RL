# Value-Based Reinforcement Learning 

## Overview
Implementation of Deep Q-Networks (DQN) for reinforcement learning tasks.

## Tasks
| Task | Environment | Method | Result |
|------|-------------|--------|--------|
| Task 1 | CartPole-v1 | Vanilla DQN | Avg: 500/500 ✅ |
| Task 2 | Pong-v5 | Vanilla DQN (CNN) | Avg: 17.85 ✅ |
| Task 3 | Pong-v5 | DDQN + PER + Multi-step | Avg: 14.85 ✅ |

## Enhancements (Task 3)
- **Double DQN** — reduces Q-value overestimation
- **Prioritized Experience Replay** — samples important transitions more often
- **Multi-step Return (n=3)** — faster reward propagation

## How to Run

### Setup
```bash
pip install -r RL_Code/requirements.txt
```

### Evaluate Models
```bash
cd RL_Code

# Task 1
python test_model_task1.py --model-path ..\RL_task1.pt

# Task 2
python test_model_task2.py --model-path ..\RL_task2.pt

# Task 3
python test_model_task3.py --model_path ..\RL_task3_best.pt
```

## Training Curves
![Task 1](plots/task1_training_curve.png)
![Task 2](plots/task2_training_curve.png)
![Task 3](plots/task3_training_curve.png)
![Comparison](plots/comparison_curve.png)