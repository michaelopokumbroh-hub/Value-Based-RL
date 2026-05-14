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
<img width="1935" height="693" alt="task1_training_curve" src="https://github.com/user-attachments/assets/97ebc4fa-6fb5-4e75-a645-9615559ed0b1" />
<img width="1935" height="693" alt="task2_training_curve" src="https://github.com/user-attachments/assets/7a9d88b4-2a83-4604-bbee-914da9775086" />
<img width="2235" height="1181" alt="task3_per_seed" src="https://github.com/user-attachments/assets/f7608e91-4d20-4d40-9d39-0be32722357d" />
<img width="2085" height="770" alt="task3_training_curve" src="https://github.com/user-attachments/assets/a5876337-2ef5-4f5f-98f9-e67a8d906d17" />
<img width="1635" height="733" alt="comparison_curve" src="https://github.com/user-attachments/assets/1c815300-4f83-4db9-9922-0553d67dba7e" />




