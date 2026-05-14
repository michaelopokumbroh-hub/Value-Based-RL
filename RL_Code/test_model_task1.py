# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL - Task 1 Evaluation: CartPole-v1
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import numpy as np
import random
import gymnasium as gym
import argparse


class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.network(x)


def eval_task1(model_path, episodes=20, seed=0):
    """
    Evaluates Task 1 model on CartPole-v1.
    Prints per-episode reward and final average.
    Seeds 0 to episodes-1 are used (matching TA protocol).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env         = gym.make("CartPole-v1", render_mode="human")
    num_actions = env.action_space.n   # 2

    # Load model
    model = DQN(num_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done   = False
        total  = 0

        while not done:
            t = torch.from_numpy(
                np.array(obs, dtype=np.float32)
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(t).argmax().item()
            obs, r, terminated, truncated, _ = env.step(action)
            env.render()
            done   = terminated or truncated
            total += r

        rewards.append(total)
        print(f"seed: {ep}, eval reward: {total:.0f}")

    avg = np.mean(rewards)
    print(f"\nAverage reward: {avg:.2f}")
    print(f"Min reward:     {np.min(rewards):.0f}")
    print(f"Max reward:     {np.max(rewards):.0f}")

    
    grade = min(avg, 480) / 480 * 15
    print(f"\nEstimated Task 1 score: {grade:.2f} / 15%")

    env.close()
    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to saved .pt model")
    parser.add_argument("--episodes",   type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed",       type=int, default=0,
                        help="Starting random seed")
    args = parser.parse_args()
    eval_task1(args.model_path, args.episodes, args.seed)