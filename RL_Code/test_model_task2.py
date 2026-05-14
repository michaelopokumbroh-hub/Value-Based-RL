# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL - Task 2 Evaluation: ALE/Pong-v5
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
import argparse
from collections import deque

gym.register_envs(ale_py)


class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.network(x / 255.0)


class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray    = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque(
            [frame for _ in range(self.frame_stack)],
            maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


def eval_task2(model_path, episodes=20, seed=0,
               output_dir="./eval_videos_task2",
               save_video=False):
    """
    Evaluates Task 2 model on ALE/Pong-v5.
    Prints per-episode reward in TA format and final average.
    Seeds 0 to episodes-1 are used (matching TA protocol).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")

    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Environment
    env = gym.make("ALE/Pong-v5", render_mode="human")
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    preprocessor = AtariPreprocessor()
    num_actions  = env.action_space.n   # 6

    # Load model 
    model = DQN(num_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if save_video:
        os.makedirs(output_dir, exist_ok=True)

    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        state  = preprocessor.reset(obs)
        done   = False
        total  = 0
        frames = []

        while not done:
            if save_video:
                frames.append(env.render())

            t = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(t).argmax().item()

            next_obs, r, terminated, truncated, _ = env.step(action)
            env.render()
            done   = terminated or truncated
            total += r
            state  = preprocessor.step(next_obs)

        rewards.append(total)
    
        print(f"seed: {ep}, eval reward: {total:.0f}")

        if save_video and frames:
            out = os.path.join(output_dir, f"eval_task2_ep{ep}.mp4")
            with imageio.get_writer(out, fps=30) as v:
                for f in frames:
                    v.append_data(f)
            print(f"  Saved video: {out}")

    avg = np.mean(rewards)
    print(f"\nAverage reward: {avg:.2f}")
    print(f"Min reward:     {np.min(rewards):.0f}")
    print(f"Max reward:     {np.max(rewards):.0f}")

    # Task 2 grade formula from HW spec:
    # Score % = (min(avg, 19) + 21) / 40 * 20%
    grade = (min(avg, 19) + 21) / 40 * 20
    print(f"\nEstimated Task 2 score: {grade:.2f} / 20%")

    env.close()
    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",  type=str, required=True,
                        help="Path to saved .pt model")
    parser.add_argument("--output-dir",  type=str,
                        default="./eval_videos_task2")
    parser.add_argument("--episodes",    type=int, default=20)
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--save-video",  action="store_true", default=False)
    args = parser.parse_args()
    eval_task2(args.model_path, args.episodes, args.seed,
               args.output_dir, args.save_video)