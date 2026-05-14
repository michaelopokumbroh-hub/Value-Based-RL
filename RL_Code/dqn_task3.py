# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL - Task 3: Enhanced DQN on ALE/Pong-v5
# Enhancements: Double DQN + Prioritized Experience Replay + Multi-step Return
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse

gym.register_envs(ale_py)
os.environ['WANDB_MODE'] = 'offline'


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        ########## YOUR CODE HERE (5~10 lines) ##########
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
        ########## END OF YOUR CODE ##########

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


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (Schaul et al. 2016)
    Priority:    p_i = |delta_i| + epsilon
    Probability: P(i) = p_i^alpha / sum(p_k^alpha)
    IS weight:   w_i = (1 / N*P(i))^beta  normalized by max weight
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity   = capacity
        self.alpha      = alpha
        self.beta       = beta
        self.buffer     = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos        = 0

    def add(self, transition, error=1.0):
        ########## YOUR CODE HERE (for Task 3) ##########
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        ########## END OF YOUR CODE (for Task 3) ##########

    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########
        n          = len(self.buffer)
        priorities = self.priorities[:n]
        probs      = priorities ** self.alpha
        probs     /= probs.sum()

        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        weights  = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights  = weights.astype(np.float32)

        states, actions, rewards, next_states, dones = zip(*samples)
        ########## END OF YOUR CODE (for Task 3) ##########
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########
        eps = 1e-6
        for idx, err in zip(indices, errors):
            self.priorities[idx] = abs(err) + eps
        ########## END OF YOUR CODE (for Task 3) ##########

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env          = gym.make(env_name, render_mode="rgb_array")
        self.test_env     = gym.make(env_name, render_mode="rgb_array")
        self.num_actions  = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.q_net      = DQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size              = args.batch_size
        self.gamma                   = args.discount_factor
        self.epsilon                 = args.epsilon_start
        self.epsilon_min             = args.epsilon_min
        self.epsilon_decay_steps     = args.epsilon_decay_steps
        self.env_count               = 0
        self.train_count             = 0
        self.best_reward             = -21
        self.max_episode_steps       = args.max_episode_steps
        self.replay_start_size       = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step          = args.train_per_step
        self.save_dir                = args.save_dir
        self.student_id              = args.student_id
        self.n_step                  = args.n_step

        
        self.epsilon_decay_per_step = (
            (args.epsilon_start - args.epsilon_min) / args.epsilon_decay_steps
        )

        
        self.memory = PrioritizedReplayBuffer(
            capacity=args.memory_size,
            alpha=args.per_alpha,
            beta=args.per_beta
        )

        
        self.n_step_buffer = deque(maxlen=self.n_step)

        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Task 3 | DoubleDQN=True | PER=True | n_step={self.n_step} | "
              f"eps_decay_per_step={self.epsilon_decay_per_step:.8f}")

    def _get_n_step_info(self):
        """
        R_t^(n) = sum_{k=0}^{n-1} gamma^k * r_{t+k}
                  + gamma^n * max_a' Q(s_{t+n}, a')
        """
        reward = 0.0
        for k, (s, a, r, ns, d) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** k) * r
            if d:
                return (self.n_step_buffer[0][0],
                        self.n_step_buffer[0][1],
                        reward, ns, True)
        first = self.n_step_buffer[0]
        last  = self.n_step_buffer[-1]
        return first[0], first[1], reward, last[3], last[4]

    def _store_transition(self, s, a, r, ns, done):
        self.n_step_buffer.append((s, a, r, ns, done))
        if len(self.n_step_buffer) == self.n_step:
            transition = self._get_n_step_info()
            self.memory.add(transition)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(
            np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=3500):
        for ep in range(episodes):
            obs, _       = self.env.reset()
            state        = self.preprocessor.reset(obs)
            done         = False
            total_reward = 0
            step_count   = 0
            self.n_step_buffer.clear()

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done       = terminated or truncated
                next_state = self.preprocessor.step(next_obs)

                self._store_transition(state, action, reward, next_state, done)

                for _ in range(self.train_per_step):
                    self.train()

                # Epsilon decays
                if self.epsilon > self.epsilon_min:
                    self.epsilon = max(
                        self.epsilon_min,
                        self.epsilon - self.epsilon_decay_per_step
                    )

                state        = next_state
                total_reward += reward
                self.env_count += 1
                step_count   += 1

                if self.env_count % 1000 == 0:
                    print(f"[Collect] Ep: {ep} Step: {step_count} "
                          f"SC: {self.env_count} UC: {self.train_count} "
                          f"Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Env Step Count": self.env_count,
                        "Epsilon":        self.epsilon,
                        "Buffer Size":    len(self.memory),
                        "Update Count":   self.train_count,
                    })

            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} "
                  f"SC: {self.env_count} UC: {self.train_count} "
                  f"Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode":        ep,
                "Total Reward":   total_reward,
                "Env Step Count": self.env_count,
                "Epsilon":        self.epsilon,
            })

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                wandb.log({
                    "Eval Reward":    eval_reward,
                    "Env Step Count": self.env_count,
                })
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} "
                      f"Best: {self.best_reward:.2f}")
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(
                        self.save_dir,
                        f"LAB5_{self.student_id}_task3_best.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved best → {model_path} "
                          f"(reward={eval_reward:.1f})")

            # Milestone snapshots 
            for milestone in [600000, 1000000, 1500000, 2000000, 2500000]:
                snap = os.path.join(
                    self.save_dir,
                    f"LAB5_{self.student_id}_task3_{milestone}.pt")
                if self.env_count >= milestone and not os.path.exists(snap):
                    torch.save(self.q_net.state_dict(), snap)
                    print(f"[Milestone] Saved {snap}")

    def evaluate(self, num_episodes=3):
        total_rewards = []
        for _ in range(num_episodes):
            obs, _ = self.test_env.reset()
            state  = self.preprocessor.reset(obs)
            done, ep_reward = False, 0
            while not done:
                t = torch.from_numpy(
                    np.array(state)).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.q_net(t).argmax().item()
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done       = terminated or truncated
                ep_reward += reward
                state      = self.preprocessor.step(next_obs)
            total_rewards.append(ep_reward)
        return np.mean(total_rewards)

    def train(self):
        if len(self.memory) < self.replay_start_size:
            return

        self.train_count += 1

        ########## YOUR CODE HERE (<5 lines) ##########
        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample(self.batch_size)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        ########## END OF YOUR CODE ##########

        states      = torch.from_numpy(
            np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(
            np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones   = torch.tensor(dones,   dtype=torch.float32).to(self.device)

        q_values = self.q_net(states).gather(
            1, actions.unsqueeze(1)).squeeze(1)

        ########## YOUR CODE HERE (~10 lines) ##########
        with torch.no_grad():
            # Double DQN
            next_actions = self.q_net(next_states).argmax(dim=1)
            next_q       = self.target_net(next_states).gather(
                               1, next_actions.unsqueeze(1)).squeeze(1)

            
            gamma_n  = self.gamma ** self.n_step
            target_q = rewards + gamma_n * next_q * (1 - dones)

        td_errors = target_q - q_values

        
        loss = (weights * td_errors.pow(2)).mean()

        
        self.memory.update_priorities(
            indices, td_errors.detach().abs().cpu().numpy())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        ########## END OF YOUR CODE ##########

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} "
                  f"Q mean: {q_values.mean().item():.3f}")
            wandb.log({
                "Train Loss":     loss.item(),
                "Q Mean":         q_values.mean().item(),
                "Env Step Count": self.env_count,
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir",               type=str,
                        default="./results_task3")
    parser.add_argument("--wandb-run-name",          type=str,
                        default="pong-enhanced-dqn")
    parser.add_argument("--batch-size",              type=int,   default=32)
    parser.add_argument("--memory-size",             type=int,   default=100000)
    parser.add_argument("--lr",                      type=float, default=0.0001)
    parser.add_argument("--discount-factor",         type=float, default=0.99)
    parser.add_argument("--epsilon-start",           type=float, default=1.0)
    parser.add_argument("--epsilon-min",             type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps",     type=int,   default=1000000)
    parser.add_argument("--target-update-frequency", type=int,   default=1000)
    parser.add_argument("--replay-start-size",       type=int,   default=20000)
    parser.add_argument("--max-episode-steps",       type=int,   default=10000)
    parser.add_argument("--train-per-step",          type=int,   default=1)
    parser.add_argument("--student-id",              type=str,   default="114202103")
    parser.add_argument("--n-step",                  type=int,   default=3)
    parser.add_argument("--per-alpha",               type=float, default=0.6)
    parser.add_argument("--per-beta",                type=float, default=0.4)

    args = parser.parse_args([])  # KEY fix for Jupyter

    wandb.init(project="DLP-Lab5-DQN-Pong",
               name=args.wandb_run_name,
               mode="offline")
    agent = DQNAgent(env_name="ALE/Pong-v5", args=args)
    agent.run(episodes=3500)
    wandb.finish()