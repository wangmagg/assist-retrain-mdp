from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
import torch.nn.functional as F

from src.environment import HumanAICollabEnv
from src.simulate_baselines import (
    generate_data_stream,
    setup_env,
    run_episode,
    baseline_policies,
    baseline_policy_labels,
    baseline_policy_colors,
    baseline_policy_linestyles,
)

Transition = namedtuple("Transition", ["state", "action", "next_state", "reward"])

class ReplayMemory(object):
    """
    Replay memory for experience replay in DQN
    """

    def __init__(self, capacity, seed=42):
        self.memory = deque([], maxlen=capacity)
        self.rng = np.random.default_rng(seed)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        which_idx = self.rng.choice(len(self.memory), batch_size)
        return [self.memory[i] for i in which_idx]
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """
    Deep Q-Network for DQN
    """
    def __init__(self, n_obs, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_obs, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, n_actions)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def decay_eps(steps, eps, eps_end, eps_decay):
    # Decay epsilon for epsilon-greedy exploration policy
    return eps_end + (eps - eps_end) * np.exp(-1. * steps / eps_decay)

def select_action(state, policy_net, n_actions, eps, rng, device):
    # Select action based on epsilon-greedy policy
    if rng.uniform() > eps:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[rng.choice(n_actions)]], device=device, dtype=torch.long)

def optimize(memory, policy_net, target_net, optimizer, batch_size, gamma):
    # Update policy network using minibatch from experience replay buffer
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    with torch.no_grad():
        next_state_values = target_net(next_state_batch).max(1).values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize model
    optimizer.zero_grad()
    loss.backward()

    # clip gradient
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss

def train_dqn(
    env: HumanAICollabEnv,
    n_time_steps: int,
    n_episodes=100,
    lr=0.0001,
    gamma_lr=0.1,
    memory_size=10000,
    batch_size=128,
    eps=0.9,
    eps_end=0.05,
    eps_decay=100,
    gamma=0.9,
    tau=0.005,
    device="cpu",
    seed=42,
    fig_dir="figs",
    **data_kwargs
):
    # Train DQN model using multiple episodes of data collection
    # in the human-AI collaboration environment

    policy_net = DQN(env.n_obs, env.n_actions).to(device)
    target_net = DQN(env.n_obs, env.n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[100, 200, 300], gamma=gamma_lr)
    memory = ReplayMemory(memory_size, seed=seed)
    n_actions = len(env.action_space())
    rng = np.random.default_rng(seed)

    avg_rewards_per_episode = []
    avg_qvals_per_episode = []
    avg_losses_per_episode = []
    steps = 0
    for _ in tqdm(range(n_episodes)):
        env.reset()

        # generate a data stream for the covariate and the ground truth decision
        dat_stream = generate_data_stream(rng=rng, n_time_steps=n_time_steps, **data_kwargs)
        env.set_dat_stream(dat_stream)

        rewards = []
        qvals = []
        losses = []
        actions = []
        while env.stream_idx < env.len_stream - 1:
            # select an action based on the current state and update the environment
            state = env.get_state()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = select_action(state, policy_net, n_actions, eps, rng, device)
            eps = decay_eps(steps, eps, eps_end, eps_decay)
            reward = env.step(action)
            rewards.append(reward)
            qvals.append(policy_net(state).max(1).values.item())
            actions.append(action.item())

            next_state = env.get_state()
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            reward = torch.tensor([reward], dtype=torch.float32, device=device)
            
            # store the transition in the experience replay buffer
            memory.push(state, action, next_state, reward)

            # update the policy and target networks
            loss = optimize(memory, policy_net, target_net, optimizer, batch_size, gamma)
            if loss is not None:
                losses.append(loss.item())

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = tau * policy_net_state_dict[key] + (1 - tau) * target_net_state_dict[key]
            target_net.load_state_dict(target_net_state_dict)

            steps += 1

        # update learning rate
        scheduler.step()
        avg_rewards_per_episode.append(np.mean(rewards))
        avg_qvals_per_episode.append(np.mean(qvals))

        with torch.no_grad():
            avg_losses_per_episode.append(np.mean(losses))

        # plot the training progress
        fig, ax = plt.subplots(1, 4, figsize=(24, 6))
        ax[0].plot(avg_rewards_per_episode)
        ax[1].plot(avg_qvals_per_episode)
        ax[2].plot(avg_losses_per_episode)
        ax[3].hist(actions, bins=n_actions)
        fig.savefig(fig_dir / "dqn_training.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    
    return policy_net

def run_episode_dqn(
    env: HumanAICollabEnv,
    dat_stream,
    policy_net: DQN,
    device: str,
    eps:float=0.05,
    seed:int=42
):
    # Run a single episode in the human-AI collaboration environment
    # Used for evaluating the DQN model
    env.reset()
    env.set_dat_stream(dat_stream)
    rewards = []
    actions = []
    rng = np.random.default_rng(seed)
    while env.stream_idx < env.len_stream:
        state = env.get_state()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = select_action(state, policy_net, env.n_actions, eps, rng, device)
        reward = env.step(action)
        rewards.append(reward)
        actions.append(action.item())

    return rewards, actions

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--human-decision-config-path", type=str, default="configs/human/human.json")
    parser.add_argument("--human-update-config-path", type=str, default="configs/update/update_human.json")
    parser.add_argument("--ai-decision-config-path", type=str, default="configs/ai/ai_ambig.json")
    parser.add_argument("--ai-update-config-path", type=str, default="configs/update/update_ai_linear.json")
    parser.add_argument("--switch-config-path", type=str, default="configs/switch/switch_ambig.json")
    parser.add_argument("--switch-update-config-path", type=str, default="configs/update/update_switch_linear.json")
    parser.add_argument("--reward-config-path", type=str, default="configs/reward/reward_med_retrain_cost.json")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--model-fname", type=str, default="dqn.pt")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--fig-dir", type=str, default="figs")

    parser.add_argument("--n-time-steps", type=int, default=5000)
    parser.add_argument("--data-a", type=float, default=5)
    parser.add_argument("--data-b", type=float, default=2)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--n-inference-episodes", type=int, default=100)

    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--memory-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eps", type=float, default=0.9)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # setup environment
    env = setup_env(args)

    # make figure directory
    ai = Path(args.ai_decision_config_path).stem
    switch = Path(args.switch_config_path).stem
    reward = Path(args.reward_config_path).stem
    ai_update = Path(args.ai_update_config_path).stem
    switch_update = Path(args.switch_update_config_path).stem
    fig_dir = Path(args.fig_dir) / reward / f"{ai}_{switch}" / f"{ai_update}_{switch_update}"
    fig_dir.mkdir(exist_ok=True, parents=True)

    print(ai, switch, ai_update, switch_update)

    # load existing DQN model
    model_dir = Path(args.model_dir) / reward / f"{ai}_{switch}" / f"{ai_update}_{switch_update}"    
    if not args.train and (model_dir / args.model_fname).exists():
        policy_net = DQN(env.n_obs, env.n_actions).to(args.device)
        policy_net.load_state_dict(torch.load(model_dir / args.model_fname))

    # train DQN model
    else:
        policy_net = train_dqn(
            env, 
            n_episodes=args.n_episodes,
            lr=args.lr,
            memory_size=args.memory_size,
            batch_size=args.batch_size,
            eps=args.eps,
            eps_end=args.eps_end,
            eps_decay=args.eps_decay,
            gamma=args.gamma,
            tau=args.tau,
            seed=args.seed,
            a = args.data_a,
            b = args.data_b,
            n_time_steps = args.n_time_steps,
            fig_dir=fig_dir)
        model_dir.mkdir(exist_ok=True, parents=True)
        torch.save(policy_net.state_dict(), model_dir / args.model_fname)

    # compare with baselines
    rewards_dqn_reps = []
    actions_dqn_reps = []
    rewards_baselines = {policy: [] for policy in baseline_policies}
    for i in tqdm(range(args.n_inference_episodes)):
        rng = np.random.default_rng(args.seed + args.n_episodes + i)
        dat_stream = generate_data_stream(rng=rng, n_time_steps=args.n_time_steps, a=args.data_a, b=args.data_b)

        rewards_dqn, actions_dqn = run_episode_dqn(
            env=env,
            dat_stream=dat_stream,
            policy_net=policy_net,
            device=args.device
        )
        rewards_dqn_reps.append(rewards_dqn)
        actions_dqn_reps.append(actions_dqn)

        for policy in baseline_policies:
            rewards = run_episode(
                dat_stream=dat_stream,
                env=env,
                policy=policy,
                seed=args.seed
            )
            rewards_baselines[policy].append(rewards)

    # Plot average cumulative rewards and actions
    fig, ax = plt.subplots(2, 1, figsize=(9, 9), height_ratios=[3, 1], sharex=True)
    start_plt_t = 20
    rewards_dqn_cum_mean = np.cumsum(rewards_dqn_reps, axis=1) / np.arange(1, args.n_time_steps + 1).T
    rewards_dqn_mean_cum_mean = np.mean(rewards_dqn_cum_mean, axis=0)
    rewards_dqn_sem_cum_mean = np.std(rewards_dqn_cum_mean, axis=0) / np.sqrt(args.n_inference_episodes)
    ax[0].plot(rewards_dqn_mean_cum_mean[start_plt_t:], label="DQN", color="tab:purple")
    ax[0].fill_between(
        np.arange(start_plt_t, args.n_time_steps), 
        rewards_dqn_mean_cum_mean[start_plt_t:] - 1.96 * rewards_dqn_sem_cum_mean[start_plt_t:],
        rewards_dqn_mean_cum_mean[start_plt_t:] + 1.96 * rewards_dqn_sem_cum_mean[start_plt_t:],
        alpha=0.1,
        color="tab:purple")
    for policy, label, color, linestyle in zip(
        baseline_policies,
        baseline_policy_labels,
        baseline_policy_colors,
        baseline_policy_linestyles,
    ):
        rewards_cum_mean = (
            np.cumsum(rewards_baselines[policy], axis=1)
            / np.arange(1, args.n_time_steps + 1).T
        )
        rewards_mean_cum_mean = np.mean(rewards_cum_mean, axis=0)
        rewards_sem_cum_mean = np.std(rewards_cum_mean, axis=0) / np.sqrt(args.n_inference_episodes)
        ax[0].plot(rewards_mean_cum_mean[start_plt_t:], label=label, color=color, linestyle=linestyle)
        ax[0].fill_between(
            np.arange(start_plt_t, args.n_time_steps), 
            rewards_mean_cum_mean[start_plt_t:] - 1.96 * rewards_sem_cum_mean[start_plt_t:],
            rewards_mean_cum_mean[start_plt_t:] + 1.96 * rewards_sem_cum_mean[start_plt_t:],
            alpha=0.1,
            color=color)

    ax[0].legend()
    ax[0].set_xlabel("Time Step", fontsize=16)
    ax[0].set_ylabel("Avg Cumulative Reward", fontsize=16)
    ax[0].tick_params(axis='both', which='major', labelsize=12)

    action_bincounts = []
    actions_dqn_reps = np.array(actions_dqn_reps)
    for i in range(args.n_time_steps):
        cnt = np.bincount(actions_dqn_reps[:, i], minlength=env.n_actions)
        action_bincounts.append(cnt)
    action_bincounts = np.array(action_bincounts)
    action_bincounts_cum = action_bincounts.cumsum(axis=0) / args.n_inference_episodes
    ax[1].stackplot(
        np.arange(args.n_time_steps),
        action_bincounts_cum.T,
        labels=["Withhold", "Show", "Retrain"],
        colors=["gainsboro", "darkgrey", "black"]
    )
    ax[1].legend()
    ax[1].set_xlabel("Time Step", fontsize=16)
    ax[1].set_ylabel("Cumulative Actions", fontsize=16)
    ax[1].tick_params(axis='both', which='major', labelsize=12)

    fig.savefig(fig_dir / "dqn_vs_baselines.png", dpi=300, bbox_inches="tight")
