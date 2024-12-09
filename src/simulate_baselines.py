from matplotlib import pyplot as plt
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

from src.actor import DecisionPDF, SwitchPDF, Actor
from src.environment import Reward, HumanAICollabEnv

baseline_policies = [
    "human_only",
    "ai_only",
    "always_show",
    "random_no_retrain",
    "random_with_retrain",
    "show_if_ai_better",
]
baseline_policy_labels = [
    "Human Only",
    "AI Only",
    "HAI Naive (Always Show)",
    "HAI Naive (Random Show)",
    "HAI Naive (Random Show + Retrain)", 
    "HAI 1-Step Optimal"
]
baseline_policy_colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:green",
    "tab:green",
    "tab:red",
]
baseline_policy_linestyles = [
    "solid",
    "solid",
    "solid",
    "dashed",
    "dotted",
    "solid",
]

def generate_data_stream(n_time_steps, a, b, rng):
    # Generate a data stream of length n_time_steps
    x = rng.beta(a, b, n_time_steps)
    y = rng.binomial(1, x)
    return x, y

def setup_env(args):
    # Set up the human-AI decision-making environment 
    with open(args.reward_config_path, "r") as f:
        reward_config = json.load(f)
    reward = Reward(**reward_config)

    with open(args.human_decision_config_path, "r") as f:
        human_decision_config = json.load(f)
    if args.human_update_config_path is not None:
        with open(args.human_update_config_path, "r") as f:
            human_update_config = json.load(f)
    else:
        human_update_config = None
    human_pdf = DecisionPDF(**human_decision_config)
    human = Actor(human_pdf)

    with open(args.ai_decision_config_path, "r") as f:
        ai_decision_config = json.load(f)
    if args.ai_update_config_path is not None:
        with open(args.ai_update_config_path, "r") as f:
            ai_update_config = json.load(f)
    else:
        ai_update_config = None
    ai_pdf = DecisionPDF(**ai_decision_config)
    ai = Actor(ai_pdf)

    with open(args.switch_config_path, "r") as f:
        switch_config = json.load(f)
    with open(args.switch_update_config_path, "r") as f:
        switch_update_config = json.load(f)
    switch_pdf = SwitchPDF(**switch_config)

    env = HumanAICollabEnv(
        human=human,
        ai=ai,
        switch_pdf=switch_pdf,
        reward_mdl=reward,
        seed=args.seed,
        human_update_config=human_update_config,
        ai_update_config=ai_update_config,
        switch_update_config=switch_update_config
    )
    return env

def run_episode(
    env: HumanAICollabEnv,
    dat_stream,
    policy='random',
    seed=42
):
    # Run a single episode in the human-AI collaboration environment
    rng = np.random.default_rng(seed)
    env.reset()
    env.set_dat_stream(dat_stream)
    rewards = []
    while env.stream_idx < env.len_stream:
        # Determine the action to take based on the policy

        # ai_only: AI makes decision
        if policy == 'ai_only':
            action = None
            ai_only = True
        else:
            ai_only = False

            # human_only: Human makes unassisted decision 
            if policy == 'human_only':
                action = 0
            # random_with_retrain: Randomly decide whether to show the AI's decision and retrain
            elif policy == "random_with_retrain":
                action = rng.choice(env.action_space())
            # random_no_retrain: Randomly decide whether to show the AI's decision
            elif policy == 'random_no_retrain':
                action = rng.choice([0, 1])
            # always_show: Always show the AI's decision
            elif policy == 'always_show':
                action = 1
            # show_if_ai_better: Show the AI's decision if it more likely to be correct
            elif policy == 'show_if_ai_better':
                ai_pdf = env.ai.decision_pdf(env.x_stream[env.stream_idx])
                human_pdf = env.human.decision_pdf(env.x_stream[env.stream_idx])
                action = ai_pdf > human_pdf
        
        # Take the action and observe the reward
        reward = env.step(action, ai_only=ai_only)
        rewards.append(reward)

    return rewards

def run_experiment(
    env,
    a, 
    b,
    policy='random',
    n_time_steps=1000,
    n_reps=500,
    seed=42
):
    # Run multiple episodes in the human-AI collaboration environment
    rewards_reps = []
    for i in tqdm(range(n_reps)):
        rng = np.random.default_rng(seed + i)
        dat_stream = generate_data_stream(n_time_steps=n_time_steps, a=a, b=b, rng=rng)
        rewards = run_episode(
            dat_stream=dat_stream,
            env=env,
            policy=policy,
            seed=seed + i
        )
        rewards_reps.append(rewards)
    return rewards_reps

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--fig-dir", type=str, default="figs")

    parser.add_argument("--human-decision-config-path", type=str, default="configs/human/human.json")
    parser.add_argument("--human-update-config-path", type=str, default="configs/update/update_human.json")
    parser.add_argument("--ai-decision-config-path", type=str, default="configs/ai/ai_ambig.json")
    parser.add_argument("--ai-update-config-path", type=str, default="configs/update/update_ai_linear.json")
    parser.add_argument("--switch-config-path", type=str, default="configs/switch/switch_ambig.json")
    parser.add_argument("--switch-update-config-path", type=str, default="configs/update/update_switch_linear.json")
    parser.add_argument("--reward-config-path", type=str, default="configs/reward/reward.json")

    parser.add_argument("--data-a", type=float, default=5)
    parser.add_argument("--data-b", type=float, default=2)

    parser.add_argument("--tn-reward", type=float, default=1)
    parser.add_argument("--tp-reward", type=float, default=1)
    parser.add_argument("--type1-error-cost", type=float, default=-1)
    parser.add_argument("--type2-error-cost", type=float, default=-1)

    parser.add_argument("--n-time-steps", type=int, default=1000)
    parser.add_argument("--n-reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    env = setup_env(args)

    ai = Path(args.ai_decision_config_path).stem
    switch = Path(args.switch_config_path).stem
    ai_update = Path(args.ai_update_config_path).stem
    switch_update = Path(args.switch_update_config_path).stem
    fig_dir = Path(args.fig_dir) / f"{ai}_{switch}" / f"{ai_update}_{switch_update}"
    print(ai, switch, ai_update, switch_update)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    start_plt_t = 20

    # Run experiment for each baseline policy
    # and plot the average cumulative rewards
    for policy, lbl, color, style in zip(
        baseline_policies,
        baseline_policy_labels,
        baseline_policy_colors,
        baseline_policy_linestyles,
    ):
        print(policy)
        rewards = run_experiment(
            env,
            n_reps=args.n_reps,
            n_time_steps=args.n_time_steps,
            policy=policy,
            seed=args.seed,
            a=args.data_a,
            b=args.data_b
        )

        # plot average cumulative rewards
        cum_rewards = np.cumsum(rewards, axis=1) / np.arange(1, args.n_time_steps + 1)
        mean_cum_rewards = cum_rewards.mean(axis=0)
        se_cum_rewards = cum_rewards.std(axis=0) / np.sqrt(cum_rewards.shape[0])
        ax.plot(mean_cum_rewards, label=lbl, color=color, linestyle=style)
        ax.fill_between(
            np.arange(start_plt_t, args.n_time_steps),
            mean_cum_rewards[start_plt_t:] - 1.96 * se_cum_rewards[start_plt_t:],
            mean_cum_rewards[start_plt_t:] + 1.96 * se_cum_rewards[start_plt_t:],
            alpha=0.1,
            color=color
        )
        ax.legend()
        ax.set_xlabel("Time Step", fontsize=16)
        ax.set_ylabel("Avg Cumulative Reward", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Save figure
        fig_dir.mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_dir / "baselines.png", dpi=300, bbox_inches="tight")
