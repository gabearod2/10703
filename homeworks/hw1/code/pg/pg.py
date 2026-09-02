#! python3

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np  # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PolicyGradient(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        lr_actor=1e-3,
        lr_critic=1e-3,
        mode="REINFORCE",
        n=0,
        gamma=0.99,
        device="cpu",
    ):
        super(PolicyGradient, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.mode = mode
        self.n = n
        self.gamma = gamma

        self.device = device

        hidden_layer_size = 256

        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, action_size),
            # BEGIN STUDENT SOLUTION
            # Output will be linear logits.
            # This is labeled as "score = h_\theta(s, a)" in lecture03.
            # END STUDENT SOLUTION
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            # Output will be scalar Value estimate
            nn.Linear(hidden_layer_size, 1),
            # END STUDENT SOLUTION
        )

        # initialize networks, optimizers, move networks to device
        # BEGIN STUDENT SOLUTION
        self.actor.to(self.device)
        self.actor.to(self.device)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr_actor)
        self.optim_critic = optim.Adam(self.actor.parameters(), lr_critic)
        pass
        # END STUDENT SOLUTION

    def forward(self, state):
        return (self.actor(state), self.critic(state))

    def get_action(self, state, stochastic):
        # if stochastic, sample using the action probabilities, else get the argmax
        # BEGIN STUDENT SOLUTION
        logits = self.actor(state)
        if stochastic:
            return torch.distributions.Categorical(logits=logits).sample()
        else:
            return torch.argmax(logits)
        # END STUDENT SOLUTION

    def calculate_n_step_bootstrap(self, rewards_tensor, values):
        # calculate n step bootstrap
        # BEGIN STUDENT SOLUTION
        pass
        # END STUDENT SOLUTION

    def train(self, states=None, actions=None, rewards=None):
        if actions is None and rewards is None and (
            states is None or isinstance(states, bool)
        ):
            return super().train(True if states is None else states)

        # train the agent using states, actions, and rewards
        # BEGIN STUDENT SOLUTION

        # STEP 1: retrieve the sampled N trajectories
        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.array(actions), dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        num_steps = rewards.shape[0]

        if self.mode == "REINFORCE":
            # STEP 2: Compute every reward-to-go 
            rewards_to_go = torch.zeros(num_steps, dtype=torch.float32, device=self.device)
            reward_to_go = 0.0
            for step in reversed(range(num_steps)):
                reward_to_go = rewards[step] + self.gamma * reward_to_go
                rewards_to_go[step] = reward_to_go

            # STEP 3: estimate the policy gradient
            logits = self.actor(states) 
            log_probs = torch.distributions.Categorical(logits=logits).log_prob(actions)
            actor_loss = -(log_probs * rewards_to_go).mean() # mean over N trajectories

            # update actor
            self.optim_actor.zero_grad()
            actor_loss.backward() # computes gradients
            self.optim_actor.step() # updates policy
        elif self.mode == "REINFORCE_WITH_BASELINE":
            # STEP 2: Compute every reward-to-go 
            rewards_to_go = torch.zeros(num_steps, dtype=torch.float32, device=self.device)
            reward_to_go = 0.0
            for step in reversed(range(num_steps)):
                reward_to_go = rewards[step] + self.gamma * reward_to_go
                rewards_to_go[step] = reward_to_go

            # STEP 3: estimate the policy and critic gradients
            logits = self.actor(states) 
            baselines = self.critic(states).squeeze(-1)
            log_probs = torch.distributions.Categorical(logits=logits).log_prob(actions)
            actor_loss = -(log_probs * (rewards_to_go - baselines.detach())).mean() # mean over N trajectories
            critic_loss = ((rewards_to_go - baselines)**2).mean()

            # update actor/policy
            self.optim_actor.zero_grad()
            actor_loss.backward() # computes gradients
            self.optim_actor.step() # updates policy

            # update critic/baseline
            self.optim_critic.zero_grad()
            critic_loss.backward()
            self.optim_critic.step()
        pass
        # END STUDENT SOLUTION

    def run(self, env, max_steps, num_episodes, train):
        total_rewards = []

        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        for _ in range(num_episodes):
            state, _ = env.reset()
            states = []
            actions = []
            rewards = []
            episode_rewards = 0.0

            for _ in range(max_steps):
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                action = self.get_action(state_tensor, stochastic=train)
                action_item = action.item()

                # step the environment using sampled action and record results
                next_state, reward, terminated, truncated, _ = env.step(action_item)

                states.append(state)
                actions.append(action_item)
                rewards.append(reward)
                episode_rewards += reward

                state = next_state

                if terminated or truncated:
                    break

            # perform update at the end of each episode
            if train:
                self.train(states, actions, rewards)
            total_rewards.append(episode_rewards) # NOTE: undiscounted, as desired
        # END STUDENT SOLUTION
        return total_rewards


def graph_agents(
    graph_name,
    agents,
    env,
    max_steps,
    num_episodes,
    num_test_episodes,
    graph_every,
):
    print(f"Starting: {graph_name}")

    if agents[0].n != 0:
        graph_name += "_" + str(agents[0].n)

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    num_checkpoints = num_episodes // graph_every
    all_trials = np.zeros((len(agents), num_checkpoints))
    for trial_idx, agent in enumerate(agents):
        for checkpoint in range(num_checkpoints):
            # train for graph_every number of episodes
            agent.train() # train mode
            agent.run(env, max_steps, graph_every, train=True)

            # freeze the current policy and evaluate
            agent.train(False)  # eval mode
            test_rewards = agent.run(env, max_steps, num_test_episodes, train=False)
            all_trials[trial_idx, checkpoint] = np.mean(test_rewards)

            print(
                f"algo: {graph_name} ", 
                f"trial num: {trial_idx+1}/{len(agents)} ",
                f"episode num: {(checkpoint+1)*graph_every}/{num_episodes} ", 
                f"mean test reward: {all_trials[trial_idx, checkpoint]:.2f}"
            )

    # average along trial number axis for plotting
    average_total_rewards = all_trials.mean(axis=0) 
    min_total_rewards = all_trials.min(axis=0)
    max_total_rewards = all_trials.max(axis=0)
    # END STUDENT SOLUTION

    # plot the total rewards
    xs = [(i + 1) * graph_every for i in range(len(average_total_rewards))]
    fig, ax = plt.subplots()
    plt.fill_between(xs, min_total_rewards, max_total_rewards, alpha=0.1)
    ax.plot(xs, average_total_rewards)
    ax.set_ylim(-max_steps * 0.01, max_steps * 1.1)
    ax.set_title(graph_name, fontsize=10)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Total Reward")
    fig.savefig(graph_dir / f"{graph_name}.png")
    plt.close(fig)
    print(f"Finished: {graph_name}")


def parse_args():
    mode_choices = ["REINFORCE", "REINFORCE_WITH_BASELINE", "A2C"]

    parser = argparse.ArgumentParser(description="Train an agent.")
    parser.add_argument(
        "--mode",
        type=str,
        default="REINFORCE",
        choices=mode_choices,
        help="Mode to run the agent in",
    )
    parser.add_argument("--n", type=int, default=0, help="The n to use for n step A2C")
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of runs to average over for graph",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=3500, help="Number of episodes to train for"
    )
    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=20,
        help="Number of episodes to test for every eval step",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Maximum number of steps in the environment",
    )
    parser.add_argument(
        "--env_name", type=str, default="CartPole-v1", help="Environment name"
    )
    parser.add_argument(
        "--graph_every", type=int, default=100, help="Graph every x episodes"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # init args, agents, and call graph_agents on the initialized agents
    # BEGIN STUDENT SOLUTION
    global graph_dir
    graph_dir = Path("graphs")

    env = gym.make(args.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    device = torch.device("cpu")

    agents = [
        PolicyGradient(
            state_size,
            action_size,
            mode=args.mode,
            n=args.n,
            device=device,
        )
        for _ in range(args.num_runs)
    ]

    graph_agents(
        args.mode,
        agents,
        env,
        args.max_steps,
        args.num_episodes,
        args.num_test_episodes,
        args.graph_every,
    )
    # END STUDENT SOLUTION


if "__main__" == __name__:
    main()
