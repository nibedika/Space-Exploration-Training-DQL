import gym
import pygame
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from padm_env import SpaceExplorerEnv
from DQN_model import DQN, create_target_network
from utils import preprocess, plot_training_and_heatmap

def main():
    # Create an instance of the SpaceExplorer environment
    env = SpaceExplorerEnv()

    # Create a Deep Q-network (DQN) agent
    input_dim = np.prod(env.observation_space.shape)  # Flatten the observation space
    output_dim = env.action_space.n
    dqn_agent = DQN(input_dim, output_dim)
    target_net = create_target_network(dqn_agent)

    # Define hyperparameters
    learning_rate = .01
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995
    target_update = 10
    buffer_capacity = 10000
    batch_size = 64

    # Initialize optimizer and loss function
    optimizer = optim.Adam(dqn_agent.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss()

    # Initialize variables for tracking rewards, steps, and epsilon
    num_episodes = 100
    epsilon = epsilon_start
    all_rewards = []
    state_visits = np.zeros(env.observation_space.shape, dtype=int)

    # Main training loop
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = dqn_agent(preprocess(state))
                    action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Track state visits
            x, y = env.current_position
            state_visits[x, y] += 1

            # Compute target and loss
            with torch.no_grad():
                next_q_values = target_net(preprocess(next_state))
                target_value = reward + gamma * next_q_values.max().item() * (1 - done)

            q_values = dqn_agent(preprocess(state))
            target = torch.tensor([target_value], dtype=torch.float32, device=q_values.device)

            # Ensure action is within bounds
            assert 0 <= action < q_values.size(1), f"Action {action} out of bounds for q_values shape {q_values.shape}"

            loss = criterion(q_values[:, action], target)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update state
            state = next_state

            # Render the environment
            env.render()
            time.sleep(0.000001)

        # Update epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Track rewards
        all_rewards.append(total_reward)

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

        # Optionally update target network
        if episode % target_update == 0:
            target_net.load_state_dict(dqn_agent.state_dict())

    # Save the trained model
    torch.save(dqn_agent.state_dict(), 'dqn.pth')

    # Plot training rewards and heat map
    plot_training_and_heatmap(all_rewards, state_visits, 'training_curve.png')

    # Test the trained agent
    test_agent(env, dqn_agent)

    # Close the environment
    env.close()

def test_agent(env, agent):
    num_episodes = 10
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                q_values = agent(preprocess(state))
                action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

            # Render the environment
            env.render()
            time.sleep(0.1)

        print(f"Test Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
        total_rewards.append(total_reward)

    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

if __name__ == "__main__":
    main()
