import matplotlib.pyplot as plt
import numpy as np
import torch

def preprocess(state):
    """
    Preprocesses the state before feeding it into the neural network.
    Here, we convert the state to a PyTorch tensor.
    
    Args:
    - state (numpy array or list): The state to be preprocessed.
    
    Returns:
    - processed_state (PyTorch tensor): The preprocessed state as a PyTorch tensor.
    """
    processed_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    return processed_state

def plot_training_and_heatmap(rewards, state_visits, save_path):
    plt.figure(figsize=(12, 6))

    # Plot training rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    # Plot heat map of state visits
    plt.subplot(1, 2, 2)
    plt.imshow(state_visits, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Visits')
    plt.title('State Visits Heat Map')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
