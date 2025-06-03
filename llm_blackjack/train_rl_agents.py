import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import os
import json
from tqdm import tqdm
try:
    from tqdm import tqdm
except ImportError:
    # Simple fallback if tqdm is not installed
    def tqdm(iterable, *args, **kwargs):
        return iterable

from rl_interface import MonteCarloAgent, DeepQAgent, DeepQNetwork

class BlackjackEnvironment:
    """Simplified Blackjack environment for training RL agents"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the environment for a new game"""
        # Player starts with two random cards
        self.player_cards = [self._draw_card(), self._draw_card()]
        # Dealer shows one card
        self.dealer_visible_card = self._draw_card()
        self.dealer_hidden_card = self._draw_card()
        self.done = False
        self.reward = 0
        
        return self._get_state()
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action: 0 (STAND) or 1 (HIT)
            
        Returns:
            (next_state, reward, done)
        """
        if action == 1:  # HIT
            self.player_cards.append(self._draw_card())
            player_sum = self._get_player_sum()
            
            if player_sum > 21:  # Bust
                self.done = True
                self.reward = -1
            else:
                self.done = False
                self.reward = 0
        else:  # STAND
            self.done = True
            
            # Dealer's turn
            dealer_cards = [self.dealer_visible_card, self.dealer_hidden_card]
            dealer_sum = self._get_sum(dealer_cards)
            
            # Dealer hits until sum >= 17
            while dealer_sum < 17:
                dealer_cards.append(self._draw_card())
                dealer_sum = self._get_sum(dealer_cards)
            
            player_sum = self._get_player_sum()
            
            if dealer_sum > 21:  # Dealer busts
                self.reward = 1
            elif player_sum > dealer_sum:  # Player wins
                self.reward = 1
            elif player_sum < dealer_sum:  # Dealer wins
                self.reward = -1
            else:  # Tie
                self.reward = 0
        
        return self._get_state(), self.reward, self.done
    
    def _draw_card(self):
        """Draw a random card value (simplified)"""
        # 1-10 for number cards, 10 for face cards, 11 for ace
        card = random.randint(1, 13)
        if card > 10:  # Face card
            return 10
        if card == 1:  # Ace
            return 11
        return card
    
    def _get_sum(self, cards):
        """Calculate sum of cards, handling aces"""
        total = sum(cards)
        # Handle aces
        num_aces = cards.count(11)
        while total > 21 and num_aces > 0:
            total -= 10  # Convert an ace from 11 to 1
            num_aces -= 1
        return total
    
    def _get_player_sum(self):
        """Get the player's current sum"""
        return self._get_sum(self.player_cards)
    
    def _get_state(self):
        """Get the current state representation"""
        player_sum = self._get_player_sum()
        usable_ace = 11 in self.player_cards and player_sum <= 21
        
        return player_sum, self.dealer_visible_card, usable_ace


def train_monte_carlo(episodes=100000):
    """Train a Monte Carlo agent"""
    print("Training Monte Carlo agent...")
    env = BlackjackEnvironment()
    agent = MonteCarloAgent()
    
    # Initialize state-action values and returns
    Q = {}  # (state, action) -> value
    returns = {}  # (state, action) -> list of returns
    
    for episode in tqdm(range(episodes)):
        # Generate an episode
        state = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        
        # Play until game is done
        done = False
        while not done:
            # Get action from epsilon-greedy policy
            if random.random() < agent.epsilon:
                action = random.choice([0, 1])
            else:
                # Use current policy
                if state not in agent.policy:
                    # Default policy: hit if sum < 17, stand otherwise
                    agent.policy[state] = 1 if state[0] < 17 else 0
                action = agent.policy[state]
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Store state, action, reward
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            
            state = next_state
        
        # Update Q-values and policy
        G = 0  # Return
        for t in range(len(episode_states) - 1, -1, -1):
            G = episode_rewards[t] + G  # No discount factor in this simple version
            
            state = episode_states[t]
            action = episode_actions[t]
            
            # Only update if this is the first occurrence of (state, action) in episode
            if (state, action) not in [(episode_states[i], episode_actions[i]) for i in range(t)]:
                if (state, action) not in returns:
                    returns[(state, action)] = []
                
                returns[(state, action)].append(G)
                Q[(state, action)] = sum(returns[(state, action)]) / len(returns[(state, action)])
                
                # Update policy to be greedy with respect to Q
                if state not in agent.policy or Q.get((state, 0), -float('inf')) != Q.get((state, 1), -float('inf')):
                    agent.policy[state] = 0 if Q.get((state, 0), 0) > Q.get((state, 1), 0) else 1
    
    # Save the trained policy
    agent.Q = Q
    agent.returns = returns
    
    # Save model
    model_data = {
        "policy": {str(k): v for k, v in agent.policy.items()},
        "Q": {str(k): v for k, v in Q.items()},
    }
    
    with open("monte_carlo_model.json", "w") as f:
        json.dump(model_data, f)
    
    print("Monte Carlo training complete. Model saved.")
    return agent


def train_deep_q(episodes=10000, batch_size=64):
    """Train a Deep Q-Learning agent"""
    print("Training Deep Q-Learning agent...")
    env = BlackjackEnvironment()
    
    # Initialize Deep Q-Network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepQNetwork().to(device)
    target_model = DeepQNetwork().to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Replay memory
    memory = []
    memory_capacity = 10000
    
    # Training parameters
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    gamma = 0.99  # Discount factor
    
    # Training loop
    for episode in tqdm(range(episodes)):
        state = env.reset()
        done = False
        
        while not done:
            # Convert state to tensor
            state_tensor = torch.tensor([state[0], state[1], int(state[2])], 
                                       dtype=torch.float32).to(device)
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                with torch.no_grad():
                    q_values = model(state_tensor)
                    action = q_values.argmax().item()
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Store transition in replay memory
            memory.append((state, action, reward, next_state, done))
            if len(memory) > memory_capacity:
                memory.pop(0)
            
            # Update state
            state = next_state
            
            # Learn from replay memory
            if len(memory) >= batch_size:
                # Sample batch
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # Convert to tensors
                states_tensor = torch.tensor([[s[0], s[1], int(s[2])] for s in states], 
                                           dtype=torch.float32).to(device)
                actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states_tensor = torch.tensor([[s[0], s[1], int(s[2])] for s in next_states], 
                                                dtype=torch.float32).to(device)
                dones_tensor = torch.tensor(dones, dtype=torch.float32).to(device)
                
                # Compute Q-values
                q_values = model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                
                # Compute target Q-values
                with torch.no_grad():
                    next_q_values = target_model(next_states_tensor).max(1)[0]
                    target_q_values = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)
                
                # Compute loss
                loss = criterion(q_values, target_q_values)
                
                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Update target model periodically
        if episode % 100 == 0:
            target_model.load_state_dict(model.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # Save model
    torch.save(model.state_dict(), "deep_q_model.pth")
    print("Deep Q-Learning training complete. Model saved.")
    
    # Create and return agent with trained model
    agent = DeepQAgent()
    agent.model = model
    return agent


if __name__ == "__main__":
    print("Starting RL agent training...")
    
    # Train Monte Carlo agent
    monte_carlo_agent = train_monte_carlo(episodes=100000)
    
    # Train Deep Q-Learning agent
    deep_q_agent = train_deep_q(episodes=10000)
    
    print("Training complete!")