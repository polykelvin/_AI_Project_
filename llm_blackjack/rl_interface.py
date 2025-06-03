import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import time
from typing import Dict, Any, Tuple, List, Optional

class RLInterface:
    """Interface for Reinforcement Learning algorithms to play Blackjack"""
    
    def __init__(self, algorithm="monte_carlo"):
        """
        Initialize the RL interface
        
        Args:
            algorithm: Algorithm to use ('monte_carlo' or 'deep_q')
        """
        self.algorithm = algorithm
        self.supported_algorithms = ["monte_carlo", "deep_q"]
        
        if algorithm not in self.supported_algorithms:
            print(f"Warning: Algorithm {algorithm} not supported. Using monte_carlo instead.")
            self.algorithm = "monte_carlo"
        
        # Initialize the appropriate algorithm
        if self.algorithm == "monte_carlo":
            self.agent = MonteCarloAgent()
        else:  # deep_q
            self.agent = DeepQAgent()
    
    def get_response(self, state_description: str) -> Tuple[str, Dict]:
        """
        Get a decision from the RL agent based on the current game state
        
        Args:
            state_description: Text description of the game state
            
        Returns:
            Tuple of (response_text, full_response_data)
        """
        start_time = time.time()
        print(f"Requesting decision from {self.algorithm} agent")
        
        # Parse the state description to extract game state
        player_cards, dealer_visible_card = self._parse_state(state_description)
        
        # Get action from the agent
        action = self.agent.get_action(player_cards, dealer_visible_card)
        
        # Convert action to response
        response = "HIT" if action == 1 else "STAND"
        
        end_time = time.time()
        duration = end_time - start_time
        
        return response, {
            "response": response,
            "thinking": f"Algorithm: {self.algorithm}\nPlayer cards: {player_cards}\nDealer visible: {dealer_visible_card}\nAction: {response}",
            "duration": duration,
            "model": self.algorithm,
            "status": "success"
        }
    
    def _parse_state(self, state_description: str) -> Tuple[List[int], int]:
        """
        Parse the state description to extract player cards and dealer visible card
        
        Args:
            state_description: Text description of the game state
            
        Returns:
            Tuple of (player_cards, dealer_visible_card)
        """
        # This is a simplified parser - in production, you'd want more robust parsing
        player_cards = []
        dealer_visible_card = 0
        
        # Extract player cards
        if "Your cards:" in state_description:
            player_part = state_description.split("Your cards:")[1].split("Your hand value:")[0]
            # Extract card values (simplistic approach)
            for card in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]:
                if card in player_part:
                    if card == "A":
                        player_cards.append(11)  # Ace is 11 by default
                    elif card in ["J", "Q", "K"]:
                        player_cards.append(10)
                    else:
                        player_cards.append(int(card))
        
        # Extract dealer visible card
        if "Dealer's visible cards:" in state_description:
            dealer_part = state_description.split("Dealer's visible cards:")[1].split("\n")[0]
            # Extract card value (simplistic approach)
            for card in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]:
                if card in dealer_part:
                    if card == "A":
                        dealer_visible_card = 11  # Ace is 11 by default
                    elif card in ["J", "Q", "K"]:
                        dealer_visible_card = 10
                    else:
                        dealer_visible_card = int(card)
                    break
        
        return player_cards, dealer_visible_card
    
    def reset_conversation(self):
        """Reset any conversation history (for compatibility with LLMInterface)"""
        pass
    
    def update_with_result(self, won: bool):
        """
        Update the agent with the game result for learning
        
        Args:
            won: Whether the player won the game
        """
        self.agent.update_with_result(won)


class MonteCarloAgent:
    """Monte Carlo agent for Blackjack"""
    
    def __init__(self):
        # State: (player_sum, dealer_card, usable_ace)
        # Action: 0 (STAND) or 1 (HIT)
        self.Q = {}  # Q-values
        self.returns = {}  # Returns for each state-action pair
        self.policy = {}  # Current policy
        self.epsilon = 0.1  # Exploration rate
        
        # Load pre-trained model if available
        self.load_model()
    
    def get_action(self, player_cards: List[int], dealer_visible_card: int) -> int:
        """
        Get action (0=STAND, 1=HIT) based on current state
        
        Args:
            player_cards: List of player card values
            dealer_visible_card: Dealer's visible card value
            
        Returns:
            Action (0=STAND, 1=HIT)
        """
        # Calculate player sum and check for usable ace
        player_sum = sum(player_cards)
        usable_ace = 11 in player_cards and player_sum > 21
        
        # If we have a usable ace, convert one ace from 11 to 1
        if usable_ace:
            player_sum -= 10
        
        # Get state
        state = (player_sum, dealer_visible_card, usable_ace)
        
        # If player sum >= 21, always STAND
        if player_sum >= 21:
            return 0  # STAND
        
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        
        # If state not in policy, initialize it
        if state not in self.policy:
            # Default policy: hit if sum < 17, stand otherwise
            self.policy[state] = 1 if player_sum < 17 else 0
        
        return self.policy[state]
    
    def update_with_result(self, won: bool):
        """
        Update the agent with the game result
        
        Args:
            won: Whether the player won the game
        """
        # In a real implementation, we would update Q-values and policy here
        # This is simplified for demonstration purposes
        pass
    
    def load_model(self):
        """Load pre-trained model if available"""
        try:
            # In a real implementation, load model from file
            pass
        except:
            print("No pre-trained Monte Carlo model found. Using default policy.")


class DeepQNetwork(nn.Module):
    """Deep Q-Network for Blackjack"""
    
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 128)  # Input: player_sum, dealer_card, usable_ace
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)  # Output: Q-values for STAND and HIT
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DeepQAgent:
    """Deep Q-Learning agent for Blackjack"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepQNetwork().to(self.device)
        self.epsilon = 0.1  # Exploration rate
        
        # Load pre-trained model if available
        self.load_model()
    
    def get_action(self, player_cards: List[int], dealer_visible_card: int) -> int:
        """
        Get action (0=STAND, 1=HIT) based on current state
        
        Args:
            player_cards: List of player card values
            dealer_visible_card: Dealer's visible card value
            
        Returns:
            Action (0=STAND, 1=HIT)
        """
        # Calculate player sum and check for usable ace
        player_sum = sum(player_cards)
        usable_ace = 11 in player_cards and player_sum > 21
        
        # If we have a usable ace, convert one ace from 11 to 1
        if usable_ace:
            player_sum -= 10
        
        # If player sum >= 21, always STAND
        if player_sum >= 21:
            return 0  # STAND
        
        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        
        # Convert state to tensor
        state = torch.tensor([player_sum, dealer_visible_card, int(usable_ace)], 
                            dtype=torch.float32).to(self.device)
        
        # Get Q-values and select action with highest Q-value
        with torch.no_grad():
            q_values = self.model(state)
            return q_values.argmax().item()
    
    def update_with_result(self, won: bool):
        """
        Update the agent with the game result
        
        Args:
            won: Whether the player won the game
        """
        # In a real implementation, we would update the model here
        # This is simplified for demonstration purposes
        pass
    
    def load_model(self):
        """Load pre-trained model if available"""
        try:
            self.model.load_state_dict(torch.load("deep_q_model.pth"))
            self.model.eval()
            print("Loaded pre-trained Deep Q-Network model.")
        except:
            print("No pre-trained Deep Q-Network model found. Using random policy.")