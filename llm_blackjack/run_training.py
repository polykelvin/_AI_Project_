#!/usr/bin/env python
"""
Script to run the RL agent training with progress tracking
"""

import os
import sys
import time
from train_rl_agents import train_monte_carlo, train_deep_q

def main():
    print("=" * 50)
    print("LLM Blackjack - RL Agent Training")
    print("=" * 50)
    print("\nThis script will train both Monte Carlo and Deep Q-Learning agents.")
    print("The training process may take some time.")
    
    # Check if models already exist
    monte_carlo_exists = os.path.exists("monte_carlo_model.json")
    deep_q_exists = os.path.exists("deep_q_model.pth")
    
    if monte_carlo_exists or deep_q_exists:
        print("\nExisting model files detected:")
        if monte_carlo_exists:
            print("- monte_carlo_model.json")
        if deep_q_exists:
            print("- deep_q_model.pth")
        
        choice = input("\nDo you want to retrain these models? (y/n): ").strip().lower()
        if choice != 'y':
            print("Training cancelled. Using existing models.")
            return
    
    # Train Monte Carlo agent
    print("\n" + "=" * 30)
    print("Training Monte Carlo Agent")
    print("=" * 30)
    start_time = time.time()
    train_monte_carlo(episodes=50000)  # Reduced for quicker training
    mc_time = time.time() - start_time
    print(f"Monte Carlo training completed in {mc_time:.2f} seconds")
    
    # Train Deep Q-Learning agent
    print("\n" + "=" * 30)
    print("Training Deep Q-Learning Agent")
    print("=" * 30)
    start_time = time.time()
    train_deep_q(episodes=5000)  # Reduced for quicker training
    dq_time = time.time() - start_time
    print(f"Deep Q-Learning training completed in {dq_time:.2f} seconds")
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print("\nThe trained models have been saved and will be used automatically")
    print("when selecting these algorithms as players in the game.")

if __name__ == "__main__":
    main()