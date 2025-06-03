# LLM Blackjack

A web-based Blackjack game where AI plays as the dealer or player.

## Features

- Web-based interface using HTML, CSS, and JavaScript
- Python backend with Flask
- Multiple AI options:
  - LLM-powered dealer and player using Ollama API
  - Monte Carlo reinforcement learning agent
  - Deep Q-Learning reinforcement learning agent
- Real-time gameplay with turn-based mechanics
- Leaderboard to compare different AI approaches

## Architecture

1. **Game Engine**: Manages game state, rules, and card deck
2. **LLM Interface**: Converts game state to prompts and processes LLM responses
3. **RL Interface**: Implements Monte Carlo and Deep Q-Learning algorithms
4. **Web UI**: Displays game and handles human player interaction

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python app.py
   ```

3. Open your browser and navigate to `http://localhost:5000`

## Training RL Agents

To train the Monte Carlo and Deep Q-Learning agents:

```
python train_rl_agents.py
```

This will create model files that will be loaded automatically when selecting these algorithms as players.

## How It Works

The game supports multiple AI approaches:

1. **LLM-based AI**: Uses large language models to make decisions based on the current game state
2. **Monte Carlo**: Uses reinforcement learning with Monte Carlo methods to learn optimal policies
3. **Deep Q-Learning**: Uses neural networks to approximate Q-values for state-action pairs

You can select which AI plays as the player, while the dealer is always controlled by an LLM.

## Comparing AI Approaches

The leaderboard tracks performance across different AI types, allowing you to compare:
- How well LLMs perform at Blackjack
- How reinforcement learning compares to LLMs
- Which specific models perform best