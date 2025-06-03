# LLM Blackjack

A web-based Blackjack game where AI plays as the dealer or player, featuring both LLM-based and reinforcement learning approaches.

## Features

- Web-based interface using HTML, CSS, and JavaScript
- Python backend with Flask
- Multiple AI options:
  - LLM-powered player using Ollama API (Gemma, Qwen, DeepSeek)
  - Rule-based dealer following standard casino rules
  - Monte Carlo reinforcement learning agent
  - Deep Q-Learning reinforcement learning agent
- Real-time gameplay with turn-based mechanics
- Collapsible LLM reasoning panels to see AI thought processes
- Comprehensive leaderboard to compare different AI approaches
- Automatic handling of edge cases like hitting to 21

## Architecture

1. **Game Engine**: Manages game state, rules, and card deck
2. **LLM Interface**: Converts game state to prompts and processes LLM responses
3. **RL Interface**: Implements Monte Carlo and Deep Q-Learning algorithms
4. **Web UI**: Displays game and handles human player interaction

## Environment Setup

### Local LLM: Setting up Ollama (Windows)

1. Download and install Ollama from [ollama.ai](https://ollama.ai)

2. Set environment variables (via System Properties > Environment Variables):
   - `OLLAMA_HOST`: `0.0.0.0:11434`
   - `OLLAMA_MODELS`: `C:\Users\<your_username>\_AI_Project_\ollama\models`
   - Add to `PATH`: `C:\Users\<your_username>\AppData\Local\Programs\Ollama`

3. Start Ollama in PowerShell or Command Prompt:
   ```
   ollama serve
   ```

4. Pull the required models:
   ```
   ollama pull gemma3:latest
   ollama pull qwen3:latest
   ollama pull deepseek-r1:8b
   ```

### Cloud LLM Integration (Work in Progress)

We plan to integrate cloud-based LLM APIs in future updates:
- OpenAI API (GPT-4, GPT-3.5)
- Anthropic Claude API
- Google Gemini API
- Custom API endpoint support

This will allow comparison between local and cloud-based LLMs for Blackjack strategy.

## Application Setup

1. Install Python dependencies:
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
python run_training.py
```

This will create model files that will be loaded automatically when selecting these algorithms as players.

## How It Works

The game supports multiple AI approaches:

1. **LLM-based AI**: Uses large language models to make decisions based on the current game state
2. **Monte Carlo**: Uses reinforcement learning with Monte Carlo methods to learn optimal policies
3. **Deep Q-Learning**: Uses neural networks to approximate Q-values for state-action pairs

You can select which AI plays as the player, while the dealer follows standard Blackjack rules:
- Dealer must hit until they have at least 17
- Dealer must stand on 17 or higher

## Models and Training

### LLM Models

We currently support three local LLM models through Ollama:

1. **Gemma 3** (gemma3:latest)
   - Google's lightweight and efficient model
   - Good balance of performance and speed
   - Size: 7B parameters

2. **Qwen 3** (qwen3:latest)
   - Alibaba's advanced model with strong reasoning capabilities
   - Excellent at explaining its decision-making process
   - Size: 7B parameters

3. **DeepSeek-R1** (deepseek-r1:8b)
   - Specialized for reasoning tasks
   - Strong mathematical and logical reasoning
   - Size: 8B parameters

### Reinforcement Learning Models

1. **Monte Carlo Agent**
   - Training: 50,000 episodes
   - Learning rate: 0.1
   - Discount factor (gamma): 0.9
   - Exploration strategy: Epsilon-greedy with decay
   - State representation: (player_sum, dealer_visible_card, usable_ace)
   - Model file: monte_carlo_model.json

2. **Deep Q-Learning Agent**
   - Training: 5,000 episodes
   - Neural network: 3-layer MLP (64, 32 hidden units)
   - Learning rate: 0.001
   - Discount factor (gamma): 0.95
   - Replay buffer size: 10,000
   - Batch size: 64
   - Target network update frequency: 10 episodes
   - State representation: (player_sum, dealer_visible_card, usable_ace)
   - Model file: deep_q_model.pth

## Comparing AI Approaches

The leaderboard tracks performance across different AI types, allowing you to compare:
- How well LLMs perform at Blackjack
- How reinforcement learning compares to LLMs
- Which specific models perform best

## Latest Features

- **Collapsible Reasoning**: Toggle visibility of LLM reasoning with "Show reasoning" buttons
- **Rule-based Dealer**: Dealer now follows standard casino rules rather than using an LLM
- **Reinforcement Learning**: Added Monte Carlo and Deep Q-Learning algorithms as player options
- **Automatic Stand on 21**: Players automatically stand when hitting to exactly 21
- **Improved Response Parsing**: Better handling of LLM responses with thinking sections
- **Dynamic UI**: LLM model selection is disabled when using RL algorithms