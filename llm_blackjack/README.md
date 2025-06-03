# LLM Blackjack

A web-based Blackjack game where an LLM plays as the dealer against a human player.

## Features

- Web-based interface using HTML, CSS, and JavaScript
- Python backend with Flask
- LLM-powered dealer that makes decisions based on game state
- Real-time gameplay with turn-based mechanics

## Architecture

1. **Game Engine**: Manages game state, rules, and card deck
2. **LLM Interface**: Converts game state to prompts and processes LLM responses
3. **Web UI**: Displays game and handles human player interaction

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

## How It Works

The LLM acts as the dealer in the game, making decisions based on the current game state. The game state is converted into a prompt that describes the dealer's hand, the player's visible cards, and the possible actions. The LLM then responds with its decision, which is processed by the game engine.