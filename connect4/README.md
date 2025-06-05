# LLM Connect 4

A Connect 4 game that allows you to play against LLM-powered AI opponents.

## Features

- Play Connect 4 against different LLM models (Gemma, Qwen, DeepSeek)
- View the AI's thinking process in real-time
- Track game statistics and view a leaderboard
- Human vs. AI or AI vs. AI gameplay

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Make sure you have Ollama installed and running with the required models

## Usage

1. Run the Flask application:
   ```
   python app.py
   ```
2. Open your browser and navigate to `http://localhost:5000`
3. Configure player types and start a new game

## Requirements

- Python 3.8+
- Flask
- Ollama with LLM models (gemma3:latest, qwen3:latest, deepseek-r1:8b)

## Project Structure

- `app.py`: Main Flask application
- `game_engine.py`: Connect 4 game logic
- `llm_interface.py`: Interface for communicating with LLM models
- `game_stats.py`: Game statistics tracking
- `templates/`: HTML templates for the web interface
- `resources/`: Static resources (logos, etc.)

## License

MIT