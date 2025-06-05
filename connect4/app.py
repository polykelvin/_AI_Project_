from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import time
import threading
from dotenv import load_dotenv
from game_engine import Connect4Game
from llm_interface import LLMInterface
from game_stats import GameStats

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize game and AI interfaces
game = Connect4Game()
llm = LLMInterface(model=os.getenv("LLM_MODEL", "gemma3:latest"))
stats = GameStats()

# Track player types
player1_type = "human"  # Options: "human", "llm"
player2_type = "llm"    # Options: "human", "llm"

# Store conversation history
conversation_history = []

# Track LLM response times
llm_response_times = []

# Lock for thread safety
api_lock = threading.Lock()

@app.route('/')
def index():
    """Render the main game page"""
    return render_template('index.html')

@app.route('/leaderboard')
def leaderboard():
    """Render the leaderboard page"""
    return render_template('leaderboard.html')

@app.route('/resources/<path:path>')
def send_resource(path):
    """Serve static resources"""
    return send_from_directory('resources', path)

@app.route('/api/start-game', methods=['POST'])
def start_game():
    """Start a new game"""
    global player1_type, player2_type, conversation_history, llm_response_times
    
    # Acquire lock to ensure thread safety
    with api_lock:
        data = request.json
        player1_type = data.get('player1_type', 'human')
        player2_type = data.get('player2_type', 'llm')
    
    # Reset AI conversation/state
    llm.reset_conversation()
    
    # Clear conversation history and response times
    conversation_history = []
    llm_response_times = []
    
    # Add game start message
    conversation_history.append({
        "role": "system",
        "content": "Starting a new game of Connect 4",
        "type": "game_action"
    })
    
    game.reset_game()
    game_state = game.get_game_state()
    
    # Add initial game state message
    conversation_history.append({
        "role": "system",
        "content": f"Player {game_state['current_player']}'s turn",
        "type": "game_state"
    })
    
    # If AI is player 1 and it's their turn, get AI decision
    if player1_type == 'llm' and game_state['current_player'] == 1:
        ai_move = get_ai_move()
        game_state = game.get_game_state()
    
    # Add conversation history to game state
    game_state['conversation_history'] = conversation_history
    
    return jsonify(game_state)

@app.route('/api/make-move', methods=['POST'])
def make_move():
    """Make a move in the specified column"""
    global llm_response_times
    
    # Acquire lock to ensure thread safety
    with api_lock:
        data = request.json
        column = data.get('column')
        
        if column is None:
            return jsonify({"error": "No column specified"}), 400
        
        # Get current player
        current_player = game.current_player
        player_type = player1_type if current_player == 1 else player2_type
        
        # Add human move to conversation
        if player_type == 'human':
            conversation_history.append({
                "role": "system",
                "content": f"Player {current_player} chose column {column}",
                "type": "player_action"
            })
        
        # Make the move
        success = game.make_move(column)
        
        if not success:
            return jsonify({"error": "Invalid move"}), 400
        
        # Get updated game state
        game_state = game.get_game_state()
        
        # If game is over, record the result
        if game_state['game_over']:
            # Add game result to conversation
            result_msg = "It's a draw!" if game_state['winner'] == 0 else f"Player {game_state['winner']} wins!"
            conversation_history.append({
                "role": "system",
                "content": f"Game over: {result_msg}",
                "type": "game_result"
            })
            
            # Record match result
            duration = sum(llm_response_times)
            stats.record_match(player1_type, player2_type, game_state['winner'], duration)
        
        # If it's AI's turn now, get AI move
        elif ((game_state['current_player'] == 1 and player1_type == 'llm') or 
              (game_state['current_player'] == 2 and player2_type == 'llm')):
            ai_move = get_ai_move()
            game_state = game.get_game_state()
            
            # If game is over after AI move, record the result
            if game_state['game_over']:
                # Add game result to conversation
                result_msg = "It's a draw!" if game_state['winner'] == 0 else f"Player {game_state['winner']} wins!"
                conversation_history.append({
                    "role": "system",
                    "content": f"Game over: {result_msg}",
                    "type": "game_result"
                })
                
                # Record match result
                duration = sum(llm_response_times)
                stats.record_match(player1_type, player2_type, game_state['winner'], duration)
        
        # Add conversation history to game state
        game_state['conversation_history'] = conversation_history
        
        return jsonify(game_state)

def get_ai_move():
    """Get a move from the AI player"""
    global llm_response_times
    
    # Get current player
    current_player = game.current_player
    
    # Generate prompt for LLM
    prompt = game.get_player_llm_prompt()
    
    # Add prompt to conversation history
    conversation_history.append({
        "role": "system",
        "content": prompt,
        "type": "player_prompt"
    })
    
    # Get response from LLM
    llm_response, full_response = llm.get_response(prompt)
    
    # Track response time
    if full_response.get("duration"):
        llm_response_times.append(full_response.get("duration"))
    
    # Extract thinking content if present
    thinking_content = ""
    clean_response = llm_response
    
    if "<think>" in llm_response and "</think>" in llm_response:
        import re
        think_match = re.search(r"<think>([\s\S]*?)</think>", llm_response)
        if think_match:
            thinking_content = think_match.group(1).strip()
            clean_response = llm_response.replace(think_match.group(0), "").strip()
    
    # Add thinking process to conversation history if available
    if thinking_content:
        conversation_history.append({
            "role": f"player{current_player}",
            "content": thinking_content,
            "type": "player_thinking",
            "model": llm.model
        })
    
    # Add response to conversation history
    conversation_history.append({
        "role": f"player{current_player}",
        "content": clean_response,
        "type": "player_response",
        "model": llm.model,
        "duration": full_response.get("duration", 0),
        "status": full_response.get("status", "unknown"),
        "fallback": full_response.get("fallback", False),
        "error": full_response.get("error", None)
    })
    
    # Process the response
    result = game.process_player_llm_response(llm_response)
    
    if result["success"]:
        # Add action to conversation history
        conversation_history.append({
            "role": "system",
            "content": f"Player {current_player} chose column {result['column']}",
            "type": "player_action"
        })
    else:
        # Add error to conversation history
        conversation_history.append({
            "role": "system",
            "content": f"Error: {result.get('error', 'Invalid move')}",
            "type": "error"
        })
    
    return result

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available LLM models"""
    llm_models = llm.get_available_models()
    
    return jsonify({"models": llm_models})

@app.route('/api/set-model', methods=['POST'])
def set_model():
    """Set the LLM model to use"""
    # Acquire lock to ensure thread safety
    with api_lock:
        data = request.json
        model = data.get('model')
        if model:
            global llm
            llm = LLMInterface(model=model)
            return jsonify({"success": True, "model": model})
        return jsonify({"success": False, "error": "No model specified"})

@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    """Get the leaderboard"""
    # Acquire lock to ensure thread safety when reading stats
    with api_lock:
        leaderboard = stats.get_leaderboard()
        recent_matches = stats.get_recent_matches()
        
        return jsonify({
            "leaderboard": leaderboard,
            "recent_matches": recent_matches
        })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create resources directory if it doesn't exist
    os.makedirs('resources/logo', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True)