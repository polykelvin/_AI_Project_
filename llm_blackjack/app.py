from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import time
from dotenv import load_dotenv
from game_engine import BlackjackGame
from llm_interface import LLMInterface
from game_stats import GameStats

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize game and LLM interface
game = BlackjackGame()
llm = LLMInterface(model=os.getenv("LLM_MODEL", "gemma3:latest"))
stats = GameStats()

# Track if LLM is playing as player
llm_is_player = False

# Store conversation history
conversation_history = []

# Track LLM response times
llm_response_times = []

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
    global llm_is_player, conversation_history, llm_response_times
    data = request.json
    llm_is_player = data.get('llm_is_player', False)
    
    # Reset LLM conversation history
    llm.reset_conversation()
    
    # Clear conversation history and response times
    conversation_history = []
    llm_response_times = []
    
    # Add game start message
    conversation_history.append({
        "role": "system",
        "content": "Starting a new game of Blackjack",
        "type": "game_action"
    })
    
    game.start_game()
    game_state = game.get_game_state()
    
    # Add initial game state message
    conversation_history.append({
        "role": "system",
        "content": f"Dealer shows: {game.dealer_hand.cards[1]} (hidden card not shown). Player has: {', '.join(str(card) for card in game.player_hand.cards)} (value: {game.player_hand.get_value()})",
        "type": "game_state"
    })
    
    # Check for immediate game over (e.g., player blackjack)
    if game_state['game_over']:
        player_type = 'llm' if llm_is_player else 'human'
        player_model = llm.model if llm_is_player else None
        duration = 0  # No LLM thinking time for immediate game over
        print(f"Immediate game over. Recording match: player_type={player_type}, model={player_model}, winner={game_state['winner']}")
        stats.record_match(player_type, player_model, game_state['winner'], duration)
    
    # If LLM is player and it's player's turn, get LLM decision
    if llm_is_player and not game_state['game_over']:
        # Let LLM make decisions until it stands or busts
        while not game_state['game_over']:
            prompt = game.get_player_llm_prompt()
            
            # Add prompt to conversation history
            conversation_history.append({
                "role": "system",
                "content": prompt,
                "type": "player_prompt"
            })
            
            llm_response, full_response = llm.get_response(prompt)
            
            # Track response time
            if full_response.get("duration"):
                llm_response_times.append(full_response.get("duration"))
            
            # Add thinking process to conversation history if available
            if full_response.get("thinking"):
                conversation_history.append({
                    "role": "player",
                    "content": full_response.get("thinking"),
                    "type": "player_thinking",
                    "model": llm.model
                })
            
            # Add response to conversation history with all response data
            conversation_history.append({
                "role": "player",
                "content": llm_response,
                "type": "player_response",
                "model": llm.model,
                "duration": full_response.get("duration", 0),
                "status": full_response.get("status", "unknown"),
                "fallback": full_response.get("fallback", False),
                "error": full_response.get("error", None)
            })
            
            action = game.process_player_llm_response(llm_response)
            
            # Add action to conversation history
            conversation_history.append({
                "role": "system",
                "content": f"Player ({llm.model}) chose to {action}",
                "type": "player_action"
            })
            
            # If player hit, add the new card info
            if action == "HIT" and len(game.player_hand.cards) > 0:
                new_card = game.player_hand.cards[-1]
                conversation_history.append({
                    "role": "system",
                    "content": f"Player drew {new_card} (new value: {game.player_hand.get_value()})",
                    "type": "game_state"
                })
            
            # Update game state
            game_state = game.get_game_state()
            
            # If player stands or busts, break the loop
            if action == "STAND" or game_state['game_over']:
                break
        
        # If player stood (didn't bust), let dealer play
        if not game_state['game_over'] and action == "STAND":
            # Player stood, now dealer plays
            game.player_stand()
            
            # Add dealer's hidden card reveal to conversation
            conversation_history.append({
                "role": "system",
                "content": f"Dealer reveals hidden card: {game.dealer_hand.cards[0]} (dealer value: {game.dealer_hand.get_value()})",
                "type": "game_state"
            })
            
            # Reset LLM conversation for dealer's turn
            llm.reset_conversation()
            
            # Let dealer make decisions until game is over
            while not game.game_over:
                prompt = game.get_dealer_llm_prompt()
                
                # Add prompt to conversation history
                conversation_history.append({
                    "role": "system",
                    "content": prompt,
                    "type": "dealer_prompt"
                })
                
                llm_response, full_response = llm.get_response(prompt)
                
                # Track response time
                if full_response.get("duration"):
                    llm_response_times.append(full_response.get("duration"))
                
                # Add thinking process to conversation history if available
                if full_response.get("thinking"):
                    conversation_history.append({
                        "role": "dealer",
                        "content": full_response.get("thinking"),
                        "type": "dealer_thinking",
                        "model": llm.model
                    })
                
                # Add response to conversation history with all response data
                conversation_history.append({
                    "role": "dealer",
                    "content": llm_response,
                    "type": "dealer_response",
                    "model": llm.model,
                    "duration": full_response.get("duration", 0),
                    "status": full_response.get("status", "unknown"),
                    "fallback": full_response.get("fallback", False),
                    "error": full_response.get("error", None)
                })
                
                result = game.process_dealer_llm_response(llm_response)
                
                # Add dealer action to conversation
                if "hits" in result:
                    conversation_history.append({
                        "role": "system",
                        "content": f"Dealer ({llm.model}) chose to HIT",
                        "type": "dealer_action"
                    })
                    
                    # Add new card info
                    new_card = game.dealer_hand.cards[-1]
                    conversation_history.append({
                        "role": "system",
                        "content": f"Dealer drew {new_card} (new value: {game.dealer_hand.get_value()})",
                        "type": "game_state"
                    })
                elif "stand" in result.lower():
                    conversation_history.append({
                        "role": "system",
                        "content": f"Dealer ({llm.model}) chose to STAND",
                        "type": "dealer_action"
                    })
                
                # If dealer stands or busts, game is over
                if "stand" in result.lower() or game.game_over:
                    break
            
            # Update game state after dealer's turn
            game_state = game.get_game_state()
        
        # Add game result to conversation if game is over
        if game_state['game_over']:
            conversation_history.append({
                "role": "system",
                "content": f"Game over: {game_state['message']}",
                "type": "game_result"
            })
            
            # Record the match result with total LLM response time
            total_response_time = sum(llm_response_times)
            print(f"Game over after LLM play. Recording match: player_type=llm, model={llm.model}, winner={game_state['winner']}, total_response_time={total_response_time}")
            stats.record_match('llm', llm.model, game_state['winner'], total_response_time)
    
    # Add conversation history to game state
    game_state['conversation_history'] = conversation_history
    
    return jsonify(game_state)

@app.route('/api/player-hit', methods=['POST'])
def player_hit():
    """Player takes another card"""
    global llm_response_times
    
    # If human player hits, add to conversation
    if not llm_is_player:
        conversation_history.append({
            "role": "system",
            "content": "Human player chose to HIT",
            "type": "player_action"
        })
    
    game.player_hit()
    
    # Add new card info to conversation
    new_card = game.player_hand.cards[-1]
    conversation_history.append({
        "role": "system",
        "content": f"Player drew {new_card} (new value: {game.player_hand.get_value()})",
        "type": "game_state"
    })
    
    game_state = game.get_game_state()
    
    # If game is over, record the result
    if game_state['game_over']:
        player_type = 'llm' if llm_is_player else 'human'
        player_model = llm.model if llm_is_player else None
        
        # For human players, use a nominal duration
        duration = 0 if player_type == 'human' else sum(llm_response_times)
        
        print(f"Game over after hit. Recording match: player_type={player_type}, model={player_model}, winner={game_state['winner']}")
        stats.record_match(player_type, player_model, game_state['winner'], duration)
    
    # Add conversation history to game state
    game_state['conversation_history'] = conversation_history
    
    return jsonify(game_state)

@app.route('/api/player-stand', methods=['POST'])
def player_stand():
    """Player stands"""
    global llm_response_times
    
    # If human player stands, add to conversation
    if not llm_is_player:
        conversation_history.append({
            "role": "system",
            "content": "Human player chose to STAND",
            "type": "player_action"
        })
    
    game.player_stand()
    
    # Add dealer's hidden card reveal to conversation
    conversation_history.append({
        "role": "system",
        "content": f"Dealer reveals hidden card: {game.dealer_hand.cards[0]} (dealer value: {game.dealer_hand.get_value()})",
        "type": "game_state"
    })
    
    # If game is not over, let the LLM make decisions for the dealer
    if not game.game_over:
        # Reset LLM conversation for dealer's turn
        llm.reset_conversation()
        
        # Let dealer make decisions until game is over
        while not game.game_over:
            prompt = game.get_dealer_llm_prompt()
            
            # Add prompt to conversation history
            conversation_history.append({
                "role": "system",
                "content": prompt,
                "type": "dealer_prompt"
            })
            
            llm_response, full_response = llm.get_response(prompt)
            
            # Track response time
            if full_response.get("duration"):
                llm_response_times.append(full_response.get("duration"))
            
            # Add thinking process to conversation history if available
            if full_response.get("thinking"):
                conversation_history.append({
                    "role": "dealer",
                    "content": full_response.get("thinking"),
                    "type": "dealer_thinking",
                    "model": llm.model
                })
            
            # Add response to conversation history with all response data
            conversation_history.append({
                "role": "dealer",
                "content": llm_response,
                "type": "dealer_response",
                "model": llm.model,
                "duration": full_response.get("duration", 0),
                "status": full_response.get("status", "unknown"),
                "fallback": full_response.get("fallback", False),
                "error": full_response.get("error", None)
            })
            
            result = game.process_dealer_llm_response(llm_response)
            
            # Add dealer action to conversation
            if "hits" in result:
                conversation_history.append({
                    "role": "system",
                    "content": f"Dealer ({llm.model}) chose to HIT",
                    "type": "dealer_action"
                })
                
                # Add new card info
                new_card = game.dealer_hand.cards[-1]
                conversation_history.append({
                    "role": "system",
                    "content": f"Dealer drew {new_card} (new value: {game.dealer_hand.get_value()})",
                    "type": "game_state"
                })
            elif "stand" in result.lower():
                conversation_history.append({
                    "role": "system",
                    "content": f"Dealer ({llm.model}) chose to STAND",
                    "type": "dealer_action"
                })
            
            # If dealer stands or busts, game is over
            if "stand" in result.lower() or game.game_over:
                break
    
    # Add game result to conversation if game is over
    game_state = game.get_game_state()
    if game_state['game_over']:
        conversation_history.append({
            "role": "system",
            "content": f"Game over: {game_state['message']}",
            "type": "game_result"
        })
        
        # Record the match result with total LLM response time
        player_type = 'llm' if llm_is_player else 'human'
        player_model = llm.model if llm_is_player else None
        
        # For human players, use a nominal duration or sum of dealer response times
        duration = sum(llm_response_times)
        
        print(f"Game over after stand. Recording match: player_type={player_type}, model={player_model}, winner={game_state['winner']}, total_response_time={duration}")
        stats.record_match(player_type, player_model, game_state['winner'], duration)
    
    # Add conversation history to game state
    game_state['conversation_history'] = conversation_history
    
    return jsonify(game_state)

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available LLM models"""
    models = llm.get_available_models()
    return jsonify({"models": models})

@app.route('/api/set-model', methods=['POST'])
def set_model():
    """Set the LLM model to use"""
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
    leaderboard = stats.get_leaderboard()
    recent_matches = stats.get_recent_matches()
    
    print("Leaderboard data:", leaderboard)
    print("Recent matches:", recent_matches)
    
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