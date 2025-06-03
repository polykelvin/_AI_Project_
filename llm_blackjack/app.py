from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import time
from dotenv import load_dotenv
from game_engine import BlackjackGame
from llm_interface import LLMInterface
from rl_interface import RLInterface
from game_stats import GameStats

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize game and AI interfaces
game = BlackjackGame()
llm = LLMInterface(model=os.getenv("LLM_MODEL", "gemma3:latest"))
monte_carlo = RLInterface(algorithm="monte_carlo")
deep_q = RLInterface(algorithm="deep_q")
stats = GameStats()

# Track player type
player_type = "human"  # Options: "human", "llm", "monte_carlo", "deep_q"

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
    global player_type, conversation_history, llm_response_times
    data = request.json
    player_type = data.get('player_type', 'human')
    
    # Reset AI conversation/state
    llm.reset_conversation()
    monte_carlo.reset_conversation()
    deep_q.reset_conversation()
    
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
        player_model = None
        if player_type == 'llm':
            player_model = llm.model
        elif player_type == 'monte_carlo':
            player_model = 'monte_carlo'
        elif player_type == 'deep_q':
            player_model = 'deep_q'
            
        duration = 0  # No AI thinking time for immediate game over
        print(f"Immediate game over. Recording match: player_type={player_type}, model={player_model}, winner={game_state['winner']}")
        stats.record_match(player_type, player_model, game_state['winner'], duration)
    
    # If AI is player and it's player's turn, get AI decision
    if player_type != 'human' and not game_state['game_over']:
        # Let AI make decisions until it stands or busts
        while not game_state['game_over']:
            prompt = game.get_player_llm_prompt()
            
            # Add prompt to conversation history
            conversation_history.append({
                "role": "system",
                "content": prompt,
                "type": "player_prompt"
            })
            
            # Get response from appropriate AI
            if player_type == 'llm':
                ai_response, full_response = llm.get_response(prompt)
                model_name = llm.model
            elif player_type == 'monte_carlo':
                ai_response, full_response = monte_carlo.get_response(prompt)
                model_name = 'Monte Carlo'
            elif player_type == 'deep_q':
                ai_response, full_response = deep_q.get_response(prompt)
                model_name = 'Deep Q-Learning'
            
            # Track response time
            if full_response.get("duration"):
                llm_response_times.append(full_response.get("duration"))
            
            # Extract thinking content from <think> tags if present
            thinking_content = ""
            clean_response = ai_response
            
            if "<think>" in ai_response and "</think>" in ai_response:
                import re
                think_match = re.search(r"<think>([\s\S]*?)</think>", ai_response)
                if think_match:
                    thinking_content = think_match.group(1).strip()
                    clean_response = ai_response.replace(think_match.group(0), "").strip()
            
            # Add thinking process to conversation history if available
            if thinking_content:
                conversation_history.append({
                    "role": "player",
                    "content": thinking_content,
                    "type": "player_thinking",
                    "model": model_name
                })
                
                # Update the response to remove the thinking part
                ai_response = clean_response
            
            # Add response to conversation history with all response data
            conversation_history.append({
                "role": "player",
                "content": ai_response,
                "type": "player_response",
                "model": model_name,
                "duration": full_response.get("duration", 0),
                "status": full_response.get("status", "unknown"),
                "fallback": full_response.get("fallback", False),
                "error": full_response.get("error", None)
            })
            
            action = game.process_player_llm_response(ai_response)
            
            # Add action to conversation history
            conversation_history.append({
                "role": "system",
                "content": f"Player ({model_name}) chose to {action}",
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
                
                # Extract thinking content from <think> tags if present
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
                        "role": "dealer",
                        "content": thinking_content,
                        "type": "dealer_thinking",
                        "model": llm.model
                    })
                    
                    # Update the response to remove the thinking part
                    llm_response = clean_response
                
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
            
            # Record the match result with total response time
            total_response_time = sum(llm_response_times)
            
            # Use the correct player type and model
            player_model = None
            if player_type == 'llm':
                player_model = llm.model
            elif player_type == 'monte_carlo':
                player_model = 'monte_carlo'
            elif player_type == 'deep_q':
                player_model = 'deep_q'
                
            print(f"Game over after AI play. Recording match: player_type={player_type}, model={player_model}, winner={game_state['winner']}, total_response_time={total_response_time}")
            stats.record_match(player_type, player_model, game_state['winner'], total_response_time)
            
            # Update RL agents with game result if they were playing
            if player_type == 'monte_carlo':
                monte_carlo.update_with_result(game_state['winner'] == 'player')
            elif player_type == 'deep_q':
                deep_q.update_with_result(game_state['winner'] == 'player')
    
    # Add conversation history to game state
    game_state['conversation_history'] = conversation_history
    
    return jsonify(game_state)

@app.route('/api/player-hit', methods=['POST'])
def player_hit():
    """Player takes another card"""
    global llm_response_times
    
    # If human player hits, add to conversation
    if player_type == 'human':
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
        player_model = None
        if player_type == 'llm':
            player_model = llm.model
        elif player_type == 'monte_carlo':
            player_model = 'monte_carlo'
        elif player_type == 'deep_q':
            player_model = 'deep_q'
        
        # For human players, use a nominal duration
        duration = 0 if player_type == 'human' else sum(llm_response_times)
        
        print(f"Game over after hit. Recording match: player_type={player_type}, model={player_model}, winner={game_state['winner']}")
        stats.record_match(player_type, player_model, game_state['winner'], duration)
        
        # Update RL agents with game result if they were playing
        if player_type == 'monte_carlo':
            monte_carlo.update_with_result(game_state['winner'] == 'player')
        elif player_type == 'deep_q':
            deep_q.update_with_result(game_state['winner'] == 'player')
    
    # Add conversation history to game state
    game_state['conversation_history'] = conversation_history
    
    return jsonify(game_state)

@app.route('/api/player-stand', methods=['POST'])
def player_stand():
    """Player stands"""
    global llm_response_times
    
    # If human player stands, add to conversation
    if player_type == 'human':
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
            
            # Extract thinking content from <think> tags if present
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
                    "role": "dealer",
                    "content": thinking_content,
                    "type": "dealer_thinking",
                    "model": llm.model
                })
                
                # Update the response to remove the thinking part
                llm_response = clean_response
            
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
        
        # Record the match result with total response time
        player_model = None
        if player_type == 'llm':
            player_model = llm.model
        elif player_type == 'monte_carlo':
            player_model = 'monte_carlo'
        elif player_type == 'deep_q':
            player_model = 'deep_q'
        
        # For human players, use a nominal duration or sum of dealer response times
        duration = sum(llm_response_times)
        
        print(f"Game over after stand. Recording match: player_type={player_type}, model={player_model}, winner={game_state['winner']}, total_response_time={duration}")
        stats.record_match(player_type, player_model, game_state['winner'], duration)
        
        # Update RL agents with game result if they were playing
        if player_type == 'monte_carlo':
            monte_carlo.update_with_result(game_state['winner'] == 'player')
        elif player_type == 'deep_q':
            deep_q.update_with_result(game_state['winner'] == 'player')
    
    # Add conversation history to game state
    game_state['conversation_history'] = conversation_history
    
    return jsonify(game_state)

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available LLM models and RL algorithms"""
    llm_models = llm.get_available_models()
    
    # Add RL algorithms
    all_models = llm_models + ["monte_carlo", "deep_q"]
    
    return jsonify({"models": llm_models, "algorithms": ["monte_carlo", "deep_q"], "all": all_models})

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