import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any

class GameStats:
    """Track game statistics and match history"""
    
    def __init__(self, stats_file="game_stats.json"):
        """
        Initialize the game statistics tracker
        
        Args:
            stats_file: Path to the JSON file for storing stats
        """
        self.stats_file = stats_file
        self.stats = self._load_stats()
    
    def _load_stats(self) -> Dict:
        """Load statistics from file or create new if not exists"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except:
                return self._create_default_stats()
        else:
            return self._create_default_stats()
    
    def _create_default_stats(self) -> Dict:
        """Create default statistics structure"""
        return {
            "matches": [],
            "players": {
                "human": {"wins": 0, "losses": 0, "ties": 0},
                "gemma3:latest": {"wins": 0, "losses": 0, "ties": 0},
                "qwen3:latest": {"wins": 0, "losses": 0, "ties": 0},
                "deepseek-r1:8b": {"wins": 0, "losses": 0, "ties": 0}
            }
        }
    
    def _save_stats(self):
        """Save statistics to file"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            print(f"Stats saved successfully to {self.stats_file}")
        except Exception as e:
            print(f"Error saving stats: {e}")
    
    def record_match(self, player_type: str, player_model: str, winner: str, duration: float):
        """
        Record a match result
        
        Args:
            player_type: 'human' or 'llm'
            player_model: Model name if player_type is 'llm', otherwise None
            winner: 'player', 'dealer', or 'tie'
            duration: Match duration in seconds
        """
        # Create player identifier
        player = player_model if player_type == 'llm' else 'human'
        
        print(f"Recording match with player={player}, winner={winner}, duration={duration}")
        
        # Create match record
        match = {
            "timestamp": datetime.now().isoformat(),
            "player": player,
            "winner": winner,
            "duration": duration
        }
        
        # Add to matches list
        self.stats["matches"].append(match)
        
        # Update player stats
        if player not in self.stats["players"]:
            print(f"Adding new player to stats: {player}")
            self.stats["players"][player] = {"wins": 0, "losses": 0, "ties": 0}
        
        if winner == 'player':
            self.stats["players"][player]["wins"] += 1
        elif winner == 'dealer':
            self.stats["players"][player]["losses"] += 1
        else:  # tie
            self.stats["players"][player]["ties"] += 1
        
        print(f"Updated stats for {player}: {self.stats['players'][player]}")
        
        # Save updated stats
        self._save_stats()
    
    def get_leaderboard(self) -> List[Dict]:
        """Get leaderboard sorted by win rate"""
        leaderboard = []
        
        for player, stats in self.stats["players"].items():
            total_games = stats["wins"] + stats["losses"] + stats["ties"]
            if total_games > 0:
                win_rate = stats["wins"] / total_games
            else:
                win_rate = 0
                
            leaderboard.append({
                "player": player,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "ties": stats["ties"],
                "total_games": total_games,
                "win_rate": win_rate
            })
        
        # Sort by win rate (descending)
        return sorted(leaderboard, key=lambda x: x["win_rate"], reverse=True)
    
    def get_recent_matches(self, limit=10) -> List[Dict]:
        """Get recent matches"""
        # Return most recent matches first
        return sorted(self.stats["matches"], key=lambda x: x["timestamp"], reverse=True)[:limit]