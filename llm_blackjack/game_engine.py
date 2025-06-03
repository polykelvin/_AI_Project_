import random
from typing import List, Dict, Tuple, Optional

# Card suits and values
SUITS = ['hearts', 'diamonds', 'clubs', 'spades']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
CARD_VALUES = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 10, 'Q': 10, 'K': 10, 'A': 11  # Ace is 11 by default, can be 1 if needed
}

class Card:
    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank
        self.value = CARD_VALUES[rank]
        self.hidden = False
    
    def __str__(self):
        if self.hidden:
            return "Hidden Card"
        return f"{self.rank} of {self.suit}"
    
    def to_dict(self):
        if self.hidden:
            return {"hidden": True}
        return {
            "suit": self.suit,
            "rank": self.rank,
            "value": self.value,
            "hidden": self.hidden
        }

class Deck:
    def __init__(self):
        self.cards = []
        self.reset()
    
    def reset(self):
        self.cards = [Card(suit, rank) for suit in SUITS for rank in RANKS]
        self.shuffle()
    
    def shuffle(self):
        random.shuffle(self.cards)
    
    def deal(self) -> Card:
        if not self.cards:
            self.reset()
        return self.cards.pop()

class Hand:
    def __init__(self):
        self.cards: List[Card] = []
    
    def add_card(self, card: Card):
        self.cards.append(card)
    
    def get_value(self) -> int:
        value = sum(card.value for card in self.cards if not card.hidden)
        
        # Handle aces (count as 1 instead of 11 if total would exceed 21)
        num_aces = sum(1 for card in self.cards if not card.hidden and card.rank == 'A')
        while value > 21 and num_aces > 0:
            value -= 10  # Convert an ace from 11 to 1
            num_aces -= 1
            
        return value
    
    def is_blackjack(self) -> bool:
        return len(self.cards) == 2 and self.get_value() == 21
    
    def is_busted(self) -> bool:
        return self.get_value() > 21
    
    def to_dict(self):
        return {
            "cards": [card.to_dict() for card in self.cards],
            "value": self.get_value(),
            "is_blackjack": self.is_blackjack(),
            "is_busted": self.is_busted()
        }

class BlackjackGame:
    def __init__(self):
        self.deck = Deck()
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        self.game_over = False
        self.winner = None
        self.message = ""
    
    def start_game(self):
        # Reset everything
        self.deck = Deck()
        self.player_hand = Hand()
        self.dealer_hand = Hand()
        self.game_over = False
        self.winner = None
        self.message = ""
        
        # Deal initial cards
        self.player_hand.add_card(self.deck.deal())
        
        dealer_card = self.deck.deal()
        dealer_card.hidden = True  # Dealer's first card is hidden
        self.dealer_hand.add_card(dealer_card)
        
        self.player_hand.add_card(self.deck.deal())
        self.dealer_hand.add_card(self.deck.deal())
        
        # Check for blackjack
        if self.player_hand.is_blackjack():
            self.dealer_hand.cards[0].hidden = False  # Reveal dealer's hidden card
            if self.dealer_hand.is_blackjack():
                self.game_over = True
                self.winner = "tie"
                self.message = "Both have Blackjack! It's a tie!"
            else:
                self.game_over = True
                self.winner = "player"
                self.message = "Blackjack! Player wins!"
    
    def player_hit(self):
        if self.game_over:
            return
        
        self.player_hand.add_card(self.deck.deal())
        
        # Check if player busts
        if self.player_hand.is_busted():
            self.game_over = True
            self.winner = "dealer"
            self.message = "Player busts! Dealer wins!"
            self.dealer_hand.cards[0].hidden = False  # Reveal dealer's hidden card
        
        # Automatically stand if player hits to exactly 21
        elif self.player_hand.get_value() == 21:
            self.player_stand()  # This will handle the dealer's turn and determine the winner
    
    def player_stand(self):
        if self.game_over:
            return
        
        # Reveal dealer's hidden card
        self.dealer_hand.cards[0].hidden = False
        
        # Dealer must hit until they have at least 17
        while self.dealer_hand.get_value() < 17:
            self.dealer_hand.add_card(self.deck.deal())
        
        # Determine winner
        dealer_value = self.dealer_hand.get_value()
        player_value = self.player_hand.get_value()
        
        if self.dealer_hand.is_busted():
            self.winner = "player"
            self.message = "Dealer busts! Player wins!"
        elif dealer_value > player_value:
            self.winner = "dealer"
            self.message = "Dealer wins!"
        elif player_value > dealer_value:
            self.winner = "player"
            self.message = "Player wins!"
        else:
            self.winner = "tie"
            self.message = "It's a tie!"
        
        self.game_over = True
    
    def get_game_state(self) -> Dict:
        return {
            "player_hand": self.player_hand.to_dict(),
            "dealer_hand": self.dealer_hand.to_dict(),
            "game_over": self.game_over,
            "winner": self.winner,
            "message": self.message
        }
    
    def get_dealer_llm_prompt(self) -> str:
        """Generate a prompt for the LLM acting as dealer based on the current game state."""
        visible_dealer_cards = [card for card in self.dealer_hand.cards if not card.hidden]
        visible_dealer_value = sum(card.value for card in visible_dealer_cards)
        
        prompt = f"""
You are the dealer in a game of Blackjack. Here's the current game state:

Your visible cards: {', '.join(str(card) for card in visible_dealer_cards)}
Your visible hand value: {visible_dealer_value}

Player's cards: {', '.join(str(card) for card in self.player_hand.cards)}
Player's hand value: {self.player_hand.get_value()}

What would you like to do? Respond with only one of the following options:
- HIT (take another card)
- STAND (end your turn)

Remember, as the dealer, you must hit until your hand value is at least 17.
"""
        return prompt
    
    def get_player_llm_prompt(self) -> str:
        """Generate a prompt for the LLM acting as player based on the current game state."""
        visible_dealer_cards = [card for card in self.dealer_hand.cards if not card.hidden]
        visible_dealer_value = sum(card.value for card in visible_dealer_cards)
        
        prompt = f"""
You are playing Blackjack against a dealer. Here's the current game state:

Your cards: {', '.join(str(card) for card in self.player_hand.cards)}
Your hand value: {self.player_hand.get_value()}

Dealer's visible cards: {', '.join(str(card) for card in visible_dealer_cards)}
Dealer's visible hand value: {visible_dealer_value}

What would you like to do? Respond with only one of the following options:
- HIT (take another card)
- STAND (end your turn)

Remember:
- If your hand value exceeds 21, you bust and lose
- The dealer must hit until they have at least 17
- Your goal is to have a higher hand value than the dealer without busting
"""
        return prompt
    
    def process_dealer_llm_response(self, response: str) -> str:
        """Process the LLM's response when acting as dealer and take the appropriate action."""
        # First, remove any thinking section if present
        if "</think>" in response:
            # Extract only the part after the thinking section
            response = response.split("</think>", 1)[1].strip()
        
        # Clean and normalize the response
        response = response.strip().upper()
        
        # Check for exact matches first (most reliable)
        if response == "HIT":
            # Dealer takes another card
            self.dealer_hand.add_card(self.deck.deal())
            
            if self.dealer_hand.is_busted():
                self.game_over = True
                self.winner = "player"
                self.message = "Dealer busts! Player wins!"
                return "Dealer hits and busts! Player wins!"
            
            return f"Dealer hits and now has a visible hand value of {self.dealer_hand.get_value()}"
        elif response == "STAND":
            # Process stand action
            return self._process_dealer_stand()
        
        # If no exact match, check the last occurrence of HIT or STAND
        words = response.split()
        if words:
            last_word = words[-1]
            if "HIT" in last_word:
                # Dealer takes another card
                self.dealer_hand.add_card(self.deck.deal())
                
                if self.dealer_hand.is_busted():
                    self.game_over = True
                    self.winner = "player"
                    self.message = "Dealer busts! Player wins!"
                    return "Dealer hits and busts! Player wins!"
                
                return f"Dealer hits and now has a visible hand value of {self.dealer_hand.get_value()}"
            elif "STAND" in last_word:
                # Process stand action
                return self._process_dealer_stand()
        
        # As a fallback, check if either word appears in the response
        # Prioritize STAND over HIT if both appear (safer option)
        if "STAND" in response:
            return self._process_dealer_stand()
        elif "HIT" in response:
            # Dealer takes another card
            self.dealer_hand.add_card(self.deck.deal())
            
            if self.dealer_hand.is_busted():
                self.game_over = True
                self.winner = "player"
                self.message = "Dealer busts! Player wins!"
                return "Dealer hits and busts! Player wins!"
            
            return f"Dealer hits and now has a visible hand value of {self.dealer_hand.get_value()}"
        
        return "Invalid response. Please respond with HIT or STAND."
    
    def _process_dealer_stand(self):
        """Helper method to process dealer stand action"""
        # Determine winner
        dealer_value = self.dealer_hand.get_value()
        player_value = self.player_hand.get_value()
        
        if dealer_value < 17:
            return "As the dealer, you must hit until your hand value is at least 17."
        
        if dealer_value > player_value:
            self.winner = "dealer"
            self.message = "Dealer wins!"
            result = "Dealer wins!"
        elif player_value > dealer_value:
            self.winner = "player"
            self.message = "Player wins!"
            result = "Player wins!"
        else:
            self.winner = "tie"
            self.message = "It's a tie!"
            result = "It's a tie!"
        
        self.game_over = True
        return result
    
    def process_player_llm_response(self, response: str) -> str:
        """Process the LLM's response when acting as player and take the appropriate action."""
        # First, remove any thinking section if present
        if "</think>" in response:
            # Extract only the part after the thinking section
            response = response.split("</think>", 1)[1].strip()
        
        # Clean and normalize the response
        response = response.strip().upper()
        
        # Check for exact matches first (most reliable)
        if response == "HIT":
            self.player_hit()
            return "HIT"
        elif response == "STAND":
            self.player_stand()
            return "STAND"
        
        # If no exact match, check the last occurrence of HIT or STAND
        # This handles cases where the model explains its decision
        words = response.split()
        if words:
            last_word = words[-1]
            if "HIT" in last_word:
                self.player_hit()
                return "HIT"
            elif "STAND" in last_word:
                self.player_stand()
                return "STAND"
        
        # As a fallback, check if either word appears in the response
        # Prioritize STAND over HIT if both appear (safer option)
        if "STAND" in response:
            self.player_stand()
            return "STAND"
        elif "HIT" in response:
            self.player_hit()
            return "HIT"
        
        return "Invalid response. Please respond with HIT or STAND."