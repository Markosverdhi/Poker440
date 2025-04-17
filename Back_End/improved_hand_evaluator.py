import functools
from collections import Counter
from enum import IntEnum
import time

class HandRank(IntEnum):
    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9

class Card:
    RANKS = '23456789TJQKA'
    SUITS = 'CDHS'
    
    def __init__(self, card_str):
        if isinstance(card_str, int):
            self.card_int = card_str
            self._init_from_int()
        else:
            self._init_from_str(card_str)
    
    def _init_from_str(self, card_str):
        if len(card_str) < 2:
            raise ValueError(f"Invalid card string: {card_str}")
        
        rank = card_str[:-1].upper()
        suit = card_str[-1].upper()
        
        if rank == '10':
            rank = 'T'
        
        if rank not in self.RANKS or suit not in self.SUITS:
            raise ValueError(f"Invalid card: {card_str}")
        
        self.rank_idx = self.RANKS.index(rank)
        self.suit_idx = self.SUITS.index(suit)
        self.card_int = self.rank_idx + 13 * self.suit_idx
    
    def _init_from_int(self):
        self.rank_idx = self.card_int % 13
        self.suit_idx = self.card_int // 13
    
    @property
    def rank(self):
        return self.RANKS[self.rank_idx]
    
    @property
    def suit(self):
        return self.SUITS[self.suit_idx]
    
    @property
    def rank_value(self):
        return self.rank_idx + 2  # 2-14 (Ace is 14)
    
    def __str__(self):
        return f"{self.rank}{self.suit}"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        if isinstance(other, Card):
            return self.card_int == other.card_int
        return False
    
    def __hash__(self):
        return hash(self.card_int)

class HandEvaluator:
    # Class-level cache for evaluated hands
    _hand_cache = {}
    _cache_hits = 0
    _cache_misses = 0
    _cache_size = 10000  # Maximum cache size
    
    def __init__(self):
        # Pre-compute straights for faster lookup
        self.straights = []
        for i in range(10):
            # Handle Ace-low straight (A-5)
            if i == 0:
                self.straights.append(set([12, 0, 1, 2, 3]))  # A,2,3,4,5
            else:
                self.straights.append(set(range(i, i+5)))
    
    @staticmethod
    def _get_cache_key(cards):
        """Generate a unique key for caching hand evaluations."""
        return tuple(sorted(card.card_int for card in cards))
    
    @classmethod
    def clear_cache_stats(cls):
        """Reset cache hit/miss statistics."""
        cls._cache_hits = 0
        cls._cache_misses = 0
    
    @classmethod
    def get_cache_stats(cls):
        """Return cache statistics."""
        return {
            'hits': cls._cache_hits,
            'misses': cls._cache_misses,
            'size': len(cls._hand_cache),
            'hit_ratio': cls._cache_hits / (cls._cache_hits + cls._cache_misses) if (cls._cache_hits + cls._cache_misses) > 0 else 0
        }
    
    @classmethod
    def _manage_cache_size(cls):
        """Ensure cache doesn't exceed maximum size."""
        if len(cls._hand_cache) > cls._cache_size:
            # Remove oldest 20% of entries
            remove_count = int(cls._cache_size * 0.2)
            keys_to_remove = list(cls._hand_cache.keys())[:remove_count]
            for key in keys_to_remove:
                del cls._hand_cache[key]
    
    def evaluate_hand(self, hole_cards, community_cards=None):
        """
        Evaluate a poker hand and return its rank and description.
        
        Args:
            hole_cards: List of card strings (e.g., ['AS', 'KH'])
            community_cards: Optional list of community card strings
            
        Returns:
            Tuple of (score, description)
            Score is a tuple of (hand_rank, tiebreakers...)
            Description is a string describing the hand
        """
        # Convert string cards to Card objects if needed
        hole_cards_obj = [Card(card) if isinstance(card, str) else card for card in hole_cards]
        
        if community_cards:
            community_cards_obj = [Card(card) if isinstance(card, str) else card for card in community_cards]
            all_cards = hole_cards_obj + community_cards_obj
        else:
            all_cards = hole_cards_obj
        
        # Check cache first
        cache_key = self._get_cache_key(all_cards)
        if cache_key in self._hand_cache:
            self.__class__._cache_hits += 1
            return self._hand_cache[cache_key]
        
        self.__class__._cache_misses += 1
        
        # Find best 5-card hand if more than 5 cards
        if len(all_cards) > 5:
            best_score = None
            best_description = None
            
            # Evaluate all 5-card combinations
            for i in range(len(all_cards)):
                for j in range(i+1, len(all_cards)):
                    remaining_cards = all_cards[:i] + all_cards[i+1:j] + all_cards[j+1:]
                    score, description = self._evaluate_five_card_hand(remaining_cards)
                    
                    if best_score is None or score > best_score:
                        best_score = score
                        best_description = description
            
            result = (best_score, best_description)
        else:
            result = self._evaluate_five_card_hand(all_cards)
        
        # Cache the result
        self._hand_cache[cache_key] = result
        self._manage_cache_size()
        
        return result
    
    def _evaluate_five_card_hand(self, cards):
        """Evaluate a 5-card poker hand."""
        if len(cards) != 5:
            raise ValueError(f"Expected 5 cards, got {len(cards)}")
        
        # Extract ranks and suits
        ranks = [card.rank_idx for card in cards]
        suits = [card.suit_idx for card in cards]
        rank_values = [card.rank_value for card in cards]
        
        # Count occurrences of each rank and suit
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        
        # Check for flush
        is_flush = max(suit_counts.values()) == 5
        
        # Check for straight
        rank_set = set(ranks)
        is_straight = False
        straight_high = None
        
        for straight in self.straights:
            if straight.issubset(rank_set):
                is_straight = True
                # For A-5 straight, high card is 5
                if straight == self.straights[0]:
                    straight_high = 5
                else:
                    straight_high = max(straight) + 2
                break
        
        # Determine hand type and score
        if is_straight and is_flush:
            if max(rank_values) == 14 and straight_high == 14:  # Royal flush
                return ((HandRank.ROYAL_FLUSH,), "Royal Flush")
            else:  # Straight flush
                return ((HandRank.STRAIGHT_FLUSH, straight_high), f"Straight Flush, {self._rank_to_name(straight_high)} high")
        
        # Four of a kind
        if 4 in rank_counts.values():
            quads_rank = next(r for r, count in rank_counts.items() if count == 4)
            kicker = next(r for r in ranks if r != quads_rank)
            return ((HandRank.FOUR_OF_A_KIND, quads_rank + 2, kicker + 2), 
                    f"Four of a Kind, {self._rank_to_name(quads_rank + 2)}s")
        
        # Full house
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            trips_rank = next(r for r, count in rank_counts.items() if count == 3)
            pair_rank = next(r for r, count in rank_counts.items() if count == 2)
            return ((HandRank.FULL_HOUSE, trips_rank + 2, pair_rank + 2), 
                    f"Full House, {self._rank_to_name(trips_rank + 2)}s full of {self._rank_to_name(pair_rank + 2)}s")
        
        # Flush
        if is_flush:
            flush_ranks = sorted(rank_values, reverse=True)
            return ((HandRank.FLUSH, *flush_ranks), 
                    f"Flush, {self._rank_to_name(flush_ranks[0])} high")
        
        # Straight
        if is_straight:
            return ((HandRank.STRAIGHT, straight_high), 
                    f"Straight, {self._rank_to_name(straight_high)} high")
        
        # Three of a kind
        if 3 in rank_counts.values():
            trips_rank = next(r for r, count in rank_counts.items() if count == 3)
            kickers = sorted([r + 2 for r in ranks if r != trips_rank], reverse=True)
            return ((HandRank.THREE_OF_A_KIND, trips_rank + 2, *kickers), 
                    f"Three of a Kind, {self._rank_to_name(trips_rank + 2)}s")
        
        # Two pair
        if list(rank_counts.values()).count(2) == 2:
            pairs = sorted([r for r, count in rank_counts.items() if count == 2], reverse=True)
            kicker = next(r for r in ranks if rank_counts[r] == 1)
            return ((HandRank.TWO_PAIR, pairs[0] + 2, pairs[1] + 2, kicker + 2), 
                    f"Two Pair, {self._rank_to_name(pairs[0] + 2)}s and {self._rank_to_name(pairs[1] + 2)}s")
        
        # One pair
        if 2 in rank_counts.values():
            pair_rank = next(r for r, count in rank_counts.items() if count == 2)
            kickers = sorted([r + 2 for r in ranks if r != pair_rank], reverse=True)
            return ((HandRank.ONE_PAIR, pair_rank + 2, *kickers), 
                    f"Pair of {self._rank_to_name(pair_rank + 2)}s")
        
        # High card
        high_cards = sorted(rank_values, reverse=True)
        return ((HandRank.HIGH_CARD, *high_cards), 
                f"High Card, {self._rank_to_name(high_cards[0])}")
    
    def _rank_to_name(self, rank_value):
        """Convert numerical rank to name."""
        if rank_value == 14:
            return "Ace"
        elif rank_value == 13:
            return "King"
        elif rank_value == 12:
            return "Queen"
        elif rank_value == 11:
            return "Jack"
        elif rank_value == 10:
            return "Ten"
        else:
            return str(rank_value)
    
    def compare_hands(self, hand1, hand2):
        """
        Compare two hands and return the winner.
        
        Args:
            hand1: Tuple of (hole_cards, community_cards)
            hand2: Tuple of (hole_cards, community_cards)
            
        Returns:
            1 if hand1 wins, -1 if hand2 wins, 0 if tie
        """
        score1, _ = self.evaluate_hand(*hand1)
        score2, _ = self.evaluate_hand(*hand2)
        
        if score1 > score2:
            return 1
        elif score1 < score2:
            return -1
        else:
            return 0
    
    @staticmethod
    def get_best_hand(hole_cards, community_cards):
        """
        Find the best 5-card hand from hole cards and community cards.
        
        Args:
            hole_cards: List of card strings (e.g., ['AS', 'KH'])
            community_cards: List of community card strings
            
        Returns:
            List of 5 Card objects representing the best hand
        """
        # Convert string cards to Card objects if needed
        hole_cards_obj = [Card(card) if isinstance(card, str) else card for card in hole_cards]
        community_cards_obj = [Card(card) if isinstance(card, str) else card for card in community_cards]
        
        all_cards = hole_cards_obj + community_cards_obj
        evaluator = HandEvaluator()
        
        best_score = None
        best_hand = None
        
        # Check all 5-card combinations
        from itertools import combinations
        for hand in combinations(all_cards, 5):
            score, _ = evaluator._evaluate_five_card_hand(hand)
            
            if best_score is None or score > best_score:
                best_score = score
                best_hand = hand
        
        return list(best_hand)

# Compatibility function for the existing code
def evaluate_hand(hole_cards, community_cards):
    """
    Evaluate a poker hand and return its score and description.
    This function maintains compatibility with the existing code.
    
    Args:
        hole_cards: List of card strings (e.g., ['AS', 'KH'])
        community_cards: List of community card strings
        
    Returns:
        Tuple of (score, description)
    """
    evaluator = HandEvaluator()
    score_tuple, description = evaluator.evaluate_hand(hole_cards, community_cards)
    
    # Convert the tuple score to a single integer for backward compatibility
    # Base score on hand type (multiply by 1000 to ensure higher hands always win)
    base_score = score_tuple[0] * 1000
    
    # Add tiebreaker values
    for i, value in enumerate(score_tuple[1:]):
        base_score += value / (100 ** (i + 1))
    
    return base_score, description

# Performance testing function
def test_performance(num_iterations=10000):
    """Test the performance of the hand evaluator."""
    import random
    
    # Create a deck of cards
    deck = [f"{r}{s}" for r in "23456789TJQKA" for s in "CDHS"]
    evaluator = HandEvaluator()
    
    start_time = time.time()
    
    for _ in range(num_iterations):
        # Shuffle the deck
        random.shuffle(deck)
        
        # Deal hole cards and community cards
        hole_cards = deck[:2]
        community_cards = deck[2:7]
        
        # Evaluate the hand
        evaluator.evaluate_hand(hole_cards, community_cards)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Evaluated {num_iterations} hands in {elapsed:.2f} seconds")
    print(f"Average time per hand: {(elapsed / num_iterations) * 1000:.2f} ms")
    
    # Print cache statistics
    stats = HandEvaluator.get_cache_stats()
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")
    print(f"Cache size: {stats['size']}")
    print(f"Hit ratio: {stats['hit_ratio']:.2%}")

if __name__ == "__main__":
    # Example usage
    evaluator = HandEvaluator()
    
    # Royal flush
    hand1 = ["AS", "KS", "QS", "JS", "TS"]
    score1, desc1 = evaluator.evaluate_hand(hand1)
    print(f"{hand1}: {desc1} (Score: {score1})")
    
    # Straight flush
    hand2 = ["9H", "8H", "7H", "6H", "5H"]
    score2, desc2 = evaluator.evaluate_hand(hand2)
    print(f"{hand2}: {desc2} (Score: {score2})")
    
    # Four of a kind
    hand3 = ["AC", "AS", "AH", "AD", "KS"]
    score3, desc3 = evaluator.evaluate_hand(hand3)
    print(f"{hand3}: {desc3} (Score: {score3})")
    
    # Compare hands
    print(f"Hand 1 vs Hand 2: {evaluator.compare_hands((hand1, []), (hand2, []))}")
    print(f"Hand 2 vs Hand 3: {evaluator.compare_hands((hand2, []), (hand3, []))}")
    
    # Test with hole cards and community cards
    hole_cards = ["AS", "KS"]
    community_cards = ["QS", "JS", "TS", "2H", "3D"]
    score, desc = evaluator.evaluate_hand(hole_cards, community_cards)
    print(f"Hole cards: {hole_cards}, Community cards: {community_cards}")
    print(f"Best hand: {desc} (Score: {score})")
    
    # Find best 5-card hand
    best_hand = HandEvaluator.get_best_hand(hole_cards, community_cards)
    print(f"Best 5 cards: {best_hand}")
    
    # Test performance
    test_performance(1000)
