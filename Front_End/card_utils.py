import random
from Back_End.improved_hand_evaluator import Card, HandEvaluator, evaluate_hand

# Unicode suit symbols
SUITS_UNICODE = {
    'S': '\u2660',  # Spades (Black ♠)
    'H': '\u2665',  # Hearts (Red ♥)
    'D': '\u2666',  # Diamonds (Red ♦)
    'C': '\u2663'   # Clubs (Black ♣)
}

# Suit colors for display
SUIT_COLORS = {
    'S': 'black',
    'H': 'red',
    'D': 'red',
    'C': 'black'
}

# Ranks for internal representation
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

def get_card_parts(card_str):
    if not isinstance(card_str, str) or len(card_str) < 2:
        return "?", "?"

    rank = card_str[:-1]
    suit = card_str[-1]

    if rank not in RANKS or suit not in SUITS_UNICODE:
        if len(card_str) == 3 and card_str[:2] == '10':
            rank = 'T'
            suit = card_str[2]
            if suit not in SUITS_UNICODE:
                return "?", "?"
        else:
            return "?", "?"

    return rank, suit

def render_card_ascii(card_str):
    rank, suit = get_card_parts(card_str)
    suit_symbol = SUITS_UNICODE.get(suit, '?')
    
    display_rank = "10" if rank == 'T' else rank

    top_border    = "+-----+"
    rank_line_top = f"|{display_rank:<2}   |"
    middle_line   = f"|  {suit_symbol}  |"
    rank_line_bot = f"|   {display_rank:>2}|"
    bottom_border = "+-----+"

    return f"{top_border}\n{rank_line_top}\n{middle_line}\n{rank_line_bot}\n{bottom_border}"

def render_hand(card_list, separator="  "):
    if not card_list:
        return ""
    rendered_cards = []
    for c in card_list:
        rank, suit_char = get_card_parts(c)
        display_rank = "10" if rank == 'T' else rank
        suit_symbol = SUITS_UNICODE.get(suit_char, '?')
        rendered_cards.append(f"{display_rank}{suit_symbol}")

    return separator.join(rendered_cards)

def render_hand_for_labels(card_list):
    if not card_list:
        return [render_card_ascii("??"), render_card_ascii("??")]
    renders = [render_card_ascii(card) for card in card_list]
    while len(renders) < 2:
        renders.append(render_card_ascii("??"))
    return renders[:2]

def render_community_cards_for_labels(card_list):
    max_cards = 5
    renders = []
    for card in card_list:
        renders.append(render_card_ascii(card))

    placeholder_art = "+-----+\n|     |\n|  ?  |\n|     |\n+-----+"
    while len(renders) < max_cards:
        renders.append(placeholder_art)
    return renders[:max_cards]

# New utility functions

def create_deck():
    return [f"{r}{s}" for r in RANKS for s in SUITS_UNICODE.keys()]

def shuffle_deck():
    deck = create_deck()
    random.shuffle(deck)
    return deck

def deal_cards(deck, num_players, num_cards=2):
    if len(deck) < num_players * num_cards:
        raise ValueError("Not enough cards in deck")
    
    hands = []
    for i in range(num_players):
        hand = []
        for j in range(num_cards):
            hand.append(deck.pop())
        hands.append(hand)
    
    return hands

def deal_community_cards(deck, num_cards=5):
    if len(deck) < num_cards:
        raise ValueError("Not enough cards in deck")
    
    return [deck.pop() for _ in range(num_cards)]

def find_best_hand(hole_cards, community_cards):
    evaluator = HandEvaluator()
    return evaluator.get_best_hand(hole_cards, community_cards)

def compare_hands(hands, community_cards):
    evaluator = HandEvaluator()
    scores = []
    
    for hand in hands:
        score, description = evaluator.evaluate_hand(hand, community_cards)
        scores.append((hand, score, description))
    
    # Sort by score (highest first)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Group hands with the same score (ties)
    result = []
    current_score = None
    current_group = []
    
    for hand, score, description in scores:
        if score != current_score:
            if current_group:
                result.append(current_group)
            current_group = [(hand, description)]
            current_score = score
        else:
            current_group.append((hand, description))
    
    if current_group:
        result.append(current_group)
    
    return result

def get_hand_odds(hole_cards, community_cards=None, num_opponents=1, num_simulations=1000):
    if community_cards is None:
        community_cards = []
    
    evaluator = HandEvaluator()
    wins = 0
    ties = 0
    
    for _ in range(num_simulations):
        # Create a new deck without the known cards
        deck = create_deck()
        for card in hole_cards + community_cards:
            if card in deck:
                deck.remove(card)
        
        random.shuffle(deck)
        
        # Deal opponent hands
        opponent_hands = []
        for _ in range(num_opponents):
            opponent_hand = []
            for _ in range(2):  # 2 cards per opponent
                opponent_hand.append(deck.pop())
            opponent_hands.append(opponent_hand)
        
        # Complete community cards if needed
        remaining_community = 5 - len(community_cards)
        simulation_community = community_cards + [deck.pop() for _ in range(remaining_community)]
        
        # Evaluate all hands
        player_score, _ = evaluator.evaluate_hand(hole_cards, simulation_community)
        
        # Check if player wins or ties
        player_wins = True
        player_ties = False
        
        for opponent_hand in opponent_hands:
            opponent_score, _ = evaluator.evaluate_hand(opponent_hand, simulation_community)
            
            if opponent_score > player_score:
                player_wins = False
                player_ties = False
                break
            elif opponent_score == player_score:
                player_wins = False
                player_ties = True
        
        if player_wins:
            wins += 1
        elif player_ties:
            ties += 1
    
    win_percentage = (wins / num_simulations) * 100
    tie_percentage = (ties / num_simulations) * 100
    loss_percentage = 100 - win_percentage - tie_percentage
    
    return {
        'win': win_percentage,
        'tie': tie_percentage,
        'loss': loss_percentage
    }

def get_hand_strength_description(hole_cards, community_cards=None):
    if community_cards is None:
        community_cards = []
    
    # Get hand evaluation
    _, description = evaluate_hand(hole_cards, community_cards)
    
    # Get odds against 1 opponent
    odds = get_hand_odds(hole_cards, community_cards, num_opponents=1, num_simulations=500)
    
    strength_levels = [
        (80, "Very Strong"),
        (60, "Strong"),
        (45, "Above Average"),
        (30, "Average"),
        (15, "Weak"),
        (0, "Very Weak")
    ]
    
    strength = "Unknown"
    for threshold, level in strength_levels:
        if odds['win'] >= threshold:
            strength = level
            break
    
    return {
        'description': description,
        'strength': strength,
        'win_percentage': odds['win'],
        'tie_percentage': odds['tie'],
        'loss_percentage': odds['loss']
    }

if __name__ == "__main__":
    # Test the card utilities
    print("Testing card utilities...")
    
    # Create and shuffle a deck
    deck = shuffle_deck()
    print(f"Deck created with {len(deck)} cards")
    
    # Deal hands
    hands = deal_cards(deck, 4)
    print(f"Dealt 4 hands: {hands}")
    
    # Deal community cards
    community = deal_community_cards(deck, 5)
    print(f"Community cards: {community}")
    
    # Render a hand
    print(f"Rendered hand: {render_hand(hands[0])}")
    
    # Evaluate hands
    for i, hand in enumerate(hands):
        score, desc = evaluate_hand(hand, community)
        print(f"Hand {i+1}: {hand} - {desc} (Score: {score})")
    
    # Compare hands
    results = compare_hands(hands, community)
    print("\nHand rankings:")
    for i, group in enumerate(results):
        print(f"Rank {i+1}:")
        for hand, desc in group:
            print(f"  {hand} - {desc}")
    
    # Get hand odds
    odds = get_hand_odds(hands[0], community, num_opponents=3, num_simulations=1000)
    print(f"\nHand odds with 3 opponents (1000 simulations):")
    print(f"Win: {odds['win']:.2f}%")
    print(f"Tie: {odds['tie']:.2f}%")
    print(f"Loss: {odds['loss']:.2f}%")
    
    # Get hand strength description
    strength = get_hand_strength_description(hands[0], community)
    print(f"\nHand strength: {strength['strength']} ({strength['description']})")
    print(f"Win: {strength['win_percentage']:.2f}%, Tie: {strength['tie_percentage']:.2f}%, Loss: {strength['loss_percentage']:.2f}%")
