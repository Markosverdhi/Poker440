# card_utils.py

# Unicode suit symbols
SUITS_UNICODE = {
    'S': '♠', # Spades (Black)
    'H': '♥', # Hearts (Red)
    'D': '♦', # Diamonds (Red)
    'C': '♣'  # Clubs (Black)
}

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

def get_card_parts(card_str):
    """ Parses '10H' into ('10', 'H') or 'AS' into ('A', 'S') """
    if len(card_str) == 3: # Handle '10' rank
        rank = card_str[:2]
        suit = card_str[2]
    elif len(card_str) == 2:
        rank = card_str[0]
        suit = card_str[1]
    else:
        return "?", "?" # Invalid card string
    return rank, suit

def render_card(card_str):
    """
    Renders a single card string (e.g., 'KH', '10S') into a simple
    Unicode representation with a border.
    """
    rank, suit = get_card_parts(card_str)
    suit_symbol = SUITS_UNICODE.get(suit, '?')

    # Basic fixed-width representation
    # Adjust spacing based on rank length ('10' vs 'K')
    rank_display = rank.ljust(2) # Left-justify rank in 2 spaces

    # Simple border
    top_border = "+-----+"
    middle = f"| {rank_display}  |"
    suit_line = f"|  {suit_symbol}  |"
    bottom_border= "+-----+"

    # Combine lines for the label text
    # Note: Tkinter Label doesn't directly support multi-line text easily
    # without complex configurations. Returning a simple representation for now.
    # For better display, one might use a Canvas or multiple labels per card.
    # This simplified version fits better in standard Labels.
    # return f"{rank}{suit_symbol}" # Simplest possible
    return f"{rank.ljust(2)} {suit_symbol}" # Rank + Suit Symbol


def render_hand(card_list):
    """
    Renders a list of card strings into a single string with spacing.
    (Less useful now as we render cards into separate labels).
    """
    if not card_list:
        return ""
    return "   ".join([render_card(card) for card in card_list])