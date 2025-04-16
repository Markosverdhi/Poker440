# filename: card_utils.py
"""
Utilities for card representation and rendering.
Includes ASCII art rendering.

MODIFIED: Changed display of Ten card from 'T' to '10' in rendering functions.
MODIFIED: Reverted render_community_cards_for_labels to output multi-line
          ASCII/Unicode art using render_card_ascii.
"""

# Unicode suit symbols
SUITS_UNICODE = {
    'S': '\u2660', # Spades (Black ♠)
    'H': '\u2665', # Hearts (Red ♥)
    'D': '\u2666', # Diamonds (Red ♦)
    'C': '\u2663'  # Clubs (Black ♣)
}
# For simple color distinction (won't render in basic Tkinter labels)
SUIT_COLORS = {
    'S': 'black',
    'H': 'red',
    'D': 'red',
    'C': 'black'
}

# Internal representation uses 'T' for Ten
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

def get_card_parts(card_str):
    """ Parses 'TH' into ('T', 'H') or 'AS' into ('A', 'S') """
    if not isinstance(card_str, str) or len(card_str) < 2:
        return "?", "?" # Invalid card string

    rank = card_str[:-1] # Handles 'T'(10), 'J', 'Q', 'K', 'A'
    suit = card_str[-1]

    # Validate rank and suit based on internal representation
    if rank not in RANKS or suit not in SUITS_UNICODE:
         # Fallback for potential '10H' format if needed, though 'TH' is standard
         if len(card_str) == 3 and card_str[:2] == '10':
              rank = 'T' # Convert '10' to 'T' internally
              suit = card_str[2]
              if suit not in SUITS_UNICODE: return "?", "?"
         else:
              return "?", "?"

    return rank, suit # Returns 'T' for Ten

def render_card_ascii(card_str):
    """
    Renders a single card string (e.g., 'KH', 'TS') into a simple
    multi-line ASCII/Unicode art representation. Displays '10' for Ten.
    Best viewed with a monospace font.
    """
    rank, suit = get_card_parts(card_str)
    suit_symbol = SUITS_UNICODE.get(suit, '?')
    # color = SUIT_COLORS.get(suit, 'black') # Color info isn't used by label

    # Display '10' instead of 'T'
    display_rank = "10" if rank == 'T' else rank

    # Simple ASCII Art Card representation
    top_border    = "+-----+"
    rank_line_top = f"|{display_rank:<2}   |"
    middle_line   = f"|  {suit_symbol}  |"
    rank_line_bot = f"|   {display_rank:>2}|"
    bottom_border = "+-----+"

    # Combine lines for the label text (use newline characters)
    return f"{top_border}\n{rank_line_top}\n{middle_line}\n{rank_line_bot}\n{bottom_border}"

def render_hand(card_list, separator="  "):
    """
    Renders a list of card strings into a simple single-line string
    using Rank + Unicode Suit (e.g., "A♥ K♦"). Displays '10' for Ten.
    Used for showdown text display.
    """
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
     """
     Returns a list of individual card ASCII art strings suitable for
     separate labels (e.g., player hand).
     """
     if not card_list:
          # Return placeholder art if no cards
          return [render_card_ascii("??"), render_card_ascii("??")]
     renders = [render_card_ascii(card) for card in card_list]
     # Ensure list has 2 elements for the two player card labels
     while len(renders) < 2:
          renders.append(render_card_ascii("??")) # Placeholder for missing card
     return renders[:2]

# --- Reverted Function ---
def render_community_cards_for_labels(card_list):
    """
    Returns a list of individual card ASCII art strings for community cards,
    padded with placeholder art.
    """
    max_cards = 5
    renders = []
    # Render existing cards using ASCII art function
    for card in card_list:
         renders.append(render_card_ascii(card)) # Use ASCII art

    # Pad with ASCII art placeholders up to max_cards
    # Define the placeholder art string
    placeholder_art = "+-----+\n|     |\n|  ?  |\n|     |\n+-----+"
    while len(renders) < max_cards:
         renders.append(placeholder_art) # Append placeholder art
    return renders[:max_cards]

