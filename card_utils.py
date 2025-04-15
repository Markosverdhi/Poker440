# filename: card_utils.py
"""
Utilities for card representation and rendering.
Includes ASCII art rendering.
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

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

def get_card_parts(card_str):
    """ Parses 'TH' into ('T', 'H') or 'AS' into ('A', 'S') """
    if not isinstance(card_str, str) or len(card_str) < 2:
        return "?", "?" # Invalid card string

    rank = card_str[:-1] # Handles 'T'(10), 'J', 'Q', 'K', 'A'
    suit = card_str[-1]

    # Validate rank and suit
    if rank not in RANKS or suit not in SUITS_UNICODE:
         # Fallback for potential '10H' format if needed, though 'TH' is standard
         if len(card_str) == 3 and card_str[:2] == '10':
              rank = 'T' # Convert '10' to 'T' internally if needed
              suit = card_str[2]
              if suit not in SUITS_UNICODE: return "?", "?"
         else:
              return "?", "?"

    return rank, suit

def render_card_ascii(card_str):
    """
    Renders a single card string (e.g., 'KH', 'TS') into a simple
    multi-line ASCII/Unicode art representation. Best viewed with a
    monospace font.
    """
    rank, suit = get_card_parts(card_str)
    suit_symbol = SUITS_UNICODE.get(suit, '?')
    # color = SUIT_COLORS.get(suit, 'black') # Color info isn't used by label

    # Simple ASCII Art Card representation
    top_border    = "+-----+"
    rank_line_top = f"|{rank:<2}   |" # Rank left-aligned, 2 spaces
    middle_line   = f"|  {suit_symbol}  |"
    rank_line_bot = f"|   {rank:>2}|" # Rank right-aligned, 2 spaces
    bottom_border = "+-----+"

    # Combine lines for the label text (use newline characters)
    return f"{top_border}\n{rank_line_top}\n{middle_line}\n{rank_line_bot}\n{bottom_border}"

def render_hand(card_list, separator="  "):
    """
    Renders a list of card strings into a single multi-line string,
    attempting to align the ASCII art cards horizontally.

    NOTE: Aligning multi-line strings horizontally within a single Tkinter
          label is difficult. This function might be better used to generate
          text for individual card labels placed side-by-side.
          Returning simple joined rendering for now.
    """
    if not card_list:
        return ""
    # Simple rendering for showdown display (less alignment needed)
    return separator.join([f"{c[:-1]}{SUITS_UNICODE.get(c[-1], '?')}" for c in card_list])

def render_hand_for_labels(card_list):
     """ Returns a list of individual card ASCII art strings """
     if not card_list:
          return ["", ""] # Return empty strings for 2 labels
     renders = [render_card_ascii(card) for card in card_list]
     # Ensure list has 2 elements for the two player card labels
     while len(renders) < 2:
          renders.append(render_card_ascii("??")) # Placeholder for missing card
     return renders[:2]

def render_community_cards_for_labels(card_list):
    """ Returns a list of individual card ASCII art strings for community cards """
    max_cards = 5
    renders = []
    # Render existing cards
    for card in card_list:
         renders.append(render_card_ascii(card))
    # Pad with empty card placeholders up to max_cards
    empty_card = "+-----+\n|     |\n|  ?  |\n|     |\n+-----+" # Placeholder visual
    while len(renders) < max_cards:
         renders.append(empty_card)
    return renders[:max_cards]

