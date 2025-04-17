import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys
import time
import random
import math
# Conditional import for torch, only if needed for AI models
try:
    import torch
except ImportError:
    print("Warning: PyTorch not found. AI model functionality will be disabled.")
    torch = None # Define torch as None if not available

from PIL import Image, ImageTk, ImageDraw, ImageFont

# --- Adjust imports based on your project structure ---
# This block attempts to import modules relative to the script's location first,
# then falls back to assuming a specific project structure if relative imports fail.
try:
    # Assume relative imports work from within a package (e.g., running as part of Front_End)
    from .poker_theme import PokerTheme
    from .animations import AnimationManager
    from .seat_config import SeatConfigManager
    # Add Back_End to sys.path relative to this file's directory
    # This allows finding modules in the sibling 'Back_End' directory
    backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Back_End'))
    if backend_path not in sys.path:
        sys.path.append(backend_path)
    from Back_End.envs import BaseFullPokerEnv # Assuming BaseFullPokerEnv is directly in Back_End/envs
    from Back_End.utils import load_agent_model, get_opponent_policy, encode_obs_eval # Assuming these are in Back_End/utils
    from Back_End.constants import NUM_PLAYERS, ACTION_LIST, NUM_ACTIONS, STARTING_STACK # Assuming these are in Back_End/constants
    from .card_utils import render_hand # Assuming relative import for card_utils works
    print("Relative imports successful.")
except ImportError as e:
    print(f"Error importing modules using relative paths: {e}")
    print("Attempting fallback imports (assuming script is run from project root or similar)...")
    # Fallback: Assumes 'Front_End' and 'Back_End' are top-level directories accessible from cwd
    try:
        # Get parent directory of the current script's directory might be needed if running script directly
        # parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # if parent_dir not in sys.path:
        #      sys.path.append(parent_dir) # Add project root to path

        # Try importing assuming Front_End and Back_End are importable packages
        from .poker_theme import PokerTheme
        from .animations import AnimationManager
        from .seat_config import SeatConfigManager
        from Back_End.envs import BaseFullPokerEnv
        from Back_End.utils import load_agent_model, get_opponent_policy, encode_obs_eval
        from Back_End.constants import NUM_PLAYERS, ACTION_LIST, NUM_ACTIONS, STARTING_STACK
        from .card_utils import render_hand
        print("Fallback imports successful.")
    except ImportError as e2:
        print(f"Fallback import failed: {e2}")
        messagebox.showerror("Import Error", f"Could not import necessary game modules.\nPlease ensure the project structure allows importing 'Front_End' and 'Back_End' modules.\nError: {e2}")
        sys.exit(1)
# --- End Imports ---


class CardImageGenerator:
    """
    Generates and caches images for playing cards using PIL and Tkinter.
    Handles card faces, backs, placeholders, and resizing.
    """
    def __init__(self, card_width=80, card_height=120):
        """
        Initializes the generator with base card dimensions.

        Args:
            card_width (int): Base width for generated cards.
            card_height (int): Base height for generated cards.
        """
        self.base_card_width = card_width
        self.base_card_height = card_height
        self.card_images = {} # Cache for generated ImageTk.PhotoImage objects { (code, size): PhotoImage }

        # Generate default placeholder immediately with base size
        self._default_placeholder_image = self._create_placeholder_image((self.base_card_width, self.base_card_height))
        if self._default_placeholder_image:
            # Convert the default PIL image to PhotoImage for direct use
            self.default_placeholder = ImageTk.PhotoImage(self._default_placeholder_image)
        else:
            # Fallback if placeholder creation failed (should not happen ideally)
            self.default_placeholder = None
            print("CRITICAL WARNING: Failed to create default placeholder image.")

    def get_card_image(self, card_code, size=None):
        """
        Retrieves or generates a PhotoImage for a given card code and size.

        Args:
            card_code (str): Standard 2-character card code (e.g., "As", "Td", "Kc") or "??", or "placeholder".
            size (tuple, optional): (width, height) for the desired image size. Defaults to base size.

        Returns:
            ImageTk.PhotoImage: The requested card image, or a placeholder if generation fails.
        """
        # Determine target size, defaulting to base size
        target_size = size or (self.base_card_width, self.base_card_height)
        target_size = (int(target_size[0]), int(target_size[1])) # Ensure integers
        cache_key = (card_code, target_size)

        # --- Check Cache ---
        if cache_key in self.card_images:
            # Check if the cached PhotoImage is still valid (Tkinter objects can become invalid)
            try:
                 # Accessing a property like width will raise TclError if invalid
                 _ = self.card_images[cache_key].width()
                 return self.card_images[cache_key] # Return cached image if valid
            except (tk.TclError, AttributeError):
                # print(f"Cached image for {cache_key} is invalid, regenerating.")
                del self.card_images[cache_key] # Remove invalid entry

        # --- Validate Size ---
        if target_size[0] <= 0 or target_size[1] <= 0:
            # print(f"Warning: Requested invalid size {target_size} for card '{card_code}'. Using placeholder.")
            return self.get_placeholder_image() # Return default placeholder

        # --- Generate PIL Image ---
        img = None # Initialize img to None
        if card_code == "??" or card_code is None:
            img = self._create_card_back(target_size)
        elif card_code == "placeholder": # Explicit request for placeholder
             img = self._create_placeholder_image(target_size)
        else:
            # Basic validation for card code format (optional but helpful)
            if isinstance(card_code, str) and len(card_code) == 2 and card_code[0] in "23456789TJQKA" and card_code[1] in "SHDC":
                 img = self._create_card_image(card_code, target_size)
            else:
                 print(f"Warning: Invalid card code format '{card_code}'. Generating placeholder.")
                 img = self._create_placeholder_image(target_size)


        # Handle PIL image creation failure
        if img is None:
            # print(f"Warning: Failed to create PIL image for {cache_key}. Using placeholder.")
            return self.get_placeholder_image(target_size) # Attempt specific size placeholder

        # --- Convert PIL Image to PhotoImage ---
        try:
            photo_img = ImageTk.PhotoImage(img)
            self.card_images[cache_key] = photo_img # Store in cache
            return photo_img
        except Exception as e:
            # This can happen if Tkinter is shutting down, image data is corrupt, etc.
            print(f"Error creating PhotoImage for {cache_key}: {e}")
            # Fallback to placeholder on PhotoImage creation error
            return self.get_placeholder_image(target_size)

    def get_card_back(self, size=None):
        """Gets a card back image at the specified size."""
        target_size = size or (self.base_card_width, self.base_card_height)
        return self.get_card_image("??", target_size)

    def get_placeholder_image(self, size=None):
        """Gets a placeholder image at the specified size."""
        target_size = size or (self.base_card_width, self.base_card_height)
        target_size = (int(target_size[0]), int(target_size[1])) # Ensure integers

        # Use default if no specific size requested and default exists
        if size is None and self.default_placeholder:
            return self.default_placeholder

        cache_key = ("placeholder", target_size)

        # Check cache for specific size placeholder
        if cache_key in self.card_images:
             # Check if the cached PhotoImage is still valid
            try:
                 _ = self.card_images[cache_key].width()
                 return self.card_images[cache_key]
            except (tk.TclError, AttributeError):
                # print(f"Cached placeholder image for {cache_key} is invalid, regenerating.")
                del self.card_images[cache_key]

        # Ensure size is valid before generating
        if target_size[0] <= 0 or target_size[1] <= 0:
            # print(f"Warning: Requested invalid size {target_size} for placeholder. Using default.")
            return self.default_placeholder # Fallback to default

        # Generate PIL image for placeholder
        img = self._create_placeholder_image(target_size)

        if img is None:
             # print(f"Warning: Failed to create PIL image for placeholder {target_size}. Using default.")
             return self.default_placeholder # Fallback to default

        # Convert PIL Image to PhotoImage
        try:
            photo_img = ImageTk.PhotoImage(img)
            self.card_images[cache_key] = photo_img # Cache the generated placeholder
            return photo_img
        except Exception as e:
            print(f"Error creating PhotoImage for placeholder {target_size}: {e}")
            return self.default_placeholder # Fallback to default

    def _create_rounded_rectangle(self, draw, xy, radius, fill, outline=None, width=1):
        """
        Draws a rounded rectangle using PIL Draw methods.
        Handles potential issues with radius size and line width adjustments.

        Args:
            draw (ImageDraw.Draw): The PIL Draw object.
            xy (tuple): Coordinates (x1, y1, x2, y2) of the bounding box.
            radius (int): Corner radius.
            fill (str): Fill color.
            outline (str, optional): Outline color. Defaults to None.
            width (int, optional): Outline width. Defaults to 1.
        """
        x1, y1, x2, y2 = xy
        # Ensure coordinates are valid (width and height > 0)
        if x2 <= x1 or y2 <= y1:
            # print(f"Warning: Invalid dimensions for rounded rectangle: {xy}")
            return

        # Ensure radius is not larger than half the shortest side and non-negative
        max_radius = min((x2 - x1) / 2, (y2 - y1) / 2)
        radius = max(0, min(radius, max_radius)) # Clamp radius: 0 <= radius <= max_radius

        # If radius is effectively zero, draw a simple rectangle
        if radius < 0.5: # Use a small threshold instead of < 1
            draw.rectangle(xy, fill=fill, outline=outline, width=width)
            return

        # Draw the main body rectangles (no outline here, fill only)
        # Horizontal rectangle
        draw.rectangle((x1 + radius, y1, x2 - radius, y2), fill=fill, outline=None)
        # Vertical rectangle
        draw.rectangle((x1, y1 + radius, x2, y2 - radius), fill=fill, outline=None)

        # Draw the corner arcs (pieslices for fill)
        diam = 2 * radius
        draw.pieslice((x1, y1, x1 + diam, y1 + diam), 180, 270, fill=fill, outline=None) # Top-left
        draw.pieslice((x2 - diam, y1, x2, y1 + diam), 270, 360, fill=fill, outline=None) # Top-right
        draw.pieslice((x1, y2 - diam, x1 + diam, y2), 90, 180, fill=fill, outline=None)  # Bottom-left
        draw.pieslice((x2 - diam, y2 - diam, x2, y2), 0, 90, fill=fill, outline=None)    # Bottom-right

        # Draw the outline if specified
        if outline and width > 0:
             # Draw corner arcs for outline
             draw.arc((x1, y1, x1 + diam, y1 + diam), 180, 270, fill=outline, width=width)
             draw.arc((x2 - diam, y1, x2, y1 + diam), 270, 360, fill=outline, width=width)
             draw.arc((x1, y2 - diam, x1 + diam, y2), 90, 180, fill=outline, width=width)
             draw.arc((x2 - diam, y2 - diam, x2, y2), 0, 90, fill=outline, width=width)

             # Draw straight line segments for outline
             # Adjust coordinates slightly for better line connection with arcs if width > 1
             adj = width / 2.0
             draw.line([(x1 + radius, y1 + adj), (x2 - radius, y1 + adj)], fill=outline, width=width) # Top edge
             draw.line([(x1 + radius, y2 - adj), (x2 - radius, y2 - adj)], fill=outline, width=width) # Bottom edge
             draw.line([(x1 + adj, y1 + radius), (x1 + adj, y2 - radius)], fill=outline, width=width) # Left edge
             draw.line([(x2 - adj, y1 + radius), (x2 - adj, y2 - radius)], fill=outline, width=width) # Right edge

    def _create_placeholder_image(self, size):
        """Creates a simple gray rounded rectangle PIL image as a placeholder."""
        width, height = int(size[0]), int(size[1])
        if width <= 0 or height <= 0: return None
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0)) # Transparent background
        draw = ImageDraw.Draw(img)
        radius = max(1, int(min(width, height) * 0.1)) # Dynamic radius
        # Draw slightly inset to avoid border clipping
        inset = 1
        self._create_rounded_rectangle(draw, (inset, inset, width - 1 - inset, height - 1 - inset),
                                       radius, fill="#555555", outline="#888888", width=1) # Dark gray fill, light gray border
        return img

    def _create_card_image(self, card_code, size):
        """Creates a PIL image for a specific playing card face."""
        width, height = int(size[0]), int(size[1])
        if width <= 0 or height <= 0: return None

        # --- Card Properties ---
        rank, suit = card_code[:-1], card_code[-1]
        symbols = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}
        colors = {'S': "#000000", 'C': "#000000", 'H': "#C14953", 'D': "#C14953"} # Black and Red
        symbol = symbols.get(suit, '?') # Get suit symbol, default to '?'
        color = colors.get(suit, "#000000") # Get suit color, default to black

        # --- Base Image and Background ---
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0)) # Start with transparent background
        draw = ImageDraw.Draw(img)
        radius = max(1, int(min(width, height) * 0.1)) # Dynamic radius based on size
        inset = 1 # Inset drawing slightly from edge
        self._create_rounded_rectangle(draw, (inset, inset, width - 1 - inset, height - 1 - inset),
                                       radius, fill="#FFFFFF", outline="#333333", width=1) # White fill, dark gray border

        # --- Font Scaling ---
        # Scale font sizes based on card height, with min/max limits for readability
        rank_font_size = max(8, min(24, int(height * 0.18)))
        suit_font_size = max(10, min(26, int(height * 0.20)))
        center_font_size = max(15, min(60, int(height * 0.45)))

        # --- Font Loading (with fallbacks) ---
        try:
            # Prioritize common system fonts (Arial/Helvetica/DejaVu Sans)
            rank_font = ImageFont.truetype("arialbd.ttf", rank_font_size) # Bold for rank
            suit_font = ImageFont.truetype("arial.ttf", suit_font_size)
            center_font = ImageFont.truetype("arial.ttf", center_font_size)
        except IOError:
            try:
                rank_font = ImageFont.truetype("DejaVuSans-Bold.ttf", rank_font_size)
                suit_font = ImageFont.truetype("DejaVuSans.ttf", suit_font_size)
                center_font = ImageFont.truetype("DejaVuSans.ttf", center_font_size)
            except IOError:
                 try:
                     rank_font = ImageFont.truetype("HelveticaNeue-Bold.ttf", rank_font_size)
                     suit_font = ImageFont.truetype("HelveticaNeue.ttf", suit_font_size)
                     center_font = ImageFont.truetype("HelveticaNeue.ttf", center_font_size)
                 except IOError:
                    # Absolute fallback if no preferred fonts found
                    print("Warning: Could not load preferred fonts (Arial, DejaVu Sans, Helvetica Neue). Using default.")
                    rank_font = ImageFont.load_default()
                    suit_font = ImageFont.load_default()
                    center_font = ImageFont.load_default()

        # --- Text Rendering ---
        rank_text = rank if rank != 'T' else '10' # Handle 'T' for Ten

        # Helper to get text dimensions using getbbox (more accurate) or fallback getsize
        def get_text_dims(text, font):
            try:
                # getbbox returns (left, top, right, bottom) relative to origin
                bbox = font.getbbox(text)
                return bbox[2] - bbox[0], bbox[3] - bbox[1] # width, height
            except AttributeError:
                # Fallback for older PIL/Pillow versions
                return font.getsize(text)

        r_w, r_h = get_text_dims(rank_text, rank_font)
        s_w, s_h = get_text_dims(symbol, suit_font)
        c_w, c_h = get_text_dims(symbol, center_font)

        # Margins (scale slightly with card size)
        margin_x = max(3, int(width * 0.07))
        margin_y = max(3, int(height * 0.05))
        suit_offset_y = max(1, int(height * 0.02)) # Small vertical gap between rank and suit

        # --- Draw Text Elements ---
        # Top-left rank and suit
        draw.text((margin_x, margin_y), rank_text, fill=color, font=rank_font)
        draw.text((margin_x, margin_y + r_h + suit_offset_y), symbol, fill=color, font=suit_font)

        # Bottom-right rank and suit (mirrored placement, text remains upright)
        # Calculate positions carefully
        br_rank_x = width - margin_x - r_w
        br_rank_y = height - margin_y - r_h - s_h - suit_offset_y # Rank Y position (higher)
        br_suit_x = width - margin_x - s_w
        br_suit_y = height - margin_y - s_h # Suit Y position (lower)

        draw.text((br_rank_x, br_rank_y), rank_text, fill=color, font=rank_font)
        draw.text((br_suit_x, br_suit_y), symbol, fill=color, font=suit_font)

        # Center symbol (adjust y slightly upwards from true center)
        center_x = (width - c_w) / 2
        center_y = (height - c_h) / 2 - int(height * 0.03) # Nudge up slightly
        draw.text((center_x, center_y), symbol, fill=color, font=center_font)

        return img

    def _create_card_back(self, size):
        """Creates a PIL image for the back of a playing card."""
        width, height = int(size[0]), int(size[1])
        if width <= 0 or height <= 0: return None
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0)) # Transparent background
        draw = ImageDraw.Draw(img)
        radius = max(1, int(min(width, height) * 0.1)) # Dynamic radius
        inset = 1 # Draw slightly inset

        # Main card back color and border
        self._create_rounded_rectangle(draw, (inset, inset, width - 1 - inset, height - 1 - inset),
                                       radius, fill="#0D5C3C", outline="#000000", width=1) # Dark Green, Black border

        # --- Optional: Add a subtle pattern ---
        # This adds visual interest but can be removed if performance is critical
        pattern_color = "#0A4A30" # Darker green for pattern
        try:
            # Simple repeating pattern (e.g., small dots) within the card bounds
            # Adjust step based on card size for consistent density
            step = max(4, int(min(width, height) * 0.08))
            dot_size = max(1, int(step * 0.2)) # Small dots

            # Iterate within the rounded rectangle's inner area
            for x in range(int(radius + inset + step/2), int(width - radius - inset), step):
                for y in range(int(radius + inset + step/2), int(height - radius - inset), step):
                    # Simple dot pattern
                    draw.ellipse((x - dot_size // 2, y - dot_size // 2,
                                  x + dot_size // 2 + 1, y + dot_size // 2 + 1), # +1 for inclusive coords
                                 fill=pattern_color, outline=None)
        except Exception as e:
            print(f"Warning: Error drawing card back pattern: {e}") # Non-critical error

        return img

# --- End CardImageGenerator ---


class PokerApp:
    """
    Main application class for the Poker GUI.
    Handles UI creation, game state management, and interaction with the poker environment.
    """
    def __init__(self, root):
        """
        Initializes the PokerApp.

        Args:
            root (tk.Tk): The main Tkinter window.
        """
        self.root = root
        self.root.title("Enhanced Poker Game")
        # Increased initial size slightly for better spacing on larger screens
        self.root.geometry("1250x900")
        self.root.minsize(1100, 750) # Minimum size to prevent layout issues
        # Configure root window background color for a consistent look
        self.root.configure(bg="#2E2E2E") # Dark background

        # --- Theme and Styles ---
        self.theme = PokerTheme(root) # Initialize custom theme styles
        # Define specific styles used in the application
        self._configure_styles()

        # --- Core Components ---
        self.card_generator = CardImageGenerator() # For creating card images
        self.animation_manager = AnimationManager(root) # For potential animations (not fully implemented here)
        self.seat_config_manager = SeatConfigManager(NUM_PLAYERS) # Manages seat type configurations

        # --- Game State Variables ---
        self.env = None # The poker environment instance
        self.agent_model = None # Loaded AI model for default opponents
        self.opponent_models = {} # Specific models loaded for individual seats {seat_id: model}
        self.device = None # PyTorch device ('cpu' or 'cuda')
        if torch: # Setup device only if torch was imported successfully
             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
             print(f"Using device: {self.device}")
        else:
             print("PyTorch not available, running without AI model support.")

        self.current_game_state = None # Stores the latest observation from the environment
        self.human_player_id = 0 # Default seat index for the human player
        self.seat_config = self.seat_config_manager.default_config.copy() # Current seat setup {id: type}
        self.checkpoint_paths = {i: None for i in range(NUM_PLAYERS)} # Paths to specific AI models {id: path}
        self.action_map = {i: s for i, s in enumerate(ACTION_LIST)} # Map action index to string
        self.string_to_action = {s: i for i, s in enumerate(ACTION_LIST)} # Map action string to index
        self.is_game_running = False # Flag indicating if a tournament is active
        self.is_human_turn = False # Flag indicating if it's the human's turn to act
        self.last_round_summary = "No previous round data." # Stores text summary of the last hand

        # --- UI Widget References ---
        # Keep references to key widgets to update them later
        self.main_frame = None
        self.header_frame = None
        self.table_container = None # Frame holding the table_frame, used for centering
        self.table_frame = None # Themed frame representing the table surface
        self.pot_label = None
        self.seat_positions = [] # List of dicts defining relative seat placement
        self.seat_frames_widgets = {} # { seat_id: { 'frame': LabelFrame, 'status': Label, ... } }

        # Base dimensions for player seats (used for scaling)
        # **FIX**: Reduced base seat size slightly for better spacing
        self.base_seat_width = 280
        self.base_seat_height = 125

        self.community_container = None # Frame holding community card labels
        self.community_card_labels = [] # List of Labels for community cards
        self.community_card_size = (0, 0) # Current size of community cards (updated dynamically)

        self.action_panel = None # LabelFrame holding action buttons
        self.action_buttons_frame = None # Frame inside action_panel for button layout
        self.action_buttons = {} # { action_name: Button }

        self.info_panel = None # LabelFrame for game info (round, blinds, etc.)
        self.round_value = None # Label for current round/stage
        self.dealer_value = None # Label for dealer button position
        self.blinds_value = None # Label for small/big blind amounts
        self.current_value = None # Label for the current player's turn
        self.tocall_value = None # Label for amount needed to call

        self.last_round_panel = None # LabelFrame for last round summary
        self.last_round_summary_label = None # Label displaying the summary text

        # **FIX**: Frame to hold all bottom panels, used with grid layout for centering table
        self.bottom_panels_frame = None

        self._resize_job = None # Stores ID for scheduled resize task (debouncing)

        # --- Build UI ---
        self._create_ui() # Create all the widgets

        # --- Bind Resize Event ---
        # Use add='+' to avoid overriding other potential bindings
        self.root.bind("<Configure>", self._on_window_resize, add='+')

        # --- Initial Draw and Game Setup ---
        self.root.update_idletasks() # Ensure widgets are created and sizes known
        self._resize_table() # Perform initial layout calculation based on window size
        self.show_initial_message() # Setup game logic and start the first game
        self.root.deiconify() # Show the main window (was hidden initially)

    def _configure_styles(self):
        """Configures the ttk styles used in the application."""
        # Distinct human seat frame style
        self.theme.style.configure(
            'Human.Seat.TLabelframe',
            background=self.theme.COLORS.get('bg_frame_human', self.theme.COLORS['bg_frame']), # Use specific color if defined
            bordercolor=self.theme.COLORS['accent'], # Bright accent color
            borderwidth=3, # Make it thicker
            relief=tk.GROOVE
        )
        # Label inside the human frame (optional, if different color needed)
        self.theme.style.configure(
            'Human.Seat.TLabelframe.Label',
             foreground=self.theme.COLORS.get('text_accent', self.theme.COLORS['text_primary']),
             background=self.theme.COLORS.get('bg_frame_human', self.theme.COLORS['bg_frame'])
        )


        # Style for the currently active player's seat
        self.theme.style.configure(
            'Active.Seat.TLabelframe',
            background=self.theme.COLORS.get('bg_frame_active', '#44475a'), # Different background
            bordercolor=self.theme.COLORS.get('accent_active', '#ffb86c'), # Different highlight color (e.g., orange)
            borderwidth=2,
            relief=tk.RAISED
        )
        self.theme.style.configure(
            'Active.Seat.TLabelframe.Label',
             foreground=self.theme.COLORS.get('text_active', '#f8f8f2'), # Bright text
             background=self.theme.COLORS.get('bg_frame_active', '#44475a')
        )

        # Styles for smaller text labels in info panels
        self.theme.style.configure('SmallInfoText.TLabel', font=self.theme.small_font, foreground=self.theme.COLORS['text_secondary'])
        self.theme.style.configure('SmallInfoValue.TLabel', font=self.theme.small_font, foreground=self.theme.COLORS['text_primary'])

        # Add other styles as needed (e.g., for specific buttons if not covered by theme defaults)


    def _create_ui(self):
        """Creates the main UI structure using Tkinter widgets and grid layout."""
        # Main Frame - takes up the whole window
        self.main_frame = ttk.Frame(self.root, style='TFrame', padding=0)
        # **FIX**: Use grid layout for main sections to achieve vertical centering of the table
        self.main_frame.pack(fill=tk.BOTH, expand=True) # Pack main_frame into root

        # Configure grid columns/rows for main_frame
        self.main_frame.columnconfigure(0, weight=1) # Single column, takes all width
        self.main_frame.rowconfigure(0, weight=0)    # Row 0: Header (fixed height)
        self.main_frame.rowconfigure(1, weight=1)    # Row 1: Table container (expands vertically)
        self.main_frame.rowconfigure(2, weight=0)    # Row 2: Bottom panels (fixed height)

        # --- Header ---
        self._create_header()
        # Place header in the top row (row 0)
        self.header_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=(5, 0))

        # --- Table Area Container ---
        # This frame's purpose is to fill the central expanding row (row 1)
        # Its background should match the main window background
        self.table_container = ttk.Frame(self.main_frame, style='TFrame')
        self.table_container.grid(row=1, column=0, sticky='nsew', padx=0, pady=0)
        # The actual table_frame will be placed *inside* this container

        # --- Create Table Frame and Contents (but don't place table_frame yet) ---
        self._create_table_area_contents()

        # --- Bottom Panels Container ---
        # **FIX**: Create a single container frame for all bottom panels
        self.bottom_panels_frame = ttk.Frame(self.main_frame, style='TFrame')
        self.bottom_panels_frame.grid(row=2, column=0, sticky='nsew', padx=10, pady=(0, 5))

        # Create the individual bottom panels (they will be packed inside bottom_panels_frame)
        self._create_last_round_panel()
        self._create_game_info_panel()
        self._create_action_panel()

        # **FIX**: Pack bottom panels into their container frame using pack(side=BOTTOM)
        # This ensures they stack vertically from the bottom edge upwards. Order matters.
        self.action_panel.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 2), padx=0)
        self.info_panel.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 2), padx=0)
        self.last_round_panel.pack(side=tk.BOTTOM, fill=tk.X, pady=(2, 2), padx=0)


    def show_initial_message(self):
        """Called after UI creation to start the game setup process."""
        # Currently just starts the game setup directly.
        # Could be extended to show a welcome dialog or instructions first.
        self.setup_game()

    # --- Game Setup and Control Methods ---

    def setup_game(self):
        """
        Initializes or resets the game environment, loads AI models based on config,
        and prepares the UI for a new tournament.
        """
        print("Setting up game...")
        print(f"Using Seat Config: {self.seat_config}")
        print(f"Checkpoint Paths: {self.checkpoint_paths}")

        self.opponent_models = {} # Clear previously loaded opponent models

        # --- Model Loading Logic ---
        # Only attempt loading if PyTorch is available
        if torch and self.device:
            script_dir = os.path.dirname(__file__) or "."
            checkpoint_dir = os.path.abspath(os.path.join(script_dir, '..', 'checkpoints'))
            print(f"Looking for checkpoints in: {checkpoint_dir}")

            # Define potential paths for the primary/default AI model
            default_checkpoint_path = os.path.join(checkpoint_dir, 'final_agent_model.pt')
            fallback_checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_10000.pt') # Example fallback

            primary_model_path = None
            if os.path.exists(default_checkpoint_path):
                primary_model_path = default_checkpoint_path
                print(f"Found default checkpoint: {primary_model_path}")
            elif os.path.exists(fallback_checkpoint_path):
                primary_model_path = fallback_checkpoint_path
                print(f"Default checkpoint not found. Using fallback: {primary_model_path}")
            else:
                print("Warning: No default or fallback agent model found in checkpoints directory.")

            # Load the primary agent model (used for opponents if no specific model is set)
            self.agent_model = None
            if primary_model_path:
                try:
                    self.agent_model = load_agent_model(primary_model_path, NUM_ACTIONS, self.device)
                    if self.agent_model is None:
                        messagebox.showwarning("Model Load Warning", f"Primary agent model function returned None from {primary_model_path}.")
                    else:
                        print("Primary agent model loaded successfully.")
                except Exception as e:
                     messagebox.showerror("Model Load Error", f"Error loading primary model from {primary_model_path}:\n{e}")
                     self.agent_model = None # Ensure it's None on error
            else:
                 print("No primary model path found to load.")


            # Load specific models for seats if paths are provided in self.checkpoint_paths
            for seat_id, model_path in self.checkpoint_paths.items():
                # Only load if it's a 'model' seat, has a valid path, and isn't the human player
                if seat_id != self.human_player_id and self.seat_config.get(seat_id) == 'model' and model_path and os.path.exists(model_path):
                    print(f"Loading specific model for Seat {seat_id+1} from {model_path}")
                    try:
                        specific_model = load_agent_model(model_path, NUM_ACTIONS, self.device)
                        if specific_model:
                            self.opponent_models[seat_id] = specific_model
                            print(f"Successfully loaded specific model for Seat {seat_id+1}.")
                        else:
                            messagebox.showwarning("Model Load Warning", f"Specific model function returned None for Seat {seat_id+1} from {model_path}. It will use the default model or random policy.")
                    except Exception as e:
                        messagebox.showerror("Model Load Error", f"Error loading specific model for Seat {seat_id+1} from {model_path}:\n{e}")
        else:
             print("Skipping AI model loading: PyTorch not available or no device found.")
             self.agent_model = None # Ensure models are None if torch isn't used


        # --- Environment Initialization ---
        try:
            # Close previous environment cleanly if it exists
            if hasattr(self, 'env') and self.env:
                try:
                     self.env.close()
                     print("Previous environment closed.")
                except Exception as close_err:
                     print(f"Warning: Error closing previous environment: {close_err}")

            # Create the poker environment instance
            self.env = BaseFullPokerEnv(
                num_players=NUM_PLAYERS,
                agent_id=self.human_player_id, # Inform env which seat is human-controlled
                render_mode=None, # GUI handles rendering, not the environment
                seat_config=self.seat_config, # Pass the current seat configuration
                starting_stack=STARTING_STACK # Pass starting stack constant
            )
            print("Poker Environment Initialized/Reset.")
        except Exception as e:
            messagebox.showerror("Fatal Error", f"Failed to initialize game environment: {e}\nCheck Back_End dependencies and constants.")
            import traceback
            traceback.print_exc()
            self.root.quit()
            return

        # --- Set Opponent Policies in Environment ---
        # Ensure the environment instance has the necessary method
        if not hasattr(self.env, 'set_opponent_policy'):
             messagebox.showerror("Fatal Error", "Environment object is missing the 'set_opponent_policy' method. Cannot configure opponents.")
             self.root.quit()
             return

        for seat_id, seat_type in self.seat_config.items():
            # Skip configuration for human player, empty seats, or seats marked as 'player'
            if seat_id == self.human_player_id or seat_type in ['empty', 'player']:
                continue

            # Determine which model to use: specific, default, or None
            model_to_use = self.opponent_models.get(seat_id, self.agent_model)

            # If seat is configured as 'model' but no model could be loaded, default policy to 'random'
            effective_seat_type = seat_type
            if seat_type == 'model' and model_to_use is None:
                print(f"Warning: No model available for Seat {seat_id+1} (configured as 'model'). Setting policy to 'random'.")
                effective_seat_type = 'random' # Override type for policy selection

            # Get the policy function based on the effective type and model
            try:
                policy_func = get_opponent_policy(
                    opponent_type=effective_seat_type,
                    agent_model=model_to_use, # Pass the loaded model (can be None)
                    action_list=ACTION_LIST,
                    num_actions=NUM_ACTIONS,
                    device=self.device if torch else None # Pass device only if torch exists
                )
                # Assign the policy function to the environment for this seat
                self.env.set_opponent_policy(seat_id, policy_func)
                print(f"Set Seat {seat_id+1} policy to: {effective_seat_type}")
            except Exception as e:
                 messagebox.showerror("Policy Error", f"Failed to get or set policy for Seat {seat_id+1} (Type: {effective_seat_type}):\n{e}")
                 # Attempt to set a fallback 'random' policy if possible
                 try:
                      fallback_policy = get_opponent_policy('random', None, ACTION_LIST, NUM_ACTIONS, None)
                      self.env.set_opponent_policy(seat_id, fallback_policy)
                      print(f"Error setting policy for Seat {seat_id+1}. Defaulted to 'random'.")
                 except Exception as fb_e:
                      print(f"FATAL: Could not even set fallback random policy for Seat {seat_id+1}: {fb_e}")
                      # Consider stopping the game here as configuration failed critically
                      messagebox.showerror("Fatal Error", "Failed to set even fallback policies. Exiting.")
                      self.root.quit()
                      return

        # --- Reset UI and Start Game ---
        self.update_last_round_display("New tournament started.") # Update summary label
        self._clear_table_state() # Reset visual elements (cards, highlights, etc.)

        # Start the first round after a short delay to allow UI to draw/update
        self.root.after(200, self.start_new_tournament)


    def _clear_table_state(self):
        """Resets visual elements like cards, highlights, bets to a default state."""
        print("Clearing table state visuals...")
        if not self.root.winfo_exists(): return # Don't update if window closed

        try:
            self.update_pot(0)
            self.update_community_cards([]) # Clear community cards

            for i in range(NUM_PLAYERS):
                 # Reset player status, stack display, and action text
                 # Determine initial status based on config
                 seat_type = self.seat_config.get(i, "Unknown")
                 status_text = "Empty"
                 if seat_type != 'empty':
                      status_text = "Human" if i == self.human_player_id else seat_type.capitalize()
                 self.update_player_status(i, status_text, STARTING_STACK, "Last: - | Bet: $0")

                 self.update_player_cards(i, ["??", "??"]) # Reset cards to hidden

                 # Reset seat frame style (remove active highlight, ensure human style)
                 widgets = self.seat_frames_widgets.get(i)
                 if widgets and widgets['frame'].winfo_exists():
                     style = 'Human.Seat.TLabelframe' if i == self.human_player_id else 'Seat.TLabelframe'
                     widgets['frame'].configure(style=style)

            self.highlight_active_player(None) # Ensure no player is highlighted initially

            # Reset game info panel to defaults
            self.update_game_info("Waiting...", "-", 0, 0, "-", 0)
            # Disable action buttons
            self.update_action_buttons(False)
        except tk.TclError as e:
             print(f"Warning: TclError during UI clear ({e}). Window might be closing.")
        except Exception as e:
             print(f"Error during UI clear: {e}")


    def start_new_tournament(self):
        """Starts a new tournament by resetting the environment and processing the initial state."""
        if not self.env:
            messagebox.showerror("Error", "Game environment not initialized. Cannot start tournament.")
            return

        print("\n--- Starting New Tournament ---")
        self.is_game_running = True # Set flag indicating game is active
        try:
            # Reset the environment to get the initial state (observation) and info dictionary
            # A seed could be passed here for reproducibility: self.env.reset(seed=...)
            encoded_state, info = self.env.reset()

            # Check for critical errors reported by the environment during reset
            if info.get("error"):
                messagebox.showerror("Environment Error", f"Failed to start new tournament:\n{info['error']}")
                self.is_game_running = False
                return

            self.current_game_state = encoded_state # Store the initial observation

            # Update the summary display (redundant with _clear_table_state, but safe)
            self.update_last_round_display("New tournament started.")

            # Perform a full UI update based on the initial state from the info dictionary
            self.update_ui_from_info(info) # This sets player cards, stacks, pot, etc.

            # Process the first game step based on the initial info
            # This will determine whose turn it is (human or opponent) and proceed accordingly
            self.process_game_step(info)

        except AttributeError as ae:
             # Catch errors like calling reset on None if env failed to initialize
             if "'NoneType' object has no attribute 'reset'" in str(ae):
                 messagebox.showerror("Error", "Game environment (self.env) is not properly initialized.")
             else:
                 messagebox.showerror("Error", f"An unexpected attribute error occurred starting the tournament: {ae}")
             self.is_game_running = False
             import traceback
             traceback.print_exc()
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred starting the tournament: {e}")
            self.is_game_running = False
            import traceback
            traceback.print_exc()


    def update_ui_from_info(self, info):
        """
        Updates the entire UI based on the info dictionary received from the environment step/reset.
        This is the central function for synchronizing the GUI with the game state.
        """
        if not self.root.winfo_exists(): return # Stop if window is closed
        if not info or not isinstance(info, dict):
            print("Warning: update_ui_from_info received invalid info object.")
            return

        # --- Extract data from info dictionary with safe defaults ---
        pot = info.get('pot', 0)
        community_cards = info.get('community_cards', [])
        stacks = info.get('stacks', {p: 0 for p in range(NUM_PLAYERS)})
        current_bets = info.get('current_bets', {}) # Bets in the current betting round
        active_players_in_round = info.get('active_players', []) # Still eligible to win pot
        folded_players = info.get('folded_players', []) # Folded this hand
        all_in_players = info.get('all_in_players', []) # All-in this hand
        showdown_hands = info.get('showdown_hands') # {player_id: {'hand': [], 'desc': ''}} or None
        agent_hand = info.get('agent_hand') # Try to get human hand directly from info first

        # Fallback: if info doesn't provide agent_hand, try getting from env (less reliable during transitions)
        if agent_hand is None and hasattr(self.env, 'hands') and isinstance(self.env.hands, dict):
             agent_hand = self.env.hands.get(self.human_player_id)

        is_round_over = info.get('round_over', False)
        is_terminated = info.get('terminated', False) # Tournament finished
        current_player_id = info.get('current_player_id') # Whose turn? (can be None)
        button_pos = info.get('button_pos', -1) # Dealer button seat index
        last_action_data = info.get('last_action', {}) # {player_id: action_str}

        # --- Update Central Table Elements ---
        try:
            self.update_pot(pot)
            self.update_community_cards(community_cards)
        except tk.TclError: return # Stop updates if UI is being destroyed

        # --- Update Each Player Seat ---
        for i in range(NUM_PLAYERS):
            if not self.root.winfo_exists(): return # Check again inside loop

            try:
                # Get Seat Info
                stack = stacks.get(i, 0)
                seat_type = self.seat_config.get(i, "Unknown")
                is_playing_tournament = seat_type != 'empty'
                is_active_in_hand = i in active_players_in_round
                is_folded = i in folded_players
                is_all_in = i in all_in_players
                bet_this_round = current_bets.get(i, 0)

                # Determine Player Status String
                status_parts = []
                if not is_playing_tournament:
                    status_parts.append("Empty")
                else:
                    # Base status (Human, Model, etc.)
                    base_status = "Human" if i == self.human_player_id else seat_type.capitalize()
                    status_parts.append(base_status)
                    # Add state info
                    if stack <= 0 and not is_active_in_hand: status_parts.append("(Out)") # Busted previously
                    elif is_folded: status_parts.append("(Folded)")
                    elif is_all_in: status_parts.append("(All-In)")
                    # Note: A player can be all-in AND active

                status = " ".join(status_parts)

                # Determine Last Action String
                last_action = last_action_data.get(i, "-")
                # Format action string nicely (e.g., "bet_small" -> "Bet Small")
                last_action_formatted = last_action.replace('_', ' ').title() if last_action != "-" else "-"

                action_text = f"Bet: ${bet_this_round:,.0f} | Last: {last_action_formatted}"

                # Update Status, Stack, Action Labels
                self.update_player_status(i, status, stack, action_text)

                # --- Determine Player Cards to Display ---
                player_hand_to_display = ["??", "??"] # Default hidden

                # 1. Human Player's Hand (priority if available and valid)
                if i == self.human_player_id and agent_hand:
                    if isinstance(agent_hand, list) and len(agent_hand) == 2 and all(isinstance(c, str) for c in agent_hand):
                        player_hand_to_display = agent_hand
                    # else: print(f"DEBUG: Human hand format issue from info/env: {agent_hand}")

                # 2. Showdown Hands (Only show if round is over AND player was involved)
                if is_round_over and showdown_hands and i in showdown_hands:
                     showdown_data = showdown_hands[i]
                     actual_hand = []
                     if isinstance(showdown_data, dict): actual_hand = showdown_data.get('hand', [])
                     elif isinstance(showdown_data, list): actual_hand = showdown_data

                     if isinstance(actual_hand, list) and len(actual_hand) == 2 and all(isinstance(c, str) for c in actual_hand):
                         player_hand_to_display = actual_hand # Use the showdown hand

                # 3. If not human and not showdown, ensure cards remain hidden
                elif i != self.human_player_id:
                     player_hand_to_display = ["??", "??"]

                # 4. Update the Card Images on the UI
                self.update_player_cards(i, player_hand_to_display)
                # --- End Card Display Logic ---

            except tk.TclError: return # Stop updates if UI is being destroyed
            except Exception as e:
                 print(f"Error updating seat {i}: {e}")
                 import traceback
                 traceback.print_exc() # Log error but try to continue updating other seats


        # --- Highlight Active Player ---
        active_player_name = "N/A"
        self.is_human_turn = False
        try:
            if current_player_id is not None and self.seat_config.get(current_player_id) != 'empty':
                active_player_name = f"P{current_player_id + 1}" # Default name
                if current_player_id == self.human_player_id:
                    active_player_name = "Human"
                self.highlight_active_player(current_player_id) # Apply highlight style
                self.is_human_turn = (current_player_id == self.human_player_id)
            else:
                # No active player (e.g., between rounds, game over)
                self.highlight_active_player(None) # Remove all highlights
        except tk.TclError: return # Stop if UI destroyed

        # --- Update Action Buttons and Game Info Panel ---
        legal_actions = []
        amount_to_call = 0
        player_stack = stacks.get(self.human_player_id, 0) # Get human stack for context

        # Get legal actions only if it's human's turn and env allows it
        if self.is_human_turn and self.env and hasattr(self.env, '_get_legal_actions'):
            try:
                legal_actions = self.env._get_legal_actions(self.human_player_id)

                # Calculate amount to call accurately
                max_bet = 0
                if current_bets:
                    active_bets = {p: b for p, b in current_bets.items() if p in active_players_in_round}
                    if active_bets: max_bet = max(active_bets.values())
                player_bet = current_bets.get(self.human_player_id, 0)
                required_to_call = max(0, max_bet - player_bet)
                amount_to_call = min(required_to_call, player_stack) # Capped by stack

            except Exception as e:
                print(f"Error getting legal actions or calculating call amount: {e}")
                legal_actions = []
                amount_to_call = 0

        # Update action buttons based on turn status and legal actions
        try:
            self.update_legal_actions_display(legal_actions, amount_to_call)
        except tk.TclError: return # Stop if UI destroyed

        # --- Update Game Info Panel ---
        try:
            game_stage = info.get('stage', 'PREFLOP') # Default if missing
            game_stage_str = game_stage.replace('_', ' ').title()
            dealer_str = f"Player {button_pos + 1}" if button_pos != -1 else "N/A"
            sb = getattr(self.env, 'small_blind', 0)
            bb = getattr(self.env, 'big_blind', 0)

            self.update_game_info(
                round_name=game_stage_str, dealer=dealer_str,
                small_blind=sb, big_blind=bb,
                current_player=active_player_name, to_call=amount_to_call
            )
        except tk.TclError: return # Stop if UI destroyed

        # --- Handle End-of-Round/Tournament Conditions AFTER UI is updated ---
        # Use 'after' to schedule handlers, allowing UI to refresh first
        if is_terminated:
            # Schedule tournament end handler slightly delayed
            self.root.after(250, lambda info_copy=info.copy(): self.handle_tournament_end(info_copy))
        elif is_round_over:
            # Schedule round end handler with a longer delay to see showdown cards
            self.root.after(500, lambda info_copy=info.copy(): self.handle_round_end(info_copy))


    def process_game_step(self, info):
        """
        Determines the next step in the game flow based on the current state info.
        If it's the human's turn, enables controls.
        If it's an opponent's turn, schedules their action.
        If the round/tournament ended, does nothing (handled by delayed callbacks).
        """
        if not self.is_game_running or not self.root.winfo_exists():
            # print("process_game_step skipped: game not running or window closed.")
            return

        current_player_id = info.get('current_player_id')
        is_terminated = info.get('terminated', False)
        is_round_over = info.get('round_over', False)

        # Stop processing if round/tournament ended or no valid player turn
        if is_terminated or is_round_over or current_player_id is None:
            # print(f"process_game_step: Halting turn processing (Terminated: {is_terminated}, Round Over: {is_round_over}, Current Player: {current_player_id})")
            return # End-of-round/tournament logic handled by delayed calls

        # Determine next action based on current player
        if current_player_id == self.human_player_id:
            print("Human player's turn.")
            self.is_human_turn = True
            # Action buttons are updated via update_ui_from_info -> update_legal_actions_display
        else:
            # It's an opponent's turn
            self.is_human_turn = False
            # Ensure human buttons are disabled (should be handled by update_legal_actions_display, but belt-and-suspenders)
            try:
                 self.update_action_buttons(False)
            except tk.TclError: return # Stop if UI destroyed

            opponent_type = self.seat_config.get(current_player_id, 'Unknown')
            print(f"Opponent {current_player_id + 1} ({opponent_type}) turn. Scheduling action...")

            # Schedule the opponent's move after a delay for visual pacing
            delay_ms = 750 # Adjust delay (milliseconds) as desired
            self.root.after(delay_ms, self.execute_opponent_turn)


    def execute_opponent_turn(self):
        """
        Executes a single step for the current AI/random opponent by calling env.step(-1).
        The environment uses the pre-assigned policy for the current player.
        """
        # Double-check conditions before executing turn
        if not self.is_game_running or not self.root.winfo_exists():
             # print("execute_opponent_turn skipped: Game stopped or window closed.")
             return
        if self.is_human_turn:
             print("execute_opponent_turn skipped: Became human's turn unexpectedly.")
             return
        if not self.env:
             print("execute_opponent_turn skipped: Environment not available.")
             return

        print("Executing opponent turn...")
        try:
            # Call env.step with a dummy action (-1).
            # The environment's internal logic should use the assigned policy function
            # for the current_player_id when the action is -1.
            encoded_state, reward, terminated, truncated, info = self.env.step(-1)

            self.current_game_state = encoded_state # Update state observation

            # Update UI based on the result of the opponent's action
            self.update_ui_from_info(info)

            # Recursively call process_game_step to handle the *next* player's turn
            # This allows chains of opponent actions without extra delays between them.
            # The next call will either schedule another opponent turn or activate human controls.
            self.process_game_step(info)

        except AttributeError as ae:
            # Catch errors if self.env becomes None or lacks 'step'
            messagebox.showerror("Environment Error", f"Error during opponent's turn: {ae}")
            self.is_game_running = False
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred during opponent's turn: {e}")
            import traceback
            traceback.print_exc()
            self.is_game_running = False # Stop the game on unexpected error


    def handle_player_action(self, action_name):
        """Handles the button click for a player action (Fold, Check, Call, Bet, All-in)."""
        # Validate state before processing action
        if not self.is_human_turn or not self.is_game_running or not self.root.winfo_exists():
            print("Player action ignored: Not human's turn, game not running, or window closed.")
            return
        if not self.env:
            messagebox.showerror("Error", "Game environment not initialized.")
            return

        # Get action index from action name
        action_idx = self.string_to_action.get(action_name)
        if action_idx is None:
            messagebox.showerror("Internal Error", f"Invalid action name received from button: {action_name}")
            return

        # --- Pre-Action Legality Check (Optional but Recommended) ---
        # Double-check if the action is still legal right before sending it.
        # This helps catch race conditions where the state might change slightly.
        try:
             if hasattr(self.env, '_get_legal_actions'):
                 current_legal_actions = self.env._get_legal_actions(self.human_player_id)
                 if action_name not in current_legal_actions:
                     messagebox.showwarning("Invalid Action", f"Action '{action_name}' is no longer legal. The game state may have changed.", parent=self.root)
                     # Refresh button states based on the actual current legal actions
                     try:
                         amount_to_call_str = self.tocall_value.cget("text")
                         amount_to_call = float(amount_to_call_str.replace("$","").replace(",",""))
                     except: amount_to_call = 0 # Fallback
                     self.update_legal_actions_display(current_legal_actions, amount_to_call)
                     return # Do not proceed with the illegal action
             # else: print("Warning: Environment missing '_get_legal_actions' for pre-action check.")
        except Exception as e:
             messagebox.showerror("Error", f"Could not verify legal actions before sending: {e}", parent=self.root)
             return # Don't proceed if legality check fails
        # --- End Pre-Action Check ---


        print(f"Player action chosen: {action_name} (Index: {action_idx})")
        self.is_human_turn = False # Player has acted, turn is over
        self.update_action_buttons(False) # Disable buttons immediately

        try:
            # Send the chosen action index to the environment
            encoded_state, reward, terminated, truncated, info = self.env.step(action_idx)
            self.current_game_state = encoded_state # Store new state observation

            # Update UI based on the result of the human action
            self.update_ui_from_info(info)

            # Process the next game step (likely an opponent's turn now, or round/game end)
            self.process_game_step(info)

        except AttributeError as ae:
            messagebox.showerror("Environment Error", f"Error processing action '{action_name}': {ae}")
            self.is_game_running = False # Stop game on critical env error
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred processing action '{action_name}': {e}")
            import traceback
            traceback.print_exc()
            # Consider stopping the game on error, as state might be inconsistent
            # self.is_game_running = False


    def handle_round_end(self, info):
        """
        Handles the end of a poker hand/round.
        Updates the summary display with results and schedules the next round if applicable.
        """
        if not self.root.winfo_exists(): return # Stop if window closed
        print("--- Round Ended ---")
        # UI should already reflect the final state of the round due to update_ui_from_info

        # Extract results from info dictionary
        winners = info.get('winners', []) # List of winner IDs or potentially dict {id: amount}
        showdown_hands = info.get('showdown_hands', {}) # {id: {'hand':[], 'desc':''}} or None
        final_pot = info.get('final_pot', info.get('pot', 0)) # Pot value at end
        winnings = info.get('winnings', {}) # Preferred way: {player_id: amount_won}

        # --- Construct Round Summary Text ---
        result_parts = [f"Prev. Round Pot: ${final_pot:,.0f}."] # Start with pot size

        if winnings: # Use winnings dictionary if available
             winners_summary = []
             # Sort by amount won (descending) for clarity
             sorted_winnings = sorted(winnings.items(), key=lambda item: item[1], reverse=True)

             for w_id, amount_won in sorted_winnings:
                 if amount_won > 0: # Only list players who actually won something
                     w_name = f"P{w_id+1}" if w_id != self.human_player_id else "Human"
                     summary_part = f"{w_name} wins ${amount_won:,.0f}"

                     # Try to add hand description if available from showdown hands
                     if showdown_hands and w_id in showdown_hands:
                          hand_data = showdown_hands[w_id]
                          desc = "N/A"
                          hand_list = []
                          if isinstance(hand_data, dict):
                              desc = hand_data.get('desc', 'N/A')
                              hand_list = hand_data.get('hand', [])
                          elif isinstance(hand_data, list): # Older format?
                               hand_list = hand_data

                          # Add description if valid
                          if desc and desc != 'N/A' and len(desc) > 3:
                              summary_part += f" ({desc})"
                          # Fallback: try rendering hand list if description missing but hand exists
                          elif hand_list:
                               try:
                                   # Use card_utils if available to render hand string
                                   hand_str = render_hand(hand_list) # Assumes render_hand imported
                                   summary_part += f" ({hand_str})"
                               except NameError: pass # card_utils or render_hand not available

                     winners_summary.append(summary_part)

             if winners_summary:
                  result_parts.append("Outcome: " + "; ".join(winners_summary) + ".")
             else:
                 result_parts.append("Round ended, but no winnings recorded.")

        elif winners and not winnings: # Fallback if only 'winners' list is present (less ideal)
            winner_names = [f"P{w_id+1}" if w_id != self.human_player_id else "Human" for w_id in winners]
            result_parts.append(f"Winner(s): {', '.join(winner_names)}.")
            # Try getting hand desc for first winner as fallback
            if winners and showdown_hands and winners[0] in showdown_hands:
                 hand_data = showdown_hands[winners[0]]
                 if isinstance(hand_data, dict): desc = hand_data.get('desc')
                 else: desc = None # Cannot get desc from list
                 if desc: result_parts.append(f"Hand: {desc}.")

        else:
            result_parts.append("Round ended, outcome unclear.") # e.g., error or unusual fold scenario

        summary_text = " ".join(result_parts)
        self.update_last_round_display(summary_text) # Update label

        # --- Schedule Next Round ---
        # Only start the next round if the tournament hasn't terminated overall
        if self.is_game_running and not info.get('terminated', False):
            print("Scheduling next round...")
            # Delay before starting the next hand (e.g., 3 seconds) to allow players to read results
            self.root.after(3000, self.start_next_round_in_tournament)
        elif not self.is_game_running:
             print("Round ended, but game is stopped. Not scheduling next round.")
        else: # Tournament terminated
            print("Round ended, and tournament also terminated.")
            # Tournament end message is handled by handle_tournament_end


    def start_next_round_in_tournament(self):
        """
        Initiates the next hand/round within the current tournament.
        Calls env.step(-1) assuming the environment handles new hand setup internally.
        """
        # Check conditions before proceeding
        if not self.is_game_running or not self.root.winfo_exists():
            print("Cannot start next round: game not running or window closed.")
            return
        if not self.env:
             print("Cannot start next round: environment not available.")
             return

        print("Starting next round...")
        try:
            # Call step with -1; environment should handle dealing new cards, moving blinds, etc.
            # This relies on the environment's step function correctly interpreting -1 after a round end.
            encoded_state, reward, terminated, truncated, info = self.env.step(-1)
            self.current_game_state = encoded_state

            # Update UI with the state for the start of the new round
            self.update_ui_from_info(info)

            # Process the first turn of the new round
            self.process_game_step(info)

        except AttributeError as ae:
            messagebox.showerror("Environment Error", f"Error starting next round: {ae}")
            self.is_game_running = False
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error starting next round: {e}")
            import traceback
            traceback.print_exc()
            self.is_game_running = False # Stop game on error


    def handle_tournament_end(self, info):
        """
        Handles the end of the entire tournament.
        Displays final results and prompts the user to play again or quit.
        """
        if not self.root.winfo_exists(): return # Stop if window closed
        print("===== Tournament Ended =====")
        self.is_game_running = False # Stop game loop processing
        self.is_human_turn = False
        try:
            self.update_action_buttons(False) # Disable all action buttons
        except tk.TclError: pass # Ignore if UI closing

        # UI should already reflect the final state from the last update_ui_from_info call

        stacks = info.get('stacks', {})
        # Determine winner(s) - usually the last player(s) with chips
        active_players = [p for p, s in stacks.items() if s > 0 and self.seat_config.get(p) != 'empty']

        winner_text = "Tournament Over!\n\n" # Start of message dialog text

        if len(active_players) == 1:
            # Single winner scenario
            winner_id = active_players[0]
            winner_name = "Human" if winner_id == self.human_player_id else f"Player {winner_id + 1}"
            final_stack = stacks.get(winner_id, 0)
            winner_text += f"{winner_name} wins the tournament with ${final_stack:,.0f}!"
        elif len(active_players) > 1:
            # Multiple players left (e.g., timed tournament, error state?)
            winner_text += "Tournament ended.\nFinal Stacks:"
            # Sort by stack descending for ranking
            sorted_stacks = sorted(stacks.items(), key=lambda item: item[1], reverse=True)
            for p_id, stack in sorted_stacks:
                if self.seat_config.get(p_id) != 'empty': # Only show non-empty seats
                    player_name = "Human" if p_id == self.human_player_id else f"Player {p_id + 1}"
                    winner_text += f"\n  {player_name}: ${stack:,.0f}"
        else:
            # No players left with chips? Should not happen in standard poker.
            winner_text += "Tournament ended unexpectedly with no players having chips."

        # Display the result in a dialog box using the internal helper method
        # This is called after a delay from update_ui_from_info, so show immediately
        self._show_tournament_end_dialog(winner_text)


    def _show_tournament_end_dialog(self, winner_text):
        """Shows a modal dialog box with tournament results and asks to play again."""
        # Ensure the dialog is modal to the root window using 'parent'
        play_again = messagebox.askyesno("Tournament Over",
                                         f"{winner_text}\n\nStart a new tournament?",
                                         parent=self.root)
        if play_again:
            # User wants to play again, re-run the setup process
            self.setup_game()
        else:
            # User chose not to play again, close the application
            self.root.quit()


    # --- UI Creation Methods ---

    def _create_header(self):
        """Creates the header section with title and control buttons."""
        # Use TFrame for consistency with theme
        self.header_frame = ttk.Frame(self.main_frame, style='Header.TFrame', padding=(10, 5))
        self.header_frame.columnconfigure(0, weight=1) # Allow title label to potentially expand

        # Title Label on the left
        ttk.Label(self.header_frame, text="Enhanced Poker", style='Title.TLabel').grid(row=0, column=0, sticky='w')

        # Button container frame, aligned right using grid
        button_container = ttk.Frame(self.header_frame, style='Header.TFrame')
        button_container.grid(row=0, column=1, sticky='e') # Place in column 1, align East

        # Buttons packed horizontally within the button container
        ttk.Button(button_container, text="Configure Seats", command=self._show_seat_config_dialog, style='Header.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_container, text="New Tournament", command=self.setup_game, style='Header.TButton').pack(side=tk.LEFT, padx=5)


    def _create_table_area_contents(self):
        """Creates the widgets that go *inside* the table_container, including the table_frame itself."""
        # Table frame (the visible oval/rectangle) - placed inside table_container
        # This frame uses the 'Table.TFrame' style defined in poker_theme.py
        self.table_frame = ttk.Frame(self.table_container, style='Table.TFrame')
        # Initial placement; size/position will be updated by _resize_table
        self.table_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=800, height=500)

        # Pot label - placed relative to table_frame
        self.pot_label = ttk.Label(self.table_frame, text="Pot: $0", style='Pot.TLabel')
        self.pot_label.place(relx=0.5, rely=0.25, anchor=tk.CENTER) # Positioned near top-center of table

        # Player Seats - Create the seat widgets
        self._create_player_seats()
        # Note: Seats are *not* placed here; _reposition_seats handles their placement within table_frame

        # Community Cards - Create the container and labels
        self._create_community_cards()
        # Note: Community card container is placed relative to table_frame


    def _create_player_seats(self):
        """Creates the LabelFrames and internal widgets for each player seat."""
        # **FIX**: Adjusted relative positions for more buffer from table edges
        # These define the anchor points for each seat relative to table_frame dimensions
        self.seat_positions = [
            # Seat 0 (Bottom Left - Default Human)
            {'relx': 0.08, 'rely': 0.88, 'anchor': tk.SW }, # Anchor South-West
            # Seat 1 (Middle Left)
            {'relx': 0.0, 'rely': 0.50, 'anchor': tk.W },  # Anchor West
            # Seat 2 (Top Left)
            {'relx': 0.08, 'rely': 0.12, 'anchor': tk.NW }, # Anchor North-West
            # Seat 3 (Top Right)
            {'relx': 0.92, 'rely': 0.12, 'anchor': tk.NE }, # Anchor North-East
            # Seat 4 (Middle Right)
            {'relx': 1.0, 'rely': 0.50, 'anchor': tk.E },  # Anchor East
            # Seat 5 (Bottom Right)
            {'relx': 0.92, 'rely': 0.88, 'anchor': tk.SE }  # Anchor South-East
        ][:NUM_PLAYERS] # Ensure we only take positions needed for the number of players

        self.seat_frames_widgets = {} # Clear any previous widgets before creating new ones
        for i in range(NUM_PLAYERS):
            # Determine the correct style for the seat frame
            seat_style = 'Human.Seat.TLabelframe' if i == self.human_player_id else 'Seat.TLabelframe'

            # Create the main LabelFrame for the seat
            seat = ttk.LabelFrame(
                self.table_frame, # Parent is the table_frame
                text=f"Seat {i+1}", # Label text for the frame
                style=seat_style,
                padding=(5, 5) # Internal padding around contents
            )
            # Prevent the frame from shrinking/growing based on its content size
            # Its size will be explicitly set by 'place' in _reposition_seats
            seat.pack_propagate(False)

            # --- Internal structure using grid for better layout control ---
            seat.columnconfigure(0, weight=3, minsize=60) # Column for text info (more weight)
            seat.columnconfigure(1, weight=2, minsize=50) # Column for cards (less weight)
            seat.rowconfigure(0, weight=1) # Single row spanning vertically

            # Frame for text info (Status, Stack, Action)
            text_f = ttk.Frame(seat, style='TFrame', padding=(2, 0))
            text_f.grid(row=0, column=0, sticky='nsew', padx=(3, 2), pady=2)

            # Configure rows within text_f to space out labels vertically
            text_f.rowconfigure(0, weight=1) # Status row
            text_f.rowconfigure(1, weight=1) # Stack row
            text_f.rowconfigure(2, weight=1) # Action row
            text_f.columnconfigure(0, weight=1) # Single column for text labels

            # Create text labels
            status = ttk.Label(text_f, text="Status: -", anchor=tk.W, style='SeatText.TLabel')
            status.grid(row=0, column=0, sticky='ew', pady=1)
            stack  = ttk.Label(text_f, text="Stack: $0", anchor=tk.W, style='SeatText.TLabel')
            stack.grid(row=1, column=0, sticky='ew', pady=1)
            action = ttk.Label(text_f, text="Last: - | Bet: $0", anchor=tk.W, style='SeatText.TLabel')
            action.grid(row=2, column=0, sticky='ew', pady=1)

            # Frame for card images
            # **FIX**: Increased horizontal padding around the card frame
            cards_f = ttk.Frame(seat, style='TFrame')
            cards_f.grid(row=0, column=1, sticky='nsew', padx=(5, 10), pady=5) # More padding left/right

            # Center cards vertically and horizontally within cards_f using grid
            cards_f.columnconfigure(0, weight=1) # Card 1 column
            cards_f.columnconfigure(1, weight=1) # Card 2 column
            cards_f.rowconfigure(0, weight=1) # Single row, centers vertically

            # Initial card size calculation (small placeholder for initialization)
            init_w = max(20, int(self.card_generator.base_card_width * 0.40))
            init_h = max(30, int(self.card_generator.base_card_height * 0.40))
            placeholder = self.card_generator.get_placeholder_image((init_w, init_h))

            # Create Card Labels
            c1 = ttk.Label(cards_f, image=placeholder)
            c1.image = placeholder # Keep reference to prevent garbage collection
            # **FIX**: Added padding between cards using grid padx
            c1.grid(row=0, column=0, sticky='nse', padx=(0, 2)) # Align East, pad right

            c2 = ttk.Label(cards_f, image=placeholder)
            c2.image = placeholder # Keep reference
            # **FIX**: Added padding between cards using grid padx
            c2.grid(row=0, column=1, sticky='nsw', padx=(2, 0)) # Align West, pad left

            # Store references to all widgets for this seat
            self.seat_frames_widgets[i] = {
                'frame': seat,
                'status': status,
                'stack': stack,
                'action': action,
                'cards': [c1, c2], # List of card labels
                'card_size': (init_w, init_h) # Store initial/current card size
            }


    def _create_community_cards(self):
        """Creates the container and labels for the community cards."""
        # Container for community cards - placed relative to table_frame center
        self.community_container = ttk.Frame(self.table_frame, style='TFrame')
        self.community_container.place(relx=0.5, rely=0.55, anchor=tk.CENTER) # Position below pot

        self.community_card_labels = [] # Clear previous labels if any
        # Initial size calculation (will be updated by resize logic)
        init_w = int(self.card_generator.base_card_width * 0.6)
        init_h = int(self.card_generator.base_card_height * 0.6)
        self.community_card_size = (max(30, init_w), max(45, init_h)) # Store initial size
        placeholder = self.card_generator.get_placeholder_image(self.community_card_size)

        # Create 5 labels for the community cards (Flop, Turn, River)
        for i in range(5):
            lbl = ttk.Label(self.community_container, image=placeholder)
            lbl.image = placeholder # Keep reference
            # Pack horizontally within their container, add padding between cards
            lbl.pack(side=tk.LEFT, padx=4)
            self.community_card_labels.append(lbl)


    def _create_action_panel(self):
        """Creates the panel containing player action buttons."""
        # Panel for action buttons - Gets packed into bottom_panels_frame
        self.action_panel = ttk.LabelFrame(self.bottom_panels_frame, text="Player Actions", style='TLabelframe')

        # Frame to hold the buttons themselves, allows centering within the panel
        self.action_buttons_frame = ttk.Frame(self.action_panel, style='TFrame')
        self.action_buttons_frame.pack(pady=5) # Use pack's default centering

        # Define button styles (these should be configured in PokerTheme)
        styles = {
            'fold':'Fold.TButton', 'check':'Check.TButton', 'call':'Call.TButton',
            'bet_small':'Bet.TButton', 'bet_big':'Bet.TButton', 'all_in':'AllIn.TButton'
        }
        self.action_buttons = {} # Clear previous button references

        # Define the order and text for action buttons
        action_order = ['fold', 'check', 'call', 'bet_small', 'bet_big', 'all_in']
        action_texts = { # More descriptive text
             'fold': 'Fold', 'check': 'Check', 'call': 'Call',
             'bet_small': 'Bet (1/2 Pot)', # Example bet sizing text
             'bet_big': 'Bet (Pot)',     # Example bet sizing text
             'all_in': 'All-In'
        }

        # Create buttons
        for action_name in action_order:
            btn_text = action_texts.get(action_name, action_name.title()) # Get text, fallback to title case
            style_key = styles.get(action_name, 'TButton') # Get style, fallback to default

            b = ttk.Button(
                self.action_buttons_frame,
                text=btn_text,
                style=style_key,
                # Lambda function captures the action_name for the command
                command=lambda act=action_name: self.handle_player_action(act),
                state=tk.DISABLED # Start all buttons disabled
            )
            # Pack buttons horizontally with padding
            b.pack(side=tk.LEFT, padx=5, pady=5, ipady=2) # Add internal vertical padding
            self.action_buttons[action_name] = b # Store button reference


    def _create_game_info_panel(self):
        """Creates the panel displaying general game information (round, blinds, etc.)."""
        # Panel for game info - Gets packed into bottom_panels_frame
        self.info_panel = ttk.LabelFrame(self.bottom_panels_frame, text="Game Information", style='TLabelframe')

        # Use grid inside the panel for neat alignment of labels
        grid = ttk.Frame(self.info_panel, style='TFrame')
        grid.pack(fill=tk.X, padx=10, pady=(2, 5)) # Padding around the grid

        # Configure columns to distribute space somewhat evenly
        num_info_cols = 6 # 3 pairs of Label:Value
        for col in range(num_info_cols):
            grid.columnconfigure(col, weight=1, minsize=80) # Give weight and minimum size

        # Define styles for info labels for consistency
        ts = 'SmallInfoText.TLabel' # Style for text labels (e.g., "Round:")
        vs = 'SmallInfoValue.TLabel' # Style for value labels (e.g., "Flop")

        # --- Row 0: Round, Blinds, Current Player ---
        ttk.Label(grid, text="Round:", style=ts).grid(row=0, column=0, sticky=tk.W, padx=5)
        self.round_value = ttk.Label(grid, text="-", style=vs, anchor=tk.W)
        self.round_value.grid(row=0, column=1, sticky=tk.EW, padx=5) # EW sticky to fill column

        ttk.Label(grid, text="SB/BB:", style=ts).grid(row=0, column=2, sticky=tk.W, padx=5)
        self.blinds_value = ttk.Label(grid, text="$0 / $0", style=vs, anchor=tk.W)
        self.blinds_value.grid(row=0, column=3, sticky=tk.EW, padx=5)

        ttk.Label(grid, text="Player Turn:", style=ts).grid(row=0, column=4, sticky=tk.W, padx=5)
        self.current_value = ttk.Label(grid, text="-", style=vs, anchor=tk.W)
        self.current_value.grid(row=0, column=5, sticky=tk.EW, padx=5)

        # --- Row 1: Dealer, To Call ---
        ttk.Label(grid, text="Dealer:", style=ts).grid(row=1, column=0, sticky=tk.W, padx=5)
        self.dealer_value = ttk.Label(grid, text="-", style=vs, anchor=tk.W)
        self.dealer_value.grid(row=1, column=1, sticky=tk.EW, padx=5)

        # Columns 2 and 3 are empty in this row for spacing

        ttk.Label(grid, text="To Call:", style=ts).grid(row=1, column=4, sticky=tk.W, padx=5)
        self.tocall_value = ttk.Label(grid, text="$0", style=vs, anchor=tk.W)
        self.tocall_value.grid(row=1, column=5, sticky=tk.EW, padx=5)


    def _create_last_round_panel(self):
        """Creates the panel displaying the summary of the previous hand."""
        # Panel for last round summary - Gets packed into bottom_panels_frame
        self.last_round_panel = ttk.LabelFrame(self.bottom_panels_frame, text="Last Round Summary", style='TLabelframe')

        # Label for the summary text itself
        self.last_round_summary_label = ttk.Label(
            self.last_round_panel,
            text=self.last_round_summary, # Initial text
            wraplength=300, # Initial wrap length, updated dynamically on resize
            anchor=tk.W, justify=tk.LEFT, # Align text left
            padding=(5, 3), # Internal padding inside the label
            style='SmallInfoText.TLabel' # Use smaller font style
        )
        # Pack to fill horizontally, allowing text wrapping
        self.last_round_summary_label.pack(fill=tk.X, expand=True, pady=1, padx=5)


    def update_last_round_display(self, summary_text=None):
        """Updates the text and wraplength of the last round summary label."""
        # Update internal state if new text is provided
        if summary_text is not None:
             self.last_round_summary = summary_text

        # Check if the label widget exists before trying to configure it
        if hasattr(self, 'last_round_summary_label') and self.last_round_summary_label and self.last_round_summary_label.winfo_exists():
            try:
                # Configure the text content
                self.last_round_summary_label.configure(text=self.last_round_summary)

                # Update wraplength based on the panel's current width for proper wrapping
                panel_width = self.last_round_panel.winfo_width()
                # Set wraplength slightly less than panel width to avoid edge cases/padding issues
                wrap_length = max(100, panel_width - 20) # Ensure a minimum wrap length
                self.last_round_summary_label.configure(wraplength=wrap_length)
            except tk.TclError:
                 # Handle cases where widget might be destroyed during update
                 # print("Warning: TclError updating last round display.")
                 pass


    # --- Resizing Logic ---

    def _on_window_resize(self, event):
        """Callback function triggered when the main window is resized."""
        # Debounce resize events: If multiple resize events occur quickly,
        # cancel any pending resize task and schedule a new one after a short delay.
        # Only trigger the actual resize logic for events on the root window itself.
        if event.widget == self.root:
            if self._resize_job:
                self.root.after_cancel(self._resize_job)
            # Schedule the _resize_table method to run after 150ms of inactivity
            self._resize_job = self.root.after(150, self._resize_table)

    def _resize_table(self):
        """
        Recalculates positions and sizes of table elements (table frame, seats, cards)
        based on the current size of the table_container frame.
        This is the core layout adjustment logic called after a window resize.
        """
        # Clear the pending resize job ID as this function is now running
        self._resize_job = None

        # Ensure the container widget exists and has valid dimensions before proceeding
        if not hasattr(self, 'table_container') or not self.table_container.winfo_exists():
            # print("Resize skipped: table_container not ready.")
            return

        try:
            container_width = self.table_container.winfo_width()
            container_height = self.table_container.winfo_height()
        except tk.TclError:
            # print("Resize skipped: Failed to get table_container dimensions (TclError).")
            return # Error getting dimensions (e.g., during shutdown)

        # Avoid calculations if container size is unrealistically small (can happen during init)
        if container_width < 100 or container_height < 100:
            # print(f"Resize deferred: container size invalid ({container_width}x{container_height}). Retrying...")
            # Schedule another attempt slightly later if size is still invalid
            self._resize_job = self.root.after(200, self._resize_table)
            return

        # print(f"Resizing table area. Container: {container_width}x{container_height}") # Debug

        # --- Calculate Table Frame Dimensions ---
        # Aim for a target aspect ratio for the table surface itself (e.g., 1.6 : 1)
        target_aspect_ratio = 1.6 # Width to Height ratio
        padding_x = 0 # Horizontal padding inside the container around the table frame
        padding_y = 0     # Vertical padding inside the container around the table frame

        # Calculate available space within the container, minus padding
        available_width = container_width - 2 * padding_x
        available_height = container_height - 2 * padding_y

        if available_width <= 0 or available_height <= 0:
             # print("Resize skipped: No available space after padding.")
             return # No space to draw the table

        # Determine table size based on available space and aspect ratio
        # Start by assuming width is the limiting factor
        table_width = available_width
        table_height = table_width / target_aspect_ratio

        # If calculated height exceeds available height, then height is the limiting factor
        if table_height > available_height:
            table_height = available_height
            table_width = table_height * target_aspect_ratio

        # Apply minimum size constraints for the table frame itself
        min_table_width = 500 # Minimum reasonable width for the table surface
        min_table_height = min_table_width / target_aspect_ratio
        table_width = max(table_width, min_table_width)
        table_height = max(table_height, min_table_height)

        # --- Apply Resized Dimensions and Reposition Internal Elements ---
        if hasattr(self, 'table_frame') and self.table_frame.winfo_exists():
            try:
                 # Update the size of the table_frame using place_configure.
                 # Centering is handled by place(relx=0.5, rely=0.5, anchor=center).
                 self.table_frame.place_configure(width=int(table_width), height=int(table_height))
                 # print(f"Table frame resized to: {int(table_width)}x{int(table_height)}") # Debug

                 # --- Reposition elements *within* the resized table_frame ---
                 # Pot label (relative placement should adapt, but explicit helps ensure it)
                 if hasattr(self, 'pot_label') and self.pot_label.winfo_exists():
                     self.pot_label.place_configure(relx=0.5, rely=0.25, anchor=tk.CENTER)

                 # Community card container (relative placement should adapt)
                 if hasattr(self, 'community_container') and self.community_container.winfo_exists():
                     self.community_container.place_configure(relx=0.5, rely=0.55, anchor=tk.CENTER)
                     # Resize the community cards themselves based on new table width
                     self._resize_community_cards(table_width)

                 # Reposition and resize player seats based on the new table dimensions
                 self._reposition_seats(table_width, table_height)

            except tk.TclError as e:
                 # Catch errors if widgets are destroyed during the update process
                 print(f"Warning: TclError during table element resize/reposition: {e}")

        # Update wraplength for the last round summary label based on its panel's current width
        self.update_last_round_display()


    def _reposition_seats(self, table_width, table_height):
        """
        Recalculates positions and sizes of player seats based on the current table_frame size.
        Also resizes the card images within each seat.
        """
        if not hasattr(self, 'seat_frames_widgets') or not self.seat_frames_widgets:
            return # Seats not created yet

        # --- Calculate Scaling Factor ---
        # Determine how much to scale seats based on table width relative to a base design width
        base_table_width_for_scaling = 800 # The table width at which base_seat_width/height look good
        scale_factor = table_width / base_table_width_for_scaling
        # Clamp scale factor: Allow shrinking, but limit growth (e.g., max 100% of base size)
        # **FIX**: Clamp ensures seats don't get *larger* than base size * scale_factor=1.0
        scale_factor = max(0.65, min(scale_factor, 1.0)) # Allow shrinking down to 65%, max is 100%

        # --- Calculate New Seat Dimensions ---
        # Apply scale factor to base dimensions defined in __init__
        seat_width = int(self.base_seat_width * scale_factor)
        seat_height = int(self.base_seat_height * scale_factor)

        # Apply absolute minimum dimensions for seats to prevent them becoming unusable
        min_seat_width = 140
        min_seat_height = 100
        seat_width = max(seat_width, min_seat_width)
        seat_height = max(seat_height, min_seat_height)
        # No explicit maximum needed due to scale_factor clamp above

        # --- Calculate New Card Dimensions within Seats ---
        # Scale card size based on the same scale factor, using base card dimensions
        card_w = self.card_generator.base_card_width
        card_h = self.card_generator.base_card_height
        card_scale_multiplier = 0.40 # How much of the base card size to use inside the seat
        scaled_card_w = int(card_w * card_scale_multiplier * scale_factor)
        scaled_card_h = int(card_h * card_scale_multiplier * scale_factor)
        # Apply minimum dimensions for cards within seats
        min_card_w = 25
        min_card_h = 35
        scaled_card_w = max(min_card_w, scaled_card_w)
        scaled_card_h = max(min_card_h, scaled_card_h)
        new_card_size = (scaled_card_w, scaled_card_h) # Store the target size for cards in seats

        # --- Reposition and Resize Each Seat Frame ---
        # Iterate through the defined seat positions
        for i, pos_info in enumerate(self.seat_positions):
            if i not in self.seat_frames_widgets: continue # Skip if seat data doesn't exist

            seat_widgets = self.seat_frames_widgets[i]
            seat_frame = seat_widgets['frame']

            if not seat_frame.winfo_exists(): continue # Skip if frame was destroyed

            # Calculate absolute (x, y) coordinates based on relative position and table dimensions
            abs_x = pos_info['relx'] * table_width
            abs_y = pos_info['rely'] * table_height

            # Place the seat frame using calculated position, size, and anchor
            try:
                 seat_frame.place_configure(x=int(abs_x), y=int(abs_y),
                                            width=int(seat_width), height=int(seat_height),
                                            anchor=pos_info['anchor'])
            except tk.TclError as e:
                 print(f"Warning: TclError placing seat {i}: {e}")
                 continue # Skip updating cards if place failed

            # --- Update Card Images if Size Changed ---
            # Check if the calculated card size is different from the currently stored size
            if seat_widgets.get('card_size') != new_card_size:
                seat_widgets['card_size'] = new_card_size # Store the new target size

                # Attempt to retrieve the current card codes being displayed (best effort)
                # This avoids needing to store card codes separately in the UI state
                current_codes = ["??", "??"] # Default if lookup fails
                try:
                    # Access the cached image object associated with the label
                    img1 = seat_widgets['cards'][0].image
                    img2 = seat_widgets['cards'][1].image
                    # Search the generator's cache for these image objects to find their codes
                    # Iterate over a copy of items to avoid issues if cache modified during iteration
                    for (code, size), img_cached in list(self.card_generator.card_images.items()):
                         # Use 'is' for object identity comparison
                         if img_cached is img1: current_codes[0] = code
                         if img_cached is img2: current_codes[1] = code
                         # Optimization: break if both found? (might compare same image twice)
                except Exception as e:
                     # This lookup can fail if images were never set or cache cleared etc.
                     # print(f"DEBUG: Could not retrieve current card codes for seat {i} during resize: {e}")
                     pass # Fallback to "??" is acceptable

                # Call update_player_cards to regenerate/fetch images at the new size
                self.update_player_cards(i, current_codes)


    def _resize_community_cards(self, table_width):
        """ Resizes community card images based on the current table width. """
        if not hasattr(self, 'community_card_labels') or not self.community_card_labels:
            return # Community cards not created yet

        # --- Calculate Scaling Factor ---
        # Scale based on table width relative to a base design width
        base_table_width_for_scaling = 800
        scale_factor = table_width / base_table_width_for_scaling
        # Clamp scale factor (allow shrinking, limit growth slightly)
        scale_factor = max(0.7, min(scale_factor, 1.1)) # Example range: 70% to 110%

        # --- Calculate New Card Dimensions ---
        card_w = self.card_generator.base_card_width
        card_h = self.card_generator.base_card_height
        comm_card_multiplier = 0.5 # Base size multiplier for community cards
        scaled_card_w = int(card_w * comm_card_multiplier * scale_factor)
        scaled_card_h = int(card_h * comm_card_multiplier * scale_factor)

        # Apply minimum dimensions for community cards
        min_comm_card_w = 30
        min_comm_card_h = 45
        new_size = (max(min_comm_card_w, scaled_card_w), max(min_comm_card_h, scaled_card_h))

        # --- Update Images Only if Size Changed ---
        if self.community_card_size != new_size:
            self.community_card_size = new_size # Store the new size
            # print(f"DEBUG Resize Community Cards to: {new_size}")

            # Attempt to retrieve current codes (similar fragile method as seats)
            current_codes = ["??"] * 5 # Default if lookup fails
            try:
                for idx, label in enumerate(self.community_card_labels):
                    if label.winfo_exists():
                        img_current = label.image
                        # Iterate over cache copy
                        for (code, size), img_cached in list(self.card_generator.card_images.items()):
                            if img_cached is img_current:
                                current_codes[idx] = code
                                break # Found code for this label
            except Exception as e:
                # print(f"DEBUG: Could not retrieve current community card codes during resize: {e}")
                pass

            # Call the update function with the retrieved codes (or defaults)
            # This forces regeneration/fetching images at the new 'self.community_card_size'
            self.update_community_cards(current_codes)


    # --- UI Update Methods (Called by game logic) ---

    def update_pot(self, amount):
        """Updates the pot label text with comma formatting."""
        if hasattr(self, 'pot_label') and self.pot_label and self.pot_label.winfo_exists():
             try:
                  # Format amount with commas for thousands separator
                  self.pot_label.configure(text=f"Pot: ${amount:,.0f}")
             except tk.TclError: pass # Ignore errors if widget destroyed during update


    def update_player_cards(self, seat_index, cards):
        """
        Updates the card images displayed for a specific player seat.
        Uses the 'card_size' currently stored for that seat.
        """
        # Validate seat index
        if seat_index not in self.seat_frames_widgets:
            # print(f"Warning: Attempted to update cards for non-existent seat {seat_index}")
            return

        seat_widgets = self.seat_frames_widgets[seat_index]
        card_labels = seat_widgets.get('cards')
        # Get the *current* required card size for this seat from storage
        card_size = seat_widgets.get('card_size', (20, 30)) # Use stored size, with fallback

        # Ensure card_labels list exists and has two labels
        if not card_labels or len(card_labels) != 2:
             print(f"Error: Invalid card labels structure for seat {seat_index}")
             return

        # Ensure 'cards' input is a list of exactly 2 card codes (use "??" for missing/invalid)
        if isinstance(cards, list) and len(cards) == 2:
            # Ensure elements are strings, replace None or empty strings with "??"
            display_codes = [str(c) if c else "??" for c in cards]
        else:
             # If input is invalid (None, wrong length, etc.), default to two hidden cards
             display_codes = ["??", "??"]
             # print(f"Warning: Invalid 'cards' data for seat {seat_index}: {cards}. Displaying hidden.")

        # Update the two card labels for the seat
        for i, label in enumerate(card_labels):
            if not label or not label.winfo_exists(): continue # Skip if label widget is gone

            card_code = display_codes[i]
            # print(f"DEBUG: Seat {seat_index+1}, Card {i+1}: Code='{card_code}', Size={card_size}") # Reduce noise

            try:
                # Get the card image (cached or newly generated) at the required size
                card_img = self.card_generator.get_card_image(card_code, size=card_size)

                if card_img:
                    label.configure(image=card_img)
                    label.image = card_img # IMPORTANT: Keep reference to prevent garbage collection
                else:
                    # This *shouldn't* happen if get_card_image always returns a placeholder
                    print(f"ERROR: get_card_image returned None for seat {seat_index}, card {i}. Attempting placeholder.")
                    placeholder = self.card_generator.get_placeholder_image(size=card_size)
                    if placeholder:
                         label.configure(image=placeholder, text="")
                         label.image = placeholder
                    else: # Absolute fallback if even placeholder fails
                         label.configure(image='', text="IMG ERR")
                         label.image = None

            except tk.TclError:
                 # Handle cases where the label widget might be destroyed between checks
                 # print(f"Warning: TclError updating card image for seat {seat_index}, card {i}.")
                 pass
            except Exception as e:
                print(f"ERROR updating card image for seat {seat_index}, card {i} (code: {card_code}, size: {card_size}): {e}")
                import traceback
                traceback.print_exc()
                # Attempt to show placeholder on unexpected error
                try:
                    placeholder = self.card_generator.get_placeholder_image(size=card_size)
                    if placeholder:
                         label.configure(image=placeholder, text="")
                         label.image = placeholder
                    else:
                         label.configure(image='', text="ERR")
                         label.image = None
                except Exception as e2:
                    print(f"ERROR: Failed to get/set placeholder after card update error: {e2}")
                    label.configure(image='', text="ERR") # Final fallback display
                    label.image = None


    def update_community_cards(self, cards):
        """
        Updates the community card images displayed on the table.
        Uses the 'community_card_size' currently stored in the instance.
        """
        # Use the currently calculated community card size
        card_size = self.community_card_size
        # print(f"DEBUG: update_community_cards called with cards: {cards}, Size: {card_size}") # Reduce noise

        # Ensure 'cards' is a list, pad with None up to 5 elements if needed
        if not isinstance(cards, list): cards = []
        # Pad with None to ensure 5 elements, then take the first 5
        display_codes_padded = (cards + [None] * 5)[:5]

        # Iterate through the community card labels
        for i, label in enumerate(self.community_card_labels):
            if not label or not label.winfo_exists(): continue # Skip if label is gone

            # Determine card code: use "??" for None or invalid entries in the padded list
            card_code = str(display_codes_padded[i]) if display_codes_padded[i] else "??"

            try:
                # Get the card image at the correct size
                card_img = self.card_generator.get_card_image(card_code, size=card_size)

                if card_img:
                    label.configure(image=card_img)
                    label.image = card_img # Keep reference
                else:
                    # Should not happen if get_card_image has fallbacks
                    print(f"ERROR: get_card_image returned None for community card {i}. Attempting placeholder.")
                    placeholder = self.card_generator.get_placeholder_image(size=card_size)
                    if placeholder:
                         label.configure(image=placeholder, text="")
                         label.image = placeholder
                    else: # Absolute fallback
                         label.configure(image='', text="IMG ERR")
                         label.image = None

            except tk.TclError:
                 # print(f"Warning: TclError updating community card image {i}.")
                 pass
            except Exception as e:
                print(f"ERROR updating community card image {i} (code: {card_code}, size: {card_size}): {e}")
                import traceback
                traceback.print_exc()
                # Attempt to show placeholder
                try:
                    placeholder = self.card_generator.get_placeholder_image(size=card_size)
                    if placeholder:
                         label.configure(image=placeholder, text="")
                         label.image = placeholder
                    else:
                         label.configure(image='', text="ERR")
                         label.image = None
                except Exception as e2:
                    print(f"ERROR: Failed to get/set placeholder after comm card update error: {e2}")
                    label.configure(image='', text="ERR")
                    label.image = None


    def update_player_status(self, seat_index, status, stack, action_text):
        """Updates the status, stack, and action labels for a player seat."""
        if seat_index not in self.seat_frames_widgets: return

        widgets = self.seat_frames_widgets[seat_index]
        # Check if widgets dictionary and the frame widget itself exist
        if widgets and widgets['frame'].winfo_exists():
             try:
                 # Update status label if it exists
                 if widgets.get('status') and widgets['status'].winfo_exists():
                     widgets['status'].configure(text=f"{status}")
                 # Update stack label if it exists, with comma formatting
                 if widgets.get('stack') and widgets['stack'].winfo_exists():
                     widgets['stack'].configure(text=f"${stack:,.0f}")
                 # Update action label if it exists
                 if widgets.get('action') and widgets['action'].winfo_exists():
                     widgets['action'].configure(text=f"{action_text}")
             except tk.TclError:
                 # Handle cases where a label might be destroyed during update
                 # print(f"Warning: TclError updating status for seat {seat_index}.")
                 pass


    def highlight_active_player(self, active_seat_index):
        """Changes the style of the active player's seat frame and resets others."""
        for i, widgets in self.seat_frames_widgets.items():
            # Ensure the seat frame widget exists
            if widgets and widgets.get('frame') and widgets['frame'].winfo_exists():
                try:
                    is_human = (i == self.human_player_id)
                    is_active = (i == active_seat_index)

                    # Determine the correct style based on active status and human status
                    if is_active:
                        # Use 'Active' style for the player whose turn it is
                        style_to_use = 'Active.Seat.TLabelframe'
                    elif is_human:
                        # Use the specific 'Human' style if it's the human player but not their turn
                        style_to_use = 'Human.Seat.TLabelframe'
                    else:
                        # Use the default 'Seat' style for inactive opponents
                        style_to_use = 'Seat.TLabelframe'

                    # Apply the determined style
                    widgets['frame'].configure(style=style_to_use)
                except tk.TclError:
                    # Handle error if widget is destroyed during style update
                    # print(f"Warning: TclError highlighting seat {i}.")
                    pass


    def update_game_info(self, round_name, dealer, small_blind, big_blind, current_player, to_call):
        """Updates the labels in the Game Information panel."""
        # Create a dictionary mapping widget references to their new values
        ui_elements = {
            'round': (self.round_value, str(round_name)),
            'dealer': (self.dealer_value, str(dealer)),
            'blinds': (self.blinds_value, f"${small_blind:,.0f} / ${big_blind:,.0f}"), # Comma format blinds
            'current': (self.current_value, str(current_player)),
            'tocall': (self.tocall_value, f"${to_call:,.0f}") # Comma format amount to call
        }
        # Iterate and update each widget safely
        for key, (widget, value) in ui_elements.items():
            if widget and widget.winfo_exists(): # Check if widget exists
                try:
                    widget.configure(text=value) # Update the text
                except tk.TclError: pass # Ignore errors if widget destroyed during update


    def update_action_buttons(self, enable):
        """Enables or disables ALL action buttons uniformly."""
        state = tk.NORMAL if enable else tk.DISABLED
        for action_name, button in self.action_buttons.items():
            if button and button.winfo_exists():
                try:
                    button.configure(state=state)
                except tk.TclError: pass # Ignore errors if button destroyed


    def update_legal_actions_display(self, legal_actions, amount_to_call=0):
        """
        Enables/disables action buttons based on the list of legal action strings
        provided by the environment, also considering the amount to call.
        """
        # If it's not the human's turn, disable all buttons and return
        if not self.is_human_turn:
            self.update_action_buttons(False)
            return

        # Ensure legal_actions is a set for efficient lookup
        if not isinstance(legal_actions, (list, set)):
            print(f"Warning: Invalid legal_actions received: {legal_actions}")
            legal_actions_set = set() # Treat as no legal actions
        else:
             legal_actions_set = set(legal_actions)

        # --- Initial Enable/Disable based on raw legal_actions ---
        for action_name, button in self.action_buttons.items():
            if button and button.winfo_exists():
                is_legal = action_name in legal_actions_set
                button_state = tk.NORMAL if is_legal else tk.DISABLED
                try:
                     button.configure(state=button_state)
                except tk.TclError: pass # Ignore if button destroyed

        # --- Refined Logic: Check vs Call ---
        # Get the Check and Call buttons if they exist
        check_button = self.action_buttons.get('check')
        call_button = self.action_buttons.get('call')

        try:
            # Only apply refinement if both buttons exist
            if check_button and check_button.winfo_exists() and call_button and call_button.winfo_exists():
                can_check = 'check' in legal_actions_set
                can_call = 'call' in legal_actions_set

                if amount_to_call > 0:
                    # If there's an amount to call:
                    # - Disable 'Check' button, even if technically legal from env.
                    # - Ensure 'Call' button is enabled if it was legal.
                    if can_check: check_button.configure(state=tk.DISABLED)
                    # Ensure call button state reflects its legality if check was disabled
                    # if can_call: call_button.configure(state=tk.NORMAL) # This line might re-enable call incorrectly if it wasn't legal

                else: # amount_to_call is 0
                    # If amount to call is 0:
                    # - Disable 'Call' button, even if technically legal from env.
                    # - Ensure 'Check' button is enabled if it was legal.
                    if can_call: call_button.configure(state=tk.DISABLED)
                    # Ensure check button state reflects its legality if call was disabled
                    # if can_check: check_button.configure(state=tk.NORMAL) # This line might re-enable check incorrectly

        except tk.TclError:
             # print("Warning: TclError during check/call refinement.")
             pass # Ignore if widgets destroyed during logic
        except Exception as e:
             print(f"Error during check/call refinement: {e}")


    # --- Seat Configuration Dialog ---

    def _show_seat_config_dialog(self):
        """Displays a Toplevel window for configuring player seats and AI models."""
        # Create the Toplevel window
        config_window = tk.Toplevel(self.root)
        config_window.title("Configure Seats")
        config_window.transient(self.root) # Keep window on top of main app
        config_window.grab_set() # Make it modal (block interaction with main window)
        config_window.resizable(False, False) # Prevent resizing the dialog
        # Use a theme-consistent background color
        config_window.configure(background=self.theme.COLORS.get('bg_dialog', self.theme.COLORS['bg_main']))

        # Main content frame inside the dialog
        content_frame = ttk.Frame(config_window, padding="15", style='TFrame')
        content_frame.pack(expand=True, fill="both")

        # Dialog Title
        ttk.Label(content_frame, text="Configure Player Seats", style='Header.TLabel').pack(pady=(0, 20))

        # Frame to hold the rows of seat options
        options_frame = ttk.Frame(content_frame, style='TFrame')
        options_frame.pack(pady=5)

        # --- Data Structures for UI Elements ---
        # Store Tkinter variables and widget references for each seat row
        seat_vars = [] # List of tk.StringVar holding the selected seat type (e.g., 'player', 'model')
        checkpoint_vars = [] # List of tk.StringVar holding the checkpoint file path
        checkpoint_widgets = [] # List of dicts: {'entry': ttk.Entry, 'button': ttk.Button}

        # --- Helper Function: Browse for Checkpoint File ---
        def browse_checkpoint(index, cp_var):
            """Opens a file dialog to select a checkpoint file (.pt) and updates the cp_var."""
            # Determine initial directory for browsing (prefer checkpoints folder)
            script_dir = os.path.dirname(__file__) or "."
            initial_dir = os.path.abspath(os.path.join(script_dir, '..', 'checkpoints'))
            # Fallback to user's home directory if checkpoints folder doesn't exist
            if not os.path.isdir(initial_dir): initial_dir = os.path.expanduser("~")

            filepath = filedialog.askopenfilename(
                title=f"Select Model Checkpoint for Seat {index+1}",
                filetypes=[("PyTorch Checkpoints", "*.pt"), ("All Files", "*.*")],
                initialdir=initial_dir,
                parent=config_window # Ensure dialog is modal relative to the config window itself
            )
            # If a file was selected, update the corresponding StringVar
            if filepath:
                cp_var.set(filepath)

        # --- Create Configuration Row for Each Player ---
        for i in range(NUM_PLAYERS):
            row_frame = ttk.Frame(options_frame, style='TFrame')
            row_frame.pack(fill=tk.X, pady=4)

            # Seat Label (e.g., "Seat 1:")
            if i == 1:
                ttk.Label(row_frame, text=f"Player", width=8).pack(side=tk.LEFT, padx=(0, 10))
            else:
                ttk.Label(row_frame, text=f"Seat {i+1}:", width=8).pack(side=tk.LEFT, padx=(0, 10))

            # --- Seat Type Dropdown ---
            current_type = self.seat_config.get(i, 'empty') # Get current type from app state
            var = tk.StringVar(value=current_type) # Create StringVar for this seat's type
            seat_vars.append(var) # Store the variable

            # Get available seat type options from the manager
            options = self.seat_config_manager.get_options() # e.g., ['empty', 'player', 'model', 'random']

            # Ensure the currently stored type is valid, default to 'empty' if not
            if current_type not in options:
                 print(f"Warning: Saved type '{current_type}' for seat {i+1} is invalid. Resetting to 'empty'.")
                 current_type = 'empty'
                 var.set(current_type)

            # Configure dropdown based on whether it's the human player's seat
            if i == self.human_player_id:
                # Human player seat is fixed to 'player'
                human_opts = ["player"]
                if var.get() != "player": var.set("player") # Ensure variable is correct
                dropdown = ttk.OptionMenu(row_frame, var, "player", *human_opts)
                dropdown.configure(state=tk.DISABLED) # Disable changing the human player type
            else:
                # Other seats can be configured using all available options
                dropdown = ttk.OptionMenu(row_frame, var, current_type, *options)

            dropdown.pack(side=tk.LEFT, padx=5)

            # --- Checkpoint Path Entry and Button (Conditional) ---
            # Create StringVar for the checkpoint path for this seat
            cp_var = tk.StringVar(value=self.checkpoint_paths.get(i) or "") # Get saved path or empty string
            checkpoint_vars.append(cp_var) # Store the variable

            # Create Entry and Button widgets (initially packed, state controlled by callback)
            cp_entry = ttk.Entry(row_frame, textvariable=cp_var, width=40, state=tk.DISABLED)
            cp_browse = ttk.Button(row_frame, text="...", width=3, state=tk.DISABLED,
                                   # Pass the specific cp_var to the browse command using lambda
                                   command=lambda idx=i, var=cp_var: browse_checkpoint(idx, var))

            # Pack widgets: Browse button first (right), then Entry fills remaining space
            cp_browse.pack(side=tk.RIGHT, padx=(5, 0))
            cp_entry.pack(side=tk.RIGHT, padx=(5, 0), fill=tk.X, expand=True)

            # Store references to these widgets
            checkpoint_widgets.append({'entry': cp_entry, 'button': cp_browse})

            # --- Callback Function to Enable/Disable Checkpoint Widgets ---
            # Uses a factory function (create_toggle_callback) to correctly capture loop variables (index, var, widgets)
            def create_toggle_callback(index, seat_type_var, widgets_dict):
                """Creates a callback function that knows its specific index, variable, and widgets."""
                def toggle_widgets_state(*args): # Callback signature for trace_add
                     entry_widget = widgets_dict['entry']
                     button_widget = widgets_dict['button']
                     # Check if widgets still exist before configuring them
                     if not entry_widget.winfo_exists() or not button_widget.winfo_exists(): return

                     # Enable entry/button only if seat type is 'model' AND PyTorch is available
                     is_model_seat = seat_type_var.get() == 'model'
                     enable_state = tk.NORMAL if (is_model_seat and torch) else tk.DISABLED

                     entry_widget.configure(state=enable_state)
                     button_widget.configure(state=enable_state)

                     # Add a visual cue if model selected but torch unavailable? (Optional)
                     # if is_model_seat and not torch:
                     #     entry_widget.configure(foreground='gray') # Example
                     # else:
                     #      entry_widget.configure(foreground=self.theme.COLORS['text_input'])

                return toggle_widgets_state

            # Add trace to the seat type variable (only for non-human seats)
            if i != self.human_player_id:
                callback = create_toggle_callback(i, var, checkpoint_widgets[i])
                var.trace_add("write", callback) # Call callback whenever the dropdown value changes
                # Call initially to set the correct state based on the loaded config
                callback()
            else:
                 # Ensure checkpoint entry/button are always disabled for the human player seat
                 cp_entry.configure(state=tk.DISABLED)
                 cp_browse.configure(state=tk.DISABLED)


        # --- Dialog Action Buttons (Apply/Cancel) ---
        button_frame = ttk.Frame(content_frame, style='TFrame')
        button_frame.pack(pady=(25, 5)) # Add space above the buttons

        def apply_config():
            """Validates the selected configuration, applies it to the app state, and closes the dialog."""
            temp_config = {} # Temporary dict to store new seat types {id: type}
            temp_paths = {} # Temporary dict to store new checkpoint paths {id: path or None}
            selected_types = [] # List of selected types for validation check

            # Read values from the UI variables into temporary storage
            for idx, seat_type_var in enumerate(seat_vars):
                seat_type = seat_type_var.get()
                temp_config[idx] = seat_type
                selected_types.append(seat_type) # Collect type for validation

                # Get checkpoint path only if seat type is 'model'
                if seat_type == 'model':
                    path = checkpoint_vars[idx].get().strip() # Get path from corresponding StringVar
                    # Basic check if path exists (optional, but helpful)
                    if path and not os.path.exists(path):
                         # Warn user if path specified but doesn't exist
                         messagebox.showwarning("Path Warning",
                                                f"Checkpoint path for Seat {idx+1} does not exist:\n{path}\n\nModel loading will likely fail.",
                                                parent=config_window) # Ensure warning is modal to dialog
                         temp_paths[idx] = path # Store the invalid path anyway
                    elif path:
                         temp_paths[idx] = path # Store the valid path
                    else:
                         temp_paths[idx] = None # Store None if path is empty
                         # Optionally warn if type is 'model' but no path is provided
                         messagebox.showwarning("Path Missing",
                                                f"Seat {idx+1} is set to 'Model' but no checkpoint path is provided.",
                                                parent=config_window)
                else:
                    temp_paths[idx] = None # No path needed for non-model types

            # --- Validate the overall configuration ---
            # Example validation: Ensure at least one opponent is selected
            is_valid, message = self.seat_config_manager.validate_config(selected_types)
            if not is_valid:
                messagebox.showerror("Configuration Error", message, parent=config_window)
                return # Stop processing if validation fails

            # --- Apply Configuration to Main App State ---
            self.seat_config = temp_config
            self.checkpoint_paths = temp_paths
            print("Seat configuration updated:", self.seat_config)
            print("Checkpoint paths updated:", self.checkpoint_paths)

            config_window.destroy() # Close the configuration dialog

            # --- Ask User to Restart ---
            # Prompt user if they want to start a new tournament with the applied settings
            if messagebox.askyesno("Restart Tournament?",
                                   "Apply new seat configuration and start a new tournament?",
                                   parent=self.root): # Parent should be main window
                self.setup_game() # Re-run the setup process with the new config/paths

        # Create Apply and Cancel buttons
        apply_btn = ttk.Button(button_frame, text="Apply & Restart", command=apply_config, style='Accent.TButton') # Use accent style for apply
        apply_btn.pack(side=tk.LEFT, padx=10)

        cancel_btn = ttk.Button(button_frame, text="Cancel", command=config_window.destroy)
        cancel_btn.pack(side=tk.LEFT, padx=10)

        # --- Finalize Dialog ---
        config_window.update_idletasks() # Ensure window dimensions are calculated before centering

        # Center the dialog relative to the main application window
        root_x = self.root.winfo_rootx() # Get main window's screen X
        root_y = self.root.winfo_rooty() # Get main window's screen Y
        root_w = self.root.winfo_width() # Get main window's width
        root_h = self.root.winfo_height() # Get main window's height
        dlg_w = config_window.winfo_width() # Get dialog's width
        dlg_h = config_window.winfo_height() # Get dialog's height

        # Calculate position for top-left corner of dialog
        x = root_x + (root_w - dlg_w) // 2
        y = root_y + (root_h - dlg_h) // 2
        config_window.geometry(f"+{x}+{y}") # Set dialog position

        config_window.focus_set() # Set focus to the dialog window
        config_window.wait_window() # Wait until the dialog is closed before returning


    def run(self):
        """Starts the Tkinter main event loop."""
        print("Starting Poker Application UI...")
        self.root.mainloop() # Enter the Tkinter event loop


# --- Main Execution Block ---
if __name__ == "__main__":
    # This block runs only when the script is executed directly
    root = tk.Tk() # Create the main Tkinter window instance
    root.withdraw() # Hide the main window initially until setup is complete
    app = PokerApp(root) # Create the application instance
    # The main window is shown via root.deiconify() at the end of app.__init__
    app.run() # Start the application's main loop
