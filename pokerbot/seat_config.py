# seat_config.py
import tkinter as tk # Only needed for example usage demonstration below
from tkinter import ttk # Only needed for example usage demonstration below

class SeatConfigManager:
    """
    Helper class to manage seat configuration settings for the poker application.

    This class centralizes the logic for:
      - Defining available seat types (player, model, etc.).
      - Setting default configurations for seats.
      - Providing validation logic for seat assignments.
      - (Potentially) Saving/loading configurations in future extensions.

    While the main application (main_ui.py) currently integrates much of this
    logic directly for simplicity, this class structure promotes modularity.
    """
    def __init__(self, num_players):
        """
        Initializes the SeatConfigManager.

        Args:
            num_players (int): The total number of seats in the game.
        """
        if not isinstance(num_players, int) or num_players <= 0:
            raise ValueError("Number of players must be a positive integer.")

        self.num_players = num_players
        # Define the available types for seat assignment
        self.options = ["player", "model", "random", "variable", "empty"]

        # Define the default configuration
        self.default_config = {}
        # Seat 0 (index) defaults to player
        self.default_config[0] = "player"
        # All other seats default to model
        for i in range(1, num_players):
             self.default_config[i] = "model"

    def get_options(self):
        """Returns the list of available seat type options."""
        return self.options

    def get_default(self, seat_index):
         """
         Gets the default configuration type for a specific seat index.

         Args:
             seat_index (int): The index of the seat (0 to num_players - 1).

         Returns:
             str: The default seat type (e.g., "player", "model").
                  Returns "empty" if the index is out of range as a safe default.
         """
         return self.default_config.get(seat_index, "empty")

    def validate_config(self, seat_variables):
        """
        Validates a list of seat configuration selections (e.g., from UI widgets).

        Ensures that exactly one seat is assigned as 'player'. Can be extended
        with more complex rules if needed.

        Args:
            seat_variables (list): A list where each element represents the
                                   selected type for a seat (e.g., strings like
                                   "player", "model", or Tkinter StringVars).

        Returns:
            tuple: (bool, str) where bool indicates validity (True/False)
                   and str provides an error message if invalid, or a success message.
        """
        # Extract string values if Tkinter variables are passed
        if seat_variables and hasattr(seat_variables[0], 'get'): # Check if elements have a 'get' method
            selected_types = [var.get() for var in seat_variables]
        else:
            selected_types = seat_variables # Assume list of strings

        if len(selected_types) != self.num_players:
             return False, f"Configuration length ({len(selected_types)}) does not match number of players ({self.num_players})."

        player_count = selected_types.count("player")

        if player_count == 0:
            return False, "Configuration Error: No seat assigned as 'player'."
        elif player_count > 1:
            return False, f"Configuration Error: {player_count} seats assigned as 'player'. Exactly one is required."

        # Add other potential validations here, e.g., minimum number of non-empty seats?
        # num_empty = selected_types.count("empty")
        # if num_empty >= self.num_players - 1:
        #    return False, "Configuration Error: At least two non-empty seats are required to play."


        return True, "Configuration is valid."

    # Example function showing how this class could generate UI elements
    # (Note: This exact function isn't directly called by the current main_ui.py)
    def create_config_widgets(self, parent_frame):
        """
        Creates Tkinter widgets (Labels and OptionMenus) for seat configuration
        within a given parent frame.

        Args:
            parent_frame: The Tkinter Frame or Toplevel window to place widgets in.

        Returns:
            list: A list of Tkinter StringVar objects, one for each seat, holding
                  the selected configuration type.
        """
        seat_vars = []
        ttk.Label(parent_frame, text="Configure Seats:", font="-weight bold").grid(row=0, column=0, columnspan=2, pady=(0, 10))
        for i in range(self.num_players):
            ttk.Label(parent_frame, text=f"Seat {i+1}:").grid(row=i+1, column=0, sticky=tk.W, padx=5, pady=2)
            default_val = self.get_default(i)
            var = tk.StringVar(value=default_val)

            # Using OptionMenu (dropdown)
            dropdown = ttk.OptionMenu(parent_frame, var, default_val, *self.get_options())
            dropdown.grid(row=i+1, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
            seat_vars.append(var)
        return seat_vars

# Example Usage (demonstrates how the class works independently)
if __name__ == '__main__':
    NUM_SEATS_EXAMPLE = 6
    manager = SeatConfigManager(NUM_SEATS_EXAMPLE)

    print(f"Available seat options: {manager.get_options()}")
    print(f"Default for Seat 1 (index 0): {manager.get_default(0)}")
    print(f"Default for Seat 2 (index 1): {manager.get_default(1)}")

    # --- Simulate getting config from UI ---
    # Example 1: Valid config
    simulated_ui_vars_valid = [tk.StringVar(value="player")] + [tk.StringVar(value="model") for _ in range(NUM_SEATS_EXAMPLE - 1)]
    is_valid, message = manager.validate_config(simulated_ui_vars_valid)
    print(f"\nValidating config 1: {is_valid} - {message}")
    if is_valid:
        print("  Selected types:", [var.get() for var in simulated_ui_vars_valid])

    # Example 2: Invalid config (no player)
    simulated_ui_vars_no_player = [tk.StringVar(value="model") for _ in range(NUM_SEATS_EXAMPLE)]
    is_valid, message = manager.validate_config(simulated_ui_vars_no_player)
    print(f"\nValidating config 2 (no player): {is_valid} - {message}")

    # Example 3: Invalid config (two players)
    simulated_ui_vars_two_players = [tk.StringVar(value="player"), tk.StringVar(value="player")] + [tk.StringVar(value="random") for _ in range(NUM_SEATS_EXAMPLE - 2)]
    is_valid, message = manager.validate_config(simulated_ui_vars_two_players)
    print(f"\nValidating config 3 (two players): {is_valid} - {message}")

    # --- Example of creating widgets (requires Tkinter running) ---
    # try:
    #     root = tk.Tk()
    #     root.title("Seat Config Widget Demo")
    #     main_frame = ttk.Frame(root, padding="10")
    #     main_frame.pack()
    #     created_vars = manager.create_config_widgets(main_frame)
    #
    #     def show_current_config():
    #         is_valid, message = manager.validate_config(created_vars)
    #         print("\nCurrent UI Selection Valid:", is_valid, "-", message)
    #         print("  Values:", [v.get() for v in created_vars])
    #
    #     validate_button = ttk.Button(main_frame, text="Validate Current Selection", command=show_current_config)
    #     validate_button.grid(row=NUM_SEATS_EXAMPLE + 1, column=0, columnspan=2, pady=10)
    #
    #     root.mainloop()
    # except Exception as e:
    #      print("\nTkinter widget creation skipped (requires graphical environment). Error:", e)