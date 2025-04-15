# human_action_handler.py

class HumanActionHandler:
    """
    Helper class to manage mappings and potentially validation for human player actions
    originating from the UI.

    In the current implementation of main_ui.py, the core logic (checking legal
    actions based on game state) is handled directly within the UI class for
    simplicity. This handler class primarily serves as a conceptual placeholder
    or could be expanded for more complex scenarios.
    """
    def __init__(self, action_list):
        """
        Initializes the handler with the master list of possible actions.

        Args:
            action_list (list): A list of all possible action strings recognized
                                by the game environment (e.g., ['fold', 'call', ...]).
        """
        self.action_list = action_list
        # Create a mapping from display text (e.g., "Bet Big") back to action string
        self.button_text_to_action = {
            action.replace('_', ' ').title(): action for action in action_list
        }

    def get_action_from_button(self, button_text):
        """
        Maps the display text of a UI button back to its corresponding internal
        action string.

        Args:
            button_text (str): The text displayed on the button (e.g., "Call", "Bet Small").

        Returns:
            str or None: The corresponding action string (e.g., "call", "bet_small")
                         if found, otherwise None.
        """
        return self.button_text_to_action.get(button_text)

    def validate_action(self, chosen_action, legal_actions):
        """
        Checks if the action chosen by the human is currently legal according
        to the game state.

        Args:
            chosen_action (str): The action string selected by the human (e.g., "bet_big").
            legal_actions (list): A list of action strings currently allowed by
                                  the game environment for the human player.

        Returns:
            bool: True if the action is legal, False otherwise.
        """
        # Basic validation: is the chosen action in the list of legal ones?
        return chosen_action in legal_actions

    # Example of combining getting and validating (how it might be used):
    def process_button_click(self, button_text, legal_actions):
        """
        Processes a button click, gets the action string, and validates it.

        Args:
            button_text (str): Text from the clicked button.
            legal_actions (list): List of currently legal actions from the env.

        Returns:
            str or None: The validated action string if legal, otherwise None.
        """
        action_str = self.get_action_from_button(button_text)
        if action_str and self.validate_action(action_str, legal_actions):
            return action_str
        else:
            print(f"Debug: Invalid action attempt. Clicked: '{button_text}' (maps to '{action_str}'). Legal: {legal_actions}")
            return None

# Example Usage (demonstrates how the class could be used, not part of main_ui.py directly):
if __name__ == '__main__':
    ACTION_LIST_EXAMPLE = ['fold', 'call', 'check', 'bet_small', 'bet_big', 'all_in']
    handler = HumanActionHandler(ACTION_LIST_EXAMPLE)

    # Simulate a button click
    clicked_button = "Bet Big"
    current_legal_actions = ['fold', 'call', 'bet_small', 'bet_big', 'all_in'] # Example state

    action_to_perform = handler.process_button_click(clicked_button, current_legal_actions)

    if action_to_perform:
        print(f"Button '{clicked_button}' corresponds to valid action: '{action_to_perform}'")
    else:
        print(f"Button '{clicked_button}' corresponds to an invalid action in the current state.")

    clicked_button_invalid = "Check"
    action_to_perform = handler.process_button_click(clicked_button_invalid, current_legal_actions)

    if action_to_perform:
        print(f"Button '{clicked_button_invalid}' corresponds to valid action: '{action_to_perform}'")
    else:
        print(f"Button '{clicked_button_invalid}' corresponds to an invalid action in the current state.")