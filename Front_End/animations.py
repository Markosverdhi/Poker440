import tkinter as tk
from tkinter import ttk
import time
import math

class AnimationManager:
    def __init__(self, root):
        """
        Initialize the AnimationManager.

        Args:
            root: The root Tkinter window.
        """
        self.root = root
        self.animations = {} # Dictionary to store active animations {animation_id: animation_data}
        self.animation_speed = 1.0 # Speed multiplier (1.0 = normal)
        self.next_id = 0 # Counter for generating unique animation IDs

    def set_animation_speed(self, speed):
        """
        Set the global animation speed multiplier.

        Args:
            speed: New speed multiplier (e.g., 0.5 for half speed, 2.0 for double).
                   Clamped between 0.1 and 3.0.
        """
        self.animation_speed = max(0.1, min(speed, 3.0))
        print(f"Animation speed set to: {self.animation_speed:.1f}x")

    def move_widget(self, widget, start_x, start_y, end_x, end_y, duration=500, easing="ease_out_quad", callback=None, repeat=0):
        """
        Animate moving a widget from a start position to an end position.

        Args:
            widget: The Tkinter widget to move.
            start_x, start_y: The starting coordinates.
            end_x, end_y: The ending coordinates.
            duration (int): Duration of the animation in milliseconds.
            easing (str): The easing function to use (e.g., 'linear', 'ease_out_quad').
            callback (callable, optional): Function to call when animation completes. Defaults to None.
            repeat (int): Number of times to repeat the animation (0 for once, -1 for infinite). Defaults to 0.

        Returns:
            str: The unique ID assigned to this animation.
        """
        animation_id = self._get_next_id()

        animation = {
            'widget': widget,
            'property': 'position',
            'start_value': (start_x, start_y),
            'end_value': (end_x, end_y),
            'start_time': time.time(),
            'duration': duration / 1000 / self.animation_speed, # Duration in seconds, adjusted by speed
            'easing': easing,
            'callback': callback,
            'repeat': repeat,
            'repeat_count': 0,
            'active': True
        }

        self.animations[animation_id] = animation

        # Start the main animation loop if it's not already running
        if len(self.animations) == 1:
            self._start_animation_loop()

        return animation_id

    def highlight_widget(self, widget, color, duration=500, callback=None, repeat=0):
        """
        Temporarily changes the background color of a widget for a highlight effect.
        Handles potential errors if the widget doesn't support background configuration.

        Args:
            widget: The Tkinter widget to highlight.
            color (str): The color to flash to.
            duration (int): Duration of one highlight cycle in milliseconds.
            callback (callable, optional): Function to call when animation completes. Defaults to None.
            repeat (int): Number of times to repeat the highlight flash. Defaults to 0 (flash once).

        Returns:
            str: The unique ID assigned to this animation.
        """
        # ---- START FIX for ttk background error ----
        original_bg = None
        try:
            # Attempt to get the background color
            # Check if it's a ttk widget first, as they might handle styling differently
            if isinstance(widget, ttk.Widget):
                # ttk widgets might use style configurations rather than direct cget
                # For now, we'll assume we can't reliably get/set background directly
                # A more complex solution would involve manipulating ttk styles
                print(f"Note: Highlight animation may not visually affect ttk.Widget background directly: {widget}")
                # Get the default style background if possible as a fallback
                try:
                    # Ensure self.root.style exists (might need to pass theme/style object)
                    if hasattr(self.root, 'style'):
                        style_name = widget.cget('style') or widget.winfo_class()
                        original_bg = self.root.style.lookup(style_name, 'background')
                    else:
                         original_bg = self.root.cget('background') # Fallback to root background
                except Exception:
                     original_bg = self.root.cget('background') # Fallback to root background
            else:
                # Try standard cget for non-ttk widgets
                original_bg = widget.cget('background')

        except tk.TclError as e:
            # Handle cases where 'background' option is unknown
            print(f"Warning: Could not get background for widget {widget}. Highlight animation might be skipped. Error: {e}")
            # Use a default color or skip background part of animation
            original_bg = self.root.cget('background') # Use root's background as a fallback default
        # ---- END FIX ----

        animation_id = self._get_next_id()

        animation = {
            'widget': widget,
            'property': 'highlight',
            'start_value': original_bg, # Use the obtained or fallback original_bg
            'end_value': color,
            'start_time': time.time(),
            'duration': duration / 1000 / self.animation_speed, # Duration in seconds, adjusted
            'easing': 'ease_in_out_quad', # Easing for the flash transition (optional)
            'callback': callback,
            'repeat': repeat,
            'repeat_count': 0,
            'active': True
        }

        self.animations[animation_id] = animation

        # Start the main animation loop if it's not already running
        if len(self.animations) == 1:
            self._start_animation_loop()

        return animation_id

    def pulse_widget(self, widget, scale_factor=1.2, duration=500, callback=None, repeat=0):
        """
        Creates a pulsing effect by scaling the widget up and down.

        Args:
            widget: The Tkinter widget to pulse.
            scale_factor (float): The maximum scale factor (e.g., 1.2 for 120%). Defaults to 1.2.
            duration (int): Duration of one pulse cycle in milliseconds. Defaults to 500.
            callback (callable, optional): Function to call when animation completes. Defaults to None.
            repeat (int): Number of times to repeat the pulse (0 for once, -1 for infinite). Defaults to 0.

        Returns:
            str: The unique ID assigned to this animation.
        """
        # Get original size after ensuring widget is rendered
        original_width = widget.winfo_width()
        original_height = widget.winfo_height()
        if original_width <= 1 or original_height <= 1: # Widget might not be fully drawn yet
            self.root.update_idletasks() # Force update to get dimensions
            original_width = widget.winfo_width()
            original_height = widget.winfo_height()
            if original_width <= 1 or original_height <= 1:
                 print(f"Warning: Could not get valid dimensions for widget {widget} for pulse animation.")
                 return None # Cannot animate if size is unknown

        animation_id = self._get_next_id()

        animation = {
            'widget': widget,
            'property': 'pulse',
            'start_value': (original_width, original_height), # Store original size
            'end_value': (original_width * scale_factor, original_height * scale_factor), # Target size
            'start_time': time.time(),
            'duration': duration / 1000 / self.animation_speed, # Duration of one pulse cycle (in/out)
            'easing': 'ease_in_out_sine', # Smooth easing for pulse
            'callback': callback,
            'repeat': repeat,
            'repeat_count': 0,
            'active': True,
            'original_size': (original_width, original_height) # Keep original size for reset
        }

        self.animations[animation_id] = animation

        # Start the main animation loop if it's not already running
        if len(self.animations) == 1:
            self._start_animation_loop()

        return animation_id

    def fade_widget(self, widget, start_alpha, end_alpha, duration=500, callback=None, repeat=0):
        """
        Animates the widget's transparency (alpha). Requires the window to support transparency.

        Args:
            widget: The Tkinter widget to fade (must be a Toplevel or have transparency support).
            start_alpha (float): Starting alpha value (0.0 to 1.0).
            end_alpha (float): Ending alpha value (0.0 to 1.0).
            duration (int): Duration of the fade in milliseconds. Defaults to 500.
            callback (callable, optional): Function to call when animation completes. Defaults to None.
            repeat (int): Number of times to repeat the fade (0 for once, -1 for infinite). Defaults to 0.

        Returns:
            str: The unique ID assigned to this animation.
        """
        animation_id = self._get_next_id()

        animation = {
            'widget': widget,
            'property': 'alpha',
            'start_value': start_alpha,
            'end_value': end_alpha,
            'start_time': time.time(),
            'duration': duration / 1000 / self.animation_speed,
            'easing': 'linear', # Linear fade is common
            'callback': callback,
            'repeat': repeat,
            'repeat_count': 0,
            'active': True
        }

        self.animations[animation_id] = animation

        # Start the main animation loop if it's not already running
        if len(self.animations) == 1:
            self._start_animation_loop()

        return animation_id

    def stop_animation(self, animation_id):
        """
        Stops a specific active animation by its ID.

        Args:
            animation_id (str): The ID of the animation to stop.
        """
        if animation_id in self.animations:
            self.animations[animation_id]['active'] = False
            # Note: The animation will be fully removed in the _clean_inactive_animations step

    def stop_all_animations(self):
        """ Stops all currently active animations. """
        print("Stopping all animations.")
        for animation_id in list(self.animations.keys()):
            self.animations[animation_id]['active'] = False

    def _get_next_id(self):
        """ Generates a unique ID for a new animation. """
        self.next_id += 1
        return f"anim_{self.next_id}"

    def _start_animation_loop(self):
        """ Initiates the main animation loop. """
        # print("Starting animation loop...") # Debug
        self._animation_loop()

    def _animation_loop(self):
        """
        The main loop that updates all active animations frame by frame.
        Schedules itself to run again if animations are still active.
        """
        current_time = time.time()
        active_animations_exist = False # Flag to check if any animations are still running

        # Iterate over a copy of keys because the dictionary might change during iteration
        for animation_id in list(self.animations.keys()):
            # Get the animation data, check if it still exists (might be stopped)
            animation = self.animations.get(animation_id)
            if not animation or not animation['active']:
                continue # Skip inactive or removed animations

            active_animations_exist = True # Mark that at least one animation is active
            widget = animation['widget']

            # Ensure widget still exists before trying to update it
            if not widget.winfo_exists():
                print(f"Widget for animation {animation_id} no longer exists. Stopping animation.")
                animation['active'] = False
                continue

            # Calculate animation progress
            elapsed = current_time - animation['start_time']
            duration = animation['duration']
            # Ensure duration is not zero to avoid division errors
            if duration <= 0:
                progress = 1.0
            else:
                progress = min(elapsed / duration, 1.0) # Clamp progress between 0 and 1

            # Apply easing function
            eased_progress = self._apply_easing(progress, animation['easing'])

            # --- Update widget based on animation property ---
            try:
                if animation['property'] == 'position':
                    start_x, start_y = animation['start_value']
                    end_x, end_y = animation['end_value']
                    current_x = start_x + (end_x - start_x) * eased_progress
                    current_y = start_y + (end_y - start_y) * eased_progress
                    widget.place(x=int(current_x), y=int(current_y)) # Use int for pixel coords

                elif animation['property'] == 'highlight':
                    # ---- START FIX for ttk background error ----
                    try:
                        # Determine target color based on progress (simple flash effect)
                        # Flashes to end_value color for the first half, then back to start_value
                        target_color = animation['end_value'] if (elapsed / duration * (animation['repeat'] + 1)) % 1.0 < 0.5 else animation['start_value']

                        # Attempt to configure background, skip if it fails for this widget type
                        # Check if it's a ttk widget - direct configure likely won't work as expected
                        if not isinstance(widget, ttk.Widget):
                             if target_color is not None: # Ensure color is valid
                                widget.configure(background=target_color)
                        # Else: For ttk widgets, ideally we'd change the style temporarily,
                        # but that's more complex. For now, we just don't apply the color change.

                    except tk.TclError as e:
                        # Ignore error if configure background fails for this widget
                        # print(f"Debug: Could not configure background for {widget}. Error: {e}")
                        pass # Silently ignore the error for ttk widgets
                    # ---- END FIX ----

                elif animation['property'] == 'pulse':
                    start_width, start_height = animation['original_size'] # Use original size as base
                    target_width, target_height = animation['end_value'] # Max size defined in end_value

                    # Use a sine wave for smooth pulsing in and out within the duration
                    pulse_progress = math.sin(eased_progress * math.pi) # Goes 0 -> 1 -> 0

                    current_width = start_width + (target_width - start_width) * pulse_progress
                    current_height = start_height + (target_height - start_height) * pulse_progress

                    # Recalculate position to keep center anchored (approximate)
                    # This assumes the widget's anchor is CENTER or placement is relative to center
                    # A more robust solution might require knowing the anchor point.
                    x = widget.winfo_x()
                    y = widget.winfo_y()
                    current_widget_width = widget.winfo_width() # Get current actual width
                    current_widget_height = widget.winfo_height()

                    # Calculate position adjustment based on size change
                    # This keeps the top-left corner stationary, not the center.
                    # For center anchor, adjust x and y:
                    # x_adjust = (current_widget_width - current_width) / 2
                    # y_adjust = (current_widget_height - current_height) / 2
                    # widget.place(x=x + x_adjust, y=y + y_adjust, width=int(current_width), height=int(current_height))

                    # Simple placement update (might shift if not center anchored)
                    widget.place(width=int(current_width), height=int(current_height))


                elif animation['property'] == 'alpha':
                    start_alpha = animation['start_value']
                    end_alpha = animation['end_value']
                    current_alpha = start_alpha + (end_alpha - start_alpha) * eased_progress
                    # Clamp alpha between 0.0 and 1.0
                    current_alpha = max(0.0, min(1.0, current_alpha))
                    try:
                        # This only works on Toplevel windows or if using specific libraries
                        widget.attributes('-alpha', current_alpha)
                    except tk.TclError:
                        # Alpha transparency not supported for this widget/platform
                        if progress >= 1.0: # Only print warning once at the end
                            print(f"Warning: Alpha animation not supported for widget {widget}.")
                        pass # Ignore the error and continue

            except Exception as e:
                 # Catch any other unexpected errors during widget update
                 print(f"Error updating animation {animation_id} for widget {widget}: {e}")
                 import traceback
                 traceback.print_exc()
                 animation['active'] = False # Stop problematic animation
                 continue

            # --- Check for Animation Completion ---
            if progress >= 1.0:
                animation['repeat_count'] += 1
                # Check if animation should repeat
                if animation['repeat'] == -1 or animation['repeat_count'] <= animation['repeat']:
                    # Reset start time for next repetition
                    animation['start_time'] = current_time
                    # Optionally reverse direction for properties like position/alpha
                    if animation['property'] in ['position', 'alpha']:
                         # Swap start and end values for ping-pong effect if desired
                         # animation['start_value'], animation['end_value'] = animation['end_value'], animation['start_value']
                         pass # Currently repeats from start
                else:
                    # Animation is finished
                    animation['active'] = False

                    # --- Reset to Final State ---
                    # Ensure widget is at its final intended state after animation
                    if animation['property'] == 'position':
                         end_x, end_y = animation['end_value']
                         widget.place(x=int(end_x), y=int(end_y))
                    elif animation['property'] == 'pulse':
                         # Reset to original size after pulsing
                         original_width, original_height = animation['original_size']
                         widget.place(width=original_width, height=original_height)
                    elif animation['property'] == 'alpha':
                         # Set final alpha value
                         try: widget.attributes('-alpha', animation['end_value'])
                         except tk.TclError: pass # Ignore if not supported
                    elif animation['property'] == 'highlight':
                         # ---- START FIX for ttk background error ----
                         # Reset background to original, handle potential errors
                         try:
                             if widget.winfo_exists() and animation['start_value'] is not None:
                                 # Reset to original background, skip if it fails
                                 if not isinstance(widget, ttk.Widget):
                                     widget.configure(background=animation['start_value'])
                                 # Else: No direct background reset needed for ttk (style controls it)
                         except tk.TclError:
                              pass # Ignore reset error for background
                         except Exception as e:
                             print(f"Error resetting highlight animation for {widget}: {e}")
                         # ---- END FIX ----

                    # --- Execute Callback ---
                    if animation.get('callback'):
                        try:
                            animation['callback']()
                        except Exception as e:
                            print(f"Error executing animation callback for {animation_id}: {e}")

        # --- Cleanup and Reschedule ---
        # Remove completed/inactive animations from the dictionary
        self._clean_inactive_animations()

        # If there are still active animations, schedule the next loop iteration
        if active_animations_exist:
            self.root.after(16, self._animation_loop) # Aim for ~60 FPS
        # else:
            # print("Animation loop stopped.") # Debug

    def _clean_inactive_animations(self):
        """ Removes animations marked as inactive from the dictionary. """
        inactive_ids = [anim_id for anim_id, anim_data in self.animations.items() if not anim_data['active']]
        for anim_id in inactive_ids:
            del self.animations[anim_id]

    def _apply_easing(self, progress, easing_type):
        """
        Applies a specified easing function to the progress value (0.0 to 1.0).

        Args:
            progress (float): The linear progress of the animation (0.0 to 1.0).
            easing_type (str): The name of the easing function (e.g., 'linear', 'ease_out_quad').

        Returns:
            float: The eased progress value.
        """
        # Simple Quadratic easing functions
        if easing_type == "linear":
            return progress
        elif easing_type == "ease_in_quad":
            return progress * progress
        elif easing_type == "ease_out_quad":
            return -progress * (progress - 2)
        elif easing_type == "ease_in_out_quad":
            progress *= 2
            if progress < 1:
                return 0.5 * progress * progress
            progress -= 1
            return -0.5 * (progress * (progress - 2) - 1)
        # Cubic easing functions
        elif easing_type == "ease_in_cubic":
            return progress * progress * progress
        elif easing_type == "ease_out_cubic":
            progress -= 1
            return progress * progress * progress + 1
        elif easing_type == "ease_in_out_cubic":
            progress *= 2
            if progress < 1:
                return 0.5 * progress * progress * progress
            progress -= 2
            return 0.5 * (progress * progress * progress + 2)
        # Sine easing functions
        elif easing_type == "ease_in_sine":
            return -math.cos(progress * (math.pi / 2)) + 1
        elif easing_type == "ease_out_sine":
            return math.sin(progress * (math.pi / 2))
        elif easing_type == "ease_in_out_sine":
            return -0.5 * (math.cos(math.pi * progress) - 1)
        # Exponential easing functions
        elif easing_type == "ease_in_expo":
            return 0.0 if progress == 0 else math.pow(2, 10 * (progress - 1))
        elif easing_type == "ease_out_expo":
            return 1.0 if progress == 1 else -math.pow(2, -10 * progress) + 1
        elif easing_type == "ease_in_out_expo":
            if progress == 0: return 0.0
            if progress == 1: return 1.0
            progress *= 2
            if progress < 1:
                return 0.5 * math.pow(2, 10 * (progress - 1))
            progress -= 1
            return 0.5 * (-math.pow(2, -10 * progress) + 2)
        # Circular easing functions
        elif easing_type == "ease_in_circ":
            return -1 * (math.sqrt(1 - progress * progress) - 1)
        elif easing_type == "ease_out_circ":
            progress -= 1
            return math.sqrt(1 - progress * progress)
        elif easing_type == "ease_in_out_circ":
            progress *= 2
            if progress < 1:
                return -0.5 * (math.sqrt(1 - progress * progress) - 1)
            progress -= 2
            return 0.5 * (math.sqrt(1 - progress * progress) + 1)
        # Default to linear if type is unknown
        else:
            print(f"Warning: Unknown easing function '{easing_type}'. Using linear.")
            return progress

# Example usage for testing
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Animation Manager Demo")
    root.geometry("800x600")
    # Use a default background color easily accessible
    root.configure(background="#CCCCCC") # Light gray

    animation_manager = AnimationManager(root)

    # Create a standard Tkinter button (supports background) and a ttk Button
    test_widget_tk = tk.Button(root, text="Animate Me (tk)", width=20, height=3, bg="#33AA33")
    test_widget_tk.place(x=100, y=100)

    test_widget_ttk = ttk.Button(root, text="Animate Me (ttk)", width=20)
    test_widget_ttk.place(x=100, y=200)

    # --- Control Buttons ---
    controls_frame = tk.Frame(root, bg="#CCCCCC")
    controls_frame.place(x=50, y=300)

    # Test Move
    move_btn = tk.Button(controls_frame, text="Move Both",
                        command=lambda: [
                            animation_manager.move_widget(test_widget_tk, 100, 100, 400, 100, 1000, "ease_out_quad", repeat=1),
                            animation_manager.move_widget(test_widget_ttk, 100, 200, 400, 200, 1000, "ease_out_quad", repeat=1)
                        ])
    move_btn.pack(side=tk.LEFT, padx=5)

    # Test Highlight
    highlight_btn = tk.Button(controls_frame, text="Highlight Both",
                             command=lambda: [
                                 animation_manager.highlight_widget(test_widget_tk, "#FFDD33", 500, repeat=3), # Yellow highlight
                                 animation_manager.highlight_widget(test_widget_ttk, "#FFDD33", 500, repeat=3) # Will print warning
                             ])
    highlight_btn.pack(side=tk.LEFT, padx=5)

    # Test Pulse
    pulse_btn = tk.Button(controls_frame, text="Pulse Both",
                         command=lambda: [
                             animation_manager.pulse_widget(test_widget_tk, 1.2, 500, repeat=3),
                             animation_manager.pulse_widget(test_widget_ttk, 1.2, 500, repeat=3)
                         ])
    pulse_btn.pack(side=tk.LEFT, padx=5)

    # Test Stop
    stop_btn = tk.Button(controls_frame, text="Stop All",
                        command=animation_manager.stop_all_animations)
    stop_btn.pack(side=tk.LEFT, padx=5)

    # Test Speed Control
    speed_frame = tk.Frame(root, bg="#CCCCCC")
    speed_frame.place(x=50, y=350)
    speed_label = tk.Label(speed_frame, text="Animation Speed:", bg="#CCCCCC")
    speed_label.pack(side=tk.LEFT, padx=5)
    speed_var = tk.DoubleVar(value=1.0)
    speed_scale = tk.Scale(speed_frame, from_=0.1, to=3.0, variable=speed_var, resolution=0.1,
                          orient=tk.HORIZONTAL, length=300, bg="#CCCCCC", highlightthickness=0,
                          command=lambda v: animation_manager.set_animation_speed(float(v)))
    speed_scale.pack(side=tk.LEFT, padx=5)


    root.mainloop()
