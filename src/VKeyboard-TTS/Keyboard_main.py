# Keyboard_main.py
# DESCRIPTION:
# This script implements an adjustable virtual keyboard designed for hands-free usage, incorporating cursor tracking and Text-to-Speech (TTS) functionality. The keyboard layout is customizable, with adjustable button sizes, fonts, and dimensions. It tracks the user's cursor coordinates and integrates real-time updates for interaction.

# FEATURES:
# Virtual Keyboard: Fully functional keyboard built with tkinter, supporting basic input actions like "Clear," "Enter," and "Speak."
# Adjustable Design: Includes sliders to customize button width, height, and font size dynamically.
# Mouse Tracking: Captures and displays cursor coordinates relative to the window in real-time.
# Text-to-Speech Integration: Converts entered text into speech using pyttsx3, with adjustable speech rate, volume, and voice.
# Threading for TTS: Ensures the GUI remains responsive while TTS processes text in the background.
# How to Use:
# Run the script using Python 3.
# Adjust button sizes and font via the provided sliders.
# Type into the text field using the virtual keyboard.
# Use the "Speak" button to vocalize the entered text.
# Key Functions:
# create_keyboard: Generates the virtual keyboard layout.
# speak_text: Converts the text input to speech.
# capture_mouse_coordinates: Tracks and updates the cursor position every 10ms.
# create_size_controls: Provides sliders for customizing the keyboard dimensions.

import tkinter as tk
import pyttsx3
import time
import threading

class AdjustableKeyboard:
    def __init__(self):
        # Initialize the main application window
        self.root = tk.Tk()
        self.root.title("Adjustable Virtual Keyboard with Cursor Coordinates")
        self.root.geometry("1600x800")
        self.root.minsize(1000, 600)
        # Initialize TTS engine
        self.speak_init()

        # Initialize button size variables
        self.button_width = tk.IntVar(value=3)
        self.button_height = tk.IntVar(value=1)
        self.button_font_size = tk.IntVar(value=30)

        # Buffer to store the submitted text
        self.buffer = ""

        # List to store keyboard buttons
        self.buttons = []

        # Create the text entry box and keyboard UI
        self.create_text_entry()
        self.create_keyboard()
        self.create_size_controls()

        # Create a label for displaying coordinates
        self.coordinates_label = tk.Label(self.root, text="Cursor coordinates: x=0, y=0", font=("Timesnewroman", 10))
        self.coordinates_label.grid(row=6, column=0, columnspan=12, sticky="w", padx=10, pady=5)

        # Start updating the coordinates
        self.get_cursor_coordinates()

        # self.mouse_coordinates = []  # Buffer to store recent 20 mouse coordinates
        # self.capture_mouse_coordinates()

        # Start the main loop
        self.root.mainloop()

    def create_text_entry(self):
        """Create the text entry field."""
        self.text_entry = tk.Entry(self.root, font=("Arial", 20))
        self.text_entry.grid(row=0, column=0, columnspan=12, sticky="nsew", padx=10, pady=10)

    def on_key_press(self, key):
        """Insert the pressed key character into the text entry."""
        self.text_entry.insert(tk.END, key)

    def delete_last(self):
        """Delete the last character from the text entry."""
        current_text = self.text_entry.get()
        if current_text:
            self.text_entry.delete(len(current_text) - 1)

    def clear_text(self):
        """Clear all text from the text entry."""
        self.text_entry.delete(0, tk.END)

    def update_button_sizes(self):
        """Update the width and height of all buttons based on slider values."""
        for row in self.buttons:
            if row == 1:
                for button in row:
                    button.config(width=self.button_width.get() * 20, height=self.button_height.get())

    def update_button_font_size(self):
        """Update the font size of all buttons based on slider value."""
        new_font_size = self.button_font_size.get()
        for row in self.buttons:
            for button in row:
                button.config(font=("Arial", new_font_size))

    def submit_and_clear_input(self):
        """Stores the current input in the buffer, prints it, and clears the entry field."""
        self.buffer = self.text_entry.get()  # Store the current text in buffer
        print(self.buffer)  # Print the stored text to the console
        self.clear_text()  # Clear the text entry field

    def create_keyboard(self):
        """Create the keyboard layout with wider first row buttons using columnspan."""
        keyboard_layout = [
            ['←', 'Clear', 'Enter', 'Speak'],  # First row
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],  # Second row
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],  # Third row
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '.', ' ']  # Fourth row
        ]

        for row_index, row in enumerate(keyboard_layout, start=1):
            self.root.grid_rowconfigure(row_index, weight=1)  # Configure row resizing

            button_row = []

            for col_index, key in enumerate(row):
                # Configure grid columns for uniform resizing
                self.root.grid_columnconfigure(col_index, weight=1)

                if row_index == 1:
                    # For the first row, make buttons span two columns for wider appearance
                    btn = tk.Button(self.root, text=key, command=self.get_button_command(key), height=2)
                    btn.grid(row=row_index, column=col_index * 2, columnspan=2, sticky="nsew", padx=2, pady=2)
                else:
                    # Default size and placement for other rows
                    btn = tk.Button(self.root, text=key, command=self.get_button_command(key), width=5, height=2)
                    btn.grid(row=row_index, column=col_index, sticky="nsew", padx=2, pady=2)
                button_row.append(btn)
            self.buttons.append(button_row)
        self.update_button_sizes()
        self.update_button_font_size()

    def get_button_command(self, key):
        """Return the appropriate command for a given key."""
        if key == "←":
            return self.delete_last
        elif key == "Clear":
            return self.clear_text
        elif key == "Enter":
            return self.submit_and_clear_input
        elif key == "Speak":
            return self.speak_text
        else:
            return lambda k=key: self.on_key_press(k)

    def create_keyboard1(self):
        """Create the keyboard layout and add buttons."""
        keyboard_layout = [
            ['←', 'Clear', 'Enter', 'Speak'],
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '.', ' ']
        ]

        for row_index, row in enumerate(keyboard_layout, start=1):
            self.root.grid_rowconfigure(row_index, weight=1)
            button_row = []
            for col_index, key in enumerate(row):
                self.root.grid_columnconfigure(col_index, weight=1)

                if key == "←":
                    btn = tk.Button(self.root, text=key, command=self.delete_last,
                                    width=5, height=2)
                elif key == "Clear":
                    btn = tk.Button(self.root, text=key, command=self.clear_text,
                                    width=5, height=2)
                elif key == "Enter":
                    btn = tk.Button(self.root, text=key, command=self.submit_and_clear_input,
                                    width=self.button_width.get(), height=self.button_height.get())
                elif key == "Speak":
                    btn = tk.Button(self.root, text=key, command=self.speak_text,
                                    width=5, height=2)
                else:
                    btn = tk.Button(self.root, text=key,
                                    command=lambda k=key: self.on_key_press(k),
                                    width=5, height=2)

                btn.grid(row=row_index, column=col_index, sticky="nsew", padx=2, pady=2)
                button_row.append(btn)

            self.buttons.append(button_row)
            self.update_button_sizes()
            self.update_button_font_size()

    def create_size_controls(self):
        """Create sliders for adjusting button width, height, and font size."""
        control_frame = tk.Frame(self.root)
        control_frame.grid(row=5, column=0, columnspan=12, sticky="ew", pady=20)

        # Button width slider
        tk.Label(control_frame, text="Button Width:").pack(side=tk.LEFT, padx=10)
        width_slider = tk.Scale(control_frame, from_=1, to=20, orient="horizontal",
                                variable=self.button_width,
                                command=lambda x: self.update_button_sizes())
        width_slider.pack(side=tk.LEFT, padx=10)

        # Button height slider
        tk.Label(control_frame, text="Button Height:").pack(side=tk.LEFT, padx=10)
        height_slider = tk.Scale(control_frame, from_=1, to=10, orient="horizontal",
                                 variable=self.button_height,
                                 command=lambda x: self.update_button_sizes())
        height_slider.pack(side=tk.LEFT, padx=10)

        # Button font size slider
        tk.Label(control_frame, text="Button Font Size:").pack(side=tk.LEFT, padx=10)
        font_size_slider = tk.Scale(control_frame, from_=25, to=38, orient="horizontal",
                                    variable=self.button_font_size,
                                    command=lambda x: self.update_button_font_size())
        font_size_slider.pack(side=tk.LEFT, padx=10)

    def get_cursor_coordinates(self):
        """Get and display the current cursor coordinates relative to the application window."""
        x = self.root.winfo_pointerx() - self.root.winfo_rootx()
        y = self.root.winfo_pointery() - self.root.winfo_rooty()
        self.coordinates_label.config(text=f"Cursor coordinates: x={x}, y={y}")
        # Schedule the function to run again after a short delay for continuous updating
        self.root.after(100, self.get_cursor_coordinates)

    def speak_text(self):
        """Convert the current text input to speech in a separate thread."""
        text = self.text_entry.get()
        if text.strip():  # Check if text is not empty
            threading.Thread(target=self._tts_worker, args=(text,), daemon=True).start()

    def _tts_worker(self, text):
        """Worker function to handle TTS in a separate thread."""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def create_settings_ui(self):
        """Create UI for adjusting TTS parameters dynamically."""
        settings_frame = tk.Frame(self.root)
        settings_frame.grid(row=1, column=0, columnspan=12, sticky="ew", pady=10)

        # Rate adjustment slider
        tk.Label(settings_frame, text="Rate:").pack(side=tk.LEFT, padx=5)
        rate_slider = tk.Scale(settings_frame, from_=50, to=300, orient="horizontal",
                               command=lambda x: self.update_rate(int(x)))
        rate_slider.set(self.default_rate)
        rate_slider.pack(side=tk.LEFT, padx=5)

        # Volume adjustment slider
        tk.Label(settings_frame, text="Volume:").pack(side=tk.LEFT, padx=5)
        volume_slider = tk.Scale(settings_frame, from_=0, to=1, resolution=0.1, orient="horizontal",
                                 command=lambda x: self.update_volume(float(x)))
        volume_slider.set(self.default_volume)
        volume_slider.pack(side=tk.LEFT, padx=5)

        # Voice selection dropdown
        tk.Label(settings_frame, text="Voice:").pack(side=tk.LEFT, padx=5)
        voice_var = tk.StringVar()
        voice_menu = tk.OptionMenu(settings_frame, voice_var, *[voice.name for voice in self.voices],
                                   command=self.update_voice)
        voice_var.set(self.voices[self.current_voice_index].name)
        voice_menu.pack(side=tk.LEFT, padx=5)

    def speak_init(self):
        """Initialize the TTS engine, set default parameters, and print properties."""
        self.tts_engine = pyttsx3.init()
        self.default_rate = 120  # Default speech rate
        self.default_volume = 0.8  # Default volume
        self.current_voice_index = 28  # Default voice index
        self.voices = self.tts_engine.getProperty("voices")

        # Adjust voice settings
        self.tts_engine.setProperty("voice", self.voices[self.current_voice_index].id)  # Example: male voice

        # Apply default settings
        self.tts_engine.setProperty("rate", self.default_rate)
        self.tts_engine.setProperty("volume", self.default_volume)

        # Print current engine properties for debugging
        print(f"Rate: {self.tts_engine.getProperty('rate')}")
        print(f"Volume: {self.tts_engine.getProperty('volume')}")
        print(f"Voice: {self.voices[self.current_voice_index].name} (ID: {self.voices[self.current_voice_index].id})")

        # for i in range(100):
        #     print("Voice ",i," : ",self.voices[i].name)

    def capture_mouse_coordinates(self):
        """Capture and store the recent 20 changes of mouse coordinates."""
        # Get the current cursor coordinates
        x = self.root.winfo_pointerx() - self.root.winfo_rootx()
        y = self.root.winfo_pointery() - self.root.winfo_rooty()

        # Add the current coordinates to the buffer
        self.mouse_coordinates.append((x, y))

        # Keep only the last 20 entries
        if len(self.mouse_coordinates) > 20:
            self.mouse_coordinates.pop(0)

        # Print the buffer for debugging (optional)
        print(f"Mouse Coordinates Buffer: {self.mouse_coordinates}")

        # Schedule the next update after 10ms
        self.root.after(1000, self.capture_mouse_coordinates)


# Run the AdjustableKeyboard
if __name__ == "__main__":
    x1 = AdjustableKeyboard()
