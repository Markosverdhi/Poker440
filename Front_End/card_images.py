import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import random

class CardImageGenerator:
    def __init__(self):
        self.card_width = 80
        self.card_height = 120
        self.corner_radius = 10
        self.card_images = {}
        self.placeholder_image = self._create_placeholder_image()
        
    def get_card_image(self, card_code):
        if card_code in self.card_images:
            return self.card_images[card_code]
            
        if card_code == "??" or card_code is None:
            return self.get_card_back()
            
        img = self._create_card_image(card_code)
        self.card_images[card_code] = ImageTk.PhotoImage(img)
        return self.card_images[card_code]
    
    def get_card_back(self):
        if "back" in self.card_images:
            return self.card_images["back"]
            
        img = self._create_card_back()
        self.card_images["back"] = ImageTk.PhotoImage(img)
        return self.card_images["back"]
    
    def get_placeholder_image(self):
        return self.placeholder_image
    
    def _create_rounded_rectangle(self, draw, xy, radius, fill, outline=None):
        x1, y1, x2, y2 = xy
        draw.rectangle((x1 + radius, y1, x2 - radius, y2), fill=fill, outline=outline)
        draw.rectangle((x1, y1 + radius, x2, y2 - radius), fill=fill, outline=outline)
        draw.pieslice((x1, y1, x1 + radius * 2, y1 + radius * 2), 180, 270, fill=fill, outline=outline)
        draw.pieslice((x2 - radius * 2, y1, x2, y1 + radius * 2), 270, 360, fill=fill, outline=outline)
        draw.pieslice((x1, y2 - radius * 2, x1 + radius * 2, y2), 90, 180, fill=fill, outline=outline)
        draw.pieslice((x2 - radius * 2, y2 - radius * 2, x2, y2), 0, 90, fill=fill, outline=outline)
    
    def _create_placeholder_image(self):
        img = Image.new('RGBA', (self.card_width, self.card_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        self._create_rounded_rectangle(
            draw, 
            (0, 0, self.card_width, self.card_height), 
            self.corner_radius, 
            fill="#263742", 
            outline="#4E5D6C"
        )
        
        placeholder = ImageTk.PhotoImage(img)
        return placeholder
    
    def _create_card_image(self, card_code):
        rank = card_code[:-1]
        suit = card_code[-1]
        
        suit_symbols = {
            'S': '♠',
            'H': '♥',
            'D': '♦',
            'C': '♣'
        }
        
        suit_colors = {
            'S': "#000000",
            'H': "#C14953",
            'D': "#C14953",
            'C': "#000000"
        }
        
        symbol = suit_symbols.get(suit, '?')
        color = suit_colors.get(suit, "#000000")
        
        img = Image.new('RGBA', (self.card_width, self.card_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        self._create_rounded_rectangle(
            draw, 
            (0, 0, self.card_width, self.card_height), 
            self.corner_radius, 
            fill="#FFFFFF", 
            outline="#000000"
        )
        
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
        
        draw.text((10, 10), rank, fill=color, font=font)
        draw.text((10, 35), symbol, fill=color, font=font)
        
        draw.text((self.card_width - 25, self.card_height - 35), rank, fill=color, font=font)
        draw.text((self.card_width - 25, self.card_height - 60), symbol, fill=color, font=font)
        
        center_symbol_size = 40
        try:
            center_font = ImageFont.truetype("arial.ttf", center_symbol_size)
        except IOError:
            center_font = ImageFont.load_default()
        
        draw.text(
            (self.card_width // 2 - center_symbol_size // 2, 
             self.card_height // 2 - center_symbol_size // 2),
            symbol,
            fill=color,
            font=center_font
        )
        
        return img
    
    def _create_card_back(self):
        img = Image.new('RGBA', (self.card_width, self.card_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        self._create_rounded_rectangle(
            draw, 
            (0, 0, self.card_width, self.card_height), 
            self.corner_radius, 
            fill="#0D5C3C", 
            outline="#000000"
        )
        
        pattern_color = "#0A4A30"
        
        for x in range(0, self.card_width, 10):
            for y in range(0, self.card_height, 10):
                if (x + y) % 20 == 0:
                    draw.rectangle((x, y, x + 5, y + 5), fill=pattern_color)
        
        return img
    
    def get_random_card(self):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['S', 'H', 'D', 'C']
        
        rank = random.choice(ranks)
        suit = random.choice(suits)
        
        return f"{rank}{suit}"

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Card Image Generator Demo")
    root.configure(background="#1B2932")
    
    card_gen = CardImageGenerator()
    
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill=tk.BOTH, expand=True)
    
    title = ttk.Label(frame, text="Card Image Generator Demo", font=("Arial", 16, "bold"))
    title.pack(pady=10)
    
    cards_frame = ttk.Frame(frame)
    cards_frame.pack(pady=10)
    
    test_cards = ['AS', 'KH', 'QD', 'JC', '10S']
    
    for card_code in test_cards:
        card_img = card_gen.get_card_image(card_code)
        card_label = ttk.Label(cards_frame, image=card_img)
        card_label.pack(side=tk.LEFT, padx=5)
    
    back_img = card_gen.get_card_back()
    back_label = ttk.Label(cards_frame, image=back_img)
    back_label.pack(side=tk.LEFT, padx=5)
    
    placeholder_img = card_gen.get_placeholder_image()
    placeholder_label = ttk.Label(cards_frame, image=placeholder_img)
    placeholder_label.pack(side=tk.LEFT, padx=5)
    
    root.mainloop()
