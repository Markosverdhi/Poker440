import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont

class PokerTheme:
    COLORS = {
        'bg_main': '#1B2932',
        'bg_table': '#0D5C3C',
        'accent': '#D4AF37',
        'accent_secondary': '#C14953',
        'text_primary': '#FFFFFF',
        'text_secondary': '#E0E0E0',
        'text_muted': '#9EADB9',
        'success': '#4CAF50',
        'warning': '#FF9800',
        'danger': '#F44336',
        'info': '#2196F3',
        'neutral': '#9E9E9E',
        'bg_frame': '#263742',
        'pl_frame': '#FFFDD0',
        'bg_button': '#2C3E50',
        'bg_button_hover': '#34495E',
        'bg_button_active': '#1B2631',
        'border': '#4E5D6C',
        'highlight': '#F1C40F',
        'shadow': '#121A21',
    }
    
    FONTS = {
        'primary': ('Segoe UI', 'Roboto', 'Helvetica', 'Arial'),
        'monospace': ('Consolas', 'Courier New', 'monospace'),
        'accent': ('Playfair Display', 'Georgia', 'serif'),
    }
    
    def __init__(self, root):
        self.root = root
        self.style = ttk.Style(root)
        self._configure_fonts()
        self._create_custom_theme()
        
    def _configure_fonts(self):
        available_fonts = tkfont.families()
        
        primary_font = next((f for f in self.FONTS['primary'] if f in available_fonts), None)
        if not primary_font:
            primary_font = self.style.lookup("TLabel", "font").split()[0]
        
        monospace_font = next((f for f in self.FONTS['monospace'] if f in available_fonts), None)
        if not monospace_font:
            monospace_font = "TkFixedFont"
        
        accent_font = next((f for f in self.FONTS['accent'] if f in available_fonts), None)
        if not accent_font:
            accent_font = primary_font
        
        self.default_font = tkfont.Font(family=primary_font, size=11)
        self.title_font = tkfont.Font(family=accent_font, size=16, weight="bold")
        self.header_font = tkfont.Font(family=primary_font, size=14, weight="bold")
        self.button_font = tkfont.Font(family=primary_font, size=12)
        self.card_font = tkfont.Font(family=monospace_font, size=12, weight="bold")
        self.status_font = tkfont.Font(family=primary_font, size=11)
        self.small_font   = tkfont.Font(family=primary_font, size=9)
        
    def _create_custom_theme(self):
        self.style.theme_use('default')
        self.root.configure(background=self.COLORS['bg_main'])
        
        self.style.configure('.',
             background=self.COLORS['bg_main'],
             foreground=self.COLORS['text_primary'],
             font=self.default_font,
             borderwidth=1,
             relief=tk.FLAT
        )
        
        self.style.configure('TFrame',
            background=self.COLORS['bg_frame'],
            borderwidth=0
        )
        
        self.style.configure('SmallInfoText.TLabel',
             font=self.small_font,                    # now defined
             foreground=self.COLORS['text_secondary']
        )
        self.style.configure('SmallInfoValue.TLabel',
             font=self.small_font,                    # now defined
             foreground=self.COLORS['text_primary']
        )
        
        self.style.configure('Table.TFrame',
            background=self.COLORS['bg_table'],
            borderwidth=2,
            relief=tk.RAISED
        )
        
        self.style.configure('TLabel',
            background=self.COLORS['bg_frame'],
            foreground=self.COLORS['text_primary'],
            font=self.default_font
        )
        
        self.style.configure('Title.TLabel',
            font=self.title_font,
            foreground=self.COLORS['accent'],
            background=self.COLORS['bg_main'],
            padding=(10, 5)
        )
        
        self.style.configure('Header.TLabel',
            font=self.header_font,
            foreground=self.COLORS['text_primary'],
            background=self.COLORS['bg_frame'],
            padding=(5, 2)
        )
        
        self.style.configure('Card.TLabel',
            font=self.card_font,
            background=self.COLORS['bg_main'],
            foreground=self.COLORS['text_primary'],
            padding=5,
            borderwidth=2,
            relief=tk.RAISED
        )
        
        self.style.configure('PlayerHand.TLabel',
            font=self.card_font,
            background=self.COLORS['bg_main'],
            foreground=self.COLORS['accent'],
            padding=5,
            borderwidth=2,
            relief=tk.RAISED
        )
        
        self.style.configure('Pot.TLabel',
            font=self.header_font,
            foreground=self.COLORS['accent'],
            background=self.COLORS['bg_table'],
            padding=(10, 5)
        )
        
        self.style.configure('Overview.TLabel',
            font=self.default_font,
            foreground=self.COLORS['text_primary'],
            background=self.COLORS['bg_table'],
            padding=(10, 5)
        )
        
        self.style.configure('Showdown.TLabel',
            font=self.card_font,
            foreground=self.COLORS['text_secondary'],
            background=self.COLORS['bg_frame'],
            padding=5
        )
        
        self.style.configure('TButton',
            font=self.button_font,
            background=self.COLORS['bg_button'],
            foreground=self.COLORS['text_primary'],
            borderwidth=2,
            relief=tk.RAISED,
            padding=(10, 5)
        )
        
        self.style.map('TButton',
            background=[('active', self.COLORS['bg_button_active']), 
                        ('hover', self.COLORS['bg_button_hover'])],
            foreground=[('active', self.COLORS['text_primary'])],
            relief=[('pressed', tk.SUNKEN), ('active', tk.RAISED)]
        )
        
        self.style.configure('Fold.TButton',
            background=self.COLORS['danger'],
            foreground=self.COLORS['text_primary']
        )
        self.style.map('Fold.TButton',
            background=[('active', '#D32F2F'), ('hover', '#E53935')],
            foreground=[('active', self.COLORS['text_primary'])]
        )
        
        self.style.configure('Call.TButton',
            background=self.COLORS['info'],
            foreground=self.COLORS['text_primary']
        )
        self.style.map('Call.TButton',
            background=[('active', '#1976D2'), ('hover', '#2196F3')],
            foreground=[('active', self.COLORS['text_primary'])]
        )
        
        self.style.configure('Check.TButton',
            background=self.COLORS['neutral'],
            foreground=self.COLORS['text_primary']
        )
        self.style.map('Check.TButton',
            background=[('active', '#757575'), ('hover', '#9E9E9E')],
            foreground=[('active', self.COLORS['text_primary'])]
        )
        
        self.style.configure('Bet.TButton',
            background=self.COLORS['success'],
            foreground=self.COLORS['text_primary']
        )
        self.style.map('Bet.TButton',
            background=[('active', '#388E3C'), ('hover', '#4CAF50')],
            foreground=[('active', self.COLORS['text_primary'])]
        )
        
        self.style.configure('AllIn.TButton',
            background=self.COLORS['warning'],
            foreground=self.COLORS['text_primary']
        )
        self.style.map('AllIn.TButton',
            background=[('active', '#E65100'), ('hover', '#FF9800')],
            foreground=[('active', self.COLORS['text_primary'])]
        )
        
        self.style.configure('TLabelframe',
            background=self.COLORS['bg_frame'],
            foreground=self.COLORS['text_primary'],
            borderwidth=2,
            relief=tk.GROOVE
        )
        
        self.style.configure('TLabelframe.Label',
            font=self.header_font,
            background=self.COLORS['bg_frame'],
            foreground=self.COLORS['accent']
        )
        
        self.style.configure('Seat.TLabelframe',
            background=self.COLORS['bg_frame'],
            borderwidth=2,
            relief=tk.GROOVE
        )
        
        self.style.configure('Active.Seat.TLabelframe',
            background=self.COLORS['pl_frame'],
            borderwidth=2,
            relief=tk.SUNKEN
        )
        self.style.map('Active.Seat.TLabelframe',
            background=[('active', self.COLORS['pl_frame'])],
            bordercolor=[('active', self.COLORS['accent'])]
        )
        
        self.style.configure('TMenubutton',
            font=self.default_font,
            background=self.COLORS['bg_button'],
            foreground=self.COLORS['text_primary'],
            padding=(10, 5)
        )
        
        self.style.configure('TEntry',
            font=self.default_font,
            fieldbackground=self.COLORS['bg_main'],
            foreground=self.COLORS['text_primary'],
            bordercolor=self.COLORS['border'],
            lightcolor=self.COLORS['bg_frame'],
            darkcolor=self.COLORS['shadow'],
            insertcolor=self.COLORS['text_primary'],
            padding=(5, 2)
        )
        
        self.style.configure('TSeparator',
            background=self.COLORS['border']
        )
        
        self.style.configure('TProgressbar',
            background=self.COLORS['accent'],
            troughcolor=self.COLORS['bg_frame'],
            bordercolor=self.COLORS['border'],
            lightcolor=self.COLORS['bg_frame'],
            darkcolor=self.COLORS['shadow']
        )
        
    def apply_theme(self):
        pass
    
    def get_color(self, color_name):
        return self.COLORS.get(color_name, self.COLORS['text_primary'])
    
    def get_font(self, font_name):
        fonts = {
            'default': self.default_font,
            'title': self.title_font,
            'header': self.header_font,
            'button': self.button_font,
            'card': self.card_font,
            'status': self.status_font
        }
        return fonts.get(font_name, self.default_font)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Poker Theme Demo")
    theme = PokerTheme(root)
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill=tk.BOTH, expand=True)
    title = ttk.Label(frame, text="Poker Theme Demo", style='Title.TLabel')
    title.pack(pady=10)
    header = ttk.Label(frame, text="Button Styles", style='Header.TLabel')
    header.pack(pady=10)
    buttons_frame = ttk.Frame(frame)
    buttons_frame.pack(pady=10)
    fold_btn = ttk.Button(buttons_frame, text="Fold", style='Fold.TButton')
    fold_btn.pack(side=tk.LEFT, padx=5)
    check_btn = ttk.Button(buttons_frame, text="Check", style='Check.TButton')
    check_btn.pack(side=tk.LEFT, padx=5)
    call_btn = ttk.Button(buttons_frame, text="Call", style='Call.TButton')
    call_btn.pack(side=tk.LEFT, padx=5)
    bet_btn = ttk.Button(buttons_frame, text="Bet", style='Bet.TButton')
    bet_btn.pack(side=tk.LEFT, padx=5)
    allin_btn = ttk.Button(buttons_frame, text="All In", style='AllIn.TButton')
    allin_btn.pack(side=tk.LEFT, padx=5)
    card_header = ttk.Label(frame, text="Card Display", style='Header.TLabel')
    card_header.pack(pady=10)
    card_frame = ttk.Frame(frame)
    card_frame.pack(pady=10)
    card1 = ttk.Label(card_frame, text="A♠", style='Card.TLabel')
    card1.pack(side=tk.LEFT, padx=5)
    card2 = ttk.Label(card_frame, text="K♥", style='Card.TLabel')
    card2.pack(side=tk.LEFT, padx=5)
    card3 = ttk.Label(card_frame, text="Q♦", style='Card.TLabel')
    card3.pack(side=tk.LEFT, padx=5)
    card4 = ttk.Label(card_frame, text="J♣", style='Card.TLabel')
    card4.pack(side=tk.LEFT, padx=5)
    hand_header = ttk.Label(frame, text="Player Hand", style='Header.TLabel')
    hand_header.pack(pady=10)
    hand_frame = ttk.Frame(frame)
    hand_frame.pack(pady=10)
    hand1 = ttk.Label(hand_frame, text="A♠", style='PlayerHand.TLabel')
    hand1.pack(side=tk.LEFT, padx=5)
    hand2 = ttk.Label(hand_frame, text="A♥", style='PlayerHand.TLabel')
    hand2.pack(side=tk.LEFT, padx=5)
    pot = ttk.Label(frame, text="Pot: $1,000", style='Pot.TLabel')
    pot.pack(pady=10)
    showdown = ttk.Label(frame, text="Player 1 wins with Pair of Aces", style='Overview.TLabel')
    showdown.pack(pady=10)
    root.mainloop()
