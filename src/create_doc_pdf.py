from fpdf import FPDF
from pathlib import Path
import re

# Theme Colors
THEME_ORANGE = (255, 98, 0) 
DARK_GREY = (50, 50, 50)
LIGHT_GREY = (245, 245, 245)

# Font Paths (Local to project)
FONT_DIR = Path("docs/fonts")
FONT_REGULAR = str(FONT_DIR / "Arial.ttf")
FONT_BOLD = str(FONT_DIR / "Arial Bold.ttf")
FONT_ITALIC = str(FONT_DIR / "Arial Italic.ttf")

class ProPDF(FPDF):
    def __init__(self):
        super().__init__()
        # Register Fonts for Unicode Support
        self.add_font('ArialCustom', '', FONT_REGULAR, uni=True)
        self.add_font('ArialCustom', 'B', FONT_BOLD, uni=True)
        self.add_font('ArialCustom', 'I', FONT_ITALIC, uni=True)
    
    def header(self):
        # Clean header, just space
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('ArialCustom', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Sayfa {self.page_no()}', 0, 0, 'C')

    def add_main_title(self, title):
        self.set_font('ArialCustom', 'B', 16) # Match chapter title size (16)
        self.set_text_color(*THEME_ORANGE)
        self.multi_cell(0, 10, title, 0, 'L')
        self.ln(2)

    def chapter_title(self, title):
        self.ln(5)
        self.set_font('ArialCustom', 'B', 16)
        self.set_text_color(*THEME_ORANGE)
        self.multi_cell(0, 8, title)
        self.ln(3)

    def chapter_subtitle(self, title):
        self.ln(4)
        self.set_font('ArialCustom', 'B', 13)
        self.set_text_color(*THEME_ORANGE)
        self.multi_cell(0, 6, title)
        self.ln(2)
        
    def _render_rich_text(self, text):
        """Helper to render text with **bold** and *italic* support."""
        # Regex to split by **...** OR *...*
        parts = re.split(r'(\*\*.*?\*\*|\*.*?\*)', text)
        
        for part in parts:
            if part.startswith('**') and part.endswith('**') and len(part) >= 4:
                # Bold
                content = part[2:-2]
                self.set_font('ArialCustom', 'B', 10)
                self.write(5, content)
            elif part.startswith('*') and part.endswith('*') and len(part) >= 2:
                # Italic
                content = part[1:-1]
                self.set_font('ArialCustom', 'I', 10)
                self.write(5, content)
            else:
                # Normal
                if part:
                    # Inherit current style (could be italic for quotes, but we want standard logic here)
                    # We assume base font is set before calling this, but we need to switch variants.
                    # To be safe, we switch to Regular, but for Quotes we might want 'I'.
                    # Let's check current font style? FPDF doesn't make it super easy to "pop" stack.
                    # We will force Regular for body text parts.
                    self.set_font('ArialCustom', '', 10)
                    self.write(5, part)

    def write_markdown_line(self, text, prefix=""):
        # Reset to body font default
        self.set_text_color(0)
        self.set_font('ArialCustom', '', 10)
        
        # If list item indent
        if prefix:
             self.set_x(15) 
             self.set_font('ArialCustom', 'B', 10)
             self.write(5, prefix + " ")
             # Revert to normal for body
             self.set_font('ArialCustom', '', 10)
        
        self._render_rich_text(text)
        self.ln(6)

    def quote_block(self, text):
        # Professional Quote Style: Indent + Dark Grey text + Vertical Line
        original_margin = self.l_margin
        
        # Draw vertical gray line
        self.set_draw_color(200, 200, 200)
        self.set_line_width(1)
        
        x = self.get_x()
        y_start = self.get_y()
        
        # Process lines
        lines = text.split('\n')
        
        # Indent text
        indent = 10
        self.set_left_margin(original_margin + indent)
        self.set_text_color(80, 80, 80) # Dark Grey
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Use rich text rendering
            # We want the base to be slightly italic or just normal? 
            # Let's keep normal for readability, but grey.
            self._render_rich_text(line)
            self.ln(5)
            
        # Draw line after we know height? No, we need start and end Y.
        y_end = self.get_y()
        self.line(original_margin + 2, y_start, original_margin + 2, y_end - 1)
        
        # Reset
        self.set_left_margin(original_margin)
        self.set_text_color(0)
        self.set_draw_color(0)
        self.ln(3)

def create_pdf(source_file, output_file):
    pdf = ProPDF()
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.add_page()
    
    # Add Main Title explicitly
    pdf.add_main_title("Banka Dokümanları Uyum ve Denetim Rapor Portalı")
    
    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    current_quote_buffer = []

    for line in lines:
        # NO TRANSLITERATION - Keep Turkish Chars
        line_clean = line.strip()
        
        if not line_clean:
            # Flush quote buffer on empty line
            if current_quote_buffer:
                pdf.quote_block("\n".join(current_quote_buffer))
                current_quote_buffer = []
            continue
            
        # FILTER SEPARATORS
        if line_clean == '---' or line_clean == '...':
            continue

        # Quote handling
        if line_clean.startswith('>'):
            quote_content = line_clean.replace('> ', '').replace('>', '').strip()
            current_quote_buffer.append(quote_content)
            continue
        else:
            if current_quote_buffer:
                pdf.quote_block("\n".join(current_quote_buffer))
                current_quote_buffer = []

        # Headers
        if line_clean.startswith('## '):
            pdf.chapter_title(line_clean.replace('## ', ''))
        elif line_clean.startswith('### '):
            pdf.chapter_subtitle(line_clean.replace('### ', ''))
        
        # Lists
        elif line_clean.startswith('* ') or line_clean.startswith('- '):
            content = line_clean[2:]
            pdf.write_markdown_line(content, prefix="-")
            
        # Numbered Lists
        elif (line_clean[0].isdigit() and line_clean[1] == '.') or (line_clean[0].isalpha() and line_clean[1] == '.' and len(line_clean) > 2 and line_clean[2] == ' '):
            parts = line_clean.split(' ', 1)
            if len(parts) == 2:
                pdf.write_markdown_line(parts[1], prefix=parts[0])
            else:
                 pdf.write_markdown_line(line_clean)
        
        # Ignore Main Title (#) as it is in Header
        elif line_clean.startswith('# '):
            continue
            
        # Normal Text
        else:
             pdf.write_markdown_line(line_clean)
            
    if current_quote_buffer:
        pdf.quote_block("\n".join(current_quote_buffer))
        
    pdf.output(output_file)
    print(f"✅ PDF Created: {output_file}")

if __name__ == "__main__":
    source = Path("PROJE_CALISMA_MANTIGI_DETAYLI.txt")
    output = Path("docs/Proje_Calisma_Mantigi_Detayli.pdf")
    create_pdf(source, output)
