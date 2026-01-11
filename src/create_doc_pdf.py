from fpdf import FPDF
from pathlib import Path
import re

# Theme Colors (Orange for titles, no explicit ING name in code)
THEME_ORANGE = (255, 98, 0) 
DARK_GREY = (50, 50, 50)
LIGHT_GREY = (245, 245, 245)

class ProPDF(FPDF):
    def header(self):
        self.set_font('Arial', '', 9)
        self.set_text_color(100, 100, 100)
        # Clean text-only header
        self.cell(0, 10, 'Projeyi Calisma Mantigi ve Teknik Detaylar', 0, 0, 'R')
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Sayfa {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.ln(5)
        self.set_font('Arial', 'B', 16)
        self.set_text_color(*THEME_ORANGE)
        self.multi_cell(0, 8, title)
        self.ln(3)

    def chapter_subtitle(self, title):
        self.ln(4)
        self.set_font('Arial', 'B', 13)
        self.set_text_color(*THEME_ORANGE)
        self.multi_cell(0, 6, title)
        self.ln(2)
        
    def write_markdown_line(self, text, prefix=""):
        # Reset to body font default for this line
        self.set_text_color(0)
        
        # If list item indent
        if prefix:
             self.set_x(15) 
             self.set_font('Arial', 'B', 10)
             self.write(5, prefix + " ")
        
        # Parsing **bold** 
        # Split by **...**
        parts = re.split(r'(\*\*.*?\*\*)', text)
        
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                # Bold content
                content = part[2:-2]
                self.set_font('Arial', 'B', 10)
                self.write(5, content)
            else:
                # Normal content
                self.set_font('Arial', '', 10)
                self.write(5, part)
        
        self.ln(6)

    def quote_block(self, text):
        self.set_fill_color(*LIGHT_GREY)
        self.set_font('Courier', '', 9)
        self.set_text_color(60, 60, 60)
        # Use multi_cell for block
        self.multi_cell(0, 5, text, 0, 'L', True)
        self.ln(2)

def clean_text(text):
    """
    Transliterate Turkish characters to Latin-1 compatible equivalents
    """
    replacements = {
        'ğ': 'g', 'Ğ': 'G',
        'ü': 'u', 'Ü': 'U',
        'ş': 's', 'Ş': 'S',
        'ı': 'i', 'İ': 'I',
        'ö': 'o', 'Ö': 'O',
        'ç': 'c', 'Ç': 'C',
        '–': '-', 
        '’': "'", 
        '“': '"', 
        '”': '"', 
        '…': '...',
        '●': '-',
    }
    for search, replace in replacements.items():
        text = text.replace(search, replace)
    return text

def create_pdf(source_file, output_file):
    pdf = ProPDF()
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.add_page()
    
    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    current_quote_buffer = []

    for line in lines:
        raw_line = line.strip()
        line_clean = clean_text(raw_line)
        
        if not line_clean:
            # Empty line, if we have buffered quote, flush it
            if current_quote_buffer:
                pdf.quote_block("\n".join(current_quote_buffer))
                current_quote_buffer = []
            continue

        # Quote handling
        if line_clean.startswith('>'):
            quote_content = line_clean.replace('> ', '').replace('>', '').strip()
            current_quote_buffer.append(quote_content)
            continue
        else:
            # If we had a quote buffer, flush it now as we met non-quote line
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
            
        # Numbered Lists (Simple check for "1. ", "A. ")
        elif (line_clean[0].isdigit() and line_clean[1] == '.') or (line_clean[0].isalpha() and line_clean[1] == '.' and len(line_clean) > 2 and line_clean[2] == ' '):
            # Split "1. Text" -> prefix="1.", text="Text"
            parts = line_clean.split(' ', 1)
            if len(parts) == 2:
                pdf.write_markdown_line(parts[1], prefix=parts[0])
            else:
                 pdf.write_markdown_line(line_clean)
        
        # Normal Text
        else:
             pdf.write_markdown_line(line_clean)
            
    # Flush remaining quotes
    if current_quote_buffer:
        pdf.quote_block("\n".join(current_quote_buffer))
        
    pdf.output(output_file)
    print(f"✅ PDF Created: {output_file}")

if __name__ == "__main__":
    source = Path("PROJE_CALISMA_MANTIGI_DETAYLI.txt")
    output = Path("docs/Proje_Calisma_Mantigi_Detayli.pdf")
    create_pdf(source, output)
