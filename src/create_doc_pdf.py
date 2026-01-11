from fpdf import FPDF
from pathlib import Path

# ING Corporate Colors
ING_ORANGE = (255, 98, 0)
ING_GREY = (51, 51, 51)
ING_LIGHT_GREY = (240, 240, 240)

class PDF(FPDF):
    def header(self):
        # ING Orange Line at top
        self.set_fill_color(*ING_ORANGE)
        self.rect(0, 0, 210, 5, 'F')
        
        # Title
        self.ln(10)
        self.set_font('Arial', 'B', 24)
        self.set_text_color(*ING_ORANGE)
        self.cell(0, 10, 'Proje Calisma Mantigi ve Teknik Detaylar', 0, 1, 'L')
        
        # Subtitle / Line
        self.set_draw_color(*ING_ORANGE)
        self.set_line_width(0.5)
        self.line(10, 35, 200, 35)
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Sayfa {self.page_no()}', 0, 0, 'R')
        
        # ING Orange Line at bottom
        self.set_fill_color(*ING_ORANGE)
        self.rect(0, 292, 210, 5, 'F')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(*ING_ORANGE)
        self.ln(5)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def chapter_subtitle(self, title):
        self.set_font('Arial', 'B', 13)
        self.set_text_color(*ING_GREY)
        self.ln(2)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(1)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, body)
        self.ln()

    def quote_block(self, text):
        self.set_fill_color(*ING_LIGHT_GREY)
        self.set_font('Courier', '', 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5, text, 0, 'L', True)
        self.ln(2)

def clean_text(text):
    """
    Transliterate Turkish characters to Latin-1 compatible equivalents
    because standard FPDF fonts do not support Turkish characters natively
    without external TTF files.
    """
    replacements = {
        'ğ': 'g', 'Ğ': 'G',
        'ü': 'u', 'Ü': 'U',
        'ş': 's', 'Ş': 'S',
        'ı': 'i', 'İ': 'I',
        'ö': 'o', 'Ö': 'O',
        'ç': 'c', 'Ç': 'C',
        '–': '-',  # En-dash to hyphen
        '’': "'",  # Smart quote to straight quote
        '“': '"',  # Smart quote to straight quote
        '”': '"',  # Smart quote to straight quote
        '…': '...',
        '●': '-',
    }
    for search, replace in replacements.items():
        text = text.replace(search, replace)
    return text

def create_pdf(source_file, output_file):
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.add_page()
    
    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    current_body = ""
    
    for line in lines:
        raw_line = line.strip()
        line_clean = clean_text(raw_line)
        
        if not line_clean:
            continue
            
        # Is it a quote or special block?
        if line_clean.startswith('>'):
             # Flush previous body
            if current_body:
                pdf.chapter_body(current_body)
                current_body = ""
            
            quote_text = line_clean.replace('> ', '').replace('>', '')
            pdf.quote_block(quote_text)
            continue
            
        if line_clean.startswith('# '):
            # Main Title check - usually skipped as we have a header
            continue
            
        if line_clean.startswith('## '):
            # Flush previous body
            if current_body:
                pdf.chapter_body(current_body)
                current_body = ""
            
            # Add Title
            title = line_clean.replace('## ', '')
            pdf.chapter_title(title)
        
        elif line_clean.startswith('### '):
             # Flush previous body
            if current_body:
                pdf.chapter_body(current_body)
                current_body = ""
            
            # Add SubTitle
            title = line_clean.replace('### ', '')
            pdf.chapter_subtitle(title)
            
        elif line_clean.startswith('* ') or line_clean.startswith('- '):
             # List Item - Flush body if it was paragraph
             # We treat list items as part of body but lets format them slightly nicely?
             # For now append to body
             current_body += line_clean + "\n"
             
        elif line_clean[0].isdigit() and line_clean[1] == '.':
             # Numbered list
             current_body += line_clean + "\n"
             
        else:
            # Accumulate body text
            current_body += line_clean + "\n"
            
    # Flush last body
    if current_body:
        pdf.chapter_body(current_body)
        
    pdf.output(output_file)
    print(f"✅ PDF Created: {output_file}")

if __name__ == "__main__":
    source = Path("PROJE_CALISMA_MANTIGI_DETAYLI.txt")
    output = Path("docs/Proje_Calisma_Mantigi_Detayli.pdf")
    create_pdf(source, output)
