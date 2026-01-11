from fpdf import FPDF
from pathlib import Path

class PDF(FPDF):
    def header(self):
        # Logo could go here
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Projeyi Calisma Mantigi ve Teknik Detaylar', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Sayfa {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, body)
        self.ln()

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
    }
    for search, replace in replacements.items():
        text = text.replace(search, replace)
    return text

def create_pdf(source_file, output_file):
    pdf = PDF()
    pdf.add_page()
    
    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    final_text = []
    
    # Simple parser to handle headers vs body
    current_body = ""
    
    for line in lines:
        line_clean = clean_text(line.strip())
        
        if not line_clean:
            continue
            
        if line_clean.startswith('# '):
            # Main Title - already handled by header mostly, but let's add it if strictly needed or skip
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
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 10, title, 0, 1, 'L')
            
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
