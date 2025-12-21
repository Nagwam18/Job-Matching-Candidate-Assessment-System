import os
import pdfplumber
import pdfplumber
import pytesseract
import docx2txt
import re
from PIL import Image

def load_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext==".pdf":
        text=""
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: text+=t+"\n"
        if text.strip():
            return text
    
        ocr=""
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                img=p.to_image(resolution=300).original
                ocr+=pytesseract.image_to_string(img)
        return ocr
    if ext==".docx":
        return docx2txt.process(path) or ""
    if ext==".txt":
        return open(path,"r",encoding="utf-8", errors="ignore").read()
    if ext in [".jpg",".jpeg",".png"]:
        return pytesseract.image_to_string(Image.open(path))
    return ""
