#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Project Title: PDF-Translator (https://github.com/KafetzisThomas/PDF-Translator)
# Author / Project Owner: KafetzisThomas (https://github.com/KafetzisThomas)
# License: MIT

import sys
import torch
import nltk
import pdfplumber
from nltk.tokenize import sent_tokenize
from transformers import MarianMTModel, MarianTokenizer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from textwrap import wrap

nltk.download("punkt")
nltk.download("punkt_tab")


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a pdf file.
    """
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(
            [page.extract_text() for page in pdf.pages if page.extract_text()]
        )


def split_text(text):
    """
    Split text into chunks (sentences).
    """
    return sent_tokenize(text.strip())


def translate_text(sentences, batch_size=5):
    """
    Translate sentences using a pre-trained model.
    """
    translations = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

        outputs = model.generate(**inputs)
        translations.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return translations


def save_to_pdf(translated_text, output_pdf):
    """
    Save translated sentences to a pdf file.
    """
    c = canvas.Canvas(output_pdf, pagesize=letter)

    # Set DejaVu as font (supports multiple languages)
    pdfmetrics.registerFont(
        TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    )
    c.setFont("DejaVu", 12)

    y = 750  # Start position
    for sentence in translated_text:
        wrapped_lines = wrap(sentence.strip(), width=70)
        for line in wrapped_lines:
            c.drawString(50, y, line)
            y -= 20  # Move down per line
            if y < 50:  # New page if needed
                c.showPage()
                y = 750

    c.save()


if __name__ == "__main__":
    try:
        input_pdf_path = sys.argv[1]
        input_text = extract_text_from_pdf(input_pdf_path)
        print("\nOriginal Text:")
        print(input_text)

        chunks = split_text(input_text)

        model_name = sys.argv[3]  # for specific language translation
        print(f"\nLoading translation model: {model_name}")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        # Move model to GPU if available
        if torch.cuda.is_available():
            model.to("cuda")
            print("Using GPU for translation.")
        else:
            print("GPU not available. Using CPU for translation.")

        translated_chunks = translate_text(chunks)
        print("\nTranslated Text:")
        print(translated_chunks)

        output_pdf_path = sys.argv[2]
        save_to_pdf(translated_chunks, output_pdf_path)
        print(f"\nTranslated text saved to: {output_pdf_path}")
    except IndexError:
        print(
            "\nUsage: python3 main.py <input_pdf_path> <output_pdf_path> <translation_model>"
        )
        sys.exit()
    except FileNotFoundError:
        print(f"Error: The file '{input_pdf_path}' was not found.")
        sys.exit()
