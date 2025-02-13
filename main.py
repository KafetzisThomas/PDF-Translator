import nltk
import pdfplumber
from nltk.tokenize import sent_tokenize
from transformers import MarianMTModel, MarianTokenizer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

nltk.download("punkt")
nltk.download("punkt_tab")


def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(
            [page.extract_text() for page in pdf.pages if page.extract_text()]
        )


print("\nOriginal Text:")
result = extract_text_from_pdf("media/test.pdf")
print(result)


def split_text(text):
    """
    Split into sentences.
    """
    return sent_tokenize(text.strip())


chunks = split_text(result)
# print(type(chunks))


model_name = "Helsinki-NLP/opus-mt-en-el"  # for specific language translation
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


def translate_text(sentences, batch_size=5):
    translations = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs)
        translations.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return translations


print("\nTranslated Text:")
translated_chunks = translate_text(chunks)
print(translated_chunks)


def save_to_pdf(translated_text, output_pdf):
    c = canvas.Canvas(output_pdf, pagesize=letter)

    # Register and set a font that supports greek characters
    pdfmetrics.registerFont(
        TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    )
    c.setFont("DejaVu", 12)

    y = 750  # Start position
    for sentence in translated_text:
        c.drawString(50, y, sentence.strip())
        y -= 20  # Move down per line
        if y < 50:  # New page if needed
            c.showPage()
            y = 750

    c.save()


save_to_pdf(translated_chunks, "translated.pdf")
