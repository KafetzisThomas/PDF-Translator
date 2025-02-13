import nltk
import pymupdf
from nltk.tokenize import sent_tokenize
from transformers import MarianMTModel, MarianTokenizer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

nltk.download("punkt")
nltk.download("punkt_tab")

doc = pymupdf.open("media/test.pdf")

result = ""
for page in doc:
    result += page.get_text()

print("Original Text:\n")
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


print("Translated Text:\n")
translated_chunks = translate_text(chunks)
print(translated_chunks)


def save_to_pdf(translated_text, output_pdf):
    c = canvas.Canvas(output_pdf)
    c.setFont("Helvetica", 12)

    y = 750  # Start position
    for sentence in translated_text:
        c.drawString(50, y, sentence.encode("utf-8").decode("utf-8"))
        y -= 20  # Move down per line
        if y < 50:  # New page if needed
            c.showPage()
            y = 750

    c.save()


save_to_pdf(translated_chunks, "translated.pdf")
