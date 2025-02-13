import nltk
import pymupdf
from nltk.tokenize import PunktTokenizer

nltk.download("punkt")
nltk.download("punkt_tab")

doc = pymupdf.open("media/test.pdf")

result = ""
for page in doc:
    result += page.get_text()

print(result[:1000])


def split_text(text):
    """
    Split into sentences.
    """
    sent_detector = PunktTokenizer()
    return sent_detector.tokenize(text.strip())


print("\n")
print(split_text(result[:1000]))
print(type(split_text(result[:1000])))
