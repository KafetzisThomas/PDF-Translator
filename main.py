import pymupdf

doc = pymupdf.open("test.pdf")

result = ""
for page in doc:
    result += page.get_text()

print(result[:1000])
