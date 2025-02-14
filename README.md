# PDF-Translator

**What is this?**  
A tool that translates pdf files using pre-trained language models. It supports various models, such as those from Helsinki-NLP.

## Usage

```bash
➜ input_pdf_path=media/test.pdf  # Path to pdf file
➜ output_pdf_path=media/translated.pdf  # Path to output file
➜ translation_model=Helsinki-NLP/opus-mt-de-en  # https://huggingface.co/Helsinki-NLP

$ pip3 install -r requirements.txt
$ python3 main.py '$input_pdf_path' '$output_pdf_path' '$translation_model'
```

## How It Works

1. Extract text from a pdf file.
2. Split text into chunks (sentences).
3. Translate sentences using a pre-trained model.
4. Save translated sentences to a new pdf file.
