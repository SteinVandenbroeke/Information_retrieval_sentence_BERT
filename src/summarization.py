from transformers.agents.evaluate_agent import summarizer
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Summarization:
    def __init__(self, tokenizer, summarizer):
        self.tokenizer = tokenizer
        self.summarizer = summarizer

    def summarize_large_document(self, document):

        # Tokenize the input text
        inputs = self.tokenizer(document, return_tensors="pt", max_length=16384, truncation=True)

        # Generate a summary with a maximum of 512 tokens
        summary_ids = self.summarizer.generate(
            inputs["input_ids"],
            max_length=512,
            min_length=50,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )

        # Decode and print the summary
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)