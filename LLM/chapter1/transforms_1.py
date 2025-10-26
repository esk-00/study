from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("abeja/gpt2-large-japanese", legacy=False)
print(tokenizer.tokenize("今日は天気が良いので"))
