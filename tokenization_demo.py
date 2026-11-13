from transformers import GPT2Tokenizer
tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
text="Hello, how are you doing today?"
tokens=tokenizer.tokenize(text)
token_ids=tokenizer.convert_tokens_to_ids(tokens)
print("original text:", text)
print("Tokens:", tokens)
print("Token IDs:", token_ids)
encoded=tokenizer(text)
print("Encoded Output:",encoded)
decoded_text=tokenizer.decode(token_ids)
print("Decoded text:", decoded_text)