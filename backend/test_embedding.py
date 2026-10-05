from app.retrieval.embeddings import generate_embedding

text = "We introduced idempotency keys to prevent duplicate payment processing."

embedding = generate_embedding(text)

print("Embedding dimensions:", len(embedding))
print("First 5 values:", embedding[:5])