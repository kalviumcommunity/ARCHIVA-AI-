from app.retrieval.search import search

query = "Have we solved duplicate payment processing before?"

results = search(query, top_k=3)

for result in results:
    print("\nTitle:", result["title"])
    print("Type:", result["document_type"])
    print("Score:", round(result["score"], 4))
    print("Content:", result["content"])