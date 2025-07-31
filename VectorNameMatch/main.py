import os
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient

# Setup Persistent Chroma Client (stores data in current path)
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "chroma_db_storage")

client = PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))

# Get or Create Collection
collection = client.get_or_create_collection(name="person_names")

# Sample Names to Add (only if not already added)
names_to_add = ["Geetha", "Gita", "Gitu", "Geethu", "Gitanjali", "Anjali", "Anuj", "Ajit", "Gopal", "Gaurav", "Sheetal"]
ids = [f"name_{i}" for i in range(len(names_to_add))]

if collection.count() == 0:
    collection.add(documents=names_to_add, ids=ids)
    print("Added names to persistent vector DB.")
else:
    print("Vector DB already contains data. Skipping addition.")

# Search Function
def search_name(user_input, top_k=5):
    results = collection.query(query_texts=[user_input], n_results=top_k)
    matched_names = results['documents'][0]
    scores = results['distances'][0]  # Lower is better
    return list(zip(matched_names, scores))

# Run Search
if __name__ == "__main__":
    user_input = input("Enter a name to search: ")
    matches = search_name(user_input, top_k=5)

    print("\nTop Match:")
    print(f"{matches[0][0]} (score: {matches[0][1]:.4f})")

    print("\nAll Matches:")
    for name, score in matches:
        print(f"{name} (score: {score:.4f})")
