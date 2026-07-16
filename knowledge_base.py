
import os
from config import DATA_FOLDER


def create_knowledge_base():
    os.makedirs(DATA_FOLDER, exist_ok=True)
    existing_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith((".pdf", ".txt"))]

    if existing_files:
        print(f"Knowledge base already loaded: {existing_files}")
        return

    print("No knowledge base document found. Please upload the company PDF or TXT file.")
    from google.colab import files
    uploaded = files.upload()

    for filename in uploaded.keys():
        os.rename(filename, f"{DATA_FOLDER}/{filename}")

    print("Knowledge base uploaded successfully.")
