import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    PyPDFLoader,
    )
# from .config import Config

def load_text_file():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(
            b"Hello, ths is a sapmle text file.\nThis file is used to demonstrate"
        )
        temp_file_path = temp_file.name
    
    try:
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        print(f"loaded {len(documents)} documents(s)")
        print(f"Content preview: {documents[0].page_content[:100]}...")
        print(f"Metadata: { documents[0].metadata}")

        for doc in documents:
            print("Document Content:")
            print(doc)
            print(doc.page_content)

    finally:
        os.remove(temp_file_path)


def pdf_loader(pdf_path:str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} document (s) from PDF")
    for i, doc in enumerate(documents):
        print(f"Document {i+1} Content Preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")

    

if __name__ == "__main__":
    # load_text_file()
    pdf_loader("./docs/langchain_demo.pdf")
