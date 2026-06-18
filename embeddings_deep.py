from langchain_mistralai.embeddings import MistralAIEmbeddings
from config import Config
import numpy as np


embeddings = MistralAIEmbeddings(model="mistral-embed-2312", api_key=Config.mistral_api_key)

def basic_embeddings():
    text = "What is Machine Learning?"
    single_embedding = embeddings.embed_query(text)
    print(f"Vector Dimendions: {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")
    print(f"Vector norm: {np.linalg.norm(single_embedding): .4f}")


if __name__ == "__main__":
    basic_embeddings()