from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
from config import Config
import tempfile

# Sample Knowledge Base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""


if not Config.mistral_api_key:
    raise ValueError("Mistral API key is missing! Please check your .env file or Config class.")

embeddings_model = MistralAIEmbeddings(api_key=Config.mistral_api_key, model="mistral-embed-2312")


def create_kb():
    """Create a vector store from knowledge base."""
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    doc = Document(page_content= KNOWLEDGE_BASE, metadata={"source": "langchain_knowledge_base.md"})

    chunks = splitter.split_documents([doc])
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding= embeddings_model,
        persist_directory=tempfile.mkdtemp()
    )
    return vector_store


def demo_basic_rag():
    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type ="similarity", search_kwargs={"k":2})

    llm = init_chat_model(
        api_key = Config.mistral_api_key,
        model="mistral-large-latest",
        temperature= 0.2,   
    )

    #RAG Promt Template
    prompt = ChatPromptTemplate.from_template(
        """
            Answer the question based only on the following context:
            {context}\n
            Question: {question}\n
            Make sure to answer in a concise manner, and if you don't know the answer, just say "I don't know"
        """
    )

    def format_docs(docs) -> str:
        return "\n\n".join([doc.page_content for doc in docs])
    
    rag_chain = (
        {"context":retriever | format_docs, "question": RunnablePassthrough()} 
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What is LangChain?",
        "Who created Langchain?",
        "What is LangGraph used for?",
    ]

    try:
        print("Basic RAG Demo:\n")
        # for q in questions:
        answer = rag_chain.invoke("What are the core components of langcchain?")
        print(f"Q: What are the core components of langcchain?")
        print(f"A: {answer}\n")
    except Exception as e:
        print(f"{str(e)}")




if __name__ == "__main__":
    demo_basic_rag()