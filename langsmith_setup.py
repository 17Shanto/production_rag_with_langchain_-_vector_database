import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from langsmith.run_trees import RunTree
from dotenv import load_dotenv
from config import Config

# langsmith_tracing = Config.langsmith_tracing
# langsmith_tracing = True
load_dotenv()

# Enable tracing
os.environ["LANGSMITH_TRACING"] = "true"

@traceable(name="basic_chaining")
def demo_basic_tracing():
    """Basic LangSmith Tracing"""

    llm = ChatMistralAI(model_name="mistral-large-latest", api_key=Config.mistral_api_key, temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in One Sentence."
    )

    chain = prompt | llm | StrOutputParser()

    print("Basic Tracing Demo:\n")
    print("Running chain with LangSmith tracing enabled...")

    result = chain.invoke({"topic": "machine learning"})
    print(f"Result: {result}")
    print("\nCheck LangSmith dashboard for tracing details.")


@traceable(name="name_runs_demo", tags=["production", "summarization"])
def demo_name_runs():
    """Name your runs for easier identification."""

    llm = ChatMistralAI(model_name="mistral-large-latest", api_key=Config.mistral_api_key, temperature=0)
    prompt = ChatPromptTemplate.from_template("Summarize: {text}")
    chain = prompt | llm | StrOutputParser()
    print("\nNamed Runs Demo:\n")

    result = chain.invoke(
        {"text": "LangSmith provides observability for LLM applications."}
    )

    print(f"Result: {result}")
    print("Run tagged with 'production', 'summarization'")


@traceable(name="trace_with_metadata_demo", tags=["metadata", "filtering"])
def demo_trace_with_metadata(user_id: str, request_type: str):
    """Add metadata to traces for filtering."""

    llm = ChatMistralAI(model_name="mistral-large-latest", api_key=Config.mistral_api_key,temperature=0)

    # Metadata is automatically captured
    result = llm.invoke(f"Hello from user {user_id}")

    return result.content



if __name__ == "__main__":
    demo_basic_tracing()
    demo_name_runs()
    demo_trace_with_metadata(user_id="user_123", request_type="greetings")