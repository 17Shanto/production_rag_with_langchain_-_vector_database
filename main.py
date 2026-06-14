from typing import cast
from pydantic import SecretStr
import os
from dotenv import load_dotenv
from langchain_core import __version__ as core_version
from importlib.metadata import version
from langchain_mistralai import ChatMistralAI
from langchain_anthropic import ChatAnthropic

lg_version = version("langgraph")
load_dotenv()
mistral_api_key = SecretStr(os.getenv("MISTRAL_API_KEY", ""))
anthropic_api_key= SecretStr(os.getenv("ANTHROPIC_API_KEY",""))
mistral_large_latest = cast(str,os.getenv("MODEL_NAME"))

print(f"LangChain-core version a: {core_version}")
print(f"LangGraph versions: {lg_version}")


def main():
    # print("Hello from production-rag-with-langchain-vector-database!")

    try:

        mistral_llm = ChatMistralAI(model_name=mistral_large_latest, api_key=mistral_api_key, temperature=0)
        response = mistral_llm.invoke("Say 'setup complete!' in one word")
        print(f"Response from ChatOpenAI: {response}")

        # Test Anthropic
        # llm_anthropic = ChatAnthropic(model_name="claude-haiku-4-5", api_key=anthropic_api_key, temperature=0, timeout=120, stop=["\n"])
        # response_anthropic = llm_anthropic.invoke("Say 'setup complete!' in one word")
        # print(f"Response from ChatAnthropic: {response_anthropic}")

    except Exception as e:
        print(f"{str(e)}")

    print("Setup Complete")


if __name__ == "__main__":
    main()


