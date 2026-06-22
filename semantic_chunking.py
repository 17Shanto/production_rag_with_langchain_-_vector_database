import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

embeddings = MistralAIEmbeddings(
    model="mistral-embed-2312"
)

document = '''
#Authentication Guide

##OAuth2 Authentication
To authenticate with our API, you need OAuth2 credentials.
First, obtain a client_id and client_secret from the developer portal.
Make a POST request to /oauth/token with grant_type=client_credentials.
The response contains an access_token valid for 3600 seconds.
Include this token in the Authorization header as 'Bearer '.

##Rate Limiting
Our API implements rate limiting using a token bucket algorithm.
Free tier: 100 requests per minute.
Pro tier: 1000 requests per minute.
Enterprise tier: Custom limits.
When rate limited, you receive a 429 status code.
The Retry-After header indicates when to retry.

##Error Handling
All errors return a standard JSON format.
The 'code' field contains a machine-readable error code.
The 'message' field contains a human-readable description.
Common errors: AUTH_FAILED, RATE_LIMITED, INVALID_REQUEST.
Always check the HTTP status code first, then parse the error body.

##Webhooks
Configure webhooks in your dashboard settings.
We support HTTP and HTTPS endpoints.
Webhook payloads are signed with HMAC-SHA256.
Verify signatures using your webhook secret.
Failed deliveries are retried with exponential backoff.
'''

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=['\n\n', '\n', '. ', ' ']
)


recursive_chunks = recursive_splitter.split_text(document)

# print(f"Recursive chunks: {len(recursive_chunks)}")
# for i, chunk in enumerate(recursive_chunks):
#     print(f"\\n--- Chunk {i+1} ({len(chunk)} chars) ---")
#     print(chunk[:100] + "..." if len(chunk) > 100 else chunk)


semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=90
)

semantic_chunks = semantic_chunker.split_text(document)
# print(f"Semantic Chunks: {len(semantic_chunks)}")
# for i, chunk in enumerate(recursive_chunks):
#     print(f"\\n--- Chunk {i+1} ({len(chunk)} chars) ---")
#     print(chunk[:100] + "..." if len(chunk) > 100 else chunk)


recursive_vectorstore = Chroma.from_texts(
    recursive_chunks,
    embeddings,
    collection_name='recursive_chunks'
)

semantic_vectorstore = Chroma.from_texts(
    semantic_chunks,
    embeddings,
    collection_name='semantic_chunks'
)


# Test Queries
test_queries = [
    'How do I authenticate with OAuth2?',
    'What happens when I hit the rate limit?',
    'How are webhooks secured?',
    'What format are errors returned in?'
]

def test_retrieval(query,vectorstore, name):
    results = vectorstore.similarity_search(query,k=3)
    print(f'\\n{name} - Query: "{query}')
    # print(f"Retrieved: {results[0].page_content[:150]}...")
    print(f"Retrieved: {results[0].page_content}")
    return results[0].page_content


if __name__ == '__main__':
    test_retrieval('How do I authenticate with OAuth2?',recursive_vectorstore, "semantic_chunking")
    # pass