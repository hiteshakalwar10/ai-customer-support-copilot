# AI Customer Support Copilot

An AI-powered customer support assistant that answers customer queries using a RAG (Retrieval-Augmented Generation) pipeline. Instead of relying on the model's own guesses, it retrieves relevant information from actual support documentation and generates answers grounded in that content.

## What it does

The assistant is built on top of a real-world e-commerce support knowledge base covering returns, cancellations, shipping, the rewards program, warranty, and payments. When a customer asks a question, it retrieves the most relevant chunks from this documentation and generates an answer based only on that context. If a question falls outside the scope of the knowledge base, it says so instead of making something up.

## Tech stack

- Python
- LangChain (RAG pipeline)
- ChromaDB (vector store)
- Sentence Transformers (embeddings)
- Groq (LLaMA-3.1 for response generation)
- Streamlit (UI)
- Streamlit Community Cloud (deployment)

## How it works

1. Support documentation is loaded and split into chunks
2. Each chunk is converted into embeddings using Sentence Transformers
3. Embeddings are stored in ChromaDB
4. When a query comes in, the most relevant chunks are retrieved
5. Groq's LLaMA-3.1 model generates an answer using only those chunks as context
6. If the answer isn't found in the retrieved context, the assistant says it doesn't have that information instead of guessing

## Running it locally

```
pip install -r requirements.txt
streamlit run app_ui.py
```

You'll need a Groq API key (free at console.groq.com/keys) set as an environment variable:

```
GROQ_API_KEY=your_key_here
```

## Live demo

https://ai-customer-support-copilot-ci5cnjkgcazqsub3fursgw.streamlit.app/
