
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from config import (
    DATA_FOLDER, CHROMA_FOLDER, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME,
    TEMPERATURE, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS, CONFIDENCE_THRESHOLD,
)
from prompts import SYSTEM_PROMPT


def load_documents():
    documents = []
    for filename in os.listdir(DATA_FOLDER):
        filepath = os.path.join(DATA_FOLDER, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        else:
            continue
        documents.extend(loader.load())
    return documents


def build_rag_chain():
    documents = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_model, persist_directory=CHROMA_FOLDER)
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

    llm = ChatGroq(model=LLM_MODEL_NAME, temperature=TEMPERATURE)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{input}")])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain, vector_store


def ask_support(rag_chain, vector_store, query: str) -> dict:
    try:
        results_with_scores = vector_store.similarity_search_with_score(query, k=TOP_K_RESULTS)
        best_score = results_with_scores[0][1] if results_with_scores else None
        is_confident = best_score is not None and best_score < CONFIDENCE_THRESHOLD

        response = rag_chain.invoke({"input": query})
        source_snippet = response["context"][0].page_content if response.get("context") else ""

        return {"answer": response["answer"], "source": source_snippet, "confident": is_confident}
    except Exception as e:
        return {"answer": f"Error processing request: {e}", "source": "", "confident": False}
