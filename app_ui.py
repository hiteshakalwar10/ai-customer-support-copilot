
import streamlit as st
from rag_pipeline import build_rag_chain, ask_support

st.set_page_config(page_title="AI Customer Support Copilot", page_icon="🤖", layout="centered")
st.title("🤖 AI Customer Support Copilot")
st.caption("Ask a question — answers are grounded in our official documentation.")

@st.cache_resource
def load_pipeline():
    return build_rag_chain()

rag_chain, vector_store = load_pipeline()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for question, answer, source, confident in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        if confident:
            st.success(answer)
        else:
            st.warning(answer)
        if source:
            with st.expander("View source"):
                st.write(source)

user_question = st.chat_input("Type your question here...")
if user_question:
    result = ask_support(rag_chain, vector_store, user_question)
    st.session_state.chat_history.append((user_question, result["answer"], result["source"], result["confident"]))
    st.rerun()
