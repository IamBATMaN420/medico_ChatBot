import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

def main():
    st.set_page_config(page_title="Elevana", layout="wide")
    
    with st.sidebar:
        st.header("Chatbot Settings")
        st.write("Adjust the chatbot preferences here.")
        st.text_input("HuggingFace Token", type="password")
        st.selectbox("Model", ["Mistral-7B-Instruct-v0.3", "GPT-3.5", "Llama-2"])
    
    st.title("🧠 Ask Elevana!")
    st.markdown("""
    <style>
        .stChatMessage {border-radius: 15px; padding: 15px; margin: 10px 0; transition: all 0.3s ease-in-out;}
        .stChatMessage.user {background-color: #2b6cb0; color: white; animation: fadeIn 0.5s ease-in-out;}
        .stChatMessage.assistant {background-color: #f0f4ff; color: black; animation: fadeIn 0.5s ease-in-out;}
        .answer {background-color: #1e293b; color: white; padding: 20px; border-radius: 10px; margin-top: 15px; font-weight: bold; transition: all 0.3s ease-in-out;}
        .source {background-color: #374151; color: white; padding: 20px; border-radius: 10px; margin-top: 10px; font-style: italic; transition: all 0.3s ease-in-out;}
        @keyframes fadeIn {
            from {opacity: 0; transform: translateY(10px);}
            to {opacity: 1; transform: translateY(0);}
        }
    </style>
    """, unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        role_class = "user" if message['role'] == 'user' else "assistant"
        st.markdown(f'<div class="stChatMessage {role_class}">{message["content"]}</div>', unsafe_allow_html=True)

    prompt = st.chat_input("Type your message here...")

    if prompt:
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.markdown(f'<div class="stChatMessage user">{prompt}</div>', unsafe_allow_html=True)

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        Don't provide anything outside of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """
        
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]
            
            st.markdown(f'<div class="answer">{result}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="source"><strong>Source Docs:</strong><br>{str(source_documents)}</div>', unsafe_allow_html=True)
            
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
