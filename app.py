import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(memory):
    prompt_template = """
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "history", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt, memory=memory)
    return chain

def user_input(user_question, memory):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    if not docs:
        # If no relevant documents found, use LLM for response directly
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        response = model(user_question)
        return f"Answer is not available in the PDF. Here's what I know about it: {response}"

    chain = get_conversational_chain(memory)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    return response["output_text"]

def main():
    st.set_page_config(page_title="InsightPDF", page_icon="✨")
    st.title("InsightPDF ✨")
    st.subheader("Chat with your PDFs using Google Generative AI")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="history", input_key="question")

    # Sidebar for PDF upload
    with st.sidebar:
        st.title("Menu:")
        st.info("Upload your PDF files and click on 'Submit & Process' to start.")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete!")
            else:
                st.warning("Please upload at least one PDF file.")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])

    # Chat input for user question
    user_question = st.chat_input("Ask a Question from the PDF Files")
    if user_question:
        st.session_state.chat_history.append({"role": "user", "text": user_question})
        st.chat_message("user").markdown(user_question)
        
        with st.spinner("Generating response..."):
            response = user_input(user_question, st.session_state.memory)
            st.session_state.chat_history.append({"role": "assistant", "text": response})
            st.chat_message("assistant").markdown(response)

if __name__ == "__main__":
    main()
