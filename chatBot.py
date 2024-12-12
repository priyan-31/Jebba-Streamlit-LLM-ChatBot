import streamlit as st
import PyPDF2
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import create_retrieval_chain
import os
import time

# Function to read PDF file and extract text
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() + '\n'
    return text

# Function to create a knowledge base from the extracted text
def create_knowledge_base(text):
    openai_api_key = os.getenv('OPENAI_API_KEY')  # Ensure this environment variable is set
    if not openai_api_key:
        raise ValueError("OpenAI API key not found. Please set it as an environment variable.")

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    chunks = text.split('\n\n')  # Simple chunking by paragraphs
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

# Function to handle API requests with retries for RateLimitError
def safe_run_chain(qa_chain, user_question):
    retries = 5  # Number of retries before failing
    for attempt in range(retries):
        try:
            return qa_chain.run(user_question)
        except Exception as e:
            if "RateLimitError" in str(e):
                wait_time = 2 ** attempt  # Exponential backoff
                st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)  # Wait before retrying
            else:
                st.error(f"An error occurred: {e}")
                break
    return None

# Main Streamlit app logic
def main():
    st.title("PDF Upload and Chatbot")

    # File uploader widget for PDF files
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Read and display the content of the PDF directly from the uploaded file
        text = read_pdf(uploaded_file)
        st.subheader("Extracted Text:")
        st.text_area("Text from PDF", text, height=300)

        # Create a knowledge base from the extracted text
        knowledge_base = create_knowledge_base(text)

        # User input for questions related to the PDF content
        user_question = st.text_input("Ask a question about your PDF:")

        if user_question:
            # Initialize LLM and create retrieval chain
            llm = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))
            retriever = knowledge_base.as_retriever()
            qa_chain = create_retrieval_chain(llm=llm, retriever=retriever)

            # Get answer from QA chain with error handling
            answer = safe_run_chain(qa_chain, user_question)
            if answer:
                st.subheader("Answer:")
                st.write(answer)

if _name_ == '_main_':
    main()