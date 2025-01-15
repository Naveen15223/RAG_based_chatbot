import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Load API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Set up Streamlit page configuration
st.set_page_config(page_title="University Chatbot", layout="wide")

# Streamlit app title
st.title("University Chatbot")

# Initialize language model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Multiple prompts for different questions and summarization
prompts = [
    ChatPromptTemplate.from_template(
        """
        Answer the question based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Question: {input}
        """
    ),
    ChatPromptTemplate.from_template(
        """
        Provide detailed information based on the given context.
        Make sure the response is accurate and relevant to the question.
        <context>
        {context}
        <context>
        Query: {input}
        """
    ),
    ChatPromptTemplate.from_template(
        """
        Summarize the information relevant to the following question based on the provided context.
        <context>
        {context}
        <context>
        Summary of the answer: {input}
        """
    ),
]

# Function to load PDF documents and create embeddings
def vector_embedding():
    if "vectors" not in st.session_state:
        # Load documents from the specified directory
        st.session_state.loader = PyPDFDirectoryLoader("./BU")  # Update the path to your PDF folder
        st.session_state.docs = st.session_state.loader.load()

        # Split documents into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        # Create embeddings and vector store
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Input for user questions
user_question = st.text_input("Ask a question:")

# Button to get the answer and summary
if st.button("Get the answer"):
    vector_embedding()

    # Choose the prompt for retrieving information
    document_prompt = prompts[0]

    # Create document chain and retrieval chain
    document_chain = create_stuff_documents_chain(llm, document_prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Process the user question
    if user_question:
        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_question})
        st.write(f"Response time: {time.process_time() - start} seconds")

        # Display the detailed response
        detailed_answer = response['answer']
        st.write("Detailed Answer:")
        st.write(detailed_answer)

        # Summarize the response using LLM
        summary_prompt = prompts[2].format(input=detailed_answer, context=response['context'])
        summary_response = llm({'input': summary_prompt})
        st.write("Summary of the Answer:")
        st.write(summary_response['answer'])

        # With a Streamlit expander for document similarity search
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response.get("context", [])):
                st.write(doc.page_content)
                st.write("--------------------------------")

# Main function to run the Streamlit app
def main():
    st.title("University Chatbot")
    
    # Call the vector embedding function to prepare the data
    vector_embedding()

    # Input for user's question
    user_input = st.text_input("Ask a question about the university:", value="")

    # Handle user input
    if st.button("Send"):
        if user_input.strip():
            document_prompt = prompts[1]
            document_chain = create_stuff_documents_chain(llm, document_prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response = retrieval_chain.invoke({'input': user_input})
            detailed_answer = response['answer']
            st.write("Detailed Answer:")
            st.write(detailed_answer)

            # Summarize the detailed answer using LLM
            summary_prompt = prompts[2].format(input=detailed_answer, context=response['context'])
            summary_response = llm({'input': summary_prompt})
            st.write("Summary of the Answer:")
            st.write(summary_response['answer'])

            # Display document similarity search results
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response.get("context", [])):
                    st.write(doc.page_content)
                    st.write("--------------------------------")

# Run the Streamlit app
if __name__ == "__main__":
    main()