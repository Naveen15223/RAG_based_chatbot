# Bharathiar University Chatbot (RAG-based)

## Project Overview

This project is a Retrieval-Augmented Generation (RAG) based chatbot designed for Bharathiar University. The chatbot is built using LangChain and Streamlit to provide concise and accurate answers in a summary format based on 72+ PDF documents related to the university. The chatbot covers a wide range of topics, including affiliated colleges, departments, courses, fees, admissions, research institutes, scholarships, and more.

## Features

- **Question-Answering**: Users can ask questions about various aspects of Bharathiar University, and the chatbot provides detailed responses.
- **Document Retrieval**: The chatbot uses vector embeddings to retrieve relevant information from a large corpus of university-related PDFs.
- **Summary Generation**: After retrieving information, the chatbot summarizes the content to provide concise answers.
- **Contextual Search**: Users can explore similar documents and view relevant content through a document similarity search.

## Technologies Used

- **Streamlit**: For building the web interface.
- **LangChain**: For integrating language models and creating retrieval chains.
- **FAISS (Facebook AI Similarity Search)**: For efficient document retrieval.
- **Google Generative AI Embeddings**: For creating embeddings of the document content.
- **OpenAI's ChatGroq**: For generating conversational responses.
- **Python**: Core programming language for development.
- **PyPDFDirectoryLoader**: For loading and processing multiple PDF documents.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/university-chatbot.git
   ```
2. Navigate to the project directory:
   ```bash
   cd university-chatbot
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Ask a question in the text input field.
3. The chatbot will retrieve and summarize the relevant information from the university's documents.
4. Explore additional document details through the document similarity search feature.

## Files

- **app.py**: Contains the Streamlit application code for handling user input, retrieving documents, and generating responses.
- **preprocessing.py**: Handles data preprocessing, including loading and splitting PDF documents and creating vector embeddings.
- **test.py**: Provides functionality for evaluating the model (if required).
- **requirements.txt**: Lists all the dependencies required for the project.

## Key Functionalities

- **PDF Document Processing**: Loads and processes over 72 PDFs related to Bharathiar University.
- **Vector Embedding and Retrieval**: Uses FAISS and Google Generative AI Embeddings to create and search vector embeddings of the documents.
- **Conversational AI**: Utilizes ChatGroq for generating human-like conversational responses.
- **Summary Generation**: Generates concise summaries of the retrieved information for user questions.

## How It Works

1. **Document Loading**: Loads all PDF documents from the specified directory.
2. **Vector Embedding**: Creates vector embeddings for document chunks using Google Generative AI Embeddings.
3. **User Interaction**: Users input their questions through the Streamlit interface.
4. **Retrieval and Response**: The chatbot retrieves relevant document content, generates a detailed answer, and provides a summary.
5. **Document Similarity Search**: Offers an expandable section to view related documents and their content.

## Target Audience

This project is designed for:

- **Prospective Students**: To get detailed information about courses, admissions, and scholarships.
- **Current Students and Faculty**: To find information about departments, research institutes, and faculty.
- **University Administration**: To assist in providing automated responses to common inquiries.
- **Researchers**: To access a summarized view of research and academic information available in the university documents.

## Acknowledgements

- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Google Generative AI](https://cloud.google.com/generative-ai)
- [OpenAI ChatGroq](https://www.openai.com/)

---

This project exemplifies the integration of RAG techniques to enhance information retrieval and summarization, providing a seamless user experience for querying university-related data.

