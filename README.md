# RAG Streamlit Application in Google Colab

This repository contains a Google Colab notebook (`RAG_Streamlit_Application.ipynb`) that sets up and deploys a Retrieval-Augmented Generation (RAG) pipeline as a Streamlit web application. The application runs within Google Colab and is exposed via `ngrok`, making it accessible publicly.

## Project Overview

This project demonstrates how to build and deploy a RAG system that answers questions based on a corpus of documents (web pages, PDFs, and text files). It leverages HuggingFace models for embeddings and language generation, LangChain for orchestrating the RAG pipeline, and FAISS for efficient vector storage. The entire setup is designed to run seamlessly in a Google Colab environment.

## Features

*   **Document Loading**: Loads data from web pages, PDF, and text files.
*   **Text Splitting**: Chunks documents for efficient processing and retrieval.
*   **Embeddings**: Uses `HuggingFaceEmbeddings` with `all-MiniLM-L6-v2` for creating vector representations of text.
*   **Vector Store**: Utilises FAISS for fast similarity search.
*   **LLM Integration**: Integrates a quantized `TinyLlama/TinyLlama-1.1B-Chat-v1.0` model via `HuggingFacePipeline` for generative answering.
*   **RAG Chain**: Implements a LangChain RAG pipeline combining retrieval and generation.
*   **Streamlit UI**: Provides an interactive web interface for the RAG application.
*   **ngrok Deployment**: Exposes the Streamlit application running on Colab to the internet via `ngrok`.
*   **Virtual Environment**: Supports installation into a virtual environment on Google Drive for persistence.

## Setup and Execution

Follow these steps to set up and run the RAG Streamlit application in Google Colab.

### 1. Open the Notebook

Open the `RAG_Streamlit_Application.ipynb` notebook in Google Colab.

### 2. Mount Google Drive

Mount your Google Drive to access virtual environment (optional) and data files (e.g., `attention.pdf`). Run the following cell in the notebook:

```python
from google.colab import drive
drive.mount("/content/drive")
```

### 3. Install Python Packages

Install the necessary Python packages. If you plan to use a persistent virtual environment on Google Drive, ensure it is activated before installing. Otherwise, the packages will be installed in the Colab session's environment.

```python
# If you have a virtual environment on Google Drive, install packages into it:
# !source /content/drive/MyDrive/virtual_env/bin/activate; pip install langchain langchain-core langchain-community
# !source /content/drive/MyDrive/virtual_env/bin/activate; pip install chromadb torch transformers accelerate sentence-transformers openpyxl pacmap datasets

# Install required packages for current notebook session
!pip install pypdf bitsandbytes>=0.46.1 faiss-cpu langchain-huggingface streamlit pyngrok
```

### 4. Add Virtual Environment to System Path (if applicable)

If you are using a virtual environment, ensure its `site-packages` are added to the system path. Adjust the Python version (`python3.12`) if yours is different.

```python
import sys
import os

sys.path.append("/content/drive/MyDrive/virtual_env/lib/python3.12/site-packages")
```

### 5. Create `app.py`

The Streamlit application code is defined in `app.py`. The notebook contains a cell that writes this code to a file. Run the cell starting with `%%writefile app.py` to create this file.

This file contains the logic for:
*   Loading documents (web/PDF/text).
*   Chunking documents.
*   Creating FAISS vector store with HuggingFace embeddings.
*   Loading a quantized TinyLlama model.
*   Setting up the RAG pipeline with LangChain.
*   The Streamlit UI components.

### 6. Set up ngrok Authtoken

`ngrok` is used to create a public URL for your Streamlit app. You need an `ngrok` authtoken, which should be stored as a Colab secret named `NGROK__AUTH_TOKEN`. Use any name here as per choice and replace `NGROK__AUTH_TOKEN` with the same in the code as well. 

1.  Go to [ngrok.com](https://ngrok.com/) and sign up for a free account.
2.  Obtain your authtoken from your ngrok dashboard.
3.  In Google Colab, click the "🔑 Secrets" icon on the left sidebar.
4.  Add a new secret with the name `NGROK__AUTH_TOKEN` and paste your authtoken as the value.

### 7. Run the Streamlit App with ngrok

Execute the cell that uses `pyngrok` to start your Streamlit application and create a public tunnel. This will print a public URL where your app is accessible.

```python
from pyngrok import ngrok
from google.colab import userdata
import time

ngrok.kill()

NGROK_AUTHTOKEN = userdata.get('NGROK__AUTH_TOKEN')
ngrok.set_auth_token(NGROK_AUTHTOKEN)

!nohup streamlit run app.py --server.port 8501 &> streamlit_app.log &
time.sleep(5)

public_url = ngrok.connect(addr='8501')
print(f"Your Streamlit app is live at: {public_url}")
```

### 8. Interact with the App

Click on the `public_url` printed in the output of the previous cell. This will open your Streamlit application in a new browser tab. You can then enter questions into the text input field to interact with your RAG model.

## Document Sources

The RAG pipeline is configured to use the following documents:
*   Web: `https://www.ibm.com/think/topics/transformer-model/`
*   PDF: `/content/drive/MyDrive/Colab Notebooks/RAG/attention.pdf` (ensure this PDF path exists on your Google Drive or replace it with your own path)

This is an example RAG application in which the documents relate to the _Transformers_ topic. You can use any documents related to any topic and create a particular RAG application for that topic. Try different types of documents from [LangChain_document_loaders](https://docs.langchain.com/oss/python/integrations/document_loaders). Feel free to modify `app.py` or the notebook to include your own document sources.
