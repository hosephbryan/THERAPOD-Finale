from langchain_community.document_loaders import DirectoryLoader, CSVLoader, UnstructuredWordDocumentLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer, util
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from textwrap import wrap
from datetime import datetime

def prepare_and_split_docs(directory):
    # Load the documents
    loaders = [
        DirectoryLoader(directory, glob="**/*.pdf",show_progress=True, loader_cls=PyPDFLoader),
        DirectoryLoader(directory, glob="**/*.docx",show_progress=True),
        DirectoryLoader(directory, glob="**/*.csv",loader_cls=CSVLoader)
    ]


    documents=[]
    for loader in loaders:
        data =loader.load()
        documents.extend(data)

    # Initialize a text splitter
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=256,
        disallowed_special=(),
        separators=["\n\n", "\n", " "]
    )

    # Split the documents and keep metadata
    split_docs = splitter.split_documents(documents)

    return split_docs


def ingest_into_vectordb(split_docs, embeddings):
    db = FAISS.from_documents(split_docs, embeddings)

    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)
    print("Documents are inserted into FAISS vectorstore")
    return db

def add_to_session_history(session_history, client_message, ai_response):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session_history.append({
        "timestamp": timestamp,
        "Client": client_message,
        "Therapod": ai_response
    })

def save_session_to_pdf(session_history, folder_name="Conversation-history/session_history"): 
    # Generate a filename with the current date
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"session_history_{current_date}.pdf"
    
    # Full path for the PDF
    file_path = os.path.join(folder_name, file_name)
    
    # Create the PDF
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter

    # Set title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, height - 50, "Therapod Session History")
    c.setFont("Helvetica", 12)

    # Set margins and text wrapping width
    margin = 50
    line_height = 15
    text_wrap_width = width - 2 * margin

    # Add conversation history
    y = height - 80  # Start position for text
    for turn in session_history:
        timestamp = turn["timestamp"]
        client_message = turn["Client"]
        ai_response = turn["Therapod"]

        # Wrap and write text
        for text, label in [(timestamp, "Timestamp"), (client_message, "Client"), (ai_response, "Therapod")]:
            wrapped_lines = wrap(f"{label}: {text}", width=100)  # Adjust width for wrapping
            for line in wrapped_lines:
                c.drawString(margin, y, line)
                y -= line_height
                if y < margin:  # Check if text exceeds page
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y = height - margin

        y -= 20  # Add space between turns

    # Save the PDF
    c.save()
    print(f"Session history saved to {file_path}")