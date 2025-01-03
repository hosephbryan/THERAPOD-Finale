
#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_huggingface import HuggingFaceEmbeddings

from rag_utils import prepare_and_split_docs, ingest_into_vectordb


def summary_conversation_chain(model:str, retriever):
    llm = ChatOllama(
        model=model,
        verbose=False,
        top_k=40,
        top_p=0.4,
        num_predict=512,
        num_ctx=2048,
        num_gpu=1,
        keep_alive=False,
    )

    ### Answer question ###
    system_prompt = (
        "You the summarizer for an AI Mental Health Care Companion, Therapod"
        "Given the conversation history between the Psychologists and Client, generate a summary of the conversation"
        "Emphasize important events, and entities such as names, places, or experiences"
        "Your main goal is to provide an informative summary that will be used to generate a conversation report"
        "Format the summary in paragraph form and don't include other information besides the conversation history"
        "Do not add additional information"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
        ]
    )
    answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    print("Conversational chain created")

    return answer_chain


def summarize():
    file_directory="Conversation-history/session_history"
    embedding_model='sentence-transformers/all-mpnet-base-v2'
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    session_history = prepare_and_split_docs(file_directory)
    vectorstore = ingest_into_vectordb(session_history, embeddings)
    retriever = vectorstore.as_retriever()
    model = 'qwen2.5:3b-instruct-q4_K_M'

    summary_chain = summary_conversation_chain(model, retriever)
    response_chain = summary_chain.invoke(
        {"context": session_history},
    )
    print(response_chain)
    context_summary = response_chain
    context_summary = f'{context_summary}'
    return context_summary

    
    