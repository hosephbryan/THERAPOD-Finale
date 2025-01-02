
#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from sentence_transformers import SentenceTransformer, util

from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_huggingface import HuggingFaceEmbeddings

from DetectEmotions import listen_for_commands
from TTSpeech import output_with_piper
from summary import summarize
from critical import critical_notif
import torch
from rag_utils import prepare_and_split_docs, ingest_into_vectordb
import time
from post_summary import send_post_request

def initialize_conversation_chain(model:str, retriever):
    llm = ChatOllama(
        model=model,
        verbose=False,
        temperature=0.9,
        top_k=70,
        top_p=0.7,
        num_predict=512,
        num_ctx=2048,
        num_gpu=1, 
    )

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "Generate an appriorate response as an AI Therapist for the user input"
    )


    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


    ### Answer question ###
    system_prompt = (
        "You are a conversational AI companion designed to facilitate empathetic and supportive therapeutic conversations. Do not pretend to be an actual" 
        "psychologist/therapist and always be transparent with the user, always avoid inapproriate and sexual conversations."
        "Avoid bias related to culture, race, or gender, treat all users equitably and ensure responses are inclusive and free from discrimination. "
        "Your main goal is to create a safe, emphatetic, and non-judgemental space where users feel heard and supported. "
        "Maintain the highest standards of ethical and responsible behavior throughout the conversation."
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    ### Statefully manage chat history ###
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    print("Conversational chain created")

    return conversational_rag_chain


def main():
    output_wavfile_1 = "temp_output_1.wav"
    output_wavfile_2 = "temp_output_2.wav"
    current_wavfile = output_wavfile_1

    file_directory="Conversation-history"
    embedding_model='sentence-transformers/all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    past_chat_history = prepare_and_split_docs(file_directory)
    vectorstore = ingest_into_vectordb(past_chat_history, embeddings)
    retriever = vectorstore.as_retriever()
    model = 'therapodLM'

    chat_chain = initialize_conversation_chain(model, retriever)

    session_history = []

    while True:

        user_input, raw_text = listen_for_commands()

        critical_notif(raw_text)

        session_history.append(f'Client: {raw_text}')
        

        if "exit" in user_input or "goodbye" in user_input:
            ai_response = "Alright, Take care! Feel free to reach out anytime."
            output_with_piper(ai_response, current_wavfile)
            break
        # Record the start time
        start_time = time.time()
        response_chain =  chat_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "001-1"}}
        )
        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        # Print the elapsed time
        print(f"RESPONSE GENERATION ran for {elapsed_time:.2f} seconds")
        
        ai_response = response_chain["answer"]

        session_history.append(f'THERAPOD: {ai_response}')

        output_with_piper(ai_response, current_wavfile)
        current_wavfile = output_wavfile_2 if current_wavfile == output_wavfile_1 else output_wavfile_1
        print(f"AI Therapist: {ai_response}")

        torch.cuda.empty_cache()

    start_time = time.time()
    summarized_history = summarize(session_history)
    print(summarized_history)
    send_post_request(1, str(summarized_history))
    print(send_post_request)
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time


    # Print the elapsed time
    print(f"SUMMARY GENERATION ran for {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
    