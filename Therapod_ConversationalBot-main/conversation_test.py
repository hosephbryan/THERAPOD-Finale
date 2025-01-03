
#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from sentence_transformers import SentenceTransformer, util

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import trim_messages, AIMessage, SystemMessage, HumanMessage
from langchain.chains import create_history_aware_retriever
from langchain_huggingface import HuggingFaceEmbeddings

#from DetectEmotions import listen_for_commands
#from TTSpeech import output_with_piper
#from summary import summarize
#from critical import critical_notif
import torch
from rag_utils import prepare_and_split_docs, ingest_into_vectordb, save_session_to_pdf, add_to_session_history
from summary import summarize
import time
#from post_summary import send_post_request

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


    trimmer = trim_messages(
        strategy="last",
        token_counter=len,
        max_tokens=7,
        start_on="human",
        allow_partial=False,
        include_system=True,
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

    return rag_chain


def main():
    file_directory="Conversation-history"
    embedding_model='sentence-transformers/all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    past_chat_history = prepare_and_split_docs(file_directory)
    vectorstore = ingest_into_vectordb(past_chat_history, embeddings)
    retriever = vectorstore.as_retriever()
    model = 'therapodLM'

    chat_chain = initialize_conversation_chain(model, retriever)

    ### Statefully manage chat history ###
    store = {}

    def get_session_history(session_id: str, chat_history=store) -> BaseChatMessageHistory:
        if session_id not in chat_history:
            chat_history[session_id] = ChatMessageHistory()
        return chat_history[session_id]
    
    conversational_rag_chain = RunnableWithMessageHistory(
        chat_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
        )
    print("Conversational chain created")

    session_history = []

    while True:

        #user_input, raw_text = listen_for_commands()
        print("You:")
        user_input = input()

        #critical_notif(user_input)
        
        if "exit" in user_input or "goodbye" in user_input:
            ai_response = "Alright, Take care! Feel free to reach out anytime."
            break

        # Record the start time
        start_time = time.time()

        response_chain =  conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "001-1"}}
        )
        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        if len(response_chain["chat_history"]) >= 8:
            message_list = get_session_history("001-1", store)
            save_message = message_list.messages[-6:]
            print(save_message)
            message_list.clear()
            message_list.add_messages(save_message)

        # Print the elapsed time
        print(f"RESPONSE GENERATION ran for {elapsed_time:.2f} seconds")
        #response_chain["chat_history"] = trim_history(response_chain["chat_history"], 3)
        ai_response = response_chain["answer"]

        add_to_session_history(session_history, user_input, ai_response)
        #print(session_history)
        print(f"AI Therapist: {ai_response}")

        torch.cuda.empty_cache()
    
    save_session_to_pdf(session_history)
    summarized_history = summarize()
    print(summarized_history)

    '''
    start_time = time.time()
    summarized_history = summarize(session_history)
    print(summarized_history)
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"SUMMARY GENERATION ran for {elapsed_time:.2f} seconds")
    '''



if __name__ == "__main__":
    main()
    