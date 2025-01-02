
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
    )

    contextualize_q_system_prompt = (
        "Given a chat history and pdf for past sessions "
        "Generate an appriorate summary between the AI and Client"
        "Format the summary in paragraph form and don't include other information besides the conversation history"
        "Do not add additional information"
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
        "You are an assistant of a Psychologists"
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


file_directory="Conversation-history"
embedding_model='sentence-transformers/all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

past_chat_history = prepare_and_split_docs(file_directory)
vectorstore = ingest_into_vectordb(past_chat_history, embeddings)
retriever = vectorstore.as_retriever()
model = 'therapodLM'

summary_chain = summary_conversation_chain(model, retriever)

def summarize(history):
    input_summary = '''
    Refer only to this conversation history and do not add additional information. Please create a summary in paragraph form that preserves the context of the conversation
    "{history}"
    '''

    response_chain = summary_chain.invoke(
        {"input": input_summary},
        config={"configurable": {"session_id": "summary"}}
    )

    context_summary = response_chain["answer"]
    context_summary = f'{context_summary}'
    return context_summary

    
    