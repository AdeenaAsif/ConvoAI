import streamlit as st
import os
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
##f  rom langchain.embeddings import OpenAIEmbeddings , HuggingFaceInstructEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from htmlTemplates import css,bot_template,user_template
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks=text_splitter.split_text(text)
    return chunks
def get_vectorstore(text_chunks):
    # if os.getenv("HUGGINGFACEHUB_API_TOKEN") is None:
    #     st.error("OpenAI API key not found. Please check your .env file.")
    #     st.stop()
    # #embeddings= OpenAIEmbeddings()
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    google_api_key=os.getenv("GOOGLE_API_KEY")
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore= FAISS.from_texts(texts=text_chunks, embedding= embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore
def get_conversation_chain(vectorstore):
   # llm= ChatOpenAI()
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            #google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
            #convert_system_message_to_human=True
        )
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error initializing the model: {str(e)}")
        st.error("Please check your Google API key and internet connection")
        return None
def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.error("Please upload a PDF file first!")
        return
    response=st.session_state.conversation({'question':user_question})
    st.session_state.chat_history=response['chat_history']
    for i,message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
# sidebar contents
with st.sidebar:
    st.title('🤖 CONVOAI')
    st.markdown(
        """
        <style>
             [data-testid="stFileUploader"] button {
            background-color: #353740 !important;
            color: white !important;
            padding: 0.2rem 0.7rem !important;
            margin: 0.5rem 0 !important;
        }
        [data-testid="stFileUploader"] p {
            color: #666;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    pdf_docs=st.file_uploader("Upload your PDF here", type='pdf',accept_multiple_files=True)
    st.markdown(
        """
        <style>
        div.stButton > button {
            display: block;
            margin: 0 auto;
            width: 30%;
            margin-top: -10px;
            background-color: #3e3f4b !important;
            color: white !important;
        }
        
        </style>
        """,
        unsafe_allow_html=True
    )
    if st.button('Upload'):
        with st.spinner("Processing"):
    # get pdf text
            raw_text=get_pdf_text(pdf_docs)
          #  st.write(raw_text)
    # get text chunks 
            text_chunks=get_text_chunks(raw_text)
         #   st.write(text_chunks)
    # create vector store
            vectorstore = get_vectorstore(text_chunks)
    # create conversational chain
            st.session_state.conversation=get_conversation_chain(vectorstore)
            st.success("Done")   
                
def main():
    
    st.write(css,unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    st.header('Chat with PDF 💭')
    
    #st.session_state.conversation
    user_question=st.text_input('Ask a question about your PDF',placeholder="Type here...")
    if user_question:
        handle_userinput(user_question)
    #st.write(user_template.replace("{{MSG}}","Hello Bot"),unsafe_allow_html=True)
    #st.write(bot_template.replace("{{MSG}}","Hello Human"),unsafe_allow_html=True)
if __name__=='__main__':
    main()