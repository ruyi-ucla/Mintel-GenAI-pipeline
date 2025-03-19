import pandas as pd
import os
import streamlit as st
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

# The data is only from 2022 to 2023 for drink only
# This is a very simple RAG model

# Set your OpenAI API key

# api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = "Enter Your Key Here or Retrieve from Environment"
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Flavor Trends Chatbot")

def process_in_chunks(df, chunk_size=1000):
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    documents = []
    for chunk in chunks:
        loader = DataFrameLoader(
            data_frame=chunk,
            page_content_column="combined_text"
        )
        documents.extend(loader.load())
    return documents

@st.cache_resource
def initialize_chain():
    DATA_PATH = "flavor_df_cut.csv"
    try:
        # Optimize CSV reading with PyArrow
        df = pd.read_csv(
            DATA_PATH,
            engine='pyarrow',
            dtype_backend='pyarrow'
        )
        
        if df.empty:
            st.error("The dataset is empty")
            return None
        
        # Process in chunks
        chunk_size = 1000
        docs = process_in_chunks(df, chunk_size)
        
        # Initialize embeddings and vectorstore
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        
        # Create and return the optimized chain
        return ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(
                temperature=0, 
                model="gpt-3.5-turbo",
                max_tokens=500
            ),
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            verbose=True
        )
    except FileNotFoundError:
        st.error(f"Dataset file not found at {DATA_PATH}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Initialize the chain with a loading spinner
if st.session_state.chain is None:
    with st.spinner("Initializing the chatbot..."):
        st.session_state.chain = initialize_chain()

# Add a sidebar with information
with st.sidebar:
    st.header("About")
    st.write("""
    This chatbot analyzes flavor trends data across different regions, 
    products, and time periods. Ask questions about:
    - Regional flavor preferences
    - Product category trends
    - Temporal changes in flavor popularity
    - Flavor prevalence in specific countries
    """)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask about flavor trends:"):
    if st.session_state.chain is None:
        st.error("Chatbot initialization failed. Please check your dataset and API key.")
    else:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    # Include chat history in the query
                    response = st.session_state.chain({
                        "question": prompt,
                        "chat_history": st.session_state.chat_history
                    })
                    response_text = response['answer']
                    
                    # Update chat history
                    st.session_state.chat_history.append((prompt, response_text))
                    
                    # Display response
                    st.write(response_text)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text
                    })
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

# Add a footer
st.markdown("---")
st.markdown("*Powered by LangChain and OpenAI*")
