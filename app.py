import streamlit as st

# Custom CSS for modern dark theme with gradients
st.set_page_config(
    page_title="Insurance Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme with gradients
st.markdown("""
<style>
    /* Main background gradient */
    .main {
        background: #000000;
        color: #ffffff;
    }
    
    /* Sidebar gradient */
    .css-1d391kg {
        background: #000000;
    }
    
    /* Header styling */
    .main-header {
        background: transparent;
        padding: 2rem 0;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: #00ff88;
        text-align: center;
        margin: 0 0 0.2rem 0;
        font-size: 4.5rem;
        font-weight: 900;
        text-shadow: 
            0 0 5px #00ff88,
            0 0 30px #00ff88,
            2px 2px 4px rgba(0, 0, 0, 0.7);
        letter-spacing: -3px;
        line-height: 1.1;
    }
    
    .main-header h2 {
        color: #00d4ff;
        text-align: center;
        margin: 0;
        font-size: 1.2rem;
        font-weight: 400;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
        letter-spacing: 1px;
        opacity: 0.9;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: #000000;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 10px 0;
        position: relative;
        z-index: 0;
    }
    
    /* Chat input styling */
    .stChatInput {
        background: #000000;
        border-radius: 25px;
        border: 2px solid rgba(255, 255, 255, 0.6);
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
        position: relative;
        z-index: 1;
        margin-bottom: 20px;
    }
    
    /* Chat input focus state */
    .stChatInput:focus {
        border: 2px solid rgba(255, 255, 255, 0.8);
        box-shadow: 0 0 15px rgba(255, 255, 255, 0.3);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 25px;
        color: white;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: #000000;
        border-radius: 15px;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        padding: 2rem;
    }
    
    /* Sidebar sections */
    .sidebar-section {
        background: #000000;
        border-radius: 15px;
        padding: 0.5rem 1rem 1rem 1rem;
        margin: 0rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Info box styling */
    .stAlert {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 15px;
    }
    
    /* Success message styling */
    .stSuccess {
        background: rgba(76, 175, 80, 0.1);
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-radius: 15px;
    }
    
    /* Error message styling */
    .stError {
        background: rgba(244, 67, 54, 0.1);
        border: 1px solid rgba(244, 67, 54, 0.3);
        border-radius: 15px;
    }
    
    /* Warning message styling */
    .stWarning {
        background: rgba(255, 152, 0, 0.1);
        border: 1px solid rgba(255, 152, 0, 0.3);
        border-radius: 15px;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: #000000;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Spinner styling */
    .stSpinner {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 15px;
    }
    
    /* Text styling */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
    }
    
    p, div {
        color: #e0e0e0;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.5);
    }
    
    /* Force black background for all elements */
    .stApp {
        background: #000000;
    }
    
    .block-container {
        background: #000000;
    }
    
    .main .block-container {
        background: #000000;
    }
    
    /* Ensure text visibility */
    .stMarkdown {
        color: #ffffff;
    }
    
    .stText {
        color: #ffffff;
    }
    
    /* Chat container background */
    .stChatFloatingInputContainer {
        background: #000000;
        position: relative;
        z-index: 1;
        margin-top: 20px;
    }
    
    /* Sidebar text visibility */
    .css-1d391kg .stMarkdown {
        color: #ffffff;
    }
    
    .css-1d391kg p {
        color: #ffffff;
    }
    
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {
        color: #ffffff;
    }
    
    /* Remove sidebar scrollbar */
    .css-1d391kg {
        background: #000000;
        overflow: hidden;
    }
    
    .css-1d391kg .css-1lcbmhc {
        overflow: hidden;
    }
    
    /* Sidebar content alignment */
    .css-1d391kg .stMarkdown {
        color: #ffffff;
        margin: 0;
        padding: 0;
    }
    
    .css-1d391kg p {
        color: #ffffff;
        margin: 0.5rem 0;
        line-height: 1.4;
    }
    
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg h5, .css-1d391kg h6 {
        color: #ffffff;
        margin: 0.5rem 0;
        line-height: 1.2;
    }
    
    .css-1d391kg ul {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    
    .css-1d391kg li {
        margin: 0.2rem 0;
        line-height: 1.3;
    }
    
    /* Chat area background */
    .stChatMessageContent {
        background: #000000;
        position: relative;
        z-index: 0;
    }
    
    /* Input area background */
    .stTextInput > div > div > input {
        background: #000000;
        color: #ffffff;
        position: relative;
        z-index: 1;
    }
    
    /* Fix overlapping issues */
    .stChatFloatingInputContainer {
        margin-bottom: 100px;
    }
    
    .stChatMessage {
        margin-bottom: 15px;
    }
    
    /* Remove unwanted rectangular elements in prompt area */
    .stChatInput > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    .stChatInput > div > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    .stChatInput input {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    .stChatInput textarea {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Remove any default Streamlit styling */
    .stChatInput * {
        background: transparent !important;
    }
    
    /* Ensure only our custom border shows */
    .stChatInput {
        background: #000000 !important;
        border: 2px solid rgba(255, 255, 255, 0.6) !important;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

import pandas as pd
import os
from ocr import extract_policy_info
import tempfile
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from main_langraph_memory import (
    StateGraph, START, END, add_messages, classify_message, confid_checker, RAG, FT, State, HumanMessage, AIMessage, manage_memory
)
from langgraph.checkpoint.memory import MemorySaver
import functools

@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer():
    base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = PeftModel.from_pretrained(base_model, "llama-insurance-v1_tinyllama")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Build the graph with FT node using partial to inject model and tokenizer
ft_node = functools.partial(FT, model=model, tokenizer=tokenizer)
graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", confid_checker)
graph_builder.add_node("RAG_model", RAG)
graph_builder.add_node("FT_model", ft_node)
graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")
graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"),
    {
        "RAG_model": "RAG_model",
        "FT_model": "FT_model"
    }
)
graph_builder.add_edge("RAG_model", END)
graph_builder.add_edge("FT_model", END)
graph = graph_builder.compile(checkpointer=MemorySaver())

def update_chromadb(df):
    """Update ChromaDB with new insurance data"""
    try:
        # Initialize embeddings
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        
        # Connect to existing ChromaDB
        db_location = "chrome_langchain_db"
        vector_store = Chroma(
            collection_name="insurance_data",
            persist_directory=db_location,
            embedding_function=embeddings
        )
        
        # Convert new data to documents
        documents = []
        ids = []
        
        # Get the current count of documents to continue ID sequence
        current_count = len(vector_store.get()['ids'])
        
        # Get all existing PolicyNumbers from ChromaDB
        existing_docs = vector_store.get()
        existing_policy_numbers = set()
        for meta in existing_docs['metadatas']:
            if meta and 'PolicyNumber' in meta:
                existing_policy_numbers.add(meta['PolicyNumber'])
        
        for i, row in df.iterrows():
            if row["PolicyNumber"] in existing_policy_numbers:
                continue  # Skip duplicates
            try:
                dob = pd.to_datetime(row["DOB"], format="%d/%m/%Y").strftime("%Y-%m-%d")
                issue_date = pd.to_datetime(row["IssueDate"], format="%d/%m/%Y").strftime("%Y-%m-%d")
                expiry_date = pd.to_datetime(row["ExpiryDate"], format="%d/%m/%Y").strftime("%Y-%m-%d")
            except:
                # If dates are already in YYYY-MM-DD format
                dob = row["DOB"]
                issue_date = row["IssueDate"]
                expiry_date = row["ExpiryDate"]
            
            document = Document(
                page_content = f"""
                    Name: {row["Name"]}, born on {dob}, holds policy number {row["PolicyNumber"]} ({row["InsuranceType"]}), issued on {issue_date} and expiring on {expiry_date}.
                    Premium: ₹{row["PremiumAmount"]}. Details: {row["Details"]}.
                    """,
                metadata={
                    "PolicyID": row["PolicyID"],
                    "Name": row["Name"],
                    "DOB": dob,
                    "PolicyNumber": row["PolicyNumber"],
                    "InsuranceType": row["InsuranceType"],
                    "IssueDate": issue_date,
                    "ExpiryDate": expiry_date,
                    "PremiumAmount": row["PremiumAmount"],
                    "AccountNumber": row["AccountNumber"],
                    "IFSCCode": row["IFSCCode"],
                    "GSTNumber": row["GSTNumber"],
                    "Details": row["Details"]
                },
                id=str(current_count + i)
            )
            ids.append(str(current_count + i))
            documents.append(document)
        
        # Add new documents to the existing vector store
        if documents:
            vector_store.add_documents(documents=documents, ids=ids)
            return True, f"Successfully added {len(documents)} new documents to ChromaDB"
        return True, "No new documents to add (all were duplicates)"
    except Exception as e:
        return False, f"Error updating ChromaDB: {str(e)}"

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def pdf_upload_sidebar():
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <h3>📄 Upload Insurance Policy</h3>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.sidebar.file_uploader(
        " ", 
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.sidebar.markdown("""
        <div class="sidebar-section">
            <h4>🔄 Processing Status</h4>
        </div>
        """, unsafe_allow_html=True)
        
        with st.sidebar:
            with st.spinner("Processing PDFs..."):
                all_data = []
                for uploaded_file in uploaded_files:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
                                border-radius: 10px; 
                                padding: 1rem; 
                                margin: 0.5rem 0;
                                border: 1px solid rgba(102, 126, 234, 0.3);">
                        <p style="margin: 0; color: white;">📋 Processing: {uploaded_file.name}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    temp_file_path = save_uploaded_file(uploaded_file)
                    try:
                        pdf_data = extract_policy_info(temp_file_path)
                        all_data.extend(pdf_data)
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, rgba(76, 175, 80, 0.2) 0%, rgba(139, 195, 74, 0.2) 100%);
                                    border-radius: 10px; 
                                    padding: 1rem; 
                                    margin: 0.5rem 0;
                                    border: 1px solid rgba(76, 175, 80, 0.3);">
                            <p style="margin: 0; color: white;">✅ Processed: {uploaded_file.name}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, rgba(244, 67, 54, 0.2) 0%, rgba(229, 57, 53, 0.2) 100%);
                                    border-radius: 10px; 
                                    padding: 1rem; 
                                    margin: 0.5rem 0;
                                    border: 1px solid rgba(244, 67, 54, 0.3);">
                            <p style="margin: 0; color: white;">❌ Error processing {uploaded_file.name}: {str(e)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    finally:
                        os.unlink(temp_file_path)
                
                if all_data:
                    df = pd.DataFrame(all_data)
                    columns = [
                        'PolicyID', 'Name', 'DOB', 'PolicyNumber', 'InsuranceType',
                        'IssueDate', 'ExpiryDate', 'PremiumAmount', 'AccountNumber',
                        'IFSCCode', 'GSTNumber', 'Details'
                    ]
                    df = df[columns]
                    csv_path = 'extracted_insurance_data.csv'
                    df.to_csv(csv_path, index=False)
                    success, message = update_chromadb(df)
                    
                    if not success:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, rgba(244, 67, 54, 0.2) 0%, rgba(229, 57, 53, 0.2) 100%);
                                    border-radius: 15px; 
                                    padding: 1.5rem; 
                                    margin: 1rem 0;
                                    border: 1px solid rgba(244, 67, 54, 0.3);">
                            <p style="margin: 0; color: white;">{message}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.sidebar.markdown("""
                    <div class="sidebar-section">
                        <h4>📊 Extracted Data Preview</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.sidebar.dataframe(df, use_container_width=True)
                    
                    st.sidebar.markdown("""
                    <div class="sidebar-section">
                        <h4>💾 Download Extracted Data</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with open(csv_path, 'rb') as file:
                        csv_contents = file.read()
                        st.sidebar.download_button(
                            label="📥 Download CSV File",
                            data=csv_contents,
                            file_name='extracted_insurance_data.csv',
                            mime='text/csv'
                        )
                else:
                    st.sidebar.markdown("""
                    <div style="background: linear-gradient(135deg, rgba(255, 152, 0, 0.2) 0%, rgba(255, 193, 7, 0.2) 100%);
                                border-radius: 15px; 
                                padding: 1.5rem; 
                                margin: 1rem 0;
                                border: 1px solid rgba(255, 152, 0, 0.3);">
                        <p style="margin: 0; color: white;">⚠️ No data could be extracted from the uploaded PDFs.</p>
                    </div>
                    """, unsafe_allow_html=True)

def main():
    pdf_upload_sidebar()
    
    # Modern header with gradient
    st.markdown("""
    <div class="main-header">
        <h1>InsurAI</h1>
        <h2>Smart Insurance Agentic Workflow</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Add family information section with modern styling
    st.sidebar.markdown("""
    <div class="sidebar-section">
        <h3>Family Insurance Features</h3>
        <p>The AI agent can now provide family insurance details!</p>
        <p><strong>Example queries:</strong></p>
        <ul>
            <li>"Show me family insurance details for John Smith"</li>
            <li>"What are the family policies for policy number 1234567890"</li>
            <li>"Get insurance details for my family members"</li>
            <li>"Show me all policies for my spouse and children"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chat state
    if "chat_state" not in st.session_state:
        st.session_state.chat_state = {
            "messages": [],
            "message_type": None,
            "memory": {},
            "summary": None
        }
    
    
    
    # Display chat history
    for msg in st.session_state.chat_state["messages"]:
        if isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)
        else:
            st.chat_message("user").write(msg.content)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Chat input with modern styling
    user_input = st.chat_input("Ask insurance queries...")
    if user_input:
        st.session_state.chat_state["messages"].append(HumanMessage(content=user_input))
        config = {"configurable": {"thread_id": 1}}
        with st.spinner("Processing..."):
            st.session_state.chat_state = graph.invoke(st.session_state.chat_state, config=config)
        st.rerun()

if __name__ == "__main__":
    main() 