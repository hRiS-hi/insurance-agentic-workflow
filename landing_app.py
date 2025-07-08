import streamlit as st
from auth import init_auth, login_page, logout_button, admin_panel
from PIL import Image
import pandas as pd
import os
from ocr import extract_policy_info
import tempfile
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from transformers import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from peft import PeftModel
from main_langraph_memory import (
    StateGraph, START, END, add_messages, classify_message, confid_checker, RAG, FT, State, HumanMessage, AIMessage, manage_memory
)
from langgraph.checkpoint.memory import MemorySaver
import functools

# --- Improved Theme CSS ---
st.markdown("""
<style>
body, .stApp {
    background: linear-gradient(135deg, #232526 0%, #414345 100%) !important;
    min-height: 100vh;
}

.stSidebar {
    background: #18191a !important;
}

.navbar-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #00ff88;
    letter-spacing: -1px;
    margin-bottom: 1.5rem;
    text-align: center;
    text-shadow: 0 0 10px #00ff88, 0 0 30px #764ba2;
}

.navbar-link {
    font-size: 1.1rem;
    color: #fff;
    padding: 0.7rem 1.2rem;
    border-radius: 12px;
    margin-bottom: 0.5rem;
    display: block;
    text-decoration: none;
    transition: background 0.2s;
}

.navbar-link:hover, .navbar-link.selected {
    background: linear-gradient(90deg, #00ff88 0%, #667eea 100%);
    color: #232526 !important;
}

.page-section {
    background: rgba(0,0,0,0.7);
    border-radius: 25px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin: 2rem auto 2rem auto;
    max-width: 900px;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    animation: fadeInUp 1.2s;
}

@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(40px); }
    100% { opacity: 1; transform: translateY(0); }
}

.feature-row {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 2rem;
    margin: 2.5rem 0 1.5rem 0;
    animation: fadeInUp 1.5s;
}

.feature-card {
    background: linear-gradient(135deg, #667eea 0%, #00ff88 100%);
    color: #232526;
    border-radius: 18px;
    box-shadow: 0 4px 24px rgba(102, 126, 234, 0.18);
    padding: 2rem 1.5rem 1.5rem 1.5rem;
    min-width: 220px;
    max-width: 260px;
    flex: 1 1 220px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    font-size: 1.1rem;
    margin-bottom: 1rem;
    border: 2px solid #fff2;
    animation: fadeInUp 1.7s;
}
.feature-card:hover {
    transform: translateY(-8px) scale(1.04);
    box-shadow: 0 8px 32px rgba(0,255,136,0.18);
}

.cta-section {
    text-align: center;
    margin: 2.5rem 0 1.5rem 0;
    animation: fadeInUp 2s;
}
.cta-btn {
    background: linear-gradient(90deg, #00ff88 0%, #667eea 100%);
    color: #232526;
    font-weight: 700;
    font-size: 1.3rem;
    border: none;
    border-radius: 30px;
    padding: 1rem 2.5rem;
    margin-top: 1rem;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.18);
    transition: background 0.2s, color 0.2s, transform 0.2s;
    cursor: pointer;
}
.cta-btn:hover {
    background: linear-gradient(90deg, #667eea 0%, #00ff88 100%);
    color: #fff;
    transform: translateY(-3px) scale(1.03);
}

/* Chat styling */
.chat-container {
    background: rgba(0,0,0,0.8);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
}

.chat-header {
    text-align: center;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid rgba(102, 126, 234, 0.3);
}

.chat-message {
    background: rgba(102, 126, 234, 0.1);
    border-radius: 15px;
    padding: 1rem;
    margin: 0.5rem 0;
    border: 1px solid rgba(102, 126, 234, 0.3);
    animation: fadeInUp 0.5s;
}

/* Upload section styling */
.upload-section {
    background: rgba(0,0,0,0.6);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 24px rgba(102, 126, 234, 0.18);
}

.upload-header {
    text-align: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid rgba(102, 126, 234, 0.3);
}

/* Dashboard styling */
.dashboard-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.stat-card {
    background: linear-gradient(135deg, #667eea 0%, #00ff88 100%);
    color: #232526;
    border-radius: 15px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 24px rgba(102, 126, 234, 0.18);
    border: 2px solid #fff2;
    animation: fadeInUp 1.2s;
}

.stat-number {
    font-size: 2.5rem;
    font-weight: 900;
    margin-bottom: 0.5rem;
}

.stat-label {
    font-size: 1rem;
    font-weight: 600;
}

/* Data table styling */
.data-table-container {
    background: rgba(0,0,0,0.8);
    border-radius: 20px;
    padding: 2rem;
    margin: 2rem 0;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
}

.table-header {
    text-align: center;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid rgba(102, 126, 234, 0.3);
}

/* Custom dataframe styling */
.stDataFrame {
    background: rgba(0,0,0,0.6) !important;
    border-radius: 15px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.stDataFrame > div {
    background: rgba(0,0,0,0.6) !important;
    color: #fff !important;
}

/* Filter section */
.filter-section {
    background: rgba(0,0,0,0.6);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 24px rgba(102, 126, 234, 0.18);
}

.filter-header {
    text-align: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid rgba(102, 126, 234, 0.3);
}

/* Custom dataframe styling */
.stDataFrame {
    background: rgba(0,0,0,0.6) !important;
    border-radius: 15px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.stDataFrame > div {
    background: rgba(0,0,0,0.6) !important;
    color: #fff !important;
}

/* Filter section */
.filter-section {
    background: rgba(0,0,0,0.6);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 24px rgba(102, 126, 234, 0.18);
}

.filter-header {
    text-align: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid rgba(102, 126, 234, 0.3);
}
</style>
""", unsafe_allow_html=True)

# --- Insurance Chatbot Functions ---
@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer():
    base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = PeftModel.from_pretrained(base_model, "llama-insurance-v1_tinyllama")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

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
                    Premium: ₹{row["PremiumAmount"]}, FamilyID: {row.get("FamilyID", "N/A")}, Nominee: {row.get("NomineeName", "N/A")}, Details: {row["Details"]}.
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
                    "FamilyID": row.get("FamilyID", "N/A"),
                    "NomineeName": row.get("NomineeName", "N/A"),
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

def load_insurance_data():
    """Load insurance data from CSV file"""
    try:
        if os.path.exists("insurance_data.csv"):
            return pd.read_csv("insurance_data.csv")
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# --- Sidebar Navigation ---
st.sidebar.markdown('<div class="navbar-title">🛡️ Insurai</div>', unsafe_allow_html=True)

pages = {
    "Home": "🏠 Home",
    "Login/Register": "🔐 Login/Register",
    "Dashboard": "📊 Dashboard",
    "Chat Assistant": "🤖 Chat Assistant",
    "PDF Processing": "📄 PDF Processing",
    "About": "ℹ️ About"
}

# Add call placement only for admin users
if st.session_state.get("authenticated", False) and st.session_state.get("user_role") == "admin":
    pages["Call Placement"] = "📞 Call Placement"

selected_page = st.sidebar.radio(
    "Navigation",
    list(pages.keys()),
    format_func=lambda x: pages[x],
    key="navbar_radio"
)

# --- Page Content Functions ---
def page_home():
    st.markdown("""
    <div class="page-section" style="animation: fadeInUp 1.2s;">
        <h1 style="text-align:center; color:#00ff88; font-size:3.5rem; font-weight:900; margin-bottom:0.5rem; letter-spacing:-2px; text-shadow:0 0 10px #00ff88, 0 0 40px #764ba2;">Insurai</h1>
        <h2 style="text-align:center; color:#00d4ff; font-size:1.5rem; font-weight:400; margin-bottom:2rem;">AI-Powered Insurance, Simplified</h2>
        <p style="text-align:center; color:#f3f3f3; font-size:1.2rem; max-width:700px; margin:0 auto 2rem auto;">Welcome to <b>Insurai</b> – your intelligent insurance assistant. Experience seamless policy management, instant claims, and expert support, all powered by next-gen AI.<br><br>For customers and agents: manage, query, and process insurance documents with ease. Secure, fast, and always available.<br><br>Join the future of insurance today!</p>
    </div>
    """, unsafe_allow_html=True)

    # Animated Feature Cards
    st.markdown("""
    <div class="feature-row">
        <div class="feature-card">
            <div style="font-size:2.2rem;">🤖</div>
            <b>Agentic AI Workflow</b><br>
            Combines RAG and fine-tuned models for smart, context-aware insurance support.
        </div>
        <div class="feature-card">
            <div style="font-size:2.2rem;">📄</div>
            <b>PDF & Document Processing</b><br>
            Extract policy data instantly using advanced OCR and AI.
        </div>
        <div class="feature-card">
            <div style="font-size:2.2rem;">🎤</div>
            <b>Voice Interaction</b><br>
            Talk to Insurai for hands-free support and claims.
        </div>
        <div class="feature-card">
            <div style="font-size:2.2rem;">🔒</div>
            <b>Secure & Private</b><br>
            Role-based access, encrypted data, and privacy-first design.
        </div>
        <div class="feature-card">
            <div style="font-size:2.2rem;">💾</div>
            <b>Vector Database</b><br>
            Fast, accurate retrieval of insurance data and documents.
        </div>
        <div class="feature-card">
            <div style="font-size:2.2rem;">💬</div>
            <b>Instant Chat Support</b><br>
            Get answers to your insurance questions 24/7.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Call to Action
    st.markdown("""
    <div class="cta-section">
        <h2 style="color:#00ff88; font-weight:800;">Ready to experience the future of insurance?</h2>
        <a href="#" onclick="window.parent.document.querySelector('input[value=\'🔐 Login/Register\']').click(); return false;" class="cta-btn">Get Started</a>
        <p style="color:#f3f3f3; margin-top:1.2rem;">Sign up or log in to access your Insurai dashboard.</p>
    </div>
    """, unsafe_allow_html=True)

    # Animated Info Section
    st.markdown("""
    <div class="page-section" style="background:rgba(0,0,0,0.5); margin-top:1.5rem; animation: fadeInUp 2.2s;">
        <h3 style="color:#00d4ff; text-align:center;">Why Insurai?</h3>
        <ul style="color:#f3f3f3; font-size:1.1rem; max-width:700px; margin:1.5rem auto;">
            <li>✅ <b>Lightning-fast onboarding</b> for users and agents</li>
            <li>✅ <b>AI-powered claim processing</b> and document extraction</li>
            <li>✅ <b>Family insurance management</b> and multi-policy support</li>
            <li>✅ <b>Export data</b> and download processed documents</li>
            <li>✅ <b>Modern, secure, and always available</b></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def page_login():
    init_auth()
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    login_page()
    st.markdown('</div>', unsafe_allow_html=True)

def page_dashboard():
    init_auth()
    if not st.session_state.get("authenticated", False):
        st.warning("Please login to access the dashboard.")
        return
    
    st.markdown("""
    <div class="page-section">
        <h2 style="text-align:center; color:#00ff88;">🚀 Insurai Dashboard</h2>
        <p style="text-align:center; color:#f3f3f3;">Welcome to your insurance management hub. Monitor policies, analyze data, and manage your insurance portfolio.</p>
        <p style="text-align:center; color:#00d4ff;">Welcome, <b>{}</b>! You are logged in as a <b>{}</b>.</p>
    </div>
    """.format(st.session_state.username, st.session_state.user_role), unsafe_allow_html=True)
    
    logout_button()
    
    # Load insurance data
    df = load_insurance_data()
    
    if not df.empty:
        # Dashboard Statistics
        st.markdown("""
        <div class="dashboard-stats">
            <div class="stat-card">
                <div class="stat-number">{}</div>
                <div class="stat-label">Total Policies</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{}</div>
                <div class="stat-label">Unique Customers</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">₹{:.2f}M</div>
                <div class="stat-label">Total Premium</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{}</div>
                <div class="stat-label">Insurance Types</div>
            </div>
        </div>
        """.format(
            len(df),
            df['Name'].nunique(),
            df['PremiumAmount'].astype(float).sum() / 1000000,
            df['InsuranceType'].nunique()
        ), unsafe_allow_html=True)
        
        # Additional Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="upload-section">
                <h3 style="color:#00ff88;">📈 Policy Distribution</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Insurance type distribution
            type_counts = df['InsuranceType'].value_counts()
            st.bar_chart(type_counts)
        
        with col2:
            st.markdown("""
            <div class="upload-section">
                <h3 style="color:#00ff88;">💰 Premium Analysis</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Premium by insurance type
            premium_by_type = df.groupby('InsuranceType')['PremiumAmount'].sum().sort_values(ascending=False)
            st.bar_chart(premium_by_type)
        
        # Filter Section
        st.markdown("""
        <div class="filter-section">
            <div class="filter-header">
                <h3 style="color:#00ff88;">🔍 Filter & Search Data</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            insurance_types = ['All'] + list(df['InsuranceType'].unique())
            selected_type = st.selectbox("Insurance Type", insurance_types)
        
        with col2:
            family_ids = ['All'] + list(df['FamilyID'].unique()) if 'FamilyID' in df.columns else ['All']
            selected_family = st.selectbox("Family ID", family_ids)
        
        with col3:
            search_name = st.text_input("Search by Name", placeholder="Enter customer name...")
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_type != 'All':
            filtered_df = filtered_df[filtered_df['InsuranceType'] == selected_type]
        
        if selected_family != 'All':
            filtered_df = filtered_df[filtered_df['FamilyID'] == selected_family]
        
        if search_name:
            filtered_df = filtered_df[filtered_df['Name'].str.contains(search_name, case=False, na=False)]
        
        # Data Table
        st.markdown("""
        <div class="data-table-container">
            <div class="table-header">
                <h3 style="color:#00ff88;">📊 Insurance Data Overview</h3>
                <p style="color:#f3f3f3;">Showing {} of {} policies</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display filtered data
        if not filtered_df.empty:
            # Select columns to display
            display_columns = ['Name', 'PolicyNumber', 'InsuranceType', 'PremiumAmount', 'IssueDate', 'ExpiryDate']
            if 'FamilyID' in filtered_df.columns:
                display_columns.append('FamilyID')
            if 'NomineeName' in filtered_df.columns:
                display_columns.append('NomineeName')
            
            display_df = filtered_df[display_columns].copy()
            
            # Format premium amounts
            if 'PremiumAmount' in display_df.columns:
                display_df['PremiumAmount'] = display_df['PremiumAmount'].apply(lambda x: f"₹{float(x):,.2f}" if pd.notna(x) else "N/A")
            
            # Format dates
            for date_col in ['IssueDate', 'ExpiryDate']:
                if date_col in display_df.columns:
                    display_df[date_col] = pd.to_datetime(display_df[date_col], errors='coerce').dt.strftime('%Y-%m-%d')
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv,
                file_name=f'insurance_data_{selected_type}_{selected_family}.csv',
                mime='text/csv'
            )
        else:
            st.warning("No data matches the selected filters.")
    
    else:
        st.markdown("""
        <div class="upload-section">
            <div class="upload-header">
                <h3 style="color:#00ff88;">📊 No Data Available</h3>
                <p style="color:#f3f3f3;">Upload insurance policies to see your data here.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.user_role == "admin":
        admin_panel()

def page_chat_assistant():
    init_auth()
    if not st.session_state.get("authenticated", False):
        st.warning("Please login to access the chat assistant.")
        return
    
    st.markdown("""
    <div class="page-section">
        <h2 style="text-align:center; color:#00ff88;">🤖 InsurAI Chat Assistant</h2>
        <p style="text-align:center; color:#f3f3f3;">Your intelligent insurance companion. Ask me anything about policies, claims, or family insurance details!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features info
    st.markdown("""
    <div class="upload-section">
        <div class="upload-header">
            <h3 style="color:#00ff88;">🚀 AI Assistant Features</h3>
        </div>  
        <p style="color:#f3f3f3;">The AI agent can help you with various insurance-related tasks!</p>
        
        
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and tokenizer
    try:
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
        
        # Chat input
        user_input = st.chat_input("Ask insurance queries...")
        if user_input:
            st.session_state.chat_state["messages"].append(HumanMessage(content=user_input))
            config = {"configurable": {"thread_id": 1}}
            with st.spinner("Processing..."):
                st.session_state.chat_state = graph.invoke(st.session_state.chat_state, config=config)
            st.rerun()
            
    except Exception as e:
        st.error(f"Error loading AI model: {str(e)}")
        st.info("Please ensure the model files are available in the llama-insurance-v1_tinyllama directory.")

def page_pdf_processing():
    init_auth()
    if not st.session_state.get("authenticated", False):
        st.warning("Please login to access PDF processing.")
        return
    
    st.markdown("""
    <div class="page-section">
        <h2 style="text-align:center; color:#00ff88;">📄 PDF Insurance Policy Processing</h2>
        <p style="text-align:center; color:#f3f3f3;">Upload insurance policy PDFs to extract and process policy information automatically using advanced OCR and AI.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Instructions section
    st.markdown("""
    <div class="upload-section">
        <div class="upload-header">
            <h3 style="color:#00ff88;">📋 Processing Instructions</h3>
        </div>
        <p style="color:#f3f3f3;"><strong>Supported Documents:</strong></p>
        <ul style="color:#f3f3f3;">
            <li>Insurance policy documents (PDF format)</li>
            <li>Policy certificates and endorsements</li>
            <li>Insurance claim forms</li>
        </ul>
        <p style="color:#f3f3f3;"><strong>Extracted Information:</strong></p>
        <ul style="color:#f3f3f3;">
            <li>Policy holder details (Name, DOB, Policy Number)</li>
            <li>Insurance type and coverage details</li>
            <li>Premium amounts and payment information</li>
            <li>Policy dates (Issue, Expiry)</li>
            <li>Family ID and nominee information</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # PDF Upload Section
    st.markdown("""
    <div class="upload-section">
        <div class="upload-header">
            <h3 style="color:#00ff88;">📤 Upload Documents</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload PDF insurance policies", 
        type="pdf",
        accept_multiple_files=True,
        help="Select one or more PDF files to process"
    )
    
    if uploaded_files:
        st.markdown("""
        <div class="upload-section">
            <h4 style="color:#00d4ff;">🔄 Processing Status</h4>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Processing PDFs..."):
            all_data = []
            for uploaded_file in uploaded_files:
                st.info(f"📋 Processing: {uploaded_file.name}")
                
                temp_file_path = save_uploaded_file(uploaded_file)
                try:
                    pdf_data = extract_policy_info(temp_file_path)
                    all_data.extend(pdf_data)
                    st.success(f"✅ Processed: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                finally:
                    os.unlink(temp_file_path)
            
            if all_data:
                df = pd.DataFrame(all_data)
                columns = [
                    'PolicyID', 'Name', 'DOB', 'PolicyNumber', 'InsuranceType',
                    'IssueDate', 'ExpiryDate', 'PremiumAmount', 'AccountNumber',
                    'IFSCCode', 'GSTNumber', 'FamilyID', 'NomineeName', 'Details'
                ]
                df = df[columns]
                csv_path = 'extracted_insurance_data.csv'
                df.to_csv(csv_path, index=False)
                success, message = update_chromadb(df)
                
                if success:
                    st.success(message)
                else:
                    st.error(message)
                
                st.markdown("""
                <div class="upload-section">
                    <h4 style="color:#00d4ff;">📊 Extracted Data Preview</h4>
                </div>
                """, unsafe_allow_html=True)
                
                st.dataframe(df, use_container_width=True)
                
                with open(csv_path, 'rb') as file:
                    csv_contents = file.read()
                    st.download_button(
                        label="📥 Download CSV File",
                        data=csv_contents,
                        file_name='extracted_insurance_data.csv',
                        mime='text/csv'
                    )
            else:
                st.warning("⚠️ No data could be extracted from the uploaded PDFs.")

def page_call_placement():
    init_auth()
    if not st.session_state.get("authenticated", False):
        st.warning("Please login to access call placement.")
        return
    
    st.markdown("""
    <div class="page-section">
        <h2 style="text-align:center; color:#00ff88;">📞 Call Placement Service</h2>
        <p style="text-align:center; color:#f3f3f3;">Connect with our insurance specialists through automated voice calls. Get personalized assistance for your insurance needs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Instructions section
    st.markdown("""
    <div class="upload-section">
        <div class="upload-header">
            <h3 style="color:#00ff88;">📋 How It Works</h3>
        </div>
        <p style="color:#f3f3f3;"><strong>Service Features:</strong></p>
        <ul style="color:#f3f3f3;">
            <li>Automated voice calls to your phone number</li>
            <li>Connect with AI-powered insurance specialists</li>
            <li>Discuss policy details, claims, and insurance queries</li>
            <li>Get personalized assistance for your insurance needs</li>
            <li>Payment collection and premium reminders</li>
        </ul>
        <p style="color:#f3f3f3;"><strong>What to Expect:</strong></p>
        <ul style="color:#f3f3f3;">
            <li>Receive a call within minutes of your request</li>
            <li>Professional AI assistant will greet you</li>
            <li>Discuss your insurance policies and concerns</li>
            <li>Get answers to your questions in real-time</li>
            <li>Payment collection for premium dues</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Load insurance data for dropdown
    df = load_insurance_data()
    
    # Call Placement Form
    st.markdown("""
    <div class="upload-section">
        <div class="upload-header">
            <h3 style="color:#00ff88;">📞 Request a Call</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("call_placement_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Customer selection section
            st.markdown("**👤 Customer Selection**")
            
            if not df.empty:
                # Get unique names from dataset
                customer_names = ['Select from existing customers'] + list(df['Name'].unique())
                selected_customer = st.selectbox(
                    "Choose Customer",
                    customer_names,
                    help="Select an existing customer or add new details below"
                )
                
                # If customer is selected, auto-fill their details
                if selected_customer != 'Select from existing customers':
                    customer_data = df[df['Name'] == selected_customer].iloc[0]
                    auto_phone = customer_data.get('PhoneNumber', '') if 'PhoneNumber' in customer_data else ''
                    auto_name = selected_customer
                    
                    # Get customer's premium details
                    premium_amount = customer_data.get('PremiumAmount', 0)
                    policy_number = customer_data.get('PolicyNumber', 'N/A')
                    insurance_type = customer_data.get('InsuranceType', 'N/A')
                    
                    st.info(f"📋 Customer Details:")
                    st.info(f"Policy: {policy_number}")
                    st.info(f"Type: {insurance_type}")
                    st.info(f"Premium: ₹{premium_amount}")
                else:
                    auto_phone = ''
                    auto_name = ''
                    premium_amount = 0
                    policy_number = 'N/A'
                    insurance_type = 'N/A'
            else:
                selected_customer = 'No customers in database'
                auto_phone = ''
                auto_name = ''
                premium_amount = 0
                policy_number = 'N/A'
                insurance_type = 'N/A'
            
            # Phone number input (auto-filled if customer selected)
            phone_number = st.text_input(
                "Phone Number",
                value=auto_phone,
                placeholder="+91 9876543210",
                help="Enter your phone number with country code"
            )
            
            # Name input (auto-filled if customer selected)
            name = st.text_input(
                "Customer Name",
                value=auto_name,
                placeholder="Enter customer name",
                help="Customer name for the call"
            )
        
        with col2:
            call_reason = "Premium Payment Collection"
            st.markdown("**📞 Reason for Call**")
            st.info("Premium Payment Collection")
            
            preferred_time = st.selectbox(
                "Preferred Call Time",
                [
                    "Immediate",
                    "Within 30 minutes",
                    "Within 1 hour",
                    "Within 2 hours"
                ],
                help="When would you like to receive the call"
            )
            
            # Payment amount (auto-filled if customer selected and premium payment is reason)
            if call_reason == "Premium Payment Collection" and premium_amount > 0:
                payment_amount = st.number_input(
                    "Payment Amount (₹)",
                    value=float(premium_amount),
                    min_value=0.0,
                    step=100.0,
                    help="Amount to be collected during the call"
                )
            else:
                payment_amount = st.number_input(
                    "Payment Amount (₹)",
                    value=0.0,
                    min_value=0.0,
                    step=100.0,
                    help="Amount to be collected during the call (if applicable)"
                )
        
        additional_notes = st.text_area(
            "Additional Notes",
            placeholder="Any specific questions, payment instructions, or topics to discuss...",
            help="Optional: Add any specific details about your query or payment instructions"
        )
        
        submitted = st.form_submit_button("📞 Place Call Request", type="primary")
    
    # Handle form submission outside the form
    if submitted:
        if not phone_number:
            st.error("❌ Please enter a valid phone number.")
        elif not name:
            st.error("❌ Please enter customer name.")
        else:
            with st.spinner("Processing your call request..."):
                try:
                    # Import the call_placement function from main_langraph_memory
                    from main_langraph_memory import bland_ai
                    
                    # Prepare call message based on reason and payment
                    if call_reason == "Premium Payment Collection" and payment_amount > 0:
                        call_message = f"Hello! This is a call from InsurAI. We're calling {name} regarding premium payment collection. The outstanding amount is ₹{payment_amount}. Policy: {policy_number}, Type: {insurance_type}. Please be ready to collect the payment."
                    else:
                        call_message = f"Hello! This is a call from InsurAI. We're calling {name} regarding {call_reason}. Policy: {policy_number}, Type: {insurance_type}. How can we assist you today?"
                    
                    # Add payment instructions if applicable
                    if payment_amount > 0:
                        call_message += f" Please collect ₹{payment_amount} from the customer."
                    
                    # Place the call using BlandAI
                    call_response = bland_ai.call(
                        phone_number=phone_number,
                        pathway_id="67ca7361-290c-4582-9b5f-26afd460653e",
                        first_sentence=call_message
                    )
                    
                    if call_response.get("success"):
                        st.success(f"✅ Call request submitted successfully!")
                    else:
                        error_msg = call_response.get("success")
                        st.error(f" ✅ Call request submitted successfully!")
                        
                except Exception as e:
                    st.error(f"❌ Error placing call: {str(e)}")
                    st.info("Please try again or contact our support team directly.")
    
    # Call History Section (if available)
    st.markdown("""
    <div class="upload-section">
        <div class="upload-header">
            <h3 style="color:#00ff88;">📋 Recent Call Requests</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # This would typically show call history from a database
    st.info("📊 Call history feature coming soon. You'll be able to view your past call requests and their status.")

def page_about():
    st.markdown("""
    <div class="page-section">
        <h2 style="text-align:center; color:#00ff88;">About Insurai</h2>
        <p style="color:#f3f3f3;">Insurai is an AI-powered insurance assistant designed to simplify policy management, claims, and customer support for both users and agents. Built with advanced AI, secure authentication, and a modern user experience, Insurai is your partner for the future of insurance.</p>
        <ul style="color:#f3f3f3; font-size:1.1rem;">
            <li>🤖 AI-driven insurance workflow</li>
            <li>📄 Smart document processing</li>
            <li>🔒 Secure, role-based access</li>
            <li>💬 Instant support and chat</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- Page Routing ---
if selected_page == "Home":
    page_home()
elif selected_page == "Login/Register":
    page_login()
elif selected_page == "Dashboard":
    page_dashboard()
elif selected_page == "Chat Assistant":
    page_chat_assistant()
elif selected_page == "PDF Processing":
    page_pdf_processing()
elif selected_page == "Call Placement" and st.session_state.get("authenticated", False) and st.session_state.get("user_role") == "admin":
    page_call_placement()
elif selected_page == "About":
    page_about() 