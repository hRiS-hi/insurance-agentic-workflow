# InsurAI - Insurance Agentic AI Workflow

<div align="center">

![InsurAI](https://img.shields.io/badge/InsurAI-Insurance%20AI-blue?style=for-the-badge&logo=shield-check)

**AI-powered insurance agent with document processing, voice-enabled call placement, and intelligent customer support**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.4.8+-green.svg)](https://langchain.com/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red.svg)](https://streamlit.io/)

</div>

## 🚀 Overview

InsurAI is an intelligent insurance agent that helps insurance companies manage customer data, process policy documents, and provide AI-powered customer support. Built with LangGraph for robust AI workflows and pdfplumber for document processing. The app features a modern, dark-themed UI for an intuitive user experience.

## ✨ Key Features

- **🤖 Agentic AI Workflow**: Combines RAG and fine-tuned models using LangGraph for smart, context-aware insurance support
- **📄 PDF Processing**: Extract policy data using pdfplumber and OCR
- **🎤 Voice-Enabled Call Placement**: Admins can place automated voice calls for premium collection and support (not direct voice chat yet)
- **🔐 User Authentication & Roles**: Secure login system with role-based access (admin and user); admins have access to extra features like call placement and admin panel
- **📊 Dashboard & Analytics**: Visualize insurance data, filter by type/family, and export as CSV
- **💬 AI Chat Assistant**: Query customer insurance details and general questions with context-aware responses
- **💾 Vector Database**: ChromaDB for efficient data storage and retrieval
- **👨‍👩‍👧‍👦 Family Insurance Management**: Manage and filter policies by FamilyID and nominee (core features implemented; more planned)
- **🔒 Privacy-Aware**: Intelligent classification of confidential vs. general queries, secure routing, and memory management
- **🌙 Modern UI/UX**: Responsive, dark-themed interface with feature cards, animated sections, and easy navigation

## 📁 Project Files

| File | Purpose |
|------|---------|
| **`landing_app.py`** | Main Streamlit web application with modern UI, navigation, and all features |
| **`app.py`** | Alternate/legacy Streamlit app (see landing_app.py for main UI) |
| **`auth.py`** | User authentication, login, registration, and admin panel logic |
| **`data_gen.py`** | Generate synthetic insurance data for testing |
| **`ocr.py`** | Extract policy information from PDFs using pdfplumber |
| **`main_langraph_memory.py`** | LangGraph AI workflow with memory management and call placement logic |
| **`vector.py`** | Vector database setup and retrieval for RAG system |
| **`requirements.txt`** | Python dependencies and packages |
| **`chrome_langchain_db/`** | ChromaDB vector database storage |
| **`llama-insurance-v1_tinyllama/`** | Fine-tuned TinyLlama model for insurance queries |
| **`insurance_data.csv`** | Sample insurance data for testing |
| **`extracted_insurance_data.csv`** | Processed PDF data output |
| **`users.json`** | User credentials and roles |

## 🛠️ Quick Setup

### Prerequisites
- Python 3.8+
- Ollama with `mxbai-embed-large` model
- Tesseract OCR
- Groq API key

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd ai_voice

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Setup Ollama
ollama pull mxbai-embed-large

# Create .env file
echo "GROQ_API_KEY=your_key_here" > .env

# Run the application
streamlit run landing_app.py
```

## 🚀 Usage

1. **Login/Register**: Create an account or log in (admins get extra features)
2. **Upload PDFs**: Use sidebar to upload insurance policy PDFs
3. **Ask Questions**: Query customer data or general insurance questions in the Chat Assistant
4. **Get AI Responses**: Receive intelligent, context-aware answers
5. **Export Data**: Download processed data as CSV
6. **(Admin) Place Calls**: Use the Call Placement page to request automated voice calls for premium collection or support

### Example Queries

**General Questions:**
- "What is the claim process for my policy?"
- "How do I file a claim?"

**Customer Data Queries:**
- "What is the premium for policy number 1234567890?"
- "Show me John Smith's policy details"
- "Who is the nominee for policy 1234567890?"
- "Show me all policies where Mary Johnson is the nominee"

**Call Placement (Admin):**
- Select a customer and request a call for premium payment collection or support

## 🏗️ Architecture

```
User Input → Classifier → Router → RAG/FT Model → Response
     ↓           ↓         ↓         ↓
PDF Upload → OCR Processing → Data Storage → Vector DB
Call Placement (Admin) → Automated Voice Call Service
```

## 🔧 Configuration

### AI Models Used
- **Fine-tuned TinyLlama**: `llama-insurance-v1_tinyllama/`
- **Ollama Embeddings**: `mxbai-embed-large`
- **Groq LLM**: `llama-3.3-70b-versatile`

### Memory Settings
```python
MAX_MESSAGES = 10  # Conversation memory limit
SUMMARY_THRESHOLD = 8  # Summarization trigger
```

## 🔒 Security Features

- **User Authentication & Roles**: Secure login, admin/user roles, admin panel
- **Confidential Data Detection**: Automatically identifies sensitive information
- **Secure Routing**: Routes confidential queries to RAG system
- **Data Validation**: Validates extracted data before storage
- **Memory Management**: Conversation summarization for privacy

## 🚧 Current & Future Features

- **Family Member Info Collection**: Core features (FamilyID, nominee) implemented; more advanced management planned
- **Call Placement**: Automated voice call feature for admins is implemented
- **Voice Chat**: Voice-enabled chat is planned (currently, only call placement is voice-enabled)
- **Enhanced Analytics**: Advanced reporting and insights planned
- **Mobile Application**: Native mobile app for field agents planned

## 📊 Data Processing

### Extracted Fields
- **Policy Info**: PolicyID, Name, DOB, PolicyNumber, InsuranceType
- **Dates**: IssueDate, ExpiryDate
- **Financial**: PremiumAmount, AccountNumber, IFSCCode, GSTNumber
- **Family**: FamilyID
- **Nominee**: NomineeName
- **Details**: Additional policy information

### Generate Test Data
```bash
python data_gen.py  # Creates insurance_data.csv
```

## 🤝 Support

- **Documentation**: Check code comments for detailed usage
- **Issues**: Create GitHub issues for bugs or feature requests
- **Testing**: Run `python ocr.py` to test PDF processing

---

<div align="center">

**Built with ❤️ for intelligent insurance solutions**

</div> 