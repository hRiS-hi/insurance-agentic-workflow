# InsurAI - Insurance Agentic AI Workflow

<div align="center">

![InsurAI](https://img.shields.io/badge/InsurAI-Insurance%20AI-blue?style=for-the-badge&logo=shield-check)

**AI-powered insurance agent with document processing, voice interaction, and intelligent customer support**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.4.8+-green.svg)](https://langchain.com/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red.svg)](https://streamlit.io/)

</div>

## 🚀 Overview

InsurAI is an intelligent insurance agent that helps insurance companies manage customer data, process policy documents, and provide AI-powered customer support. Built with LangGraph for robust AI workflows and pdfplumber for document processing.

## ✨ Key Features

- **🤖 Agentic AI Workflow**: Combines RAG and fine-tuned models using LangGraph
- **📄 PDF Processing**: Extract policy data using pdfplumber and OCR
- **🎤 Voice Interaction**: Speech-to-text and text-to-speech capabilities
- **🔍 Smart Querying**: Query customer insurance details instantly
- **💾 Vector Database**: ChromaDB for efficient data storage and retrieval
- **🔒 Privacy-Aware**: Intelligent classification of confidential vs. general queries

## 📁 Project Files

| File | Purpose |
|------|---------|
| **`app.py`** | Main Streamlit web application with modern dark UI |
| **`data_gen.py`** | Generate synthetic insurance data for testing |
| **`ocr.py`** | Extract policy information from PDFs using pdfplumber |
| **`main_langraph_memory.py`** | LangGraph AI workflow with memory management |
| **`requirements.txt`** | Python dependencies and packages |
| **`chrome_langchain_db/`** | ChromaDB vector database storage |
| **`llama-insurance-v1_tinyllama/`** | Fine-tuned TinyLlama model for insurance queries |
| **`insurance_data.csv`** | Sample insurance data for testing |
| **`extracted_insurance_data.csv`** | Processed PDF data output |

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
streamlit run app.py
```

## 🚀 Usage

1. **Upload PDFs**: Use sidebar to upload insurance policy PDFs
2. **Ask Questions**: Query customer data or general insurance questions
3. **Get AI Responses**: Receive intelligent, context-aware answers
4. **Export Data**: Download processed data as CSV

### Example Queries

**General Questions:**
- "What is the claim process for my policy?"
- "How do I file a claim?"

**Customer Data Queries:**
- "What is the premium for policy number 1234567890?"
- "Show me John Smith's policy details"

## 🏗️ Architecture

```
User Input → Classifier → Router → RAG/FT Model → Response
     ↓           ↓         ↓         ↓
PDF Upload → OCR Processing → Data Storage → Vector DB
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

- **Confidential Data Detection**: Automatically identifies sensitive information
- **Secure Routing**: Routes confidential queries to RAG system
- **Data Validation**: Validates extracted data before storage
- **Memory Management**: Conversation summarization for privacy

## 🚧 Future Features

- **Family Member Info Collection**: Comprehensive family policy management
- **Calling Agent Feature**: AI-powered voice calling capabilities
- **Enhanced Analytics**: Advanced reporting and insights
- **Mobile Application**: Native mobile app for field agents

## 📊 Data Processing

### Extracted Fields
- PolicyID, Name, DOB, PolicyNumber
- InsuranceType, IssueDate, ExpiryDate
- PremiumAmount, AccountNumber, IFSCCode, GSTNumber
- Details and additional policy information

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