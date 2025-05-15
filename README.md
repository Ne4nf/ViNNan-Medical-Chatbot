# 💊 ViNNan - AI-Powered Medical Chatbot System

## Overview

ViNNan is an intelligent medical chatbot system designed to assist users in diagnosing diseases and retrieving detailed medical information in Vietnamese. The system leverages advanced natural language processing (NLP) techniques, vector search, and machine learning models to provide accurate and context-aware responses. It supports multi-turn conversations and integrates with a structured medical knowledge base.

---

https://github.com/user-attachments/assets/1b040075-034d-4511-8d60-f5a08be3ac91

---

## 📸 Screenshots
![Image](https://github.com/user-attachments/assets/c448a1d0-44b4-4bf3-8fe8-ff6a85d7f3b2)

---

## 🚀 Features

* **Disease Diagnosis**: Analyze user symptoms and provide potential disease predictions.
* **Medical Information Retrieval**: Retrieve detailed information about diseases from a structured knowledge base.
* **Context Management**: Combine previous and current symptoms for accurate diagnosis when needed.
* **Multi-Turn Conversations**: Maintain context across multiple user interactions.
* **Interactive User Interface**: Built with Streamlit for real-time interaction.

---

## 🛠️ Technologies 

🔹 **Programming Language**: Python

🔹 **Frameworks**: LangChain, Streamlit

🔹 **Tools**: Qdrant, OpenRouter API, BeautifulSoup

🔹 **ML Libraries**: HuggingFace Transformers, Cross-Encoder

---

## 🏠 Setup & Installation

### Prerequisites

* Python 3.8 or higher
* Virtual environment (recommended)
* Qdrant server (for vector search)

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/vinnan-medical-chatbot.git
   cd vinnan-medical-chatbot
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   * Create a `.env` file in the root directory and add the following:

   ```plaintext
   OPENROUTER_API_KEY=your_openrouter_api_key
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   ```
5. Start the chatbot:

   ```bash
   streamlit run src/interface.py
   ```

---

## 🧠 Project Structure

```
Vimedical/
├── data/                  # Folder for raw medical data (HTML, CSV)
├── scripts/               # Preprocessing scripts
│   └── pre_html.py        # Script to parse and clean HTML files
├── src/                   # Source code for the chatbot
│   ├── interface.py       # Streamlit-based user interface
│   ├── llm_chain.py       # Logic for LLM integration
│   ├── rag_chain.py       # Retrieval-Augmented Generation (RAG) logic
│   ├── tools.py           # Utility functions for context and intent processing
├── .gitignore             # Git ignore file
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 👤 Author

Developed by **THUC TU**

---


## Contact

For any questions or feedback, please contact:

* **Email**: [tuthucdz@gmail.com](mailto:tuthucdz@gmail.com)
* **GitHub**: [Ne4nf](https://github.com/Ne4nf)


