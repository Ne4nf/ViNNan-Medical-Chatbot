ViNNan - AI-Powered Medical Chatbot System
Overview
ViNNan is an intelligent medical chatbot system designed to assist users in diagnosing diseases and retrieving detailed medical information in Vietnamese. The system leverages advanced natural language processing (NLP) techniques, vector search, and machine learning models to provide accurate and context-aware responses. It supports multi-turn conversations and integrates with a structured medical knowledge base.

Features
Disease Diagnosis: Analyze user symptoms and provide potential disease predictions.
Medical Information Retrieval: Retrieve detailed information about diseases from a structured knowledge base.
Multi-Turn Conversations: Maintain context across multiple user interactions.
Interactive User Interface: Built with Streamlit for real-time interaction.
Context Management: Combine previous and current symptoms for accurate diagnosis when needed.
Technologies Used
Programming Language: Python
Frameworks & Libraries:
Frontend: Streamlit
HTML Parsing: BeautifulSoup
LLM Integration: LangChain
Vector Search: Qdrant
Reranking: SentenceTransformers (Cross-Encoder)
Machine Learning Models:
OpenAI GPT-based LLM
SentenceTransformers for embeddings
Tools:
Qdrant (Vector Database)
TQDM (Progress Tracking)
JSON (Data Serialization)
Data Sources:
Medical knowledge base in HTML format
CSV files for disease metadata
