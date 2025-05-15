# ViNNan - AI-Powered Medical Chatbot System

## Overview

ViNNan is an intelligent medical chatbot system designed to assist users in diagnosing diseases and retrieving detailed medical information in Vietnamese. The system leverages advanced natural language processing (NLP) techniques, vector search, and machine learning models to provide accurate and context-aware responses. It supports multi-turn conversations and integrates with a structured medical knowledge base.

---

## Features

* **Disease Diagnosis**: Analyze user symptoms and provide potential disease predictions.
* **Medical Information Retrieval**: Retrieve detailed information about diseases from a structured knowledge base.
* **Multi-Turn Conversations**: Maintain context across multiple user interactions.
* **Interactive User Interface**: Built with Streamlit for real-time interaction.
* **Context Management**: Combine previous and current symptoms for accurate diagnosis when needed.

---

## Technologies Used

* **Programming Language**: Python
* **Frameworks & Libraries**:

  * **Frontend**: Streamlit
  * **HTML Parsing**: BeautifulSoup
  * **LLM Integration**: LangChain
  * **Vector Search**: Qdrant
  * **Reranking**: SentenceTransformers (Cross-Encoder)
* **Machine Learning Models**:

  * OpenAI GPT-based LLM
  * SentenceTransformers for embeddings
* **Tools**:

  * Qdrant (Vector Database)
  * TQDM (Progress Tracking)
  * JSON (Data Serialization)
* **Data Sources**:

  * Medical knowledge base in HTML format
  * CSV files for disease metadata

---

## Installation

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

5. Prepare the medical knowledge base:

   * Place your HTML files in the `data/Corpus` folder.
   * Run the preprocessing script to generate a clean JSON file:

   ```bash
   python pre_html.py
   ```

6. Start the chatbot:

   ```bash
   streamlit run src/interface.py
   ```

---

## Project Structure

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

---

## Usage

1. Open the chatbot in your browser after running the Streamlit app.
2. Interact with the chatbot by:

   * Providing symptoms for diagnosis.
   * Asking for detailed information about specific diseases.
3. The chatbot will maintain context across multiple interactions and provide accurate responses.

---

## Key Functionalities

1. **Preprocessing Medical Data**:

   * The `pre_html.py` script parses raw HTML files, removes unnecessary content, and structures the data into clean JSON chunks for efficient retrieval.

2. **Retrieval-Augmented Generation (RAG)**:

   * Combines vector search (Qdrant) and reranking (SentenceTransformers) to retrieve the most relevant medical information.

3. **LLM Integration**:

   * Uses OpenAI GPT-based LLM (via OpenRouter API) to generate natural language responses based on retrieved data and user queries.

4. **Context Management**:

   * Tracks user symptoms and previous interactions to provide accurate multi-turn conversation support.

---

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact

For any questions or feedback, please contact:

* **Email**: [your-email@example.com](mailto:your-email@example.com)
* **GitHub**: [your-username](https://github.com/your-username)


