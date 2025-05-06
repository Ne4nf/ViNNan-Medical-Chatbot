
# from langchain.vectorstores import Qdrant
# from langchain_huggingface import HuggingFaceEmbeddings
# import json
# import uuid
# import logging
# from pathlib import Path
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import Distance, VectorParams, PointStruct
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv
# import os
# from tqdm import tqdm
# load_dotenv()
# # Qdrant Cloud configuration
# QDRANT_URL = os.getenv("QDRANT_URL")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
# questions_vectorstore = Qdrant(
#     client=qdrant_client,
#     collection_name="vimedical-questions",
#     embeddings=embedding,
#     content_payload_key="text",
#     metadata_payload_key=None 
# )

# query = "Tôi hay bị nôn ói nhiều và kéo dài kèm theo chóng mặt. Tôi có thể đang bị bệnh gì?"
# docs_with_scores = questions_vectorstore.similarity_search_with_score(query, k=3)
# for doc, score in docs_with_scores:
#     print(f"Text: {doc.page_content}, Score: {score}, Metadata: {doc.metadata}")
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Qdrant Cloud configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "vimedical-questions"  # hoặc vimedical-information tùy kiểm tra

# Câu truy vấn
query = "Tôi hay bị nôn ói nhiều và kéo dài kèm theo chóng mặt. Tôi có thể đang bị bệnh gì?"

# Tải model embedding và Qdrant client
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Khởi tạo vectorstore từ Qdrant
vectorstore = Qdrant(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embeddings=embedding,
    content_payload_key="text",
    metadata_payload_key="metadata"
)

# Truy vấn top-k
docs_with_scores = vectorstore.similarity_search_with_score(query, k=5)

# In kết quả
print(f"\n🔎 Truy vấn: {query}\n{'-'*80}")
for i, (doc, score) in enumerate(docs_with_scores, 1):
    print(f"[{i}] 📄 Text: {doc.page_content}")
    print(f"    📌 Score: {score}")
    print(f"    📑 Metadata: {doc.metadata}\n")

