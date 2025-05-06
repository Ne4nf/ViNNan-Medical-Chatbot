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

# # Load environment variables
# load_dotenv()

# # Qdrant Cloud configuration
# QDRANT_URL = os.getenv("QDRANT_URL")
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# if not QDRANT_URL or not QDRANT_API_KEY:
#     raise ValueError("⚠️ QDRANT_URL hoặc QDRANT_API_KEY không được cấu hình trong .env")

# # File paths
# CLEAN_CHUNKS_PATH = "D:/Vimedical/scripts/clean_chunks.json"
# QUESTIONS_PATH = "D:/Vimedical/scripts/questions_merged.json"

# # Collection names
# COLLECTION_QUESTIONS = "vimedical-questions"
# COLLECTION_INFORMATION = "vimedical-information"

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize Qdrant Cloud Client
# qdrant_client = QdrantClient(
#     url=QDRANT_URL,
#     api_key=QDRANT_API_KEY,
#     timeout=120,
#     check_compatibility=False
# )

# # Load embedding model
# try:
#     model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
#     logger.info("✅ Mô hình SentenceTransformer đã được tải thành công!")
# except Exception as e:
#     logger.error(f"❌ Lỗi khi tải mô hình SentenceTransformer: {e}")
#     raise

# def load_json_file(path):
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#             if not data:
#                 raise ValueError(f"⚠️ File {path} rỗng.")
#             return data
#     except Exception as e:
#         logger.error(f"❌ Lỗi khi tải file {path}: {e}")
#         raise

# def extract_chunks(data):
#     chunks = []
#     for item in data:
#         disease = item.get('title', '').split(':')[0].strip()
#         for section in item.get("sections", []):
#             text = section.get("content", "").strip()
#             if len(text.split()) > 10:
#                 chunks.append({
#                     "text": text,
#                     "metadata": {
#                         "disease": disease,
#                         "source": item.get("source", ""),
#                         "type": "information"
#                     }
#                 })
#     return chunks

# def extract_questions(data):
#     questions = []
#     for disease, qs in data.items():
#         for q in qs:
#             q_clean = q.strip()
#             if len(q_clean.split()) > 5:
#                 questions.append({
#                     "text": q_clean,
#                     "metadata": {
#                         "disease": disease,
#                         "source": "questions_merged",
#                         "type": "question"
#                     }
#                 })
#     return questions

# def create_collection_with_index(collection_name):
#     try:
#         collections = qdrant_client.get_collections().collections
#         if collection_name in [c.name for c in collections]:
#             logger.info(f"🧹 Xóa collection cũ: {collection_name}")
#             qdrant_client.delete_collection(collection_name)

#         qdrant_client.create_collection(
#             collection_name=collection_name,
#             vectors_config=VectorParams(size=384, distance=Distance.COSINE)
#         )
#         logger.info(f"✅ Created collection {collection_name}")

#         # Tạo payload index cho trường metadata.disease (sử dụng API cũ)
#         qdrant_client.create_payload_index(
#             collection_name=collection_name,
#             field_name="metadata.disease",
#             field_type="keyword"
#         )
#         logger.info(f"✅ Created index on 'metadata.disease' in {collection_name}")

#     except Exception as e:
#         logger.error(f"❌ Lỗi khi tạo collection {collection_name}: {e}")
#         raise

# def embed_and_upsert(texts, collection_name, batch_size=100):
#     if not texts:
#         logger.warning(f"⚠️ Không có dữ liệu để upsert vào {collection_name}.")
#         return

#     try:
#         for i in tqdm(range(0, len(texts), batch_size), desc=f"Upserting {collection_name}"):
#             batch = texts[i:i + batch_size]
#             vectors = model.encode([item["text"] for item in batch], show_progress_bar=False)
#             points = [
#                 PointStruct(
#                     id=str(uuid.uuid4()),
#                     vector=vec.tolist(),
#                     payload={
#                         "text": item["text"],
#                         "metadata": item["metadata"]
#                     }
#                 ) for item, vec in zip(batch, vectors)
#             ]
#             qdrant_client.upsert(collection_name=collection_name, points=points)
#             logger.info(f"✅ Uploaded {len(points)} points to {collection_name}")
#     except Exception as e:
#         logger.error(f"❌ Lỗi khi upsert vào collection {collection_name}: {e}")
#         raise

# def main():
#     try:
#         chunks_data = load_json_file(CLEAN_CHUNKS_PATH)
#         questions_data = load_json_file(QUESTIONS_PATH)

#         chunks = extract_chunks(chunks_data)
#         questions = extract_questions(questions_data)

#         create_collection_with_index(COLLECTION_INFORMATION)
#         create_collection_with_index(COLLECTION_QUESTIONS)

#         embed_and_upsert(chunks, COLLECTION_INFORMATION)
#         embed_and_upsert(questions, COLLECTION_QUESTIONS)

#         logger.info(f"✅ Đã xử lý tổng cộng {len(chunks)} thông tin và {len(questions)} câu hỏi.")
#     except Exception as e:
#         logger.error(f"❌ Lỗi trong quá trình chính: {e}")
#         raise

# if __name__ == "__main__":
#     main()
import json
import uuid
import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Qdrant Cloud configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("⚠️ QDRANT_URL hoặc QDRANT_API_KEY không được cấu hình trong .env")

# File paths
CLEAN_CHUNKS_PATH = "D:/Vimedical/scripts/clean_chunks.json"
QUESTIONS_PATH = "D:/Vimedical/scripts/questions_merged.json"

# Collection names
COLLECTION_QUESTIONS = "vimedical-questions"
COLLECTION_INFORMATION = "vimedical-information"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Qdrant Cloud Client
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=120,
    check_compatibility=False
)

# Load embedding model
try:
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    logger.info("✅ Mô hình SentenceTransformer đã được tải thành công!")
except Exception as e:
    logger.error(f"❌ Lỗi khi tải mô hình SentenceTransformer: {e}")
    raise


# Hàm chuẩn hóa và sửa lỗi chính tả tên bệnh
def normalize_disease_name(disease_name):
    # Loại bỏ dấu cách thừa
    normalized = disease_name.strip().replace("  ", " ")
    
    
    # Viết hoa chữ đầu mỗi từ
    normalized = " ".join(word.capitalize() for word in normalized.split())
    return normalized

def load_json_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not data:
                raise ValueError(f"⚠️ File {path} rỗng.")
            return data
    except Exception as e:
        logger.error(f"❌ Lỗi khi tải file {path}: {e}")
        raise

def extract_chunks(data):
    chunks = []
    for item in data:
        disease = normalize_disease_name(item.get('title', '').split(':')[0].strip())
        for section in item.get("sections", []):
            text = section.get("content", "").strip()
            if len(text.split()) > 10:
                chunks.append({
                    "text": text,
                    "metadata": {
                        "disease": disease,
                        "source": item.get("source", ""),
                        "type": "information"
                    }
                })
    return chunks

def extract_questions(data):
    questions = []
    for disease, qs in data.items():
        disease = normalize_disease_name(disease)
        for q in qs:
            q_clean = q.strip()
            if len(q_clean.split()) > 5:
                questions.append({
                    "text": q_clean,
                    "metadata": {
                        "disease": disease,
                        "source": "questions_merged",
                        "type": "question"
                    }
                })
    return questions

def create_collection_with_index(collection_name):
    try:
        collections = qdrant_client.get_collections().collections
        if collection_name in [c.name for c in collections]:
            logger.info(f"🧹 Xóa collection cũ: {collection_name}")
            qdrant_client.delete_collection(collection_name)

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        logger.info(f"✅ Created collection {collection_name}")

        # Tạo payload index cho trường metadata.disease
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.disease",
            field_type="keyword"
        )
        logger.info(f"✅ Created index on 'metadata.disease' in {collection_name}")

    except Exception as e:
        logger.error(f"❌ Lỗi khi tạo collection {collection_name}: {e}")
        raise

def embed_and_upsert(texts, collection_name, batch_size=100):
    if not texts:
        logger.warning(f"⚠️ Không có dữ liệu để upsert vào {collection_name}.")
        return

    try:
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Upserting {collection_name}"):
            batch = texts[i:i + batch_size]
            vectors = model.encode([item["text"] for item in batch], show_progress_bar=False)
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec.tolist(),
                    payload={
                        "text": item["text"],
                        "metadata": item["metadata"]
                    }
                ) for item, vec in zip(batch, vectors)
            ]
            qdrant_client.upsert(collection_name=collection_name, points=points)
            logger.info(f"✅ Uploaded {len(points)} points to {collection_name}")
    except Exception as e:
        logger.error(f"❌ Lỗi khi upsert vào collection {collection_name}: {e}")
        raise

def main():
    try:
        chunks_data = load_json_file(CLEAN_CHUNKS_PATH)
        questions_data = load_json_file(QUESTIONS_PATH)

        chunks = extract_chunks(chunks_data)
        questions = extract_questions(questions_data)

        create_collection_with_index(COLLECTION_INFORMATION)
        create_collection_with_index(COLLECTION_QUESTIONS)

        embed_and_upsert(chunks, COLLECTION_INFORMATION)
        embed_and_upsert(questions, COLLECTION_QUESTIONS)

        logger.info(f"✅ Đã xử lý tổng cộng {len(chunks)} thông tin và {len(questions)} câu hỏi.")
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình chính: {e}")
        raise

if __name__ == "__main__":
    main()