import os
import logging
from dotenv import load_dotenv
import json
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
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

# Normalize disease name
def normalize_disease_name(disease_name):
    normalized = disease_name.strip().replace("  ", " ")
    normalized = " ".join(word.capitalize() for word in normalized.split())
    return normalized

# Load JSON file
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

# Remove advertisements
def remove_ads(text):
    ad_keywords = ["hotline", "liên hệ", "bệnh viện", "đăng ký", "ưu đãi", "giảm giá","Fangpage", "website", "đặt lịch hẹn", "mục lục"]
    return not any(keyword in text.lower() for keyword in ad_keywords)

# Extract chunks with subsections
def extract_chunks(data):
    chunks = []
    for item in tqdm(data, desc="Extracting chunks"):
        disease = normalize_disease_name(item.get('title', '').split(':')[0].strip())
        section_title = item.get('title', 'No Title')
        for section in item.get("sections", []):
            text = section.get("content", "").strip()
            if text and len(text) > 100 and len(text.split()) > 10 and remove_ads(text):
                chunks.append({
                    "text": text,
                    "metadata": {
                        "disease": disease,
                        "source": item.get("source", ""),
                        "type": "information",
                        "section_title": section_title,
                        "subsection_title": "Main Content"
                    }
                })
            # Handle subsections
            for subsection in section.get("subsections", []):
                sub_text = subsection.get("content", "").strip()
                sub_title = subsection.get("title", "No Subsection")
                if sub_text and len(sub_text) > 100 and len(sub_text.split()) > 10 and remove_ads(sub_text):
                    chunks.append({
                        "text": sub_text,
                        "metadata": {
                            "disease": disease,
                            "source": item.get("source", ""),
                            "type": "information",
                            "section_title": section_title,
                            "subsection_title": sub_title
                        }
                    })
    return chunks

# Extract questions
def extract_questions(data):
    questions = []
    for disease, qs in tqdm(data.items(), desc="Extracting questions"):
        disease = normalize_disease_name(disease)
        for q in qs:
            q_clean = q.strip()
            if len(q_clean.split()) > 5 and remove_ads(q_clean):
                questions.append({
                    "text": q_clean,
                    "metadata": {
                        "disease": disease,
                        "source": "questions_merged",
                        "type": "question"
                    }
                })
    return questions

# Create collection with index
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

        # Create payload index for metadata fields
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.disease",
            field_type="keyword"
        )
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.section_title",
            field_type="keyword"
        )
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.subsection_title",
            field_type="keyword"
        )
        logger.info(f"✅ Created indexes on metadata fields in {collection_name}")

    except Exception as e:
        logger.error(f"❌ Lỗi khi tạo collection {collection_name}: {e}")
        raise

# Embed and upsert
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

        # Extract chunks (without deduplication)
        chunks = extract_chunks(chunks_data)
        logger.info(f"✅ Extracted {len(chunks)} chunks.")

        # Extract questions
        questions = extract_questions(questions_data)
        logger.info(f"✅ Extracted {len(questions)} questions.")

        # Create collections
        create_collection_with_index(COLLECTION_INFORMATION)
        create_collection_with_index(COLLECTION_QUESTIONS)

        # Upsert data
        embed_and_upsert(chunks, COLLECTION_INFORMATION, batch_size=100)
        embed_and_upsert(questions, COLLECTION_QUESTIONS, batch_size=100)

        logger.info(f"✅ Đã xử lý tổng cộng {len(chunks)} thông tin và {len(questions)} câu hỏi.")
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình chính: {e}")
        raise

if __name__ == "__main__":
    main()