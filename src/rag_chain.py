import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import logging

# Load environment variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_QUESTIONS = "vimedical-questions"
COLLECTION_INFORMATION = "vimedical-information"

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("⚠️ Thiếu QDRANT_URL hoặc QDRANT_API_KEY trong file .env")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hàm chuẩn hóa tên bệnh (giống với create_index.py)
def normalize_disease_name(disease_name):
    # Loại bỏ dấu cách thừa
    normalized = disease_name.strip().replace("  ", " ")
    
    # Viết hoa chữ đầu mỗi từ
    normalized = " ".join(word.capitalize() for word in normalized.split())
    return normalized

# Khởi tạo các vectorstore từ Qdrant Cloud
def load_vectorstores():
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        questions_vs = Qdrant(
            client=client,
            collection_name=COLLECTION_QUESTIONS,
            embeddings=embedding,
            content_payload_key="text",
            metadata_payload_key="metadata"
        )

        information_vs = Qdrant(
            client=client,
            collection_name=COLLECTION_INFORMATION,
            embeddings=embedding,
            content_payload_key="text",
            metadata_payload_key="metadata"
        )

        logger.info("✅ Đã tải thành công vectorstores từ Qdrant")
        return questions_vs, information_vs
    except Exception as e:
        logger.error(f"❌ Lỗi khi tải vectorstores: {e}")
        raise

# Tạo RAG Chain với 2 bước truy vấn
def get_qa_chain(memory=None):
    questions_vs, information_vs = load_vectorstores()

    def run(query, use_memory=False):
        try:
            # Nếu use_memory=True, truy xuất tên bệnh từ bộ nhớ
            if use_memory and memory:
                disease_name = memory.get("last_disease", None)
                if not disease_name:
                    logger.warning("⚠️ Không tìm thấy tên bệnh trong bộ nhớ.")
                    return {
                        "result": "Tôi không nhớ bệnh nào đã được đề cập trước đó. Vui lòng cung cấp thêm thông tin hoặc hỏi lại về triệu chứng.",
                        "disease": "",
                        "context": "",
                        "source_documents": []
                    }
                disease_name = normalize_disease_name(disease_name)
                logger.info(f"🔍 Sử dụng tên bệnh từ bộ nhớ: {disease_name}")

            else:
                # Bước 1: Truy xuất câu hỏi gần nhất để lấy metadata bệnh
                question_retriever = questions_vs.as_retriever(search_kwargs={"k": 1})
                question_docs = question_retriever.invoke(query)

                if not question_docs:
                    logger.warning(f"⚠️ Không tìm thấy câu hỏi tương tự cho: {query}")
                    return {
                        "result": "Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu y tế hiện có.",
                        "disease": "",
                        "context": "",
                        "source_documents": []
                    }

                # Truy xuất disease từ metadata và chuẩn hóa
                disease_name = question_docs[0].metadata.get("disease", "").strip()
                if not disease_name:
                    logger.warning(f"⚠️ Không tìm thấy tên bệnh trong metadata: {question_docs[0].metadata}")
                    return {
                        "result": "Không thể xác định tên bệnh từ câu hỏi tương tự.",
                        "disease": "",
                        "context": "",
                        "source_documents": []
                    }

                # Chuẩn hóa tên bệnh
                disease_name = normalize_disease_name(disease_name)
                logger.info(f"🔍 Tên bệnh tìm được: {disease_name}")

                # Lưu tên bệnh vào bộ nhớ
                if memory:
                    memory["last_disease"] = disease_name
                    logger.info(f"💾 Đã lưu tên bệnh vào bộ nhớ: {disease_name}")

            # Bước 2: Truy xuất thông tin liên quan tới bệnh đó
            filter_condition = Filter(
                must=[FieldCondition(key="metadata.disease", match=MatchValue(value=disease_name))]
            )
            info_retriever = information_vs.as_retriever(
                search_kwargs={"k": 4, "filter": filter_condition}
            )
            info_docs = info_retriever.invoke(query if not use_memory else disease_name)

            if not info_docs:
                logger.warning(f"⚠️ Không tìm thấy thông tin chi tiết cho bệnh: {disease_name}")
                return {
                    "result": f"Dựa trên triệu chứng bạn mô tả, bạn có thể đang gặp phải {disease_name}. Tôi khuyên bạn nên đi khám bác sĩ để được chẩn đoán và điều trị chính xác.",
                    "disease": disease_name,
                    "context": "",
                    "source_documents": [{"content": doc.page_content, "metadata": doc.metadata} for doc in question_docs] if not use_memory else []
                }

            context = "\n\n".join([doc.page_content for doc in info_docs])
            return {
                "context": context,
                "disease": disease_name,
                "source_documents": [{"content": doc.page_content, "metadata": doc.metadata} for doc in info_docs]
            }

        except Exception as e:
            logger.error(f"❌ Lỗi trong quá trình truy vấn: {e}")
            return {
                "result": f"Đã xảy ra lỗi trong quá trình truy vấn: {str(e)}",
                "disease": "",
                "context": "",
                "source_documents": []
            }

    return run