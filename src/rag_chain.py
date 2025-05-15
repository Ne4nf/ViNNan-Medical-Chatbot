import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import CrossEncoder
import logging
from tools import process_context

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_QUESTIONS = "vimedical-questions"
COLLECTION_INFORMATION = "vimedical-information"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def normalize_disease_name(name):
    return " ".join(w.capitalize() for w in name.strip().split())

def is_disease_name(query, known_diseases):
    from fuzzywuzzy import fuzz
    query_lower = query.lower()
    for disease in known_diseases:
        disease_lower = disease.lower()
        if fuzz.partial_ratio(disease_lower, query_lower) > 90 or f"bệnh {disease_lower}" in query_lower:
            return disease
    return None

def load_vectorstores():
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

    return questions_vs, information_vs

def get_qa_chain():
    questions_vs, information_vs = load_vectorstores()

    known_diseases = set()
    try:
        info_docs = information_vs.as_retriever(search_kwargs={"k": 8317}).invoke("all diseases")
        for doc in info_docs:
            disease = doc.metadata.get("disease", "").strip()
            if disease:
                known_diseases.add(normalize_disease_name(disease))
        logger.info(f"🔍 Đã tải {len(known_diseases)} bệnh từ collection_information")
    except Exception as e:
        logger.error(f"❌ Lỗi khi lấy danh sách bệnh: {e}")

    def run(query, previous_symptoms=""):
        try:
            logger.info(f"🔍 Xử lý câu hỏi: {query}")
            context_result = process_context(query, previous_symptoms)
            processed_query = context_result["query"]
            new_symptoms = context_result["symptoms"]
            reset = context_result.get("reset", False)
            ask_confirmation = context_result.get("ask_confirmation", False)

            if ask_confirmation:
                logger.info("🔍 Yêu cầu xác nhận bệnh, không truy xuất thông tin.")
                return {
                    "result": processed_query,
                    "disease": "",
                    "possible_diseases": [],
                    "context": "",
                    "source_documents": [],
                    "symptoms": new_symptoms,
                    "ask_confirmation": True
                }

            if reset:
                logger.info(f"🔍 Reset ngữ cảnh")
                new_symptoms = ""

            if (disease := is_disease_name(processed_query, known_diseases)):
                logger.info(f"🔍 Phát hiện tên bệnh: {disease}")
                disease_detected = disease
            else:
                question_retriever = questions_vs.as_retriever(search_kwargs={"k": 20})
                question_docs = question_retriever.invoke(processed_query)
                if not question_docs:
                    return {
                        "result": "Tôi không tìm thấy thông tin phù hợp. Vui lòng mô tả rõ hơn hoặc nêu tên bệnh.",
                        "disease": "",
                        "possible_diseases": [],
                        "context": "",
                        "source_documents": [],
                        "symptoms": new_symptoms,
                        "ask_confirmation": False
                    }

                ranked_docs = []
                for doc in question_docs:
                    score = reranker.predict([(processed_query, doc.page_content)])[0]
                    ranked_docs.append({"content": doc.page_content, "metadata": doc.metadata, "score": score})
                ranked_docs = sorted(ranked_docs, key=lambda x: x["score"], reverse=True)

                disease_scores = {}
                for doc in ranked_docs:
                    disease = doc["metadata"].get("disease", "").strip()
                    if disease:
                        disease = normalize_disease_name(disease)
                        disease_scores[disease] = disease_scores.get(disease, 0) + doc["score"]

                if not disease_scores:
                    return {
                        "result": "Tôi chưa xác định được bệnh cụ thể. Vui lòng cung cấp thêm thông tin.",
                        "disease": "",
                        "possible_diseases": [],
                        "context": "",
                        "source_documents": [],
                        "symptoms": new_symptoms,
                        "ask_confirmation": False
                    }

                sorted_candidates = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
                top1_score = sorted_candidates[0][1]
                top2_score = sorted_candidates[1][1] if len(sorted_candidates) > 1 else 0

                if top1_score > 1.125 * top2_score and top1_score >= 0.92:
                    disease_detected = sorted_candidates[0][0]
                else:
                    top3 = [name for name, _ in sorted_candidates[:3]]
                    return {
                        "result": f"Tôi chưa chắc chắn. Bạn có thể đang mắc một trong các bệnh: {', '.join(top3)}. Vui lòng chọn bệnh hoặc cung cấp thêm thông tin.",
                        "disease": "",
                        "possible_diseases": top3,
                        "context": "",
                        "source_documents": [],
                        "symptoms": new_symptoms,
                        "ask_confirmation": False
                    }

            filter_condition = Filter(must=[FieldCondition(key="metadata.disease", match=MatchValue(value=disease_detected))])
            info_docs = information_vs.as_retriever(search_kwargs={"k": 6, "filter": filter_condition}).invoke(disease_detected)
            if info_docs:
                context = "\n\n".join([doc.page_content for doc in info_docs])
                return {
                    "result": f"Đây là thông tin chi tiết về {disease_detected}:",
                    "disease": disease_detected,
                    "possible_diseases": [disease_detected],
                    "context": context,
                    "source_documents": [{"content": doc.page_content, "metadata": doc.metadata} for doc in info_docs],
                    "symptoms": new_symptoms,
                    "ask_confirmation": False
                }
            else:
                return {
                    "result": f"Tôi chưa tìm thấy thông tin chi tiết về {disease_detected}.",
                    "disease": disease_detected,
                    "possible_diseases": [disease_detected],
                    "context": "",
                    "source_documents": [],
                    "symptoms": new_symptoms,
                    "ask_confirmation": False
                }

        except Exception as e:
            logger.error(f"❌ Lỗi trong truy vấn: {e}")
            return {
                "result": f"Đã xảy ra lỗi: {str(e)}",
                "disease": "",
                "possible_diseases": [],
                "context": "",
                "source_documents": [],
                "symptoms": new_symptoms,
                "ask_confirmation": False
            }

    return run