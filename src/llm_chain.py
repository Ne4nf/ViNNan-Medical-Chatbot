import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag_chain import get_qa_chain
from tools import process_context
import logging

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = ChatOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model_name="mistralai/mixtral-8x7b-instruct",
    temperature=0.5
)

prompt_template = """
Bạn là một trợ lý y tế thông minh, trả lời câu hỏi y tế bằng tiếng Việt một cách chính xác và rõ ràng. Dựa trên ngữ cảnh và câu hỏi, thực hiện như sau:

1. Nếu yêu cầu thông tin về bệnh, cung cấp chi tiết về bệnh đó.
2. Nếu là chuẩn đoán, phân tích và đưa ra bệnh khả năng cao hoặc gợi ý nếu không chắc chắn.
3. Nếu câu hỏi yêu cầu xác nhận (ví dụ: 'Bạn đang đề cập đến bệnh nào?'), trả lời theo đúng yêu cầu đó.
4. Luôn phân biệt 'đau đầu', 'sốt', 'ho', 'khó thở', 'mệt' hoặc các từ khóa tương tự là triệu chứng khi đi kèm mô tả, không phải tên bệnh.
5. Luôn khuyến khích người dùng đến bác sĩ nếu triệu chứng nghiêm trọng.

**Ngữ cảnh:** {context}
**Câu hỏi:** {question}
**Triệu chứng trước đó (nếu có):** {previous_symptoms}

**Phản hồi:**
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
output_parser = StrOutputParser()

def get_llm_chain():
    qa_chain = get_qa_chain()

    def run(query, previous_symptoms=""):
        try:
            logger.info(f"🔍 Xử lý câu hỏi LLM: {query}")
            context_result = process_context(query, previous_symptoms)
            processed_query = context_result["query"]
            new_symptoms = context_result["symptoms"]
            ask_confirmation = context_result.get("ask_confirmation", False)

            result = qa_chain(processed_query, previous_symptoms=new_symptoms)

            if result.get("ask_confirmation", False):
                logger.info("🔍 ask_confirmation được kích hoạt, trả về câu hỏi xác nhận mà không gọi LLM.")
                return result

            context = result.get("context", "")
            if not context and result.get("possible_diseases"):
                context = f"Các bệnh có thể liên quan: {', '.join(result['possible_diseases'])}"

            input_data = {
                "context": context,
                "question": query,
                "previous_symptoms": new_symptoms if new_symptoms else ""
            }

            response = prompt | llm | output_parser
            final_response = response.invoke(input_data)

            result["result"] = final_response
            return result

        except Exception as e:
            logger.error(f"❌ Lỗi trong LLM chain: {e}")
            return {
                "result": f"Đã xảy ra lỗi: {str(e)}",
                "disease": "",
                "possible_diseases": [],
                "context": "",
                "source_documents": [],
                "symptoms": previous_symptoms,
                "ask_confirmation": False
            }

    return run

def is_reference_to_last_disease(query):
    from tools import detect_intent
    return detect_intent(query).get("intent") == "reference_last"