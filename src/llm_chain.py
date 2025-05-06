# from langchain_community.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate
# import os
# from dotenv import load_dotenv
# from rag_chain import get_qa_chain
# import logging

# # Load environment variables
# load_dotenv()

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Tải mô hình LLM hỗ trợ tiếng Việt
# def load_llm():
#     """Tải LLM từ OpenRouter"""
#     return ChatOpenAI(
#         model="deepseek/deepseek-r1-distill-llama-70b:free",
#         openai_api_key=os.getenv("OPENROUTER_API_KEY"),
#         base_url="https://openrouter.ai/api/v1",
#         temperature=0.5
#     )

# # Prompt tùy chỉnh để trả lời bằng tiếng Việt
# def custom_prompt():
#     prompt = """
# Bạn là một trợ lý y tế thông minh. Hãy trả lời câu hỏi của người dùng dựa trên thông tin bên dưới.
# Đừng cố gắng đưa ra câu trả lời nếu không có thông tin liên quan trong tài liệu y tế (context).
# Nếu không biết câu trả lời, hãy trả lời như sau: "Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu y tế hiện có."

# Thông tin y tế:
# {context}

# Câu hỏi: {question}

# Hãy trả lời bằng tiếng Việt, ngắn gọn, rõ ràng. Nếu có thể, hãy đề cập đến tên bệnh lý hoặc triệu chứng liên quan.
# """
#     return PromptTemplate(template=prompt, input_variables=["context", "question"])

# # Hàm kiểm tra xem câu hỏi có yêu cầu thông tin thêm hay không
# def is_requesting_more_info(query):
#     more_info_keywords = [
#         "thông tin thêm", "chi tiết", "nói thêm", "giải thích", "tìm hiểu thêm",
#         "thêm thông tin", "hiểu thêm", "thông tin chi tiết", "bệnh này", "về bệnh này"
#     ]
#     query_lower = query.lower()
#     return any(keyword in query_lower for keyword in more_info_keywords)

# # Tạo LLM Chain
# def get_llm_chain():
#     llm = load_llm()
#     prompt = custom_prompt()
#     # Khởi tạo rag_chain với bộ nhớ
#     memory = {"last_disease": None}
#     rag_chain = get_qa_chain(memory=memory)

#     def run(query):
#         try:
#             # Xác định loại câu hỏi
#             if is_requesting_more_info(query):
#                 # Nếu yêu cầu thông tin thêm, sử dụng tên bệnh từ bộ nhớ
#                 logger.info("🔍 Yêu cầu thông tin thêm, sử dụng tên bệnh từ bộ nhớ...")
#                 response = rag_chain(query, use_memory=True)
#             else:
#                 # Nếu không, thực hiện truy vấn bình thường (bước 1: đoán bệnh)
#                 logger.info("🔍 Xử lý câu hỏi triệu chứng...")
#                 response = rag_chain(query, use_memory=False)

#             # Xử lý phản hồi từ rag_chain
#             if "result" in response:
#                 return {
#                     "context": response.get("context", ""),
#                     "disease": response.get("disease", ""),
#                     "result": response["result"],
#                     "source_documents": response.get("source_documents", [])
#                 }

#             # Lấy context và disease từ response
#             context = response.get("context", "")
#             disease = response.get("disease", "bệnh không xác định")
#             source_docs = response.get("source_documents", [])

#             # Chuẩn bị câu trả lời từ LLM nếu có context
#             if context:
#                 prompt_input = prompt.format(context=context, question=query)
#                 answer = llm.invoke(prompt_input).content
#             else:
#                 # Nếu không có context, tạo câu trả lời cơ bản từ disease
#                 answer = f"Tôi không tìm thấy thông tin chi tiết về {disease} trong tài liệu y tế hiện có. Tuy nhiên, dựa trên triệu chứng bạn mô tả, bạn có thể đang gặp phải {disease}. Hãy đến khám bác sĩ để được chẩn đoán chính xác."

#             # Ghi log xử lý
#             logger.info(f"🔍 Context: {context[:50]}... | Disease: {disease} | Answer: {answer}")

#             # Trả về dữ liệu cho agent.py xử lý
#             return {
#                 "context": context,
#                 "disease": disease,
#                 "result": answer,
#                 "source_documents": source_docs
#             }

#         except Exception as e:
#             logger.error(f"❌ Lỗi trong LLM chain: {e}")
#             return {
#                 "context": "",
#                 "disease": "",
#                 "result": f"Đã xảy ra lỗi: {str(e)}",
#                 "source_documents": []
#             }

#     return run 

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from rag_chain import get_qa_chain
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tải mô hình LLM hỗ trợ tiếng Việt
def load_llm():
    """Tải LLM từ OpenRouter"""
    return ChatOpenAI(
        model="deepseek/deepseek-r1-distill-llama-70b:free",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0.5
    )

# Prompt tùy chỉnh để trả lời bằng tiếng Việt
def custom_prompt():
    prompt = """
Bạn là một trợ lý y tế thông minh. Hãy trả lời câu hỏi của người dùng dựa trên thông tin bên dưới.
Đừng cố gắng đưa ra câu trả lời nếu không có thông tin liên quan trong tài liệu y tế (context).
Nếu không biết câu trả lời, hãy trả lời như sau: "Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu y tế hiện có."

Thông tin y tế:
{context}

Câu hỏi: {question}

Hãy trả lời bằng tiếng Việt, ngắn gọn, rõ ràng. Nếu có thể, hãy đề cập đến tên bệnh lý hoặc triệu chứng liên quan.
"""
    return PromptTemplate(template=prompt, input_variables=["context", "question"])

# Hàm kiểm tra xem câu hỏi có yêu cầu thông tin thêm hay không
def is_requesting_more_info(query):
    more_info_keywords = [
        "thông tin thêm", "chi tiết", "nói thêm", "giải thích", "tìm hiểu thêm",
        "thêm thông tin", "hiểu thêm", "thông tin chi tiết", "bệnh này", "về bệnh này"
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in more_info_keywords)

# Tạo LLM Chain
def get_llm_chain():
    llm = load_llm()
    prompt = custom_prompt()
    # Khởi tạo rag_chain với bộ nhớ
    memory = {"last_disease": None}
    rag_chain = get_qa_chain(memory=memory)

    def run(query, last_disease=None):
        try:
            # Đồng bộ last_disease từ interface nếu có
            if last_disease and last_disease != "bệnh không xác định":
                memory["last_disease"] = last_disease
                logger.info(f"🔄 Đã đồng bộ last_disease từ interface: {last_disease}")

            # Xác định loại câu hỏi
            if is_requesting_more_info(query):
                # Nếu yêu cầu thông tin thêm, sử dụng tên bệnh từ bộ nhớ
                logger.info("🔍 Yêu cầu thông tin thêm, sử dụng tên bệnh từ bộ nhớ...")
                if not memory["last_disease"]:
                    return {
                        "context": "",
                        "disease": "",
                        "result": "Vui lòng cung cấp tên bệnh hoặc hỏi về triệu chứng trước.",
                        "source_documents": []
                    }
                response = rag_chain(query, use_memory=True)
            else:
                # Nếu không, thực hiện truy vấn bình thường (bước 1: đoán bệnh)
                logger.info("🔍 Xử lý câu hỏi triệu chứng...")
                response = rag_chain(query, use_memory=False)

            # Xử lý phản hồi từ rag_chain
            if "result" in response:
                return {
                    "context": response.get("context", ""),
                    "disease": response.get("disease", ""),
                    "result": response["result"],
                    "source_documents": response.get("source_documents", [])
                }

            # Lấy context và disease từ response
            context = response.get("context", "")
            disease = response.get("disease", "bệnh không xác định")
            source_docs = response.get("source_documents", [])

            # Chuẩn bị câu trả lời từ LLM nếu có context
            if context:
                prompt_input = prompt.format(context=context, question=query)
                answer = llm.invoke(prompt_input).content
            else:
                # Nếu không có context, tạo câu trả lời cơ bản từ disease
                answer = f"Tôi không tìm thấy thông tin chi tiết về {disease} trong tài liệu y tế hiện có. Tuy nhiên, dựa trên triệu chứng bạn mô tả, bạn có thể đang gặp phải {disease}. Hãy đến khám bác sĩ để được chẩn đoán chính xác."

            # Ghi log xử lý
            logger.info(f"🔍 Context: {context[:50]}... | Disease: {disease} | Answer: {answer}")

            # Trả về dữ liệu
            return {
                "context": context,
                "disease": disease,
                "result": answer,
                "source_documents": source_docs
            }

        except Exception as e:
            logger.error(f"❌ Lỗi trong LLM chain: {e}")
            return {
                "context": "",
                "disease": "",
                "result": f"Đã xảy ra lỗi: {str(e)}",
                "source_documents": []
            }

    return run