from llm_chain import get_llm_chain
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    qa_chain = get_llm_chain()
    print("🤖 Chatbot Y tế đã sẵn sàng! Nhập câu hỏi của bạn hoặc gõ 'exit' để thoát.")
    while True:
        query = input("👤 Câu hỏi: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Tạm biệt! Chúc bạn sức khỏe!")
            break

        try:
            # Gọi hàm xử lý truy vấn
            result = qa_chain(query)

            # Trích xuất nội dung từ phản hồi
            answer = result.get("result", "Xin lỗi, tôi không thể trả lời câu hỏi này.")
            print("\n🤖 Trả lời:")
            print(answer)

            # Hiển thị tên bệnh nếu có
            disease = result.get("disease", "")
            if disease:
                print(f"\n🩺 Bệnh liên quan: {disease}")

            # Hiển thị nguồn dữ liệu (nếu có)
            if "source_documents" in result and result["source_documents"]:
                print("\n📚 Nguồn dữ liệu:")
                for doc in result["source_documents"]:
                    print(f"- {doc['metadata'].get('source', 'unknown')}: {doc['content'][:100]}...")
                print("\n" + "-" * 50)
        except Exception as e:
            print(f"❌ Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    main()