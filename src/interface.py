import os
import streamlit as st
from llm_chain import get_llm_chain
from datetime import datetime

# Cấu hình trang
st.set_page_config(
    page_title="🧠 Chatbot Y tế Thông minh",
    page_icon="💊",
    layout="wide"
)

# Tiêu đề và mô tả
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>💊 Trợ lý Y Tế Thông minh</h1>
    <p style='text-align: center; color: gray;'>Hỏi đáp y tế bằng tiếng Việt, trả lời nhanh chóng và chính xác.</p>
    <hr>
    """, 
    unsafe_allow_html=True
)

# Lưu qa_chain vào session_state để tránh khởi tạo lại
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = get_llm_chain()

# Lưu lịch sử hội thoại và bệnh gần nhất
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_disease" not in st.session_state:
    st.session_state.last_disease = None

# Sidebar với cấu trúc danh mục
with st.sidebar:
    st.markdown("### 📅 Menu")
    
    # Danh mục Thông tin hôm nay
    st.markdown("#### Hôm nay")
    if st.button("Thông tin bệnh gần nhất", key="today_info"):
        if st.session_state.last_disease:
            st.write(f"**Bệnh gần nhất:** {st.session_state.last_disease}")
        else:
            st.write("Chưa có thông tin bệnh gần nhất.")
    
    # Danh mục Lịch sử 7 ngày
    st.markdown("#### Lịch sử 7 ngày")
    if st.button("Xem lịch sử 7 ngày", key="history_7days"):
        st.write("Chức năng xem lịch sử 7 ngày (chưa triển khai).")
    
    # Danh mục Lịch sử 30 ngày
    st.markdown("#### Lịch sử 30 ngày")
    if st.button("Xem lịch sử 30 ngày", key="history_30days"):
        st.write("Chức năng xem lịch sử 30 ngày (chưa triển khai).")
    
    # Nút xóa lịch sử
    st.markdown("#### Cài đặt")
    if st.button("Xóa lịch sử hội thoại", key="clear_history"):
        clear_chat_history()
        st.success("Lịch sử hội thoại đã được xóa!")

# Hàm để xóa lịch sử hội thoại
def clear_chat_history():
    st.session_state.messages = []
    st.session_state.last_disease = None
    st.session_state.qa_chain = get_llm_chain()

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    timestamp = msg["timestamp"]
    avatar = "🧑‍⚕️" if role == "user" else "💊"
    color = "#0B5394" if role == "user" else "#4CAF50"
    with st.chat_message(role, avatar=avatar):
        st.markdown(f"<div style='color:{color}'>{content}</div>", unsafe_allow_html=True)
        st.caption(f"{timestamp}")

# Nhận input từ người dùng
query = st.chat_input("Nhập câu hỏi của bạn bằng tiếng Việt...")

if query:
    # Thêm tin nhắn người dùng vào lịch sử
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({"role": "user", "content": query, "timestamp": timestamp})
    st.chat_message("user", avatar="🧑‍⚕️").markdown(query)

    # Lấy phản hồi từ LLM
    with st.chat_message("assistant", avatar="💊"):
        with st.spinner("🤖 Đang truy xuất dữ liệu y khoa..."):
            result = st.session_state.qa_chain(query)

            # Lấy câu trả lời và bệnh từ kết quả
            answer = result.get("result", "Xin lỗi, tôi không thể trả lời câu hỏi này.")
            disease = result.get("disease", None)

            # Lưu bệnh gần nhất vào trạng thái phiên nếu có
            if disease and disease != "bệnh không xác định":
                st.session_state.last_disease = disease

            # Hiển thị câu trả lời
            st.markdown(f"<div style='color:#4CAF50'>{answer}</div>", unsafe_allow_html=True)

            # Hiển thị tên bệnh nếu có
            if disease and disease != "bệnh không xác định":
                st.markdown(f"🩺 **Bệnh liên quan:** {disease}")

            # Hiển thị nguồn tài liệu nếu có
            if "source_documents" in result and result["source_documents"]:
                with st.expander("📚 Nguồn tham khảo"):
                    for doc in result["source_documents"]:
                        st.markdown(f"""
                        - **Nguồn:** `{doc['metadata'].get('source', 'Không rõ')}`  
                        - **Trích đoạn:** {doc['content'][:100]}...
                        """)
            else:
                st.info("Không có nguồn tài liệu liên quan được tìm thấy.")

            st.session_state.messages.append({"role": "assistant", "content": answer, "timestamp": timestamp})

# Thêm footer
# st.markdown(
#     """
#     <hr>
#     <p style='text-align: center; color: gray;'>© 2023 Trợ lý Y Tế Thông minh. Tất cả quyền được bảo lưu.</p>
#     """, 
#     unsafe_allow_html=True
# )