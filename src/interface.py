import streamlit as st
from llm_chain import get_llm_chain
from datetime import datetime

st.markdown(
    """
    <h1 style='text-align: center; color: #1E88E5;'>💊 ViNNan - Trợ lý Y Tế Thông minh</h1>
    <p style='text-align: center; color: #555555;'>Hỏi đáp y tế bằng Tiếng Việt, chuẩn đoán và truy xuất nhanh chóng, chính xác.</p>
    <hr style='border-color: #BBDEFB;'>
    """,
    unsafe_allow_html=True
)

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = get_llm_chain()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = {"symptoms": ""}

if not st.session_state.messages:
    timestamp = datetime.now().strftime("%H:%M:%S")
    greeting = "Xin chào! Tôi là ViNNan - Chatbot y tế tự động, hỗ trợ chuẩn đoán bệnh và truy xuất thông tin y tế. Bạn có thể nêu triệu chứng hoặc tên bệnh để tôi giúp bạn một cách chi tiết và chính xác nhất. Hãy bắt đầu nào!"
    st.session_state.messages.append({
        "role": "assistant",
        "content": greeting,
        "timestamp": timestamp
    })

for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    timestamp = msg["timestamp"]
    avatar = "😷" if role == "user" else "💊"
    with st.chat_message(role, avatar=avatar):
        st.markdown(f"{content} <span style='color:#888888;font-size:0.8em'>[{timestamp}]</span>", unsafe_allow_html=True)

query = st.chat_input("Nhập câu hỏi của bạn bằng tiếng Việt...")

if query:
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    with st.chat_message("user", avatar="😷"):
        st.markdown(f"{query} <span style='color:#888888;font-size:0.8em'>[{timestamp}]</span>", unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": query, "timestamp": timestamp})

    
    with st.chat_message("assistant", avatar="💊"):
        with st.spinner("🤖 Đang xử lý..."):
            
            if "bệnh này" in query.lower() or "căn bệnh này" in query.lower():
                confirmation_query = "Bạn đang đề cập đến bệnh nào? Vui lòng cung cấp tên bệnh để tôi hỗ trợ tốt hơn."
                st.markdown(f"{confirmation_query} <span style='color:#888888;font-size:0.8em'>[{timestamp}]</span>", unsafe_allow_html=True)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": confirmation_query,
                    "timestamp": timestamp
                })
            else:
                
                result = st.session_state.qa_chain(
                    query,
                    previous_symptoms=st.session_state.memory["symptoms"]
                )
                answer = result.get("result", "Xin lỗi, tôi không thể trả lời câu hỏi này.")
                st.session_state.memory["symptoms"] = result.get("symptoms", st.session_state.memory["symptoms"])
                possible_diseases = result.get("possible_diseases", [])

                
                response_content = answer
                if possible_diseases:
                    response_content += "\n\n🩺 **Các bệnh có thể liên quan:**\n"
                    for i, disease in enumerate(possible_diseases[:3], 1):
                        response_content += f"{i}. {disease}\n"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_content,
                    "timestamp": timestamp
                })

                
                st.markdown(f"{answer} <span style='color:#888888;font-size:0.8em'>[{timestamp}]</span>", unsafe_allow_html=True)
                if possible_diseases:
                    st.markdown("🩺 **Các bệnh có thể liên quan:**")
                    for i, disease in enumerate(possible_diseases[:3], 1):
                        st.markdown(f"{i}. {disease}")