from sentence_transformers import CrossEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

intent_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

REFERENCE_LAST_PATTERNS = [
    "Bệnh này",
    "Bệnh vừa {context}",
    "Bệnh vừa đề cập",
    "Thông tin thêm về {disease}",
    "Chi tiết về bệnh này",
    "Nói thêm về bệnh vừa nêu",
    "Bệnh vừa nói đến"
]

INTENT_PATTERNS = {
    "info_new_disease": [
        "Cho tôi thông tin về {disease}",
        "Thông tin về {disease}",
        "Giải thích về {disease}",
        "Bệnh {disease} là gì",
        "Tìm hiểu về {disease}",
        "Tôi muốn biết về {disease}",
        "Tôi cần thông tin về {disease}",
        "Thông tin bệnh {disease}",
    ],
    "diagnose_new": [
        "Tôi bị {symptom} tôi có thể bị bệnh gì",
        "Tôi hiện đang có {symptom} tôi bị gì",
        "Có triệu chứng {symptom} tôi bị gì",
        "Cảm thấy {symptom} tôi có thể mắc bệnh gì",
        "Tôi thấy {symptom} tôi có bị bệnh không",
        "Tôi đang {symptom} có thể bị bệnh gì",
        "Tôi hiện đang có triệu chứng {symptom} tôi có thể bị bệnh gì",
        "Tôi đang gặp triệu chứng {symptom} tôi có thể bị bệnh gì",
        "Tôi thường xuyên bị {symptom} tôi có thể bị bệnh gì",
        "Tôi hay bị {symptom} tôi có thể bị bệnh gì",
        "Tôi hay gặp triệu chứng {symptom} tôi có thể bị bệnh gì",
    ],
    "diagnose_update": [
        "Tôi còn bị {symptom}",
        "Tôi còn thấy {symptom}",
        "Tôi còn",
        "Ngoài ra tôi còn {symptom}",
        "Ngoài ra",
        "Ngoài triêu chứng {symptom}",
        "Thêm triệu chứng {symptom}",
        "Cập nhật triệu chứng {symptom}",
        "Tôi cũng bị {symptom}",
        "Thêm vào {symptom}",
        "Tôi còn có triệu chứng {symptom}",
    ]
}

COMMON_SYMPTOMS = ["đau đầu", "sốt", "ho", "khó thở", "mệt", "chóng mặt", "buồn nôn", "ra máu", "đau ngực", "sưng phù"]

def check_symptom_overlap(query_symptoms, previous_symptoms):
    if not previous_symptoms:
        return False
    query_lower = query_symptoms.lower()
    prev_lower = previous_symptoms.lower()
    for symptom in COMMON_SYMPTOMS:
        if symptom in query_lower and symptom in prev_lower:
            return True
    return False

def extract_symptoms(query):
    query_lower = query.lower()
    found_symptoms = [symptom for symptom in COMMON_SYMPTOMS if symptom in query_lower]
    return " ".join(found_symptoms) if found_symptoms else query

def check_reference_last(query):
    query_lower = query.lower()
    for pattern in REFERENCE_LAST_PATTERNS:
        if "{context}" in pattern:
            pattern = pattern.format(context="nêu")
        pattern = pattern.replace("{disease}", "").strip()
        if pattern.lower() in query_lower:
            logger.info(f"🔍 Phát hiện từ khóa reference_last: {pattern}")
            return True
    return False

def detect_intent(query, previous_symptoms=""):
    logger.info(f"🔍 Đang phân tích ý định cho câu hỏi: {query}")
    query_symptoms = extract_symptoms(query)

    if check_reference_last(query):
        logger.info("🔍 Phát hiện ý định reference_last")
        return {"intent": "reference_last", "context": {"ask_confirmation": True}, "reset": False}

    intent_scores = {}
    for intent, patterns in INTENT_PATTERNS.items():
        scores = []
        for pattern in patterns:
            if "{symptom}" in pattern:
                pattern = pattern.format(symptom=query_symptoms if query_symptoms else "triệu chứng")
            score = intent_model.predict([(query, pattern)])[0]
            scores.append(score)
        intent_scores[intent] = max(scores) if scores else 0.0

    best_intent = max(intent_scores.items(), key=lambda x: x[1])[0] if intent_scores else None
    best_score = intent_scores.get(best_intent, 0.0)

    logger.info(f"🔍 Ý định phát hiện: {best_intent} với điểm {best_score}")

    if best_score < 0.5:
        return {"intent": None, "context": {"reset": True}}

    context = {}
    reset = False

    if best_intent == "info_new_disease":
        reset = True
        context["reset"] = True
    elif best_intent == "diagnose_new":
        if not check_symptom_overlap(query_symptoms, previous_symptoms):
            reset = True
            context["reset"] = True
        else:
            context["symptoms"] = query_symptoms
    elif best_intent == "diagnose_update":
        if previous_symptoms:
            context["symptoms"] = previous_symptoms
        else:
            reset = True
            context["reset"] = True

    return {"intent": best_intent, "context": context, "reset": reset}

def process_context(query, previous_symptoms=""):
    result = detect_intent(query, previous_symptoms)
    intent = result["intent"]
    context = result["context"]
    reset = result.get("reset", False)

    if intent == "reference_last":
        if context.get("ask_confirmation"):
            return {"query": "Bạn đang đề cập đến bệnh nào? Vui lòng cung cấp tên bệnh để tôi hỗ trợ tốt hơn.", "symptoms": previous_symptoms, "reset": False, "ask_confirmation": True}
        else:
            return {"query": query, "symptoms": previous_symptoms, "reset": False, "ask_confirmation": False}
    elif intent == "info_new_disease" or (intent == "diagnose_new" and context.get("reset")):
        return {"query": query, "symptoms": "", "reset": True, "ask_confirmation": False}
    elif intent == "diagnose_update" and context.get("symptoms"):
        combined_query = f"{context['symptoms']} {query}"
        return {"query": combined_query, "symptoms": context["symptoms"], "reset": False, "ask_confirmation": False}
    else:
        return {"query": query, "symptoms": previous_symptoms, "reset": reset, "ask_confirmation": False}