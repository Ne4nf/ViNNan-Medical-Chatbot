import pandas as pd
import json

df = pd.read_csv("data/ViMedical_Disease.csv")

disease_questions = {}
for _, row in df.iterrows():
    disease = row["Disease"]
    question = row["Question"]

    if disease not in disease_questions:
        disease_questions[disease] = []
    disease_questions[disease].append(question)

with open("scripts/questions_merged.json", "w", encoding="utf-8") as f:
    json.dump(disease_questions, f, ensure_ascii=False, indent=2)

print(f"✅ Saved questions_merged.json with {len(disease_questions)} diseases.")
