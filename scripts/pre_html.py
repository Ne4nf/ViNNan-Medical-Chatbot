import os
import json
from bs4 import BeautifulSoup
from tqdm import tqdm

INPUT_FOLDER = "data/Corpus"
OUTPUT_FILE = "scripts/clean_chunks.json"
CHUNK_WORDS = 150
EXCLUDE_KEYWORDS = ["HỆ THỐNG BỆNH VIỆN", "Fanpage", "Hotline", "Website", "Đặt lịch hẹn", "Mục lục"]


def collect_content(start_element, stop_tags):
    content = ""
    next_element = start_element.next_sibling
    while next_element and (not hasattr(next_element, 'name') or next_element.name not in stop_tags):
        if hasattr(next_element, 'get_text'):
            content += next_element.get_text().strip() + "\n"
        next_element = next_element.next_sibling
    return content.strip(), next_element

def parse_html_file(path, source_name):
    with open(path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    root = {
        "source": source_name,
        "title": "",
        "content": "",
        "sections": []
    }

    h1 = soup.find('h1')
    if h1:
        root["title"] = h1.get_text().strip()
        intro_content, next_element = collect_content(h1, ['h2'])
        root["content"] = intro_content
    else:
        root["title"] = "Untitled"

    h2_list = soup.find_all('h2')
    for h2 in h2_list:
        section = {
            "title": h2.get_text().strip(),
            "content": "",
            "subsections": []
        }

        section_content, next_element = collect_content(h2, ['h2', 'h3'])
        section["content"] = section_content

        while next_element and next_element.name == 'h3':
            h3 = next_element
            subsection = {
                "title": h3.get_text().strip(),
                "content": ""
            }

            subsection_content, next_element = collect_content(h3, ['h2', 'h3'])
            subsection["content"] = subsection_content
            section["subsections"].append(subsection)

        root["sections"].append(section)

    return root


def process_all_html():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    all_data = []

    files = sorted(f for f in os.listdir(INPUT_FOLDER) if f.endswith(".html"))
    print(f"📁 Đang xử lý {len(files)} file HTML...")

    for fname in tqdm(files):
        path = os.path.join(INPUT_FOLDER, fname)
        result = parse_html_file(path, source_name=fname)
        all_data.append(result)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Hoàn tất! Đã lưu kết quả vào {OUTPUT_FILE}")


if __name__ == "__main__":
    process_all_html()