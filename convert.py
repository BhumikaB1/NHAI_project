import json

with open(r"C:\Users\harsh\Desktop\startup_manufacturing.jsonl", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("startup.jsonl", "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")