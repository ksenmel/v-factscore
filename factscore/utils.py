import json

def extract_outputs_from_jsonl(file_path: str) -> list[str]:
    outputs = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            outputs.append(data["output"])
    return outputs
