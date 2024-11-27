import os
import json

def extract_texts_from_json(file_path):
    """
    Extracts the values associated with the 'text' key from JSON objects in a file.

    :param file_path: Path to the JSON file.
    :return: List of extracted text values.
    """
    extracted_texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'text' in data:
                    extracted_texts.append(data['text'])
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {file_path}: {e}")
    return "".join(extracted_texts)


def extract_text_from_directory(directory, text):
    """
    Walk through all folders in a directory and extract text from JSON files.

    :param directory: Path to the main directory.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jsonl'):
                file_path = os.path.join(root, file)
                text += extract_texts_from_json(file_path) +'\n'
    return text

if __name__ == "__main__":
    main_directory = '/scratch/anxtem001/filtered_data'
    text =  extract_text_from_directory(main_directory,'')
    file = open("/scratch/anxtem001/extracted_data.txt", "w")
    file.write(text)