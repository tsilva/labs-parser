import os
import json
import PyPDF2
from langchain_openai import OpenAI

# Initialize Langchain with OpenAI
llm = OpenAI(api_key='your_openai_api_key')

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_data_with_langchain(text):
    # Create a prompt for extracting the specific information
    prompt_text = f"Extract date, lab name, value, unit, and range from the following text:\n{text}"
    response = llm.generate(prompt_text, max_tokens=100)
    return response

def save_as_json(data, json_path):
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)

def process_pdfs_in_directory(directory):
    for filename in os.listdir(directory):
        if not filename.endswith('.pdf'): continue
        if not "analises" in filename: continue

        base_name = filename[:-4]
        pdf_path = os.path.join(directory, filename)

        # Extract text from PDF
        try: text = extract_text_from_pdf(pdf_path)
        except: continue
        text_file_path = os.path.join(directory, base_name + '.txt')
        with open(text_file_path, 'w') as text_file:
            print(text_file_path)
            text_file.write(text)

        # Extract data using Langchain and save as JSON
        #extracted_data = extract_data_with_langchain(text)
        #json_file_path = os.path.join(directory, base_name + '.json')
        #save_as_json(extracted_data, json_file_path)

# Specify your directory path here
process_pdfs_in_directory("data/labs")
