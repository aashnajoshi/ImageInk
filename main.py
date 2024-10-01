import tkinter as tk
from tkinter import filedialog
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI 
from dotenv import load_dotenv
import os

load_dotenv()

vision_endpoint = os.getenv('AZURE_VISION_ENDPOINT')
vision_key = os.getenv('AZURE_VISION_KEY')
openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
openai_key = os.getenv('AZURE_OPENAI_KEY')
openai_deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')

vision_client = ImageAnalysisClient(endpoint=vision_endpoint, credential=AzureKeyCredential(vision_key))
openai_client = AzureOpenAI(azure_endpoint=openai_endpoint, api_key=openai_key, api_version="2024-02-15-preview")

def analyze_image(image_path):
    print('\nAnalyzing image...')
    try:
        with open(image_path, "rb") as f:
            result = vision_client.analyze(image_data=f.read(), visual_features=["caption", "denseCaptions", "tags", "objects"])
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

def display_analysis_results(result):
    captions = [result.caption.text] if result.caption else []
    for caption in result.dense_captions.list:
        captions.append(f"'{caption.text}' (confidence: {caption.confidence * 100:.2f}%)")
    return captions[:5]  # Return only the top 5 captions

def generate_story(caption):
    prompt = f"Write a creative story in 200 words based on this caption: {caption}"
    response = openai_client.chat.completions.create(
        model=openai_deployment_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=400)
    return response.choices[0].message.content.strip()

def main():
    tk.Tk().withdraw()
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if not image_path:
        print("No file selected.")
        return

    result = analyze_image(image_path)

    if result:
        print("Captions derived from image:")
        for i, caption in enumerate(display_analysis_results(result), start=1):
            print(f"{i}. {caption}")
        
        if result.caption:
            creative_title = f"An Imaginative Tale Based on: '{result.caption.text}'"
            print("\n" + creative_title)
            print("\n" + generate_story(result.caption.text))

if __name__ == "__main__":
    main()