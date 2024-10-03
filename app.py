import streamlit as st
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

def analyze_image(image_data):
    """Analyze the image using Azure Vision API."""
    try:
        result = vision_client.analyze(image_data=image_data, visual_features=["caption", "denseCaptions", "tags", "objects"])
        return result
    except Exception as e:
        st.error(f"Error analyzing image: {e}")
        return None

def display_analysis_results(result):
    """Extract captions from analysis results."""
    captions = [result.caption.text] if result.caption else []
    for caption in result.dense_captions.list:
        captions.append(f"'{caption.text}' (confidence: {caption.confidence * 100:.2f}%)")
    return captions[:5]  # Return only the top 5 captions

def generate_story(caption):
    """Generate a story based on the caption using OpenAI API."""
    prompt = f"Write a creative story in 200 words based on this caption: {caption}"
    response = openai_client.chat.completions.create(
        model=openai_deployment_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=400)
    return response.choices[0].message.content.strip()

def main():
    """Main function to run the Streamlit app."""
    st.title("Imageink")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        image_data = uploaded_file.read()
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        st.write("Analyzing image...")
        result = analyze_image(image_data)

        if result:
            st.write("Captions derived from the image:")
            captions = display_analysis_results(result)
            for i, caption in enumerate(captions, start=1):
                st.write(f"{i}. {caption}")

            if result.caption:
                creative_title = f"An Imaginative Tale Based on: '{result.caption.text}'"
                st.subheader(creative_title)
                story = generate_story(result.caption.text)
                st.write(story)

if __name__ == "__main__":
    main()