import streamlit as st
import os
import pandas as pd
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from src.config import DPI, JPEGOPT, TESSDATA_DIR_CONFIG, OUTPUT_DIR
import tempfile
import markdown
from pathlib import Path
import base64
from tqdm import tqdm

from src.extract_utils import *

# Import all the extract_info_mode* functions here
# from your original code

def process_image(image, mode, language_model):
    result = []
    gray_image = image.convert('L')
    x, y = gray_image.size
    
    try:
        if mode in ['1', '2']:
            txt = pytesseract.image_to_string(gray_image, lang=language_model, config=TESSDATA_DIR_CONFIG)
            result = extract_info_mode1(txt)
        elif mode == '3':
            txt = pytesseract.image_to_string(gray_image, lang=language_model, config=TESSDATA_DIR_CONFIG)
            result = extract_info_mode3(txt)
        elif mode == '4':
            txt = pytesseract.image_to_data(gray_image, lang=language_model, config=TESSDATA_DIR_CONFIG)
            result = extract_info_mode4(txt)
        elif mode == '5':
            txt = pytesseract.image_to_string(gray_image, lang=language_model, config=TESSDATA_DIR_CONFIG)
            result = extract_info_mode5(txt)
        elif mode == '6':
            result = extract_info_mode6(image, language_model, TESSDATA_DIR_CONFIG)
        elif mode == '7':
            txt = pytesseract.image_to_data(gray_image, lang=language_model, config=TESSDATA_DIR_CONFIG)
            result = extract_info_mode7(txt, x)
        elif mode == '8':
            txt = pytesseract.image_to_string(gray_image, lang=language_model, config=TESSDATA_DIR_CONFIG)
            result = extract_info_mode8(txt)
        elif mode == '9':
            result = extract_info_mode9(image, language_model, TESSDATA_DIR_CONFIG)
        elif mode == '10':
            result = extract_info_mode10(image, language_model, TESSDATA_DIR_CONFIG)
        elif mode == '11':
            txt = pytesseract.image_to_string(gray_image, lang=language_model, config=TESSDATA_DIR_CONFIG)
            result = extract_info_mode11(txt)
        elif mode == '12':
            result = extract_info_mode12(image, language_model, TESSDATA_DIR_CONFIG)
        elif mode == '13':
            result = extract_info_mode13(image, language_model, TESSDATA_DIR_CONFIG)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
    
    return result

def display_readme():
    # Read the README.md file
    readme_path = Path("README.md")
    with readme_path.open("r") as f:
        readme_content = f.read()
    
    # Function to encode image to base64
    def img_to_base64(img_path):
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded_string}"
    
    # Replace image path with base64 encoded image
    img_path = "modes.png"
    if os.path.exists(img_path):
        base64_image = img_to_base64(img_path)
        readme_content = readme_content.replace(f'src="{img_path}"', f'src="{base64_image}"')
    
    # Convert markdown to HTML
    html_content = markdown.markdown(readme_content)
    
    # Display the HTML content
    st.markdown(html_content, unsafe_allow_html=True)

def main():
    st.title("IITB-CSTT: Glossary OCR Application")
    
    display_readme()

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Options
        language_model = st.selectbox("Select language model", ["eng+hin", "eng+san", "eng+ori","eng+tam", "eng+tel", "eng+hin+guj", "eng+hin+bod"])
        mode = st.selectbox("Select mode", ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'])
        start_page = st.number_input("Start page", min_value=1, value=1)
        end_page = st.number_input("End page", min_value=1, value=1)

        if st.button("Process PDF"):
            # Convert PDF to images
            images = convert_from_path(tmp_file_path, dpi=DPI, fmt='jpeg', jpegopt=JPEGOPT)

            result = []
            progress_bar = st.progress(0)
            for i, image in enumerate(images[start_page-1:end_page], start=start_page):
                result.extend(process_image(image, mode, language_model))
                progress_bar.progress((i - start_page + 1) / (end_page - start_page + 1))

            # Create DataFrame based on mode
            columns = {
                '1': ['English Word', 'Indic Meaning', 'Indic Description'],
                '2': ['English Word', 'Indic Meaning', 'Indic Description'],
                '3': ['Sanskrit Word', 'English Vocab', 'Description'],
                '4': ['English Word', 'Indic Meaning'],
                '5': ['English Word', 'Word 1', 'Word 2', 'Word 3', 'Word 4'],
                '6': ['English Word', 'Oriya Word'],
                '7': ['Col1', 'Col2', 'Col3'],
                '8': ['Col1', 'Col2', 'Col3'],
                '9': ['col1', 'col2'],
                '10': ['col1', 'col2'],
                '11': ['col1', 'col2', 'col3'],
                '12': ['col1', 'col2', 'col3'],
                '13': ['col1', 'col2']
            }

            df = pd.DataFrame(result, columns=columns.get(mode, ['Column 1', 'Column 2', 'Column 3']))

            # Display results
            st.write(df)

            # Option to download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="ocr_results.csv",
                mime="text/csv",
            )

        # Clean up temporary file
        os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()