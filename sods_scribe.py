import streamlit as st
import joblib
import re
from pdfminer.high_level import extract_text
import pytesseract
from pdf2image import convert_from_path
from fpdf import FPDF

# Set up the page configuration
st.set_page_config(page_title="SODScribe", page_icon="📖", layout="wide")

with open("styles.css") as css_file:
    st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    tfidf_vectorizer = joblib.load(r"C:\Users\medha\Downloads\tfidf_vectorizer.joblib")
    model = joblib.load(r"C:\Users\medha\Downloads\best_model.joblib")
    return tfidf_vectorizer, model

# Load the models at the start
tfidf_vectorizer, model = load_models()

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    extracted_text = {}
    text = extract_text(pdf_file)
    articles = re.split(r'\n\n+', text)
    for idx, article_text in enumerate(articles, start=1):
        formatted_text = " ".join(article_text.split())  # Clean and format the text
        extracted_text[idx] = formatted_text
    return extracted_text

# OCR extraction from scanned PDFs
def extract_pdf_text_with_ocr(pdf_file):
    try:
        with open("temp_pdf.pdf", "wb") as f:
            f.write(pdf_file.getbuffer())  # Save the PDF temporarily

        pages = convert_from_path("temp_pdf.pdf", 300)
        extracted_text = ""

        for page_num, page in enumerate(pages):
            text = pytesseract.image_to_string(page)
            cleaned_text = clean_pdf_text(text)

            paragraphs = cleaned_text.split('\n\n')
            extracted_text += f"Page {page_num + 1}:\n"
            for i, paragraph in enumerate(paragraphs):
                extracted_text += f"Paragraph {i + 1}: {paragraph.strip()}\n"
            extracted_text += "\n"

        return extracted_text
    except Exception as e:
        print(f"Error occurred: {e}")
        return None

# Clean up text extracted from OCR or PDF
def clean_pdf_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()
    text = re.sub(r'\s([?.!,;])', r'\1', text)  # Remove spaces before punctuation
    return text

# Classify text based on the model
def classify_text(text_by_articles, vectorizer, model):
    sods_articles = {}
    for article_num, text in text_by_articles.items():
        transformed_text = vectorizer.transform([text])
        prediction = model.predict(transformed_text)
        if prediction[0] == 1:  # SODS-related content is marked as 1
            sods_articles[article_num] = text
    return sods_articles

# Save classified SODS-related articles to a file
def save_sods_to_file(sods_articles):
    with open("sods_articles.txt", "w", encoding='utf-8') as f:
        for article_num, text in sods_articles.items():
            f.write(f"Article {article_num} (SODS):\n")
            f.write(f"{text}\n\n")
    return "sods_articles.txt"

# Load the external CSS file
st.markdown('<link href="assets/style.css" rel="stylesheet" type="text/css">', unsafe_allow_html=True)

# Page Title
st.title("📖 Welcome to **SODScribe**")

# Sidebar for navigation
st.sidebar.header("📂 Navigation")
page = st.sidebar.radio("Select a Page", ["Home", "About the App", "About Us", "Feedback"])

# About the App Section
if page == "About the App":
    st.sidebar.subheader("🌟 About the App")
    st.sidebar.write("""
    **SODScribe** is an intelligent tool designed to extract and classify SODS-related content from university newsletters using machine learning techniques.

    **Key Features:**
    - Text extraction from both regular and OCR-based PDF files.
    - Automated classification of articles related to the SODS department.
    - Download results as a text file for further analysis or sharing.
    """)

# About Us Section
elif page == "About Us":
    st.sidebar.subheader("🌟 About Us")
    st.sidebar.write("""
    Hi! We are a team of three - Medha, Anju, and Anand. We developed **SODScribe** as part of our third-semester project, aiming to help our SODS department efficiently identify relevant content in university newsletters.

    **Meet the Team:**

    - **Medha**:
        - [GitHub](https://github.com/cherimedz) 
        - [LinkedIn](https://linkedin.com/in/medha-reju-pillai-42551b277)
        - [Kaggle](https://www.kaggle.com/cherimedz)
    
    - **Anju**:
        - [GitHub](https://github.com/Anju-B-J) 
        - [LinkedIn](https://www.linkedin.com/in/anjubj/)
    
    - **Anand**:
        - [GitHub](https://github.com/anandsj7) 
        - [LinkedIn](https://www.linkedin.com/in/anand-sankar-j-938a14326/)
        - [Kaggle](https://www.kaggle.com/anandsj7)
    """)

# Home Page - File Upload Section
elif page == "Home":
    st.markdown("""**SODScribe** is an intelligent tool designed to extract and classify SODS-related content from university newsletters. By leveraging machine learning techniques, **SODScribe** can extract and classify articles based on their relevance to the SODS department.""")

    st.markdown("<h2 style='font-size: 24px; font-weight: bold;'>Upload Your PDF Documents Below</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        
        with st.spinner("Extracting text..."):
            extracted_text = ""
            if uploaded_file.name.endswith('.pdf'):
                # Check if it's a scanned PDF for OCR
                extracted_text = extract_pdf_text_with_ocr(uploaded_file)
                if not extracted_text:
                    st.warning("OCR extraction failed, trying regular text extraction.")
                    extracted_text = extract_text_from_pdf(uploaded_file)
        
            st.success("Text extraction completed!")

        with st.spinner("Classifying articles..."):
            sods_articles = classify_text(extracted_text, tfidf_vectorizer, model)
        
        st.success("Classification completed!")

        # Provide download link for the .txt file
        file_name = save_sods_to_file(sods_articles)
        if file_name:
            st.success(f"SODS-related content saved to {file_name}. You can download it below.")
            with open(file_name, "rb") as f:
                st.download_button("Download SODS Articles", f, file_name)

# Feedback Section
elif page == "Feedback":
    st.sidebar.subheader("💬 Feedback")
    feedback = st.sidebar.text_area("We value your feedback! Share your thoughts here:")
    if st.sidebar.button("Submit Feedback"):
        if feedback:
            st.sidebar.write("Thank you for your feedback! 😊 Your input helps us improve.")
            with open("feedback.txt", "a") as f:
                f.write(feedback + "\n")
        else:
            st.sidebar.write("Please enter your feedback before submitting.")

# End of the App Section (footer-like section)
st.markdown("---")
st.markdown("""
    **Thank you for using SODScribe!**  
    We hope this tool helps you identify relevant content from university newsletters. Feel free to reach out to us through the **About Us** section if you have any questions or feedback.

    **Happy exploring!** 😊
""")
