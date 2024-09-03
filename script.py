import shutil
import os
import PyPDF2
import argparse
import pickle
import os
import joblib
import pandas as pd

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF file. Returns an empty string if an error occurs.
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            extracted_text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                extracted_text += page.extract_text()
            return extracted_text
    except Exception as e:
        print(f"An error occurred while extracting text from {pdf_path}: {e}")
        return ""

def formatting_data(text, model_path):
    """
    Transform the input text using a pre-trained vectorizer.

    Args:
        text (str or list of str): The text data to be transformed.
        model_path (str): The path to the directory containing the vectorizer model file.

    Returns:
        scipy.sparse.csr.csr_matrix: The transformed text data as a sparse matrix.
    """
    try:
        vectorizer = joblib.load(model_path + 'bow_vectorizer.pkl')
        X_test_ifidf = vectorizer.transform(text)
        return X_test_ifidf
    except Exception as e:
        print(f"An error occurred while formatting the data: {e}")
        return None

def read_pdfs_from_directory(directory_path, destination_path, model_path):
    """
    Reads PDF files from a directory, categorizes them using a machine learning model,
    and saves the categorized files to respective folders. The results are also saved in a CSV file.

    Args:
        directory_path (str): The path to the directory containing PDF files.
        destination_path (str): The path to the directory where categorized PDFs will be saved.
        model_path (str): The path to the directory containing the vectorizer and model files.

    Returns:
        None
    """
    file_name = []
    category = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory_path, filename)
            try:
                print(f"Reading {filename}...")
                text = extract_text_from_pdf(pdf_path)
                formatted_data = formatting_data([text], model_path)
                classification_model = pickle.load(open(model_path+"gradient_boosting_model.sav", 'rb'))
                predicted_class = classification_model.predict(formatted_data)
                destination_path_save = os.path.join(destination_path, predicted_class[0])
                os.makedirs(destination_path_save, exist_ok=True)
                shutil.copy2(pdf_path, destination_path_save)
                print('Resume shifted to respective category! ', predicted_class)
                file_name.append(filename)
                category.append(predicted_class[0])
            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")
    try:
        df = pd.DataFrame({
            'filename': file_name,
            'category': category
        })
        csv_file_path = './categorized_resume.csv'
        df.to_csv(csv_file_path, index=False)
        print('CSV Saved Successfully!')
    except Exception as e:
        print(f"An error occurred while saving the CSV file: {e}")

def main():
    """
    Main function to process and categorize PDF resumes based on a pre-trained model.
    """
    parser = argparse.ArgumentParser(description='Process and categorize PDF resumes.')
    parser.add_argument('directory_path', default="./test_resumes/", type=str,
                        help='Path to the directory containing PDF resumes')
    parser.add_argument('--destination_path', default="./test_resumes_categorization/", required=False, type=str,
                        help='Path to the directory where categorized resumes will be saved')
    parser.add_argument('--model_path', default='./model/', required=False, type=str,
                        help='Path to the classification model file')
    args = parser.parse_args()
    directory_path = args.directory_path
    destination_path = args.destination_path
    model_path = args.model_path
    read_pdfs_from_directory(directory_path, destination_path, model_path)


if __name__ == "__main__":
    main()
