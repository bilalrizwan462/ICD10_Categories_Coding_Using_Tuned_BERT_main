import pandas as pd
import re

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import streamlit as st



# print('model loaded')
# medical_stopwords.extend(['speaking', 'none', 'time', 'flush'])

def process_clinical_note(clinical_note):
    medical_stopwords = stopwords.words("english")
    clinical_note = clinical_note.replace('\\n', '\n')
    # Define the sections to remove
    sections_to_remove = [
        "Name:",
        "Unit No:",
        "Admission Date:",
        "Discharge Date:",
        "Date of Birth:",
        "Sex:",
        "Service:",
        "Allergies:",
        "Attending:",
        "Past Medical History:",
        "Social History:",
        "Family History:",
        "Vitals:",
        "Pertinent Results:",
        "Medications on Admission:",
        "Discharge Medications:",
        "Discharge Disposition:",
        "Discharge Condition:",
        "Discharge Instructions:",
        "Followup Instructions:"
    ]

    # Split the clinical note into lines
    lines = clinical_note.split('\n')
    # Initialize the processed note
    processed_note = []

    # Flag to exclude lines within unwanted sections
    exclude_section = False

    # Iterate through the lines and filter unwanted sections
    for line in lines:
        if any(section in line for section in sections_to_remove):
            exclude_section = True
        elif line.strip() == "":
            # Empty lines separate sections, so reset the flag
            exclude_section = False

        if not exclude_section:
            processed_note.append(line)

    # Join the lines to create the final note
    final_note = '\n '.join(processed_note)
    
    sections_to_remove = [
        r'chief complaint',
        r'history of present illness',
        r'Major Surgical or Invasive Procedure',
        r'physical exam',
        r'brief hospital course',
        r'Discharge',
        
        r'completed by',
    ]
    
    for pattern in sections_to_remove:
        final_note = re.sub(pattern, '', final_note, flags=re.IGNORECASE)

    # Define patterns to identify negations
    negation_patterns = [
        r'no\s+\w+',
        r'not\s+\w+',
        r'did\s+not\s+have\s+\w+',
        r'denies\s+\w+',
        r's+\w+clear'
    ]
    
    # Filter out sentences with negations
    sentences = [sentence for sentence in final_note.split('\n') if not any(re.search(pattern, sentence, re.IGNORECASE) for pattern in negation_patterns)]

    # Remove keys and special characters
    cleaned_note = re.sub(r'\w+:', '', '\n'.join(sentences), flags=re.IGNORECASE)  # Remove keys (case-insensitive)
    cleaned_note = re.sub(r'[^a-zA-Z\s]', '', cleaned_note)  # Remove special characters

    # Tokenize the note into sentences based on '\n'
    sentences = [sentence.strip() for sentence in cleaned_note.split('\n') if sentence.strip()]
    # Remove stop words and empty sentences
    sentences = [
        ' '.join(word for word in sentence.split() if word.lower() not in medical_stopwords)
        for sentence in sentences
    ]
    sentences = [item for item in sentences if item != '']

    return sentences


def predict_codes_using_BERT(processed_example, threshold = 0.3):

    Category_model = BertForSequenceClassification.from_pretrained('models/classifier/')
    print("Model loaded")
    Category_tokenizer = BertTokenizer.from_pretrained('models/tokenizer/')
    print("Tokenizer Loaded")
    Category_classifier = pipeline('text-classification', model = Category_model, tokenizer = Category_tokenizer)
    print("Classifier Made!")

    categories_description = pd.read_csv('categories.csv')

    print("predicting codes")
    predictions = Category_classifier(processed_example)
    print("codes predicted")
    for data, sentence in zip(predictions, processed_example):
        # Add a new key-value pair to each dictionary
        data['support_sentence'] = sentence

    predicted_codes = [
    item for item in predictions if item['score'] >= threshold
    ]
    if predicted_codes ==[]:
        return ("No Codes Found")
    else:
        final_results = pd.merge(pd.DataFrame(predicted_codes), categories_description, left_on = 'label', right_on = 'icd_code')
        return final_results


def main():

    st.title("ICD-10 Categories Coder Using BERT")


    text = st.text_area("Please input your clinical text")

    if st.button("Predict"):
        text = text.encode('utf-8').decode('utf-8')
        processed_text = process_clinical_note(text)

        results = predict_codes_using_BERT(processed_text, threshold = 0.35)
        st.write(results)


if __name__=='__main__':
    main()