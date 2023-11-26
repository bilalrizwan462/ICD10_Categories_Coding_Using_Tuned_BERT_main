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

    html_temp = """
    <div style="background-color:green;padding:10px">
    <h2 style="color:white;text-align:center;">ICD10 Categories Coding App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    
    # Example texts
    example_text = """ \nName:  ___                   Unit No:   ___\n \nAdmission Date:  ___              Discharge Date:   ___\n \nDate of Birth:  ___             Sex:   F\n \nService: MEDICINE\n \nAllergies: \nNo Known Allergies / Adverse Drug Reactions\n \nAttending: ___\n \nChief Complaint:\nInpatient Hospice\n \nMajor Surgical or Invasive Procedure:\nnone\n\n \nHistory of Present Illness:\nPatient was admitted to inpatient hospice at ___. For full H&P \nsee prior discharge summary or admission note.\n \nPast Medical History:\nEtOH cirrhosis  \nCalculus of gallbladder without cholecystitis  \nCOPD  \nFatty liver  \nIron deficiency anemia  \nh/o aspiration pneumonitis  \n\nPast Surgical History:\ns/p partial gastrectomy ___ for ulcers\n"revision" of gastric ulcer surgery in ___ \n\n \nSocial History:\n___\nFamily History:\nMother w/ DM and ESRD on HD  \nNo family members with liver disease. \n\n \nPhysical Exam:\nAdmission to inpatient hospice exam:\nPatient lying comfortably in bed, sleeping. Breathing unlabored.\n\nDeath exam:\nPatient not breathing. No breath or heart sounds. No palpable\npulse. No pupillary reflexes. No withdrawal to firm nailbed\npressure.\nTime of death: ___\n \nPertinent Results:\nSee prior discharge summary for full lab results and other \nstudies. No labs or studies performed while patient on inpatient \nhospice.\n \nBrief Hospital Course:\nPatient admitted to inpatient hospice. Pain and agitation \nmedications were titrated. Medications to manage secretions were \nadded. Patient expired on ___ at ___.\n \nMedications on Admission:\nThe Preadmission Medication list is accurate and complete.\n1. Acetaminophen 500 mg PO Q6H:PRN Pain - Mild/Fever \n2. Albuterol 0.083% Neb Soln 1 NEB IH Q6H:PRN Wheezing, SOB \n3. Cepacol (Sore Throat Lozenge) 1 LOZ PO Q2H:PRN Throat \nirritation  \n4. GuaiFENesin ___ mL PO Q6H:PRN Cough  \n5. Haloperidol 0.5-2 mg IV Q4H:PRN delirium \n6. Heparin Flush (10 units/ml) 1 mL IV DAILY and PRN, VIP line \nflush \n7. Heparin Flush (10 units/ml) 2 mL IV DAILY and PRN, line flush \n\n8. HYDROmorphone (Dilaudid) 0.25-0.5 mg IV Q15MIN:PRN \nmoderate-severe pain or respiratory distress \n9. LORazepam 0.5-2 mg IV Q2H:PRN anxiety, nausea (first line) \n10. Midodrine 20 mg PO Q6H \n11. Ondansetron 4 mg IV Q8H:PRN Nausea/Vomiting - Second Line \n12. Polyethylene Glycol 17 g PO DAILY:PRN Constipation - First \nLine \n13. rifAXIMin 550 mg PO BID \n14. Simethicone 40-80 mg PO QID:PRN bloating \n15. Sodium Chloride 0.9%  Flush 10 mL IV DAILY and PRN, line \nflush \n16. Sodium Chloride 0.9%  Flush 10 mL IV DAILY and PRN, line \nflush \n\n \nDischarge Medications:\nnone\n \nDischarge Disposition:\nExpired\n \nDischarge Diagnosis:\n#Renal failure\n#Liver failure\n#VRE bloodstream infection\n#Spontaneous bacterial peritonitis\n#Alcoholic cirrhosis\n\n \nDischarge Condition:\npatient expired\n\n \nDischarge Instructions:\nPatient expired\n \nFollowup Instructions:\n___\n"""

    st.code("Example clinical note:" + example_text)
    text = st.text_area("Please input your clinical text")

    if st.button("Predict"):
        text = text.encode('utf-8').decode('utf-8')
        processed_text = process_clinical_note(text)

        results = predict_codes_using_BERT(processed_text, threshold = 0.35)
        st.write(results)


if __name__=='__main__':
    main()