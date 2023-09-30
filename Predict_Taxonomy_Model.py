import streamlit as st
import joblib
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import requests


specialcharcters=["0","1","2","3","4","5","6","7","8","9","s","S","$","@","•","º","°","%","Σ","Ω","μ","α","×","π","&","½","■","—","–","™","#","’","®","!","^","?","*","\"","€","±","~","“","”","µ","‘","ü","‑","⁰","","²","³",",","−","…","","（","）","≥"," ","≤","″","¬"," ","：","℃","Ć","―","Â","＠","］","［","˚","¼","Δ","©","□","�","・","é","ø","･","〞","～","„","、","‐","－","φ","£","ｍｍ","●","ô","【","】","※","à","·","§","å","ä","«","√","，","△","‟","´","λ","….","\"","","/","\\",":","s","-","(",")"]
def claening_desc(desc):
    desc= "".join([char for char in desc if char not in punctuation])
    desc= "".join([char for char in desc if char not in specialcharcters]).lower().replace("  "," ").strip()
    return " ".join([word for word in desc.split(" ") if len(word)>1]).strip()


load_model=joblib.load("svm_Model.pkl")
tf_idf_vectorizer=TfidfVectorizer()

with open('Taxonomies_Labels_dict.txt',encoding='utf-8',mode='r',newline="") as file:
    Taxonomies_Labels_dict=ast.literal_eval(file.read())
    
with open('saved_features_names_all.txt',mode='r',newline='',encoding='utf-8') as fileread:
    load_feature_names=ast.literal_eval(fileread.readlines()[0])
    
df_original_fetures=pd.DataFrame(np.zeros((1,32559),float),columns=load_feature_names)

st.set_page_config(layout="wide")

input_desc=st.text_input(label="Enter Description to Predict Taxonomy",placeholder='Marzouk',value='Marzouk')

cleaned_desc=claening_desc(input_desc)

if cleaned_desc:
    tf_input_desc=tf_idf_vectorizer.fit_transform([cleaned_desc])
    
    df_input_desc=pd.DataFrame(tf_input_desc.toarray(),columns=tf_idf_vectorizer.get_feature_names_out())
    
    found_feature_names=[found_feature for found_feature in df_input_desc.columns if found_feature in df_original_fetures.columns]
    df_original_fetures[found_feature_names]=df_input_desc[found_feature_names]
    fiinal_array=df_original_fetures.to_numpy()
    if st.button("Click to predict taxonomy"):
        if found_feature_names !=[]:
            predicted_taxonomy=load_model.predict(fiinal_array)
            predicted_accuracy= f"{(load_model.predict_proba(fiinal_array).max() * 100):.2f}"
            Taxonomy_Name=Taxonomies_Labels_dict[predicted_taxonomy[0]]
            print(Taxonomy_Name,predicted_accuracy,sep='\t')
            st.text(f"predicted Taxonomy is: {Taxonomy_Name} \nAccuracy is: {predicted_accuracy} %")
            #accuracy=load_model.predict_proba(df_final).max()
            #print(f"predicted Taxonomy is: {predicted_taxonomy[0]} \nAccuracy is: {round(accuracy*100,2)} %")
        else:
            st.text("Descriptions you entered is under training or is not related to any of silicon taxonomies, please enter another descriptions")





# extract pdf parts and title and predict taxonomy

input_datasheet=st.text_input(label="Enter PDF to Get all data you need, we support till now only supplier (Texas Instruments)",placeholder='Marzouk',value='Marzouk')

# functions to return table in the page that contain 'keyword'
def extract_table_from_page(pdf, keyword):
      
    if "http" in pdf:
        response = requests.get(pdf)
        # Check if the request was successful
        if response.status_code == 200:
            # Open the PDF document
            pdf_document = fitz.open("pdf", stream=response.content)
        else:
            print("Wrong PDF Link")    
        
    else:
        pdf_document=fitz.open(pdf)

    pg_numbers=[]    
    for page_number, page in enumerate(pdf_document):
        page_text = page.get_text()
        if keyword in page_text:
            pg_numbers.append(page_number)
    all_df_tables=pd.DataFrame()
    for  pg_no in pg_numbers:
        #print(pg_no)       
        page=pdf_document.load_page(pg_no)
        all_tables=page.find_tables()
        #print(all_tables)
        if all_tables.tables != []:
            df_table=all_tables[0].to_pandas()
            all_df_tables=pd.concat([all_df_tables,df_table],axis=0,ignore_index=True)
    return all_df_tables

# function to return title of pdf
def get_title(pdf):
    
    if "http" in pdf:
        response = requests.get(pdf)
        # Check if the request was successful
        if response.status_code == 200:
            # Open the PDF document
            pdf_document = fitz.open("pdf", stream=response.content)
        else:
            print("Wrong PDF Link")    
        
    else:
        pdf_document=fitz.open(pdf)
    
    meta_data=pdf_document.metadata
    title=meta_data.get('title')
    subject=meta_data.get('subject')
    return title.replace('\t','')

try:

    if st.button("Click to get all data"):
        df_table=extract_table_from_page(input_datasheet,'Addendum-Page')
        orderable_parts=pd.DataFrame(df_table.iloc[:,0])
        title = get_title(input_datasheet)
        title = title.split('datasheet')[0].strip()

        orderable_parts['Descriptions']= title

        print(orderable_parts)


        input_desc=title

        cleaned_desc=claening_desc(input_desc)

        if cleaned_desc:
            tf_input_desc=tf_idf_vectorizer.fit_transform([cleaned_desc])
            
            df_input_desc=pd.DataFrame(tf_input_desc.toarray(),columns=tf_idf_vectorizer.get_feature_names_out())
            
            found_feature_names=[found_feature for found_feature in df_input_desc.columns if found_feature in df_original_fetures.columns]
            df_original_fetures[found_feature_names]=df_input_desc[found_feature_names]
            fiinal_array=df_original_fetures.to_numpy()
            if found_feature_names !=[]:
                predicted_taxonomy=load_model.predict(fiinal_array)
                predicted_accuracy= f"{(load_model.predict_proba(fiinal_array).max() * 100):.2f}"
                Taxonomy_Name=Taxonomies_Labels_dict[predicted_taxonomy[0]]
                print(Taxonomy_Name,predicted_accuracy,sep='\t')
                #st.text(f"predicted Taxonomy is: {Taxonomy_Name} \nAccuracy is: {predicted_accuracy} %")
                #accuracy=load_model.predict_proba(df_final).max()
                #print(f"predicted Taxonomy is: {predicted_taxonomy[0]} \nAccuracy is: {round(accuracy*100,2)} %")
            else:
                st.text("Descriptions you entered is under training or is not related to any of silicon taxonomies, please enter another descriptions")

        orderable_parts['Predicted_Taxonomy']=Taxonomy_Name
        orderable_parts['Accuracy']=predicted_accuracy
        
        orderable_parts.dropna(inplace=True)
        
        st.dataframe(orderable_parts,use_container_width=True)
except Exception as e:
    st.text("Be Noticed that we support till now supplier (Texas Instruments) with new PDFs only.\nUnexpected Error as (pdf link or path not right ), or can't understand document.\nPlease check your PDF Link again.\nAny way We will retrain module to get your expectation.")
