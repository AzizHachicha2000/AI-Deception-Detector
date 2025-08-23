#app.py
import streamlit as st
import joblib
from preprocessor import preprocess_text
import matplotlib.pyplot as plt


model = joblib.load('Sherlock_Holmes.pkl')


st.title("🔍 Sherlock - Détection de Tromperie")
text = st.text_area("Entrez le texte à analyser:")

if st.button("Analyser"):
    
    clean_text = preprocess_text(text)
    
    
    proba = model.predict_proba([clean_text])[0][1]
    pred = "Véridique ✅" if proba > 0.5 else "Trompeur ❌"
    
    
    st.metric("Probabilité de véracité", f"{proba:.2%}", pred)
    
    
    fig, ax = plt.subplots()
    ax.bar(['Trompeur', 'Véridique'], model.predict_proba([clean_text])[0])
    ax.set_ylim(0,1)
    st.pyplot(fig)