import pandas as pd
import streamlit as st


@st.cache_data
def load_data():
    """
    Load and cache the dataset from file
    """
    df = pd.read_csv(r'C:\Modul2_220711682\streamlit_modul2\web\multilabel\data\train_preprocess.csv')
    return df
