import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title("Rasmlarni klassifikatsiya qilish")

# Rasmni yuklab oluvchi fayl
file = st.file_uploader("Rasm yuklang", type=['svg', 'jpg', 'jpeg'])

# Agar fayl yuklab olingan bo'lsa
if file is not None:
    img = PILImage.create(file)
    st.image(file)
    # Modelni yuklash
    model = load_learner('model.pkl')

    # Rasmini klassifikatsiya qilish
    prediction = model.predict(img)
    
    # Natijani ko'rsatish
    st.write("Natija:", prediction[0])
