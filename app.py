import streamlit as st
from PIL import Image
from model import predict
from translator import predict_trans
import json
import pandas as pd
st.set_page_config(page_title="EfficientNet Image Classifier")
st.markdown("<h1 style='text-align: center; font-family: sans-serif'; color: #191926>EfficientNetB0 — Класифікація зображень</h1>", unsafe_allow_html=True)
st.markdown("<p style ='text-align: center; font-family: sans-serif; color: #18182C; font-size: 1em; font-weight: 1000'>Завантажте картинку для класифікації за допомогою EfficientNetB0</p>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Завантажте JPG/PNG", type=["jpg", "jpeg", "png"])
if uploaded_file:
  if st.button("Перевірити"):
    image = Image.open(uploaded_file)
    st.image(image, caption="Завантажене зображення", use_container_width=True)
    st.write("Виконується аналіз...")
    preds = predict(image)
    st.write(preds)
    data = []
    new_text = preds['class']
    text_ = str(new_text[3:])
    text_ = predict_trans(f"This is {text_} car.")     
    data.append({"Об'єкт": f"{text_[0]['translation_text']}", "Ймовірність": f"{preds['confidence']:.2f}"})
    table = pd.DataFrame(data, columns = ["Об'єкт", 'Ймовірність'])
    styled_df = table.style.set_table_styles([
          {'selector': '', 
     'props': [
         ('margin-left', 'auto'),
         ('margin-right', 'auto'),
         ('width', '100%')
     ]},
        {'selector': 'thead tr', 
        'props': [
            ('border', '2px solid black'), ('text-align', 'center')
        ]},
        
        {'selector': 'tbody tr:nth-child(even)', 
        'props': [('background-color', "#0d9e12"), ('color', '#ffffff'), 
                  ('text-align', 'center'), 
                  ('border', '2px solid black')]},

        {'selector': 'tbody tr:nth-child(odd)', 
        'props': [('background-color', '#000000'),
                  ('color', 'white'),('text-align', 'center'),
                  ('border', '2px solid black')]},
        
        {'selector': 'tbody tr:hover', 
        'props': [('background-color', "#3d4256"),('text-align', 'center'), 
                  ('border', '2px solid black')]},
    ])

    st.write(styled_df.to_html(), unsafe_allow_html=True)