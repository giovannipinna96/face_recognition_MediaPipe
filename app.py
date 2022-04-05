import streamlit as st

st.title('Face Mash using MediaPipe')
with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
