import streamlit as st

#--- Replicate API Settings & llama2
#test_secret = st.secrets['test']
st.write("Hello World")
test_secret = st.secrets["REPLICATE_API_TOKEN"]
st.write(test_secret)