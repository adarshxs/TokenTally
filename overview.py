import streamlit as st

def display_overview():
    
    # read readme.md file and display it as markdown in streamlit
    
    with open("README.md", "r", encoding="utf-8") as f:
        readme = f.read()
        # readme.replace("![cutie](https://github.com/adarshxs/TokenTally/assets/114558126/0f584e00-5bf8-4763-a885-8ca5a7e87ee9)", "") # not working don't know why
        st.markdown(readme)