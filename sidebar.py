import streamlit as st

def sidebar():
    with st.sidebar:
        st.image("cutie.png", use_column_width=True)
        st.title("About Token Tally")
        st.info("Select your desired base model, parameters, and configuration to get an estimate of the required GPU memory and model size. Do contribute: https://github.com/adarshxs/TokenTally")
        # Add a Products section with radio buttons to select a page
        product_options = ["Overview", "LLM Cost Tool", "Transformer Memory Tool", "LLM Model Recommendation"]
        selected_product = st.radio("Products", product_options)
        
        st.warning("Notice: The logic for the final cost/token is yet to be implemented!")
        
        return selected_product
