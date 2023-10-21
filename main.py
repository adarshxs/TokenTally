import streamlit as st
import json
import pandas as pd

@st.cache_resource
def load_base_models():
    with open("models.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_quantization():
    with open("quantization.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_gpus():
    with open("gpus.json", "r") as f:
        return json.load(f)

@st.cache_resource
def load_gpu_providers():
    return pd.read_csv('cloud-gpus.csv')

def main():
    st.title("Token Tally: LLM Cost Estimator")
    st.subheader("Estimate Your LLM's Token Toll Across Various Platforms and Configurations")

    st.sidebar.image("cutie.png", use_column_width=True)
    st.sidebar.title("About Token Tally")
    st.sidebar.write("Select your desired base model, parameters, and configuration to get an estimate of the required GPU memory and model size.")
    
    # Base model and configurations data
    base_models = load_base_models()
    quantization_data = load_quantization()
    gpu_data = load_gpus()
    gpu_providers_df = load_gpu_providers()

    model_names = [model["name"] for model in base_models]
    selected_model_name = st.selectbox("Step 1: Select the Base Model", model_names)
    selected_model = next(model for model in base_models if model["name"] == selected_model_name)

    param_options = list(selected_model["params"].keys())
    selected_params = st.selectbox("Step 2: Select the Number of Parameters", param_options)

    config_names = list(quantization_data.keys())
    selected_config_name = st.selectbox("Step 3: Select the Configuration", config_names)

    # calculate model size based on selected configuration
    model_size = selected_model["params"][selected_params] * quantization_data[selected_config_name]
    
    # make stuff cute
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Size")
        st.write(f"{model_size} GB")
    with col2:
        st.subheader("Minimum GPU Memory Requirement")
        st.write(f"{model_size * 1.2:.2f} GB")
        
    cloud_providers = gpu_providers_df["Cloud"].unique()
    selected_provider = st.selectbox("Step 4: Select the Cloud Provider", cloud_providers)

    suitable_gpu_types = gpu_providers_df[gpu_providers_df["Cloud"] == selected_provider]["GPU Type"]
    selected_gpu_type = st.selectbox("Step 5: Select the GPU Type", suitable_gpu_types)

    selected_gpu_details = gpu_providers_df[(gpu_providers_df["Cloud"] == selected_provider) & (gpu_providers_df["GPU Type"] == selected_gpu_type)]

    # GPU details in a table format
    st.subheader("Available GPU Variants")
    st.table(selected_gpu_details.assign(hack='').set_index('hack'))
    
    st.subheader("Final Total Cost of Ownership")
    st.write("(To be calculated)")

if __name__ == "__main__":
    main()
