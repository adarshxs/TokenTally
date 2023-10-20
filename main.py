import streamlit as st
import json
import pandas as pd

# load the base models and configurations from the JSON file
@st.cache_data 
def load_base_models():
    with open("base_models_with_gpus.json", "r") as f:
        return json.load(f)

# load the GPU data from the cloud-gpus.csv
@st.cache_data 
def load_gpu_providers():
    return pd.read_csv('cloud-gpus.csv')


def main():
    st.title("Token Tally: LLM Cost Estimator")
    st.subheader("Estimate Your LLM's Token Toll Across Various Platforms and Configurations")
    
    # base model and configurations data
    base_models = load_base_models()["models"]
    gpu_providers_df = load_gpu_providers()

    model_names = [model["name"] for model in base_models]
    selected_model_name = st.selectbox("Step 1: Select the Base Model", model_names)

    selected_model = next(model for model in base_models if model["name"] == selected_model_name)
    selected_params = st.selectbox("Step 2: Select the Number of Parameters", selected_model["params"])

    config_names = [config["name"] for config in selected_model["configurations"]]
    selected_config_name = st.selectbox("Step 3: Select the Configuration", config_names)

    selected_config = next(config for config in selected_model["configurations"] if config["name"] == selected_config_name)
    
    # Displaying the minimum GPU memory requirement
    st.write(f"Minimum GPU Memory Requirement: {selected_config['min_gpu_memory']}")
    
    cloud_providers = gpu_providers_df["Cloud"].unique()
    selected_provider = st.selectbox("Step 4: Select the Cloud Provider", cloud_providers)

    suitable_gpu_types = gpu_providers_df[gpu_providers_df["Cloud"] == selected_provider]["GPU Type"]
    selected_gpu_type = st.selectbox("Step 5: Select the GPU Type", suitable_gpu_types)

    selected_gpu_details = gpu_providers_df[(gpu_providers_df["Cloud"] == selected_provider) & (gpu_providers_df["GPU Type"] == selected_gpu_type)]
    
    # Displaying the selected GPU details
    st.write("Selected GPU Details:")
    st.write(selected_gpu_details)
    
    # Creating a DataFrame for the selected configuration and displaying it
    selected_config_df = pd.DataFrame([selected_config], columns=["name", "cpu", "ram", "gpu_ram", "min_gpu_memory"])
    st.write("Selected Configuration Details:")
    st.write(selected_config_df)

    st.write("Final Total Cost of Ownership: (To be calculated)")

# Uncomment the following line to run the application
if __name__ == "__main__":
    main()