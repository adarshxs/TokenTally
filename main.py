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

    with st.sidebar:
        st.image("cutie.png", use_column_width=True)
        st.title("About Token Tally")
        st.info("Select your desired base model, parameters, and configuration to get an estimate of the required GPU memory and model size. Do contribute: https://github.com/adarshxs/TokenTally")
        st.warning("Notice: The logic for the final cost/token is yet to be implemented!")

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
        st.markdown(f"""
            <div class="card">
                <strong>Model Size: {model_size} GB</strong>
            </div>
            """, unsafe_allow_html=True)
    with col2:
        st.subheader("Minimum GPU Memory")
        st.markdown(f"""
            <div class="card">
                <strong>Min GPU Memory Required: {model_size * 1.2:.2f} GB</strong>
            </div>
            """, unsafe_allow_html=True)
        st.latex(r'''
                 \text{GPU Requirement} = 1.2 \times \text{Model Size}
        ''')

        
    cloud_providers = gpu_providers_df["Cloud"].unique()
    selected_provider = st.selectbox("Step 4: Select the Cloud Provider", cloud_providers)

    suitable_gpu_types = gpu_providers_df[gpu_providers_df["Cloud"] == selected_provider]["GPU Type"]
    selected_gpu_type = st.selectbox("Step 5: Select the GPU Type", suitable_gpu_types)

    selected_gpu_details = gpu_providers_df[(gpu_providers_df["Cloud"] == selected_provider) & (gpu_providers_df["GPU Type"] == selected_gpu_type)]

    # GPU details in a table format
    st.subheader("Available GPU Variants")
    selected_gpu_details = selected_gpu_details.reset_index(drop=True)
    selected_gpu_details.index = selected_gpu_details.index + 1
    st.table(selected_gpu_details)

    st.subheader("Final Total Cost of Ownership")
    MO = st.slider("Enter Maxed Out percentage", min_value=1, max_value=100, value=50)
    VMc = st.number_input("Refer to the above table($ On-Demand) and Enter Instance Cost Per Hour (in USD):", min_value=0.0, value=0.0, step=0.01)
    # calculate TS_max
    TS_max = 1 # to be implemented!!!
    TS = TS_max*(MO/100)
    # Compute Cost
    CT = VMc / (TS*3600) 
    st.latex(r'''TS = \frac{TS_{max} \times MO}{100}''')
    st.latex(r'''CT = \frac{VM_c}{TS}''')
    
    st.markdown("""
    <style>
    .card {
        background-color: #f0f2f6; 
        border-radius: 5px;
        padding: 20px 30px;
        margin: 25px 0px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card">
        <strong>Estimated Cost per Token (CT): To be Implemented</strong>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
