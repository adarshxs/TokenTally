import streamlit as st
from utilities import load_base_models, load_quantization, load_gpus, load_gpu_providers, convert_params, compute_bound_tokens_p_sec, memory_bound_tokens_p_sec, cost_per_1k_tokens

def display_llm_cost_tool():
    st.title("Token Tally: LLM Cost Estimator")
    st.subheader("Estimate Your LLM's Token Toll Across Various Platforms and Configurations")

    # Base model and configurations data
    base_models = load_base_models()
    quantization_data = load_quantization()
    gpu_data = load_gpus()
    gpu_providers_df = load_gpu_providers()

    model_names = [model["name"] for model in base_models]
    selected_model_name = st.selectbox("Step 1: Select the Base Model", model_names, key='base_model')
    selected_model = next(model for model in base_models if model["name"] == selected_model_name)

    param_options = list(selected_model["params"].keys())
    selected_params = st.selectbox("Step 2: Select the Number of Parameters", param_options, key='params')
    
    config_names = list(quantization_data.keys())
    selected_config_name = st.selectbox("Step 3: Select the Configuration", config_names, key='config')

    # calculate model size based on selected configuration
    model_size = selected_model["params"][selected_params] * quantization_data[selected_config_name]

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
    selected_provider = st.selectbox("Step 4: Select the Cloud Provider", cloud_providers, key='provider')

    suitable_gpu_types = gpu_providers_df[gpu_providers_df["Cloud"] == selected_provider]["GPU Type"].unique()
    selected_gpu_type = st.selectbox("Step 5: Select the GPU Type", suitable_gpu_types, key='gpu_type')
    selected_gpu_details = gpu_providers_df[(gpu_providers_df["Cloud"] == selected_provider) & (gpu_providers_df["GPU Type"] == selected_gpu_type)]

    st.subheader("Available GPU Variants")
    selected_gpu_details = selected_gpu_details.reset_index(drop=True)
    selected_gpu_details.index = selected_gpu_details.index + 1
    st.table(selected_gpu_details)
    st.subheader("Cost per 1,000 tokens - For Selected Model")
    # calculate TS_max
    TS_max = 1 # to be implemented!!!
    #TS = TS_max*(MO/100)
    # Compute Cost
    #CT = VMc / (TS*3600) 

    st.markdown("""
    <style>
    .card {
        background-color: #2f2f2f; 
        border-radius: 5px;
        padding: 20px 30px;
        margin: 25px 0px;
        text-align: center;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)
    cost_p_1k_tokens_compute, cost_p_1k_tokens_memory = 0,0

    calculated_flops = selected_model["params"][selected_params]
    # Populate the placeholder with the number_input, setting the default value to calculated_flops
    flops_per_token = st.number_input("FLOPs per Token = Model parameters in Billion * 2 (Considering batch size=1 and ignoring KV cache)", min_value=1.0, value=float(calculated_flops))
    flops_per_gpu = st.number_input("FLOPs per GPU (TFLOPs) - Only available for A100 80GB considering 70% MFU", min_value=1, value=200)
    num_gpus = st.number_input("Number of GPUs", min_value=1, value=8)
    cost_per_hour = st.number_input("Cost per Hour (USD) - Refer ($)On-Demand in the above table only for A100 80GB", min_value=0.01, value=40.97)
    memory_bandwidth_per_gpu = st.number_input("Memory Bandwidth per GPU (TB/s) - 2Tb/s for A100 80Gb and considering 60-70 % inference workloads", min_value=0.1, value=1.3)

    if st.button("Calculate"):
        cost_p_1k_tokens_compute, cost_p_1k_tokens_memory = cost_per_1k_tokens(
            flops_per_token, 
            flops_per_gpu, 
            num_gpus, 
            cost_per_hour, 
            memory_bandwidth_per_gpu
        )
        
    st.markdown(f"""
    <div class="card">
        <strong>Estimated Cost per 1,000 Input tokens: ${cost_p_1k_tokens_compute:.6f}</strong><br>
        <strong>Estimated Cost per 1,000 Output tokens: ${cost_p_1k_tokens_memory:.6f}</strong>
    </div>
    """, unsafe_allow_html=True)