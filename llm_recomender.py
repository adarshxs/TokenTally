import streamlit as st
import pandas as pd
from utilities import load_base_models, load_quantization

# Ensure session state initialization
if 'value' not in st.session_state:
    st.session_state.value = "8"

def recommend_model(gpu_memory_available):
    """
    Recommend suitable Large Language Models (LLM) based on available GPU memory.

    Parameters:
        gpu_memory_available (float): Available GPU memory in GB.

    Returns:
        list: A list of dictionaries containing suitable models sorted by score.
    """
    base_models = load_base_models()
    quantization_data = load_quantization()

    # Ordered by performance
    performance_order = ["LLAMA 2", "FALCON", "Mistral", "Vicuna"]
    suitable_models = []

    for model in base_models:
        for param, size in model["params"].items():
            for quant_type, quant_value in quantization_data.items():
                actual_model_size = size * quant_value
                min_gpu_req = actual_model_size * 1.2
                if min_gpu_req <= gpu_memory_available:
                    score = (1/min_gpu_req) + float(param.rstrip('B')) + (1/(1+performance_order.index(model["name"])))
                    suitable_models.append({
                        "Model Name": model["name"],
                        "Parameters": param,
                        "Quantization Type": quant_type,
                        "Actual Model Size (GB)": actual_model_size,
                        "GPU Requirement (GB)": min_gpu_req,
                        "Score": score
                    })

    # Sorting models based on score
    suitable_models.sort(key=lambda x: x["Score"], reverse=True)

    return suitable_models


def display_llm_recomender_tool():
    """
    Display the Streamlit interface for the LLM Model Recommendation tool.
    """

    # Preamble description
    st.markdown("""
    ## **Welcome to the LLM Recommendation Tool!**

    Choosing the right **Large Language Model (LLM)** can be challenging given the variety of models available 
    and the technical constraints of different systems.

    This tool is designed to assist users in selecting the most suitable LLM for their computational infrastructure, 
    specifically considering the GPU memory constraints.

    Instructions:
    - **Specify your GPU Memory**: Adjust the available GPU memory using the input field.
    - **Get Recommendations**: The tool will recommend suitable LLMs based on your input.
    - **Understand the Recommendations**: A comprehensive breakdown explains the ranking and recommendation logic.

    Enter your GPU memory below!
    """)

    # Initialize GPU Memory input field
    gpu_memory_field = st.empty()
    st.session_state.value = gpu_memory_field.text_input("Available GPU Memory (GB)", value=st.session_state.value)

    # Get recommendations and display them in a table
    results = recommend_model(float(st.session_state.value))
    results_df = pd.DataFrame(results).head(15)
    st.table(results_df)

    # Display the methodology and top recommendation
    top_model = results_df.iloc[0]["Model Name"]
    top_quantization = results_df.iloc[0]["Quantization Type"]
    top_gpu_req = results_df.iloc[0]["GPU Requirement (GB)"]
    top_model_size = results_df.iloc[0]["Actual Model Size (GB)"]
    value = st.session_state.value
    st.markdown(f"""
    ## **How it works**

    1. **Available GPU Memory**: Based on your input, the currently available GPU memory on your system is **{value} GB**. The recommendation engine filters out models that surpass this memory footprint.

    2. **Quantization**: 
        - Models undergo *quantization* to shrink their size, albeit at a minor compromise on accuracy. This results in a model that's more resource-efficient.
        - Different quantization levels impact the model's size differently. For instance, adopting a {top_quantization} approach for the {top_model} compresses it to an approximate size of **{top_model_size:.2f} GB**.
        - Depending on the balance between accuracy and resource usage you're aiming for, you can choose between differently quantized models.

    3. **Ranking & Recommendation Score**:
        - The recommendation score is a multifaceted metric, considering:
            * **GPU Memory Requirement**: Models demanding less GPU memory are scored higher. For example, {top_model} requires around **{top_gpu_req:.2f} GB**.
            * **Number of Parameters**: A higher parameter count usually indicates better performance, so these models get a favorable score.
            * **Inherent Performance Order**: Some models inherently outperform others due to architecture or training nuances. This intrinsic order also influences the ranking.
        - The recommendations you see are sorted descendingly based on this composite score.

    The top recommendation currently for your system, given the constraints, is the **{top_model}** model with {top_quantization} quantization.

    Harnessing this tool ensures you get the most bang for your buck — or in this case, the most AI prowess for your GPU memory!
    """)
