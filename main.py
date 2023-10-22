import streamlit as st
import json
import math
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

@st.cache_resource
def convert_params(params):
    if params == 0:
        return "0"
    size_name = ("", "K", "M", "B", "T", "P", "E", "Z", "Y")
    i = int(math.floor(math.log(params, 1000)))
    p = math.pow(1000, i)
    s = round(params / p, 2)
    return "%s %s" % (s, size_name[i])

# calculates the total memory necessary for training a model
def calc_mem(args):
    dp_degree = args.num_gpus / (args.tensor_parallel_size * args.pipeline_parallel_size)

    # 4 bytes in fp32, 2 bytes in fp16/bf16
    if args.fp32_model:
        bytes_per_param = 4
    else:
        bytes_per_param = 2

    # Split the model with 3D parallelism
    model_mem = (args.params * bytes_per_param) / (args.tensor_parallel_size * args.pipeline_parallel_size)
    # ZeRO stage 3 shards the model parameters across GPUs (plus the gradients and optimizer states)
    if args.zero_stage == 3:
        model_mem /= args.num_gpus

    # 4 bytes in fp32, 2 bytes in fp16/bf16
    if args.fp32_grads:
        bytes_per_grad_element = 4
    else:
        bytes_per_grad_element = 2

    gradient_mem = args.params * bytes_per_grad_element
    # ZeRO stage 2 shards the gradients across GPUs (plus the optimizer states)
    if args.zero_stage >= 2:
        gradient_mem /= args.num_gpus
    gradient_mem /= args.pipeline_parallel_size

    # For mixed-precision Adam/AdamW, the optimizer must store fp32 copies of the parameters, momentum, and variance (4 + 4 + 4 = 12 bytes per optimizer parameter)
    # Feel free to change the multiplier for your optimizer (examples include SGD (4 + 4 = 8) and 8-bit ADAM (2 + 2 + 2 = 6)
    optimizer_mem = args.params * 12
    # ZeRO stage 3 shards the optimizer states across GPUs
    if args.zero_stage >= 1:
        optimizer_mem /= args.num_gpus

    communication_mem = 0
    # The size of the communication buffer DeepSpeed uses to store ZeRO optimizer elements
    if args.zero_stage >= 1:
        communication_mem += args.zero_allgather_bucket_size * bytes_per_param
    # The number of parameters ZeRO-3 keeps alive in GPU memory at a time
    if args.zero_stage == 3:
        communication_mem += args.zero3_max_live_params * bytes_per_param

    # Taken from Table 2 in https://arxiv.org/pdf/1910.02054.pdf
    # We find these don't perfectly match with experiment, but are good approximations
    if args.checkpoint_activations:
        activation_mem = args.sequence_length * args.batch_size_per_gpu * args.hidden_size * args.num_layers * (10 + (24 / args.tensor_parallel_size))
    else:
        activation_mem = args.sequence_length * args.batch_size_per_gpu * args.hidden_size * args.num_layers * (10 + (24 / args.tensor_parallel_size) + 5 * ((args.num_attention_heads * args.sequence_length) / (args.hidden_size * args.tensor_parallel_size)))

    # DeepSpeed's ZeRO-R partitions activation memory across tensor-parallel GPUs
    if args.partition_activations:
        activation_mem /= args.tensor_parallel_size


    # We include a "Miscellaneous Memory" term because we find some 3D-parallel frameworks add a constant memory overhead (~5GB in our experiments with Megatron-DeepSpeed) that we cannot explain. If you know the source of this, add a comment!
    gradient_mem_gb = gradient_mem / 1024**3
    activation_mem_gb = activation_mem / 1024**3
    model_mem_gb = model_mem / 1024**3
    optimizer_mem_gb = optimizer_mem / 1024**3
    communication_mem_gb = communication_mem / 1024**3
    total_mem_gb = activation_mem_gb + gradient_mem_gb + model_mem_gb + optimizer_mem_gb + communication_mem_gb + args.misc_mem_gb
    st.write(f'Number of Parameters: {convert_params(args.params)}')
    st.write(f'Gradient Memory: {gradient_mem_gb:.2f} GB')
    st.write(f'Activation Memory: {activation_mem_gb:.2f} GB')
    st.write(f'Model Memory: {model_mem_gb:.2f} GB')
    st.write(f'Optimizer Memory: {optimizer_mem_gb:.2f} GB')
    st.write(f'Communication Memory: {communication_mem_gb:.2f} GB')
    st.write(f'Miscellaneous Memory: {args.misc_mem_gb:.2f} GB')
    st.markdown(f"""
            <div class="card">
                <strong>Min GPU Memory Required: {total_mem_gb:.2f} GB</strong>
            </div>
            """, unsafe_allow_html=True)

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

    st.markdown("""---""")

    st.title("Transformer Memory Calculator")

    # Creating UI elements for each argument:
    params = st.number_input("Number of Parameters", min_value=1, value=20000000000, step=1)
    num_gpus = st.number_input("Number of GPUs used for training", min_value=1, value=1, step=1)
    tensor_parallel_size = st.number_input("Tensor parallel degree", min_value=1, value=1, step=1)
    pipeline_parallel_size = st.number_input("Pipeline parallel degree", min_value=1, value=1, step=1)
    partition_activations = st.checkbox("Use ZeRO-R to partition activation memory?")
    zero_stage = st.selectbox("Stage of the ZeRO optimizer", [0, 1, 2, 3], index=1)
    checkpoint_activations = st.checkbox("Use Megatron-style activation checkpointing?")
    batch_size_per_gpu = st.number_input("Batch size per GPU", min_value=1, value=1, step=1)
    hidden_size = st.number_input("Dimension of the model's hidden size", min_value=1, value=6144, step=1)
    num_attention_heads = st.number_input("Number of attention heads used in model", min_value=1, value=64, step=1)
    sequence_length = st.number_input("Sequence length used for training", min_value=1, value=2048, step=1)
    num_layers = st.number_input("Number of transformer layers used in model", min_value=1, value=44, step=1)
    fp32_model = st.checkbox("Is model stored in fp32?")
    fp32_grads = st.checkbox("Are grads stored in fp32?")
    zero_allgather_bucket_size = st.number_input("Size of allgather buckets used by ZeRO", min_value=1, value=int(5e8), step=1)
    zero3_max_live_params = st.number_input("Maximum number of parameters ZeRO3 keeps in GPU memory", min_value=1, value=int(1e9), step=1)
    misc_mem_gb = st.number_input("Miscellaneous memory overhead", min_value=0, value=0, step=1)

    # When the user clicks this button, calculate the memory:
    if st.button("Calculate Memory"):
        # Create an object to mimic the argparse Namespace object:
        args = type('', (), {})()
        args.params = params
        args.num_gpus = num_gpus
        args.tensor_parallel_size = tensor_parallel_size
        args.pipeline_parallel_size = pipeline_parallel_size
        args.partition_activations = partition_activations
        args.zero_stage = zero_stage
        args.checkpoint_activations = checkpoint_activations
        args.batch_size_per_gpu = batch_size_per_gpu
        args.hidden_size = hidden_size
        args.num_attention_heads = num_attention_heads
        args.sequence_length = sequence_length
        args.num_layers = num_layers
        args.fp32_model = fp32_model
        args.fp32_grads = fp32_grads
        args.zero_allgather_bucket_size = zero_allgather_bucket_size
        args.zero3_max_live_params = zero3_max_live_params
        args.misc_mem_gb = misc_mem_gb
        
        calc_mem(args)

if __name__ == "__main__":
    main()
