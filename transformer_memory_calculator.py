import streamlit as st

from memory_calculation import calc_mem

def display_transformer_memory_tool():
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