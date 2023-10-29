import streamlit as st

from utilities import convert_params

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