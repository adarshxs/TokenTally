# ⚙️ TokenTally
### Estimate Your LLM's Token Toll Across Various Platforms and Configurations

![cutie](https://github.com/adarshxs/TokenTally/assets/114558126/0f584e00-5bf8-4763-a885-8ca5a7e87ee9)

🎯The goal is to be able to calculate the **minimum GPU requirements** for **Training**(Fine Tuning and Continued Pre Training) and **Inference** for any LLM along with Comparison to Self-Host these models across different GPU Cloud Platforms and Optimizations. Eventually to Calculate tokens/$ for every possible combinations of Model, Platform and Optimizations!

---
## Inference Calculations:
The current formula to calculate the minimum size for inference and training is from the findings in [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) and simplified by [Transformer Math 101](https://blog.eleuther.ai/transformer-math/)

### Minimum GPU Memort Requirement for Inference:
```
Model_size * 1.2 
# according to the blog, where 20% is a good buffer to accomodate for any overheads
```
### Cost per 1,000 tokens - For Selected Model
To provide a rough estimate of the total cost of running an LLM on a per 1000 token basis, the logic from https://cursor.sh/blog/llama-inference is followed in the calculator:

1. **Retrieve User Inputs:** 
   - The following parameters are required:
     - FLOPs per Token
     - FLOPs per GPU (TFLOPs)
     - Number of GPUs
     - Cost per Hour (USD)
     - Memory Bandwidth per GPU (TB/s)

2. **Compute Tokens per Second (Compute Bound):**
   - The number of tokens processed per second when compute bound is calculated using the formula:
     ```python
     tokens_p_sec_compute = (flops_per_gpu * num_gpus * 10**12) / (flops_per_token * 10**9)
     ```

3. **Compute Tokens per Second (Memory Bound):**
   - The number of tokens processed per second when memory bound is calculated using the formula:
     ```python
     tokens_p_sec_memory = (memory_bandwidth_per_gpu * num_gpus * 10**12) / (flops_per_token * 10**9)
     ```

4. **Calculate Cost per Token:**
   - The cost per token for both compute and memory bounds is determined using the formula:
     ```python
     cost_p_token_compute = cost_per_hour / (3600 * tokens_p_sec_compute)
     cost_p_token_memory = cost_per_hour / (3600 * tokens_p_sec_memory)
     ```

5. **Calculate Cost per 1,000 Tokens:**
   - The cost per 1,000 tokens for both compute and memory bounds is determined using the formula:
     ```python
     cost_p_1k_tokens_compute = cost_p_token_compute * 1000
     cost_p_1k_tokens_memory = cost_p_token_memory * 1000
     ```

Upon inputting the necessary parameters and clicking on the "Calculate" button, the calculator will display the estimated cost per 1,000 input tokens and cost per 1,000 output tokens based on the above logic.


---

## Transformer Memory Calculations:
Training large transformer models often poses challenges in terms of GPU memory requirements. This tool estimates the memory consumption given a range of parameters, such as model size, sequence length, batch size, and parallelization settings.
### Key Memory Components:

1. **Model Memory**: 
   - Memory occupied by the model's weights.
   - Influenced by the number of parameters and their data types (e.g., FP32 vs. FP16).
   - Adjusted based on parallelism settings, like tensor and pipeline parallel degrees.

2. **Gradient Memory**: 
   - Memory needed to store gradients during backpropagation.
   - Affected by the number of parameters, their data types, and the ZeRO optimizer stage.
   - Reduced with pipeline parallelism.

3. **Optimizer Memory**: 
   - Memory occupied by the optimizer's states.
   - For mixed-precision Adam/AdamW, three copies (parameters, momentum, variance) are stored, which means 12 bytes per parameter.
   - Modified by the ZeRO optimizer stage.

4. **Activation Memory**: 
   - Memory used for storing activations during forward and backward passes.
   - Calculated based on sequence length, batch size, hidden size, and the number of layers.
   - Can be influenced by techniques like activation checkpointing and activation partitioning.

5. **Communication Memory**: 
   - Additional memory needed for operations like all-reduce during distributed training.
   - Particularly pertinent when using ZeRO optimization.

6. **Miscellaneous Memory**: 
   - An overhead term to account for other unaccounted memory usage by deep learning frameworks, communication libraries, etc.

### Usage

To get memory estimates:

1. Input your desired model configurations.
2. The tool will calculate memory requirements for each component and provide a total estimate.



---
## Contributions!
Looking For Contributions to implement the logic and Crowdsource relevant data!

---


All props to https://github.com/cloud-gpus/cloud-gpus.github.io from where I stole the list of available GPUs and their pricing, https://huggingface.co/spaces/mithril-security/TCO_calculator from where I used the TCO logic and https://gist.github.com/Quentin-Anthony/f43939791a7ceb0b01a4937308317be5 for Transformer memory logic as is ;) 
And to [Dr. Pratik Desai](https://x.com/chheplo?s=20) for the idea!
