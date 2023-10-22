# ⚙️ TokenTally
### Estimate Your LLM's Token Toll Across Various Platforms and Configurations

![cutie](https://github.com/adarshxs/TokenTally/assets/114558126/0f584e00-5bf8-4763-a885-8ca5a7e87ee9)

🎯The goal is to be able to calculate the **minimum GPU requirements** for **Training**(Fine Tuning and Continued Pre Training) and **Inference** for any LLM along with Comparison to Self-Host these models across different GPU Cloud Platforms and Optimizations. Eventually to Calculate tokens/$ for every possible combinations of Model, Platform and Optimizations!

---
## Formula
The current formula to calculate the minimum size for inference and training is from the findings in [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) and simplified by [Transformer Math 101](https://blog.eleuther.ai/transformer-math/)

### Minimum GPU Memort Requirement for Inference:
```
Model_size * 1.2 
(according to the blog, where 20% is a good buffer to accomodate for any overheads
```
### Minimum GPU Memort Requirement for Inference:
```
Total_Memory_Train = Model_Memory + Optimiser_Memory + Activation_Memory + Gradient_Memory

Yet to be implemented according to the blog!
```

### Variables:
- `BatchSz`: Batch size
- `SeqLen`: Sequence length
- `Layers`: Number of layers
- `AttnHeads`: Number of attention heads
- `HiddenDim`: Hidden dimensions
- `BytePrec`: Bytes of precision (e.g., float32 would have 4 bytes)

### Equations:

#### Activations per Layer:
`Activations per Layer = SeqLen * BatchSz * HiddenDim * (34 + ((5 * AttnHeads * SeqLen) / HiddenDim))`

#### Total Activations:
`Activations = Layers * (5 / 2) * AttnHeads * BatchSz * SeqLen^2 + 17 * BatchSz * HiddenDim * SeqLen`

#### Total Size in GB:
`Total Size (in GB) = BytePrec * (params + Activations)`


---
## Contributions!
Looking For Contributions to implement the logic and Crowdsource some data!

---


All props to https://github.com/cloud-gpus/cloud-gpus.github.io from where I stole the list of available GPUs and their pricing ;) and to [Dr. Pratik Desai](https://x.com/chheplo?s=20) for the idea!
