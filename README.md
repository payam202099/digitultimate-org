# DigitUltra_1Q_Base

This repository contains JAX example code for loading and running the DigitUltra_1Q_Base open-weights model.

Then, run

```shell
pip install -r requirements.txt
python run.py
```

to run the code.

This model with 117 parameters can simultaneously present image, audio, video, and text in one content.
 This model has video chat capabilities and you have to pay $5 per month to use it. 
# Model Specifications

DigitUltra_1Q_Base is currently designed with the following specifications:

- **Parameters:** 117T
- **Architecture:** Mixture of 128 Experts (MoE),Holographic, quantum,MOE QUANTUM
- **Experts Utilization:** 64 experts used per token
- **Layers:** 1024
- **Attention Heads:** 150 for queries, 64 for keys/values
- **Embedding Size:** 16,384
- **Tokenization:** SentencePiece tokenizer with 800,000 tokens
- **Additional Features:**
  - Rotary embeddings (RoPE)
  - Supports activation sharding and 8-bit quantization
- **Maximum Sequence Length (context):** 2,050,000 tokens
# License

The code and associated DigitUltra_1Q_Base weights in this release are licensed under the
Apache 2.0 license. The license only applies to the source files in this
repository and the model weights of DigitUltra_1Q_Base.
