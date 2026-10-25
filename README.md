# MicroLLM
microLLM is a decoder-only Transformer language model and reasoning engine built from scratch in PyTorch. 

The project implements the complete pipeline—tokenizer integration, multi-head attention mechanisms, masking, target loss masking, and reinforcement learning alignment algorithms

---

## Features

- **From-Scratch Architecture:** Custom implementation of Multi-Head Self-Attention, GELU activation, Layer Normalization, and Causal Masking in PyTorch.
- **Phase 1: Pre-training:** Data streaming pipeline that interleaves text from `TinyStories` and `FineWeb-Edu` datasets for causal language modeling.
- **Phase 2: Supervised Fine-Tuning (SFT):** Multi-dataset fine-tuning (`Alpaca`, `GSM8K`, and `OpenOrca`) with custom target-masking cross-entropy loss to compute gradients and to teach structures <think>... </think> step by step reasoning exclusively on model generated responses.
- **Phase 3: Group Relative Policy Optimization (GRPO):** Implementation of GRPO alignment—a model-free reinforcement learning technique—using group relative rewards, KL-divergence penalties, and custom reward function for reasoning and structural output validation.
  

---

## Model Specifications

Configuration defined in `TINY_TM_CONFIG`:

- **Vocabulary Size:** 50,257 tokens (tiktoken `gpt2` encoding)
- **Context Window:** 1,024 tokens
- **Embedding Dimension:** 384
- **Attention Heads:** 6
- **Transformer Layers:** 6

---

## Pipeline Overview

1. **Pre-training (`run_pretraining`):** Streams tokenized sequences from mixed datasets and trains the core language model using next-token prediction.
2. **Supervised Fine-Tuning (`run_sft`):** Splits instruction and mathematical datasets into Train/Test/Validation sets (85/10/5) and trains the model to follow prompt structures.
3. **Reasoning Alignment (`run_grpo`):** Generates output groups per prompt, evaluates math accuracy and structural tags (`<think>...</think>`), and updates policy via relative reward advantage scaling.
