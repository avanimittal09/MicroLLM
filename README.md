# MicroLLM
microLLM is a decoder-only Transformer language model and reasoning engine built from scratch in PyTorch. 

The project implements the complete pipeline—tokenizer integration, multi-head attention mechanisms, masking, target loss masking, and reinforcement learning alignment algorithms

---

# Overview & Motivation

How do modern AI models learn to "think" step-by-step before answering a hard math question?

To find out, I built **MicroLLM**—a ~28M-parameter Large Language Model written entirely from scratch in PyTorch. Instead of using pre-built model libraries, I coded the Transformer architecture myself and trained it across three progressive stages on a single Google Colab GPU:

1. **Pre-training:** Learning English language patterns and basic math concepts from raw text.
2. **Supervised Fine-Tuning (SFT):** Learning to follow instructions and write step-by-step thoughts inside `<think>...</think>` tags.
3. **Reinforcement Learning (GRPO):** Practicing math word problems through trial-and-error using a custom reward grading function.

---

##  Model Architecture

MicroLLM is a decoder-only Transformer inspired by the GPT-2 architecture, scaled down so it can be trained on a single cloud GPU.

| Parameter / Component | Value | Why I Chose It |
| :--- | :--- | :--- |
| **Parameters** | ~28M – 30M | Small enough for a single Google Colab GPU, large enough to learn basic math reasoning. |
| **Layers (`n_layers`)** | `6` | 6 stacked Transformer blocks (Attention + Feed-Forward MLP). |
| **Embedding Dimension (`emb_dim`)** | `384` | Each token is represented as a 384-dimensional vector. |
| **Attention Heads (`n_heads`)** | `12` | 12 parallel attention heads (`32` dimensions per head) using Flash Attention. |
| **Context Length** | `1,024` tokens | Processes up to 1,024 tokens at once. |
| **Vocabulary Size** | `50,304` | Uses OpenAI's `gpt2` tokenizer (50,257 tokens), padded to a multiple of 64 for faster GPU math. |
| **Weight Tying** | Enabled | Shares weights between the input token embedding and output layer to save parameters and improve training. |

---

##  Public Datasets Used

All datasets are loaded from Hugging Face using the `datasets` library:

| Stage | Dataset | What It Contains | How It Is Used |
| :--- | :--- | :--- | :--- |
| **Phase 1: Pre-training** | `HuggingFaceFW/fineweb-edu` | High-quality educational web pages | **85% mix** — Teaches general language, grammar, and facts. |
| **Phase 1: Pre-training** | `open-web-math/open-web-math` | Math articles and problem discussions | **15% mix** — Introduces mathematical notation and logic early on. |
| **Phase 2: SFT** | `nvidia/OpenMathInstruct-1` | Step-by-step math solutions | **10,000 samples** — Teaches structured chain-of-thought reasoning. |
| **Phase 2 & 3: SFT & RL** | `gsm8k` | Grade School Math 8K word problems | Used in SFT for step-by-step training and in GRPO for reinforcement learning. |

---

##  The 3-Stage Training Pipeline

### Stage 1: Pre-training (`run_pretraining`)
In pre-training, the model starts with random weights and learns to predict the next token in a sequence.
* **Dataset Streaming:** Because `FineWeb-Edu` and `OpenWebMath` are massive, I used Hugging Face `streaming=True` and `interleave_datasets` (85% Edu / 15% Math) to stream text directly over the internet without filling up disk space.
* **Sequence Packing (`PackedDS`):** Documents are joined with `<|endoftext|>` tokens and sliced into continuous 1,024-token blocks so no GPU compute is wasted on padding.

### Stage 2: Supervised Fine-Tuning (`run_sft_pipeline`)
A pre-trained model only knows how to continue text—it doesn't know how to answer questions. In SFT, I formatted math problems using a `<think>` template:
```text
Question: If I have 3 apples and eat 1, how many do I have?
Solve step by step.
<think>
{step-by-step reasoning}
</think>
<|endoftext|>
```
* **Prompt Masking:** During training, I masked the question tokens with `-100` so the loss function only grades the model on its `<think>` reasoning steps and final answer—not on memorizing the question.

### Stage 3: Reinforcement Learning with GRPO (`run_grpo`)
To help the model reason better on its own, I implemented **Group Relative Policy Optimization (GRPO)**:
1. **Generate Multiple Attempts:** For each GSM8K math question, the model generates `4` different step-by-step attempts at once.
2. **Score Each Attempt (`get_rewards`):** A custom rule-based grading function checks every response and assigns a score:
   * **+2.0 points** if the final numerical answer matches the ground truth.
   * **+0.1 to +1.0 bonus points** for writing clear, multi-line reasoning steps (only awarded if the final answer is right).
   * **Penalties (-0.5 to -10.0)** if the model skips the `<think>` block, leaks the answer early, repeats the same line in a loop, or hallucinates code blocks.
3. **Learn from the Best:** The model compares the 4 scores against the group's average and updates its weights to favor the reasoning paths that scored above average, while using KL-divergence to stay close to the SFT model.

---

## ☁️ How I Trained It on Google Colab

Training an LLM from scratch on a single Google Colab GPU came with real hardware limits (VRAM limits and session timeouts). Here is how I solved them:

* **Google Drive Auto-Checkpointing:** Every few hundred steps, the script saves model and optimizer weights to Google Drive using a safe temporary file (`os.replace(tmp_path, full_path)`). If Colab disconnects, re-running the cell automatically resumes from the exact step and fast-forwards the dataset stream.
* **FP16 Mixed Precision (`torch.amp.autocast`):** Runs forward and backward passes in 16-bit floating point to cut GPU memory usage in half and roughly double training speed.
* **Gradient Accumulation:** Uses a micro-batch size of `4` and accumulates gradients over `8` steps (`effective batch size = 32`), allowing stable training without running out of GPU memory.
* **Batched RL Generation:** In the GRPO loop, all `batch_size × group_size` prompts (16 sequences) are stacked and generated in a single parallel GPU loop rather than one by one.
ng.
