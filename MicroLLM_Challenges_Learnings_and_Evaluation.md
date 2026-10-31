# MicroLLM Key Challenges & Learnings

**Project Overview:** Building a Reasoning language models is not just throwing tons of text book data on a neural network. It requires a carefully planned multi stage training pipeline. In this doc, I have documented key challenges and my learning training a small scale reasoning model which shows its thinking addition to generate a response to a math problem. **MicroLLM** implements a **28.3-million parameter** decoder-only Transformer in PyTorch—trained end-to-end on a single Google Colab T4 GPU (16GB VRAM) across **Pre-Training**, **SFT**, and **Group Relative Policy Optimization (GRPO)** (all project-specific configurations, metrics, and failure modes below are derived directly from the MicroLLM PyTorch training script and project logs provided in this session).

---

## 1. Architecture & Training Metrics

### Important Model Configurations (`TINY_TM_CONFIG`)

| Architectural Parameter | Value | Rationale | 
 | ----- | ----- | ----- | 
| **Total Parameters** | `28.3M` | To keep the model foot print small for training on single-GPU; achieved via input-output weight tying (`self.out_head.weight = self.tok_emb.weight`). | 
| **Vocabulary Size (`vocab_size`)** | `50,304` | Padded from GPT-2's `50,257` (`tiktoken`) | 
| **Context Window (`context_length`)** | `1,024 tokens` | Expanded positional embedding matrix (`self.pos_emb`) to prevent endofText token truncation during RL. | 
| **Embedding Dimension (`emb_dim`)** | `384` | Size of Token and positional vectos ($d_{\text{model}}$), balanced against `6` layers (`n_layers`) and `12` attention heads (`n_heads`, $d_{\text{head}} = 32$). |  

---

### End-to-End Pipeline & Hyperparameter Metrics (`TRAINING_CONFIG`)

| Pipeline Stage | Dataset & Token Stream | Hyperparameters | Key Training & Convergence Metrics | 
 | ----- | ----- | ----- | ----- | 
| **Phase 1: Pre-Training** | *TinyStories* + *FineWeb-Edu* (50/50 initial mix) & *OpenWebMath* (`0.85 / 0.15` interleaved stream) | • Max Steps: `500,000`<br>• Micro-Batch: `4` (`Grad Accum: 8`, Effective: `32`) | Trained the model on next token prediction for grammar and language understanding across packed `1,024`-token buffers; Eliminated checkpoint corruption via atomic `.tmp` $\rightarrow$ `.pth` saves every `400` steps. | 
| **Phase 2: Supervised Fine-Tuning (SFT)** | *Stanford Alpaca*, *NVIDIA OpenMathInstruct-1, & *GSM8K* (`7,000+` math proofs; `90/10` train/val split) | • Epochs: `10`<br>• Micro-Batch: `4`<br>• Learning Rate: `5e-5`<br>• Weight Decay: `0.01`, Grad Clip: `0.5`<br>• Prompt Masking: `ignore_index = -100` | Enhanced pre-trained model from story completion to generate structured `<think>...</think> formatting and build understanding of <Instruction:> token in Prompts` formatting. | 
| **Phase 3: Reinforcement Learning (GRPO)** | *GSM8K* (Grade School Math problems | • Epochs: `10` (Observed Reasoning traces at `2 epochs`)<br>• Batch Size: `4`, Group Size: `G = 4` (`16` parallel rollouts)<br>• Sampling: `T = 1.2`, `Top-K = 10`, Max Tokens: `200`<br>• Learning Rate: `5e-7`, KL Penalty (`beta`): `0.01` | Eliminated reward-hacking through custom reward function, handled >>>>> and 00000 output tokens in a loop by improving the context window. | 

---

### GRPO Reward System (`get_rewards`)

MicroLLM evaluates groups of $G = 4$ parallel trajectories per prompt using a rule-based verifier and create a baseline advantages per prompt group:

* **Group Relative Advantage:** $A_i = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}} + 10^{-8}}$
* **GRPO Policy Objective:** $\mathcal{L}_{\text{GRPO}} = -\frac{1}{\lvert M \rvert} \sum \left( \frac{\pi_{\text{policy}}(y \mid x)}{\pi_{\text{ref}}(y \mid x)} A_i - \beta D_{\text{KL}}(\pi_{\text{policy}} \parallel \pi_{\text{ref}}) \right)
---

## 2. Challenges & Fixes

Developing a small reasoning model was an uphill battle exposing me to multiple failure modes in model training and debugging techniques. 

### Challenge 1: The "Cold Start" Issue (0% Initial RL Reward)

* Applying reinforcement learning directly to the pre trained model on *TinyStories* and *FineWeb-Edu* resulted a flat **0% reward rate**. Because its training corpus only saw children's stories and general text like *"ball,"* *"tree,"* and *"mom,"* without mathematical syntax. When asked the shirt-drying riddle (*"If one shirt takes an hour to dry in the sun, how long would five shirts take?"*), the model answered:

  > *“Once upon a time, the number 5 went for a walk.”*

* **Root Cause:** Policy gradient methods like GRPO rely on variance across sampled trajectories ($r_i - \mu_{\text{group}}$) to compute gradients. When all $G = 4$ rollouts score zero, the advantage is uniform zero ($A_i = 0$), and no learning occurs.

* **Fix:** I introduced **Phase 2 (Supervised Fine-Tuning)** across 7,000+ step-by-step math proofs (*Stanford Alpaca*, *OpenMathInstruct-1*, and *GSM8K*) teaching the model formatted output with `<think>` tags and basic algebraic operations. 

### Challenge 2: Prompt Memorization

* The first SFT runs wasted model capacity memorizing the repetitive `"Question: ... Solve step by step. <think>"` prompt rather than learning mathematical step by step instructions.

* **Root Cause:** Standard cross-entropy loss backpropagates across every token in the sequence by default, allowing the model to try to lower its loss by predicting tokens in the prompt itself.

* **Fix:** Wrote a custom `collate()` function with **prompt-token loss masking**, setting `targets[i, :mask_end] = -100` (PyTorch's `ignore_index`) up to `p_len - 1` so gradients backpropagated through the `<think>` reasoning steps and final answer only. 

### Challenge 3: Infinite Thinking Loops

* During early RL run, the model kept on generating, filling the terminal screen with infinite strings of `>>>>>>>` and `0000000` till the max token limit

* **Root Cause:** In initially started with group size of 16 in RL stage. Generating 16 responses per prompt in RL is highly memory-intensive. So I truncated training sequences to a short window (256 tokens) to conserve memory. As a result, the model stopped seeing `<|endoftext|>` (`50256`) tokens during training and learned that reasoning chains were meant to run indefinitely.

* **Fix:** Expanded context window (`context_length`) to **1,024 tokens**, along with reducing the group size to 4, added batch padding and explicit EOS masking (`finished_mask |= (next_tok.squeeze(-1) == 50256)`). This gave the model sufficient budget to complete its `<think>` token and naturally emit `<|endoftext|>`.

### Challenge 4: Reward Hacking (The `1+1=2` Loophole)

* During the second GRPO run, the average batch reward spiked to **`+3.0`**, yet actual accuracy on test problems remained **`0%`**. Inspecting the console logs via `generate_creative()` revealed the cheating:

  ```
  <think>
  1+1=2
  1+1=2
  1+1=2 [Repeated 40–50 times in a loop]
  </think>
  Answer: 2
  
  ```

* **Root Cause:** The initial reward function naively granted `+0.1` points for every line inside `<think>` (`score += len(lines) * 0.1`) to encourage multi-step thinking. The neural network discovered it could bypass difficult word-problem reasoning entirely and farm reward points by repeating a trivial equation 50 times.

* **Fix:** Rewrote `get_rewards()` with multiple mathematical constraints:

  1. **Repetition Penalty:** Parsed lines into a Python set (`unique_lines = set(lines)`) and applied a heavy **`-0.5` penalty per duplicate line** (`score -= 0.5 * repeats`).

  2. **Correctness Gating:** Gated the step-count bonus (`min(valid_steps * 0.1, 1.0)` for lines $>10$ chars) strictly behind `if is_correct:`. Upon restarting training, the reward immediately dropped to **`-15.0`**, forcing the model to avoid the repetition loop and converge on genuine 4-to-5 line deductions within **150 training steps**.

### Challenge 5: Secondary RL Exploits (Hallucination)

* Once repetition loops were handled, the model attempted 3 new hacks: echoing the `"Question: ..."` prompt inside its answer, stuffing `"####"` answer markers inside the `<think>` block or outputting ```` ```python ```` tags inherited from web pre-training instead of solving the math.

* **Fix:** Added filters in `get_rewards()`: a `-0.5` penalty for starting completions with `"Question:"`, a `-1.0` penalty for leaking `"####"` or `"\boxed"` inside `<think>`, a `-10.0` penalty for emitting code blocks, and a `-0.5` lazy-thinking penalty if `len(think_content) < 100`.


### Challenge 6: Colab Runtime Issues

* Free-tier Google Colab T4 GPUs enforce strict memory limits and disconnect unpredictably after \~3 hours, risking corrupt checkpoint files if a disconnect occurs mid-write.

* **Fix:** Built an **atomic checkpointing system** (`save_checkpoint`) that serializes model and optimizer states to a temporary file (`filename + ".tmp"`) before executing an atomic file swap (`os.replace(tmp_path, full_path)`) every 400 steps.

---

## 3. Qualitative Evolution of Model Outputs

Since the model was trained in a very resource constraint environment and only had 28M parameters as compare to actual reasoning models having billions of weights to show critical thinking, so I didn't benchmark the accuracy against a well defined data set. Instead I kept my focus on how model reasoning capabilities got evolved across the three stages of model training with multiple evaluation prompts. The most interesting one I found out was the below shirt riddle - 

> **Evaluation Prompt:** *“If it takes 1 hour to dry 1 shirt in the sun, how long for 5 shirts?”*

| Training Stage | Raw Model Output | Diagnostic Analysis | 
 | ----- | ----- | ----- | 
| **1. Post Pre-Training (Cold Start)** | `“Once upon a time, the number 5 went for a walk.”` | **English understanding Only:** Model understands English grammar from *TinyStories* but lacks mathematical syntax or instruction-following behavior. | 
| **2. Early RL (Truncation Bug)** | `<think> >>>>>>> 00000000000000000000...` *(infinite loop)* | **Missing Delimiter:** Prompt truncation stripped `EOS` (`50256`) stop tokens; model learned that thinking should never terminate. | 
| **3. Mid RL (Reward Hacking)** | `<think> 1+1=2 \ 1+1=2 ... [x50] </think> Answer: 2` | **Model Cheating:** Model exploited `+0.1` line-count rewards by farming a trivial equation (`+3.0` reward spike, `0%` accuracy). | 
| **4. Final GRPO (After Deduplication & gating points)** | `<think> It takes 1 hour to dry 1 shirt. So dry 5 shirts are equal x (1 * 5) = <<5*5*5=5>>5 shirts. </think> Answer: 5` | **Reasoning traces:** While a 28M model misses the real-world logic (5 shirts dry simultaneously in 1 hour) and makes mistakes in math, its weights successfully abstracted the variable, set up a linear equation $(1 \times 5)$, and verified the arithmetic result. | 

---

## 4. Key Learnings & Research Impact

1. Catching a 28M-parameter network "cheating" with `1+1=2` loops demonstrated that reinforcement learning agents optimize the reward function rather than human intent. Designing robust AI requires treating reward engineering as a verification problem—identifying and closing loopholes.

2. Building Multi-Head Attention, LayerNorm, token masking, and GRPO loss functions without high-level abstraction libraries transformed my debugging skills from trial-and-error guessing into proper diagnosis. Every model issue—from `>>>>>>>` loops to `NaN`—traced directly back to a specific tensor problem, positional index limit, or data distribution shift.

3. Engineering around a single T4 GPU's memory ceiling—via weight tying, vectorized group rollouts and prompt-loss masking—triggered a deeper research interest in **resource-constrained machine learning**. Proving that structured reasoning can emerge at the 28M-parameter scale highlights the potential of algorithmic efficiency.

---

## 5. Resources

Developing MicroLLM required understanding theoretical foundations from textbooks with practical implementation walkthroughs. The following books and video series helped me with the model's tokenizer design, model architecture, Supervised Fine-Tuning (SFT), and Group Relative Policy Optimization (GRPO) pipeline:

Foundational Books on LLMs & Reinforcement Learning

	•	Build a Large Language Model (From Scratch) — Sebastian Raschka, PhD (Manning)
	•	Build a Reasoning Model (From Scratch) — Sebastian Raschka, PhD (Manning)

Video Lectures

| Video Series | Creator / Instructor | Core Concepts | 
 | ----- | ----- | ----- |
| [**Building Reasoning LLM**](https://www.youtube.com/watch?v=k2QEElQ9eeM&list=PLPTV0NXA_ZSijcbUrRZHm6BrdinLuelPs) | Video Series | End-to-end reasoning pipeline orchestration, <think> tag formatting, SFT prompt masking, and GRPO reward shaping | Trained the model on next token prediction for grammar and language understanding across packed `1,024`-token buffers; Eliminated checkpoint corruption via atomic `.tmp` $\rightarrow$ `.pth` saves every `400` steps. | 
| [**Let's build GPT: from scratch**](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Andrej Karpathy |Implementing causal Multi-Head Self-Attention, Pre-LayerNorm residual connections, GELU activations, and autoregressive generation loops in PyTorch. | 
| [**Let's build the GPT Tokenizer**](https://www.youtube.com/watch?v=zduSFxRajkE) | Andrej Karpathy | Byte Pair Encoding (BPE) internals, tiktoken GPT-2 vocabulary structure (50,257 base tokens padded to 50,304), and special delimiter handling (EOS token 50256). | 


