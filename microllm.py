# -*- coding: utf-8 -*-
"""MicroLLM.ipynb
"""

!pip install tiktoken datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import time
import os
import json
import ssl
import re
import copy
import random
import math
import urllib.request
import datetime
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, IterableDataset, DataLoader, random_split
from datasets import load_dataset, interleave_datasets

try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = "/content/drive/MyDrive/MicroLLM"
    print("Running on Google Colab. Saving to Drive.")
except ImportError:
    BASE_DIR = "./models"
    print("Running Locally. Saving to ./models")

os.makedirs(BASE_DIR, exist_ok=True)

TINY_TM_CONFIG = {
    "vocab_size": 50304,
    "context_length": 1024,
    "emb_dim": 384,
    "n_heads": 12,
    "n_layers": 6,
    "drop_rate": 0.05,
    "qkv_bias": True
}

TRAINING_CONFIG = {
    "pretraining": {
        "epochs": 1,
        "max_steps": 100000,
        "batch_size": 4,
        "grad_accum_steps": 8,
        "lr": 6e-4,
        "weight_decay": 0.1,
        "checkpoint_interval": 400
    },
    "sft": {
        "dataset": "nvidia/OpenMathInstruct-1",
        "subset": "default",
        "epochs": 10,
        "batch_size": 4,
        "grad_accum_steps": 8,
        "lr": 5e-5,
        "weight_decay": 0.01,
        "checkpoint_interval": 400
    },
    "grpo": {
        "dataset": "gsm8k",
        "epochs": 10,
        "batch_size": 4,
        "group_size": 4,
        "grad_accum_steps": 1,
        "lr": 5e-7,
        "beta": 0.01
    }
}

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg["n_heads"]
        self.head_dim = cfg["emb_dim"] // cfg["n_heads"]
        self.W_q = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.W_k = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.W_v = nn.Linear(cfg["emb_dim"], cfg["emb_dim"], bias=cfg["qkv_bias"])
        self.out_proj = nn.Linear(cfg["emb_dim"], cfg["emb_dim"])

    def forward(self, x):
        b, seq, _ = x.shape
        q = self.W_q(x).view(b, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(b, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(b, seq, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0 if not self.training else 0.05)
        y = y.transpose(1, 2).contiguous().view(b, seq, -1)
        return self.out_proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(cfg)
        self.ff = nn.Sequential(nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), GELU(), nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]))
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
    def forward(self, x):
        x = x + self.att(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class MicroLLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # IMPROVEMENT 1: Weight Tying (Improves performance and saves params)
        self.out_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

        # IMPROVEMENT 2: Residual Scaling for Deep Networks
        # Scale output projections by 1/sqrt(2 * n_layers)
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('ff.2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * cfg["n_layers"]))

    # IMPROVEMENT 3: GPT-2 Standard Initialization (Normal vs Xavier)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        seq = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(seq, device=idx.device))
        x = self.blocks(x)
        return self.out_head(self.final_norm(x))

# ==========================================
# 3. UTILITIES
# ==========================================
def get_path(filename):
    return os.path.join(BASE_DIR, filename)

def save_checkpoint(model, optimizer, step, epoch, filename):
    full_path = get_path(filename)
    tmp_path = full_path + ".tmp"
    torch.save({
        'model_state_dict': model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'step': step,
        'epoch': epoch
    }, tmp_path)
    if os.path.exists(tmp_path):
        os.replace(tmp_path, full_path)
        print(f"💾 Saved {filename} | Ep {epoch} Step {step}")

def load_checkpoint(model, optimizer, filename):
    full_path = get_path(filename)
    if not os.path.exists(full_path):
        print(f"No checkpoint '{filename}' found. Starting fresh.")
        return 0, 0
    print(f"Resuming from {filename}...")
    ckpt = torch.load(full_path, map_location='cpu')

    state_dict = ckpt['model_state_dict']
    if hasattr(model, '_orig_mod'):
        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith("_orig_mod."):
                new_state_dict[k] = v
        model._orig_mod.load_state_dict(new_state_dict, strict=False)
    else:
         model.load_state_dict(state_dict)

    if optimizer and ckpt['optimizer_state_dict']:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt.get('step', 0), ckpt.get('epoch', 0)

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate_creative(model, tokenizer, prompt, max_tokens, context_size, device, temp=0.8, top_k=10 ):
    model.eval()
    idx = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    eos_id = tokenizer.encode('<|endoftext|>', allowed_special={'<|endoftext|>'})[0]
    for _ in range(max_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(idx_cond)
            logits = logits[:, -1, :]
            logits[:, 50257:] = float('-inf')
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

            if temp > 0.0:
                logits = logits / temp
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            if idx_next == eos_id:
                break
            idx = torch.cat((idx, idx_next), dim=1)
    model.train()
    return token_ids_to_text(idx,tokenizer)

def create_or_load_val(tokenizer, device):
    val_file = get_path("val_set_reasoning.pt")
    if os.path.exists(val_file):
        print("Loading validation set...")
        val_data = torch.load(val_file, map_location='cpu')
        return [(x.to(device), y.to(device)) for x, y in val_data]

    print("Creating validation set (50 Edu + 50 Math)...")
    val_samples = []

    print("  - Streaming FineWeb-Edu...")
    ds_edu = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for i, item in enumerate(ds_edu):
        if i >= 50: break
        tokens = tokenizer.encode(item['text'], allowed_special={'<|endoftext|>'})[:1024]
        if len(tokens) > 1: val_samples.append((torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])))

    print("  - Streaming OpenWebMath...")
    try:
        ds_math = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
        for i, item in enumerate(ds_math):
            if i >= 50: break
            tokens = tokenizer.encode(item['text'], allowed_special={'<|endoftext|>'})[:1024]
            if len(tokens) > 1: val_samples.append((torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])))
    except Exception as e:
        print(f"Warning: Could not load OpenWebMath: {e}")

    torch.save(val_samples, val_file)
    return [(x.to(device), y.to(device)) for x, y in val_samples]

def validate(model, val_set):
  model.eval()
  total_loss = 0
  with torch.no_grad():
      with torch.amp.autocast('cuda', dtype=torch.float16):
          for x, y in val_set:
              logits = model(x.unsqueeze(0))
              loss = nn.functional.cross_entropy(logits.flatten(0,1), y.unsqueeze(0).flatten())
              total_loss += loss.item()
  model.train()
  return total_loss / max(1, len(val_set))

# ==========================================
# PHASE 1: PRE-TRAINING
# ==========================================
def run_pretraining():
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 PHASE 1: Pre-training on {device}")

    # FP32 Matmul Precision
    torch.set_float32_matmul_precision('high')

    cfg = TRAINING_CONFIG["pretraining"]
    accum_steps = cfg["grad_accum_steps"]

    tokenizer = tiktoken.get_encoding("gpt2")
    model = MicroLLM(TINY_TM_CONFIG).to(device)

    print("Compiling model...")
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    scaler = torch.amp.GradScaler('cuda')

    if os.path.exists(get_path("pretrained_model.pth")):
        print("Phase 1 Complete. Skipping.")
        return
    start_step, start_epoch = load_checkpoint(model, optimizer, "pretrain_latest.pth")

    val_set = create_or_load_val(tokenizer, device)

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["max_steps"], eta_min=cfg["lr"]/10)

    eval_prompts = [
        "One day, the sun decided not to rise. The people were ",  # Story check
        "The capital of France is",                      # Knowledge check
        "2 + 2 =",                                       # Math check
        "To solve the equation 3x + 5 = 20, we first"    # Logic check
    ]

    print("Streaming Datasets...")
    ds_edu = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).skip(100)
    ds_math = load_dataset("open-web-math/open-web-math", split="train", streaming=True).skip(100)
    mixed_ds = interleave_datasets([ds_edu, ds_math], probabilities=[0.85, 0.15])

    total_batch_size = cfg["batch_size"] * accum_steps
    skip_count = (start_epoch * cfg["max_steps"] * total_batch_size) + (start_step * total_batch_size)
    if skip_count > 0:
        print(f"Skipping {skip_count} samples...")
        mixed_ds = mixed_ds.skip(skip_count)

    class PackedDS(IterableDataset):
        def __iter__(self):
            buf = []
            for item in mixed_ds:
                tokens = tokenizer.encode(item['text'], allowed_special={'<|endoftext|>'}) + [50256]
                buf.extend(tokens)
                while len(buf) >= 1025:
                    yield torch.tensor(buf[:1024]), torch.tensor(buf[1:1025])
                    buf = buf[1024:]

    loader = DataLoader(PackedDS(), batch_size=cfg["batch_size"], pin_memory=True)

    model.train()
    optimizer.zero_grad()
    loss_history = []
    micro_step = 0
    global_step = start_step

    start_time = time.time()
    try:
        for epoch in range(start_epoch, cfg["epochs"]):
            for x, y in loader:
                if global_step >= cfg["max_steps"]: break
                model.train()
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits.flatten(0,1), y.flatten())
                    loss = loss / accum_steps

                scaler.scale(loss).backward()
                micro_step += 1

                if micro_step % accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    loss_val = loss.item() * accum_steps
                    loss_history.append(loss_val)
                    if len(loss_history) > 1000: loss_history.pop(0)

                    if global_step % 100 == 0:
                        avg = sum(loss_history)/len(loss_history)
                        val_loss = validate(model, val_set)
                        lr = scheduler.get_last_lr()[0]
                        print(f"[{time.strftime('%H:%M:%S')}] Ep {epoch} Step {global_step} | Loss: {loss_val:.4f} | Avg: {avg:.4f} | val_loss : {val_loss:.4f} |LR: {lr:.2e}")

                    if global_step % 500 == 0:
                        print(f"\n PRE-TRAIN CHECKUP (Step {global_step}):")
                        print("-" * 40)
                        for p in eval_prompts:
                            gen = generate_creative(model, tokenizer, p, 50, TINY_TM_CONFIG["context_length"], device, 0.8)
                            print(f"{p} -> {gen.strip()}")
                        print("-" * 40 + "\n")

                    if global_step % cfg["checkpoint_interval"] == 0:
                        save_checkpoint(model, optimizer, global_step, epoch, "pretrain_latest.pth")

                    global_step += 1

            global_step = 0
            save_checkpoint(model, optimizer, 0, epoch+1, f"pretrain_latest_{epoch}.pth")

    except KeyboardInterrupt:
        print("Stopped.")
        return

    save_checkpoint(model, optimizer, 0, 0, "pretrained_model.pth")
    print(f"Pre-training Complete! Total time: {(time.time() - start_time) / 3600:.2f} hours")

# ==========================================
# PHASE 2: SFT
# ==========================================
def load_sft_data_json():
    train_path = get_path("sft_train.json")
    val_path = get_path("sft_val.json")

    # A. Try loading JSON from Drive first
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"📂 Found saved JSON datasets on Drive! Loading...")
        try:
            with open(train_path, 'r', encoding='utf-8') as f: train_data = json.load(f)
            with open(val_path, 'r', encoding='utf-8') as f: val_data = json.load(f)
            print(f"✅ Loaded {len(train_data)} training samples from JSON.")
            return train_data, val_data
        except Exception as e:
            print(f"⚠️ Error reading JSON (file might be corrupt): {e}")
            print("🔄 Re-generating fresh data...")

    # B. If not found, create fresh
    print("📦 Downloading & Creating fresh datasets...")
    data = []

    # 1. OpenMath
    try:
        ds = load_dataset("nvidia/OpenMathInstruct-1", split="train", streaming=True)
        for i, item in enumerate(ds):
            if i >= 10000: break # Grab 10k
            q = item.get('question', item.get('problem', ''))
            a = item.get('generated_solution', item.get('solution', ''))
            if q and a:
                p = f"Question: {q}\nSolve step by step.\n<think>\n"
                full = f"{p}{a}\n</think>\n<|endoftext|>"
                data.append({"prompt": p, "full": full})
    except:
        print("⚠️ OpenMath failed. Skipping.")

    # 2. GSM8K (Standard)
    try:
        ds_gsm = load_dataset("gsm8k", "main", split="train")
        for item in ds_gsm:
            p = f"Question: {item['question']}\nSolve step by step.\n<think>\n"
            full = f"{p}{item['answer']}\n</think>\n<|endoftext|>"
            data.append({"prompt": p, "full": full})
    except:
        print("⚠️ GSM8K failed.")

    # 3. Emergency Synthetic
    if len(data) == 0:
        print("🚨 Generating Synthetic Data (Backup)...")
        for i in range(500):
            data.append({"prompt": "1+1", "full": "1+1=2<|endoftext|>"})

    # Split
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # SAVE AS JSON
    print(f"Saving datasets as JSON to Drive...")
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    return train_data, val_data

def test_loader(model, loader):
    model.eval()
    loss = 0
    device = next(model.parameters()).device

    with torch.no_grad():
      for vx, vy in loader:
        vx = vx.to(device, non_blocking=True)
        vy = vy.to(device, non_blocking=True)

        logits = model(vx)
        loss += nn.functional.cross_entropy(logits.flatten(0,1), vy.flatten()).item()

    model.train()
    return loss / len(loader)

def run_sft_pipeline():
    # --- SANITY CHECK ---
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n PHASE 2: SFT on {device}")

    cfg = TRAINING_CONFIG["sft"]
    accum_steps = cfg["grad_accum_steps"]

    # 1. Load Data
    tokenizer = tiktoken.get_encoding("gpt2")
    train_data, val_data = load_sft_data_json()


    print(f"Train Size: {len(train_data)} | Val Size: {len(val_data)}")

    # --- DEBUG: CHECK DATA ---
    print("\n🔍 INSPECTING DATA SAMPLE:")
    print(f"Sample Prompt: {train_data[0]['prompt'][:50]}...")
    sample_tok = tokenizer.encode(train_data[0]['full'], allowed_special={'<|endoftext|>'})
    print(f"Token Count: {len(sample_tok)}")
    if len(sample_tok) < 10:
        print("⚠️ WARNING: Tokens are suspiciously short! Check Tokenizer.")

    eval_prompts = [
        "Question: If I have 3 apples and eat 1, how many do I have?\nSolve step by step.\n<think>\n",
        "Question: What is 15 + 3 * 2?\nSolve step by step.\n<think>\n"
    ]

    # --- OPTIMIZATION: PRE-TOKENIZE NOW ---
    # This prevents the 4-minute hang during the loop.
    print("⏳ Pre-tokenizing data (this takes ~30s)...")

    def fast_tokenize(data_list):
        processed = []
        for item in data_list:
            full = tokenizer.encode(item['full'], allowed_special={'<|endoftext|>'})
            prompt = tokenizer.encode(item['prompt'], allowed_special={'<|endoftext|>'})
            processed.append({"ids": full, "p_len": len(prompt)})
        return processed

    train_data = fast_tokenize(train_data)
    val_data = fast_tokenize(val_data)
    print("✅ Pre-tokenization Complete.")

    # 2. Ultra-Lightweight Dataset Class
    class FastSFTDS(Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            # No processing, just return the pre-computed list
            return self.data[idx]

    # 3. Collate
    def collate(batch):
        # Remove empties
        batch = [x for x in batch if len(x['ids']) > 0]
        if not batch: return None, None

        max_len = min(1024, max([len(x['ids']) for x in batch]))
        seq_len = max(1, max_len - 1)

        inputs = torch.full((len(batch), seq_len), 50256, dtype=torch.long)
        targets = torch.full((len(batch), seq_len), -100, dtype=torch.long)

        for i, item in enumerate(batch):
            ids = torch.tensor(item['ids'])[:max_len]
            if len(ids) < 2: continue
            inputs[i, :len(ids)-1] = ids[:-1]
            targets[i, :len(ids)-1] = ids[1:]
            mask_end = max(0, min(item['p_len'] - 1, len(ids)-1))
            targets[i, :mask_end] = -100
        return inputs, targets

    train_loader = DataLoader(FastSFTDS(train_data), batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, pin_memory=False)
    val_loader = DataLoader(FastSFTDS(val_data), batch_size=cfg["batch_size"], collate_fn=collate)

    # 4. Model Setup
    model = MicroLLM(TINY_TM_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    scaler = torch.amp.GradScaler('cuda')

    # Force fresh start if we deleted the checkpoint
    if os.path.exists(get_path("sft_latest.pth")):
        print(f"Resuming SFT from sft_latest.pth...")
        start_step, start_epoch = load_checkpoint(model, optimizer, "sft_latest.pth")
    else:
        print(f"Starting SFT from Pretrained weights (Fresh Start)...")
        load_checkpoint(model, optimizer, "pretrained_model.pth")
        start_step, start_epoch = 0, 0

    model.train()
    optimizer.zero_grad()
    micro_step = 0
    start_time = time.time()
    global_step = 0
    print("*" * 40)
    for p in eval_prompts:
        ans = generate_creative(model, tokenizer, p, 100, TINY_TM_CONFIG["context_length"], device, 0.8)
        print("-" * 20 + "\n")
        print(f"Q: {p.strip()}]\n")
        print(f"A: {ans}\n" + "-"*20)
    print("*" * 40)
    print(f"\n🚀 Starting SFT training loop (Epochs: {cfg['epochs']})...")

    for epoch in range(start_epoch, cfg["epochs"]):
        for step, batch_data in enumerate(train_loader):

            # Handle empty batches
            if batch_data[0] is None:
              continue

            x, y = batch_data
            if step < start_step:
              continue

            model.train()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits.flatten(0,1), y.flatten())
                loss = loss / accum_steps

            scaler.scale(loss).backward()
            micro_step += 1

            # Update weights every `accum_steps`
            if micro_step % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                loss_val = loss.item() * accum_steps
                timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                global_step += 1

                if global_step % 100 == 0:
                    val_loss = test_loader(model, val_loader)
                    print(f"\n[{timestamp}] SFT Ep {epoch} | Step {global_step} |val_loss: {val_loss:.4f} | loss: {loss_val:.4f}", flush=True)

                if global_step % 300 == 0 and global_step > 0:
                    print(f"\n[{timestamp}] 🧠 SFT CHECKUP: Global Step {global_step}:\n")
                    print("*" * 40)
                    for p in eval_prompts:
                        ans = generate_creative(model, tokenizer, p, 100, TINY_TM_CONFIG["context_length"], device, 0.8)
                        print("-" * 20 + "\n")
                        print(f"Q: {p.strip()}]\n")
                        print(f"A: {ans}\n" + "-"*20)
                    print("*" * 40)
                    model.train() # Switch back to train mode

                if global_step % cfg["checkpoint_interval"] == 0 and global_step > 0:
                     save_checkpoint(model, optimizer, step, epoch, "sft_latest.pth")

        start_step = 0
        global_step = 0
        save_checkpoint(model, optimizer, 0, epoch+1, "sft_latest.pth")

    save_checkpoint(model, optimizer, 0, 0, "sft_model.pth")

    print(f"SFT Phase Complete.! Total time: {(time.time() - start_time) / 3600:.2f} hours")

# ==========================================
# PHASE 3: GRPO
# ==========================================
def load_rl_data():
    f_rl = get_path("grpo_data.json")
    if os.path.exists(f_rl):
        with open(f_rl, 'r') as f: return json.load(f)

    print("Loading GSM8K for RL...")
    gsm = load_dataset("gsm8k", "main", split="train")
    data = []
    for item in gsm:
        p = f"Question: {item['question']}\nSolve step by step.\n<think>\n"
        ans = item['answer'].split('####')[-1].strip()
        data.append({"prompt": p, "answer": ans})

    with open(f_rl, 'w') as f: json.dump(data, f)
    return data


def get_rewards(texts, answers, device):
        rewards = []
        for text, ans in zip(texts, answers):
            score = 0.0

            # --- 1. FORMAT & STRUCTURE CHECKS ---
            has_think = "<think>" in text and "</think>" in text
            think_content = ""
            if has_think:
                # Basic structure bonus
                score += 0.1
                try:
                    think_content = text.split("<think>")[1].split("</think>")[0].strip()
                except:
                    think_content = ""

            # 🛑 PENALTY 1: Answer Leakage
            # If the model puts '####' or '\boxed' INSIDE the think block, punish it.
            if "####" in think_content or "\\boxed" in think_content:
                score -= 1.0

            # 🛑 PENALTY 2: "Parrot" Mode (Repeating the Prompt)
            # Your model likes to say "Question: ..." at the start.
            if text.strip().startswith("Question:"):
                score -= 0.5

            # 🛑 PENALTY 3: Hallucinated Code
            if "<llm-code>" in text or "```python" in text:
                score -= 10.0

            # --- 2. EXTRACT PREDICTED ANSWER ---
            predicted_val = None

            # Strategy: Look for answer AFTER the </think> tag first
            answer_section = text.split("</think>")[-1] if "</think>" in text else text

            # Clean up: strict search for "####" or "\boxed" or "Answer:"
            if "####" in answer_section:
                answer_text = answer_section.split("####")[1].strip()
            elif "\\boxed" in answer_section:
                try: answer_text = answer_section.split("\\boxed{")[1].split("}")[0]
                except: answer_text = answer_section.strip()
            elif "Answer:" in answer_section:
                answer_text = answer_section.split("Answer:")[1].strip()
            else:
                answer_text = answer_section.strip()

            nums = re.findall(r"[-+]?\d*\.\d+|\d+", answer_text)
            if nums:
                try: predicted_val = float(nums[-1])
                except: pass

            # --- 3. EXTRACT EXPECTED ANSWER ---
            expected_val = None
            try:
                # Your dataloader already strips ####, but we double-check just in case
                expected_str = str(ans).split('####')[-1].strip()
                expected_clean = expected_str.replace(',', '')
                expected_nums = re.findall(r"[-+]?\d*\.\d+|\d+", expected_clean)
                if expected_nums: expected_val = float(expected_nums[-1])
            except: pass

            # --- 4. CORRECTNESS CHECK (The Big Prize) ---
            is_correct = False
            if predicted_val is not None and expected_val is not None:
                # Tolerance for float math (0.1% error allowed)
                if abs(predicted_val - expected_val) < 0.01:
                    score += 2.0
                    is_correct = True

            # --- 5. REASONING QUALITY CHECKS ---
            if has_think and think_content:
                # 🛑 PENALTY 4: Lazy Thinking
                # If thought is too short (< 100 chars), it's probably not real reasoning.
                if len(think_content) < 100:
                    score -= 0.5

                lines = [l.strip() for l in think_content.split('\n') if l.strip()]

                # 🛑 PENALTY 5: Repetition Loops
                unique_lines = set(lines)
                if len(lines) > len(unique_lines):
                    repeats = len(lines) - len(unique_lines)
                    score -= (0.5 * repeats)

                # ✅ BONUS: Valid Reasoning Steps (Gated)
                # Only reward step count if the answer is RIGHT.
                if is_correct:
                    valid_steps = 0
                    for line in lines:
                        if len(line) > 10: valid_steps += 1
                    # Cap bonus at 1.0 (requires 10 lines)
                    score += min(valid_steps * 0.1, 1.0)
                else:
                    # Tiny participation award so gradients aren't zero
                    score += 0.01

            rewards.append(score)

        return torch.tensor(rewards).to(device)

def get_logprobs(m, ids):
    with torch.amp.autocast('cuda', enabled=False):
          logits = m(ids).float()
    return torch.gather(nn.functional.log_softmax(logits, -1), -1, ids.unsqueeze(-1)).squeeze(-1)

def run_grpo():
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n PHASE 3: GRPO on {device}")

    cfg = TRAINING_CONFIG["grpo"]
    accum_steps = cfg["grad_accum_steps"]

    train_data = load_rl_data()
    tokenizer = tiktoken.get_encoding("gpt2")
    model = MicroLLM(TINY_TM_CONFIG).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    scaler = torch.amp.GradScaler('cuda')

    if os.path.exists(get_path("grpo_latest.pth")):
         start_step, start_epoch = load_checkpoint(model, None, "grpo_latest.pth")
    else:
         load_checkpoint(model, None, "sft_model.pth")
         start_epoch, start_step = 0, 0

    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters(): p.requires_grad = False
    rl_eval_prompts = [
        "Question: If it takes 10 minutes to boil 1 egg, how long does it take to boil 3 eggs?\nSolve step by step.\n<think>\n", # Parallelism
        "Question: Which is larger, 9.11 or 9.9?\nSolve step by step.\n<think>\n", # Tokenizer/Decimal bias
        "Question: Lisa has 5 apples. She buys 2 more, then drops 1. How many does she have?\nSolve step by step.\n<think>\n" # State tracking
    ]

    start_time = time.time()
    model.train()
    optimizer.zero_grad()
    micro_step = 0

    global_step = start_step
    print("Starting GRPO...")
    for epoch in range(start_epoch, cfg["epochs"]):
        for i in range(0, len(train_data), cfg["batch_size"]):
            if i < start_step: continue
            model.train()
            # 1. Prepare Batch
            batch = train_data[i : i + cfg["batch_size"]]
            if not batch: continue

            prompts, answers = zip(*batch)

            # Prepare ALL prompts at once
            input_list = []
            for p in prompts:
                p_ids = tokenizer.encode(p)
                # Repeat each prompt 'group_size' times immediately
                input_list.extend([p_ids] * cfg["group_size"])

            # Pad inputs to same length so we can stack them
            max_p_len = max(len(ids) for ids in input_list)
            curr_batch = torch.full((len(input_list), max_p_len), 50256, device=device) # Pad with EOS
            for idx, item in enumerate(input_list):
                 curr_batch[idx, :len(item)] = torch.tensor(item, device=device)

            # Generate for the WHOLE batch (Size = batch_size * group_size = 16)
            # This runs the heavy GPU loop ONCE instead of 4 times
            finished_mask = torch.zeros(curr_batch.shape[0], dtype=torch.bool, device=device)
            temperature = 1.2
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    for _ in range(200):
                        idx_cond = curr_batch[:, -1024:] # Context window
                        logits = model(idx_cond)[:, -1, :]
                        logits[:, 50257:] = float('-inf')
                        probs = torch.softmax(logits / temperature, dim=-1)
                        next_tok = torch.multinomial(probs, num_samples=1)

                        # Stop updating finished sequences (optional optimization)
                        next_tok = torch.where(finished_mask.unsqueeze(1), torch.tensor(50256, device=device), next_tok)
                        curr_batch = torch.cat((curr_batch, next_tok), 1)

                        finished_mask |= (next_tok.squeeze(-1) == 50256)
                        if finished_mask.all(): break

            # Decode all at once
            completions = []
            tokens_list = []
            for seq in curr_batch:
                decoded = tokenizer.decode(seq.tolist())
                if "<|endoftext|>" in decoded:
                    decoded = decoded.split("<|endoftext|>")[0]
                completions.append(decoded)
                tokens_list.append(seq)

            max_len = max([len(t) for t in tokens_list])
            padded = torch.full((len(tokens_list), max_len), 50256, device=device)
            mask = torch.zeros_like(padded)
            for j, t in enumerate(tokens_list):
                padded[j, :len(t)] = t
                mask[j, 20:len(t)] = 1.0

            expanded_ans = [a for a in answers for _ in range(cfg["group_size"])]
            rewards = get_rewards(completions, expanded_ans, device)
            avg_reward = rewards.mean().item()
            # Normalize Advantage
            # (reward - mean) / std
            # We reshape to [batch, group] to normalize PER PROMPT
            rewards_view = rewards.view(-1, cfg["group_size"])
            mean = rewards_view.mean(dim=1, keepdim=True)
            std = rewards_view.std(dim=1, keepdim=True)
            adv = (rewards_view - mean) / (std + 1e-8)
            adv = adv.view(-1) # Flatten back

            with torch.amp.autocast('cuda', dtype=torch.float16):
                log_probs = get_logprobs(model, padded)
                with torch.no_grad(): ref_log_probs = get_logprobs(ref_model, padded)

                ratio = torch.exp(log_probs - ref_log_probs)
                kl = log_probs - ref_log_probs
                loss = -((ratio * adv.unsqueeze(1) - cfg["beta"] * kl) * mask).sum() / mask.sum()
                loss = loss / accum_steps

            scaler.scale(loss).backward()
            micro_step += 1

            if micro_step % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                if global_step % 10 == 0:
                  print(f"\n[{timestamp}] GRPO Ep {epoch} | Step {global_step} |loss: {loss.item()*accum_steps:.4f} | reward: {avg_reward:.4f}", flush=True)

                if global_step % 50 == 0:
                    print(f"\n RL REASONING CHECK (Step {i}):")
                    print("-" * 40)
                    for p in rl_eval_prompts:
                        print(f"Q: {p.strip()}")
                        gen = generate_creative(model, tokenizer, p, 200, TINY_TM_CONFIG["context_length"], device, temp=0.8)
                        print(f"A: {gen.strip()}\n")
                    print("-" * 40)

                if global_step % 100 == 0:
                    save_checkpoint(model, optimizer, i, epoch, "grpo_latest.pth")

        start_step = 0
        global_step = 0
        save_checkpoint(model, optimizer, 0, epoch+1, "grpo_latest.pth")

    save_checkpoint(model, optimizer, 0, 0, "reasoning_model_grpo.pth")
    print(f"GRPO Phase Complete.! Total time: {(time.time() - start_time) / 3600:.2f} hours")


if __name__ == "__main__":
    run_pretraining()
    run_sft_pipeline()
    run_grpo()

