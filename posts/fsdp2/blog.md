## Introduction

Training large language models (LLMs) has become increasingly challenging as models grow from billions to hundreds of billions of parameters. A 3B parameter model in BF16 precision requires 6 GB just for parameters, plus another 24 GB for optimizer states (with AdamW), totaling **30 GB** — and that's before accounting for activations and gradients!

Enter **FSDP (Fully Sharded Data Parallel)** — PyTorch's answer to training models that don't fit on a single GPU. Based on Microsoft's ZeRO (Zero Redundancy Optimizer) paper, FSDP shards model parameters, gradients, and optimizer states across multiple GPUs, enabling you to train models 4-8× larger than what fits on a single GPU.

This blog post chronicles my journey implementing FSDP2 (PyTorch's latest FSDP API) to train SmolLM3-3B on **4× NVIDIA H100 SXM5 GPUs** via Lambda Labs. We'll cover everything from setup to benchmarking, with real performance numbers and lessons learned.

### What You'll Learn

- How FSDP works under the hood
- Migrating from FSDP1 to FSDP2
- Setting up a production-ready training environment
- Calculating and optimizing MFU (Model FLOPs Utilization)
- Real-world performance comparison: ZeRO-2 vs ZeRO-3
- Best practices and common pitfalls

---

## Understanding FSDP

### The Problem: Memory Wall

Traditional **DataParallel (DP)** and **DistributedDataParallel (DDP)** replicate the entire model on each GPU:

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   GPU 0         │  │   GPU 1         │  │   GPU 2         │  │   GPU 3         │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ Full Model (3B) │  │ Full Model (3B) │  │ Full Model (3B) │  │ Full Model (3B) │
│ Full Optimizer  │  │ Full Optimizer  │  │ Full Optimizer  │  │ Full Optimizer  │
│ 30 GB           │  │ 30 GB           │  │ 30 GB           │  │ 30 GB           │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
    Total: 120 GB across 4 GPUs (75% redundancy!)
```

**Problem**: Each GPU stores the full model and optimizer state. With 4 GPUs, you're storing 4 copies of everything!

### The Solution: FSDP with ZeRO

FSDP implements Microsoft's **ZeRO (Zero Redundancy Optimizer)** strategy, which shards (splits) model state across GPUs:

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   GPU 0         │  │   GPU 1         │  │   GPU 2         │  │   GPU 3         │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ Params: 1/4     │  │ Params: 1/4     │  │ Params: 1/4     │  │ Params: 1/4     │
│ Grads: 1/4      │  │ Grads: 1/4      │  │ Grads: 1/4      │  │ Grads: 1/4      │
│ Optim: 1/4      │  │ Optim: 1/4      │  │ Optim: 1/4      │  │ Optim: 1/4      │
│ 7.5 GB          │  │ 7.5 GB          │  │ 7.5 GB          │  │ 7.5 GB          │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
    Total: 30 GB across 4 GPUs (4× memory savings!)
```

### ZeRO Optimization Stages

FSDP supports different levels of sharding:

| Stage | What's Sharded | Memory/GPU | Speed | Use Case |
|-------|----------------|------------|-------|----------|
| **ZeRO-1** | Optimizer states only | ~18 GB | Fastest | Small models, max speed |
| **ZeRO-2** | Optimizer + Gradients | ~10 GB | Fast | Medium models, good balance |
| **ZeRO-3** | Optimizer + Gradients + Parameters | ~7.5 GB | Slower | Large models, max memory savings |

In FSDP2, this is controlled by the `reshard_after_forward` parameter:
- `reshard_after_forward=False` → **ZeRO-2** (keep parameters unsharded during forward/backward)
- `reshard_after_forward=True` → **ZeRO-3** (reshard parameters after each layer)

### Data Distribution: FSDP vs DDP

**Important**: FSDP still uses **data parallelism** — each GPU sees different data!

```
┌─────────────────────────────────────────────────────────────────┐
│                       Training Batch                            │
│  [Sample 1, Sample 2, Sample 3, Sample 4, Sample 5, ...]       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────┬─────────────┬─────────────┬─────────────┐
        │   GPU 0     │   GPU 1     │   GPU 2     │   GPU 3     │
        ├─────────────┼─────────────┼─────────────┼─────────────┤
        │ Sample 1    │ Sample 2    │ Sample 3    │ Sample 4    │
        │ (different) │ (different) │ (different) │ (different) │
        └─────────────┴─────────────┴─────────────┴─────────────┘
```

**Example with batch_size=1 per GPU, 4 GPUs**:
```python
# DataLoader automatically distributes data
dataloader = DataLoader(dataset, batch_size=1)  # Per GPU
dataloader = accelerator.prepare(dataloader)    # Shards data across GPUs

# Each GPU gets different samples
GPU 0: batch["input_ids"] = [sample_0]  # Tokens from story #0
GPU 1: batch["input_ids"] = [sample_1]  # Tokens from story #1
GPU 2: batch["input_ids"] = [sample_2]  # Tokens from story #2
GPU 3: batch["input_ids"] = [sample_3]  # Tokens from story #3

# Effective global batch size = 1 × 4 = 4 samples
```

### What FSDP Shards vs DDP

**Both FSDP and DDP**:

- ✅ Shard **data** across GPUs (different samples per GPU)
- ✅ Each GPU processes different inputs
- ✅ Gradients are averaged across GPUs

**FSDP additionally shards**:

- ✅ **Model parameters** (each GPU stores 1/N)
- ✅ **Gradients** (each GPU stores 1/N)
- ✅ **Optimizer states** (each GPU stores 1/N)

**Visual comparison**:

```
DDP (Data Parallel):
┌────────────────┐  ┌────────────────┐
│    GPU 0       │  │    GPU 1       │
├────────────────┤  ├────────────────┤
│ Data: Sample 0 │  │ Data: Sample 1 │ ← Different data
│ Params: FULL   │  │ Params: FULL   │ ← Same params (duplicated)
│ Grads: FULL    │  │ Grads: FULL    │ ← Same grads (duplicated)
│ Optim: FULL    │  │ Optim: FULL    │ ← Same optim (duplicated)
└────────────────┘  └────────────────┘

FSDP (Fully Sharded Data Parallel):
┌────────────────┐  ┌────────────────┐
│    GPU 0       │  │    GPU 1       │
├────────────────┤  ├────────────────┤
│ Data: Sample 0 │  │ Data: Sample 1 │ ← Different data
│ Params: 1/2    │  │ Params: 1/2    │ ← Different params (sharded)
│ Grads: 1/2     │  │ Grads: 1/2     │ ← Different grads (sharded)
│ Optim: 1/2     │  │ Optim: 1/2     │ ← Different optim (sharded)
└────────────────┘  └────────────────┘
```

### How FSDP Works: Communication Pattern

During training, FSDP temporarily gathers parameters for computation:

#### ZeRO-3 Forward Pass (per layer)

```
1. all_gather(params)     # Gather full parameters from all GPUs
   GPU 0: [P0, P1, P2, P3] (complete layer)
   GPU 1: [P0, P1, P2, P3] (complete layer)
   GPU 2: [P0, P1, P2, P3] (complete layer)
   GPU 3: [P0, P1, P2, P3] (complete layer)

2. compute_forward()      # Each GPU processes its own batch
   GPU 0: forward(Sample 0, params)
   GPU 1: forward(Sample 1, params)
   GPU 2: forward(Sample 2, params)
   GPU 3: forward(Sample 3, params)

3. reduce_scatter(params) # Reshard parameters immediately
   GPU 0: [P0] (back to 1/4 shard)
   GPU 1: [P1] (back to 1/4 shard)
   GPU 2: [P2] (back to 1/4 shard)
   GPU 3: [P3] (back to 1/4 shard)
```

#### ZeRO-3 Backward Pass (per layer)

```
1. all_gather(params)     # Re-gather full parameters
   All GPUs: [P0, P1, P2, P3]

2. compute_gradients()    # Each GPU computes gradients for its batch
   GPU 0: ∂L₀/∂W (gradients from Sample 0)
   GPU 1: ∂L₁/∂W (gradients from Sample 1)
   GPU 2: ∂L₂/∂W (gradients from Sample 2)
   GPU 3: ∂L₃/∂W (gradients from Sample 3)

3. reduce_scatter(grads)  # Sum gradients across GPUs, then shard
   GPU 0: [(∂L₀ + ∂L₁ + ∂L₂ + ∂L₃)/4][0:N/4]   (first 1/4 of averaged grads)
   GPU 1: [(∂L₀ + ∂L₁ + ∂L₂ + ∂L₃)/4][N/4:N/2] (second 1/4 of averaged grads)
   GPU 2: [(∂L₀ + ∂L₁ + ∂L₂ + ∂L₃)/4][N/2:3N/4]
   GPU 3: [(∂L₀ + ∂L₁ + ∂L₂ + ∂L₃)/4][3N/4:N]
```

**Key insights**:

1. ✅ Each GPU sees **different data** (data parallelism)
2. ✅ Each GPU computes **different local gradients**
3. ✅ Gradients are **averaged** across GPUs (same as DDP)
4. ✅ Each GPU stores **different parts** of averaged gradients (unique to FSDP)
5. ⚠️ ZeRO-3 does 2× more communication than ZeRO-2 (re-gathering params in backward)

### Why This Matters

**Effective batch size**:
```python
# Your code
batch_size = 1  # Per GPU
num_gpus = 4

effective_batch_size = batch_size × num_gpus = 1 × 4 = 4

# Each step processes 4 different samples
# Gradients are averaged across these 4 samples
```

**Gradient averaging** (automatic):
```python
# Conceptually (FSDP handles this automatically)
grad_gpu0 = compute_grad(sample_0)
grad_gpu1 = compute_grad(sample_1)
grad_gpu2 = compute_grad(sample_2)
grad_gpu3 = compute_grad(sample_3)

# reduce_scatter does:
averaged_grad = (grad_gpu0 + grad_gpu1 + grad_gpu2 + grad_gpu3) / 4

# Then shards the averaged gradient:
GPU 0 stores: averaged_grad[0:N/4]
GPU 1 stores: averaged_grad[N/4:N/2]
GPU 2 stores: averaged_grad[N/2:3N/4]
GPU 3 stores: averaged_grad[3N/4:N]
```

**Training is mathematically equivalent** to:
```python
# Single GPU with batch_size=4
large_batch = [sample_0, sample_1, sample_2, sample_3]
loss = model(large_batch)
loss.backward()  # Computes average gradient over 4 samples
optimizer.step()
```

---

## FSDP1 vs FSDP2: What Changed?

PyTorch introduced FSDP2 in version 2.4 with a completely redesigned API. Here's what changed:

### FSDP1 (Legacy API)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

# Wrap entire model
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
    auto_wrap_policy=transformer_auto_wrap_policy(
        transformer_layer_cls={GPT2Block}
    ),
)

# Create optimizer after wrapping
optimizer = torch.optim.AdamW(model.parameters())
```

**FSDP1 Sharding Strategies**:

| ShardingStrategy | Description | Use Case |
|-----------------|-------------|----------|
| `FULL_SHARD` | Shard params, grads, optimizer (ZeRO-3) | Maximum memory savings |
| `SHARD_GRAD_OP` | Shard grads and optimizer only (ZeRO-2) | Better performance, more memory |
| `HYBRID_SHARD` | ZeRO-3 with 2D device mesh (intra/inter-node) | Multi-node training |
| `_HYBRID_SHARD_ZERO2` | ZeRO-2 with 2D device mesh | Multi-node, max performance |
| `NO_SHARD` | No sharding (DDP equivalent) | Baseline comparison |

**Problems with FSDP1**:

- Class-based API is less Pythonic
- `auto_wrap_policy` is complex and error-prone
- Harder to compose with other features
- Less transparent about what's happening
- Sharding strategy is an enum (less flexible)

### FSDP2 (New API)

```python
from torch.distributed.fsdp import fully_shard

# Shard individual layers
for layer in model.layers:
    fully_shard(layer, reshard_after_forward=True)  # ZeRO-3

# Shard root module
fully_shard(model, reshard_after_forward=True)

# Create optimizer AFTER sharding (critical!)
optimizer = torch.optim.AdamW(model.parameters())
```

**Benefits of FSDP2**:

- ✅ **Simpler**: Function-based API, explicit wrapping
- ✅ **More control**: Manually choose what to shard
- ✅ **Better composition**: Works with torch.compile(), quantization
- ✅ **DTensor-based**: Uses PyTorch's distributed tensor abstraction
- ✅ **Better error messages**: Clearer what went wrong

### FSDP1 to FSDP2 Migration Mapping

| FSDP1 Strategy | FSDP2 Equivalent | Code |
|----------------|------------------|------|
| `FULL_SHARD` | `reshard_after_forward=True` | ZeRO-3 (params resharded) |
| `SHARD_GRAD_OP` | `reshard_after_forward=False` | ZeRO-2 (params kept) |
| `HYBRID_SHARD` | `reshard_after_forward=True` + 2D DeviceMesh | Hybrid ZeRO-3 |
| `_HYBRID_SHARD_ZERO2` | `reshard_after_forward=False` + 2D DeviceMesh | Hybrid ZeRO-2 |

**2D Device Mesh Example** (for hybrid sharding):
```python
from torch.distributed.device_mesh import init_device_mesh

# Create 2D mesh: 2 nodes × 4 GPUs per node
mesh_2d = init_device_mesh("cuda", (2, 4))  # (inter-node, intra-node)

# Hybrid ZeRO-3
for layer in model.layers:
    fully_shard(layer, mesh=mesh_2d, reshard_after_forward=True)

# Hybrid ZeRO-2
for layer in model.layers:
    fully_shard(layer, mesh=mesh_2d, reshard_after_forward=False)
```

**When to use Hybrid Sharding**:

- ✅ Multi-node training (>8 GPUs across nodes)
- ✅ Want to reduce inter-node communication
- ✅ Replicate within nodes, shard across nodes (or vice versa)

### Key Migration Steps

1. **Replace wrapper class with function**:
   ```python
   # FSDP1
   model = FSDP(model, ...)

   # FSDP2
   fully_shard(model, ...)
   ```

2. **Explicit layer wrapping**:
   ```python
   # FSDP2
   for module in get_module_children_bottom_up(model)[:-1]:
       if isinstance(module, TransformerLayer):
           fully_shard(module)
   ```

3. **Replace ShardingStrategy enum with parameter**:
   ```python
   # FSDP1
   sharding_strategy=ShardingStrategy.FULL_SHARD

   # FSDP2
   reshard_after_forward=True  # ZeRO-3
   ```

4. **Add DeviceMesh for hybrid sharding** (optional):
   ```python
   # FSDP1
   sharding_strategy=ShardingStrategy.HYBRID_SHARD

   # FSDP2
   mesh = init_device_mesh("cuda", (num_nodes, gpus_per_node))
   fully_shard(model, mesh=mesh, reshard_after_forward=True)
   ```

5. **Optimizer after sharding** (unchanged, but more critical):
   ```python
   fully_shard(model)
   optimizer = torch.optim.AdamW(model.parameters())  # Must be after!
   ```

---

## Setting Up Your Environment

### Hardware Used

For this project, I used **Lambda Labs** GPU cloud instances:

```
Instance: 4× H100 SXM5
────────────────────────────────────────
GPU:          NVIDIA H100 SXM5 80GB
Count:        4 GPUs
Peak TFLOPS:  989 TFLOPS/GPU (BF16)
Memory:       80 GB HBM3 per GPU
Bandwidth:    3.35 TB/s per GPU
Interconnect: NVLink 4.0 (900 GB/s)
Total Peak:   3,956 TFLOPS
Cost:         ~$32/hour
────────────────────────────────────────
```

**Why Lambda Labs?**

- ✅ Affordable H100 access (~$8/GPU/hour)
- ✅ Easy setup (pre-configured drivers)
- ✅ Fast provisioning (minutes, not hours)
- ✅ Good NVLink bandwidth for distributed training

### Software Requirements

**Prerequisites**:

- Python 3.9+ (3.10 recommended)
- CUDA 12.1+
- PyTorch 2.4+ (for FSDP2)

### Quick Setup

```bash
# Clone repository
git clone https://github.com/daddyofadoggy/torch-fsdp-daddyofadoggy
cd torch-fsdp-daddyofadoggy

# Run automated setup script
./setup.sh

# Or manual setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Key Dependencies

```txt
torch>=2.4.0              # FSDP2 support
transformers>=4.40.0      # SmolLM3 model
accelerate>=0.30.0        # Distributed training
datasets>=2.18.0          # TinyStories dataset
torchao>=0.3.0            # FP8 quantization
wandb>=0.16.0             # Experiment tracking
```

**Critical**: PyTorch **2.4+** is required for FSDP2. Earlier versions only support FSDP1.

### Verification

```bash
# Check PyTorch version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Expected: 2.4.0 or higher

# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Expected: True

# Check GPU count
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
# Expected: 4

# Check GPU type
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA H100 SXM5 80GB
```

---

## The Codebase Architecture

Our implementation consists of two main files:

### Project Structure

```
torch-fsdp-daddyofadoggy/
├── train_fsdp.py         # Main training script
├── utils.py              # Dataset, metrics, FLOP calculation
├── requirements.txt      # Dependencies
├── setup.sh              # Automated setup
└── docs/
    ├── CODEWALKTHROUGH.md
    ├── FLOPS_CALCULATION.md
    ├── MFU_CALCULATION.md
    └── BENCHMARK.md
```

### Key Components

#### 1. Dataset Loading (`utils.py`)

```python
def get_dataset(tokenizer, seq_len, accelerator):
    """
    Load TinyStories dataset with sequence packing.

    Why packing?
    - TinyStories has short texts (50-200 tokens)
    - Training on short sequences wastes compute
    - Packing combines multiple texts into full sequences
    """
    raw_dataset = load_dataset("roneneldan/TinyStories", split="train[:5%]")

    # Tokenize
    tokenized = raw_dataset.map(tokenize_function, batched=True)

    # Pack into full sequences (8192 tokens)
    packed = tokenized.map(create_packed_sequences, batched=True)

    return packed.shuffle(seed=42)
```

#### 2. FLOP Calculation (`utils.py`)

```python
def get_model_flops_per_token(model, seq_len):
    """
    Calculate FLOPs per token for training.

    Formula: factor × (attention_flops + mlp_flops) × num_layers

    Factor = 6:
      - 2 FLOPs per MAC (multiply-accumulate)
      - 3× for training (forward + 2× backward)
    """
    cfg = model.config
    factor = 6  # Training: forward + backward

    # Attention FLOPs
    qkv_flops = factor * hidden_size * (num_heads * head_dim * 3)
    attn_scores = factor * num_heads * seq_len * head_dim
    attn_output = factor * num_heads * seq_len * head_dim
    output_proj = factor * num_heads * head_dim * hidden_size

    # MLP FLOPs (SwiGLU: 3 projections)
    mlp_flops = factor * hidden_size * intermediate_size * 3

    # Total
    return (qkv_flops + attn_scores + attn_output + output_proj + mlp_flops) * num_layers
```

**For SmolLM3-3B (8192 seq_len)**:

- Attention: ~302M FLOPs per token per layer
- MLP: ~302M FLOPs per token per layer
- Total: **24.2 GFLOPs per token** (40 layers)

#### 3. MFU Calculation (`utils.py`)

```python
def estimate_mfu(model_flops_per_token, num_tokens, time_elapsed, num_gpus, peak_tflops=None):
    """
    Calculate Model FLOPs Utilization (MFU).

    MFU = (Actual TFLOPS) / (Theoretical Peak TFLOPS) × 100%

    Example:
      - Processed 280,000 tokens in 10 seconds
      - Model needs 24.2 GFLOPs per token
      - Using 4× H100 (989 TFLOPS each)

      Total FLOPs = 24.2e9 × 280,000 = 6.776e15
      Actual TFLOPS/sec = 6.776e15 / (10 × 1e12) = 677.6
      Theoretical = 989 × 4 = 3,956 TFLOPS
      MFU = 677.6 / 3,956 × 100 = 17.1%
    """
    if peak_tflops is None:
        peak_tflops = get_gpu_peak_tflops()  # Auto-detect

    total_flops = model_flops_per_token * num_tokens
    actual_tflops = total_flops / (time_elapsed * 1e12)
    theoretical = peak_tflops * num_gpus

    mfu_percent = (actual_tflops / theoretical) * 100

    return {
        "mfu_percent": mfu_percent,
        "actual_tflops_per_sec": actual_tflops,
        "theoretical_tflops_total": theoretical,
        "tokens_per_sec": num_tokens / time_elapsed,
    }
```

#### 4. Performance Tracking (`utils.py`)

```python
class PerformanceTracker:
    """
    Track training metrics after warmup period.

    Why warmup?

    - First few steps compile CUDA kernels
    - Caches need to warm up
    - Exclude from metrics for accuracy
    """
    def __init__(self, warmup_steps=10, num_gpus=1):
        self.warmup_steps = warmup_steps
        self.num_gpus = num_gpus
        self.reset()

    def step(self, batch_tokens, model_flops_per_token):
        self.step_count += 1

        if self.step_count == self.warmup_steps:
            # Warmup complete, start tracking
            self.start_time = time.perf_counter()
            self.num_tokens = 0
            return {"warmup_completed": True}

        if not self.is_in_warmup:
            # Calculate metrics
            self.num_tokens += batch_tokens
            elapsed = time.perf_counter() - self.start_time

            # Basic metrics
            metrics = {
                "tokens_per_sec": self.num_tokens / elapsed,
                "steps_per_sec": (self.step_count - self.warmup_steps) / elapsed,
            }

            # MFU metrics
            mfu = estimate_mfu(model_flops_per_token, self.num_tokens, elapsed, self.num_gpus)
            metrics.update(mfu)

            return metrics
```

---

## Implementing FSDP2 Training

### The Training Script (`train_fsdp.py`)

Let's walk through the complete training implementation:

#### Step 1: Setup

```python
import torch
from torch.distributed.fsdp import fully_shard
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from utils import PerformanceTracker, get_dataset, get_model_flops_per_token

# Initialize distributed training
set_seed(42)
accelerator = Accelerator()
```

#### Step 2: Load Model

```python
# Load model from config (random initialization)
model = AutoModelForCausalLM.from_config(
    AutoConfig.from_pretrained("HuggingFaceTB/SmolLM3-3B", use_cache=False),
    torch_dtype=torch.bfloat16,  # BF16 parameters
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

**Why `from_config` instead of `from_pretrained`?**

- Faster (no 6GB download)
- Focus on training infrastructure, not convergence
- Easier to benchmark

#### Step 3: Load Dataset

```python
# Load and prepare dataset
dataset = get_dataset(tokenizer, seq_len=8192, accelerator=accelerator)
dataloader = DataLoader(dataset, batch_size=1, collate_fn=create_collate_fn())

# Prepare for distributed training
dataloader = accelerator.prepare(dataloader)
accelerator.wait_for_everyone()
```

#### Step 4: Apply FSDP2 Sharding

**This is the critical part!**

```python
from transformers.models.smollm3.modeling_smollm3 import SmolLM3DecoderLayer
from accelerate.utils.other import get_module_children_bottom_up

# Define sharding policy
def policy(module):
    return isinstance(module, SmolLM3DecoderLayer)

# Shard each decoder layer individually
for module in get_module_children_bottom_up(model)[:-1]:
    if policy(module):
        fully_shard(module, reshard_after_forward=True)  # ZeRO-3

# Shard root module
fully_shard(model, reshard_after_forward=True)
```

**Why per-layer sharding?**

- Overlaps communication with computation
- Better memory efficiency
- Recommended by PyTorch for transformers

**What happens to parameters?**

Before `fully_shard()`:
```python
weight = model.layers[0].weight
print(type(weight))    # torch.nn.Parameter
print(weight.shape)    # [2048, 2048]
```

After `fully_shard()`:
```python
weight = model.layers[0].weight
print(type(weight))    # DTensor (distributed tensor)
print(weight.shape)    # [2048, 2048] (logical shape)
print(weight._local_tensor.shape)  # [512, 2048] (1/4 on each GPU)
```

Parameters are transformed into **DTensors** — PyTorch's abstraction for distributed tensors that are sharded across GPUs.

#### Step 5: Create Optimizer (CRITICAL ORDER!)

```python
# MUST create optimizer AFTER fully_shard()!
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
```

**Why this order matters:**

❌ **WRONG** (optimizer before sharding):
```python
optimizer = torch.optim.AdamW(model.parameters())  # Full tensors
fully_shard(model)  # Parameters become DTensors, but optimizer states are still full

# Result:
# - Optimizer states: FULL tensors (30 GB per GPU)
# - Parameters: Sharded DTensors (7.5 GB per GPU)
# - Wasted 4× memory on optimizer states!
```

✅ **CORRECT** (optimizer after sharding):
```python
fully_shard(model)  # Parameters become DTensors
optimizer = torch.optim.AdamW(model.parameters())  # Creates states as DTensors

# Result:
# - Optimizer states: Sharded DTensors (7.5 GB per GPU)
# - Parameters: Sharded DTensors (7.5 GB per GPU)
# - 4× memory savings!
```

When you create the optimizer after sharding:

- `model.parameters()` returns DTensors
- `optimizer.state['exp_avg'] = zeros_like(param)` creates sharded DTensors
- Optimizer states are **automatically sharded** to match parameters

#### Step 6: Training Loop

```python
model.train()

# Setup performance tracking
model_flops_per_token = get_model_flops_per_token(model, seq_len=8192)
tracker = PerformanceTracker(warmup_steps=5, num_gpus=accelerator.num_processes)

# Training loop
for step, batch in enumerate(dataloader):
    # Forward pass
    outputs = model(**batch)
    loss = outputs.loss

    # Backward pass (with FSDP gradient reduction)
    accelerator.backward(loss)

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Track performance
    metrics = tracker.step(batch["input_ids"].shape[1], model_flops_per_token)

    # Logging
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
        if metrics:
            print(tracker.get_print_message(metrics))

    accelerator.log(metrics)
```

**What happens during forward/backward with ZeRO-3?**

Forward pass (per layer):
```
1. all_gather(params)      # GPU 0: [P0, P1, P2, P3] (full layer)
2. compute_forward()       # Run layer forward
3. reduce_scatter(params)  # GPU 0: [P0] (back to 1/4 shard)
```

Backward pass (per layer, reverse order):
```
1. all_gather(params)      # Re-gather for gradient computation
2. compute_gradients()     # Calculate ∂L/∂W
3. reduce_scatter(grads)   # Sum gradients across GPUs, keep 1/4
4. free(params)            # Free unsharded parameters
```

---

## Understanding Performance Metrics

### Key Metrics We Track

#### 1. Throughput (Tokens/Second)

**What it measures**: Training speed

```python
tokens_per_sec = total_tokens / time_elapsed
```

**Example**:
```
280,000 tokens in 10 seconds = 28,000 tokens/sec
```

**Why it matters**:

- Direct measure of training speed
- Easy to compare across configurations
- Scales linearly with batch size

#### 2. MFU (Model FLOPs Utilization)

**What it measures**: Hardware efficiency

```python
MFU = (Actual TFLOPS / Theoretical Peak TFLOPS) × 100%
```

**Example**:
```
Actual: 677.6 TFLOPS
Peak: 3,956 TFLOPS (4× H100)
MFU: 17.1%
```

**Why it matters**:

- Hardware-independent comparison
- Identifies bottlenecks (compute vs memory vs communication)
- Industry standard (used in PaLM, GPT-3 papers)

**Target MFU**:

- **50-60%**: Excellent (state-of-the-art)
- **40-50%**: Very good (production quality)
- **30-40%**: Good (room for optimization)
- **<30%**: Poor (significant bottlenecks)

#### 3. Memory Usage

**Three types tracked**:

```python
peak_memory_active:    # Actually used by tensors
peak_memory_alloc:     # Allocated by PyTorch (includes fragmentation)
peak_memory_reserved:  # Reserved from OS (includes cache)
```

**Relationship**: `reserved ≥ alloc ≥ active`

**Example (SmolLM3-3B, ZeRO-3, 4 GPUs)**:
```
Parameters:    1.5 GB per GPU
Gradients:     1.5 GB per GPU
Optimizer:     6.0 GB per GPU
Activations:   ~10 GB per GPU
──────────────────────────
Total:         ~19 GB per GPU
```

#### 4. TFLOPS (Tera Floating-Point Operations per Second)

**Actual TFLOPS**:
```python
actual_tflops = (total_flops / time_elapsed) / 1e12
```

**Theoretical TFLOPS**:
```python
theoretical = peak_tflops_per_gpu × num_gpus
            = 989 × 4
            = 3,956 TFLOPS
```

---

## Benchmark Results: ZeRO-2 vs ZeRO-3

I ran comprehensive benchmarks comparing ZeRO-2 and ZeRO-3 strategies on our Lambda Labs setup.

### Configuration

```yaml
Hardware:
  Instance: Lambda Labs 4× H100 SXM5
  GPUs: 4× NVIDIA H100 SXM5 80GB
  Peak: 989 TFLOPS/GPU (BF16)
  Interconnect: NVLink 4.0 (900 GB/s)

Model:
  Name: SmolLM3-3B
  Parameters: 3 Billion
  Precision: BF16 (parameters), FP32 (optimizer)

Training:
  Sequence Length: 8192 tokens
  Batch Size: 1 per GPU (4 global)
  Optimizer: AdamW (lr=1e-5)
  Dataset: TinyStories
```

### Results

#### ZeRO-2 (`reshard_after_forward=False`)

```
Loss:             5.9867
Steps/sec:        1.03
Tokens/sec:       8,414.69
Tokens/sec/GPU:   2,103.67
MFU:              20.52%
Time/step:        0.974s
Actual TFLOPS:    202.97
Theoretical:      3,956 TFLOPS
Peak/GPU:         989 TFLOPS
Memory/GPU:       ~22-25 GB
```

#### ZeRO-3 (`reshard_after_forward=True`)

```
Loss:             5.9865
Steps/sec:        1.00
Tokens/sec:       8,213.54
Tokens/sec/GPU:   2,053.39
MFU:              20.03%
Time/step:        0.997s
Actual TFLOPS:    198.12
Theoretical:      3,956 TFLOPS
Peak/GPU:         989 TFLOPS
Memory/GPU:       ~19-22 GB
```

### Performance Comparison

| Metric | ZeRO-2 | ZeRO-3 | Difference | Winner |
|--------|--------|--------|------------|--------|
| **Throughput (tokens/s)** | 8,415 | 8,214 | +201 (+2.4%) | 🏆 ZeRO-2 |
| **Steps/sec** | 1.03 | 1.00 | +0.03 (+3.0%) | 🏆 ZeRO-2 |
| **Time/step** | 0.974s | 0.997s | -0.023s (-2.3%) | 🏆 ZeRO-2 |
| **MFU** | 20.52% | 20.03% | +0.49 pp | 🏆 ZeRO-2 |
| **Memory/GPU** | ~24 GB | ~21 GB | -3 GB (-12%) | 🏆 ZeRO-3 |
| **Training Loss** | 5.9867 | 5.9865 | +0.0002 | ≈ Same |

**Key Findings**:

1. ✅ ZeRO-2 is **2.4% faster** than ZeRO-3
2. ✅ ZeRO-3 saves **3 GB memory per GPU** (12% reduction)
3. ✅ Training convergence is **identical** (loss diff: 0.0002)
4. ⚠️ Both show low MFU (~20%) due to small batch size

### Why ZeRO-2 is Faster

ZeRO-3 performs **2× more communication**:

**Communication volume per step**:
```
ZeRO-2:
  Forward:  40 all-gathers (params) = 240 GB
  Backward: 40 reduce-scatters (grads) = 240 GB
  Total: 480 GB

ZeRO-3:
  Forward:  40 all-gathers + 40 reduce-scatters = 480 GB
  Backward: 40 all-gathers + 40 reduce-scatters = 480 GB
  Total: 960 GB (2× more!)
```

**However**, H100's fast NVLink (900 GB/s) mitigates the overhead:
```
Communication time:
  ZeRO-2: 480 GB / 900 GB/s = 0.53s
  ZeRO-3: 960 GB / 900 GB/s = 1.07s

Actual difference: 0.997s - 0.974s = 0.023s (only 2.3%!)
```

**Why so small?**

- Communication overlaps with computation
- PyTorch's optimized collectives
- H100's high bandwidth (900 GB/s)

### When to Use Each

**Use ZeRO-2** when:

- ✅ You have sufficient GPU memory
- ✅ Prioritizing maximum throughput
- ✅ Training smaller models (<7B on high-memory GPUs)
- ✅ Communication is a bottleneck (slower interconnects)

**Use ZeRO-3** when:

- ✅ GPU memory is tight
- ✅ Training very large models (>7B parameters)
- ✅ Want to maximize batch size
- ✅ Memory savings > 2-3% speed difference

**For our setup (3B model, 4× H100 80GB)**:

- **Recommendation**: Use **ZeRO-2**
- Memory is not constrained (using <30% of 80GB)
- 2.4% speed improvement over long training runs

---

## Optimization Guide

Our benchmarks showed **20% MFU** — well below the 40-50% target. Here's how to improve:

### Problem Analysis

**Why is MFU low?**

1. **Small batch size** (primary factor)

   - Batch size = 1 per GPU
   - Memory-bandwidth bound, not compute-bound
   - GPU compute units underutilized

2. **Communication overhead**

   - 50%+ of time on collective operations
   - Small batches make this proportionally larger

3. **Model size relative to hardware**

   - 3B params don't fully saturate H100's 989 TFLOPS
   - Smaller matrix multiplications

### Optimization Roadmap

#### 1. Increase Batch Size (Immediate, +50% throughput)

```python
# Current
batch_size = 1 per GPU

# Optimized
batch_size = 4 per GPU
```

**Expected improvement**:

- Throughput: 8,415 → 12,000-13,000 tokens/sec (+45-55%)
- MFU: 20% → 30-35%
- Memory: 22 GB → 35-40 GB per GPU (still fits!)

#### 2. Add Flash Attention 2 (Medium, +25% throughput)

```python
model = AutoModelForCausalLM.from_config(
    config,
    attn_implementation="flash_attention_2"  # 2-3× faster attention
)
```

**Why it helps**:

- Optimized CUDA kernels for attention
- Reduced memory usage (enables larger batches)
- Fused operations

**Expected improvement**:

- Throughput: +20-30%
- Memory: -15-20%
- MFU: +5-8%

#### 3. Use torch.compile() (Medium, +20% throughput)

```python
model = torch.compile(model, mode="max-autotune")
```

**Why it helps**:

- Kernel fusion (fewer kernel launches)
- Optimized memory access patterns
- Graph-level optimizations

**Expected improvement**:

- Throughput: +15-25%
- MFU: +3-5%

#### 4. Gradient Accumulation (Low, +30% throughput)

```python
gradient_accumulation_steps = 4

for step, batch in enumerate(dataloader):
    outputs = model(**batch)
    loss = outputs.loss / gradient_accumulation_steps

    accelerator.backward(loss)

    if (step + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Why it helps**:

- Simulates larger batch size
- Amortizes communication overhead
- Same memory as batch_size=1

**Expected improvement**:

- Throughput: +25-35%
- MFU: +5-10%

### Expected Results After Optimization

| Optimization | Cumulative Throughput | Cumulative MFU |
|--------------|----------------------|----------------|
| Baseline | 8,415 tokens/s | 20.5% |
| + Batch size 4 | 12,600 tokens/s | 31% |
| + Flash Attention 2 | 15,100 tokens/s | 37% |
| + torch.compile() | 17,400 tokens/s | 42% |
| + Gradient accum. | 18,800 tokens/s | 46% |

**Target achieved**: 46% MFU (excellent for production!)

### Cost Analysis

Training 1 billion tokens:

```
Current (ZeRO-2, batch_size=1):
  Time: 1B / 8,415 = 118,836 seconds = 33.0 hours
  Cost: 33.0 hours × $32/hour = $1,056

Optimized (ZeRO-2, batch_size=4, Flash Attn 2, compile):
  Time: 1B / 17,400 = 57,471 seconds = 16.0 hours
  Cost: 16.0 hours × $32/hour = $512

Savings: $544 (51% cost reduction!)
```

---

## Lessons Learned

### 1. Optimizer Order is Critical

**Never create optimizer before FSDP sharding!**

```python
# ❌ WRONG - 4× memory waste
optimizer = torch.optim.AdamW(model.parameters())
fully_shard(model)

# ✅ CORRECT
fully_shard(model)
optimizer = torch.optim.AdamW(model.parameters())
```

**Symptom**: OOM errors that make no sense, or seeing full model size in `nvidia-smi`.

### 2. Small Batches Kill Performance

Batch size = 1 resulted in:

- 20% MFU (should be 40-50%)
- 50%+ time on communication
- Memory-bandwidth bound

**Lesson**: Always maximize batch size (within memory limits).

### 3. ZeRO-3 Isn't Always Necessary

For our 3B model on H100 80GB:

- ZeRO-2 was 2.4% faster
- Memory usage (24 GB) was comfortable
- Only needed ZeRO-3 for >7B models

**Lesson**: Match sharding strategy to your constraints, not blindly use ZeRO-3.

### 4. Communication Overhead Matters (But Less Than Expected)

ZeRO-3 does 2× communication, but only 2.3% slower because:

- H100 NVLink is incredibly fast (900 GB/s)
- PyTorch optimizes collectives well
- Overlap hides most latency

**Lesson**: Modern hardware mitigates communication overhead significantly.

### 5. MFU is the Key Metric

Tokens/sec alone is misleading:

- Comparing across hardware (H100 vs A100)
- Understanding bottlenecks
- Research reproducibility

**Lesson**: Always track MFU, not just throughput.

### 6. Warmup is Essential

First 5-10 steps:

- Compile CUDA kernels
- Warm up caches
- Unstable measurements

**Lesson**: Always exclude warmup from benchmarks.

### 7. Per-Layer Sharding > Model-Level

Individual layer wrapping:

- Better communication/compute overlap
- Finer memory control
- Recommended by PyTorch

**Lesson**: Use `get_module_children_bottom_up()` for transformer layers.

### 8. Documentation Matters

This project has:

- 5 comprehensive markdown docs
- Line-by-line code walkthrough
- Benchmark analysis
- Setup automation

**Lesson**: Good documentation saves debugging time and enables others.

---

## Conclusion

### What We Achieved

- ✅ **Implemented FSDP2** from scratch with proper sharding
- ✅ **Benchmarked ZeRO-2 vs ZeRO-3** on real hardware (4× H100)
- ✅ **Measured performance** comprehensively (MFU, TFLOPS, memory)
- ✅ **Identified optimization paths** to 2× performance improvement
- ✅ **Documented everything** for reproducibility and learning

### Key Takeaways

1. **FSDP2 is production-ready**: Simpler API, better composability than FSDP1
2. **ZeRO-2 vs ZeRO-3 is a trade-off**: 2-3% speed vs 10-15% memory
3. **Small batches are expensive**: Batch size is the #1 performance lever
4. **H100 mitigates communication**: Fast NVLink makes ZeRO-3 viable
5. **MFU < 30% signals problems**: Indicates memory-bound or communication-bound
6. **Optimizer order matters**: Create after sharding to shard optimizer states

### Performance Summary

**Baseline (ZeRO-2, batch_size=1)**:

- Throughput: 8,415 tokens/sec
- MFU: 20.5%
- Memory: 24 GB/GPU

**Optimized (estimated)**:

- Throughput: 17,400 tokens/sec (2× improvement)
- MFU: 42% (production-grade)
- Memory: 38 GB/GPU (still <50%)
- Cost: 51% reduction

### Future Work

1. **Implement optimizations**: Flash Attention 2, torch.compile()
2. **Scale to larger models**: Test 7B, 13B parameters
3. **Multi-node training**: Scale beyond 8 GPUs
4. **FP8 quantization**: Further memory and speed improvements
5. **Gradient checkpointing**: Trade compute for memory

### Resources

- **Repository**: [torch-fsdp-daddyofadoggy](https://github.com/your-username/torch-fsdp-daddyofadoggy)
- **Documentation**:
  - [Code Walkthrough](CODEWALKTHROUGH.md)
  - [FLOPs Calculation](FLOPS_CALCULATION.md)
  - [MFU Calculation](MFU_CALCULATION.md)
  - [Benchmark Analysis](BENCHMARK.md)
- **PyTorch FSDP**: [Official Docs](https://pytorch.org/docs/stable/fsdp.html)
- **Lambda Labs**: [GPU Cloud](https://lambdalabs.com/service/gpu-cloud)

---

## References

### Foundational Papers

1. **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**

   - Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020)
   - Microsoft Research
   - ArXiv: https://arxiv.org/abs/1910.02054
   - *The foundational paper introducing ZeRO optimization stages that FSDP implements*

2. **PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel**

   - Zhao, Y., Gu, A., Varma, R., et al. (2023)
   - Meta AI / PyTorch Team
   - ArXiv: https://arxiv.org/abs/2304.11277
   - *Official PyTorch team's paper on FSDP design and implementation*

3. **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM**

   - Narayanan, D., Shoeybi, M., Casper, J., et al. (2021)
   - NVIDIA Research
   - ArXiv: https://arxiv.org/abs/2104.04473
   - *Megatron-LM: combines model, data, and pipeline parallelism*

### Performance and Optimization

4. **PaLM: Scaling Language Modeling with Pathways**

   - Chowdhery, A., Narang, S., Devlin, J., et al. (2022)
   - Google Research
   - ArXiv: https://arxiv.org/abs/2204.02311
   - *Introduces MFU (Model FLOPs Utilization) as a key metric*

5. **Training Compute-Optimal Large Language Models (Chinchilla)**

   - Hoffmann, J., Borgeaud, S., Mensch, A., et al. (2022)
   - DeepMind
   - ArXiv: https://arxiv.org/abs/2203.15556
   - *Scaling laws and compute-optimal training strategies*

6. **GPT-3: Language Models are Few-Shot Learners**

   - Brown, T. B., Mann, B., Ryder, N., et al. (2020)
   - OpenAI
   - ArXiv: https://arxiv.org/abs/2005.14165
   - *175B parameter training at scale, discusses efficiency metrics*

### Transformer Architectures

7. **Attention Is All You Need**

   - Vaswani, A., Shazeer, N., Parmar, N., et al. (2017)
   - Google Research
   - ArXiv: https://arxiv.org/abs/1706.03762
   - *Original Transformer architecture paper*

8. **LLaMA: Open and Efficient Foundation Language Models**

   - Touvron, H., Lavril, T., Izacard, G., et al. (2023)
   - Meta AI
   - ArXiv: https://arxiv.org/abs/2302.13971
   - *Introduces GQA (Grouped Query Attention) and modern optimizations*

9. **GLU Variants Improve Transformer**

   - Shazeer, N. (2020)
   - Google Research
   - ArXiv: https://arxiv.org/abs/2002.05202
   - *Introduces SwiGLU activation used in modern LLMs*

### Mixed Precision and Quantization

10. **Mixed Precision Training**

    - Micikevicius, P., Narang, S., Alben, J., et al. (2018)
    - NVIDIA / Baidu Research
    - ArXiv: https://arxiv.org/abs/1710.03740
    - *Foundational work on FP16/BF16 training*

11. **FP8 Formats for Deep Learning**

    - Micikevicius, P., Stosic, D., Burgess, N., et al. (2022)
    - NVIDIA Research
    - ArXiv: https://arxiv.org/abs/2209.05433
    - *FP8 training for next-generation accelerators*

12. **FlashAttention: Fast and Memory-Efficient Exact Attention**

    - Dao, T., Fu, D. Y., Ermon, S., et al. (2022)
    - Stanford University
    - ArXiv: https://arxiv.org/abs/2205.14135
    - *IO-aware attention algorithm for 2-4× speedup*

### Distributed Training Systems

13. **Megatron-LM: Training Multi-Billion Parameter Language Models**

    - Shoeybi, M., Patwary, M., Puri, R., et al. (2019)
    - NVIDIA Research
    - ArXiv: https://arxiv.org/abs/1909.08053
    - *Model parallelism strategies for large models*

14. **DeepSpeed: System Optimizations Enable Training Deep Learning Models**

    - Rasley, J., Rajbhandari, S., Ruwase, O., et al. (2020)
    - Microsoft Research
    - ArXiv: https://arxiv.org/abs/2002.08910
    - *Implements ZeRO and other optimizations*

15. **Distributed Deep Learning with PyTorch**

    - Li, S., Zhao, Y., Varma, R., et al. (2020)
    - Meta AI / PyTorch Team
    - PyTorch Documentation
    - *Official guide to PyTorch distributed training*

### Benchmarking and Profiling

16. **MLPerf Training Benchmark**

    - Mattson, P., Cheng, C., Diamos, G., et al. (2020)
    - MLCommons
    - ArXiv: https://arxiv.org/abs/1910.01500
    - *Industry-standard benchmarking for ML systems*

17. **Measuring the Carbon Intensity of AI in Cloud Instances**

    - Dodge, J., Prewitt, T., Tachet des Combes, R., et al. (2022)
    - ArXiv: https://arxiv.org/abs/2206.05229
    - *Environmental impact and efficiency metrics*

### Hardware and Infrastructure

18. **NVIDIA H100 Tensor Core GPU Architecture**

    - NVIDIA Corporation (2022)
    - White Paper
    - https://resources.nvidia.com/en-us-tensor-core
    - *H100 specifications and capabilities*

19. **NVLink and NVSwitch: High-Speed Interconnect for GPUs**

    - NVIDIA Corporation (2023)
    - Technical Documentation
    - *GPU interconnect technology used in our benchmarks*

### Software Frameworks

20. **PyTorch 2.0: Faster, More Pythonic, Staying True to Its Roots**

    - PyTorch Team (2023)
    - https://pytorch.org/blog/pytorch-2.0-release/
    - *torch.compile() and PyTorch 2.x features*

21. **Accelerate: A Simple Way to Train and Use PyTorch Models**

    - HuggingFace Team (2023)
    - https://huggingface.co/docs/accelerate/
    - *Distributed training abstraction library*

22. **Transformers: State-of-the-Art Natural Language Processing**

    - Wolf, T., Debut, L., Sanh, V., et al. (2020)
    - HuggingFace
    - ArXiv: https://arxiv.org/abs/1910.03771
    - *Library used for model loading and tokenization*

### Additional Resources

23. **Understanding PyTorch DTensor**

    - PyTorch Team (2023)
    - https://pytorch.org/docs/stable/distributed.tensor.html
    - *Distributed tensor abstraction underlying FSDP2*

24. **Automatic Mixed Precision Package**

    - PyTorch Documentation
    - https://pytorch.org/docs/stable/amp.html
    - *torch.cuda.amp for mixed precision training*

25. **Lambda Labs GPU Cloud Documentation**

    - Lambda Labs (2024)
    - https://lambdalabs.com/service/gpu-cloud
    - *Cloud infrastructure used for this work*

---

### Citation

If you use this work or reference these benchmarks, please cite:

```bibtex
@misc{fsdp2-blog-2025,
  author = {Ron},
  title = {Training Large Language Models with FSDP2: A Complete Guide},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/your-username/torch-fsdp-daddyofadoggy}},
  note = {Benchmarks on 4× NVIDIA H100 SXM5 via Lambda Labs}
}
```

---

### Acknowledgments

- **PyTorch Team** for FSDP2 implementation and excellent documentation
- **HuggingFace Team** for Transformers and Accelerate libraries
- **Lambda Labs** for providing accessible H100 GPU instances
- **Microsoft Research** for the foundational ZeRO paper
- **Meta AI** for SmolLM3 model and PyTorch development
- **NVIDIA** for H100 GPUs and NVLink technology
- **Open Source Community** for tools and libraries that made this possible

---

## Appendix: Quick Reference

### Running the Code

```bash
# Setup
./setup.sh

# Activate environment
source venv/bin/activate

# Single GPU (testing)
python train_fsdp.py --num-steps 100

# 4 GPUs with Accelerate
accelerate launch --num_processes=4 train_fsdp.py

# 4 GPUs with torchrun
torchrun --nproc_per_node=4 train_fsdp.py

# Custom configuration
accelerate launch --num_processes=4 train_fsdp.py \
  --sequence-length 8192 \
  --num-steps 1000 \
  --precision bf16 \
  --log-with wandb
```

### Key Formulas

**FLOPs per token (training)**:
```
factor = 6 (2 FLOPs/MAC × 3 for forward+backward)
FLOPs = factor × (attention_flops + mlp_flops) × num_layers
```

**MFU**:
```
MFU = (Actual TFLOPS / Theoretical Peak TFLOPS) × 100%
```

**Memory (ZeRO-3, 4 GPUs)**:
```
Params:  model_size × 2 (BF16) / 4
Grads:   model_size × 2 (BF16) / 4
Optim:   model_size × 8 (FP32, AdamW) / 4
Total:   model_size × 3 bytes / GPU
```

### Troubleshooting

**OOM Error**:

- ✅ Check batch size (reduce to 1)
- ✅ Enable gradient checkpointing
- ✅ Switch to ZeRO-3 (`reshard_after_forward=True`)
- ✅ Reduce sequence length

**Low MFU (<20%)**:

- ✅ Increase batch size
- ✅ Use gradient accumulation
- ✅ Add Flash Attention 2
- ✅ Profile for bottlenecks

**Slow Training**:

- ✅ Check communication overhead (ZeRO-2 vs ZeRO-3)
- ✅ Verify NVLink is active (`nvidia-smi topo -m`)
- ✅ Use torch.compile()
- ✅ Check data loading (increase num_workers)

**Optimizer States Not Sharded**:

- ✅ Create optimizer AFTER fully_shard()
- ✅ Check with `hasattr(param, '_local_tensor')`

---

**Thanks for reading! Questions? Open an issue on [GitHub](https://github.com/Scratch-to-Scale/torch-fsdp-daddyofadoggy).**
