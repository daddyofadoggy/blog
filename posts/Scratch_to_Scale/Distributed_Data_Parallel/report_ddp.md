# Understanding Distributed Data Parallelism (DDP): A Beginner's Guide

## Introduction

Welcome! In this guide, we'll explore Distributed Data Parallelism (DDP), a powerful technique for training deep learning models faster by using multiple GPUs. If you've ever trained a large model and wished it could go faster, DDP is one of the best tools to achieve that speedup.

Don't worry if you're new to distributed training - we'll break everything down step by step, starting from the core concepts and building up to a working implementation of Pytorch's `DistributedDataParallel`


## What is Distributed Data Parallelism?

Before diving into the code, let's understand the fundamental concept.

### The Core Idea

Imagine you're a teacher grading 100 homework assignments. You could:

- **Option A**: Grade all 100 assignments yourself (slow!)
- **Option B**: Split the assignments among 4 teaching assistants, each grades 25 assignments (4x faster!)

In the First Phase, Distributed Data Parallel begins with the entire batch of data being divided into equal partitions across devices. Each partition is processed independently by identical model replicas running on separate GPUs, with each performing its own forward pass computation. Following the forward pass, each model calculates its own loss value based solely on its data partition, which then initiates the backward pass where gradients are computed independently on each device.

After local gradient computation, DDP executes its most critical operation—the all-reduce synchronization—where gradients from all devices are averaged, ensuring each model receives the same update signal as if it had processed the entire batch. With synchronized gradients in hand, each model's optimizer applies identical parameter updates, maintaining perfect weight consistency across all replicas. This coordinated update completes one training iteration, and the process repeats with new data partitions in the next step, preserving model equivalence throughout training. To illustrate the DDP process I have attached a diagram below. I borrowed it Zach Mueller's Scratch to Scale cohort and one of the best diagrams I've ever seen on DDP

<div style="text-align: center;">

![DDP Architecture Diagram (ref: Scratch to Scale)](assets/ddp_diagram.png)

</div>

Distributed Data Parallel delivers remarkable efficiency through its balanced approach to parallelism, offering near-linear scaling with increasing GPU count while maintaining mathematical equivalence to single-GPU training. The communication overhead is minimized by exchanging only gradients rather than activations or weights, utilizing highly optimized all-reduce operations that leverage ring-based algorithms. DDP's elegant simplicity makes it the preferred parallelization strategy for most deep learning tasks, providing substantial speedups without the complexity of model parallelism approaches.

**Key Points:**

- We **DON'T** split the model across GPUs (the model stays whole)
- We **DO** split the training data across GPUs
- Each GPU has a complete copy of the model
- Each GPU processes a different subset of data
- At the end of each step, we average the gradients from all GPUs


### The Math Behind It

Given `n` GPUs, here's what happens:

```
B_i = B/n     → Each GPU gets a mini-batch of size B/n
g = (1/n) Σ g_i   → Gradients from all GPUs are averaged
θ_i = θ_i - g     → Each GPU updates its model using the averaged gradient
```

In plain English:
1. Split your batch of data across all GPUs
2. Each GPU computes gradients on its portion
3. Average all the gradients together
4. Update the model parameters on each GPU

The beauty of DDP is that it only requires **one communication step** - the gradient averaging. This makes it very efficient!



## Setting Up the Environment

The notebook uses a custom distributed environment with 2 GPUs (GPU 1 and GPU 2). When initialized, you get:

```python
%dist_init --num-processes 2 --gpu-ids 1,2
```

This creates:
- **Rank 0** → Worker on GPU 1
- **Rank 1** → Worker on GPU 2

Each "rank" is essentially a separate process handling one GPU.

### Auto-imported Variables

The environment automatically provides:
- `rank` - The ID of the current process (0 or 1)
- `world_size` - Total number of processes (2 in this case)
- `gpu_id` - The specific GPU assigned to this process
- `device` - The PyTorch device object for this GPU

### The `get()` Utility

The notebook introduces a handy utility function `get()` for accessing distributed information:

```python
get("ws")      # → world_size (number of GPUs)
get("rank")    # → current process rank
get("grank")   # → global rank
get("lrank")   # → local rank
```

## Building DDP from Scratch

Now comes the exciting part - implementing DDP ourselves to understand how it works!

### Step 1: The Constructor - Ensuring Model Synchronization

The first challenge: we need to ensure all GPUs start with the **exact same model**. If they don't, the training will diverge and produce incorrect results.

```python
class SimpleDistributedDataParallelism:
    def __init__(self, model: torch.nn.Module):
        self.model = model

        # Verify all GPUs have identical model parameters
        for param in model.parameters():
            rank0_param = param.data.clone()
            dist.broadcast(rank0_param, src=0)  # Broadcast from rank 0
            if not torch.equal(param.data, rank0_param):
                raise ValueError(
                    "Expected model parameters to be identical during `__init__`, "
                    "but this is not true. Make sure to set the seeds before creating your model"
                )
```

**What's happening here?**

1. For each parameter in the model:
   - Rank 0 broadcasts its parameter value to all other ranks
   - Each rank compares its local parameter to rank 0's parameter
   - If there's any mismatch, we raise an error

2. Why do we need this?
   - Random initialization could give different starting weights on each GPU
   - Solution: Set the same random seed on all GPUs before creating the model

**Testing the verification:**

The notebook demonstrates this by intentionally setting different seeds:

```python
%%rank [0]
set_seed(43)  # Rank 0 uses seed 43

# Rank 1 still uses default seed
# Result: ValueError when trying to create DDP model!
```

After fixing by setting the same seed on all ranks:

```python
set_seed(43)  # Same seed on all ranks
model = SimpleDistributedDataParallelism(model)  # Success!
```

### Step 2: Adding Forward Pass Methods

We need to make our wrapper behave like a normal PyTorch model:

```python
def __call__(self, *args, **kwargs):
    return self.model(*args, **kwargs)

def train(self):
    self.model.train()

def eval(self):
    self.model.eval()
```

These methods simply delegate to the wrapped model, making our DDP class transparent to use.

### Step 3: The Heart of DDP - Gradient Synchronization

This is where the magic happens! After computing gradients on each GPU's subset of data, we need to average them across all GPUs.

**The Problem Without Synchronization:**

The notebook shows what happens if we train without syncing:

```python
# Each GPU processes different data
item = dataset[get("rank")]  # Rank 0 gets item 0, Rank 1 gets item 1

# Train without syncing
output = model(**item)
output.loss.backward()
optimizer.step()

# Check if parameters match across GPUs
# Result: ValueError - parameters are different!
# Max difference: 0.00390625
```

The GPUs diverged because they updated their models differently!

**The Solution - `sync_gradients()` Method:**

```python
def sync_gradients(self):
    """
    Should be called after the backward pass.
    Averages gradients across all GPUs.
    """
    for param in self.model.parameters():
        if param.grad is not None:
            # Sum gradients from all GPUs
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            # Divide by number of GPUs to get average
            param.grad /= dist.get_world_size()
```

**How it works:**

1. `dist.all_reduce()` - A collective communication operation that:
   - Gathers the gradient tensor from all GPUs
   - Applies an operation (SUM in our case)
   - Returns the result to all GPUs

2. We divide by `world_size` to convert the sum into an average

3. After this, all GPUs have the **same averaged gradient** and will update identically

**The Corrected Training Loop:**

```python
output = model(**item)
output.loss.backward()
ddp_model.sync_gradients()  # Critical step!
optimizer.step()
optimizer.zero_grad()

# Verify parameters match across GPUs
# Result: Success - all parameters identical!
```

## Putting It All Together: Performance Comparison

Now let's see the speedup in action!

### Single GPU Baseline

Training on a single GPU (Rank 0 only):
```python
per_device_batch_size = 16  # Batch size of 16

# Results:
# Total training time: 1.58 seconds
# Average time per batch: 0.0751 seconds
```

### DDP with 2 GPUs

Now let's distribute the training:

1. **Data Sharding**: Split the dataset across GPUs
   ```python
   ds_length_per_rank = len(dataset) // world_size
   rank = get("rank")
   start = rank * ds_length_per_rank
   end = start + ds_length_per_rank
   train_shard = dataset.select(range(start, end))
   ```

2. **Smaller per-device batch size**: Since we're using 2 GPUs
   ```python
   per_device_batch_size = 8  # 8 per GPU = 16 total (same as single GPU)
   ```

3. **Results:**
   ```
   Rank 0: 1.13 seconds, 0.0540 seconds/batch
   Rank 1: 1.16 seconds, 0.0551 seconds/batch
   ```

### Key Insight

With 2 GPUs, we can train with an **effective global batch size of 16** (8 per GPU) in approximately **the same time** it took to train with batch size 8 on a single GPU!

This means:
- We effectively **doubled our throughput**
- The communication overhead (gradient averaging) is minimal
- We could even increase to a global batch size of 32 (16 per GPU) for even faster training

## Advanced Feature: Gradient Accumulation

Sometimes you want to train with a very large batch size, but it won't fit in GPU memory. The solution is **gradient accumulation** - accumulate gradients over multiple micro-batches before updating.

### The Challenge with DDP

With gradient accumulation, we don't want to sync gradients after every micro-batch - that would be wasteful! We only want to sync when we're ready to actually update the model.

### The Solution: Conditional Syncing

```python
class SimpleDistributedDataParallelism:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.enable_grad_sync()  # Start with syncing enabled
        # ... (initialization code)

    def sync_gradients(self):
        if not self.do_sync:
            return  # Skip syncing if disabled
        # ... (sync code)

    def enable_grad_sync(self):
        self._do_sync = True

    def disable_grad_sync(self):
        self._do_sync = False

    @contextmanager
    def no_sync(self):
        """Context manager to temporarily disable gradient syncing."""
        prev = self.do_sync
        self.disable_grad_sync()
        try:
            yield
        finally:
            self._do_sync = prev
```

### Using Gradient Accumulation

```python
grad_accum_steps = 4  # Accumulate over 4 micro-batches

for i, batch in enumerate(dataloader):
    # Only sync on the last accumulation step
    if i % grad_accum_steps == 0:
        ddp_model.enable_grad_sync()
    else:
        ddp_model.disable_grad_sync()

    output = ddp_model(batch)
    output.loss.backward()

    # Only update when syncing is enabled
    if ddp_model.do_sync:
        ddp_model.sync_gradients()
        optimizer.step()
        optimizer.zero_grad()
```

This way, we only communicate gradients every 4 steps instead of every step, reducing communication overhead!

## Summary: Key Takeaways

1. **DDP splits data, not the model**: Each GPU has a full copy of the model and processes different data

2. **One critical sync step**: After computing gradients, we average them across all GPUs using `all_reduce`

3. **Initialization matters**: All GPUs must start with identical model parameters (use the same seed!)

4. **Communication is cheap**: The gradient averaging is fast relative to the forward/backward pass

5. **Near-linear speedup**: With 2 GPUs, you can roughly double your throughput

6. **Gradient accumulation**: Can be combined with DDP by selectively disabling gradient syncing

## When to Use DDP

DDP is ideal when:
- Your model fits on a single GPU
- You want to train with larger batch sizes
- You want faster training
- You have multiple GPUs available
- Communication between GPUs is fast (same machine or fast interconnect)

## Next Steps

Now that you understand the basics of DDP:
- Experiment with different numbers of GPUs
- Try different batch sizes
- Measure the speedup on your own models
- Explore PyTorch's built-in `torch.nn.parallel.DistributedDataParallel` (which builds on these concepts with optimizations)

Happy distributed training!
