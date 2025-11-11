
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

### Auto-imported Variables

The environment automatically provides:
- `rank` - The ID of the current process (0 or 1)
- `world_size` - Total number of processes (2 in this case)
- `gpu_id` - The specific GPU assigned to this process
- `device` - The PyTorch device object for this GPU

### The `get()` Utility

We have introduced a handy utility function `get()` for accessing distributed information:

```python
get("ws")      # → world_size (number of GPUs)
get("rank")    # → current process rank
get("grank")   # → global rank
get("lrank")   # → local rank
```

To understand `get`, we need to dig into `cache_mesh` Class - A Function Decorator with State.

```python 
class cache_mesh:
    def __init__(self, func):
        self.func = func        # Store the decorated function
        self._mesh = None       # Initialize mesh cache as None

    def __call__(self, str, dm: dist.device_mesh.DeviceMesh = None):
        mesh = self._mesh if dm is None else dm     # If no device mesh (dm) is provided, it uses the cached mesh (self._mesh)
        return self.func(str, mesh)                 # It calls the original function with the string argument and the determined mesh

    def register_mesh(self, mesh: dist.device_mesh.DeviceMesh):
        self._mesh = mesh
        return self

```
Now we are going to declare the `get` function is decorated with @cache_mesh, transforming it into an instance of the cache_mesh class. This allows it to use a cached device mesh when none is provided.

```python
@cache_mesh
def get(str, dm: dist.device_mesh.DeviceMesh = None):
    """
    Applies a func to get whatever is requested.

    `ws` -> dist.get_world_size(pg)
    `pg` -> dist.get_process_group()
    `rank` -> dist.get_rank(pg) # global
    `grank` -> dist.get_rank(pg) # global
    `lrank` -> local_rank
    """

    pg = dm.get_group() if dm else None

    match str:
        case "ws":
            return dist.get_world_size(pg)
        case "pg":
            return pg
        case "rank" | "grank":
            return dist.get_rank(pg)
        case "lrank":
            return dm.get_local_rank() if dm else int(os.environ.get("LOCAL_RANK", 0))
        case _:
            raise ValueError(f"Invalid string: {str}")

```
Here is an example of how to use it in practice 

```python
# In setup code, register a mesh once
device_mesh = dist.DeviceMesh("cuda", [[0, 1, 2, 3]])  # Create a mesh with 4 GPUs
get.register_mesh(device_mesh)

# Later, easily access distributed info without passing the mesh each time
world_size = get("ws")       # Uses cached mesh
my_rank = get("rank")        # Uses cached mesh
local_rank = get("lrank")    # Uses cached mesh

# Or override with a specific mesh when needed
specific_mesh = dist.DeviceMesh("cuda", [[0, 1]])
other_world_size = get("ws", specific_mesh)  # Uses specific mesh
```
Alternatively, we can use `nbdistributed` [plugin] (https://muellerzr.github.io/scratch-to-scale/01_intro_to_jupyter.html ) and then 

```python
%load_ext nbdistributed
```

```python
%dist_init --num-processes 2 --gpu-ids 3,4
```
This creates:
- **Rank 0** → Worker on GPU 1
- **Rank 1** → Worker on GPU 2

Each "rank" is essentially a separate process handling one GPU.

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

Here is how a simple DDP class looks like 

```python
class SimpleDistributedDataParallelism:
    def __init__(self, model:torch.nn.Module):
        self.model = model

        for param in model.parameters():
            rank0_param = param.data.clone()
            dist.broadcast(rank0_param, src=0)
            if not torch.equal(param.data, rank0_param):
                raise ValueError(
                    "Expected model parameters to be identical during `__init__`, but this is not true. "
                    "Make sure to set the seeds before creating your model"
                )

    def sync_gradients(self):
        """
        Should be called before the backward pass, iterates 
        through all params, and:
        1. Check if it is `None` (not trainable)
        2. If trainable, will perform an `all_reduce` using `SUM`
        (aka: take the global average of all grads)
        """
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= dist.get_world_size()
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
```

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

## Dataset Characteristics
For dataset, we used GLUE MRPC Dataset. Here is a brief description of the dataset

- **Task Type:** Sentence pair classification (paraphrase identification)
- **Description:** The MRPC dataset contains pairs of sentences automatically extracted from online news sources with human annotations indicating whether they are semantically equivalent (paraphrases) or not
- **Size:**
    - Training set: 3,668 sentence pairs
    - Validation set: 408 sentence pairs
    - Test set: 1,725 sentence pairs
- **Labels:** Binary classification
    - 0: not_equivalent
    - 1: equivalent

Here's an example from the dataset:
```python
{
  'idx': 0,
  'label': 1,
  'sentence1': 'Amrozi accused his brother, whom he called "the witness", of deliberately distorting his evidence.',
  'sentence2': 'Referring to him as only "the witness", Amrozi accused his brother of deliberately distorting his evidence.'
}
```
For model, we used a small 360M `HuggingFaceTB/SmolLM2-360M-Instruct` model.


## Profiling 
We used `torch.profiler` to check traces, kernel and memory footprint . We ran the distributed module using a 4 H100 SXM5 GPU instatance in `LambdaLab`.  

<div style="text-align: center;">

![Profiier Execution Summary](assets/profiler_overview_1.png)

</div>

<div style="text-align: center;">

![Profiier Operator Summary](assets/profiler_operator.png)

</div>

<div style="text-align: center;">

![Profiier Kernel Summary](assets/profiler_kernel.png)

</div>


<div style="text-align: center;">

![Profiier Traces ](assets/profiler_trace.png)

</div>

### Key Observations

- **Very low GPU utilization (15.19%)** - This is extremely low for H100 GPUs, indicating significant inefficiency
- **SM Efficiency (11.08%)** - This suggests your kernels aren't fully utilizing the streaming multiprocessors
- **Occupancy (28.73%)** - The low occupancy indicates your kernels aren't keeping the GPU busy
- **CPU Execution dominates (61.1%)** of the step time
- **Kernel execution (15.2%)** is relatively small
- **Communication overhead (20.9%)** is significant but expected in DDP

### Bottlenecks and Solutions

- The AllReduce operation (42.2%) dominates kernel time, which is expected in DDP but appears to be taking too much relative time
    - Solution: Gradient Accumulation
- Unused Tensorcore as we see it in the Profiler
    - Solution: Mixed Precision Training to enable tensorcore 
- We can try to Increase batch size until memory limits to increase throughput

We have experimented with Gradient Accumulation with step size 2 and AllReduce operation  reduced to 25% in Kermel profiler.

<div style="text-align: center;">

![Profiier Kernel Summary after Gradient Accumulation](assets/profiler_kernel_grad_ac.png)

</div>

