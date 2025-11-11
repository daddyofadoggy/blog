# Distributed Data Parallelism (DDP) Explained for Beginners

## What is Distributed Data Parallelism?

Distributed Data Parallelism (DDP) is a technique for training large machine learning models faster by using multiple GPUs or computers at the same time. Instead of training on one GPU, you train on multiple GPUs simultaneously, with each GPU working on different parts of your data.

Think of it like this: if you have 1000 images to process and 4 GPUs, each GPU processes 250 images. This makes training roughly 4x faster!

---

## Code Walkthrough

### 1. Imports and Setup (Lines 1-14)

```python
import torch
import torch.distributed as dist
from accelerate import PartialState
```

**What's happening:**
- `torch`: The main PyTorch library for deep learning
- `torch.distributed (dist)`: PyTorch's library for distributed training across multiple devices
- `PartialState`: A helper from the Accelerate library that manages which GPU each process should use

```python
state = PartialState()
device = state.device
set_seed(42)
```

**What's happening:**
- `PartialState()` automatically figures out which GPU this process should use
- `device` stores the assigned GPU
- `set_seed(42)` ensures reproducibility - all processes start with the same random state

---

### 2. The SimpleDistributedDataParallelism Class (Lines 16-42)

This is the heart of the code! It shows how DDP works under the hood.

#### Initialization (__init__, Lines 17-27)

```python
def __init__(self, model:torch.nn.Module):
    self.model = model

    for param in model.parameters():
        rank0_param = param.data.clone()
        dist.broadcast(rank0_param, src=0)
        if not torch.equal(param.data, rank0_param):
            raise ValueError(...)
```

**What's happening:**

1. **Takes a model as input** and stores it
2. **Broadcasts parameters from GPU 0 to all other GPUs**:
   - In distributed training, each GPU (called a "rank") has its own copy of the model
   - "Broadcasting" means copying data from one GPU to all others
   - This ensures all GPUs start with identical model weights
3. **Verification check**: If any GPU has different parameters, raise an error

**Why this matters:** All GPUs must start with the exact same model, or they'll learn different things!

#### Gradient Synchronization (Lines 29-33)

```python
def sync_gradients(self):
    for param in self.model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= dist.get_world_size()
```

**What's happening:**

This is THE KEY operation in DDP! Let me explain with an example:

Imagine you have 2 GPUs:
- GPU 0 processes batch A and calculates gradient = [1, 2, 3]
- GPU 1 processes batch B and calculates gradient = [4, 5, 6]

After `all_reduce` with SUM operation:
- Both GPUs now have gradient = [5, 7, 9] (sum of both)

After dividing by world_size (2):
- Both GPUs have gradient = [2.5, 3.5, 4.5] (average)

**Why averaging?** This is equivalent to processing both batches on a single GPU! The gradient is the average across all data processed by all GPUs.

#### Helper Methods (Lines 35-42)

```python
def __call__(self, *args, **kwargs):
    return self.model(*args, **kwargs)

def train(self):
    self.model.train()

def eval(self):
    self.model.eval()
```

**What's happening:**
These methods allow our wrapper to behave like a regular PyTorch model.

---

### 3. Data Preparation (Lines 44-68)

```python
dataset = get_dataset()["train"]
train_ds = dataset.shuffle(seed=42)
```

**What's happening:**
- Load the training dataset
- Shuffle it with a fixed seed so all GPUs shuffle the same way

```python
def collate_func(batch):
    return tokenizer.pad(
        batch,
        padding="longest",
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
```

**What's happening:**
- This function prepares batches of text data
- It pads sequences to the same length (needed for batch processing)
- Pads to multiples of 8 (optimization for GPU efficiency)

---

### 4. Data Sharding - The Critical Part! (Lines 87-98)

```python
# Shard data for first parallel dimension
ds_length = len(train_ds)
ds_length_per_rank = ds_length // get("ws")  # ws = world_size
rank = get("rank")
start = rank * ds_length_per_rank
end = start + ds_length_per_rank if rank != get("ws") - 1 else ds_length

train_shard = train_ds.select(list(range(start, end)))
```

**What's happening:**

This splits the dataset into separate chunks for each GPU!

**Example with 1000 samples and 4 GPUs:**
- Total samples: 1000
- Samples per GPU: 1000 / 4 = 250
- GPU 0 (rank 0): samples 0-249
- GPU 1 (rank 1): samples 250-499
- GPU 2 (rank 2): samples 500-749
- GPU 3 (rank 3): samples 750-999

**Why this matters:** Each GPU only loads and processes its own portion of data, so they work on different examples simultaneously!

---

### 5. Model Setup (Lines 109-112)

```python
model = get_smol_model()
model.to(device)
optimizer = torch.optim.SGD(model.model.parameters(), lr=1e-3)
model = SimpleDistributedDataParallelism(model)
```

**What's happening:**
1. Create the model
2. Move it to the assigned GPU
3. Create an optimizer (SGD with learning rate 0.001)
4. Wrap the model with our DDP wrapper

---

### 6. Profiler Setup (Lines 114-136)

```python
if state.is_main_process:
    profiler_context = profile(...)
```

**What's happening:**
- Only the main process (GPU 0) runs the profiler
- The profiler tracks performance metrics like memory usage and computation time
- Results are saved to "ddp_trace" folder for analysis with TensorBoard

**Why only main process?** To avoid multiple GPUs writing the same profiling data and causing conflicts.

#### Understanding the Profiler Schedule

```python
profiler_schedule = schedule(
    skip_first=5,
    wait=1,
    warmup=2,
    active=5,
    repeat=1
)
```

**What's happening:**

The profiler schedule controls WHEN the profiler collects data. It doesn't run on every iteration because profiling adds overhead and generates large trace files. The schedule has four phases that cycle through iterations:

1. **skip_first=5**: Skip the first 5 iterations completely (no profiling)
   - Why? The first few iterations are often slower due to initialization and GPU warm-up
   - Skipping them gives more accurate performance measurements

2. **wait=1**: Wait for 1 iteration without profiling
   - This is a "rest" phase between profiling cycles
   - Allows the system to stabilize before starting to profile again

3. **warmup=2**: Run for 2 iterations collecting basic profiling data
   - This is a "warm-up" phase where the profiler starts but doesn't record everything yet
   - Helps the profiler itself initialize properly

4. **active=5**: Actively profile for 5 iterations with full data collection
   - This is when the profiler records detailed performance data
   - Captures CPU usage, GPU usage, memory allocations, and operation timing

5. **repeat=1**: Repeat the cycle (wait → warmup → active) 1 time
   - After the first cycle completes, it runs one more cycle
   - Total cycles = initial + repeat = 2 cycles

**Timeline example for 20 iterations:**

```
Iterations 0-4:   SKIP (skip_first=5)
Iteration 5:      WAIT (wait=1)
Iterations 6-7:   WARMUP (warmup=2)
Iterations 8-12:  ACTIVE - recording data! (active=5)
Iteration 13:     WAIT (wait=1)
Iterations 14-15: WARMUP (warmup=2)
Iterations 16-20: ACTIVE - recording data! (active=5)
```

#### Understanding the Profile Configuration

```python
profiler_context = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=profiler_schedule,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("ddp_trace"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True
)
```

**What each parameter means:**

- **activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]**
  - Track both CPU and GPU (CUDA) operations
  - Shows where time is spent on both devices

- **schedule=profiler_schedule**
  - Use the schedule defined above to control when profiling happens

- **on_trace_ready=torch.profiler.tensorboard_trace_handler("ddp_trace")**
  - When profiling data is ready, save it to the "ddp_trace" folder
  - Can be visualized with TensorBoard using: `tensorboard --logdir=ddp_trace`

- **record_shapes=True**
  - Record the shapes of tensors (e.g., [batch_size, sequence_length, hidden_size])
  - Helps identify operations working on large tensors that might be slow

- **profile_memory=True**
  - Track memory allocations and deallocations
  - Shows which operations use the most GPU memory
  - Helps identify memory bottlenecks or leaks

- **with_stack=True**
  - Record the Python call stack for each operation
  - Shows which line of code triggered each operation
  - Makes it easier to find performance bottlenecks in your code

- **with_flops=True**
  - Estimate floating-point operations (FLOPs) for each operation
  - Helps understand computational intensity
  - Higher FLOPs = more computation work

**Why this matters:** Profiling helps you understand where your training time is spent. You can identify if you're bottlenecked by data loading, forward pass, backward pass, or gradient synchronization!

#### Visualizing Profiler Data with TensorBoard (Remote Setup)

Since your code is running on **Lambda Labs** (remote GPU server) and you're accessing it from your **MacBook via VSCode**, here's how to visualize the profiler traces:

**Step 1: Run the DDP Training Script on Lambda Labs**

First, execute your training script on the Lambda Labs instance. This will generate the profiler trace files:

```bash
# On Lambda Labs (via VSCode terminal)
python ddp.py
```

After the script completes, you should see a `ddp_trace` folder created with trace files inside.

**Step 2: Install TensorBoard (if not already installed)**

On your Lambda Labs instance:

```bash
pip install tensorboard
```

**Step 3: Launch TensorBoard on Lambda Labs**

Start TensorBoard on the remote server:

```bash
tensorboard --logdir=ddp_trace --port=6006
```

This will output something like:
```
TensorBoard 2.x.x at http://localhost:6006/
```

**Important:** Keep this terminal running! Don't close it.

**Step 4: Port Forwarding via VSCode (Easy Method)**

VSCode makes port forwarding super easy!

**Option A: Automatic Port Forwarding (Recommended)**

1. VSCode should automatically detect that port 6006 is being used
2. Look for a notification in the bottom-right corner saying "Port 6006 is available"
3. Click "Open in Browser" or "Forward Port"

**Option B: Manual Port Forwarding**

1. In VSCode, press `Cmd+Shift+P` (on Mac) to open the Command Palette
2. Type "Forward a Port" and select it
3. Enter port number: `6006`
4. Press Enter

You should see the forwarded port appear in the "PORTS" panel at the bottom of VSCode.

**Step 5: Open TensorBoard in Your MacBook Browser**

Once the port is forwarded, open your web browser on your MacBook and go to:

```
http://localhost:6006
```

You should see the TensorBoard interface!

**Step 6: Navigate to the Profiler Tab**

In TensorBoard:
1. Click on the **"PYTORCH_PROFILER"** or **"PROFILE"** tab at the top
2. You'll see a dropdown to select which trace file to view
3. Select the trace file you want to analyze

**What You'll See in TensorBoard:**

The profiler visualization shows several views:

1. **Overview Page:**
   - Performance summary
   - GPU utilization over time
   - Step time breakdown (how long each training iteration took)

2. **Operator View:**
   - Shows which PyTorch operations took the most time
   - See operations like `matmul`, `conv2d`, `all_reduce`, etc.
   - Sorted by execution time

3. **Kernel View:**
   - Low-level GPU kernel performance
   - Shows actual CUDA kernels that ran on the GPU

4. **Trace View:**
   - Timeline visualization
   - Shows when each operation executed
   - You can zoom in to see individual operations
   - **Look for the `sync_grads` section** - this shows the time spent on gradient synchronization!

5. **Memory View:**
   - Memory allocation over time
   - Helps identify memory leaks or spikes

**Tips for Analysis:**

- **Look for the "sync_grads" operations** in the trace view - this is your DDP gradient synchronization time
- **Compare "forward", "backward", and "sync_grads" times** - ideally, sync time should be small compared to computation
- **Check GPU utilization** - you want this close to 100% during training
- **Identify bottlenecks** - if data loading takes longer than forward/backward, you need faster data loading

**Alternative: Using SSH Tunnel (Manual Method)**

If VSCode port forwarding doesn't work, you can use SSH tunneling:

```bash
# On your MacBook terminal (not VSCode)
ssh -L 6006:localhost:6006 username@lambda-labs-ip-address
```

Then access `http://localhost:6006` in your browser.

**Troubleshooting:**

- **Port already in use?** Change the port: `tensorboard --logdir=ddp_trace --port=6007`
- **Can't see traces?** Make sure the `ddp_trace` folder exists and contains `.pt.trace.json` files
- **Port forwarding not working?** Try restarting VSCode or manually set up SSH tunnel
- **No data in TensorBoard?** The profiler only collects data during "active" iterations (8-12 and 16-20 in this code)

---

### 7. The Training Loop (Lines 138-161)

This is where everything comes together!

```python
for (i, batch) in enumerate(train_dataloader):
    if i > 20:
        break
```

**What's happening:** Loop through batches, stopping after 20 iterations (for demonstration).

#### Step 1: Move Data to GPU (Lines 143-144)

```python
with record_function("data_movement"):
    batch = {k: v.to(device) for k, v in batch.items()}
```

**What's happening:** Transfer the batch from CPU memory to GPU memory.

#### Step 2: Forward Pass (Lines 146-147)

```python
with record_function("forward"):
    output = model(**batch)
```

**What's happening:**
- Run the model on the input data
- Each GPU processes its own batch independently
- Calculate predictions and loss

#### Step 3: Backward Pass (Lines 148-149)

```python
with record_function("backward"):
    output.loss.backward()
```

**What's happening:**
- Calculate gradients using backpropagation
- Each GPU calculates gradients based on its own batch
- At this point, gradients are still different on each GPU!

#### Step 4: Synchronize Gradients (Lines 151-152)

```python
with record_function("sync_grads"):
    model.sync_gradients()
```

**What's happening:**
- **THIS IS THE MAGIC!**
- All GPUs communicate and average their gradients
- After this step, all GPUs have identical gradients
- This makes it as if we processed all batches on a single GPU

#### Step 5: Update Model (Lines 154-158)

```python
with record_function("opt_step"):
    optimizer.step()
    optimizer.zero_grad()
```

**What's happening:**
1. Update model parameters using the averaged gradients
2. Reset gradients to zero for the next iteration
3. Since all GPUs have the same gradients, they all update identically
4. Models stay synchronized!

---

### 8. Cleanup (Lines 160-163)

```python
if profiler_context:
    profiler_context.__exit__(None, None, None)

dist.destroy_process_group()
```

**What's happening:**
- Close the profiler
- Destroy the process group (disconnect GPUs from each other)

---

## The Big Picture: How DDP Works

### The DDP Workflow

1. **Initialization:** All GPUs start with identical model copies
2. **Data Sharding:** Each GPU gets a different subset of the training data
3. **Independent Forward/Backward:** Each GPU processes its own data independently
4. **Gradient Synchronization:** GPUs communicate and average their gradients
5. **Synchronized Update:** All GPUs update their models identically
6. **Repeat:** Back to step 3 for the next batch

### Why DDP is Powerful

**Speed:** With N GPUs, you process N times more data per iteration!

**Example:**
- Single GPU: Process 8 samples per iteration
- 4 GPUs with DDP: Process 32 samples per iteration (8 per GPU)
- This is like having a batch size of 32, but the memory usage per GPU is only for batch size 8!

**Equivalence to Single GPU:**
DDP is mathematically equivalent to training on a single GPU with a larger batch size, because:
- You process more samples total (N times more)
- Gradients are averaged across all samples
- Model updates are based on the averaged gradient

### Key Concepts Recap

- **Rank:** The ID of each GPU (0, 1, 2, ...)
- **World Size:** Total number of GPUs
- **Broadcast:** Copy data from one GPU to all others
- **All-Reduce:** Combine data from all GPUs (sum, average, etc.)
- **Data Sharding:** Split dataset so each GPU gets different samples
- **Gradient Synchronization:** Average gradients across all GPUs

---

## Summary

This code demonstrates a simplified version of PyTorch's Distributed Data Parallelism. The key insight is:

> Each GPU works on different data independently, but they synchronize their gradients after backpropagation, ensuring all GPUs learn the same model together.

By splitting the work across multiple GPUs, you can train models much faster without changing the final result!
