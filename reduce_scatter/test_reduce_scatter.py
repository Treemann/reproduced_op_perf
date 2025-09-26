import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
input_shape = (32768, 1, 4096)
output_shape = (32768//world_size, 1, 4096)
dtype = torch.bfloat16

input_tensor = torch.randn(input_shape, dtype=dtype).to(device)
output_tensor = torch.zeros(output_shape, dtype=dtype).to(device)
print(f"{input_tensor.shape=}")
print(f"{output_tensor.shape=}")

for i in range(3):
    print(f"Warm up: {i} ...")
    dist.reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.SUM)

count = 1
if torch.cuda.is_available():
    torch.cuda.synchronize()

if torch.cuda.is_available():
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
else:
    start_time = time.time()

for i in range(count):
    dist.reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.SUM)

if torch.cuda.is_available():
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
else:
    elapsed_time_ms = (time.time() - start_time) * 1000

avg_iter_time_ms = elapsed_time_ms / count

print(f"Rank {rank}: avg iter time = {avg_iter_time_ms:.6f} ms, total time = {elapsed_time_ms:.6f} ms")

