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
# print(f"{input_tensor.shape=}")
# print(f"{output_tensor.shape=}")

wait_steps = 5
prof_steps = 10

profiler = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=wait_steps,
        warmup=0,
        active=prof_steps,
        repeat=1
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile_reduce_scatter"),
    record_shapes=True, 
    profile_memory=True,
    with_stack=True 
)

with profiler:
    for i in range(wait_steps + prof_steps):
        dist.reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.SUM)
        profiler.step()
print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=-1))
