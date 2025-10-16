import torch
from torch.profiler import profile, record_function, ProfilerActivity
from flash_attn import flash_attn_varlen_func
from aiter import flash_attn_varlen_func as flash_attn_varlen_func_aiter

device = torch.device("cuda:0")
q = torch.randn((9749, 28, 128), dtype = torch.bfloat16, device = device, requires_grad = True)
k = torch.randn((9749, 4, 128), dtype = torch.bfloat16, device = device, requires_grad = True)
v = torch.randn((9749, 4, 128), dtype = torch.bfloat16, device = device, requires_grad = True)
cu_q = torch.tensor([0,  266,  569,  848, 1056, 1404, 1629, 1758, 1984, 2242, 2593, 2768, 3054, 3269, 3422, 3565, 3951, 4139, 4318, 4509, 4745, 4965, 5203, 5682, 5993, 6211, 6503, 6718, 6917, 7171, 7451, 7738, 8010, 8250, 8562, 8707, 8974, 9181, 9421, 9576, 9749], dtype=torch.int32).to(device)
cu_k = torch.tensor([0,  266,  569,  848, 1056, 1404, 1629, 1758, 1984, 2242, 2593, 2768, 3054, 3269, 3422, 3565, 3951, 4139, 4318, 4509, 4745, 4965, 5203, 5682, 5993, 6211, 6503, 6718, 6917, 7171, 7451, 7738, 8010, 8250, 8562, 8707, 8974, 9181, 9421, 9576, 9749], dtype=torch.int32).to(device)
max_length_q=479
max_length_k=479
causal = True
softmax_scale = 0.08838834764831845
dropout_p = 0.0
deterministic = False

for i in range(5):
    out = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max_length_q, max_seqlen_k=max_length_k, softmax_scale=softmax_scale, dropout_p=0, deterministic=False, causal=True)
    out.sum().backward()
    q.grad = None

    out2, _ = flash_attn_varlen_func_aiter(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max_length_q, max_seqlen_k=max_length_k, softmax_scale=softmax_scale, dropout_p=0, deterministic=False, causal=True, return_lse=True)
    out2.sum().backward()
    q.grad = None


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
    out = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max_length_q, max_seqlen_k=max_length_k, softmax_scale=softmax_scale, dropout_p=0, deterministic=False, causal=True)
    out.sum().backward()
    q_grad = q.grad.clone()
    q.grad = None
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
    out2, _ = flash_attn_varlen_func_aiter(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=max_length_q, max_seqlen_k=max_length_k, softmax_scale=softmax_scale, dropout_p=0, deterministic=False, causal=True, return_lse=True)
    out2.sum().backward()
    q_grad2 = q.grad.clone()
    q.grad = None
    

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print('q_grad/q_grad2 is nan: ', q_grad.isnan().any(),q_grad2.isnan().any())
print('diff output, mean/max: ', (out2-out).abs().mean().item(),(out2-out).abs().max().item())
print('diff grad,  mean/max: ', (q_grad2-q_grad).abs().mean().item(),(q_grad2-q_grad).abs().max().item())

