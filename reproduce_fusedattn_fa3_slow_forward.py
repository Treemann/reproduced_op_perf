import os
import torch
from contextlib import nullcontext
from transformer_engine.pytorch.attention import FusedAttention
from torch.profiler import profile, record_function, ProfilerActivity

os.environ['NVTE_CK_USES_BWD_V3'] = '1'

softmax_scale=0.08838834764831843
attention_type='self'
layer_number=1
deterministic=False
attn_kwargs = {'attention_dropout': 0.0, 
              'attention_dropout_ctx': nullcontext}
seqlen = 32768
device = torch.device("cuda")
query_layer = torch.randn((seqlen,32,128), dtype = torch.bfloat16, device = device, requires_grad = True)
key_layer = torch.randn((seqlen,8,128), dtype = torch.bfloat16, device = device, requires_grad = True)
value_layer = torch.randn((seqlen,8,128), dtype = torch.bfloat16, device = device, requires_grad = True)
cu_seqlens_q = cu_seqlens_kv = torch.tensor([0,  3602,  5672,  9207, 11229, 14814, 15286, 18924, 19417, 20504, 24106, 26087, 29675, 32768], dtype=torch.int32).to(device)
attention_mask = torch.triu(torch.ones(seqlen, seqlen), diagonal=1).bool().unsqueeze(0).unsqueeze(0).to(device)
# attention_mask = torch.load('attention_mask.pt').to(device)
max_seqlen = 3638

fused_attention = FusedAttention(
    softmax_scale,
    attention_type=attention_type,
    layer_number=layer_number,
    deterministic=deterministic,
    **attn_kwargs,
)

profiler = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=5,
        warmup=0,
        active=10,
        repeat=1
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile"),
    record_shapes=True, 
    profile_memory=True,
    with_stack=True 
)

with profiler:
    for i in range(15):
        out = fused_attention(
            query_layer,
            key_layer,
            value_layer,
            qkv_layout='thd_thd_thd',
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            attn_mask_type='padding_causal',
            attention_mask=attention_mask,
            window_size=(-1,0),
            fused_attention_backend=1, #tex.NVTE_Fused_Attn_Backend.NVTE_CK
            core_attention_bias_type='no_bias',
            core_attention_bias=None,
            fast_zero_fill=True,
            cp_group=None,
            cp_global_ranks=None,
            cp_stream=None,
            cp_comm_type='p2p',
            fp8=False,
            fp8_meta={},
            quantizers={},
            pad_between_seqs=False,
            inference_params=None,
        )
        out.sum().backward()
        profiler.step()
print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=-1))

'''
Perf of FusedAttn forward is slower in rocm7.0 docker than in rocm6.4.2 docker

rocm7.0
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
autograd::engine::evaluate_function: FusedAttnFuncBa...         0.03%      90.305us         0.54%       1.877ms     187.709us       0.000us         0.00%     206.268ms      20.627ms           0 B           0 B       3.71 GB     -40.00 MB            10
                                  FusedAttnFuncBackward         0.28%     975.667us         0.51%       1.787ms     178.679us     202.793ms        58.82%     206.268ms      20.627ms           0 B           0 B       3.75 GB     -12.58 GB            10
aiter::fmha_bwd_hd128_bf16_causal_a32_rtna_pssk_grou...         0.00%       0.000us         0.00%       0.000us       0.000us     179.974ms        52.20%     179.974ms      17.997ms           0 B           0 B           0 B           0 B            10
                                          ProfilerStep*         0.00%       0.000us         0.00%       0.000us       0.000us     135.106ms        39.19%     135.106ms      13.511ms           0 B           0 B           0 B           0 B            10
                                          ProfilerStep*         1.43%       4.990ms         1.89%       6.601ms     660.147us       0.000us         0.00%     135.106ms      13.511ms           0 B           0 B           0 B      -2.54 GB            10
                                          FusedAttnFunc         0.22%     760.901us         0.32%       1.132ms     113.193us     134.171ms        38.91%     134.171ms      13.417ms           0 B           0 B       2.54 GB     -40.00 MB            10
_ZN7ck_tile6kentryILi256ELi2ENS_13FmhaFwdKernelINS_2...         0.00%       0.000us         0.00%       0.000us       0.000us     132.678ms        38.48%     132.678ms      13.268ms           0 B           0 B           0 B           0 B            10
_ZN7ck_tile6kentryILi64ELi2ENS_22FmhaBwdOGradDotOKer...         0.00%       0.000us         0.00%       0.000us       0.000us       7.776ms         2.26%       7.776ms     777.583us           0 B           0 B           0 B           0 B            10
_ZN13ck_fused_attn16dk_dv_reduce_thdIDF16bEEvmmmPKiP...         0.00%       0.000us         0.00%       0.000us       0.000us       6.556ms         1.90%       6.556ms     655.646us           0 B           0 B           0 B           0 B            10
                                        Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us       6.186ms         1.79%       6.186ms     123.723us           0 B           0 B           0 B           0 B            50
autograd::engine::evaluate_function: torch::autograd...         0.01%      51.057us         0.09%     327.268us      10.909us       0.000us         0.00%       3.415ms     113.844us           0 B           0 B      -3.75 GB           0 B            30
                        torch::autograd::AccumulateGrad         0.02%      74.387us         0.08%     276.211us       9.207us       0.000us         0.00%       3.415ms     113.844us           0 B           0 B      -3.75 GB      -3.75 GB            30
                                             aten::add_         0.03%      90.704us         0.06%     201.824us       6.727us       3.415ms         0.99%       3.415ms     113.844us           0 B           0 B           0 B           0 B            30
void at::native::vectorized_elementwise_kernel<8, at...         0.00%       0.000us         0.00%       0.000us       0.000us       3.415ms         0.99%       3.415ms     113.844us           0 B           0 B           0 B           0 B            30
_ZN7ck_tile6kentryILi256ELi2ENS_25FmhaBwdConvertQGra...         0.00%       0.000us         0.00%       0.000us       0.000us       2.746ms         0.80%       2.746ms     274.611us           0 B           0 B           0 B           0 B            10
                                       aten::contiguous         0.00%      15.974us         0.08%     272.966us      27.297us       0.000us         0.00%       1.884ms     188.383us           0 B           0 B       2.50 GB           0 B            10
                                            aten::clone         0.01%      27.832us         0.07%     256.992us      25.699us       0.000us         0.00%       1.884ms     188.383us           0 B           0 B       2.50 GB           0 B            10
                                            aten::copy_         0.02%      78.303us         0.04%     153.568us      15.357us       1.884ms         0.55%       1.884ms     188.383us           0 B           0 B           0 B           0 B            10
void at::native::elementwise_kernel_manual_unroll<12...         0.00%       0.000us         0.00%       0.000us       0.000us       1.884ms         0.55%       1.884ms     188.383us           0 B           0 B           0 B           0 B            10
                                            aten::fill_         0.03%     108.056us         0.07%     256.048us       6.401us       1.641ms         0.48%       1.641ms      41.028us           0 B           0 B           0 B           0 B            40
void at::native::vectorized_elementwise_kernel<8, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.641ms         0.48%       1.641ms      41.028us           0 B           0 B           0 B           0 B            40
                                              aten::sum         0.05%     186.322us         0.08%     281.457us      28.146us     885.416us         0.26%     885.416us      88.542us           0 B           0 B       5.00 KB       5.00 KB            10
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     837.756us         0.24%     837.756us      83.776us           0 B           0 B           0 B           0 B            10
void transformer_engine::fused_attn_rocm::remove_pad...         0.00%       0.000us         0.00%       0.000us       0.000us     638.981us         0.19%     638.981us      63.898us           0 B           0 B           0 B           0 B            10
void transformer_engine::fused_attn_rocm::add_paddin...         0.00%       0.000us         0.00%       0.000us       0.000us     407.355us         0.12%     407.355us      40.736us           0 B           0 B           0 B           0 B            10
                                        aten::ones_like         0.01%      25.246us         0.04%     147.217us      14.722us       0.000us         0.00%      49.384us       4.938us           0 B           0 B       5.00 KB           0 B            10
                     unpack(at::PhiloxCudaState, long*)         0.00%       0.000us         0.00%       0.000us       0.000us      48.422us         0.01%      48.422us       4.842us           0 B           0 B           0 B           0 B            10
                                            aten::empty         0.08%     272.267us         0.08%     272.267us       2.269us       0.000us         0.00%       0.000us       0.000us           0 B           0 B      18.91 GB      18.91 GB           120
                                         hipMemsetAsync         0.07%     245.878us         0.07%     245.878us       4.918us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            50
                                   hipStreamIsCapturing         0.00%       6.622us         0.00%       6.622us       0.331us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            20
                                        hipLaunchKernel         0.17%     611.794us         0.17%     611.794us       3.824us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B           160
                            hipGetDevicePropertiesR0600         0.01%      17.537us         0.01%      17.537us       0.877us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            20
                                             aten::view         0.01%      50.668us         0.01%      50.668us       5.067us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            10
                                       aten::as_strided         0.01%      28.875us         0.01%      28.875us       1.444us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            20
                                       aten::empty_like         0.01%      33.946us         0.03%     117.213us       5.861us       0.000us         0.00%       0.000us       0.000us           0 B           0 B       2.50 GB           0 B            20
                                    aten::empty_strided         0.01%      23.697us         0.01%      23.697us       2.370us       0.000us         0.00%       0.000us       0.000us           0 B           0 B       5.00 KB       5.00 KB            10
      autograd::engine::evaluate_function: SumBackward0         0.02%      65.128us         0.05%     175.261us      17.526us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            10
                                           SumBackward0         0.01%      48.322us         0.03%     110.133us      11.013us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            10
                                           aten::expand         0.01%      43.472us         0.02%      61.811us       6.181us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            10
     autograd::engine::evaluate_function: ViewBackward0         0.01%      23.125us         0.03%      98.927us       9.893us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            10
                                          ViewBackward0         0.00%      12.579us         0.02%      75.802us       7.580us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            10
                                          aten::reshape         0.01%      41.771us         0.02%      63.223us       6.322us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            10
                                   aten::_reshape_alias         0.01%      21.452us         0.01%      21.452us       2.145us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            10
                                      hipGetProcAddress         0.01%      17.916us         0.01%      17.916us       1.792us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            10
                                       hipCtxGetCurrent         0.00%       1.663us         0.00%       1.663us       0.166us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            10
                                  hipModuleLaunchKernel         0.01%      38.356us         0.01%      38.356us       3.836us       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B            10
                                   hipDeviceSynchronize        97.40%     340.617ms        97.40%     340.617ms     340.617ms       0.000us         0.00%       0.000us       0.000us           0 B           0 B           0 B           0 B             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 349.697ms
Self CUDA time total: 344.789ms

rocm6.4.2
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
autograd::engine::evaluate_function: FusedAttnFuncBa...         0.02%      81.411us         0.62%       2.623ms     262.322us       0.000us         0.00%     253.316ms      25.332ms           0 b           0 b       3.71 Gb     -40.00 Mb            10
                                  FusedAttnFuncBackward         0.28%       1.167ms         0.60%       2.542ms     254.181us     243.339ms        64.58%     253.316ms      25.332ms           0 b           0 b       3.75 Gb     -27.29 Gb            10
aiter::fmha_bwd_hd128_bf16_causal_a32_rtna_pssk_grou...         0.00%       0.000us         0.00%       0.000us       0.000us     179.888ms        47.74%     179.888ms      17.989ms           0 b           0 b           0 b           0 b            10
                                          ProfilerStep*         0.00%       0.000us         0.00%       0.000us       0.000us     119.472ms        31.70%     119.472ms      11.947ms           0 b           0 b           0 b           0 b            10
                                          ProfilerStep*         1.45%       6.156ms         1.96%       8.294ms     829.395us       0.000us         0.00%     119.472ms      11.947ms           0 b           0 b           0 b      -2.54 Gb            10
                                          FusedAttnFunc         0.22%     941.381us         0.37%       1.583ms     158.268us     117.309ms        31.13%     117.309ms      11.731ms           0 b           0 b       2.54 Gb      -6.29 Gb            10
void ck_tile::kentry<256, 2, ck_tile::FmhaFwdKernel<...         0.00%       0.000us         0.00%       0.000us       0.000us      95.944ms        25.46%      95.944ms       9.594ms           0 b           0 b           0 b           0 b            10
void transformer_engine::fused_attn_rocm::remove_pad...         0.00%       0.000us         0.00%       0.000us       0.000us      37.758ms        10.02%      37.758ms     471.979us           0 b           0 b           0 b           0 b            80
void transformer_engine::fused_attn_rocm::add_paddin...         0.00%       0.000us         0.00%       0.000us       0.000us      18.639ms         4.95%      18.639ms     465.969us           0 b           0 b           0 b           0 b            40
                                        Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us       9.799ms         2.60%       9.799ms     108.877us           0 b           0 b           0 b           0 b            90
void ck_tile::kentry<64, 2, ck_tile::FmhaBwdOGradDot...         0.00%       0.000us         0.00%       0.000us       0.000us       7.439ms         1.97%       7.439ms     743.949us           0 b           0 b           0 b           0 b            10
void ck_fused_attn::dk_dv_reduce_thd<unsigned short>...         0.00%       0.000us         0.00%       0.000us       0.000us       7.377ms         1.96%       7.377ms     737.652us           0 b           0 b           0 b           0 b            10
                                       aten::contiguous         0.00%      10.286us         0.08%     321.596us      32.160us       0.000us         0.00%       5.149ms     514.864us           0 b           0 b       2.50 Gb           0 b            10
                                            aten::clone         0.01%      31.492us         0.07%     311.310us      31.131us       0.000us         0.00%       5.149ms     514.864us           0 b           0 b       2.50 Gb           0 b            10
                                            aten::copy_         0.02%      96.785us         0.05%     194.141us      19.414us       5.149ms         1.37%       5.149ms     514.864us           0 b           0 b           0 b           0 b            10
void at::native::elementwise_kernel_manual_unroll<12...         0.00%       0.000us         0.00%       0.000us       0.000us       5.149ms         1.37%       5.149ms     514.864us           0 b           0 b           0 b           0 b            10
                                            aten::fill_         0.03%     119.839us         0.07%     306.973us       7.674us       4.879ms         1.29%       4.879ms     121.983us           0 b           0 b           0 b           0 b            40
void at::native::vectorized_elementwise_kernel<8, at...         0.00%       0.000us         0.00%       0.000us       0.000us       4.879ms         1.29%       4.879ms     121.983us           0 b           0 b           0 b           0 b            40
autograd::engine::evaluate_function: torch::autograd...         0.01%      51.760us         0.08%     327.016us      10.901us       0.000us         0.00%       4.038ms     134.604us           0 b           0 b      -3.75 Gb           0 b            30
                        torch::autograd::AccumulateGrad         0.02%      65.564us         0.07%     275.256us       9.175us       0.000us         0.00%       4.038ms     134.604us           0 b           0 b      -3.75 Gb      -3.75 Gb            30
                                             aten::add_         0.02%      98.508us         0.05%     209.692us       6.990us       4.038ms         1.07%       4.038ms     134.604us           0 b           0 b           0 b           0 b            30
void at::native::vectorized_elementwise_kernel<8, at...         0.00%       0.000us         0.00%       0.000us       0.000us       4.038ms         1.07%       4.038ms     134.604us           0 b           0 b           0 b           0 b            30
void ck_tile::kentry<256, 2, ck_tile::FmhaBwdConvert...         0.00%       0.000us         0.00%       0.000us       0.000us       2.745ms         0.73%       2.745ms     274.459us           0 b           0 b           0 b           0 b            10
                                              aten::sum         0.05%     206.917us         0.08%     330.532us      33.053us       2.113ms         0.56%       2.113ms     211.304us           0 b           0 b       5.00 Kb       5.00 Kb            10
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       2.063ms         0.55%       2.063ms     206.294us           0 b           0 b           0 b           0 b            10
void transformer_engine::fused_attn_rocm::remove_pad...         0.00%       0.000us         0.00%       0.000us       0.000us     640.263us         0.17%     640.263us      64.026us           0 b           0 b           0 b           0 b            10
void transformer_engine::fused_attn_rocm::add_paddin...         0.00%       0.000us         0.00%       0.000us       0.000us     416.650us         0.11%     416.650us      41.665us           0 b           0 b           0 b           0 b            10
                     unpack(at::PhiloxCudaState, long*)         0.00%       0.000us         0.00%       0.000us       0.000us      51.828us         0.01%      51.828us       5.183us           0 b           0 b           0 b           0 b            10
                                        aten::ones_like         0.01%      25.519us         0.04%     171.785us      17.178us       0.000us         0.00%      50.627us       5.063us           0 b           0 b       5.00 Kb           0 b            10
                                            aten::empty         0.07%     289.579us         0.07%     289.579us       2.413us       0.000us         0.00%       0.000us       0.000us           0 b           0 b      39.87 Gb      39.87 Gb           120
                                         hipMemsetAsync         0.11%     446.050us         0.11%     446.050us       4.956us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            90
                                   hipStreamIsCapturing         0.00%       7.639us         0.00%       7.639us       0.382us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            20
                                        hipLaunchKernel         0.29%       1.210ms         0.29%       1.210ms       4.320us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b           280
                                      hipGetProcAddress         0.01%      46.986us         0.01%      46.986us       1.175us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            40
                                       hipCtxGetCurrent         0.00%       4.524us         0.00%       4.524us       0.113us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            40
                            hipGetDevicePropertiesR0600         0.00%       7.660us         0.00%       7.660us       0.383us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            20
                                             aten::view         0.01%      52.630us         0.01%      52.630us       5.263us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            10
                                       aten::as_strided         0.01%      30.639us         0.01%      30.639us       1.532us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            20
                                       aten::empty_like         0.01%      47.840us         0.03%     136.262us       6.813us       0.000us         0.00%       0.000us       0.000us           0 b           0 b       2.50 Gb           0 b            20
                                    aten::empty_strided         0.01%      24.646us         0.01%      24.646us       2.465us       0.000us         0.00%       0.000us       0.000us           0 b           0 b       5.00 Kb       5.00 Kb            10
      autograd::engine::evaluate_function: SumBackward0         0.01%      61.482us         0.05%     190.443us      19.044us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            10
                                           SumBackward0         0.01%      58.918us         0.03%     128.961us      12.896us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            10
                                           aten::expand         0.01%      50.271us         0.02%      70.043us       7.004us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            10
     autograd::engine::evaluate_function: ViewBackward0         0.01%      22.777us         0.02%     104.778us      10.478us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            10
                                          ViewBackward0         0.00%      11.244us         0.02%      82.001us       8.200us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            10
                                          aten::reshape         0.01%      46.261us         0.02%      70.757us       7.076us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            10
                                   aten::_reshape_alias         0.01%      24.496us         0.01%      24.496us       2.450us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            10
                                  hipModuleLaunchKernel         0.01%      43.042us         0.01%      43.042us       4.304us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            10
                                   hipDeviceSynchronize        97.27%     411.593ms        97.27%     411.593ms     411.593ms       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 423.132ms
Self CUDA time total: 376.827ms
'''
