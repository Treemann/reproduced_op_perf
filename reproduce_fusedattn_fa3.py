import os
import torch
from contextlib import nullcontext
# import transformer_engine_torch as tex
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
query_layer = torch.randn((1,seqlen,32,128), dtype = torch.bfloat16, device = device, requires_grad = True)
key_layer = torch.randn((1,seqlen,8,128), dtype = torch.bfloat16, device = device, requires_grad = True)
value_layer = torch.randn((1,seqlen,8,128), dtype = torch.bfloat16, device = device, requires_grad = True)
cu_seqlens_q = cu_seqlens_kv = torch.tensor([0, 32768], dtype=torch.int32).to(device)
attention_mask = torch.triu(torch.ones(seqlen, seqlen), diagonal=1).bool().unsqueeze(0).unsqueeze(0).to(device)

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
            qkv_layout='bshd_bshd_bshd',
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=None,
            cu_seqlens_kv_padded=None,
            max_seqlen_q=seqlen,
            max_seqlen_kv=seqlen,
            attn_mask_type='causal',
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
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
autograd::engine::evaluate_function: FusedAttnFuncBa...         0.00%      49.785us         0.05%       1.193ms     238.676us       0.000us         0.00%     765.809ms     153.162ms           0 b           0 b       1.86 Gb     -20.00 Mb             5
                                  FusedAttnFuncBackward         0.03%     716.457us         0.05%       1.144ms     228.719us     763.217ms        64.91%     765.809ms     153.162ms           0 b           0 b       1.88 Gb      -6.27 Gb             5
             aiter::fmha_bwd_hd128_bf16_causal_a32_rtna         0.00%       0.000us         0.00%       0.000us       0.000us     750.585ms        63.84%     750.585ms     150.117ms           0 b           0 b           0 b           0 b             5
                                          ProfilerStep*         0.00%       0.000us         0.00%       0.000us       0.000us     407.473ms        34.66%     407.473ms      81.495ms           0 b           0 b           0 b           0 b             5
                                          ProfilerStep*         0.15%       3.402ms         0.20%       4.485ms     896.997us       0.000us         0.00%     407.473ms      81.495ms           0 b           0 b           0 b      -1.27 Gb             5
                                          FusedAttnFunc         0.03%     567.003us         0.03%     765.521us     153.104us     406.384ms        34.56%     406.384ms      81.277ms           0 b           0 b       1.27 Gb      -2.50 Kb             5
void ck_tile::kentry<256, 2, ck_tile::FmhaFwdKernel<...         0.00%       0.000us         0.00%       0.000us       0.000us     406.362ms        34.56%     406.362ms      81.272ms           0 b           0 b           0 b           0 b             5
void ck_tile::kentry<64, 2, ck_tile::FmhaBwdOGradDot...         0.00%       0.000us         0.00%       0.000us       0.000us       5.190ms         0.44%       5.190ms       1.038ms           0 b           0 b           0 b           0 b             5
void ck_fused_attn::dk_dv_reduce<unsigned short>(uns...         0.00%       0.000us         0.00%       0.000us       0.000us       3.326ms         0.28%       3.326ms     665.151us           0 b           0 b           0 b           0 b             5
                                        Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us       2.609ms         0.22%       2.609ms     130.426us           0 b           0 b           0 b           0 b            20
                                       aten::contiguous         0.00%       6.109us         0.01%     207.570us      41.514us       0.000us         0.00%       2.592ms     518.483us           0 b           0 b       1.25 Gb           0 b             5
                                            aten::clone         0.00%      17.316us         0.01%     201.461us      40.292us       0.000us         0.00%       2.592ms     518.483us           0 b           0 b       1.25 Gb           0 b             5
                                            aten::copy_         0.00%      62.274us         0.01%     126.008us      25.202us       2.592ms         0.22%       2.592ms     518.483us           0 b           0 b           0 b           0 b             5
void at::native::elementwise_kernel_manual_unroll<12...         0.00%       0.000us         0.00%       0.000us       0.000us       2.592ms         0.22%       2.592ms     518.483us           0 b           0 b           0 b           0 b             5
autograd::engine::evaluate_function: torch::autograd...         0.00%      28.450us         0.01%     197.103us      13.140us       0.000us         0.00%       2.511ms     167.427us           0 b           0 b      -1.88 Gb           0 b            15
                        torch::autograd::AccumulateGrad         0.00%      37.795us         0.01%     168.653us      11.244us       0.000us         0.00%       2.511ms     167.427us           0 b           0 b      -1.88 Gb      -1.88 Gb            15
                                             aten::add_         0.00%      66.875us         0.01%     130.858us       8.724us       2.511ms         0.21%       2.511ms     167.427us           0 b           0 b           0 b           0 b            15
void at::native::vectorized_elementwise_kernel<8, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.511ms         0.21%       2.511ms     167.427us           0 b           0 b           0 b           0 b            15
void ck_tile::kentry<256, 2, ck_tile::FmhaBwdConvert...         0.00%       0.000us         0.00%       0.000us       0.000us       1.532ms         0.13%       1.532ms     306.319us           0 b           0 b           0 b           0 b             5
                                              aten::sum         0.01%     120.340us         0.01%     190.805us      38.161us       1.064ms         0.09%       1.064ms     212.890us           0 b           0 b       2.50 Kb       2.50 Kb             5
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       1.040ms         0.09%       1.040ms     208.068us           0 b           0 b           0 b           0 b             5
                                        aten::ones_like         0.00%      15.751us         0.00%      96.343us      19.269us       0.000us         0.00%      25.075us       5.015us           0 b           0 b       2.50 Kb           0 b             5
                                            aten::fill_         0.00%      26.069us         0.00%      51.868us      10.374us      25.075us         0.00%      25.075us       5.015us           0 b           0 b           0 b           0 b             5
void at::native::vectorized_elementwise_kernel<8, at...         0.00%       0.000us         0.00%       0.000us       0.000us      25.075us         0.00%      25.075us       5.015us           0 b           0 b           0 b           0 b             5
                     unpack(at::PhiloxCudaState, long*)         0.00%       0.000us         0.00%       0.000us       0.000us      21.909us         0.00%      21.909us       4.382us           0 b           0 b           0 b           0 b             5
                                            aten::empty         0.01%     172.131us         0.01%     172.131us       2.869us       0.000us         0.00%       0.000us       0.000us           0 b           0 b       9.41 Gb       9.41 Gb            60
                                   hipStreamIsCapturing         0.00%       7.661us         0.00%       7.661us       0.766us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            10
                                        hipLaunchKernel         0.02%     334.314us         0.02%     334.314us       6.078us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            55
                                      hipGetProcAddress         0.00%      25.397us         0.00%      25.397us       1.270us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            20
                                       hipCtxGetCurrent         0.00%       2.522us         0.00%       2.522us       0.126us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            20
                            hipGetDevicePropertiesR0600         0.00%       4.047us         0.00%       4.047us       0.405us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            10
                                             aten::view         0.00%      30.105us         0.00%      30.105us       6.021us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             5
                                       aten::as_strided         0.00%      19.329us         0.00%      19.329us       1.933us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            10
                                         hipMemsetAsync         0.00%     110.237us         0.00%     110.237us       5.512us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            20
                                       aten::empty_like         0.00%      27.021us         0.00%      86.861us       8.686us       0.000us         0.00%       0.000us       0.000us           0 b           0 b       1.25 Gb           0 b            10
                                    aten::empty_strided         0.00%      13.480us         0.00%      13.480us       2.696us       0.000us         0.00%       0.000us       0.000us           0 b           0 b       2.50 Kb       2.50 Kb             5
      autograd::engine::evaluate_function: SumBackward0         0.00%      49.983us         0.01%     135.952us      27.190us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             5
                                           SumBackward0         0.00%      39.861us         0.00%      85.969us      17.194us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             5
                                           aten::expand         0.00%      32.507us         0.00%      46.108us       9.222us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             5
     autograd::engine::evaluate_function: ViewBackward0         0.00%      21.694us         0.00%      71.027us      14.205us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             5
                                          ViewBackward0         0.00%       7.249us         0.00%      49.333us       9.867us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             5
                                          aten::reshape         0.00%      26.530us         0.00%      42.084us       8.417us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             5
                                   aten::_reshape_alias         0.00%      15.554us         0.00%      15.554us       3.111us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             5
                                  hipModuleLaunchKernel         0.00%      26.390us         0.00%      26.390us       5.278us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             5
                                   hipDeviceSynchronize        99.72%        2.201s        99.72%        2.201s        2.201s       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 2.207s
Self CUDA time total: 1.176s


### H20 ###
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
autograd::engine::evaluate_function: FusedAttnFuncBa...         0.00%      48.927us         0.11%       1.165ms     233.096us       0.000us         0.00%     782.736ms     156.547ms           0 b           0 b       1.86 Gb     -20.00 Mb             5
                                  FusedAttnFuncBackward         0.07%     745.116us         0.10%       1.117ms     223.311us     781.434ms        71.12%     782.736ms     156.547ms           0 b           0 b       1.88 Gb      -6.27 Gb             5
cudnn_generated_fort_native_sdpa_sm90_flash_bprop_wg...         0.00%       0.000us         0.00%       0.000us       0.000us     777.816ms        70.79%     777.816ms     155.563ms           0 b           0 b           0 b           0 b             5
                                          ProfilerStep*         0.00%       0.000us         0.00%       0.000us       0.000us     314.547ms        28.63%     314.547ms      62.909ms           0 b           0 b           0 b           0 b             5
                                          ProfilerStep*         0.32%       3.559ms         0.64%       7.007ms       1.401ms       0.000us         0.00%     314.424ms      62.885ms           0 b           0 b           0 b      -1.27 Gb             5
                                          FusedAttnFunc         0.08%     931.726us         0.28%       3.057ms     611.426us     313.771ms        28.56%     313.771ms      62.754ms           0 b           0 b       1.27 Gb      -2.50 Kb             5
cudnn_generated_fort_native_sdpa_sm90_flash_fprop_wg...         0.00%       0.000us         0.00%       0.000us       0.000us     313.760ms        28.55%     313.760ms      62.752ms           0 b           0 b           0 b           0 b             5
autograd::engine::evaluate_function: torch::autograd...         0.00%      38.458us         0.02%     189.900us      12.660us       0.000us         0.00%       1.635ms     108.994us           0 b           0 b      -1.88 Gb           0 b            15
                        torch::autograd::AccumulateGrad         0.00%      38.203us         0.01%     151.442us      10.096us       0.000us         0.00%       1.635ms     108.994us           0 b           0 b      -1.88 Gb      -1.88 Gb            15
                                             aten::add_         0.00%      53.863us         0.01%     113.239us       7.549us       1.635ms         0.15%       1.635ms     108.994us           0 b           0 b           0 b           0 b            15
void at::native::vectorized_elementwise_kernel<8, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.635ms         0.15%       1.635ms     108.994us           0 b           0 b           0 b           0 b            15
void cudnn::fusion::compute_dot_do_o_specialized<tru...         0.00%       0.000us         0.00%       0.000us       0.000us       1.557ms         0.14%       1.557ms     311.443us           0 b           0 b           0 b           0 b             5
                                       aten::contiguous         0.00%       5.140us         0.01%     153.484us      30.697us       0.000us         0.00%       1.302ms     260.486us           0 b           0 b       1.25 Gb           0 b             5
                                            aten::clone         0.00%      15.565us         0.01%     148.344us      29.669us       0.000us         0.00%       1.302ms     260.486us           0 b           0 b       1.25 Gb           0 b             5
                                            aten::copy_         0.00%      38.989us         0.01%      90.967us      18.193us       1.302ms         0.12%       1.302ms     260.486us           0 b           0 b           0 b           0 b             5
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       1.302ms         0.12%       1.302ms     260.486us           0 b           0 b           0 b           0 b             5
void cudnn::fusion::convert_dq_to_16bits<true>(void ...         0.00%       0.000us         0.00%       0.000us       0.000us       1.120ms         0.10%       1.120ms     223.980us           0 b           0 b           0 b           0 b             5
void cudnn::fusion::fmha_reduce_head<true>(void cons...         0.00%       0.000us         0.00%       0.000us       0.000us     935.677us         0.09%     935.677us      93.568us           0 b           0 b           0 b           0 b            10
                                              aten::sum         0.01%     123.905us         0.02%     184.790us      36.958us     646.333us         0.06%     646.333us     129.267us           0 b           0 b       2.50 Kb       2.50 Kb             5
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us     641.501us         0.06%     641.501us     128.300us           0 b           0 b           0 b           0 b             5
                                        Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us      14.752us         0.00%      14.752us       0.983us           0 b           0 b           0 b           0 b            15
                                        aten::ones_like         0.00%      19.997us         0.01%     107.662us      21.532us       0.000us         0.00%       6.624us       1.325us           0 b           0 b       2.50 Kb           0 b             5
                                            aten::fill_         0.00%      24.855us         0.00%      52.532us      10.506us       6.624us         0.00%       6.624us       1.325us           0 b           0 b           0 b           0 b             5
void at::native::vectorized_elementwise_kernel<8, at...         0.00%       0.000us         0.00%       0.000us       0.000us       6.624us         0.00%       6.624us       1.325us           0 b           0 b           0 b           0 b             5
                     unpack(at::PhiloxCudaState, long*)         0.00%       0.000us         0.00%       0.000us       0.000us       6.560us         0.00%       6.560us       1.312us           0 b           0 b           0 b           0 b             5
                                            aten::slice         0.01%      63.025us         0.01%      72.743us       7.274us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            10
                                       aten::as_strided         0.00%      19.795us         0.00%      19.795us       0.990us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            20
                                            aten::empty         0.02%     222.957us         0.02%     222.957us       3.716us       0.000us         0.00%       0.000us       0.000us           0 b           0 b       9.41 Gb       9.41 Gb            60
                                Activity Buffer Request         0.17%       1.863ms         0.17%       1.863ms       1.863ms       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1
                                  cudaStreamIsCapturing         0.00%      11.319us         0.00%      11.319us       1.132us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            10
                                       cudaLaunchKernel         0.02%     241.939us         0.02%     241.939us       6.913us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            35
                                cudaGetDriverEntryPoint         0.00%      12.875us         0.00%      12.875us       0.258us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            50
                                        cudaMemsetAsync         0.01%      83.675us         0.01%      83.675us       5.578us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            15
                                         cuLaunchKernel         0.01%     134.902us         0.01%     134.902us       4.497us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            30
                                             aten::view         0.00%      25.666us         0.00%      25.666us       5.133us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             5
                                       aten::empty_like         0.00%      21.112us         0.01%      76.945us       7.695us       0.000us         0.00%       0.000us       0.000us           0 b           0 b       1.25 Gb           0 b            10
                                    aten::empty_strided         0.00%      21.899us         0.00%      21.899us       4.380us       0.000us         0.00%       0.000us       0.000us           0 b           0 b       2.50 Kb       2.50 Kb             5
      autograd::engine::evaluate_function: SumBackward0         0.00%      44.327us         0.01%      95.760us      19.152us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             5
                                           SumBackward0         0.00%      28.755us         0.00%      51.433us      10.287us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             5
                                           aten::expand         0.00%      16.443us         0.00%      22.678us       4.536us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             5
     autograd::engine::evaluate_function: ViewBackward0         0.00%      20.909us         0.01%      59.639us      11.928us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             5
                                          ViewBackward0         0.00%       6.504us         0.00%      38.730us       7.746us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             5
                                          aten::reshape         0.00%      22.965us         0.00%      32.226us       6.445us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             5
                                   aten::_reshape_alias         0.00%       9.261us         0.00%       9.261us       1.852us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             5
                                    cudaGetFuncBySymbol         0.00%       2.973us         0.00%       2.973us       0.149us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            20
                                  cudaDeviceSynchronize        99.23%        1.094s        99.23%        1.094s        1.094s       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 1.103s
Self CUDA time total: 1.099s
'''
