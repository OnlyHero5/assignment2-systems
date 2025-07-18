*** note that all results below are based on running on an A100 (rather than the H100 used for the normal class) ***


## nsys_profile

a. forward passes take around 33 ms each (for d_model=512, n_layers=4, context_length=256).  this is in line with what we observed via python profiling.  however, 

b. various matmuls (e.g. ampere_sgemm_128x64_tn) account for over 70% of total time in the forward pass, totalling 30 invocations for a single forward pass

c. most remaining time in the forward pass comes from elementwise_kernel and variations thereof

d. the optimizer step adds ~3 ms, almost entirely from vectorized_elementwise_kernel and reduce_kernel.  this reduces the time spent in matmul for the entire pass to the low 50s%.  however, not all of the clock runtime for the optimizer step is accounted for-- its span is 16 ms but there are only 3 ms of kernel time captured by the profiler

e. softmax accounts for about 25% of the runtime in each self-attention block, far higher than the proportion of flops that it accounts for


## benchmarking_mixed_precision

a.
    - model params (initialized as float32) remain float32
    - linear components output float16 (including final output)
    - relu float16 -> float16
    - layer norm returns back to float32
    - loss is float32
    - all gradients are float32

b. layer norm includes a division by the square root of variance, which could result in massive fluctuations in values from the lower precision of float16, especially when variances are quite small (close to zero).  in contrast, the linear layers consist of simple multiplications and additions, where these small differences won't have nearly the same magnitude of impact.  with bf16, we would still need to treat layer norm differently, as it has even *less* precision than standard float16 (7 bits compared to 10).  the issue here is related to the precision of the floating point values, not their dynamic ranges.

c. all results below based on including full backward pass and optimizer step.  with n_layers=4, d_model=512, using float16 and bfloat16 each yielded run times of ~68 ms, compared to 128 ms for float 32, nearly a 47% speedup.  with a larger model (n_layers=16, d_model=1024), the speedup is even more dramatic-- 760 ms -> 343 ms (~55% speedup)





