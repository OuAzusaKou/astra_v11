# DexBench2 MANO ImageWAM 10k client/server evaluation

## Target

- Training run: `imagewam-dexbench2-mano-h32-368x320-lr5e5-10k-1n-b30z-20260826-095327`
- Checkpoint: `step_010000.pt`
- Dataset/adapter: `dex_bench_2` / `teleop_mano`
- Model input: 368×320
- Action: H32, canonical 326D relative EE + relative MANO90
- Server inference: seed 42, 10 denoising steps, production compile profile
- Wire protocol: one persistent, length-prefixed msgpack-numpy TCP connection

The client sends the exact decoded 480×640 RGB dataset frames and the same
dataset-named low-dimensional fields used by the live robot client. The server
owns RGB preprocessing, canonical packing, masks, normalization, and action
denormalization. Every returned action is additionally passed through the
deploy client's relative EE and MANO90-to-q20 decoder.

## Sample selection

The training config's episode split is reused exactly. With seed 42, the
preparation script samples valid frame anchors uniformly without replacement:

| Split | Samples | Episodes represented | Split episodes | Valid anchors |
| --- | ---: | ---: | ---: | ---: |
| train | 100 | 96 | 951 | 461,806 |
| validation (`eval`) | 100 | 44 | 50 | 23,729 |

The exact positions and source paths are recorded in
`outputs/imagewam-dexbench2-mano-h32-368x320-lr5e5-10k/samples_seed42_train100_eval100.manifest.json`.

## Pred-vs-GT MSE

These metrics compare the final denoised H32 action returned by the server with
the canonical GT H32 action. Only valid action values are included. They are
not the diffusion flow-matching training objective.

| Metric | train | validation |
| --- | ---: | ---: |
| normalized unweighted MSE, macro | 0.000120376 | 0.001103515 |
| normalized unweighted MSE, micro | 0.000115062 | 0.001099595 |
| normalized EE=1/MANO=0.5 weighted MSE, macro | 0.000094763 | 0.000887184 |
| physical mixed-unit MSE, macro | 0.000060942 | 0.000497092 |
| physical mixed-unit MSE, micro | 0.000059797 | 0.000503602 |

`macro` computes each sample over its own valid values and then averages the
100 samples. `micro` pools all valid values first. The weighted metric uses the
training recipe's per-value EE=1.0/MANO90=0.5 weighting but retains the raw
valid-value denominator.

Additional physical diagnostics:

| Metric | train | validation |
| --- | ---: | ---: |
| EE translation L2 mean | 0.322 cm | 1.060 cm |
| EE rotation error mean | 1.196° | 2.702° |
| MANO joint rotation error mean | 0.596° | 1.138° |

All 200 server predictions and all 200 deploy-client 52D action decodes were
finite. Mean model inference latency was 42.6 ms on train samples and 41.9 ms
on validation samples. Mean persistent client round-trip latency was 103.5 ms
and 91.2 ms respectively.

## Reproduce

The deploy environment must contain the declared Python dependencies and a
native extension built for its Python version:

```bash
python scripts/build_native_mano_ik.py --force
SERVER_GPU=3 SERVER_PORT=29658 \
  ops/dexbench2_mano/evaluate_client_server.sh
```

The workflow reuses an existing sample payload unless `REFRESH_SAMPLES=1` is
set. Outputs are written below
`outputs/imagewam-dexbench2-mano-h32-368x320-lr5e5-10k/`:

- `server.log` and `client.log`: startup, inference, and progress logs;
- `results/metrics.json`: full metric and server metadata;
- `results/predictions.npz`: canonical predictions, decoded 52D actions,
  sample identities, and per-request latency.
