# flare-5090

**NVIDIA Flare, serving on GeForce RTX 5090.** A fork of
[NVIDIA/flare-local-inference](https://github.com/NVIDIA/flare-local-inference)
that makes Flare's TP4 inference work on consumer Blackwell workstations
(4× RTX 5090, 32 GB) — the hardware upstream does not validate today
("Qwen TP1 and TP2 … have been exercised end to end; Qwen TP4 still requires
model validation").

First result: **Qwen3.8-Flash-Next (125B MoE, NVFP4) serving on 4× RTX 5090**
— 23.50 GiB of weights per GPU, **1,007,616 KV tokens** at concurrency 8,
262K context per request, MTP speculative decoding, OpenAI/Anthropic APIs,
Rust runtime — on four cards that cost less than one data-center GPU.

## Why a fork

Out of the box, Flare cannot start multi-GPU serving on a 5090 box. Three
independent blockers, in the order we hit them:

1. **No CUDA peer access on GeForce.** `can_device_access_peer` is 0 for
   every pair (`nvidia-smi topo -p2p` = `GNS`), and Flare's single-host TP
   collectives (FlashInfer/b12x PCIe one-shot family) map peer GPU memory
   directly — preparation dies at collective bootstrap
   (`cuMemSetAccess: invalid device ordinal`). Unlocking the *capability*
   with GSP registry keys is not enough: peer copies then return zeros and
   peer kernels hang, because the direct peer-write path is not routed on
   this platform. The working combination on driver 610 is five registry
   keys including the BAR1 mailbox P2P mode, IOMMU passthrough, and baking
   the modprobe drop-in into initramfs. Full recipe, evidence, and the
   data-path verification method: [docs/rtx-5090.md](docs/rtx-5090.md).
   The unlock itself lives in
   [kacper-daftcode/rtx-p2p](https://github.com/kacper-daftcode/rtx-p2p).
2. **Host bring-up gaps** — `protoc`, `liburing-dev`, `pkg-config`, and a
   CUDA ≥12.8 toolkit pin; each missing piece fails preparation opaquely.
3. **A Helios graph-construction bug** (fixed here): packages record each
   kernel function's dynamic shared-memory opt-in as a snapshot taken
   before b12x tuning raises it, so `flare run` rejected kernels needing
   more than the snapshot (`cuGraphAddKernelNode_v2`, 57,600 B required vs
   49,152 B recorded; snapshots even differed across ranks for the same
   function). Helios now names the graph/node/kernel on failure and raises
   the function's limit to the node's requirement before adding it.

With P2P on, the same box also stops being a second-class citizen for
everyone else: NCCL picks real peer transports (2-rank all-reduce verified;
peer copies at **56.3 GB/s**, the PCIe Gen5 x16 practical ceiling), expert
parallelism and multi-GPU training collectives work. None of this applies
to RTX PRO 6000 — professional cards were never locked.

## What is measured (4× RTX 5090, one workstation)

| | |
|---|---|
| Model | Qwen3.8-Flash-Next-FP4-Local (NVFP4), TP4, MTP k=3, concurrency 8 |
| Export | ~10 min warm cache (tuning 35 s, forward compile 156 s) |
| Weights per GPU | 23.50 GiB |
| KV cache | 29.2 GiB total = **1,007,616 tokens**, 262,144 max per request |
| Serving resident per GPU | 32,048 MiB |
| Smoke | `/v1/models` + greedy chat completions, clean shutdown |

Details and exact commands: [docs/rtx-5090.md](docs/rtx-5090.md).

## Changes vs upstream (5f37bd1)

- `helios/` — dynamic shared-memory retry + graph/node/kernel context in
  graph-construction errors. Small, model-agnostic, no numerics change;
  intended as an upstream PR. (The fuller fix is on the Ignition recording
  side and is documented as such.)
- `docs/rtx-5090.md` — the consumer-Blackwell bring-up guide above.
- This README. Everything else is upstream, untouched.

## Roadmap

- Upstream the Helios fix (with the cross-rank snapshot evidence).
- Remaining catalog, sized honestly: `glm-5.3-flash` does **not** fit
  4× 32 GB (304B routed experts = 42.8 GiB/rank at NVFP4 before dense
  weights or KV). It belongs to the EXL3/hybrid tier, not this fork's
  resident path; `deepseek-v4.1-flash` is in the same weight class.
- Weight-format yardstick for this class of box: NVFP4 vs EXL3 trellis
  4–5 bpw vs FP8 (KL/top-1 vs BF16, one harness), then the serving-leg
  comparison against vLLM on the same checkpoint.
- Measure mailbox-mode small-message latency (the decode all-reduce axis);
  bandwidth already saturates the link.
- Track upstream; this fork intends to stay a thin, rebaseable delta.

## Upstream

Flare is NVIDIA's local inference server for SM120/SM121: Ignition prepares
models with PyTorch/Transformers/FlashInfer-b12x into prepared CUDA graphs;
Helios, the Rust runtime, serves them without Python. Design and usage docs
live in [docs/](docs/) (start with `docs/model-preparation.md`); the
upstream README is in git history at the fork point. Issues about stock
Flare belong upstream; issues about RTX 5090 / consumer-Blackwell support
belong here.

## License

Apache-2.0, as upstream (see `LICENSE`). The RTX 5090 bring-up notes are
provided as-is; the peer-to-peer unlock flips undocumented firmware flags
on hardware you own — no warranty.
