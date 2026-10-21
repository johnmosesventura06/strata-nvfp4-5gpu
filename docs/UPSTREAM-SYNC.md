# Upstream sync marker

This fork has **no shared git history** with upstream (it started on 2026-10-10 as a snapshot of
sergqwer/strata-nvfp4 at engine 0.1.41). So `git merge-base` doesn't work. Compare by the upstream
commits recorded below, and port by content.

**For the next upstream review, start after these commits. Everything at or before them has been reviewed.**

| Upstream | Branch | Last commit reviewed | Commit date |
|---|---|---|---|
| [Niko1221/Strata](https://github.com/Niko1221/Strata) (original) | `main` | `011c9e600a93bbc9fd451c563501b5831c32bbca` | 2026-10-21 11:07 +0200 |
| [sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) (parent fork) | `main` | `57293b8ec57203a741dc7b2abd69f24f077037ab` | 2026-10-20 04:29 +0300 |
| sergqwer/strata-nvfp4 | `nvfp4` | `cfa0f74722a4483ea8a62d69dd0ad254dea4e1c0` | 2026-10-01 |
| sergqwer/strata-nvfp4 | `glm` | `90fcc69b011bc7ae4f6156b88299b9ef9b95c41f` | 2026-10-01 |

How to list what is new:

```sh
git remote add niko https://github.com/Niko1221/Strata.git    # once
git remote add serg https://github.com/sergqwer/strata-nvfp4.git
git fetch niko && git fetch serg
git log --format='%h %cs %s' 011c9e60..niko/main
git log --format='%h %cs %s' 57293b8e..serg/main
```

Review scope: speed changes that could apply to this rig. The rig has an RTX 3090 (sm_86) trunk that runs all
dense, attention and decode-window work, and 4× RTX 5060 Ti (sm_120) tiers that run only routed experts
(decode graphs and prompt-chunk rows). Experts are NVFP4, dense weights Q8_0, KV int8, and it is multi-GPU on
consumer PCIe. Upstream is mostly single-GPU, AMD/Intel/Mac, and GGUF i-quants, and most of that does not apply.

## Reviewed 2026-10-22: ported (branch `upstream-port`)

| # | Upstream commit(s) | What | Switch | Outputs |
|---|---|---|---|---|
| A | `546d7675` + `6e6d4faf` | NVFP4 decode experts: each weight decoded once per window (`Split<40>`), down on 8-lane rows | always on | rounding-level vs our old kernel (all text changes in the A/B came from this) |
| B | `5b6c9398` | QSA decode block scores, queries in registers | `STRATA_QSA_ROWS=1` | identical |
| C | `75afe3f4` | expert gate/up + SwiGLU + q8_1 in one launch (opt-in here, not arch-default) | `STRATA_GU_SWIGLU=1` | identical |
| D | `731528be` | the Daily's decode attention chunk kernel + merge v2 (minimal port, no parity-harness refactor) | `STRATA_ATTN_DAILY=1` | identical |
| E | `ffc30b8d` | the mixer's input-only work on graph branches (single-stream windows only) | `STRATA_SIDE_MIXER=2` | identical |
| F | `1a2f95c3` | split tensor-core decode attention (also hooked on our batch path; not on the MTP draft) | `STRATA_QSA_SPLIT_ATTN=1` | rounding-level; no gain measured, not used |
| G | docs `703e63c1` (no upstream code) | NVFP4 prompt down products on two FP4 terms (5060 Ti tiers) | `STRATA_PREFILL_X2_MIN_K=640` | rounding-level, closer to FP16 per upstream |

Measured on this rig with A–E (2026-10-22, one boot each, vs production `24a57eaf`): decode 1 / 3 / 4 streams
128.4 / 161.4 / 169.8 → 143.7 / 179.1 / 188.0 tok/s (+11–12%); batch window 28.8 → 25.0 ms.

## Reviewed 2026-10-22: already in this fork (no port needed)

- serg `78855ec0` / `53001550` / `87e0faea`: fused prompt experts read in place (−6% at 32K).
- INT8 tensor-core prompt attention (`STRATA_PROMPT_ATTN_IMMA`): already the default.
- `STRATA_HC_UPMIX`: present. It defaults on only for sm_120, and the trunk is sm_86. Upstream measured +0.4–0.6%
  prompt on sm_86, bit-exact. Candidate switch, not a port.
- `STRATA_ROUTE_RESIDENT`, `STRATA_PF_FUSED` (i-quant formats only), `--pipeline-windows`: present or not applicable.

## Reviewed 2026-10-22: deliberately not taken

- **Elastic slot K/V** (`923dc36d` and related): declined by the owner.
- **`STRATA_ROUTE_TAIL_SKIP`** (+11.5% decode on NVFP4, changes answers): declined by the owner.
- `STRATA_EXPDEC` (`f7eaf2de`): upstream itself dropped it from the sm_86 set (it did not pay).
- `STRATA_GR_UP75` / `STRATA_GR_DOWN_TG`: our own `STRATA_GR_V6` hyper-connection kernels replace that path.
- `STRATA_B6_MMVQ_ROWS`, `native_mmvq_il` 5–8 column tables: K-quant / i-quant matvecs. Our dense Q8_0 runs on
  `STRATA_MMA_Q8`.
- `STRATA_ADAPT_NOWAIT` / `STRATA_ADAPT_LAG` / `STRATA_ADAPT_OVERLAP`, `STRATA_CPU_IQ_ALL_EXACT1`: adaptive tier
  and CPU experts. Production has every expert on a GPU.
- `STRATA_HC_PERSIST` (`6cbc15a8`): HIP only.
- sm_120 default set (`ba2c2d5f`, `959e6c34`, `b1e0fdaa`, `94104176`, `75fabbd9`): targets cards that run the whole
  model. Our sm_120 cards run only experts. The expert part (GU_SWIGLU) is taken above.
- Everything AMD (HIP/gfx), Intel (SYCL), Apple (Metal), Volta/Turing/Pascal, Gyro trellis formats, the new UI,
  updater, installer and setup changes.

## Our own work that upstream does not have (do not "re-port" over it)

Peer expert tiers and device tiers (`STRATA_DEVICE_TIERS`), the lend shadow, the output head split across tiers,
tier prefill (P3b compact + FP16 tier sums), the NVMe delta journal, the conversation RAM pool, `STRATA_GR_V6`,
`STRATA_MMA_Q8`, and `STRATA_MTP_ATTN_FAST`. Parked branches: `pf-pair` (two prompt paths, deadlocks),
`draft-batch` (batched MTP draft, no VRAM).
