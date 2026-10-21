# Strata NVFP4 — five-GPU fleet edition

**Qwen3.8-Flash-Next (125B hybrid MoE) with NVFP4 experts, spread across five consumer GPUs instead of squeezed
onto one.** A fork of [sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) (itself a fork of
[Niko1221/Strata](https://github.com/Niko1221/Strata)) that adds the multi-GPU work: expert tiers on the peer
cards, the output head split across all of them, tier prefill, NVMe session journaling, and mixed-generation
support. Measured on a 4× RTX 5060 Ti 16 GB rig and a mixed RTX 3090 + 3× 5060 Ti rig (9950X, 128 GB DDR5-5000,
driver 615.71.09 with the consumer-P2P patch, CUDA 13.3, Linux).

## What's new (22 October 2026): faster decode on the same five GPUs

Same rig and config of record as the 21 October edition below (RTX 3090 trunk + 4× RTX 5060 Ti tiers, 4 slots,
every expert on a GPU). Three rounds of decode work, each measured against the production build it replaced:

| decode, total tok/s | 1 stream | 3 streams | 4 streams |
|---|---|---|---|
| 21 October edition | ~124 | 150.3 | 153.6 |
| + 3090 kernels (K-split hyper-connection read, tensor-core Q8_0) | ~124 | 161.7 | 170.6 |
| + upstream decode ports (this release; one boot, live production 141.7 / 172.6 / 182.4) | **143.7** | **179.1** | **188.0** |

(The upstream ports were measured against that day's production build: 128.4 / 161.4 / 169.8 → 143.7 / 179.1 /
188.0, decode window 28.8 → 25.0 ms. The 16K-token prompt read is unchanged, 8.8-9.1 s.)

**1. The 3090's decode kernels** (`STRATA_GR_V6=5`, `STRATA_MMA_Q8=1`). The hyper-connection read splits each row's K
five ways, and the dense Q8_0 matvecs run on int8 tensor cores (`mma`), with the same kernel for every window width,
so a batch row never depends on its neighbours. 4 streams +10.8%, 16K prompt 9.01 → 8.82 s.

**2. The MTP draft head's attention** (`STRATA_MTP_ATTN_FAST=1`). The draft layer's attention splits its heads over
three blocks and merges only the chunks that hold cells: 1.88 → 1.55 ms a window (~+2% decode). Only the drafts use
it, so the verified tokens are unchanged.

**3. Upstream's decode kernels, ported** (see [docs/UPSTREAM-SYNC.md](docs/UPSTREAM-SYNC.md) for every commit
reviewed, ported or declined):
- NVFP4 experts decode each weight once per window (`Split<40>`), and the down rows run on 8 lanes. Always on, on
  the trunk and in the tiers' decode graphs. This is most of the gain. Outputs differ from the previous build at
  rounding level only.
- `STRATA_QSA_ROWS`, `STRATA_GU_SWIGLU`, `STRATA_ATTN_DAILY`, `STRATA_SIDE_MIXER=2`, `STRATA_HC_UPMIX`: upstream's
  bitwise sm_86 / sm_120 decode switches, on in the config of record.
- Built but off: `STRATA_QSA_SPLIT_ATTN` (no gain here) and `STRATA_PREFILL_X2_MIN_K=640` (FP4 prompt down
  products on the tiers: +1.4% prompt time here).

**4. The conversation journal writes deltas** (`STRATA_JOURNAL_DELTA=1`). A parked turn writes only the 1 MiB chunks
that changed against the deepest earlier file of the same conversation. A `.map` sidecar records every chunk's hash,
every chunk is verified at mount, and chains are at most four files deep. Real parks went 601 → 137 MiB and
1,046 → 208 MiB, and the rebuilt state is byte-identical (a tampered parent is refused). Parked conversations also stay
in RAM now (`--conversation-cache-mib 16384 --conversation-cache-slots 16`), and the boot warm-up stops once that
cache is full.

**5. The server.**
- `"thinking_policy"` in the run config gives every request one thinking setting, overriding the client on both the
  OpenAI and the Anthropic API. The config of record uses thinking on, effort medium, budget 32,768.
- The Monitor has a **Slots** card: each batch slot's state (reading / thinking / answering / tool call / idle) and
  its own speed. `/metrics` reports the same per slot.

**Tried and parked:** batched MTP drafts across slots (the 3090 has no VRAM left for their graphs), and reading two
chunks of one prompt at once (`pf-pair` branch: the two paths deadlock in a first-use probe).

## Earlier (21 October 2026): the fifth GPU - every expert on a GPU, 4 slots, the RAM copy gone

The production rig is now an **RTX 3090 trunk + 4× RTX 5060 Ti tiers** (each tier on a CPU-attached Gen5 x4 link;
9950X, 128 GB DDR5-5000). The fourth tier made room for every expert of the model on a GPU, and the work below turns
that into speed, concurrency and free RAM. Config of record: `configs/nvfp4-512k-np2-vision.json`.

| | 3 tiers (20 Oct, before) | 4 tiers, this edition |
|---|---|---|
| experts on a GPU | 20,446 of 24,576 | **24,576 of 24,576** |
| batch slots (concurrent conversations) | 3 | **4** |
| RAM available with the engine up | 12.7 GB | **65.7 GB** |
| a client decoding while another client's 16K prompt is read | 41.6 tok/s | **59.2 tok/s** |
| 16K-token prompt, time to first token | 9.62 s | **8.83-8.93 s** |
| 1K prompts, aggregate decode, 3 clients | 117.6 tok/s | **134.9 tok/s** |
| decode, all streams busy (sum / each) | 3: 146.7 / 49 | 3: **150.3 / 50**; 4: **153.6 / 38** |
| steady batch window, all slots decoding (both on 4 tiers) | 3 slots: ~183 tok/s | 4 slots: **~197 tok/s** |

**1. Four tiers and the head split across them.** `--peer-devices 1,2,3,4`. The verifier kept only three parts of
the split output head, so a fourth tier refused to boot ("the head parts do not reach the vocabulary's end"); the
part arrays now size to `kMaxPeerTiers` (8). `--head-split 0.7515` was swept 0-0.90 on four tiers: flat.

**2. The tiers run their share of a decode window themselves** (`STRATA_DEVICE_TIERS=2`). Each tier GPU gets one
graph per window shape that waits on the trunk's doorbell, plans its own entries on the device (a copy of its
residency table), computes them with the same grouped expert kernel, and sends **one weighted sum per token** into
the trunk's VRAM over P2P (summed in the token's own entry order, so a batch row never depends on its neighbours),
with its done word stored after the sums on the same path. The host pool no longer launches or waits for the tiers
(7.2 -> 0.15 ms of host time a window). 3-stream decode +1.5-2%; `=1` keeps the rows path (bit-identical to the pool).

**3. The lend shadow** (`STRATA_LEND_SHADOW=1`, with `--kv-resident 20480`). The prompt path borrows the trunk's last
cache slots for its buffers; their experts then sat on no GPU - streamed from RAM by the prompt path and computed on
the **CPU** by the decode windows beside the read. GPU 4 now keeps a copy of those slots: during a loan it claims
their experts (the prompt path, the pool and the tier graphs all route them there), and the loan is given back
from the copy over P2P. The smaller K/V window (the minimum, 20480 cells) gives the trunk ~228 more cache slots, so
GPU 4 holds that many fewer experts and fits the whole copy; it cost nothing measurable up to 172K tokens of
context. Decode beside a 16K read 53.7 -> 58.4 tok/s; the CPU experts in those windows 1.21 per layer-window -> 0.

**4. The expert RAM copy goes** (`--ram-budget 4`, the low-RAM mode, on a 128 GB machine). With every expert on a
GPU and the shadow, nothing reads the 63 GB RAM copy after boot - the serve path now counts it per request
("expert RAM copy since the first request: 0 blob reads"). The low-RAM mode fills the GPUs from the pack file at
boot and keeps 4 GB of experts in RAM: **0 MB of expert reads after boot**, the same outputs, the same speed.
In this mode the tiers need `--peer-reserve-mib 1024`: at 768 their prompt buffers no longer fit and the prompt
path falls back to the trunk alone (a 1K prompt's first token 0.88 -> 4.8 s; the boot log says so).

**5. A fourth slot** (`"parallel": 4`). The freed RAM carries its pinned K/V copy (~7 GB); its session takes
1.24 GB of the trunk's VRAM and ~500 experts move to GPU 4 - still 24,576 of 24,576 on the GPUs. With three
streams busy it changes nothing (150.3 vs 152.4 tok/s); with four it serves the fourth conversation at once instead
of queueing it (and evicting another conversation's slot to the journal every turn), at ~197 tok/s steady total.

**Measured and not taken:** more MTP proposals per batch slot (2: -17%, 3: -28%; acceptance 63 / 35 / 20%, a row
costs ~3 ms of trunk time); fixed placement without the adaptive tier (prompt reads +7.6% slower); moving the hot
experts between trunk and tiers (`STRATA_PEER_HOT` 0-0.7: flat below 0.45); an asynchronous batch commit (drafts
then accepted less: dropped); parallel slot drafts (exact, -0.5 ms a window, within noise; kept on a branch).
**Open:** long-context decode is ~4% lower in the low-RAM mode (88K: 102-105 vs 107-111 tok/s), cause not yet known.
**The ceiling now** is the trunk's own work per window row (~3.2 ms a row, ~8.8 ms fixed per window): a fourth
stream shares the same total. Raising it needs the trunk's dense work split across GPUs.

## Earlier (October 2026, RTX 3090 + 3× RTX 5060 Ti): concurrency and the multi-tier prompt path

Measured on the production rig (RTX 3090 trunk + 3× RTX 5060 Ti tiers), same session, the build with items 1-2
against the build with all six:

| | items 1-2 | all six |
|---|---|---|
| 16K-token prompt, time to first token | 14.3 s | **9.35 s** |
| 30K-token prompt, time to first token | 16.4 s | **11.6 s** |
| one 6,144-token prompt chunk on the trunk | 3.55 s (~1,730 tok/s) | **2.35 s (~2,600 tok/s)** |
| 1K prompts, aggregate decode, 1 / 2 / 3 clients | 81 / 94 / 102 tok/s | **91 / 98 / 119 tok/s** |
| 1K prompts, first token with 3 clients | 3.4 s | **2.2 s** |
| a client decoding while another client's 16K prompt is read | 35 tok/s (prompt first token 18.5 s) | **41 tok/s** (prompt first token **15.7 s**) |
| tier GPUs supported | 3 | **8** (a 3090 + 4× 5060 Ti fleet needs only `--peer-devices 1,2,3,4`) |

Prompt-path accuracy: the first-token logits after five prompts of 1.5K-30K tokens sit at KL 0.8-2.2 × 10⁻³ from
the previous build, with the same top token in 5 of 5 (the previous build against itself: 0).

**1. Batch slots branch inside the verify window** (`STRATA_BATCH_BRANCH=1`). With several clients, each slot's own
work in a window (GDN convolution and recurrence, the QSA KV append, scores, top-k and attention, and each slot's
commit chain) runs on its own stream inside the captured graph, forked and joined around the shared work, instead
of one slot after another. 3-slot window 28.7 → 26.9 ms, 3 clients 96.5 → 101.8 tok/s aggregate; token-identical
to the serial graph.

**2. Decode keeps running while a prompt is read** (`STRATA_BATCH_CONCURRENT=1`, `STRATA_BATCH_CONC_SHARE`). The
batch slots used to stop for the whole of a long prompt and get only a slice between chunks. Now they decode on
their own thread, on high-priority streams, beside the prompt chunks; `STRATA_BATCH_CONC_SHARE` (0.75 in the
config of record) adds an exclusive decode slice after each chunk. It engages for prompts longer than one chunk.
A client decoding while a 16K prompt is read: 25.7 → 33.7 tok/s, and the prompt's first token 19.3 → 17.8 s.

**3. Tiers no longer make the trunk copy their experts.** With several tiers, the prompt path's expert stream
skipped only the first tier's experts, so the trunk copied every expert of tiers 2 and 3 over its own PCIe link on
every chunk and threw them away. Fixing it is what makes `STRATA_PREFILL_CPU_SHARE_MAX=1024` (stream every
non-resident expert during the attention half) the fast setting again; the old config pinned 6145 to dodge it.

**4. Every tier streams its own share of the missing experts** (`STRATA_PF_TIER_STREAM`, default 1/(tiers+1)
each). Experts no card holds are copied from pinned RAM by the trunk and by every tier, each over its own link and
into its own ring, and computed where they land.

**5. Tiers send back one FP16 sum per token** (`STRATA_PF_TIER_SUMS`, on with two or more tiers). Instead of one
result row per routed expert (~18 GB a chunk through the trunk's x8 link), each tier adds its rows into a
weighted per-token sum and copies that over P2P (~4× less); a new combine kernel (`moe_combine_tiers16`) adds
the trunk's own rows and every tier's sum.

**6. The activations go out once** (`STRATA_PF_TIER_X16`, on with tier sums). The trunk writes each layer's MoE
input once, in FP16, to mapped host memory and every tier reads it over its own link, instead of one FP32 P2P copy
per tier through the trunk's link.

**7. Steady RAM with the conversation journal** (`STRATA_CONV_POOL_MIB`, 4096 in the config of record). Free RAM
used to fall through the day of 3-slot use (16 → 11 → 7 GB available) while the pinned K/V copy stayed the same
27.4 GB. The growth was the malloc heap. Every slot switch builds the whole parked conversation in RAM (up to
5 GB), writes it to the NVMe journal and frees it, and a mount reads one back the same way. glibc served those
16 MiB pieces from its heap and kept the freed memory: 6.7 GB after 13 hours (1,350 parks, 985 mounts),
measured with `malloc_trim` on the live engine. The images now come from a fixed pool of 16 MiB blocks, reserved
once at start and reused by every park and mount. Anything beyond the pool is mapped and returned to the kernel
when freed. Available RAM starts about 4 GB lower (the pool is paid up front) and stays there: over 35 parks and
30 mounts of 18K-token conversations the engine grew 0.17 GB, and an 18K park takes 284 ms (median, was 381 ms).
Decode, prompt and concurrency benches are unchanged (16K first token 9.26 s; 1K prompts, 2 / 3 clients 107 / 121
tok/s aggregate; solo decode 119-125 tok/s; a client decoding beside a 16K prompt 40 tok/s).

The prompt path also gained a working `STRATA_PREFILL_TIMING=1` with several tiers (the per-phase breakdown that
found items 3-6).

## What this fork adds

- **Peer expert tiers** — each peer card holds ~5,500-8,500 experts in its own VRAM and computes their rows
  during decode and prefill. With a 3090 in the fleet, 21,033 of the model's 24,576 experts (85.6%) live in
  VRAM; misses stream from pinned RAM.
- **Output-head split** — the 675 MB lm_head is read by every card in parallel over P2P during the MTP verify
  window. The merged pick is **bitwise-identical to the full-head pick** — proven by an in-boot verification
  harness (`STRATA_DBG_SPLIT_VERIFY=2`, 0 mismatches across np1/np2/mixed-generation runs), not argued.
- **Tier prefill + AVX-512 CPU share** — peers compute their experts' prompt rows per chunk (returns pipelined
  group-by-group on a second stream) while the 9950X computes a measured slice of non-resident experts.
- **NVMe conversation journaling (disk-direct)** — sessions park to NVMe at turn end with **zero RAM
  retention**; a restart remounts a 26K-token session in ~350 ms instead of re-prefilling. One config line
  (`--conversation-cache-mib 0`) keeps the whole journal out of pinned memory — see the RAM-claim lesson in
  [docs/HEAD-SPLIT-PORT.md](docs/HEAD-SPLIT-PORT.md).
- **Mixed-generation fleets** — an RTX 3090 (Ampere) as trunk or peer beside Blackwell cards: fat `86;120`
  build + the documented libcuda cross-generation P2P patch. Full recipe with measurements:
  [docs/RTX3090-MIXED-FLEET.md](docs/RTX3090-MIXED-FLEET.md).

## Measured (single client, temp 0, `bench-3k500` shapes)

| rig | prefill tok/s | decode tok/s | notes |
|---|---|---|---|
| 4× 5060 Ti 16 GB | 1,846-1,891 @14.5-54K prompt, 1,723-1,739 @195-212K | 94-113 | flat depth curve 14K→212K |
| 3090 trunk + 3× 5060 Ti (np3) | ~1,250-1,300 @3K, ~1,670 @57K | **132 solo** | trunk must sit on x8; peers fine on x4 |
| 3090 trunk + 3× 5060 Ti (np3), Oct 2026 build | **~2,600 per 6,144-token chunk**; 16K prompt 9.35 s, 30K 11.6 s | 117-121 solo | multi-tier prompt path, see *What's new* |

Concurrency on the np3 shape: 2 streams ~56-79 t/s each (~135 aggregate), 3 streams ~36-58 each (~139 aggregate).
(The October 2026 numbers in *What's new* use the `conc_bench` 1K-prompt shape, a different workload: 91 / 98 / 119
aggregate at 1 / 2 / 3 clients.)
The aggregate ceiling is the 3090 trunk and it is already saturated at two streams, so the third slot is nearly
free; its cost is expert cache (4,632 → ~3,150 slots, fleet residency 85.6% → ~80%). For comparison, the vLLM
golden lane on the same rig runs ~40 t/s per stream at two concurrent requests — strata's fan-out scales
concurrency better at low stream counts; TP scales prefill better.

Decode is trunk-bound: the aggregate ceiling (~105-110 t/s across concurrent streams on all-Blackwell) rises
only with a faster trunk card — which is why the 3090-as-trunk shape posts the best single-stream numbers.
Peer link width and peer ordering are **not** factors (measured: x8-peer at sequence 1, 2, 3 all within 1%).

## The model: stock, censored, verified

This lane runs the **original** Qwen3.8-Flash-Next NVFP4 by sergqwer's GPTQ
([Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata),
SHA-256 verified, 93.6% top-token agreement with the Q8_0 reference — the best of the NVFP4 quants).
The former `orca-nvfp4` default (OrcaRouter's abliterated build) was **withdrawn upstream** as "a much weaker
agent" (10-70 agent steps vs the original's 175 on the same task) and is not installed here. NVIDIA's stock
NVFP4 (`qwen-nvidia-nvfp4`) is the alternative if you prefer their quantizer.

## Requirements

- **GPUs:** four NVIDIA cards, 16 GB each (RTX 5060 Ti 16 GB reference build) — or a mixed fleet with one
  24 GB RTX 3090/4090 as trunk (x8 slot required; peers may sit on x4). Up to 8 tier cards beside the trunk
  (`--peer-devices`); the multi-tier prompt path was validated with 2 and 3 tiers. Blackwell cards run the FP4 native
  expert path; an Ampere trunk runs the sm_86 path (see the mixed-fleet doc). Driver 580+ (this rig:
  615.71.09 with the consumer-P2P kernel patch), CUDA 13.3, Linux. Without the P2P patch everything still
  runs over host-mapped transport — correct, slower.
- **Context by shape (this fork, multi-GPU):** `nvfp4-512k-np2-vision` = 524,288 tokens (YaRN) with vision
  across **np3** (3 concurrent slots; the filename predates the np3 adoption); `np4-262k` = 262,144 across np4. These are fleet numbers — upstream's single-card table
  (65,536 @16 GB) does not apply to a 4-card tier.
- **CPU:** x86-64 with AVX2 minimum; AVX-512 (Zen 4/5) unlocks the full CPU prefill share (measured +44% at
  fixed chunk); weaker CPUs' share auto-backs off.
- **RAM:** 64 GB minimum (upstream floor); 96-128 GB recommended for 512K context with the expert arena
  fully resident. NVMe strongly recommended for the conversation journal (parked sessions remount from it).

## Quick start

```
git clone https://github.com/johnmosesventura06/strata-nvfp4-4gpu
cd strata-nvfp4-4gpu
START-HERE.bat        (Linux: ./setup.sh)
```

Setup checks the PC, installs the engine and the model pack, and starts the server on `http://127.0.0.1:8080`.
For the multi-GPU fleet shapes, boot with `configs/nvfp4-512k-np2-vision.json` (production: np3, 512K context,
head split 0.7515, tier prefill, disk journal, `--conversation-cache-mib 0`) or `configs/np4-262k.json`
(np4 concurrency shape at 262K). Multi-GPU P2P needs the consumer-P2P kernel patch; without it the engine
still runs, falling back to host-mapped transport (correct, slower).

### Engine flags added in October 2026

| flag | config of record | what it does |
|---|---|---|
| `STRATA_BATCH_BRANCH` | `1` | batch slots' own work on their own streams inside the verify-window graph |
| `STRATA_BATCH_CONCURRENT` | `1` | the batch slots decode on their own thread while a prompt longer than one chunk is read |
| `STRATA_BATCH_CONC_SHARE` | `0.75` | exclusive decode time after each prompt chunk, as a fraction of the chunk's time |
| `STRATA_PREFILL_CPU_SHARE_MAX` | `1024` | chunks from this size up stream every non-resident expert during the attention half |
| `STRATA_PF_TIER_STREAM` | unset (1/(tiers+1)) | each tier's share of the streamed experts (`0`: tiers compute only the experts they hold) |
| `STRATA_PF_TIER_SUMS` | unset (on) | tiers send one FP16 sum per token back (`0`: their result rows) |
| `STRATA_PF_TIER_X16` | unset (on) | activations go out once in FP16 through host memory (`0`: one P2P copy per tier) |
| `STRATA_CONV_POOL_MIB` | `4096` | fixed RAM pool (MiB) for parked/mounted conversation images; beyond it images are mapped and returned on free (`0`: no pool) |
| `STRATA_PREFILL_TIMING` | unset | `1`: per-phase GPU timeline of each prompt read, trunk and first tier |
| `STRATA_GR_V6` | `5` | the hyper-connection read with each row's K split 5 ways (3090) |
| `STRATA_MMA_Q8` | `1` | dense Q8_0 decode matvecs on int8 tensor cores, one kernel for every window width |
| `STRATA_MTP_ATTN_FAST` | `1` | the MTP draft layer's attention split over blocks (drafts only) |
| `STRATA_JOURNAL_DELTA` | `1` | journal parks write only changed 1 MiB chunks against the conversation's deepest earlier file |
| `STRATA_QSA_ROWS` / `STRATA_GU_SWIGLU` / `STRATA_ATTN_DAILY` / `STRATA_HC_UPMIX` | `1` | upstream's bitwise decode / prompt kernels (docs/UPSTREAM-SYNC.md) |
| `STRATA_SIDE_MIXER` | `2` | single-stream windows: the mixer's input-only work on graph branches |
| `STRATA_QSA_SPLIT_ATTN` / `STRATA_PREFILL_X2_MIN_K` | unset | built, off: split tensor-core decode attention; FP4 prompt down products |

## Docs

- [docs/UPSTREAM-SYNC.md](docs/UPSTREAM-SYNC.md) — upstream review marker: the last upstream commits reviewed, what was ported, what was declined
- [docs/HEAD-SPLIT-PORT.md](docs/HEAD-SPLIT-PORT.md) — the head-split port ledger: design, verification
  harness, the record corrections (the early "+8.2%" headline was retracted after bracketed A/Bs; pooled
  gain is +2-4%, state-dependent), the RAM-claim drift mechanism, and the PLE repack retraction.
- [docs/RTX3090-MIXED-FLEET.md](docs/RTX3090-MIXED-FLEET.md) — mixed-generation recipe: fat build,
  libcuda patch, enumeration policy, measured role table, rollback paths.
- [docs/DISK-JOURNAL.md](docs/DISK-JOURNAL.md) — the NVMe conversation journal: disk-direct mode,
  self-healing re-pin, park/mount measurements.
- [docs/DETAILS.md](docs/DETAILS.md) / [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) — upstream's engine
  internals, still accurate for everything the forks share.

Upstream lineage: [Niko1221/Strata](https://github.com/Niko1221/Strata) →
[sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) → this fork. The parent README is kept as
[README.upstream.md](README.upstream.md).
