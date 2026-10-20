// src/core/peer_experts.cpp - see include/strata/core/peer_experts.hpp.
#include "strata/core/peer_experts.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

namespace strata::core {

static bool g_peer_portable = false;
void set_peer_portable(bool on) { g_peer_portable = on; }
bool peer_portable() { return g_peer_portable; }

namespace {

constexpr int64_t H = strata::kernels::cpu::H;
constexpr int64_t CAP = strata::kernels::cpu::MAXT * 10;   // entries per layer: MAXT tokens x top-10

struct Meta {                           // one block, uploaded with one copy per layer
    unsigned long long ptr[CAP];
    int32_t start[CAP + 1];
    int32_t dst[CAP];
    int32_t tok[CAP];
    int32_t count[4];                   // [0] groups, [1] entries
    int32_t out_row[CAP];               // multi-GPU: compact row -> entry row in the host `out`
};

/// Switches the calling thread to `dev` and back.  The engine's own thread lives on device 0.
struct On {
    int prev = 0;
    explicit On(int dev) { cudaGetDevice(&prev); if (prev != dev) cudaSetDevice(dev); }
    ~On() { int cur = 0; cudaGetDevice(&cur); if (cur != prev) cudaSetDevice(prev); }
};

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("peer experts: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

}  // namespace

PeerExperts::~PeerExperts() { close(); }

void PeerExperts::close() {
    if (device_ < 0) return;
    device_close();
    if (shadow_ != nullptr) { On on(device_); cudaFree(shadow_); shadow_ = nullptr; shadow_off_.clear(); }
    {
        On on(device_);
        if (stream_) cudaStreamSynchronize(stream_);
        if (refill_) cudaStreamSynchronize(refill_);
        cache_.close();
        if (d_x_) cudaFree(d_x_);
        if (d_out_) cudaFree(d_out_);
        if (d_meta_) cudaFree(d_meta_);
        if (d_q8_) cudaFree(d_q8_);
        if (d_scratch_) cudaFree(d_scratch_);
        if (h_x_) cudaFreeHost(h_x_);
        if (h_out_) cudaFreeHost(h_out_);
        if (h_meta_) cudaFreeHost(h_meta_);
        if (refill_ev_) cudaEventDestroy(refill_ev_);
        if (stream_) cudaStreamDestroy(stream_);
        if (refill_) cudaStreamDestroy(refill_);
    }
    d_x_ = d_out_ = h_x_ = h_out_ = nullptr;
    d_meta_ = h_meta_ = d_scratch_ = nullptr;
    d_q8_ = nullptr;
    stream_ = refill_ = nullptr;
    refill_ev_ = nullptr;
    device_ = -1;
}

bool PeerExperts::open(int device, const std::vector<std::pair<int32_t, int32_t>>& ranked, const ExpertCache& primary,
                       ExpertSource& src, int64_t n_layers, int64_t n_expert, int reserve_mib, int64_t max_slots,
                       std::string& err) {
    close();
    int count = 0;
    if (!ck(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err)) return false;
    if (device < 1 || device >= count) {
        err = "peer experts: CUDA device " + std::to_string(device) + " is not visible (" + std::to_string(count) +
              " devices; set CUDA_VISIBLE_DEVICES)";
        return false;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    src_ = &src;
    res_.assign((size_t) (n_layers * n_expert), kNotResident);
    {   // direct access both ways (NVLink or another P2P path): the prompt path copies activations and results over it
        int a = 0, b = 0;
        cudaDeviceCanAccessPeer(&a, 0, device);
        cudaDeviceCanAccessPeer(&b, device, 0);
        if (const char* np = std::getenv("STRATA_PEER_NO_P2P"); np != nullptr && std::atoi(np) != 0) a = 0;   // test knob: pretend the pair has no P2P
        p2p_ = a && b;
        if (p2p_) {
            On on0(0);
            cudaError_t e0 = cudaDeviceEnablePeerAccess(device, 0);
            if (e0 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e0 = cudaSuccess; }
            On on1(device);
            cudaError_t e1 = cudaDeviceEnablePeerAccess(0, 0);
            if (e1 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e1 = cudaSuccess; }
            p2p_ = e0 == cudaSuccess && e1 == cudaSuccess;
        }
    }
    On on(device);
    device_ = device;
    // scratch first, so the slots take what is really left
    int64_t ff = strata::kernels::cpu::FF;
    if (lay.native)
        for (const auto& f : lay.fmt) ff = std::max<int64_t>(ff, f.n_ff);
    const size_t scratch = std::max<size_t>((size_t) strata::kernels::moe_hit_grouped_scratch_bytes(CAP, H, ff),
                                            strata::kernels::native_expert_scratch_bytes(CAP, ff));
    const bool alloc_ok =
        ck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err) &&
        ck(cudaStreamCreateWithFlags(&refill_, cudaStreamNonBlocking), "refill stream", err) &&
        ck(cudaEventCreateWithFlags(&refill_ev_, cudaEventDisableTiming), "event", err) &&
        ck(cudaHostAlloc((void**) &h_x_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "input staging", err) &&
        ck(cudaHostAlloc((void**) &h_out_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "result staging", err) &&
        ck(cudaHostAlloc(&h_meta_, sizeof(Meta), cudaHostAllocPortable | cudaHostAllocMapped), "plan staging", err) &&
        ck(cudaMalloc((void**) &d_x_, (size_t) CAP * H * sizeof(float)), "input", err) &&
        ck(cudaMalloc((void**) &d_out_, (size_t) CAP * H * sizeof(float)), "result", err) &&
        ck(cudaMalloc(&d_meta_, sizeof(Meta)), "plan", err) &&
        ck(cudaMalloc((void**) &d_q8_, (size_t) CAP * (H / 32) * 36), "activations", err) &&
        ck(cudaMalloc(&d_scratch_, scratch), "scratch", err);
    if (!alloc_ok) { close(); return false; }

    // the pairs the primary does not hold, in rank order, as many as fit
    size_t free_b = 0, total_b = 0;
    if (!ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo", err)) { close(); return false; }
    const uint64_t reserve = (uint64_t) std::max(reserve_mib, 128) << 20;
    const uint64_t budget = free_b > reserve ? free_b - reserve : 0;
    std::vector<std::pair<int32_t, int32_t>> pick;
    std::vector<int64_t> sizes;
    uint64_t used = 0;
    for (const auto& pr : ranked) {
        if (primary.slot_of(pr.first, pr.second) >= 0) continue;
        const uint64_t b = lay.blob_bytes(pr.first);
        const uint64_t b256 = lay.native ? (b + 255) / 256 * 256 : lay.max_blob;
        if (used + b256 > budget) break;
        if (max_slots > 0 && (int64_t) pick.size() >= max_slots) break;
        used += b256;
        pick.push_back(pr);
        sizes.push_back((int64_t) b);
    }
    if (pick.empty()) { err = "peer experts: no room or no expert left for the peer"; close(); return false; }
    const bool opened = lay.native ? cache_.open_sized(sizes, n_layers, n_expert, err)
                                   : cache_.open((int64_t) pick.size(), n_layers, n_expert, (int64_t) lay.max_blob, err);
    if (!opened) { err = "peer experts: " + err; close(); return false; }
    // open() zeroes the arena with cudaMemset on the legacy stream, which is not ordered against the non-blocking
    // refill stream: wait for it, or the zeroing can land on top of the fills
    if (!ck(cudaDeviceSynchronize(), "arena zeroing", err)) { close(); return false; }
    // mapped reads: advise the next kAhead pairs so their reads overlap
    constexpr size_t kAhead = 256;
    const bool advise = src.advise_pairs(pick.data(), (int64_t) std::min(kAhead, pick.size()));
    for (size_t i = 0; i < pick.size(); ++i) {
        const auto& pr = pick[i];
        if (advise && i + kAhead < pick.size()) (void) src.advise_pairs(&pick[i + kAhead], 1);
        const int32_t slot = cache_.admit(pr.first, pr.second);
        const uint8_t* b = src.blob(pr.first, pr.second);
        if (slot < 0 || b == nullptr || !cache_.fill_slot(slot, b, refill_, err, (int64_t) lay.blob_bytes(pr.first))) {
            err = "peer experts: fill failed: " + err;
            close();
            return false;
        }
        res_[(size_t) (pr.first * n_expert + pr.second)] = slot;
    }
    if (!ck(cudaStreamSynchronize(refill_), "fill", err)) { close(); return false; }
    const auto& f0 = pick.front();
    if (!cache_.verify_slot(res_[(size_t) (f0.first * n_expert + f0.second)], src.blob(f0.first, f0.second), err,
                            (int64_t) lay.blob_bytes(f0.first))) {
        err = "peer experts: " + err;
        close();
        return false;
    }
    resident_ = (int64_t) pick.size();
    return true;
}

bool PeerExperts::launch(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                         const int32_t* kind, std::string& err, float* out) {
    static const bool direct_env = [] { const char* v = std::getenv("STRATA_PEER_DIRECT"); return v == nullptr || std::atoi(v) != 0; }();
    const bool direct = direct_env && out != nullptr;
    launched_direct_ = false;
    launched_rows_ = 0;
    row_of_.clear();
    const int64_t n = n_tok * k;
    if (n > CAP) { err = "peer experts: window too large"; return false; }
    Meta& m = *(Meta*) h_meta_;
    int groups = 0, rows = 0;
    // distinct experts in routing order; each one's entries become compact rows
    for (int64_t i = 0; i < n; ++i) {
        if (kind[i] != 2) continue;
        bool seen = false;
        for (int64_t j = 0; j < i; ++j)
            if (kind[j] == 2 && ids[j] == ids[i]) { seen = true; break; }
        if (seen) continue;
        const int32_t slot = res_[(size_t) (layer * n_expert_ + ids[i])];
        if (slot < 0) { err = "peer experts: an entry was planned for the peer but is not resident"; return false; }
        m.ptr[groups] = (unsigned long long) ptr_of(slot);   // a cache slot, or a claimed shadow slot
        m.start[groups] = rows;
        for (int64_t j = i; j < n; ++j)
            if (kind[j] == 2 && ids[j] == ids[i]) {
                m.dst[rows] = rows;
                m.tok[rows] = (int32_t) (j / k);
                m.out_row[rows] = (int32_t) j;
                row_of_.push_back((int32_t) j);
                ++rows;
            }
        ++groups;
        ++experts_;
    }
    if (groups == 0) return true;
    m.start[groups] = rows;
    m.count[0] = groups;
    m.count[1] = rows;
    entries_ += rows;
    launched_rows_ = rows;
    // direct: x is the verifier's pinned (portable) doorbell row block - copied from where it is
    if (!direct) std::memcpy(h_x_, x, (size_t) (n_tok * H) * sizeof(float));
    On on(device_);
    const cudaStream_t s = stream_;
    if (!ck(cudaMemcpyAsync(d_x_, direct ? x : h_x_, (size_t) (n_tok * H) * sizeof(float), cudaMemcpyHostToDevice, s), "input", err) ||
        !ck(cudaMemcpyAsync(d_meta_, h_meta_, sizeof(Meta), cudaMemcpyHostToDevice, s), "plan", err))
        return false;
    Meta* dm = (Meta*) d_meta_;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (lay.native) {
        strata::kernels::quantize_q8_1_rows(d_x_, n_tok, H, d_q8_, s);
        const auto& f = lay.fmt[(size_t) layer];
        const auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
        strata::kernels::native_expert_grouped(L, dm->ptr, dm->start, dm->count, dm->dst, dm->tok, groups, rows, d_q8_,
                                               d_scratch_, d_out_, s);
    } else {
        err = "peer experts: only native packs are supported";
        return false;
    }
    if (direct) {   // the rows straight into the host `out` (zero-copy, coalesced float4 writes)
        try {
            strata::kernels::scatter_rows_f32(d_out_, out, dm->out_row, rows, H, s);
        } catch (const std::exception& e) { err = std::string("peer experts: scatter: ") + e.what(); return false; }
        launched_direct_ = true;
        return true;
    }
    return ck(cudaMemcpyAsync(h_out_, d_out_, (size_t) rows * H * sizeof(float), cudaMemcpyDeviceToHost, s),
              "results", err);
}

bool PeerExperts::finish(float* out, std::string& err) {
    if (launched_rows_ == 0) return true;
    const auto t0 = std::chrono::steady_clock::now();
    {
        On on(device_);
        // spin on the stream (a blocking sync would sleep the pool thread and wake it late)
        cudaError_t e;
        while ((e = cudaStreamQuery(stream_)) == cudaErrorNotReady) {
        }
        if (!ck(e, "finish", err)) return false;
    }
    ms_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (launched_direct_) { launched_rows_ = 0; return true; }
    for (size_t r = 0; r < row_of_.size(); ++r)
        std::memcpy(out + (size_t) row_of_[r] * H, h_out_ + r * H, (size_t) H * sizeof(float));
    launched_rows_ = 0;
    return true;
}

bool PeerExperts::adapt(const float* usage, const int32_t* res0, int max_swaps, std::string& err) {
    if (!pending_.empty() || max_swaps <= 0) return true;
    const auto& lay = strata::kernels::cpu::expert_layout();
    struct Swap { float gain; int32_t layer, in, out; };
    std::vector<Swap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < n_layers_; ++l) {
        cand.clear();
        vict.clear();
        const float* u = usage + l * n_expert_;
        const int32_t* r0 = res0 + l * n_expert_;
        const int32_t* r1 = res_.data() + l * n_expert_;
        for (int32_t e = 0; e < (int32_t) n_expert_; ++e) {
            if (r1[e] >= 0 && r1[e] < (int32_t) resident_) vict.emplace_back(u[e], e);   // never a claimed shadow slot
            else if (r0[e] < 0 && u[e] >= 2.0f) cand.emplace_back(u[e], e);
        }
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + 1.5f) break;
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    if (swaps.empty()) return true;
    std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
    if ((int) swaps.size() > max_swaps) swaps.resize((size_t) max_swaps);
    On on(device_);
    for (const Swap& s : swaps) {
        const size_t in = (size_t) (s.layer * n_expert_ + s.in), out = (size_t) (s.layer * n_expert_ + s.out);
        const int32_t slot = res_[out];
        const uint8_t* b = src_->blob(s.layer, s.in);
        if (slot < 0 || b == nullptr) continue;
        if (!ck(cudaMemcpyAsync(cache_.device_slot(slot), b, (size_t) lay.blob_bytes(s.layer), cudaMemcpyHostToDevice,
                                refill_), "refill", err))
            return false;
        res_[out] = kNotResident;              // evicted now: the CPU computes it meanwhile
        pending_.emplace_back((int32_t) in, slot);
        ++swaps_;
        res_dev_dirty_ = true;                 // STRATA_DEVICE_TIERS: uploaded before the next window's graph
        ++res_version_;
    }
    return ck(cudaEventRecord(refill_ev_, refill_), "refill event", err);
}

void PeerExperts::apply_pending(bool wait) {
    if (pending_.empty()) return;
    {
        On on(device_);
        if (wait) cudaEventSynchronize(refill_ev_);
        else if (cudaEventQuery(refill_ev_) != cudaSuccess) return;
    }
    for (const auto& [i, slot] : pending_) res_[(size_t) i] = slot;
    pending_.clear();
    res_dev_dirty_ = true;
    ++res_version_;
}

// ---- STRATA_DEVICE_TIERS (see the header)

void PeerExperts::device_close() {
    if (device_ < 0) return;
    On on(device_);
    if (dstream_) cudaStreamSynchronize(dstream_);
    for (auto& g : dgraphs_) cudaGraphExecDestroy(g.exec);
    dgraphs_.clear();
    for (void* p : {(void*) d_res_, (void*) d_slot_off_, (void*) d_plan_, (void*) dx_, (void*) dout_, (void*) dq8_, dscratch_,
                    (void*) d_counter_})
        if (p != nullptr) cudaFree(p);
    if (dstream_) cudaStreamDestroy(dstream_);
    d_res_ = nullptr; d_slot_off_ = nullptr; d_plan_ = nullptr; dx_ = dout_ = nullptr; dq8_ = nullptr; dscratch_ = nullptr;
    d_counter_ = nullptr;
    dstream_ = nullptr;
    dev_ready_ = false;
}

bool PeerExperts::device_open(int max_t, int k, std::string& err) {
    if (device_ < 0) { err = "peer experts: the device path needs an open tier"; return false; }
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native) { err = "peer experts: the device path needs a native pack"; return false; }
    if (max_t < 1 || k < 1 || (long long) max_t * k > 128 || (long long) max_t * k > CAP) {
        err = "peer experts: the device path takes windows of at most 128 entries";
        return false;
    }
    device_close();
    On on(device_);
    dev_max_t_ = max_t;
    dev_k_ = k;
    dev_capx_ = (long long) max_t * k;
    const long long capx = dev_capx_;
    const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
    const size_t plan_i32 = (size_t) (ptr_off + 4 * capx + capx + 1 + 16);
    int64_t ff = strata::kernels::cpu::FF;
    for (const auto& f : lay.fmt) ff = std::max<int64_t>(ff, f.n_ff);
    int prio_lo = 0, prio_hi = 0;
    cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi);
    const uint64_t* off = cache_.slot_offsets();
    const bool ok =
        ck(cudaStreamCreateWithPriority(&dstream_, cudaStreamNonBlocking, prio_hi), "device stream", err) &&
        ck(cudaMalloc((void**) &d_res_, res_.size() * sizeof(int32_t)), "device residency", err) &&
        ck(cudaMalloc((void**) &d_plan_, plan_i32 * sizeof(int32_t)), "device plan", err) &&
        ck(cudaMalloc((void**) &dx_, (size_t) max_t * H * sizeof(float)), "device input", err) &&
        ck(cudaMalloc((void**) &dout_, (size_t) capx * H * sizeof(float)), "device result", err) &&
        ck(cudaMalloc((void**) &dq8_, (size_t) max_t * (H / 32) * 36), "device activations", err) &&
        ck(cudaMalloc(&dscratch_, strata::kernels::native_expert_scratch_bytes(capx, ff)), "device scratch", err) &&
        ck(cudaMalloc((void**) &d_counter_, sizeof(unsigned)), "device counter", err) &&
        ck(cudaMemset(d_counter_, 0, sizeof(unsigned)), "device counter", err) &&
        ck(cudaMalloc((void**) &d_slot_off_, (size_t) (resident_ + shadow_slots()) * sizeof(unsigned long long)), "slot offsets", err) &&
        ck(cudaMemcpy(d_res_, res_.data(), res_.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "residency", err);
    if (!ok) { device_close(); return false; }
    d_base_ = cache_.device_slot(0);   // slot 0 sits at offset 0 either way
    d_blob_ = 0;
    {   // every slot's offset from the arena base - the cache's, then the shadow's (a separate allocation: the
        // difference wraps in 64 bits, and base + offset is the shadow slot's address again)
        std::vector<unsigned long long> offs((size_t) (resident_ + shadow_slots()));
        for (int64_t s = 0; s < resident_; ++s) offs[(size_t) s] = off != nullptr ? off[s] : cache_.slot_offset(s);
        for (int64_t j = 0; j < shadow_slots(); ++j)
            offs[(size_t) (resident_ + j)] = (unsigned long long) (uintptr_t) shadow_slot(j) - (unsigned long long) (uintptr_t) d_base_;
        if (!ck(cudaMemcpy(d_slot_off_, offs.data(), offs.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice),
                "slot offsets", err)) { device_close(); return false; }
    }
    res_dev_dirty_ = false;
    dev_ready_ = true;
    return true;
}

bool PeerExperts::device_launch(const DeviceWindow& w, std::string& err) {
    if (!dev_ready_) { err = "peer experts: the device path is not open"; return false; }
    if (w.T < 1 || w.T > dev_max_t_ || w.k != dev_k_ || w.G < 1 || w.G > 2 || w.le <= w.lb ||
        w.seq == nullptr || w.x == nullptr || w.ids == nullptr || w.rows == nullptr || w.done == nullptr ||
        (w.sum_mode && (w.w == nullptr || w.sums == nullptr))) {
        err = "peer experts: a device window out of range";
        return false;
    }
    On on(device_);
    if (res_dev_dirty_) {   // pageable source: the copy takes its snapshot before the call returns
        if (!ck(cudaMemcpyAsync(d_res_, res_.data(), res_.size() * sizeof(int32_t), cudaMemcpyHostToDevice, dstream_),
                "residency upload", err))
            return false;
        res_dev_dirty_ = false;
    }
    cudaGraphExec_t exec = nullptr;
    for (const DevGraph& g : dgraphs_)
        if (g.T == w.T && g.G == w.G && g.lb == w.lb && g.le == w.le && g.seq == (const void*) w.seq) { exec = g.exec; break; }
    if (exec == nullptr) {
        using namespace strata::kernels;
        const auto& lay = cpu::expert_layout();
        const cudaStream_t s = dstream_;
        const long long capx = dev_capx_;
        const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        const int32_t* counts = d_plan_;
        const int32_t* start = d_plan_ + 4;
        const int32_t* dst = start + capx + 1;
        const int32_t* tok = dst + capx;
        const unsigned long long* ptr = (const unsigned long long*) (d_plan_ + ptr_off);
        if (!ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture begin", err)) return false;
        try {
            for (int64_t step = 0; step < (w.le - w.lb) * w.G; ++step) {
                const int64_t l = w.lb + step / w.G;
                const int grp = (int) (step % w.G);
                const int tb = w.gtb[grp], n = w.gte[grp] - w.gtb[grp];
                const uint32_t ring = (uint32_t) (step + 1);
                wait_flag_ge(w.seq, ring, s);
                if (n > 0) {
                    const int ne = n * w.k;
                    copy_from_mapped(dx_, w.x + (size_t) tb * H, (int64_t) n * H, s);
                    tier_plan(w.ids + (size_t) tb * w.k, ne, w.k, d_res_ + l * n_expert_, (int) n_expert_, d_base_,
                              d_slot_off_, d_blob_, d_plan_, capx, s,
                              w.main_res != nullptr ? w.main_res + l * n_expert_ : nullptr);
                    quantize_q8_1_rows(dx_, n, H, dq8_, s);
                    const auto& f = lay.fmt[(size_t) l];
                    const auto L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
                    native_expert_grouped(L, ptr, start, counts, dst, tok, ne, ne, dq8_, dscratch_, dout_, s);
                    if (w.sum_mode) {   // the sums into the main's VRAM, the done word by the kernel's last block
                        tier_sums(dout_, dst, tok, counts + 1, w.w + (size_t) tb * w.k, n, w.k, H,
                                  w.sums + (size_t) tb * H, w.done, ring, d_counter_, s);
                        continue;
                    }
                    scatter_rows_counted(dout_, w.rows + (size_t) tb * w.k * H, dst, counts + 1, ne, H, s);
                }
                mapped_store(w.done, ring, s);
            }
        } catch (const std::exception& e) {
            cudaGraph_t g = nullptr;
            cudaStreamEndCapture(s, &g);
            if (g) cudaGraphDestroy(g);
            err = std::string("peer experts: device window capture: ") + e.what();
            return false;
        }
        cudaGraph_t graph = nullptr;
        if (!ck(cudaStreamEndCapture(s, &graph), "capture end", err)) return false;
        const cudaError_t ie = cudaGraphInstantiateWithFlags(&exec, graph, 0);
        cudaGraphDestroy(graph);
        if (!ck(ie, "graph instantiate", err)) return false;
        dgraphs_.push_back({w.T, w.G, w.lb, w.le, (const void*) w.seq, exec});
    }
    return ck(cudaGraphLaunch(exec, dstream_), "device window launch", err);
}


// ---- STRATA_LEND_SHADOW (see the header)

uint64_t PeerExperts::free_bytes() const {
    if (device_ < 0) return 0;
    On on(device_);
    size_t f = 0, t = 0;
    return cudaMemGetInfo(&f, &t) == cudaSuccess ? (uint64_t) f : 0;
}

bool PeerExperts::shadow_open(const std::vector<int64_t>& slot_bytes, int reserve_mib, std::string& err) {
    if (device_ < 0 || shadow_ != nullptr || dev_ready_) { err = "peer experts: shadow_open needs an open tier, once, before device_open"; return false; }
    std::vector<uint64_t> off(slot_bytes.size());
    uint64_t total = 0;
    for (size_t j = 0; j < slot_bytes.size(); ++j) { off[j] = total; total += ((uint64_t) slot_bytes[j] + 255) / 256 * 256; }
    const uint64_t reserve = (uint64_t) std::max(reserve_mib, 0) << 20;
    if (total + reserve > free_bytes()) {
        err = "peer experts: the shadow (" + std::to_string(total >> 20) + " MiB) does not fit beside the reserve";
        return false;
    }
    On on(device_);
    if (!ck(cudaMalloc((void**) &shadow_, total), "shadow", err)) { shadow_ = nullptr; return false; }
    shadow_off_ = std::move(off);
    return true;
}

void PeerExperts::shadow_claim(int64_t layer, int64_t expert, int64_t j) {
    int32_t& r = res_[(size_t) (layer * n_expert_ + expert)];
    if (r >= 0) return;   // the tier holds it already (never the case for a lent slot's expert)
    r = (int32_t) (resident_ + j);
    res_dev_dirty_ = true;
    ++res_version_;
}

void PeerExperts::shadow_release(int64_t layer, int64_t expert, int64_t j) {
    int32_t& r = res_[(size_t) (layer * n_expert_ + expert)];
    if (r != (int32_t) (resident_ + j)) return;
    r = kNotResident;
    res_dev_dirty_ = true;
    ++res_version_;
}

}  // namespace strata::core
