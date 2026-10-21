// journal_delta_check PARENT.bin CHILD.bin WORKDIR - the delta journal on real files, CPU only:
//   the parent copied into WORKDIR as a whole file + map, the child written there as a delta against it, read back
//   through session_file_read (the mount's reader: chain rebuild + chunk verification), then compared byte for byte
//   with the child's own whole file.  Negative control: one byte of the parent copy changed -> the delta must refuse.
#include "strata/core/conversation_file.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
using namespace strata::core;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
static bool same_buf(const ConversationBuffer& a, const ConversationBuffer& b) {
    if (a.size() != b.size()) return false;
    std::vector<uint8_t> x(1 << 20), y(1 << 20);
    for (size_t at = 0; at < a.size(); at += x.size()) {
        const size_t n = std::min(x.size(), a.size() - at);
        a.visit(at, n, [&](const uint8_t* p, size_t c, size_t o) { std::memcpy(x.data() + (o - at), p, c); return true; });
        b.visit(at, n, [&](const uint8_t* p, size_t c, size_t o) { std::memcpy(y.data() + (o - at), p, c); return true; });
        if (std::memcmp(x.data(), y.data(), n) != 0) return false;
    }
    return true;
}
static bool same_ck(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used;
}
static int compare(const SavedConversation& a, const SavedConversation& b) {
    int bad = 0;
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec) { std::printf("  header differs\n"); ++bad; }
    if (!same_ck(a.live, b.live)) { std::printf("  live state differs\n"); ++bad; }
    if (a.checkpoints.size() != b.checkpoints.size()) { std::printf("  checkpoint count differs\n"); ++bad; }
    else for (size_t k = 0; k < a.checkpoints.size(); ++k) if (!same_ck(a.checkpoints[k], b.checkpoints[k])) { std::printf("  checkpoint %zu differs\n", k); ++bad; }
    if (a.kv.size() != b.kv.size()) { std::printf("  K/V layer count differs\n"); ++bad; }
    else for (size_t l = 0; l < a.kv.size(); ++l) {
        const auto &x = a.kv[l], &y = b.kv[l];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim || x.page_size != y.page_size ||
            x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim || !same_buf(x.k, y.k) || !same_buf(x.v, y.v) ||
            !same_buf(x.k_scale, y.k_scale) || !same_buf(x.v_scale, y.v_scale) || !same_buf(x.pooled, y.pooled)) {
            std::printf("  K/V layer %zu differs\n", l); ++bad;
        }
    }
    return bad;
}
int main(int argc, char** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: journal_delta_check PARENT.bin CHILD.bin WORKDIR\n"); return 2; }
    const std::string pin = argv[1], cin = argv[2], dir = argv[3];
    SessionFileIdentity id;
    {   // the identity the files were written with (header bytes 16 and 24)
        FILE* f = std::fopen(cin.c_str(), "rb");
        unsigned char h[64] = {};
        if (!f || std::fread(h, 1, 64, f) != 64) { std::fprintf(stderr, "cannot read %s\n", cin.c_str()); return 2; }
        std::fclose(f);
        std::memcpy(&id.model, h + 16, 8);
        std::memcpy(&id.config, h + 24, 8);
    }
    SessionReadLimits lim;
    SavedConversation P, C;
    size_t pb = 0, cb = 0;
    std::string e;
    auto t = Clock::now();
    if (!session_file_read(pin, id, P, pb, e, lim)) { std::fprintf(stderr, "parent: %s\n", e.c_str()); return 1; }
    if (!session_file_read(cin, id, C, cb, e, lim)) { std::fprintf(stderr, "child: %s\n", e.c_str()); return 1; }
    std::printf("read parent %zu tokens (%.0f MiB) and child %zu tokens (%.0f MiB) in %.0f ms\n", P.live.ids.size(), pb / 1048576.0,
                C.live.ids.size(), cb / 1048576.0, ms(t));
    if (P.live.ids.size() >= C.live.ids.size() || !std::equal(P.live.ids.begin(), P.live.ids.end(), C.live.ids.begin())) {
        std::fprintf(stderr, "the parent is not a prefix of the child\n"); return 2;
    }
    std::filesystem::create_directories(dir);
    const std::string pp = dir + "/conv-parent.bin", cp = dir + "/conv-child.bin";
    const uint64_t chunk = 1 << 20;
    SessionChunkMap pm, cm;
    t = Clock::now();
    session_chunk_map_build(P, chunk, pm);
    session_chunk_map_build(C, chunk, cm);
    const double hash_ms = ms(t);
    size_t w = 0, dw = 0;
    if (!session_file_write(pp, P, id, w, e) || !session_chunk_map_write(pp + ".map", pm, e)) { std::fprintf(stderr, "write parent: %s\n", e.c_str()); return 1; }
    cm.depth = 1; cm.parent = "conv-parent.bin";
    t = Clock::now();
    if (!session_file_write_delta(cp, C, id, cm.parent, pm, cm, dw, e) || !session_chunk_map_write(cp + ".map", cm, e)) { std::fprintf(stderr, "write delta: %s\n", e.c_str()); return 1; }
    const double write_ms = ms(t);
    std::printf("child as a delta: %.1f MiB instead of %.1f MiB (%.1fx less), written in %.0f ms; hashing both images %.0f ms\n",
                dw / 1048576.0, cb / 1048576.0, (double) cb / (double) dw, write_ms, hash_ms);
    SavedConversation R;
    size_t rb = 0;
    t = Clock::now();
    if (!session_file_read(cp, id, R, rb, e, lim)) { std::fprintf(stderr, "read delta: %s\n", e.c_str()); return 1; }
    std::printf("delta mounted (parent + delta + verification) in %.0f ms\n", ms(t));
    const int bad = compare(R, C);
    std::printf("rebuilt vs the child's whole file: %s\n", bad ? "DIFFERENT" : "IDENTICAL (every token, state array and K/V byte)");
    {   // negative control: the parent copy altered by one byte in its K/V
        SavedConversation Q;
        size_t qb = 0;
        session_file_read(pp, id, Q, qb, e, lim);
        if (!Q.kv.empty() && Q.kv[0].k.size() > 0) Q.kv[0].k.visit(0, 1, [](uint8_t* p, size_t, size_t) { p[0] ^= 0x5a; return true; });
        session_file_write(pp, Q, id, w, e);   // same name, same map: the chain is now wrong
        SavedConversation X;
        e.clear();
        const bool read = session_file_read(cp, id, X, rb, e, lim);
        std::printf("negative control (parent altered by one byte): %s\n", read ? "NOT DETECTED - FAIL" : ("refused: " + e).c_str());
        if (read) return 1;
    }
    std::filesystem::remove_all(dir);
    return bad ? 1 : 0;
}
