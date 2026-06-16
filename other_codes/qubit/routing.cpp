#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <limits>
#include <cmath>
#include <cstdint>
#include <string>
#include <algorithm>
#include <tuple>
// #include <omp.h>


// compile with:
// export PYBIND_INCLUDES=$(python3 -m pybind11 --includes) && export PYTHON_EXT=$(python3-config --extension-suffix)
// g++ -O3 -shared -fPIC -fopenmp -std=c++17 $PYBIND_INCLUDES routing.cpp -o routing${PYTHON_EXT}


const float INF = std::numeric_limits<float>::max();
constexpr float H_TOL = 1e-5f;   // tolerance for tie-breaking on heuristic H

// ---------------------------------------------------------------------------
// Output metric: routed depth (SWAP = 3)
// ---------------------------------------------------------------------------
// This file implements pure SABRE routing only. We return two outputs per
// trial: the total number of gates placed and the routed depth, both
// counted with the SwapGate-decomposition convention SWAP = 3 CX:
//   * Each input gate contributes 1 to num_gates and advances per-qubit
//     last_layer by 1 on the qubits it touches.
//   * Each SABRE-inserted SWAP contributes 3 to num_gates and advances
//     per-qubit last_layer by 3 on (p_a, p_b). This is exact: a SwapGate
//     decomposes to CX(a,b)*CX(b,a)*CX(a,b), three CX in series on the same
//     pair, so per-qubit layers advance by exactly 3 with no parallelism
//     loss between the three. The output therefore equals what Qiskit's
//     BasisTranslator produces from a SABRE-routed circuit when run with
//     translation kept and optimization/scheduling stripped.
//
// The post-TOS depth model that used to live here (KAK block consolidation
// with CAP_1Q Euler collapse) was removed because empirical measurement
// showed its per-mapping rank correlation with Qiskit's level-3 final depth
// was essentially zero — it approximated the magnitude well but not the
// rank. Modeling the rest of TOS faithfully requires per-gate identity and
// parameters, which we don't track here; the project now scopes the GA's
// objective to the simplified pipeline (SABRE + translation, no init / no
// optimization / no scheduling). If a faithful post-TOS fitness is ever
// needed, it would be the job of a Python-side post-routing pass that
// runs Qiskit's optimization stage on the C++-routed DAG.

namespace py = pybind11;

using namespace std;


// Build the gate DAG in a single pass over the gate list (already in topological order):
//   * successor_idx: flat [G*2] array. successor_idx[g*2 + slot] is the next gate id on
//     the qubit at `slot` of gate g (0 = q1 of g, 1 = q2 of g); -1 if none.
//   * required_predecessors: [G] array. Count of g's operand qubits that have a
//     predecessor gate — 0, 1, or 2, counted per qubit (not per distinct predecessor).
// Host-side; runs in O(G). Kept with vector inputs for convenience — called exactly once.
void precompute_dag (const vector<int>& gates_q1, const vector<int>& gates_q2, int n,
                     vector<int>& successor_idx, vector<int>& required_predecessors)
{
    const int G = (int) gates_q1.size();
    successor_idx.assign(G * 2, -1);
    required_predecessors.assign(G, 0);

    vector<int> last_touch_q(n, -1);

    for (int g = 0; g < G; ++g)
    {
        const int q1 = gates_q1[g];
        const int q2 = gates_q2[g];

        const int prev1 = last_touch_q[q1];
        if (prev1 != -1) {
            const int slot = (gates_q1[prev1] == q1) ? 0 : 1;
            successor_idx[prev1 * 2 + slot] = g;
            required_predecessors[g]++;
        }
        last_touch_q[q1] = g;

        if (q2 != -1)
        {
            const int prev2 = last_touch_q[q2];
            if (prev2 != -1) {
                const int slot = (gates_q1[prev2] == q2) ? 0 : 1;
                successor_idx[prev2 * 2 + slot] = g;
                required_predecessors[g]++;
            }
            last_touch_q[q2] = g;
        }
    }
}


// Lexicographic comparison of gates a and b on (q1, q2) — mirrors std::pair's operator<
// so we can keep the exact tie-break order while gates are stored as two parallel arrays.
static inline bool gate_lex_less (const int* gates_q1, const int* gates_q2, int a, int b)
{
    return (gates_q1[a] < gates_q1[b]) ||
           (gates_q1[a] == gates_q1[b] && gates_q2[a] < gates_q2[b]);
}


// --- Front-layer as a sorted compact list ---------------------------------
// The front layer F is represented as front_sorted[0..front_count) — gate ids
// kept sorted by Gate-lex order on (gates_q1[g], gates_q2[g]). Iteration gives
// deterministic tie-break behavior. Insert/erase are O(|F|) due to the shift;
// |F| is bounded by n, so this is cheap in practice.

// First index k such that front_sorted[k] does not compare less than gid
// (equivalent to std::lower_bound with our comparator).
static inline int front_lower_bound (const int* gates_q1, const int* gates_q2,
                                     const int* front_sorted, int front_count, int gid)
{
    int lo = 0, hi = front_count;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (gate_lex_less(gates_q1, gates_q2, front_sorted[mid], gid))
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// Caller guarantees gid is currently in the list.
static inline void front_erase (const int* gates_q1, const int* gates_q2,
                                int* front_sorted, int& front_count, int gid)
{
    const int pos = front_lower_bound(gates_q1, gates_q2, front_sorted, front_count, gid);
    for (int k = pos; k + 1 < front_count; ++k)
        front_sorted[k] = front_sorted[k + 1];
    --front_count;
}


// swap the logical qubits sitting on physical qubits p_a and p_b (O(1) via inverse map)
static inline void apply_swap (int* mapping, int* phys_to_logical, int p_a, int p_b)
{
    int q_a = phys_to_logical[p_a];
    int q_b = phys_to_logical[p_b];
    if (q_a != -1) mapping[q_a] = p_b;
    if (q_b != -1) mapping[q_b] = p_a;
    phys_to_logical[p_a] = q_b;
    phys_to_logical[p_b] = q_a;
}


// remap a single physical-qubit label under a candidate swap (p_a, p_b)
static inline int swap_phys (int p, int p_a, int p_b)
{
    return (p == p_a) ? p_b : ((p == p_b) ? p_a : p);
}


// xorshift32 — small-state PRNG used for random tie-breaking during swap selection,
// matching the standard SABRE / LightSABRE approach of uniformly sampling over the
// set of minimum-H candidate swaps. State must be non-zero (callers ensure this).
static inline uint32_t xorshift32 (uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}


// Build the lookahead (extended) set as a row-major adjacency:
//   E_adj[q * E_MAX_DEG + k]  — k-th partner of logical qubit q.
//   E_adj_count[q]            — number of valid partners for q.
//
// The BFS decrements predecessor counts to discover newly-executable successors.
// Instead of copying the full rp_live array (O(G)) at entry, we decrement
// rp_live in-place and record each decrement in `rp_undo` so we can restore on
// exit — typical BFS touches a few dozen gates, so this turns an O(G) pass into
// O(touched). Scratch buffers (to_visit, visit_now, rp_undo) and output buffers
// are supplied by the caller, pre-sized. Returns the total number of 2Q gates
// placed in the extended set.
int build_extended_front (const int* gates_q1, const int* gates_q2,
                          const int* successor_idx,
                          int* rp_live, int G,
                          const int* front_sorted, int front_count,
                          int n, int E_MAX_DEG,
                          int* rp_undo,
                          int* to_visit,
                          int* visit_now,
                          int* E_adj,
                          int* E_adj_count)
{
    (void) G; // rp_live bounds no longer referenced; keep parameter for symmetry
    const int ext_cap{20};

    std::fill(E_adj_count, E_adj_count + n, 0);

    int to_visit_n = 0;
    for (int k = 0; k < front_count; ++k)
        to_visit[to_visit_n++] = front_sorted[k];

    int i = 0;
    int visit_now_n = 0;
    int ext_size = 0;             // running count of 2Q gates in E
    int undo_n   = 0;             // number of decrements recorded for restore

    while (i < to_visit_n && ext_size < ext_cap)
    {
        visit_now[visit_now_n++] = to_visit[i];

        int j = 0;
        while (j < visit_now_n)
        {
            const int g = visit_now[j];

            // When both slots of g point to the same successor (that successor shares both
            // qubits with g), the two per-slot decrements match the per-qubit rp init; the
            // == 0 guard then adds the gate exactly once. Sort by Gate-lex first so iteration
            // order is deterministic.
            int succs[2] = { successor_idx[g * 2 + 0], successor_idx[g * 2 + 1] };
            if (succs[0] != -1 && succs[1] != -1 &&
                gate_lex_less(gates_q1, gates_q2, succs[1], succs[0]))
                std::swap(succs[0], succs[1]);

            for (int ki = 0; ki < 2; ++ki)
            {
                const int s = succs[ki];
                if (s == -1) continue;

                rp_undo[undo_n++] = s;
                if (--rp_live[s] == 0)
                {
                    const int sq2 = gates_q2[s];
                    if (sq2 != -1) // two-qubit gate
                    {
                        const int sq1 = gates_q1[s];

                        E_adj[sq1 * E_MAX_DEG + E_adj_count[sq1]++] = sq2;
                        E_adj[sq2 * E_MAX_DEG + E_adj_count[sq2]++] = sq1;

                        to_visit[to_visit_n++] = s;
                        ++ext_size;
                    }
                    else // single-qubit gate
                    {
                        visit_now[visit_now_n++] = s;
                    }
                }
            }
            j++;
        }
        visit_now_n = 0;
        i++;
    }

    // Restore rp_live to its caller-visible state.
    for (int k = 0; k < undo_n; ++k)
        ++rp_live[rp_undo[k]];

    return ext_size;
}


// LightSABRE-style relative scoring.
//
// Many inputs are p_a-invariant across the inner p_b loop in the caller
// (mapping[partner_a], E_adj row for q_a, decay[p_a], the precomputed 1/|F|
// and weight/|E| factors). The caller precomputes those once per outer SABRE
// iteration and passes them in via a small PaPrep struct so this function can
// stay focused on the p_b-dependent deltas.
struct PaPrep {
    int   q_a;                // phys_to_logical[p_a]
    int   partner_a;          // front_partner[q_a], always != -1
    int   p_partner_a;        // mapping[partner_a]
    int   dist_a_partner_a;   // dist[p_a * N + p_partner_a]  (pre-swap F term for q_a)
    int   e_cnt_a;            // E_adj_count[q_a]
    int   e_row_a;            // q_a * E_MAX_DEG
    float decay_a;            // decay[p_a]
    float H_base;             // score_F / |F| + weight * score_E / |E|
    float inv_front;          // 1 / |F|
    float inv_ext_weighted;   // weight / |E|   (or 0 if |E|==0)
};

static inline float Hdecay_relative (
    const int* dist, int N,
    const int* E_adj,
    const int* E_adj_count,
    int E_MAX_DEG,
    const int* mapping,
    const int* phys_to_logical,
    const int* front_partner,
    int p_a, int p_b,
    const float* decay,
    const PaPrep& pp)
{
    const int q_a = pp.q_a;
    const int q_b = phys_to_logical[p_b];   // may be -1 (p_b may be an unmapped physical qubit)

    // ---- delta for the basic (front-layer) component ----
    // q_a's F gate (all F gates are 2Q at swap-selection time).
    // p_partner_a is precomputed; its post-swap physical is inline-resolved.
    float delta_F;
    {
        const int p_partner_new = swap_phys(pp.p_partner_a, p_a, p_b);
        delta_F = (float) dist[p_b * N + p_partner_new] - (float) pp.dist_a_partner_a;
    }

    // q_b's F gate, if q_b is mapped and in F (and it's a distinct gate from q_a's)
    if (q_b != -1)
    {
        const int partner_b = front_partner[q_b];
        if (partner_b != -1 && partner_b != q_a)
        {
            const int p_partner = mapping[partner_b];
            const int p_partner_new = swap_phys(p_partner, p_a, p_b);
            delta_F += (float) dist[p_a * N + p_partner_new] - (float) dist[p_b * N + p_partner];
        }
    }

    // ---- delta for the extended (lookahead) component ----
    float delta_E = 0.0f;
    {
        const int cnt = pp.e_cnt_a;
        const int row = pp.e_row_a;
        for (int k = 0; k < cnt; ++k)
        {
            const int other = E_adj[row + k];
            const int p_other = mapping[other];
            const int p_other_new = swap_phys(p_other, p_a, p_b);
            delta_E += (float) dist[p_b * N + p_other_new] - (float) dist[p_a * N + p_other];
        }
    }

    if (q_b != -1)
    {
        const int cnt = E_adj_count[q_b];
        const int row = q_b * E_MAX_DEG;
        for (int k = 0; k < cnt; ++k)
        {
            const int other = E_adj[row + k];
            if (other == q_a) continue;   // edge (q_a, q_b) already counted from q_a's side
            const int p_other = mapping[other];
            const int p_other_new = swap_phys(p_other, p_a, p_b);
            delta_E += (float) dist[p_a * N + p_other_new] - (float) dist[p_b * N + p_other];
        }
    }

    // ---- assemble H_after * decay_max ----
    // H_base already incorporates the pre-swap score_F/|F| and weight*score_E/|E|;
    // we just add the deltas and multiply by the cached 1/|F| factors.
    const float H_after = pp.H_base + delta_F * pp.inv_front + delta_E * pp.inv_ext_weighted;
    return H_after * std::max(pp.decay_a, decay[p_b]);
}


// Shared read-only inputs for sabre_route_one: the circuit DAG, the device topology,
// and the scalar sizes. Built once per batch (see SABRE_routing_many) and passed by
// const reference to every per-mapping call. `dist` is borrowed from the caller's
// numpy buffer; all other arrays are owned.
struct SharedCtx
{
    std::vector<int> gates_q1;        // [G]
    std::vector<int> gates_q2;        // [G]
    std::vector<int> successor_idx;   // [2*G]
    std::vector<int> rp_init;         // [G]  — initial predecessor counts
    const int*       dist;            // [N*N], row-major, not owned
    std::vector<int> phys_nbr_off;    // [N+1] — CSR offsets into phys_nbr_list
    std::vector<int> phys_nbr_list;   // [sum of degrees] — flat device adjacency

    // Initial front layer, cached once per batch. Identical across all mappings
    // since it depends only on the circuit's DAG (rp_init), not on routing state.
    // front_sorted_init[0..front_init_count) is Gate-lex-sorted by (q1, q2).
    // front_partner_init[q] carries the logical partner of q in the initial F
    // (-1 if q isn't the 2Q side of any initial-F gate).
    std::vector<int> front_sorted_init;   // [front_init_count]
    std::vector<int> front_partner_init;  // [n]
    int              front_init_count;

    int G;                            // number of gates
    int n;                            // number of logical qubits
    int N;                            // number of physical qubits
    int max_deg;                      // max node degree in the device coupling graph
    int E_MAX_DEG;                    // per-qubit degree cap in the extended set (= 32)
    int release_threshold;            // release-valve threshold (max(4n, 20))
    int pending_cap;                  // pending-swap buffer capacity (release_threshold + 2)
    int cand_cap;                     // candidate-swap buffer capacity (n*max_deg + 1)
};


// Per-thread scratch storage for sabre_route_one. All buffers are sized once by the
// constructor from the caller's SharedCtx, and reused across every mapping the owning
// OpenMP thread routes.
struct Scratch
{
    std::vector<int>   last_layer;
    std::vector<float> decay;
    std::vector<int>   phys_to_logical;
    std::vector<int>   front_partner;
    std::vector<int>   rp_live;
    std::vector<int>   rp_undo;      // scratch for decrement-undo in build_extended_front
    std::vector<int>   pending_pa;
    std::vector<int>   pending_pb;
    std::vector<int>   gate_to_remove_gid;
    std::vector<int>   path;
    std::vector<int>   cand_pa;
    std::vector<int>   cand_pb;
    std::vector<int>   bfs_to_visit;
    std::vector<int>   bfs_visit_now;
    std::vector<int>   E_adj;
    std::vector<int>   E_adj_count;
    std::vector<int>   front_sorted;
    std::vector<char>  phys_in_F;
    std::vector<int>   phys_in_F_list;    // compact ascending list of phys_in_F==1 entries
    std::vector<int>   mapping_buf;       // mutable copy of a mapping passed to sabre_route_one

    explicit Scratch (const SharedCtx& ctx)
      : last_layer(ctx.N),
        decay(ctx.N),
        phys_to_logical(ctx.N),
        front_partner(ctx.n),
        rp_live(ctx.G),
        rp_undo(2 * ctx.G),           // upper bound: 2 successor-slots per gate
        pending_pa(ctx.pending_cap),
        pending_pb(ctx.pending_cap),
        gate_to_remove_gid(ctx.n),
        path(ctx.N),
        cand_pa(ctx.cand_cap),
        cand_pb(ctx.cand_cap),
        bfs_to_visit(ctx.G),
        bfs_visit_now(ctx.G),
        E_adj((size_t) ctx.n * ctx.E_MAX_DEG),
        E_adj_count(ctx.n),
        front_sorted(ctx.n),
        phys_in_F(ctx.N),
        phys_in_F_list(ctx.N),
        mapping_buf(ctx.n)
    {}
};


// Run SABRE-routing for a single initial mapping.
//
// Shared read-only inputs are bundled into `SharedCtx` and per-call mutable scratch
// into `Scratch`. Each invocation touches only its own `mapping` and `sc`, so multiple
// invocations can run concurrently with no shared mutable state.
void sabre_route_one (const SharedCtx& ctx,
                      int* mapping, uint32_t rng_seed,
                      Scratch& sc,
                      int* out_num_gates, int* out_depth)
{
    // --- Unpack ctx as raw pointer / scalar locals for terse body code. ---
    const int* gates_q1      = ctx.gates_q1.data();
    const int* gates_q2      = ctx.gates_q2.data();
    const int* successor_idx = ctx.successor_idx.data();
    const int* rp_init       = ctx.rp_init.data();
    const int* dist          = ctx.dist;
    const int* phys_nbr_off  = ctx.phys_nbr_off.data();
    const int* phys_nbr_list = ctx.phys_nbr_list.data();
    const int  G                 = ctx.G;
    const int  n                 = ctx.n;
    const int  N                 = ctx.N;
    const int  E_MAX_DEG         = ctx.E_MAX_DEG;
    const int  release_threshold = ctx.release_threshold;

    // --- Unpack scratch references (no copy — just aliases for readability). ---
    auto& last_layer         = sc.last_layer;
    auto& decay              = sc.decay;
    auto& phys_to_logical    = sc.phys_to_logical;
    auto& front_partner      = sc.front_partner;
    auto& rp_live            = sc.rp_live;
    auto& rp_undo            = sc.rp_undo;
    auto& pending_pa         = sc.pending_pa;
    auto& pending_pb         = sc.pending_pb;
    auto& gate_to_remove_gid = sc.gate_to_remove_gid;
    auto& path               = sc.path;
    auto& cand_pa            = sc.cand_pa;
    auto& cand_pb            = sc.cand_pb;
    auto& bfs_to_visit       = sc.bfs_to_visit;
    auto& bfs_visit_now      = sc.bfs_visit_now;
    auto& E_adj              = sc.E_adj;
    auto& E_adj_count        = sc.E_adj_count;
    auto& front_sorted       = sc.front_sorted;
    auto& phys_in_F          = sc.phys_in_F;
    auto& phys_in_F_list     = sc.phys_in_F_list;

    int num_gates = 0;
    const float increment = 1e-3f;
    constexpr int DECAY_RESET_PERIOD = 5;   // Matches Qiskit LightSABRE: decay
                                            // is reset to 1.0 every 5 swap
                                            // selections (sabre/route.rs).
    int num_search_steps = 0;

    // PRNG state for tie-breaking. xorshift32 requires a non-zero seed.
    uint32_t rng_state = (rng_seed != 0u) ? rng_seed : 1u;

    // Initialize per-call state.
    std::fill(last_layer.begin(),      last_layer.end(),      -1);
    std::fill(decay.begin(),           decay.end(),           1.0f);
    std::fill(phys_to_logical.begin(), phys_to_logical.end(), -1);

    for (int q = 0; q < n; ++q)
        phys_to_logical[mapping[q]] = q;

    // Live predecessor counts start from the DAG's initial counts.
    std::copy(rp_init, rp_init + G, rp_live.begin());

    // Initial front layer + partner map are circuit-only state, cached in ctx.
    // Copy into the per-call buffers rather than rebuilding with an O(G) scan.
    int front_count = ctx.front_init_count;
    std::copy(ctx.front_sorted_init.begin(),
              ctx.front_sorted_init.begin() + front_count,
              front_sorted.begin());
    std::copy(ctx.front_partner_init.begin(),
              ctx.front_partner_init.end(),
              front_partner.begin());

    // Release-valve state (LightSABRE, paper §II.7).
    // Swaps selected since the last routed gate are held pending: applied to `mapping`
    // but not yet committed to last_layer / num_gates. Committed on the next routed
    // gate, or reverted in place if the valve trips.
    int pending_count = 0;

    while (front_count > 0)
    {
        int gate_to_remove_count = 0;
        bool pending_committed = false;

        for (int k = 0; k < front_count; ++k)
        {
            const int g_id = front_sorted[k];
            const int qubit_1 = gates_q1[g_id];
            const int qubit_2 = gates_q2[g_id];

            int phys_qubit_1 = mapping[qubit_1];
            int phys_qubit_2 = (qubit_2 == -1) ? -1 : mapping[qubit_2];

            const bool routable = (qubit_2 == -1) || (dist[phys_qubit_1 * N + phys_qubit_2] == 1);
            if (!routable) continue;

            // Commit any pending SABRE swaps before placing this gate so that the
            // per-qubit last_layer reflects them when computing the gate's layer.
            // Each SWAP is counted as 3 in both num_gates and depth: a SwapGate
            // decomposes to CX(a,b)*CX(b,a)*CX(a,b) — three CX in series on the
            // same pair, so the per-qubit last_layer advances by exactly 3 and
            // the gate count grows by 3 per SWAP. This matches what Qiskit's
            // BasisTranslator produces when the translation stage is run.
            if (!pending_committed)
            {
                for (int i = 0; i < pending_count; ++i)
                {
                    const int pa = pending_pa[i], pb = pending_pb[i];
                    int li = std::max(last_layer[pa], last_layer[pb]) + 3;
                    last_layer[pa] = li;
                    last_layer[pb] = li;
                    num_gates += 3;
                }
                pending_count = 0;
                pending_committed = true;
            }

            if (qubit_2 == -1)
            {
                last_layer[phys_qubit_1] += 1;
                ++num_gates;
            }
            else
            {
                int li = std::max(last_layer[phys_qubit_1], last_layer[phys_qubit_2]) + 1;
                last_layer[phys_qubit_1] = li;
                last_layer[phys_qubit_2] = li;
                ++num_gates;
            }

            gate_to_remove_gid[gate_to_remove_count++] = g_id;
        }

        if (gate_to_remove_count > 0)
        {
            // Reset decay on every gate-routing event, mirroring Qiskit
            // LightSABRE (route.rs: `state.decay.fill(1.)` after each
            // `update_route` when the decay heuristic is enabled). This is
            // in addition to the every-DECAY_RESET_PERIOD-swaps periodic
            // reset further down. The effect on bias is small in practice
            // (decay is already bounded by the periodic reset), but keeping
            // both resets matches Qiskit's algorithm exactly.
            std::fill(decay.begin(), decay.end(), 1.0f);
            num_search_steps = 0;

            for (int i = 0; i < gate_to_remove_count; ++i)
            {
                const int g_id = gate_to_remove_gid[i];
                const int gq1  = gates_q1[g_id];
                const int gq2  = gates_q2[g_id];
                front_erase(gates_q1, gates_q2, front_sorted.data(), front_count, g_id);

                if (gq2 != -1)
                {
                    front_partner[gq1] = -1;
                    front_partner[gq2] = -1;
                }

                // Process successors via the precomputed DAG. When both slots of g_id point
                // to the same successor (the successor shares both qubits with g_id), the
                // two per-slot decrements match the per-qubit rp init; the == 0 guard adds
                // the gate exactly once. Sort by Gate-lex first for deterministic iteration.
                int succs[2] = { successor_idx[g_id * 2 + 0], successor_idx[g_id * 2 + 1] };
                if (succs[0] != -1 && succs[1] != -1 &&
                    gate_lex_less(gates_q1, gates_q2, succs[1], succs[0]))
                    std::swap(succs[0], succs[1]);

                for (int ki = 0; ki < 2; ++ki)
                {
                    const int s = succs[ki];
                    if (s == -1) continue;
                    if (--rp_live[s] == 0)
                    {
                        const int nq1 = gates_q1[s];
                        const int nq2 = gates_q2[s];
                        front_insert(gates_q1, gates_q2, front_sorted.data(), front_count, s);
                        if (nq2 != -1)
                        {
                            front_partner[nq1] = nq2;
                            front_partner[nq2] = nq1;
                        }
                    }
                }
            }
        }
        else if (pending_count > release_threshold)
        {
            // ---------- RELEASE VALVE (LightSABRE §II.7) ----------
            // 1. Revert pending swaps (each swap is self-inverse; reverse order cancels).
            for (int i = pending_count - 1; i >= 0; --i)
                apply_swap(mapping, phys_to_logical.data(), pending_pa[i], pending_pb[i]);
            pending_count = 0;

            // 2. Pick the F gate with the smallest current physical distance.
            int target_q1 = -1, target_q2 = -1;
            int min_d = std::numeric_limits<int>::max();
            for (int k = 0; k < front_count; ++k)
            {
                const int g_id = front_sorted[k];
                const int q1 = gates_q1[g_id];
                const int q2 = gates_q2[g_id];
                if (q2 == -1) continue;
                const int d = dist[mapping[q1] * N + mapping[q2]];
                if (d < min_d) { min_d = d; target_q1 = q1; target_q2 = q2; }
            }

            // 3. Reconstruct a shortest path via greedy distance-decrement.
            // Iterate only the actual neighbours of `cur` (CSR adjacency) rather
            // than scanning all N physical qubits per step — for heavy-hex / IBM-
            // style topologies the degree is ~3, so this is a big constant-factor
            // win when the release valve fires.
            const int p_start = mapping[target_q1];
            const int p_end   = mapping[target_q2];
            const int d = min_d;
            int path_len = 0;
            path[path_len++] = p_start;
            {
                int cur = p_start;
                for (int step = 0; step < d; ++step)
                {
                    const int remaining = d - step - 1;
                    const int nbr_lo = phys_nbr_off[cur];
                    const int nbr_hi = phys_nbr_off[cur + 1];
                    for (int k = nbr_lo; k < nbr_hi; ++k)
                    {
                        const int nxt = phys_nbr_list[k];
                        if (dist[nxt * N + p_end] == remaining)
                        {
                            path[path_len++] = nxt;
                            cur = nxt;
                            break;
                        }
                    }
                }
            }

            // 4. Apply swaps from both ends of the path so the two operands meet in the middle.
            //    These swaps are final (committed immediately via last_layer / num_gates).
            const int k = (d - 1) / 2;   // q1 at p_start takes k forward steps; q2 at p_end takes d-1-k
            for (int i = 0; i < k; ++i)
            {
                int pa = path[i], pb = path[i + 1];
                apply_swap(mapping, phys_to_logical.data(), pa, pb);
                // SWAP = 3 layers / 3 gates (CX·CX·CX in series on the same pair).
                int li = std::max(last_layer[pa], last_layer[pb]) + 3;
                last_layer[pa] = li;
                last_layer[pb] = li;
                num_gates += 3;
            }
            for (int j = 0; j < d - 1 - k; ++j)
            {
                int idx = d - j;
                int pa = path[idx], pb = path[idx - 1];
                apply_swap(mapping, phys_to_logical.data(), pa, pb);
                // SWAP = 3 layers / 3 gates (CX·CX·CX in series on the same pair).
                int li = std::max(last_layer[pa], last_layer[pb]) + 3;
                last_layer[pa] = li;
                last_layer[pb] = li;
                num_gates += 3;
            }

            // target_gate is now at distance 1; next outer iteration will route it.
            std::fill(decay.begin(), decay.end(), 1.0f);
            num_search_steps = 0;
        }
        else
        {
            // ---------- NORMAL SWAP SELECTION (LightSABRE relative scoring) ----------
            const int ext_size = build_extended_front(
                gates_q1, gates_q2, successor_idx, rp_live.data(), G,
                front_sorted.data(), front_count, n, E_MAX_DEG,
                rp_undo.data(), bfs_to_visit.data(), bfs_visit_now.data(),
                E_adj.data(), E_adj_count.data());
            const int front_size = front_count;

            float score_F = 0.0f;
            for (int k = 0; k < front_count; ++k)
            {
                const int g_id = front_sorted[k];
                score_F += (float) dist[mapping[gates_q1[g_id]] * N + mapping[gates_q2[g_id]]];
            }
            // score_E: skip qubits with no extended-set partners. Hoisting
            // mapping[q] out of the inner loop avoids the cache-miss chain on
            // qubits that wouldn't contribute anyway.
            float score_E = 0.0f;
            for (int q = 0; q < n; ++q)
            {
                const int cnt = E_adj_count[q];
                if (cnt == 0) continue;
                const int mq  = mapping[q];
                const int row = q * E_MAX_DEG;
                for (int kk = 0; kk < cnt; ++kk)
                    score_E += (float) dist[mq * N + mapping[E_adj[row + kk]]];
            }
            score_E *= 0.5f;   // each edge counted twice in E_adj

            // Precompute normalisers (H_base + inv_front + inv_ext_weighted are
            // constant across the entire swap-candidate loop).
            constexpr float weight = 0.5f;
            const float inv_front        = 1.0f / (float) front_size;
            const float inv_ext_weighted = (ext_size > 0) ? (weight / (float) ext_size) : 0.0f;
            const float H_base           = score_F * inv_front + score_E * inv_ext_weighted;

            float Hmin = INF;
            int cand_count = 0;

            // Build phys_in_F_list in ascending physical-qubit order in the same
            // pass that marks phys_in_F. The ascending order preserves
            // (seed, mapping)-reproducibility of the tie-break RNG. Iterating
            // this compact list avoids the O(N) outer scan on large backends.
            std::fill(phys_in_F.begin(), phys_in_F.end(), (char) 0);
            for (int k = 0; k < front_count; ++k)
            {
                const int g_id = front_sorted[k];
                phys_in_F[mapping[gates_q1[g_id]]] = 1;
                phys_in_F[mapping[gates_q2[g_id]]] = 1;
            }
            int phys_in_F_count = 0;
            for (int p = 0; p < N; ++p)
                if (phys_in_F[p]) phys_in_F_list[phys_in_F_count++] = p;

            for (int ia = 0; ia < phys_in_F_count; ++ia)
            {
                const int p_a = phys_in_F_list[ia];

                // Precompute everything the inner p_b loop needs that only
                // depends on p_a (and ext-loop invariants).
                PaPrep pp;
                pp.q_a               = phys_to_logical[p_a];
                pp.partner_a         = front_partner[pp.q_a];
                pp.p_partner_a       = mapping[pp.partner_a];
                pp.dist_a_partner_a  = dist[p_a * N + pp.p_partner_a];
                pp.e_cnt_a           = E_adj_count[pp.q_a];
                pp.e_row_a           = pp.q_a * E_MAX_DEG;
                pp.decay_a           = decay[p_a];
                pp.H_base            = H_base;
                pp.inv_front         = inv_front;
                pp.inv_ext_weighted  = inv_ext_weighted;

                const int nbr_lo = phys_nbr_off[p_a];
                const int nbr_hi = phys_nbr_off[p_a + 1];
                for (int k = nbr_lo; k < nbr_hi; ++k)
                {
                    const int p_b = phys_nbr_list[k];

                    // Canonicalise unordered pairs, matching Qiskit's
                    // route.rs:
                    //     if neighbor > phys || !front_layer.is_active(neighbor)
                    // — when both endpoints are in F, only emit the pair
                    // once (with p_b > p_a). Without this guard, every
                    // both-in-F pair appears twice in the candidate pool
                    // (once from each iteration direction), which biases
                    // the random tie-break toward both-in-F swaps over
                    // boundary swaps. Boundary swaps are the productive
                    // ones (they pull future gates from E into F), so
                    // doubling intra-F entries systematically inflates
                    // the SWAP count, especially on dense-connectivity
                    // circuits like QFT.
                    if (p_b < p_a && phys_in_F[p_b]) continue;

                    float H = Hdecay_relative(dist, N,
                                              E_adj.data(), E_adj_count.data(), E_MAX_DEG,
                                              mapping, phys_to_logical.data(), front_partner.data(),
                                              p_a, p_b, decay.data(), pp);

                    if (H < Hmin - H_TOL)
                    {
                        cand_pa[0] = p_a;
                        cand_pb[0] = p_b;
                        cand_count = 1;
                        Hmin = H;
                    }
                    else if (std::fabs(H - Hmin) <= H_TOL)
                    {
                        cand_pa[cand_count] = p_a;
                        cand_pb[cand_count] = p_b;
                        ++cand_count;
                    }
                }
            }

            // Standard SABRE tie-break: uniformly sample one of the cand_count swaps
            // that share the minimum H. xorshift32 is called once; modulo bias over
            // at most a few dozen candidates is negligible in practice.
            const int pick = (int)(xorshift32(rng_state) % (uint32_t) cand_count);
            const int swap_idx_1 = cand_pa[pick];
            const int swap_idx_2 = cand_pb[pick];

            // Decay update + periodic reset, matching Qiskit LightSABRE
            // (sabre/route.rs): every DECAY_RESET_PERIOD swap selections the
            // decay vector is reset to 1.0; otherwise we bump the two endpoints
            // of the chosen swap by `increment`. Keeps decay bounded over long
            // runs of consecutive swap selections (previously decay could grow
            // monotonically across many selections between gate-routings).
            ++num_search_steps;
            if (num_search_steps >= DECAY_RESET_PERIOD)
            {
                std::fill(decay.begin(), decay.end(), 1.0f);
                num_search_steps = 0;
            }
            else
            {
                decay[swap_idx_1] += increment;
                decay[swap_idx_2] += increment;
            }

            // Apply the swap to the current mapping and queue it; commit to last_layer
            // and num_gates only when the next gate actually routes.
            apply_swap(mapping, phys_to_logical.data(), swap_idx_1, swap_idx_2);
            pending_pa[pending_count] = swap_idx_1;
            pending_pb[pending_count] = swap_idx_2;
            ++pending_count;
        }
    }

    int depth = 0;
    for (int i = 0; i < N; ++i)
        if (last_layer[i] + 1 > depth)
            depth = last_layer[i] + 1;

    *out_num_gates  = num_gates;
    *out_depth      = depth;
}


// Host-side batch worker: runs the one-shot preprocessing (CSR adjacency, flatten
// circuit, gate DAG), then invokes sabre_route_one for each (mapping, trial) pair.
// The (P * num_trials) tasks are parallelised with OpenMP via `collapse(2)` on the
// nested (p, t) loops so load balance holds up at thread counts higher than P;
// each thread holds its own private scratch buffer, reused across tasks. The
// thread count is capped at min(P*num_trials, omp_get_max_threads()) so we
// don't spin up more workers than there is work.
//
// `mappings_data` points to a (P, n) row-major int array; mapping p is
// mappings_data[p*n .. p*n + n). The mappings are not mutated.
//
// `base_seed` is the starting point for the tie-break PRNG. Each mapping p uses seed
// (base_seed + p + 1), so (a) the full batch is reproducible given base_seed, and
// (b) different mappings get different tie-break streams even when base_seed is the
// same, which is needed to make random tie-breaking actually explore new paths.
//
// Returns a vector of (depth, num_gates) pairs, one per mapping (depth first,
// because depth is the primary objective metric for the GA fitness). Each
// input gate contributes 1 to both, and each inserted SWAP contributes 3 to
// both (CX-basis convention; see "Output metric" comment near the top of this
// file). Post-TOS effects (optimization, scheduling) are not modelled; for
// those, post-process the routed circuit on the Python side.
//
// `num_trials`: for each mapping, run SABRE `num_trials` times with distinct tie-
// break seeds and keep the trial with the smallest num_gates (i.e. fewest inserted
// SWAPs, since at SWAP = 3 the input-gate count is the same across trials and
// num_gates is monotonic in the SWAP count), matching Qiskit SabreSwap's trial-
// selection criterion (verified on Qiskit 1.2.4). The depth of that same winning
// trial is returned alongside num_gates, so the two outputs are internally
// consistent. Must be >= 1.
struct RoutingResult { int num_gates; int depth; };

std::vector<RoutingResult> SABRE_routing_many (
    const int* gates_flat, int num_gates_in,
    const int* dist, int N,
    int n, int P, const int* mappings_data, uint32_t base_seed,
    int num_trials, int num_threads)
{
    // --- Build the shared context (one-shot preprocessing). ---
    SharedCtx ctx;
    ctx.dist      = dist;
    ctx.N         = N;
    ctx.n         = n;
    ctx.E_MAX_DEG = 32;                           // safely exceeds ext_cap (20)

    // Device adjacency in CSR form (dist[i,j]==1 iff there is an edge i—j).
    ctx.phys_nbr_off.assign(N + 1, 0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (i != j && dist[i * N + j] == 1)
                ctx.phys_nbr_off[i + 1]++;
    for (int i = 0; i < N; ++i)
        ctx.phys_nbr_off[i + 1] += ctx.phys_nbr_off[i];

    ctx.phys_nbr_list.resize(ctx.phys_nbr_off[N]);
    {
        vector<int> cursor(ctx.phys_nbr_off.begin(), ctx.phys_nbr_off.begin() + N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (i != j && dist[i * N + j] == 1)
                    ctx.phys_nbr_list[cursor[i]++] = j;
    }

    ctx.max_deg = 0;
    for (int p = 0; p < N; ++p)
    {
        const int deg = ctx.phys_nbr_off[p + 1] - ctx.phys_nbr_off[p];
        if (deg > ctx.max_deg) ctx.max_deg = deg;
    }

    // Copy the flat (q1, q2) pairs into ctx's vectors. Validation of per-gate
    // qubit indices is done by the caller (who still has the GIL); we just
    // strip-mine the interleaved array into parallel arrays here.
    ctx.G = num_gates_in;
    ctx.gates_q1.resize(num_gates_in);
    ctx.gates_q2.resize(num_gates_in);
    for (int i = 0; i < num_gates_in; ++i) {
        ctx.gates_q1[i] = gates_flat[2 * i];
        ctx.gates_q2[i] = gates_flat[2 * i + 1];
    }

    precompute_dag(ctx.gates_q1, ctx.gates_q2, n, ctx.successor_idx, ctx.rp_init);

    // Matches Qiskit's LightSABRE attempt_limit = 10 * num_dag_qubits (see
    // crates/accelerate/src/sabre/heuristic.rs in Qiskit 1.2.4).
    ctx.release_threshold = 10 * n;
    ctx.pending_cap       = ctx.release_threshold + 2;
    ctx.cand_cap          = n * ctx.max_deg + 1;

    // Cache the initial front layer + partner map so each mapping just memcpy's
    // them (O(|F|), typically n or less) instead of scanning all G gates and
    // doing |F| lex-sorted insertions (O(G + F²)).
    ctx.front_sorted_init.resize(n);
    ctx.front_partner_init.assign(n, -1);
    ctx.front_init_count = 0;
    for (int g = 0; g < ctx.G; ++g)
    {
        if (ctx.rp_init[g] == 0)
            front_insert(ctx.gates_q1.data(), ctx.gates_q2.data(),
                         ctx.front_sorted_init.data(), ctx.front_init_count, g);
    }
    for (int k = 0; k < ctx.front_init_count; ++k)
    {
        const int g_id = ctx.front_sorted_init[k];
        const int q1   = ctx.gates_q1[g_id];
        const int q2   = ctx.gates_q2[g_id];
        if (q2 != -1) {
            ctx.front_partner_init[q1] = q2;
            ctx.front_partner_init[q2] = q1;
        }
    }

    std::vector<RoutingResult> results(P);

    // Thread count: honour the caller's request, but clamp to [1, max] and to
    // the total task count (P * num_trials). Task-level parallelism below is
    // finer-grained than mapping-level, so thread counts up to P*num_trials
    // are useful instead of capping at P.
    // const int max_threads = omp_get_max_threads();
    const int max_threads = 1; // Change by Jérôme Rouzé
    const long long total_tasks = (long long) P * (long long) num_trials;
    int nt = (num_threads <= 0) ? max_threads : num_threads;
    if (nt > max_threads)                nt = max_threads;
    if ((long long) nt > total_tasks)    nt = (int) total_tasks;
    if (nt < 1)                          nt = 1;

    // Flat buffer holding every trial's (num_gates, depth). Indexed as
    // trial_results[(p * num_trials + t) * 2 + k] for k in {0,1}. The
    // parallel region writes to disjoint slots (distinct (p,t) per iteration),
    // then a short serial reduction picks the min-num_gates trial per mapping.
    std::vector<int> trial_results((size_t) P * (size_t) num_trials * 2);

    #pragma omp parallel num_threads(nt)
    {
        // Per-thread scratch, constructed once at region entry and reused
        // across every (p, t) task the thread handles. Scratch state depends
        // only on circuit dimensions (constant across the batch), not on the
        // specific mapping or trial, so reuse is unconditional.
        Scratch sc(ctx);

        // collapse(2) fuses the (p, t) loops into a single 2000-task (for
        // P=100, num_trials=20) iteration space, giving OpenMP 20x more tasks
        // than the pure mapping-level loop. This is essential at 64+ threads:
        // with only P=100 mapping-level tasks there is not enough work to
        // keep all threads busy once some mappings finish early, so the
        // slowest mapping bottlenecks the whole generation. Per-(p,t) task
        // lets a fast mapping's threads steal trials from slower mappings.
        //
        // Dynamic scheduling with chunk 1 gives maximum load balance. The
        // OpenMP dispatch overhead per task is a few µs; amortised over ~1 ms
        // of sabre_route_one work per trial, that is negligible.
        #pragma omp for collapse(2) schedule(dynamic, 1)
        for (int p = 0; p < P; ++p)
        {
            for (int t = 0; t < num_trials; ++t)
            {
                // sabre_route_one mutates the mapping buffer in place, so we have
                // to re-seed it from the caller's original mapping each trial.
                std::copy(mappings_data + (size_t) p * n,
                          mappings_data + (size_t)(p + 1) * n,
                          sc.mapping_buf.begin());

                // Per-trial seed: base_seed + p*num_trials + t + 1. This gives every
                // (mapping, trial) pair a distinct stream, is fully reproducible from
                // base_seed+num_trials, and — usefully — reduces to the original
                // single-trial scheme (base_seed + p + 1) when num_trials == 1.
                const uint32_t rng_seed =
                    base_seed
                    + (uint32_t) p * (uint32_t) num_trials
                    + (uint32_t) t
                    + 1u;

                int num_gates = 0, depth = 0;
                sabre_route_one(ctx, sc.mapping_buf.data(), rng_seed, sc,
                                &num_gates, &depth);

                // Write to this (p, t)'s own slot — no contention, no atomic.
                const size_t base = ((size_t) p * (size_t) num_trials + (size_t) t) * 2;
                trial_results[base + 0] = num_gates;
                trial_results[base + 1] = depth;
            }
        }
    }

    // Serial reduction: per mapping, pick the trial with the smallest
    // num_gates (matching Qiskit's SabreSwap trial-selection criterion;
    // verified on Qiskit 1.2.4). Since every input gate contributes 1 to
    // num_gates and every inserted SWAP contributes 1, min(num_gates) ≡
    // min(swap_count) for a fixed circuit. The depth of the same winning
    // trial is returned alongside, so the two reported metrics come from
    // the same trial and stay internally consistent.
    //
    // Cost: P*num_trials comparisons, ~2000 for P=100 k=20 — trivial next
    // to the parallel compute.
    for (int p = 0; p < P; ++p)
    {
        int best_num_gates = std::numeric_limits<int>::max();
        int best_depth     = std::numeric_limits<int>::max();
        for (int t = 0; t < num_trials; ++t)
        {
            const size_t base = ((size_t) p * (size_t) num_trials + (size_t) t) * 2;
            const int ng = trial_results[base + 0];
            if (ng < best_num_gates)
            {
                best_num_gates = ng;
                best_depth     = trial_results[base + 1];
            }
        }
        results[p] = {best_num_gates, best_depth};
    }

    return results;
}


// Batch entry point: routes a population of P initial mappings against the same circuit
// and device, and returns two parallel 1D int32 numpy arrays of length P, in the order
// (depth, num_gates):
//   `depth[p]`     — routed depth in CX-basis layers, with parallelism: each input
//                    gate is 1 layer, each inserted SWAP is 3 layers (since
//                    SwapGate = CX(a,b)*CX(b,a)*CX(a,b) in series on the same pair).
//   `num_gates[p]` — total gate count after routing (input gates + 3 * inserted SWAPs).
// depth is returned first because it's the primary objective metric for the GA fitness.
// Post-TOS effects (optimization, scheduling) are not modelled; if you need them,
// post-process the routed circuit on the Python side.
//
// mappings_py: (P, n) int32 array; each row is one initial logical→physical mapping.
// The array is NOT mutated.
std::tuple<py::array_t<int>, py::array_t<int>>
circuit_routing_batch (py::array_t<int, py::array::c_style | py::array::forcecast> gates_py,
                       py::array_t<int> dist_py,
                       py::array_t<int, py::array::c_style | py::array::forcecast> mappings_py,
                       uint32_t seed,
                       int num_trials,
                       int num_threads)
{
    py::buffer_info dist_buf = dist_py.request();
    if (dist_buf.ndim != 2 || dist_buf.shape[0] != dist_buf.shape[1]) {
        throw std::runtime_error(
            "Distance matrix must be a 2D square array; got shape (" +
            (dist_buf.ndim >= 1 ? std::to_string(dist_buf.shape[0]) : std::string("?")) + ", " +
            (dist_buf.ndim >= 2 ? std::to_string(dist_buf.shape[1]) : std::string("?")) + ")."
        );
    }
    const int N = (int) dist_buf.shape[0];
    int* dist = static_cast<int*>(dist_buf.ptr);

    py::buffer_info map_buf = mappings_py.request();
    if (map_buf.ndim != 2) {
        throw std::runtime_error(
            "Mappings array must be 2D with shape (P, n); got ndim=" +
            std::to_string(map_buf.ndim) + "."
        );
    }
    const int P = (int) map_buf.shape[0];
    const int n = (int) map_buf.shape[1];
    const int* mappings_data = static_cast<const int*>(map_buf.ptr);

    if (n > N) {
        throw std::runtime_error(
            "Mappings width (" + std::to_string(n) +
            ") exceeds number of physical qubits (" + std::to_string(N) + ")."
        );
    }
    for (int p = 0; p < P; ++p) {
        for (int q = 0; q < n; ++q) {
            const int mpq = mappings_data[p * n + q];
            if (mpq < 0 || mpq >= N) {
                throw std::runtime_error(
                    "mappings[" + std::to_string(p) + "][" + std::to_string(q) +
                    "] = " + std::to_string(mpq) + " is out of range [0, " +
                    std::to_string(N) + ")."
                );
            }
        }
    }

    if (num_trials < 1) {
        throw std::runtime_error(
            "num_trials must be >= 1; got " + std::to_string(num_trials) + "."
        );
    }

    // Extract + validate the gates array while we still hold the GIL. After
    // this point, only raw int* into the numpy buffer is needed, and we can
    // release the GIL for the OpenMP-parallel compute below.
    py::buffer_info gates_buf = gates_py.request();
    if (gates_buf.ndim != 2 || gates_buf.shape[1] != 2) {
        throw std::runtime_error(
            "Gates array must be 2D with shape (num_gates, 2)."
        );
    }
    const int num_gates_in = static_cast<int>(gates_buf.shape[0]);
    const int* gates_flat  = static_cast<const int*>(gates_buf.ptr);
    for (int i = 0; i < num_gates_in; ++i) {
        const int q1 = gates_flat[2 * i];
        const int q2 = gates_flat[2 * i + 1];
        if (q1 < 0 || q1 >= n || (q2 != -1 && (q2 < 0 || q2 >= n))) {
            throw std::runtime_error(
                "Gate " + std::to_string(i) + " has out-of-range qubit index (q1=" +
                std::to_string(q1) + ", q2=" + std::to_string(q2) +
                "); expected 0 <= q < " + std::to_string(n) + " or q2 == -1."
            );
        }
    }

    // Release the GIL around the heavy compute. Our OpenMP region is pure C++
    // and numpy-buffer reads use raw pointers into memory that numpy won't
    // move; nothing calls back into Python, so releasing lets any concurrent
    // Python threads make progress. Re-acquired automatically when `release`
    // goes out of scope, in time for the output-array allocations below.
    std::vector<RoutingResult> results;
    {
        py::gil_scoped_release release;
        results = SABRE_routing_many(gates_flat, num_gates_in, dist, N, n, P,
                                     mappings_data, seed, num_trials, num_threads);
    }

    py::array_t<int> num_gates_out((size_t) P);
    py::array_t<int> depth_out((size_t) P);
    int* ng_ptr  = num_gates_out.mutable_data();
    int* dp_ptr  = depth_out.mutable_data();
    for (int p = 0; p < P; ++p) {
        ng_ptr[p]  = results[p].num_gates;
        dp_ptr[p]  = results[p].depth;
    }

    // Returned in (depth, num_gates) order — depth is the primary objective
    // metric for the GA fitness, so it goes first by convention.
    return {depth_out, num_gates_out};
}


PYBIND11_MODULE(routing, m) {
    m.doc() = "SABRE-based circuit routing in C++ using pybind11";
    m.def(
        "circuit_routing_batch",
        &circuit_routing_batch,
        "applies SABRE-routing to a circuit for P initial mappings (CPU, parallel); "
        "returns (depth[P], num_gates[P]). `depth` is the routed depth in "
        "CX-basis layers, with parallelism (each input gate adds 1 layer to the "
        "qubits it touches, each SWAP adds 3 layers since SwapGate = 3 CX in "
        "series). `num_gates` is the total gate count after routing (each input "
        "gate counts 1, each inserted SWAP counts 3). Post-TOS effects "
        "(optimization, scheduling) are not modelled; if you need them, run "
        "Qiskit's "
        "optimization/scheduling stages on the routed circuit on the Python "
        "side. `seed` seeds the random tie-breaker; same inputs + same seed "
        "reproduce the same outputs. `num_trials` runs SABRE that many times "
        "per mapping with distinct seeds and keeps the trial with the smallest "
        "num_gates (i.e. fewest inserted SWAPs), matching Qiskit's SabreSwap "
        "trial-selection criterion; the depth of that same winning trial is "
        "returned alongside. Defaults to 1 for backward compatibility; set to "
        "20 to match Qiskit's level-3 default. `num_threads` controls OpenMP "
        "parallelism over the P*num_trials (mapping, trial) tasks (0 = use all "
        "available cores); clamped to [1, min(P*num_trials, max_threads)].",
        py::arg("gates"),
        py::arg("dist"),
        py::arg("mappings"),
        py::arg("seed") = 42u,
        py::arg("num_trials") = 1,
        py::arg("num_threads") = 0
    );
}