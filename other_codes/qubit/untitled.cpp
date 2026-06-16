
//phisic m, N
//logic n, d
//P batch size, 
//mappings data, partial permutation
// 
std::vector<RoutingResult> SABRE_routing_many (
    const int* gates_flat, int num_gates_in,
    const int* dist, int N,
    int n, const int P = 1, const int* mappings_data, 
    int num_trials = 20, int num_threads)
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