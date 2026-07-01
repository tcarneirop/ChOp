#ifndef QUEENS_ENUMERATION_HPP
#define QUEENS_ENUMERATION_HPP

#if defined(__CUDACC__) || defined(__HIPCC__)
#define CHOP_DEVICE __device__
#else
#define CHOP_DEVICE
#endif

static constexpr unsigned long long FULL_WARP_MASK = 0xFFFFFFFFFFFFFFFFULL;

CHOP_DEVICE void CUDA_HIP__queens_dfs_enumeration(
    const int N, const unsigned int nPrefixes, const int depthGlobal,
    QueenRoot *__restrict__ root_prefixes,
    unsigned long long *__restrict__ global_tree_size,
    unsigned long long *__restrict__ sols)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    unsigned long long tree_size = 0ULL;
    unsigned long long qtd_sols_thread = 0ULL;

    if (idx < nPrefixes)
    {
        unsigned int flag = 0;
        int8_t board[MAX_SIZE];
        int N_l = N;
        int i, depth;

        for (i = 0; i < N_l; ++i)
        {
            board[i] = EMPTY;
        }

        flag = root_prefixes[idx].control;

        for (i = 0; i < depthGlobal; ++i)
            board[i] = root_prefixes[idx].board[i];

        depth = depthGlobal;

        do
        {

            board[depth]++;
            const int mask = 1 << board[depth];

            if (board[depth] == N_l)
            {
                board[depth] = EMPTY;
                depth--;
                flag &= ~(1 << board[depth]);
            }
            else if (!(flag & mask) && queens_is_legal_placement(board, depth))
            {

                ++tree_size;
                flag |= mask;

                depth++;

                if (depth == N_l)
                { // sol
                    ++qtd_sols_thread;

                    depth--;
                    flag &= ~mask;
                }
            }
        } while (depth >= depthGlobal); // FIM DO DFS_BNB

    } // if

#if defined(SYCL_LANGUAGE_VERSION) || defined(__SYCL_DEVICE_ONLY__)

    auto sg = item.get_sub_group();
    unsigned long long reduced_tree = sycl::reduce_over_group(sg, tree_size, sycl::plus<>());
    unsigned long long reduced_qtd_sols_thread = sycl::reduce_over_group(sg, qtd_sols_thread, sycl::plus<>());

    // 3. Only the leader performs the atomic operation
    if (sycl::leader(sg))
    {

        sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
                         sycl::memory_scope::device>
            atomic_tree(global_tree_size[0]);

        sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
                         sycl::memory_scope::device>
            atomic_sols(sols[0]);

        atomic_tree.fetch_add(reduced_tree);
        atomic_sols.fetch_add(reduced_qtd_sols_thread);
    }

#else
      //  Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2)
    {
        tree_size += __shfl_down_sync(FULL_WARP_MASK, tree_size, offset);
        qtd_sols_thread += __shfl_down_sync(FULL_WARP_MASK, qtd_sols_thread, offset);
    }

    // Only one thread per warp adds the warp's result to the global total
    if (threadIdx.x % 32 == 0)
    {
        atomicAdd(global_tree_size, tree_size);
        atomicAdd(sols, qtd_sols_thread);
    }

#endif

} // kernel

#endif