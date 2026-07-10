#ifndef QUEENS_ENUMERATION_HPP
#define QUEENS_ENUMERATION_HPP

#if defined(__CUDACC__) || defined(__HIPCC__)
#define CHOP_GLOBAL __global__
#else
#define CHOP_GLOBAL
#endif

static constexpr unsigned long long FULL_WARP_MASK = 0xFFFFFFFFFFFFFFFFULL;

CHOP_GLOBAL void CUDA_HIP__queens_dfs_enumeration(
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

            //board[depth]++;
            const int mask = 1 << ++board[depth];

            if (board[depth] == N_l)
            {
                board[depth] = EMPTY;
               // depth--;
                flag &= ~(1 << board[--depth]);
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


      //  Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2)
    {

        #ifdef HIP_OLD_VERSION
        tree_size += __shfl_down(tree_size, offset);
        qtd_sols_thread += __shfl_down(qtd_sols_thread, offset);
        #else
        tree_size += __shfl_down_sync(FULL_WARP_MASK, tree_size, offset);
        qtd_sols_thread += __shfl_down_sync(FULL_WARP_MASK, qtd_sols_thread, offset);
        #endif
    }

    // Only one thread per warp adds the warp's result to the global total
    if (threadIdx.x % 32 == 0)
    {
        atomicAdd(global_tree_size, tree_size);
        atomicAdd(sols, qtd_sols_thread);
    }


} // kernel

#endif
