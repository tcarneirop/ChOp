#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <sycl/sycl.hpp>

#include "../headers/queens_subproblem.hpp"
#include "../headers/queens_CPU_GPU_subproblem_eval.hpp"
#include "../headers/queens_sub_gen.hpp"

#define ONE 1

static constexpr unsigned long long FULL_WARP_MASK = 0xFFFFFFFFFFFFFFFFULL;

void SYCL_queens_dfs_enumeration(
	sycl::nd_item<1> item, const int N, const unsigned int nPrefixes, const int depthGlobal,
	QueenRoot *__restrict__ root_prefixes,
	unsigned long long *__restrict__ global_tree_size,
	unsigned long long *__restrict__ sols)
{

	unsigned long long tree_size = 0ULL;
	unsigned long long qtd_sols_thread = 0ULL;
	int idx = static_cast<int>(item.get_global_id(0));
	// Get sub_group for reductions
	auto sg = item.get_sub_group();

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

	unsigned long long reduced_tree = sycl::reduce_over_group(sg, tree_size, sycl::plus<>());
	unsigned long long reduced_qtd_sols_thread = sycl::reduce_over_group(sg, qtd_sols_thread, sycl::plus<>());

	if (sg.get_local_id()[0] == 0)
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

} // kernel

void SYCL_SGPU_call_queens(const int size, const int initialDepth, const int block_size)
{

	unsigned long long initial_tree_size = 0ULL;
	unsigned long long qtd_sols_global = 0ULL;
	unsigned long long gpu_tree_size = 0ULL;

	unsigned long long nMaxPrefixes = 75580635;

	QueenRoot *root_prefixes_h = (QueenRoot *)malloc(sizeof(QueenRoot) * nMaxPrefixes);
	unsigned long long sols_h = 0ULL;

	// initial search, getting Feasible, Valid and Incomplete solutions -- subproblems;
	unsigned long long n_explorers = queens_subproblem_generation((char)size, initialDepth, &initial_tree_size, root_prefixes_h);

	printf("\n### Queens size: %d, Initial depth: %d, Block size: %d - Num_explorers: %llu", size, initialDepth, block_size, n_explorers);

#ifdef USE_GPU
	sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
	sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

	int num_blocks = ceil((double)n_explorers / block_size);

	unsigned long long *vector_of_tree_size_d = sycl::malloc_device<unsigned long long>(ONE, q);
	unsigned long long *sols_d = sycl::malloc_device<unsigned long long>(ONE, q);
	QueenRoot *root_prefixes_d = sycl::malloc_device<QueenRoot>(n_explorers, q);

	q.memcpy(root_prefixes_d, root_prefixes_h, n_explorers * sizeof(QueenRoot));

	sycl::range<1> gws(num_blocks * block_size); // group of work items, number of threads.
	sycl::range<1> lws(block_size);				 // number of work intens on a single group, i.e., theads

	q.wait();

	auto start = std::chrono::steady_clock::now();

	static thread_local int SYCL_IDX_PROXY;

	q.submit([&](sycl::handler &cgh)
			 {
				 cgh.parallel_for<class nqueen>(
					 sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item)
					 { SYCL_queens_dfs_enumeration(
						   item,
						   size,
						   n_explorers,
						   initialDepth,
						   root_prefixes_d,
						   vector_of_tree_size_d,
						   sols_d); });
			 });

	q.wait();
	auto end = std::chrono::steady_clock::now();
	auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

	q.memcpy(&gpu_tree_size, vector_of_tree_size_d, sizeof(unsigned long long));
	q.memcpy(&qtd_sols_global, sols_d, sizeof(unsigned long long));
	q.wait();

#ifdef IMPROVED
	qtd_sols_global *= 2;
#endif

	printf("\nInitial tree size: %llu", initial_tree_size);
	printf("\nGPU Tree size: %llu\nTotal tree size: %llu\nNumber of solutions found: %llu\n", gpu_tree_size, (initial_tree_size + gpu_tree_size), qtd_sols_global);
	printf("\nElapsed total: %.3f\n", (time * 1e-9f));

#ifdef CHECKSOL
	if (qtd_sols_global == check_sols_number[size - 1])
		printf("\n####### SUCCESS - CORRECT NUMBER OF SOLS. FOR SIZE %d\n", size);
	else
		printf("########## ERROR -- INCORRECT NUMBER FOS SOLS. FOR SIZE %d: %llu vs. %llu (correct)\n", size, qtd_sols_global, check_sols_number[size - 1]);
#endif

	sycl::free(vector_of_tree_size_d, q);
	sycl::free(sols_d, q);
	sycl::free(root_prefixes_d, q);
}

int main(int argc, char *argv[])
{

	int initialDepth;
	int size;
	int block_size;

#ifdef IMPROVED
	printf("### IMPROVED SEARCH - Avoiding mirrored solutions\n");
#endif

	if (argc != 4)
	{
		printf("Usage: %s <size> <initial depth> <block size>\n", argv[0]);
		return 1;
	}

	size = atoi(argv[1]);
	initialDepth = atoi(argv[2]);
	block_size = atoi(argv[3]);

	SYCL_SGPU_call_queens(size, initialDepth, block_size);

	return 0;
}
