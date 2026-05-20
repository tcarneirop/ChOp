# ChOp - Chapel-based Optimization

Project focused on the design and implementation of large-scale distributed exact optimization algorithms in Chapel for CPU-GPU heterogeneous systems, balancing productivity, scalability, and parallel efficiency. The project investigates performance-portable approaches for multicore, multi-GPU, and distributed branch-and-bound applications, including comparisons against OpenMP, MPI, CUDA, HIP, and SYCL implementations.

## Overview

___

## ## Implementations

In ChOP, there are diverse implementations for solving two permutation-based combinatorial problems: the Flow Shop Scheduling Problem and the classic N-Queens problem. More specifically, the following implementations are available:

### ### The N-Queens problem:

There are two backtracking versions for solving the N-Queens problem: 

1. **Vector**, backtracking implemented using a vector-based data structure, slower, but it is a branch-and-bound skeleton for other problems. There are 4 implementations using the vector-based data-structure:
   
   1. Serial
   
   2. Multicore - single-node (single-locale in Chapel)
   
   3. Multi-GPU - single-node 
      
      1. **Important:** the GPU-based versions brings two versions of the kernels: Chapel-based and CUDA/HIP-based kernels, depending on the system. CUDA-based kernels for NVIDIA systems and HIP-based kernels for AMD 
   
   4. Distributed-CPU (multi-locale)
   
   5. Distributed-GPU (multi-locale)
      
      1. **Important:** the same goes for the distributed version of the code - two versions of the kernels. 

2. **Bitset**, backtracking implemented using a bitset-based data structure. There are 3 implementations using the vector-based data-structure:
   
   1. Serial
   
   2. Multicore - single-node (single-locale in Chapel)
   
   3. Distributed-CPU (multi-locale)

### The Flow-shop Scheduling Problem (FSP)

There are two branch-and-bound versions for solving the FSP:

1. 

## Building and Running ChOP

### The N-Queens Problem:

First, to run the benchmarks,  need to:

1) build the runtime for single-locale
2) clone the main branch 
3) Compile the single-locale version of the code:
4) ```shell
   make queens_singlelocale_cpu
   ```
5) If it compiles, you get two versions of the code:
6) N-Queens with vector data structure (much slower, but it is a skeleton to other problems)
7) N-Queens bitsets, which I believe is more interesting for you, 
   as the steps of the search can be expressed as bitwise operations.
   When the execution finishes you get a small report on these operations. 
   I call it ''tree size'', as it is a combinatorial search.
8) 

## 

## Solving Other Probles With ChOP

## Publications:

- Carneiro, T.; Kayraklioglu, E.; Helbecque, G.; Melab, N.  [Investigating Portability in Chapel for Tree-based Optimization on GPU-powered Clusters.](https://link.springer.com/chapter/10.1007/978-3-031-69583-4_27) ([Slides](https://chapel-lang.org/papers/Europar2024.pdf)) Euro-PAR 2024, August 28, 2024.

- Carneiro, T; Koutsantonis, L.; Melab, N.; Kieffer, E.; Bouvry, P. [A Local Search for Automatic Parameterization of Distributed Tree Search Algorithms](https://hal.archives-ouvertes.fr/hal-03619760). [PDCO 2022](https://pdco2022.sciencesconf.org/) - 12th IEEE Workshop Parallel / Distributed Combinatorics and Optimization, May 2022, Lyon, France.

- Carneiro, T.; Melab, N.; Hayashi, A.; Sarkar, V. [Towards Chapel-based Exascale Tree Search Algorithms: dealing withmultiple GPU accelerators](https://hal.archives-ouvertes.fr/hal-03149394/document). In: The International Conference on High Performance Computing & Simulation - HPCS2020 (2021). [HPCS 2020 Outstanding Paper Award winner](http://hpcs2020.cisedu.info/2-conference/outstanding-paper-poster-awards).

- Carneiro, T.; Gmys, J.; Melab, N.; Tuyttens, D. [Towards Ultra-scale Branch-and-Bound Using a High-productivity Lan-guage](https://doi.org/10.1016/J.future.2019.11.011). Future Generation Computer Systems, 105: 196-209 (2020). DOI: 10.1016/J.future.2019.11.011.

- Gmys, J.; Carneiro, T.; Melab, N.; Tuyttens, d.; Talbi, E-G. [A Comparative Study of High-productivity High-performance Programming Languages for Parallel Metaheuristics](https://doi.org/10.1016/j.swevo.2020.100720). Swarm and Evolutionary Computation, 57:100720 (2020). DOI:10.1016/j.swevo.2020.100720

- Carneiro, T.;   Melab,   N.  [An  Incremental  Parallel  PGAS-based  Tree  Search  Algorithm](https://ieeexplore.ieee.org/document/9188106). In:    The  2019  In-ternational   Conference   on   High   Performance   Computing   &   Simulation   -   HPCS2019,   pp.19-26,   DOI:10.1109/HPCS48598.2019.9188106.

- Carneiro, T.; Melab, N. [Productivity-aware Design and Implementation of Distributed Tree-based Search Algorithms](https://link.Springer.com/chapter/10.1007/978-3-030-22734-0_19). In:  The International Conference on Computational Science - ICCS2019.  Lecture notes in computer science, vol.11536 (253-266), Springer. DOI: 10.1007/978-3-030-15996-2_2
