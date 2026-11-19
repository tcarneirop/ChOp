# ChOp — Chapel-based Optimization

**An experimental research laboratory for productivity, performance, portability, and scalability in parallel combinatorial optimization.**

## Overview

ChOp is a research project focused on designing and implementing large-scale exact distributed optimization algorithms in [Chapel](https://chapel-lang.org/), targeting heterogeneous CPU–GPU systems while investigating productivity, scalability, portability, and parallel efficiency.

The project uses combinatorial algorithmic skeletons and large-scale applications as **experimental artifacts** to study how programming abstractions influence algorithm development and execution. Its central research question is:

**How can we develop productive, portable, scalable, and efficient exact optimization algorithms for heterogeneous computing architectures?**

ChOp's research includes N-Queens prototypes for tree-search experimentation and exact optimization applications for problems such as flow-shop scheduling and the asymmetric traveling salesman problem (ATSP).

A complementary repository, the **[NQueens Performance Portability Lab](https://github.com/tcarneirop/NQueens-Performance-Portability-Lab)**, extends these experimental comparisons across programming models, including MPI, OpenMP, CUDA, HIP, SYCL, Kokkos, and others. This separation supports systematic studies of performance, productivity, and portability without assuming that any single programming model is universally superior.

## Why Chapel?

[Chapel](https://chapel-lang.org/) is a high-level parallel programming language designed to make parallel computing more productive, from multicore systems to distributed-memory machines and GPU accelerators.

Its task-parallel and data-parallel constructs, distributed arrays, and PGAS-oriented abstractions make it an interesting research vehicle for irregular tree-search applications such as backtracking and Branch-and-Bound. ChOp investigates the benefits and limitations of these abstractions, particularly how they affect programming effort, performance, and scalability on heterogeneous HPC systems.

## Research Objectives

- **Productivity:** Study and quantify the effort required to develop, maintain, and extend parallel optimization algorithms.
- **Performance:** Compare computational efficiency across implementations and architectures.
- **Portability:** Investigate the effort and performance implications of moving applications across hardware and programming environments.
- **Scalability:** Study exact optimization algorithms on multicore, multi-GPU, and distributed computing systems.

## Distributed Tree Search

One of the distributed designs uses a master–worker organization. Locale 0 (the master) generates and distributes an initial pool of subproblems and coordinates the search. Worker locales maintain local search pools, with work partitioned between CPUs and GPUs.

<img src="https://tcarneirop.github.io/pictures/overview.png" width="50%" class="center">

## Selected Experimental Results

### Multi-GPU Scalability — Chapel + CUDA

A GPU-based N-Queens prototype was evaluated using **288 NVIDIA V100 GPUs across 48 computing nodes**. The original study reports **84% of linear speedup** compared with the same application on one node, and **74% of linear speedup** compared with an optimized single-node CUDA baseline.

See [Carneiro et al. (2021)](https://hal.archives-ouvertes.fr/hal-03149394/document).

<img src="https://tcarneirop.github.io/pictures/new.png" width="50%" class="center">

### Distributed Branch-and-Bound — Chapel vs. MPI/C++

The Chapel-based Branch-and-Bound implementation was compared with an MPI/C++ baseline on Taillard flow-shop scheduling instances (ta21–ta30). The experiments examined both execution time and programming productivity, from **1 node (32 CPU cores) to 32 nodes (1,024 CPU cores)**.

See [Carneiro et al. (2020)](https://doi.org/10.1016/j.future.2019.11.011).

**Performance comparison**

<img src="https://tcarneirop.github.io/pictures/performance.png" width="50%" class="center">

**Productivity comparison**

<img src="https://tcarneirop.github.io/pictures/prod.png" width="50%" class="center">

## Recognition & Impact

- **Chapel in Production:** ChOp is featured by the Chapel team among the projects showcased in [Chapel in Production](https://chapel-lang.org/).
- **Chapel developer interview:** ChOp's research and development were discussed in the official [“7 Questions” interview with Tiago Carneiro and Guillaume Helbecque](https://chapel-lang.org/blog/posts/7qs-chop/).
- **HeCBench:** The CUDA-based N-Queens implementation from ChOp has been included in [HeCBench](https://github.com/zjin-lcf/HeCBench), a heterogeneous computing benchmark collection.
- **GPU validation and performance tracking:** ChOp has been used by the Chapel team in GPU-related evaluation. See the [CHIUW 2023 presentation](https://chapel-lang.org/CHIUW/2023/KayrakliogluSlides.pdf) and [Chapel GPU performance graphs](https://chapel-lang.org/perf/1-node-mi250x/?graphs=choptime).
- **RISC-V portability and performance evaluation:** ChOp was used as an application benchmark in a study by researchers from Sandia National Laboratories, HPE, and collaborating institutions, evaluating Chapel on emerging RISC-V systems alongside ARM and x86 platforms. See [Henriksen et al. (2026)](https://arxiv.org/pdf/2608.14799), Section V.

## Publications

- Carneiro, T.; Kayraklioglu, E.; Helbecque, G.; Melab, N. (2024). **[Investigating Portability in Chapel for Tree-Based Optimization on GPU-Powered Clusters](https://doi.org/10.1007/978-3-031-69583-4_27).** *Euro-Par 2024: Parallel Processing*, Lecture Notes in Computer Science, vol. 14803, pp. 386–399. Springer.

- Carneiro, T.; Koutsantonis, L.; Melab, N.; Kieffer, E.; Bouvry, P. (2022). [A Local Search for Automatic Parameterization of Distributed Tree Search Algorithms](https://hal.archives-ouvertes.fr/hal-03619760). *12th IEEE Workshop on Parallel / Distributed Combinatorics and Optimization (PDCO)*, Lyon, France.

- Carneiro, T.; Melab, N.; Hayashi, A.; Sarkar, V. (2021). [Towards Chapel-based Exascale Tree Search Algorithms: Dealing with Multiple GPU Accelerators](https://hal.archives-ouvertes.fr/hal-03149394/document). *International Conference on High Performance Computing & Simulation (HPCS 2020).* **Outstanding Paper Award.**

- Carneiro, T.; Gmys, J.; Melab, N.; Tuyttens, D. (2020). [Towards Ultra-scale Branch-and-Bound Using a High-productivity Language](https://doi.org/10.1016/j.future.2019.11.011). *Future Generation Computer Systems*, vol. 105, pp. 196–209.

- Gmys, J.; Carneiro, T.; Melab, N.; Tuyttens, D.; Talbi, E.-G. (2020). [A Comparative Study of High-productivity High-performance Programming Languages for Parallel Metaheuristics](https://doi.org/10.1016/j.swevo.2020.100720). *Swarm and Evolutionary Computation*, vol. 57, article 100720.

- Carneiro, T.; Melab, N. (2019). [An Incremental Parallel PGAS-based Tree Search Algorithm](https://ieeexplore.ieee.org/document/9188106). *International Conference on High Performance Computing & Simulation (HPCS)*, pp. 19–26. DOI: 10.1109/HPCS48598.2019.9188106.

- Carneiro, T.; Melab, N. (2019). [Productivity-aware Design and Implementation of Distributed Tree-based Search Algorithms](https://link.springer.com/chapter/10.1007/978-3-030-22734-0_19). *International Conference on Computational Science (ICCS)*, Lecture Notes in Computer Science, vol. 11536, pp. 253–266. Springer.
