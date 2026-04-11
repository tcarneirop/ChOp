# ChOp - Chapel-based Optimization

The objective of the ChOp project is to design and implement large-scale exact distributed optimization algorithms taking into account CPU-GPU heterogeneity, but also achieving high productivity and parallel efficiency. 

Chapel truly stands out because it effectively unifies the different parallel levels of modern GPU-powered clusters, handling everything from inter-node communication to intra-node parallelism across both CPUs and GPUs.

The prototypes are programmed to enumerate all feasible and complete configurations of the N-Queens. The final versions of the distributed algorithms solve to the optimality  instances of combinatorial optimization problems, such as the flow-shop scheduling and the ATSP. This study is pioneering within the context of parallel exact optimization.

This dramatically simplifies development by letting us avoid the complex mix of different programming languages and libraries typically needed for each parallel level. For instance, we no longer had to manually program intricate MPI-based load balancing schemes (see the figure below as well as [this paper](https://www.sciencedirect.com/science/article/pii/S0167739X1930946X)). This significantly higher productivity in combinatorial search was achieved with minor parallel performance losses.

## Overview of the algorithm:

The locale 0 (master) is responsible for generating the distribute pool Pd and controlling the search. Each worker locale receives nodes from the master and generates a local pool that is partitioned into CPU and GPU portions. L locales are launched on L-1 computer nodes.

<img title="" src="https://tcarneirop.github.io/pictures/overview.png" alt="" width="432" class="center" data-align="center">

## Some productivity/performance results of using Chapel for distributed exact optimization vs MPI+Cpp, flow-shop scheduling problem:

Execution times of Chapel-BB solving to the optimality Taillard instances ta21-30. The execution time is given relative to the MPI-PBB baseline. Next, productivity achieved by Chapel compared to its counterpart written in MPI+Cpp. Experiments executed on 1 (32 cores) to 32 nodes (1024 cores). For more details, see [Carneiro et al. (2020)](https://www.sciencedirect.com/science/article/pii/S0167739X1930946X).

<img title="" src="https://tcarneirop.github.io/pictures/performance.png" alt="" data-align="center" width="436"><sub>Normalized execution time of Chapel-B&B compared to its counterpart written in MPI+Cpp.</sub>



<img title="" src="https://tcarneirop.github.io/pictures/prod.png" alt="" width="422" data-align="center">

<sub>Normalized productivity achieved by Chapel compared to its counterpart written in MPI+Cpp.</sub>

## Recent results, GPU-based prototype in Chapel + CUDA, solving the N-Queens:

### 288 NVIDIA V100, 48 computer nodes.

84% of the linear speedup vs. the same application on one computer node. 74% of the linear speedup vs. the optimized baseline in CUDA on one computer node.  [See Carneiro et al. (2021)](https://hal.archives-ouvertes.fr/hal-03149394/document).

<img title="" src="https://tcarneirop.github.io/pictures/new.png" alt="" width="503" class="center" data-align="center">

#### [August 1, 2025] [📌 7 Questions for Chapel Users - Interview Series](https://chapel-lang.org/blog/posts/7qs-chop/)

🚀 Excited to share that I had the opportunity to take part in the latest "7 Questions for Chapel Users" interview series, alongside my colleague and friend [Guillaume Helbecque](https://www.linkedin.com/in/guillaume-helbecque-4810b519a/)!

We talked about [ChOp](https://github.com/tcarneirop/ChOp), our Chapel-based Optimization project, our work on large-scale combinatorial optimization problems, and how Chapel helps us tackle these challenges efficiently and at scale.

Many thanks to [Brad Chamberlain](https://www.linkedin.com/in/brad-chamberlain-3ab358105/) and the whole Chapel team for the spotlight!

See this post on Linkedin and on the [Chapel Language's Blog](https://chapel-lang.org/blog/posts/7qs-chop/).

## Publications:

- Carneiro, T. Kayraklioglu, E.  Helbecque, G.  Melab, N. - [Investigating Portability in Chapel for Tree-Based Optimization on GPU-Powered Clusters](https://scholar.google.com/scholar?oi=bibs&cluster=8018526403536283056&btnI=1&hl=en), European Conference on Parallel Processing (Europar), 2024. **DOI:** [10.1109/IPDPSW55747.2022.00132](https://doi.org/10.1109/IPDPSW55747.2022.00132).
  
  -  [Chapel: ChapelCon '24](https://chapel-lang.org/ChapelCon24.html)[Chapel: ChapelCon '24](https://chapel-lang.org/ChapelCon24.html) Talk - [[slides](https://chapel-lang.org/chapelcon/2024/pessoa.pdf) | [video](https://www.youtube.com/watch?v=Zh0YrGDZV1o&list=PLuqM5RJ2KYFi2yV4sFLc6QeRYpS35UeKl&index=7&pp=iAQB)].

- Carneiro, T; Koutsantonis, L.; Melab, N.; Kieffer, E.; Bouvry, P. [A Local Search for Automatic Parameterization of Distributed Tree Search Algorithms](https://hal.archives-ouvertes.fr/hal-03619760). [PDCO 2022](https://pdco2022.sciencesconf.org/) - 12th
  IEEE Workshop Parallel / Distributed Combinatorics and Optimization, May 2022, Lyon, France.

- Carneiro, T.,  Nouredine, M. Towards Ultra-scale Exact Optimization Using Chapel. *The 8th Annual Chapel Implementers and Users Workshop*, Jun 2021, Seattle, United States. [⟨hal-03326294⟩](https://hal.science/hal-03326294v1)

- Carneiro, T.; Melab, N.; Hayashi, A.; Sarkar, V. [Towards Chapel-based Exascale Tree Search Algorithms: dealing withmultiple GPU accelerators](https://hal.archives-ouvertes.fr/hal-03149394/document). In: The International Conference on High Performance Computing & Simulation - HPCS2020 (2021). [HPCS 2020 Outstanding Paper Award winner](http://hpcs2020.cisedu.info/2-conference/outstanding-paper-poster-awards).

- Carneiro, T.; Gmys, J.; Melab, N.; Tuyttens, D. [Towards Ultra-scale Branch-and-Bound Using a High-productivity Lan-guage](https://doi.org/10.1016/J.future.2019.11.011). Future Generation Computer Systems, 105: 196-209 (2020). DOI: 10.1016/J.future.2019.11.011.

- Gmys, J.; Carneiro, T.; Melab, N.; Tuyttens, d.; Talbi, E-G. [A Comparative Study of High-productivity High-performance Programming Languages for Parallel Metaheuristics](https://doi.org/10.1016/j.swevo.2020.100720). Swarm and Evolutionary Computation, 57:100720 (2020). DOI:10.1016/j.swevo.2020.100720

- Carneiro, T.;   Melab,   N.  [An  Incremental  Parallel  PGAS-based  Tree  Search  Algorithm](https://ieeexplore.ieee.org/document/9188106). In:    The  2019  In-ternational   Conference   on   High   Performance   Computing   &   Simulation   -   HPCS2019,   pp.19-26,   DOI:10.1109/HPCS48598.2019.9188106.

- Carneiro, T.; Melab, N. [Productivity-aware Design and Implementation of Distributed Tree-based Search Algorithms](https://link.Springer.com/chapter/10.1007/978-3-030-22734-0_19). In:  The International Conference on Computational Science - ICCS2019.  Lecture notes in computer science, vol.11536 (253-266), Springer. DOI: 10.1007/978-3-030-15996-2_2
