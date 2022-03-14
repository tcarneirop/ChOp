#ifndef __MACROS_H
#define __MACROS_H

#define mat(i,j) mat[(i)*N+(j)]
#define mat_h(i,j) mat_h[(i)*N+(j)]
#define mat_d(i,j) mat_d[(i)*N_l+(j)]
#define mat_block(i,j) mat_block[(i)*N_l+(j)]

#define indice(i,j) ((i)*N_l+(j))

#define poolOfGuides(i,j) poolOfGuides[(i) * (N+1) + (j)]


#define _SET_   1
#define _UNSET_ 0
#define INFINITO 999999
#define ZERO 0
#define ONE 1

#define proximo(x) ((x)+1)
#define anterior(x) ((x)-1)


#define _VAZIO_      -1
#define _VISITADO_    1
#define _NAO_VISITADO_ 0
#define ZERO 0


#define MAX 64
#define __INFINITO__ 999999
#define __ZERO__ 0
#define __VISITADO__ 1
#define __UNVISITED__ 0


#define _NO_GPU_TREE 0ULL
#define _NO_CDP_TREE 0ULL
#define _NO_MC_TREE  0ULL
#define _NO_GPU_TIME 0.0
#define _NO_CDP_TIME 0.0
#define _NO_MC_TIME  0.0	
#define _NO_BLOCKS   0


#define AVG 0
#define VAR 1
#define DEVIANCE 2

	#define __HALF_GIGA_ 1073741824/2UL
	#define __ONE_GIGA_ 1073741824
	#define __TWO_GIGA_ 2UL*1073741824
	#define __FOUR_GIGA_ 4UL*1073741824
	#define __SIX_GIGA_ 6UL*1073741824
	#define __EIGHT_GIGA_ 8UL*1073741824

	#define __DEF_PARTITION__ 50

//MENU 
#define RESOURCES        "-r" 
#define QUEENS           "q"
#define ATSP             "a"
#define ATSPALLSPACE     "all"
#define QUEENSALLSPACE   "qall"
#define SOLVE            "s"
#define MCORE            "m"
#define HYBRID           "h"
#define PROFILE          "p"
#define TREEPROFILE      "-tp"
#define HELP             "-h"
#define DEVPROP          "-d"
#define BENCHMARK        "-ball"
#define BENCHCPU         "-bcpu"
#define BENCHGPU         "-bgpu"
#define MCOREPREGEN      "-mpg"
#define SERIALKKERNELS   "-skb"
#define KSTREAMSPBL      "-ksb"  
#define PLTH             "-p"
#define SEARCHDP2        "-dp2"
#define SEARCHCDPDP2     "-cdpdp2"
#define SEARCHDP3        "-dp3"
#define SEARCHCDPDP3     "-cdpdp3"
#define SEARCHPTHREADDP3 "-tbdp3"
#define SEARCHCDPTBDP3   "-cdptbdp3"
#define PERTHREAD        "-t"   
#define POOL             "-pool" 
#define RECDP3 		 	 "-rdp3"
#define RECCDP			 "-rcdp"	     
#define NONBP			 "-nonbp"
#define DOUBLE           "-double"
#define PLUSTWO          "-plustwo"
#define ASETPARTITION     "-ap"


#define _DOUBLE_ 0
#define _PLUS_TWO_ 1


#define __PERTHRD__ 0
#define _CDP_DP2_   1
#define _DP2_       2
#define _DP3_       3
#define _CDP_DP3_   4
#define _TB_DP3_    5
#define _CDP_TBDP3_ 6


#define _POOL_      0
#define _REGULAR_  1

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}


#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }                            

#endif