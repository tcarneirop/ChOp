#ifndef VEC_ADD_H
#define VEC_ADD_H

#define N 10

#ifdef __cplusplus
extern "C" {
#endif

void call_cuda_param_vector_add(int *h_a, int *h_b, int *out, int size);

#ifdef __cplusplus
}
#endif



#ifdef __cplusplus
extern "C" {
#endif
void call_cuda_vector_add();
#ifdef __cplusplus
}
#endif



#endif