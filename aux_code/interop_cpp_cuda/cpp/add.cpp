#include <utility>

// clang-format off
int add(std::pair<int,int> p) {
  return p.first + p.second;
}

template<typename T>
T sub(T a, T b) {
  return a - b;
}

#ifdef __cplusplus
extern "C" {
#endif

int add_c(int a, int b) {
  return add(std::make_pair(a,b));
}

int sub_int(int a, int b) {
  return sub(a,b);
}

double sub_double(double a, double b) {
  return sub(a,b);
}


#ifdef __cplusplus
}
#endif
