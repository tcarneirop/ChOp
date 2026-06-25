#include "add.h"
#include <stdio.h>

int main(void) {
  printf("%d\n", add_c(1,2));
  printf("%d\n", sub_int(1,2));
  printf("%lf\n", sub_double(1.0l, 2.0l));
  return 0;
}
