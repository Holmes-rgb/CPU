#include <stdio.h>

int main() {
  cudaDeviceProp prop;

  int count;
  cudaGetDeviceCount( &count );
  printf("there is/are %d device(s)\n", count);
  for (int i=0; i<count; ++i) {
    cudaGetDeviceProperties( &prop, i );
    printf("name is %s\n", prop.name);
    printf("major.minor is %d.%d\n", prop.major, prop.minor);
    printf("multiProcessorCount is %d\n", prop.multiProcessorCount);
    printf("warpSize is %d\n", prop.warpSize);
    printf("maxThreadsPerBlock is %d\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim is (%d, %d, %d)\n", prop.maxThreadsDim[0],
           prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize is (%d, %d, %d)\n", prop.maxGridSize[0],
           prop.maxGridSize[1], prop.maxGridSize[2]);
  }

  return 0;
}
