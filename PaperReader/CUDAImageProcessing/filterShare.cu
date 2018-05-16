__global__ void convolutionGPU(
................................float *d_Result,
................................float *d_Data,
................................int dataW,
................................int dataH
................................)
{
....// Data cache: threadIdx.x , threadIdx.y
....__shared__ float data[TILE_W + KERNEL_RADIUS * 2][TILE_W + KERNEL_RADIUS * 2]; 

....// global mem address of this thread
....const int gLoc = threadIdx.x + 
........................IMUL(blockIdx.x, blockDim.x) +
........................IMUL(threadIdx.y, dataW) +
........................IMUL(blockIdx.y, blockDim.y) * dataW; 

....// load cache (32x32 shared memory, 16x16 threads blocks)
....// each threads loads four values from global memory into shared mem
....// if in image area, get value in global mem, else 0
....int x, y;	// image based coordinate

....// original image based coordinate
....const int x0 = threadIdx.x + IMUL(blockIdx.x, blockDim.x);
....const int y0 = threadIdx.y + IMUL(blockIdx.y, blockDim.y); 

....// case1: upper left
....x = x0 - KERNEL_RADIUS;
....y = y0 - KERNEL_RADIUS;
....if ( x < 0 || y < 0 )
........data[threadIdx.x][threadIdx.y] = 0;
....else 
........data[threadIdx.x][threadIdx.y] = d_Data[ gLoc - KERNEL_RADIUS - IMUL(dataW, KERNEL_RADIUS)];

....// case2: upper right
....x = x0 + KERNEL_RADIUS;
....y = y0 - KERNEL_RADIUS;
....if ( x > dataW-1 || y < 0 )
........data[threadIdx.x + blockDim.x][threadIdx.y] = 0;
....else 
........data[threadIdx.x + blockDim.x][threadIdx.y] = d_Data[gLoc + KERNEL_RADIUS - IMUL(dataW, KERNEL_RADIUS)];

....// case3: lower left
....x = x0 - KERNEL_RADIUS;
....y = y0 + KERNEL_RADIUS;
....if (x < 0 || y > dataH-1)
........data[threadIdx.x][threadIdx.y + blockDim.y] = 0;
....else 
........data[threadIdx.x][threadIdx.y + blockDim.y] = d_Data[gLoc - KERNEL_RADIUS + IMUL(dataW, KERNEL_RADIUS)];

....// case4: lower right
....x = x0 + KERNEL_RADIUS;
....y = y0 + KERNEL_RADIUS;
....if ( x > dataW-1 || y > dataH-1)
........data[threadIdx.x + blockDim.x][threadIdx.y + blockDim.y] = 0;
....else 
........data[threadIdx.x + blockDim.x][threadIdx.y + blockDim.y] = d_Data[gLoc + KERNEL_RADIUS + IMUL(dataW, KERNEL_RADIUS)]; 

....__syncthreads();

....// convolution
....float sum = 0;
....x = KERNEL_RADIUS + threadIdx.x;
....y = KERNEL_RADIUS + threadIdx.y;
....for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
........for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
............sum += data[x + i][y + j] * d_Kernel[KERNEL_RADIUS + j] * d_Kernel[KERNEL_RADIUS + i];

....d_Result[gLoc] = sum; 
}
