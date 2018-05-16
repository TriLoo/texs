__global__ void gaussfilterGlo_kernel(float *d_imgOut, float *d_imgIn, int wid, int hei, 
											const float * __restrict__ d_filter, int filterW)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    if(idx > wid || idy > hei)
        return ;

    int filterR = (filterW - 1) / 2;

    float val = 0.f;

    for(int fr = -filterR; fr<= filterR; ++fr)           // row
        for(int fc = -filterR; fc <= filterR; ++fc)      // col
        {
            int ir = idy + fr;
            int ic = idx + fc;

            if((ic >= 0) && (ic <= wid - 1) && (ir >= 0) && (ir <= hei - 1))
                val += d_imgIn[INDX(ir, ic, wid)] * d_filter[INDX(fr+filterR, fc+filterR, filterW)];
        }
    d_imgOut[INDX(idy, idx, wid)] = val;
}
