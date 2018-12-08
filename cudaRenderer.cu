/*

We use a term *tile* to identify the rectangular submatrices of the image.
Not to be confused with the blocks of threads.

*/

#include <cuda_runtime.h>
#include <stdio.h>

#define DSM_MAX_TILES_PER_BLOCK 500
#define DSM_MAX_TILES_PER_THREAD 500

// threads per block
#define TPB_1D 16
#define TPB (TPB_1D * TPB_1D)
// satellite pixels per thread
#define SAT_PPT_1D 2
#define SAT_PPT (SAT_PPT_1D * SAT_PPT_1D)
// satellite pixels per block
#define SAT_PPB_1D (SAT_PPT_1D * TPB_1D)
#define SAT_PPB (SAT_PPB_1D * SAT_PPB_1D)
// DSM pixels per thread
#define DSM_PPT_1D 1
#define DSM_PPT (DSM_PPT_1D * DSM_PPT_1D)
// DSM pixels per block
#define DSM_PPB_1D (DSM_PPT_1D * TPB_1D)
// #define DSM_PPB (DSM_PPB_1D * DSM_PPB_1D)

// this needs to be large negative number
#define DSM_IGNORE_VALUE -1E5
// extern const float DSM_IGNORE_VALUE;
#define EPS 1E-3

#define SCAN_BLOCK_DIM TPB
#include "exclusiveScan.cu_inl"

#define DTYPE float

__device__ bool d_rectanglesIntersect(DTYPE* bbox1, DTYPE* bbox2) {
    if (bbox2[0] > bbox1[2] ||
        bbox2[1] > bbox1[3] ||
        bbox1[0] > bbox2[2] ||
        bbox1[1] > bbox2[3]) { return false; }
    else { return true; }
}

__device__ DTYPE d_area(DTYPE x1, DTYPE y1,
            DTYPE x2, DTYPE y2,
            DTYPE x3, DTYPE y3) {
    return abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2;
}

__device__ DTYPE d_interpolate_three(DTYPE x, DTYPE y,
                        DTYPE x1, DTYPE y1, DTYPE v1,
                        DTYPE x2, DTYPE y2, DTYPE v2, 
                        DTYPE x3, DTYPE y3, DTYPE v3) {
    DTYPE denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
    DTYPE w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom;
    DTYPE w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom;
    DTYPE w3 = 1. - w1 - w2; 

    return (w1 * v1 + w2 * v2 + w3 * v3);
}

__device__ bool d_inside_triangle(DTYPE x, DTYPE y,
                     DTYPE x1, DTYPE y1, 
                     DTYPE x2, DTYPE y2, 
                     DTYPE x3, DTYPE y3) {
    DTYPE A = d_area(x1, y1, x2, y2, x3, y3);    
    DTYPE A1 = d_area(x, y, x1, y1, x2, y2);
    DTYPE A2 = d_area(x, y, x3, y3, x1, y1);
    DTYPE A3 = d_area(x, y, x2, y2, x3, y3);
    return (abs(A1 + A2 + A3 - A) < EPS);
}

__global__ void kernelRenderSatElevation(
        DTYPE* pX, DTYPE* pY, DTYPE* pZ,
        DTYPE* pOut,
        DTYPE* pTilesBboxes,
        int numDSMTiles, int numDSMTiles_X,
        int dsm_width, int dsm_height,
        int sat_width, int sat_height) {
    // One thread block processes one sattelite tile

    // linear thread index that is sorted into thread warps
    int linearThreadIdx = threadIdx.y * blockDim.x + threadIdx.x;

    // pixels being processed
    int satTileX0 = blockIdx.x * SAT_PPB_1D;
    int satTileY0 = blockIdx.y * SAT_PPB_1D;
    int satTileX1 = satTileX0 + SAT_PPB_1D - 1;
    int satTileY1 = satTileY0 + SAT_PPB_1D - 1;
    if (blockIdx.x == gridDim.x - 1) {
        satTileX1 = sat_width - 1;
    }
    if (blockIdx.y == gridDim.y - 1) {
        satTileY1 = sat_height - 1;
    }
    DTYPE satTileBbox[] = {
        static_cast<DTYPE>(satTileX0),
        static_cast<DTYPE>(satTileY0),
        static_cast<DTYPE>(satTileX1),
        static_cast<DTYPE>(satTileY1),
    };

    __shared__ uint privateTileCount[TPB];
    __shared__ uint accumPrivateTileCount[TPB];
    // TODO: use tileIndex as scratch to save some memory
    __shared__ uint privateTileCountScratch[2 * TPB];
    __shared__ uint tileIndex [DSM_MAX_TILES_PER_BLOCK];

    int dsmTilesPerThread = (numDSMTiles + TPB - 1) / TPB;
    int dsmTilesStart = dsmTilesPerThread * linearThreadIdx;
    int dsmTilesEnd = dsmTilesStart + dsmTilesPerThread;
    // (linearThreadIdx == TPB - 1) condition is wrong, because here
    // we divide the array into TPB parts, as opposed to dividing
    // into parts of fixed size
    dsmTilesEnd = fminf(dsmTilesEnd, numDSMTiles);

    int numPrivateTiles = 0;
    uint privateTileList[DSM_MAX_TILES_PER_THREAD];
    for (int i = dsmTilesStart; i < dsmTilesEnd; ++i) {
        if (d_rectanglesIntersect(pTilesBboxes + i * 4, satTileBbox))
            privateTileList[numPrivateTiles++] = i;
    }
    privateTileCount[linearThreadIdx] = numPrivateTiles;
    __syncthreads();
    sharedMemExclusiveScan(linearThreadIdx, privateTileCount,
                           accumPrivateTileCount, privateTileCountScratch, TPB);
    __syncthreads();

    // total number of DSM tiles that intersect with the current sat tile
    int numTiles = privateTileCount[TPB - 1] + accumPrivateTileCount[TPB - 1];

/*
    // TODO: debug
    for (int dx = 0; dx < SAT_PPT_1D; ++dx) {
        for (int dy = 0; dy < SAT_PPT_1D; ++dy) {
            int x = satTileX0 + SAT_PPT_1D * threadIdx.x + dx;
            int y = satTileY0 + SAT_PPT_1D * threadIdx.y + dy;
            if (x > sat_width - 1 || y > sat_height - 1) {
                continue;
            }
            int pixelIndex = y * sat_width + x;
            pOut[pixelIndex] = static_cast<DTYPE>(numTiles);
        }
    }
    int tmpIdx = (blockIdx.y * TPB_1D + threadIdx.y) * sat_width + blockIdx.x * TPB_1D + threadIdx.x;
    if (tmpIdx < sat_height * sat_width) {
        // pOut[tmpIdx] = static_cast<int>(accumPrivateTileCount[linearThreadIdx]);
        pOut[tmpIdx] = static_cast<DTYPE>(numTiles);
        // pOut[tmpIdx] = static_cast<DTYPE>(numPrivateTiles);
    }

    // end
*/

    int curIndex = accumPrivateTileCount[linearThreadIdx];
    for (int i = 0; i < numPrivateTiles; ++i) {
        tileIndex[curIndex++] = privateTileList[i];
    }
    __syncthreads();

    for (int iTile = 0; iTile < numTiles; ++iTile) {
        int dsmTileIndex = tileIndex[iTile];

        int dsmTileX0 = (dsmTileIndex % numDSMTiles_X) * (DSM_PPB_1D - 1);
        int dsmTileY0 = (dsmTileIndex / numDSMTiles_X) * (DSM_PPB_1D - 1);
        int dsmTileX1 = dsmTileX0 + DSM_PPB_1D - 1;
        int dsmTileY1 = dsmTileY0 + DSM_PPB_1D - 1;
        if (dsmTileX1 > dsm_width - 2) { dsmTileX1 = dsm_width - 2; }
        if (dsmTileY1 > dsm_height - 2) { dsmTileY1 = dsm_height - 2; }

        for (int row_d = dsmTileY0; row_d <= dsmTileY1; ++row_d) {
            for (int col_d = dsmTileX0; col_d <= dsmTileX1; ++col_d) {
                int idx = row_d * dsm_width + col_d;

                for (int j = 0; j < 2; ++j) {
                    DTYPE x1, y1, elev1, x2, y2, elev2, x3, y3, elev3;
                    if (j == 0) {
                        x1 = pX[idx] - satTileX0;
                        y1 = pY[idx] - satTileY0;
                        elev1 = pZ[idx];
                        x2 = pX[idx + 1] - satTileX0;
                        y2 = pY[idx + 1] - satTileY0;
                        elev2 = pZ[idx + 1];
                        x3 = pX[idx + dsm_width] - satTileX0;
                        y3 = pY[idx + dsm_width] - satTileY0;
                        elev3 = pZ[idx + dsm_width];
                    }
                    else {  // j == 1
                        x1 = pX[idx + 1] - satTileX0;
                        y1 = pY[idx + 1] - satTileY0;
                        elev1 = pZ[idx + 1];
                        x2 = pX[idx + dsm_width] - satTileX0;
                        y2 = pY[idx + dsm_width] - satTileY0;
                        elev2 = pZ[idx + dsm_width];
                        x3 = pX[idx + dsm_width + 1] - satTileX0;
                        y3 = pY[idx + dsm_width + 1] - satTileY0;
                        elev3 = pZ[idx + dsm_width + 1];
                    }

                    // skip invalid faces
                    if ((elev1 < DSM_IGNORE_VALUE + 1) ||
                        (elev2 < DSM_IGNORE_VALUE + 1) ||
                        (elev3 < DSM_IGNORE_VALUE + 1)) { continue; }

                    for (int dx = 0; dx < SAT_PPT_1D; ++dx) {
                        for (int dy = 0; dy < SAT_PPT_1D; ++dy) {
                            int x = satTileX0 + SAT_PPT_1D * threadIdx.x + dx;
                            int y = satTileY0 + SAT_PPT_1D * threadIdx.y + dy;
                            if (x > sat_width - 1 || y > sat_height - 1) {
                                continue;
                            }
                            int pixelIndex = y * sat_width + x;
                            DTYPE fx = static_cast<DTYPE>(x) - satTileX0;
                            DTYPE fy = static_cast<DTYPE>(y) - satTileY0;

                            // if (d_inside_barycentric(
                            if (d_inside_triangle(
                                    fx, fy, x1, y1, x2, y2, x3, y3)) {
                                DTYPE elev = d_interpolate_three(
                                    fx, fy,
                                    x1, y1, elev1,
                                    x2, y2, elev2,
                                    x3, y3, elev3);
                                if (elev > pOut[pixelIndex]) {
                                    pOut[pixelIndex] = elev;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
/*
*/
}

__global__ void kernelFindDSMBlocksBbox(DTYPE* pX, DTYPE* pY, DTYPE* pZ,
                                        DTYPE* pBbox,
                                        int width, int height) {
    // Each block processes a DSM tile that is referenced by blockIdx

    int dsmStep = DSM_PPB_1D - 1;
    int rowTileOffset = blockIdx.y * dsmStep;
    int colTileOffset = blockIdx.x * dsmStep;
    // thread linear index within a block
    int linearThreadIdx = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ DTYPE cacheX0[TPB];
    __shared__ DTYPE cacheX1[TPB];
    __shared__ DTYPE cacheY0[TPB];
    __shared__ DTYPE cacheY1[TPB];

    // find thread-private local values
    // each thread is allowed to process up to DSM_PPB_1D pixels
    DTYPE localX0 = 1E10;
    DTYPE localY0 = 1E10;
    DTYPE localX1 = -1E10;
    DTYPE localY1 = -1E10;
    for (int i = 0; i < DSM_PPT_1D; ++i) {
        for (int j = 0; j < DSM_PPT_1D; ++j) {
            int y = rowTileOffset + threadIdx.y * DSM_PPT_1D + i;
            int x = colTileOffset + threadIdx.x * DSM_PPT_1D + j;
            if (y >= height || x >= width) { continue; }
            // global pixel index
            int pixelIdx = y * width + x;
            if (pZ[pixelIdx] < DSM_IGNORE_VALUE + 1) { continue; }

            localX0 = fminf(localX0, pX[pixelIdx]);
            localX1 = fmaxf(localX1, pX[pixelIdx]);
            localY0 = fminf(localY0, pY[pixelIdx]);
            localY1 = fmaxf(localY1, pY[pixelIdx]);
        }
    }

    cacheX0[linearThreadIdx] = localX0;
    cacheY0[linearThreadIdx] = localY0;
    cacheX1[linearThreadIdx] = localX1;
    cacheY1[linearThreadIdx] = localY1;
    __syncthreads();

    // reduction op
    int threadsPerBlock = blockDim.x * blockDim.y;
    int i = threadsPerBlock / 2;
    while (i != 0) {
        if (linearThreadIdx < i) {
            cacheX0[linearThreadIdx] = fmin(cacheX0[linearThreadIdx],
                                            cacheX0[linearThreadIdx + i]);
            cacheY0[linearThreadIdx] = fmin(cacheY0[linearThreadIdx],
                                            cacheY0[linearThreadIdx + i]);
            cacheX1[linearThreadIdx] = fmax(cacheX1[linearThreadIdx],
                                            cacheX1[linearThreadIdx + i]);
            cacheY1[linearThreadIdx] = fmax(cacheY1[linearThreadIdx],
                                            cacheY1[linearThreadIdx + i]);
        }

        __syncthreads();
        i /= 2;
    }

    if (linearThreadIdx == 0) {
        int linearBlockIdx = blockIdx.y * gridDim.x + blockIdx.x;
        pBbox[linearBlockIdx * 4 + 0] = cacheX0[0];
        pBbox[linearBlockIdx * 4 + 1] = cacheY0[0];
        pBbox[linearBlockIdx * 4 + 2] = cacheX1[0];
        pBbox[linearBlockIdx * 4 + 3] = cacheY1[0];
    }
}

void cudaRenderSatElevation(DTYPE * pX, DTYPE* pY, DTYPE* pZ, DTYPE* pOut,
                            int dsm_width, int dsm_height, int sat_width, int sat_height) {
    int dsm_npixels = dsm_width * dsm_height;
    int sat_npixels = sat_width * sat_height;
    
    DTYPE* d_pX;
    DTYPE* d_pY;
    DTYPE* d_pZ;
    DTYPE* d_pOut;
    cudaMalloc((void **)&d_pX, sizeof(DTYPE) * dsm_npixels);
    cudaMalloc((void **)&d_pY, sizeof(DTYPE) * dsm_npixels);
    cudaMalloc((void **)&d_pZ, sizeof(DTYPE) * dsm_npixels);
    cudaMalloc((void **)&d_pOut, sizeof(DTYPE) * sat_npixels);
    cudaMemcpy(d_pX, pX, sizeof(DTYPE) * dsm_npixels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pY, pY, sizeof(DTYPE) * dsm_npixels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pZ, pZ, sizeof(DTYPE) * dsm_npixels, cudaMemcpyHostToDevice);
    // output memory on host contains all min values
    cudaMemcpy(d_pOut, pOut, sizeof(DTYPE) * sat_npixels, cudaMemcpyHostToDevice);

    // number of tiles:
    // DSM tiles overlap by 1 pixel,
    // because they are split into triangular faces
    int dsmStep = DSM_PPB_1D - 1;
    int dsmTiles_X = ((dsm_width - DSM_PPB_1D) + dsmStep - 1) / dsmStep + 1;
    int dsmTiles_Y = ((dsm_height - DSM_PPB_1D) + dsmStep - 1) / dsmStep + 1;
    int dsmTiles = dsmTiles_X * dsmTiles_Y;
    // tile bounding boxes
    DTYPE* d_pDSMBbox;
    cudaMalloc((void **)&d_pDSMBbox, sizeof(DTYPE) * 4 * dsmTiles);
    // cudaMemset(d_pDSMBbox, 0, sizeof(DTYPE) * 4 * dsmTiles);

    // blocks are linked to tiles, threads are linked to pixels
    dim3 dsmBlocks(dsmTiles_X, dsmTiles_Y);
    dim3 dsmThreadsPerBlock(TPB_1D, TPB_1D);
    kernelFindDSMBlocksBbox<<<dsmBlocks, dsmThreadsPerBlock>>>(
        d_pX, d_pY, d_pZ,
        d_pDSMBbox,
        dsm_width, dsm_height
    );
    cudaThreadSynchronize();
    // cudaDeviceSynchronize();
    if ( cudaSuccess != cudaGetLastError() )
        printf( "Error in CUDA kernel attempting to find DSM blocks bounding boxes!\n" );

    // // check output:
    // DTYPE x0 = 1E10;
    // DTYPE y0 = 1E10;
    // DTYPE x1 = -1E10;
    // DTYPE y1 = -1E10;
    // int istart = 1600;
    // int jstart = 800;
    // for (int i = istart; i < istart+16; ++i) {
    //     for (int j = jstart; j < jstart+16; ++j) {
    //         int idx = i * width + j;
    //         if (pX[idx] < x0)
    //             x0 = pX[idx];
    //         if (pX[idx] > x1)
    //             x1 = pX[idx];
    //         if (pY[idx] < y0)
    //             y0 = pY[idx];
    //         if (pY[idx] > y1)
    //             y1 = pY[idx];
    //     }
    // }
    // DTYPE tmp[4];
    // cudaMemcpy(tmp, d_pDSMBbox + 4 * ((istart/16)*dsmTiles_X + (jstart/16)), sizeof(DTYPE) * 4, cudaMemcpyDeviceToHost);
    // printf("%d %d\n", dsmTiles_X, dsmTiles_Y);
    // printf("%.3f %.3f %.3f %.3f\n", tmp[0], tmp[1], tmp[2], tmp[3]);
    // printf("%.3f %.3f %.3f %.3f\n", x0, y0, x1, y1);
/*
    int tmpStartTile = 0;
    for (int i = tmpStartTile; i < tmpStartTile + dsmTiles; ++i) {
        DTYPE tmp[4];
        cudaMemcpy(tmp, d_pDSMBbox + 4 * i, sizeof(DTYPE) * 4, cudaMemcpyDeviceToHost);
        // int x = 28;
        bool flag = true;
        for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 32; ++x) {
            if (flag && (tmp[0] <= x) && (tmp[1] <= y) && (tmp[2] >= x) && (tmp[3] >= y)) {
                printf(">>>>>>>>>>>>>>> %d %d, %d\n", i, dsmTiles_X, dsmTiles_Y);
                flag = false;
            }
        }}
        // printf("%.1f %.1f %.1f %.1f\n", tmp[0], tmp[1], tmp[2], tmp[3]);
    }
*/
/*
*/
    // cudaMemcpy(pOut, d_pDSMBbox, sizeof(DTYPE) * 4 * dsmTiles,
    //            cudaMemcpyDeviceToHost);
    // printf("=========================== %d %.3f, %.3f\n", dsmTiles, pOut[0], pOut[1]);
    // return;

    // blocks per grid
    const int BPG_X = (sat_width + SAT_PPB_1D - 1) / SAT_PPB_1D;
    const int BPG_Y = (sat_height + SAT_PPB_1D - 1) / SAT_PPB_1D;
    dim3 satBlocks(BPG_X, BPG_Y);
    dim3 satThreadsPerBlock(TPB_1D, TPB_1D);
    // printf("%d %d %d %d %d %d\n", TPB_1D, TPB, SAT_PPT_1D, SAT_PPT, SAT_PPB_1D, SAT_PPB);
    // printf("============== %d %d\n", BPG_X, BPG_Y);
    kernelRenderSatElevation<<<satBlocks, satThreadsPerBlock>>>(
        d_pX, d_pY, d_pZ, d_pOut,
        d_pDSMBbox,
        dsmTiles, dsmTiles_X,
        dsm_width, dsm_height,
        sat_width, sat_height);
    cudaThreadSynchronize();
    // cudaDeviceSynchronize();
    if ( cudaSuccess != cudaGetLastError() )
        printf( "Error in CUDA kernel attempting to render satellite elevation!\n" );

    cudaMemcpy(pOut, d_pOut, sizeof(DTYPE) * sat_npixels,
               cudaMemcpyDeviceToHost);
/*
    int tmpStartI = 0;
    int tmpStartJ = 0;
    for (int i = tmpStartI; i < tmpStartI + TPB_1D; ++i) {
        for (int j = tmpStartJ; j < tmpStartJ + TPB_1D; ++j) {
            printf("%.3f ", pOut[i * out_width + j]);
        }
        printf("\n");
    }
    printf("\n");
    DTYPE tmpMax = 0;
    DTYPE tmpAvg = 0;
    int tmpCount = 0;
    for (int i = 0; i < out_height * out_width; ++i) {
        tmpMax = fmaxf(tmpMax, pOut[i]);
        tmpAvg += pOut[i];
        if (pOut[i] > 20) {
            tmpCount++;
        }
    }
    printf("%.1f\n", tmpMax);
    printf("%.1f\n", tmpAvg / (out_height * out_width));
    printf("%d\n", tmpCount);
*/
}

