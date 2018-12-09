/*

We use a term *tile* to identify the rectangular submatrices of the image.
Not to be confused with the blocks of threads.

*/

#include <cuda_runtime.h>
#include <stdio.h>

#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

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

__global__ void kernelComputePointsNum(DTYPE* pX, DTYPE* pY, DTYPE* pZ,
                                       int* dsmPixelCounts,
                                       int nfaces, int dsm_width,
                                       int sat_width, int sat_height) {
    int iface = blockIdx.x * blockDim.x + threadIdx.x;
    if (iface < nfaces) {
        int faces_per_row = 2 * (dsm_width - 1);
        int irow = iface / faces_per_row;
        int icol = (iface % faces_per_row) / 2;
        int idx = irow * dsm_width + icol;

        int idx1, idx2, idx3;
        if (iface % 2 == 0) {
            // **
            // *
            idx1 = idx;
            idx2 = idx + 1;
            idx3 = idx + dsm_width;
        } else {
            //  *
            // **
            idx1 = idx + 1;
            idx2 = idx + dsm_width;
            idx3 = idx + dsm_width + 1;
        }

        if (pZ[idx1] < DSM_IGNORE_VALUE + 1 ||
            pZ[idx2] < DSM_IGNORE_VALUE + 1 ||
            pZ[idx3] < DSM_IGNORE_VALUE + 1) { return; }
        
        float x1, y1, x2, y2, x3, y3;
        x1 = pX[idx1];
        y1 = pY[idx1];
        x2 = pX[idx2];
        y2 = pY[idx2];
        x3 = pX[idx3];
        y3 = pY[idx3];
        int ymin = static_cast<int>( ceilf(fminf(fminf(y1, y2), y3)) );
        int xmin = static_cast<int>( ceilf(fminf(fminf(x1, x2), x3)) );
        int ymax = static_cast<int>( floorf(fmaxf(fmaxf(y1, y2), y3)) );
        int xmax = static_cast<int>( floorf(fmaxf(fmaxf(x1, x2), x3)) );

        ymin = fmaxf(0, ymin);
        xmin = fmaxf(0, xmin);
        ymax = fminf(sat_height - 1, ymax);
        xmax = fminf(sat_width - 1, xmax);

        //if ((xmax - xmin) * (ymax - ymin) > 100) {
        //    dsmPixelCounts[iface] = -1;
        //} else {
        {
            for (int x = xmin; x <= xmax; ++x) {
                for (int y = ymin; y <= ymax; ++y) {
                    if (d_inside_triangle((float) x - x1, (float) y - y1,
                                          0, 0, x2-x1, y2-y1, x3-x1, y3-y1)) {
                        dsmPixelCounts[iface] += 1;
                    }
                }
            }
        }
    }
}

__global__ void kernelGetPoints(DTYPE* pX, DTYPE* pY, DTYPE* pZ,
                                int* dsmPixelCounts,
                                int* faceIDs, int* pixelIDs,
                                int nfaces, int dsm_width,
                                int sat_width, int sat_height) {
    int iface = blockIdx.x * blockDim.x + threadIdx.x;
    if (iface < nfaces) {
        int curIdx = dsmPixelCounts[iface];

        int faces_per_row = 2 * (dsm_width - 1);
        int irow = iface / faces_per_row;
        int icol = (iface % faces_per_row) / 2;
        int idx = irow * dsm_width + icol;

        int idx1, idx2, idx3;
        if (iface % 2 == 0) {
            // **
            // *
            idx1 = idx;
            idx2 = idx + 1;
            idx3 = idx + dsm_width;
        } else {
            //  *
            // **
            idx1 = idx + 1;
            idx2 = idx + dsm_width;
            idx3 = idx + dsm_width + 1;
        }

        if (pZ[idx1] < DSM_IGNORE_VALUE + 1 ||
            pZ[idx2] < DSM_IGNORE_VALUE + 1 ||
            pZ[idx3] < DSM_IGNORE_VALUE + 1) { return; }
        
        float x1, y1, x2, y2, x3, y3;
        x1 = pX[idx1];
        y1 = pY[idx1];
        x2 = pX[idx2];
        y2 = pY[idx2];
        x3 = pX[idx3];
        y3 = pY[idx3];
        int ymin = static_cast<int>( ceilf(fminf(fminf(y1, y2), y3)) );
        int xmin = static_cast<int>( ceilf(fminf(fminf(x1, x2), x3)) );
        int ymax = static_cast<int>( floorf(fmaxf(fmaxf(y1, y2), y3)) );
        int xmax = static_cast<int>( floorf(fmaxf(fmaxf(x1, x2), x3)) );

        ymin = fmaxf(0, ymin);
        xmin = fmaxf(0, xmin);
        ymax = fminf(sat_height - 1, ymax);
        xmax = fminf(sat_width - 1, xmax);

        //if ((xmax - xmin) * (ymax - ymin) > 100) {
        //    dsmPixelCounts[iface] = -1;
        //} else {
        {
            for (int x = xmin; x <= xmax; ++x) {
                for (int y = ymin; y <= ymax; ++y) {
                    if (d_inside_triangle((float) x - x1, (float) y - y1,
                                          0, 0, x2-x1, y2-y1, x3-x1, y3-y1)) {
                        faceIDs[curIdx] = iface;
                        pixelIDs[curIdx] = y * sat_width + x;
                        curIdx++;
                    }
                }
            }
        }
    }
}

__global__ void kernelFindLimits(int* ids, int* limits, int num) {
    int iel = blockIdx.x * blockDim.x + threadIdx.x;
    if (iel < num) {
        int pixelID = ids[iel];
        if (iel == 0 || ids[iel - 1] != pixelID) {
            limits[pixelID * 2 + 0] = iel;
        }
        if (iel == num - 1 || ids[iel + 1] != pixelID) {
            limits[pixelID * 2 + 1] = iel + 1;
        }
    }
}

__global__ void kernelDraw(int* faceIDs, int* pixelIDsLimits,
                           float* pX, float* pY, float* pZ,
                           float* pOut,
                           int sat_npixels, int dsm_width, int sat_width) {
    int ipixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (ipixel < sat_npixels) {
        int faces_per_row = 2 * (dsm_width - 1);
        for (int i = pixelIDsLimits[2 * ipixel + 0];
                 i < pixelIDsLimits[2 * ipixel + 1]; ++i) {
            int iface = faceIDs[i];

            int irow = iface / faces_per_row;
            int icol = (iface % faces_per_row) / 2;
            int idx = irow * dsm_width + icol;

            int idx1, idx2, idx3;
            if (iface % 2 == 0) {
                // **
                // *
                idx1 = idx;
                idx2 = idx + 1;
                idx3 = idx + dsm_width;
            } else {
                //  *
                // **
                idx1 = idx + 1;
                idx2 = idx + dsm_width;
                idx3 = idx + dsm_width + 1;
            }

            float x1, y1, elev1, x2, y2, elev2, x3, y3, elev3;
            x1 = pX[idx1];
            y1 = pY[idx1];
            elev1 = pZ[idx1];
            x2 = pX[idx2];
            y2 = pY[idx2];
            elev2 = pZ[idx2];
            x3 = pX[idx3];
            y3 = pY[idx3];
            elev3 = pZ[idx3];

            float x = static_cast<float>(ipixel % sat_width);
            float y = static_cast<float>(ipixel / sat_width);
            
            float elev = d_interpolate_three(x, y,
                                             x1, y1, elev1,
                                             x2, y2, elev2,
                                             x3, y3, elev3);

            if (elev > pOut[ipixel]) {
                pOut[ipixel] = elev;
            }
        }
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

    int nfaces = 2 * (dsm_height - 1) * (dsm_width - 1);
    int nblocks = (nfaces + TPB - 1) / TPB;
    // compute # of pixels that each face cover
    // TODO: change to int
    int* dsmPixelCounts;
    cudaMalloc((void **)&dsmPixelCounts, sizeof(int) * nfaces);
    cudaMemset(dsmPixelCounts, 0, sizeof(int) * nfaces);
    kernelComputePointsNum<<<nblocks, TPB>>>(d_pX, d_pY, d_pZ,
                                             dsmPixelCounts, nfaces,
                                             dsm_width, sat_width, sat_height);
    // cudaThreadSynchronize();
    cudaDeviceSynchronize();
    if ( cudaSuccess != cudaGetLastError() )
        printf( "Error in CUDA kernel attempting to compute number of points "
                "for each thread!\n" );

    int numPixelsLast;
    cudaMemcpy(&numPixelsLast, dsmPixelCounts + nfaces - 1, sizeof(int),
               cudaMemcpyDeviceToHost);

    // exclusive scan to get start index for each face
    thrust::exclusive_scan(thrust::device, dsmPixelCounts,
                           dsmPixelCounts + nfaces, dsmPixelCounts);

    //
    int numPixelsTotal;
    cudaMemcpy(&numPixelsTotal, dsmPixelCounts + nfaces - 1, sizeof(int),
               cudaMemcpyDeviceToHost);
    numPixelsTotal += numPixelsLast;
    printf("================= %d\n", numPixelsTotal);
    int* faceIDs;
    int* pixelIDs;
    cudaMalloc((void **)&faceIDs, sizeof(int) * numPixelsTotal);
    cudaMalloc((void **)&pixelIDs, sizeof(int) * numPixelsTotal);
    kernelGetPoints<<<nblocks, TPB>>>(d_pX, d_pY, d_pZ,
                                      dsmPixelCounts,
                                      faceIDs, pixelIDs,
                                      nfaces,
                                      dsm_width, sat_width, sat_height);
    cudaDeviceSynchronize();
    if ( cudaSuccess != cudaGetLastError() )
        printf( "Error in CUDA kernel attempting to "
                "get points ids for each face!\n" );

    // sort by key
    thrust::sort_by_key(thrust::device, pixelIDs, pixelIDs + numPixelsTotal,
                        faceIDs);
    cudaDeviceSynchronize();
    if ( cudaSuccess != cudaGetLastError() )
        printf( "Error in CUDA kernel attempting to "
                "sort!\n" );
   
    // find start and end points for each pixel
    int* pixelIDsLimits;
    cudaMalloc((void **)&pixelIDsLimits, 2 * sizeof(int) * sat_npixels);
    cudaMemset(pixelIDsLimits, 0, 2 * sizeof(int) * sat_npixels);
    nblocks = (numPixelsTotal + TPB - 1) / TPB;
    kernelFindLimits<<<nblocks, TPB>>>(pixelIDs, pixelIDsLimits, numPixelsTotal);
    cudaDeviceSynchronize();
    if ( cudaSuccess != cudaGetLastError() )
        printf( "Error in CUDA kernel attempting to "
                "find start and end positions for each pixel!\n" );

    //
    nblocks = (sat_npixels + TPB - 1) / TPB;
    kernelDraw<<<nblocks, TPB>>>(faceIDs, pixelIDsLimits, d_pX, d_pY, d_pZ,
                                d_pOut,
                                sat_npixels, dsm_width, sat_width);
    cudaDeviceSynchronize();
    if ( cudaSuccess != cudaGetLastError() )
        printf( "Error in CUDA kernel attempting to "
                "draw satellite elevation!\n" );

    
    // cudaMemcpy(pOut, dsmPixelCounts, sizeof(float) * min(sat_npixels, nfaces), cudaMemcpyDeviceToHost);
    cudaMemcpy(pOut, d_pOut, sizeof(DTYPE) * sat_npixels,
               cudaMemcpyDeviceToHost);
}

