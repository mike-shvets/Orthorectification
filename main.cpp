/*
A utility for fast ortho-rectification

Misha Shvets <mshvets@cs.unc.edu>
*/


#include <omp.h>
#include <math.h>
#include <type_traits>
#include "gdal_alg.h"
#include "gdal_priv.h"
#include "cpl_conv.h"
#include "ogr_spatialref.h"

const float DSM_IGNORE_VALUE = -1E5;

void cudaRenderSatElevation(float * pX, float* pY, float* pZ, float* pOut,
                            int width, int height, int out_width, int out_height);

void pixel2geo(int n, double * X, double * Y, double * transform) {
    for (int i = 0; i < n; ++i) {
        double x = transform[0] + X[i] * transform[1] + Y[i] * transform[2];
        double y = transform[3] + X[i] * transform[4] + Y[i] * transform[5];
        X[i] = x;
        Y[i] = y;
    }
}

template <typename T>
T minthree(T a1, T a2, T a3) {
    return (a1 < a2) ? ((a1 < a3) ? a1 : a3) \
                     : ((a2 < a3) ? a2 : a3);
}

template <typename T>
T maxthree(T a1, T a2, T a3) {
    return (a1 < a2) ? ((a2 < a3) ? a3 : a2) \
                     : ((a1 < a3) ? a3 : a1);
}

double dist(double x1, double y1,
            double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return sqrt(dx * dx + dy * dy);
}

double area(double x1, double y1,
            double x2, double y2,
            double x3, double y3) {
    return std::abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2;
}

double interpolate_thee(double x, double y,
                        double x1, double y1, double v1,
                        double x2, double y2, double v2,
                        double x3, double y3, double v3) {
   // double w1 = 1.0 / dist(x, y, x1, y1);
   // double w2 = 1.0 / dist(x, y, x2, y2);
   // double w3 = 1.0 / dist(x, y, x3, y3);
   // return (w1 * v1 + w2 * v2 + w3 * v3) / (w1 + w2 + w3);
   double denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
   double w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom;
   double w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom;
   double w3 = 1. - w1 - w2;

   return (w1 * v1 + w2 * v2 + w3 * v3);
}

bool inside_triangle(double x, double y,
                     double x1, double y1,
                     double x2, double y2,
                     double x3, double y3) {
    double A = area(x1, y1, x2, y2, x3, y3);    
    double A1 = area(x, y, x1, y1, x2, y2);
    double A2 = area(x, y, x3, y3, x1, y1);
    double A3 = area(x, y, x2, y2, x3, y3);

    return (std::abs(A1 + A2 + A3 - A) < 1E-5);
}

int draw_triangle(double x1, double y1, double elev1,
                   double x2, double y2, double elev2,
                   double x3, double y3, double elev3,
                   int cols, int rows,
                   float * pImg) {
    int imin = (int) ceil(minthree<double>(y1, y2, y3));
    int imax = (int) floor(maxthree<double>(y1, y2, y3));
    int jmin = (int) ceil(minthree<double>(x1, x2, x3));
    int jmax = (int) floor(maxthree<double>(x1, x2, x3));

    // printf("%d\n", (imax - imin + 1) * (jmax - jmin + 1));

    for (int i = imin; i < imax + 1; ++i) {
        for (int j = jmin; j < jmax + 1; ++j) {
            if ((i < 0) || (i >= rows) || (j < 0) || (j >= cols)) {
                continue;
            }

            int idx = i * cols + j;

            double x = j;
            double y = i;
            if (inside_triangle(x, y, x1, y1, x2, y2, x3, y3)) {
                double elev = interpolate_thee(x, y,
                                               x1, y1, elev1,
                                               x2, y2, elev2,
                                               x3, y3, elev3);
                // #pragma omp atomic
                // pImg[idx] = ((elev > pImg[idx]) ? static_cast<float>(elev) : pImg[idx]);
                if (elev > pImg[idx]) {
                    pImg[idx] = static_cast<float>(elev);
                }
            }
        }
    }

    return (imax - imin + 1) * (jmax - jmin + 1);
}

template <typename T>
void writeGTiff(const char* outFilename, T* pImg, int cols, int rows) {
    GDALDataType gdal_type;
    if (std::is_same<T, float>::value) {
        gdal_type = GDT_Float32;
    } else
    if (std::is_same<T, uint8_t>::value) {
        gdal_type = GDT_Byte;
    }

    const char *pszFormat = "GTiff";
    GDALDriver *pDriver;
    pDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);
    // TODO: add metadata
    // char **papszMetadata;
    // papszMetadata = pDriver->GetMetadata();

    GDALDataset *pDstDS;
    char **papszOptions = NULL;
    pDstDS = pDriver->Create(outFilename, cols, rows,
                             1, gdal_type, papszOptions);

    GDALRasterBand *poBand;
    poBand = pDstDS->GetRasterBand(1);
    CPLErr e = poBand->RasterIO(GF_Write, 0, 0, cols, rows,
                                pImg, cols, rows, gdal_type, 0, 0);

    GDALClose((GDALDatasetH) pDstDS);
}

void readImage(const std::string sat_filepath, std::string dsm_filepath) {
    int par_i;

    // TODO: use ignore value
    double time_start;

    GDALAllRegister();

    // Read satellite image
    GDALDataset * sat_dataset;
    sat_dataset = (GDALDataset *) GDALOpen(sat_filepath.c_str(), GA_ReadOnly);
    int sat_cols = sat_dataset->GetRasterXSize();
    int sat_rows = sat_dataset->GetRasterYSize();
    // int channels = sat_dataset->GetRasterCount();

    GDALRasterBand * pBand;
    pBand = sat_dataset->GetRasterBand(1);
    int bGotMin, bGotMax;
    double adfMinMax[2];
    adfMinMax[0] = GDALGetRasterMinimum(pBand, &bGotMin);
    adfMinMax[1] = GDALGetRasterMaximum(pBand, &bGotMax);
    if( ! (bGotMin && bGotMax) )
        GDALComputeRasterMinMax( pBand, TRUE, adfMinMax );
    printf("Min=%.3f, Max=%.3f\n", adfMinMax[0], adfMinMax[1]);

    time_start = omp_get_wtime();
    double *pBuffer;
    pBuffer = (double *) CPLMalloc(sizeof(double) * sat_cols * sat_rows);
    CPLErr e = pBand->RasterIO(GF_Read,           // GDALRWFlag eRWFlag
                               0, 0,              // (xOff, yOff)
                               sat_cols, sat_rows,        // (xSize, ySize)
                               pBuffer,           // buffer
                               sat_cols, sat_rows,        // buffer (xSize, ySize)
                               GDT_Float64,       // GDALDataType eBufType
                               0, 0);             // nPixelSpace, nLineSpace

    // Transform to OpenCV matrix
    uint8_t* pSatImshow;
    pSatImshow = (uint8_t *) CPLMalloc(sizeof(uint8_t) * sat_cols * sat_rows);
    for (int i = 0; i < sat_rows * sat_cols; ++i) {
        pSatImshow[i] = static_cast<uint8_t>(std::round(
            (pBuffer[i] - adfMinMax[0]) / (adfMinMax[1] - adfMinMax[0]) * 255.
        ));
    }
    const char *satOutFilename = "output/tmp.tif";
    writeGTiff<uint8_t>(satOutFilename, pSatImshow, sat_cols, sat_rows);
    printf("Finish reading the satellite image in %.3fs\n", omp_get_wtime() - time_start);

    // Read DSM image
    time_start = omp_get_wtime();
    GDALDataset * dsm_dataset;
    dsm_dataset = (GDALDataset *) GDALOpen(dsm_filepath.c_str(), GA_ReadOnly);
    int dsm_cols = dsm_dataset->GetRasterXSize();
    int dsm_rows = dsm_dataset->GetRasterYSize();
    // int dsm_channels = dsm_dataset->GetRasterCount();
    double dsmGeoTransform[6];
    dsm_dataset->GetGeoTransform(dsmGeoTransform);

    GDALRasterBand * pDSMBand;
    pDSMBand = dsm_dataset->GetRasterBand(1);

    adfMinMax[0] = GDALGetRasterMinimum(pDSMBand, &bGotMin);
    adfMinMax[1] = GDALGetRasterMaximum(pDSMBand, &bGotMax);
    if( ! (bGotMin && bGotMax) )
        GDALComputeRasterMinMax( pDSMBand, TRUE, adfMinMax );
    printf("DSM: Min=%.3f, Max=%.3f\n", adfMinMax[0], adfMinMax[1]);
    double dsmMinVal = adfMinMax[0];

    double *pDSMBuffer;
    pDSMBuffer = (double *) CPLMalloc(sizeof(double) * dsm_cols * dsm_rows);
    e = pDSMBand->RasterIO(GF_Read,             // GDALRWFlag eRWFlag
                           0, 0,                // (xOff, yOff)
                           dsm_cols, dsm_rows,  // (xSize, ySize)
                           pDSMBuffer,          // buffer
                           dsm_cols, dsm_rows,  // buffer (xSize, ySize)
                           GDT_Float64,         // GDALDataType eBufType
                           0, 0);               // nPixelSpace, nLineSpace
    printf("Finish reading the DSM image in %.3fs\n", omp_get_wtime() - time_start);


    // TODO (mshvets): check if GDAL reads RPC info from RPB or from metadata
    // RPC based transformer ... src is pixel/line/elev, dst is long/lat/elev
    GDALRPCInfo rpc;
    char ** rpc_meta;
    rpc_meta = sat_dataset->GetMetadata("RPC");
    GDALExtractRPCInfo(rpc_meta, &rpc);

    void * pRPCTransform;
    pRPCTransform = GDALCreateRPCTransformer(&rpc, false, 0.1, nullptr);

    const char * projection = dsm_dataset->GetProjectionRef();
    OGRSpatialReference * src = new OGRSpatialReference(projection);
    OGRSpatialReference * dst = new OGRSpatialReference();
    dst->SetWellKnownGeogCS("WGS84");
    OGRCoordinateTransformation * pCoordTransform;
    pCoordTransform = OGRCreateCoordinateTransformation(src, dst);

    // dsm pixel + elevation
    time_start = omp_get_wtime();
    double * pXCoord = (double *) CPLMalloc(sizeof(double) * dsm_cols * dsm_rows);
    double * pYCoord = (double *) CPLMalloc(sizeof(double) * dsm_cols * dsm_rows);
    for (int row_d = 0; row_d < dsm_rows; ++row_d) {
        for (int col_d = 0; col_d < dsm_cols; ++col_d) {
            int idx = row_d * dsm_cols + col_d;
            pXCoord[idx] = static_cast<double>(col_d);
            pYCoord[idx] = static_cast<double>(row_d);
        }
    }
    printf("Finish grid generation in %.3fs\n", omp_get_wtime() - time_start);
    

    // to geo coordinates
    // time_start = omp_get_wtime();
    // pixel2geo(dsm_cols * dsm_rows, pXCoord, pYCoord, dsmGeoTransform);
    // printf("Finish converting to geo coordinates %.3fs\n", omp_get_wtime() - time_start);
    // to latlon coordinates
    time_start = omp_get_wtime();
    int * pSuccess = (int *) CPLMalloc(sizeof(int) * dsm_cols * dsm_rows);
    
    /*
    pixel2geo(dsm_cols * dsm_rows, pXCoord, pYCoord, dsmGeoTransform);
    pCoordTransform->Transform(dsm_cols * dsm_rows, pXCoord, pYCoord, pDSMBuffer);
    GDALRPCTransform(pRPCTransform, true, dsm_cols * dsm_rows, pXCoord, pYCoord, pDSMBuffer, pSuccess);
    */

    int nthreads = omp_get_max_threads();
    printf("Num threads = %d\n", nthreads);
    int chunk_size = (dsm_cols * dsm_rows + nthreads - 1) / nthreads;
    #pragma omp parallel for shared(pDSMBuffer, pXCoord, pYCoord, pSuccess) private(par_i) schedule(static)
    for (par_i = 0; par_i < nthreads; ++par_i) {
        int offset = par_i * chunk_size;
        int cur_chunk_size = (offset + chunk_size >= dsm_cols * dsm_rows) ?
                             dsm_cols * dsm_rows - offset : chunk_size;
        pixel2geo(cur_chunk_size,
                  pXCoord + offset,
                  pYCoord + offset,
                  dsmGeoTransform);
        pCoordTransform->Transform(cur_chunk_size,
                                   pXCoord + offset,
                                   pYCoord + offset,
                                   pDSMBuffer + offset);
        GDALRPCTransform(pRPCTransform, true, cur_chunk_size,
                         pXCoord + offset,
                         pYCoord + offset,
                         pDSMBuffer + offset,
                         pSuccess + offset);
    }
    // pCoordTransform->Transform(dsm_cols * dsm_rows, pXCoord, pYCoord, pDSMBuffer);
    // printf("Finish converting to lat/long in %.3fs\n", omp_get_wtime() - time_start);
    // RPC projection to sat image coordinates
    // time_start = omp_get_wtime();
    // int * pSuccess = (int *) CPLMalloc(sizeof(int) * dsm_cols * dsm_rows);
    // GDALRPCTransform(pRPCTransform, true, dsm_cols * dsm_rows, pXCoord, pYCoord, pDSMBuffer, pSuccess);
    // printf("Finish RPC transform in %.3fs\n", omp_get_wtime() - time_start);
    printf("Finish coordinate transform in %.3fs\n", omp_get_wtime() - time_start);

    // Occlussion handling
    time_start = omp_get_wtime();
    float * pSatElevImg;
    pSatElevImg = (float *) CPLMalloc(sizeof(float) * sat_cols * sat_rows);
    std::fill(pSatElevImg, pSatElevImg + sat_cols * sat_rows, dsmMinVal);

    float* pXCoord32 = (float *) CPLMalloc(sizeof(float) * dsm_cols * dsm_rows);
    float* pYCoord32 = (float *) CPLMalloc(sizeof(float) * dsm_cols * dsm_rows);
    float* pZCoord32 = (float *) CPLMalloc(sizeof(float) * dsm_cols * dsm_rows);
    for (int idx = 0; idx < dsm_cols * dsm_rows; idx++) {
        if (std::isnormal(pDSMBuffer[idx])) {
            pXCoord32[idx] = static_cast<float>(pXCoord[idx]);
            pYCoord32[idx] = static_cast<float>(pYCoord[idx]);
            pZCoord32[idx] = static_cast<float>(pDSMBuffer[idx]);
        } else {  // missing DSM value
            pXCoord32[idx] = DSM_IGNORE_VALUE;
            pYCoord32[idx] = DSM_IGNORE_VALUE;
            pZCoord32[idx] = DSM_IGNORE_VALUE;
        }
        // pZCoord32[idx] = (std::isnormal(pDSMBuffer[idx]))
        //                  ? static_cast<float>(pDSMBuffer[idx])
        //                  : DSM_IGNORE_VALUE;
    }

    cudaRenderSatElevation(pXCoord32, pYCoord32, pZCoord32, pSatElevImg,
               dsm_cols, dsm_rows, sat_cols, sat_rows);

    writeGTiff<float>("output/sat_elevation_gpu.tif", pSatElevImg, sat_cols, sat_rows);
    printf("Finish GPU call in %.3fs\n", omp_get_wtime() - time_start);
/*
*/
    time_start = omp_get_wtime();
    std::fill(pSatElevImg, pSatElevImg + sat_cols * sat_rows, dsmMinVal);
    // long long num_triangles = 0;
    // long long num_points = 0;
    // #pragma omp parallel for shared(pDSMBuffer, pXCoord, pYCoord, pSatElevImg) private(par_i) schedule(static) collapse(2)
    for (par_i = 0; par_i < dsm_rows - 1; ++par_i) {
        for (int col_d = 0; col_d < dsm_cols - 1; ++col_d) {
            int idx = par_i * dsm_cols + col_d;
            // triangular face **
            //                 *
            // if ((pDSMBuffer[idx] != DSM_IGNORE_VALUE) and
            //         (pDSMBuffer[idx + 1] != DSM_IGNORE_VALUE) and
            //         (pDSMBuffer[idx + dsm_cols] != DSM_IGNORE_VALUE)) {
            if (std::isnormal(pDSMBuffer[idx]) and
                    std::isnormal(pDSMBuffer[idx + 1]) and
                    std::isnormal(pDSMBuffer[idx + dsm_cols])) {
                // num_points += draw_triangle(pXCoord[idx], pYCoord[idx],
                draw_triangle(pXCoord[idx], pYCoord[idx],
                              pDSMBuffer[idx],
                              pXCoord[idx + 1], pYCoord[idx + 1],
                              pDSMBuffer[idx + 1],
                              pXCoord[idx + dsm_cols], pYCoord[idx + dsm_cols],
                              pDSMBuffer[idx + dsm_cols],
                              sat_cols, sat_rows,
                              pSatElevImg);
                // num_triangles += 1;
            }

            // triangular face  *
            //                 **
            // if ((pDSMBuffer[idx + dsm_cols + 1] != DSM_IGNORE_VALUE) and
            //         (pDSMBuffer[idx + 1] != DSM_IGNORE_VALUE) and
            //         (pDSMBuffer[idx + dsm_cols] != DSM_IGNORE_VALUE)) {
            if (std::isnormal(pDSMBuffer[idx + dsm_cols + 1]) and
                    std::isnormal(pDSMBuffer[idx + 1]) and
                    std::isnormal(pDSMBuffer[idx + dsm_cols])) {
                // num_points += draw_triangle(pXCoord[idx + dsm_cols + 1], pYCoord[idx + dsm_cols + 1],
                draw_triangle(pXCoord[idx + dsm_cols + 1], pYCoord[idx + dsm_cols + 1],
                              pDSMBuffer[idx + dsm_cols + 1],
                              pXCoord[idx + 1], pYCoord[idx + 1],
                              pDSMBuffer[idx + 1],
                              pXCoord[idx + dsm_cols], pYCoord[idx + dsm_cols],
                              pDSMBuffer[idx + dsm_cols],
                              sat_cols, sat_rows,
                              pSatElevImg);
                // num_triangles += 1;
            }
        }
    }
    // printf("%lld %lld\n", num_points, num_triangles);
    //printf("Mean points per triangle: %.3f\n", float(num_points) / num_triangles);

    const char *pszDstFilename = "output/sat_elevation.tif";
    writeGTiff<float>(pszDstFilename, pSatElevImg, sat_cols, sat_rows);
    printf("Finish sattelite elevation rendering in %.3fs\n", omp_get_wtime() - time_start);

    // Image re-collection
    time_start = omp_get_wtime();
    uint8_t* pOrthoImshow;
    pOrthoImshow = (uint8_t *) CPLMalloc(sizeof(uint8_t) * dsm_cols * dsm_rows);
    // printf("%d\n", dsm_cols * dsm_rows);
    for (int row_d = 0; row_d < dsm_rows; ++row_d) {
        for (int col_d = 0; col_d < dsm_cols; ++col_d) {
            int idx = row_d * dsm_cols + col_d;
            int col_s = static_cast<int>(pXCoord[idx]);
            int row_s = static_cast<int>(pYCoord[idx]);
            int idx_s = row_s * sat_cols + col_s;
            // printf("%d\n", idx_s);
            // if (std::abs(pSatElevImg[idx_s] - pDSMBuffer[idx]) > 1.) {
            //     continue;
            // }
            if ((col_s >= 0) && (col_s < sat_cols) &&
                    (row_s >= 0) && (row_s < sat_rows) &&
                    std::abs(pSatElevImg[idx_s] - pDSMBuffer[idx]) < 1.) {
                pOrthoImshow[idx] = pSatImshow[idx_s];
            } else {
                pOrthoImshow[idx] = 0;
            }
        }
    }
    const char* orthoOutFilename = "output/ortho2.tif";
    writeGTiff<uint8_t>(orthoOutFilename, pOrthoImshow, dsm_cols, dsm_rows);
    printf("Finish image re-collection in %.3fs\n", omp_get_wtime() - time_start);

    GDALDestroyRPCTransformer(pRPCTransform);

}

int main(int argc, char** argv) {
    double time_total = omp_get_wtime();

    if (argc != 3) {
        printf("Error: exactly two arguments must be provided: "
                "satellite image path, and DSM image path\n");
        return 1;
    }

    std::string sat_filepath(argv[1]);
    std::string dsm_filepath(argv[2]);
    readImage(sat_filepath, dsm_filepath);

    printf("Total execution time: %.3fs\n", omp_get_wtime() - time_total);
    return 0;
}

