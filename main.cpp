/*
A utility for fast ortho-rectification

Misha Shvets <mshvets@cs.unc.edu>
*/


#include <omp.h>
#include <math.h>
#include "gdal_alg.h"
#include "gdal_priv.h"
#include "cpl_conv.h"
#include "ogr_spatialref.h"

#include "opencv2/opencv.hpp"

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

void draw_triangle(double x1, double y1, double elev1,
                   double x2, double y2, double elev2,
                   double x3, double y3, double elev3,
                   int cols, int rows,
                   float * pImg) {
    //               cv::Mat * pImg) {
    // int cols = pImg->cols;
    // int rows = pImg->rows;

    int imin = (int) round(minthree<double>(y1, y2, y3));
    int imax = (int) round(maxthree<double>(y1, y2, y3));
    int jmin = (int) round(minthree<double>(x1, x2, x3));
    int jmax = (int) round(maxthree<double>(x1, x2, x3));

    for (int i = imin; i < imax + 1; ++i) {
        // j1 = 
        // jmin = (int) round(minthree<double>(j1, j2, j3));
        // jmax = (int) round(maxthree<double>(j1, j2, j3));
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
                if (elev > pImg[idx]) {
                    pImg[idx] = static_cast<float>(elev);
                }
            }
        }
    }
}

void readImage(const std::string sat_filepath, std::string dsm_filepath) {
    double dsmIgnoreValue = 0.;
    double time_start;

    GDALAllRegister();

    // Read satellite image
    GDALDataset * sat_dataset;
    sat_dataset = (GDALDataset *) GDALOpen(sat_filepath.c_str(), GA_ReadOnly);
    int sat_cols = sat_dataset->GetRasterXSize();
    int sat_rows = sat_dataset->GetRasterYSize();
    int channels = sat_dataset->GetRasterCount();

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
    pBand->RasterIO(GF_Read,           // GDALRWFlag eRWFlag
                    0, 0,              // (xOff, yOff)
                    sat_cols, sat_rows,        // (xSize, ySize)
                    pBuffer,           // buffer
                    sat_cols, sat_rows,        // buffer (xSize, ySize)
                    GDT_Float64,       // GDALDataType eBufType
                    0, 0);             // nPixelSpace, nLineSpace

    // Transform to OpenCV matrix
    cv::Mat img(sat_rows, sat_cols, CV_8UC1);
    for (int i = 0; i < sat_rows * sat_cols; ++i) {
        img.data[i] = static_cast<uchar>(std::round(
            (pBuffer[i] - adfMinMax[0]) / (adfMinMax[1] - adfMinMax[0]) * 255.
        ));
    }
    cv::imwrite("tmp.png", img);
    printf("Finish reading the satellite image in %.3fs\n", omp_get_wtime() - time_start);

    // Read DSM image
    time_start = omp_get_wtime();
    GDALDataset * dsm_dataset;
    dsm_dataset = (GDALDataset *) GDALOpen(dsm_filepath.c_str(), GA_ReadOnly);
    int dsm_cols = dsm_dataset->GetRasterXSize();
    int dsm_rows = dsm_dataset->GetRasterYSize();
    int dsm_channels = dsm_dataset->GetRasterCount();
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
    pDSMBand->RasterIO(GF_Read,             // GDALRWFlag eRWFlag
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
    time_start = omp_get_wtime();
    pixel2geo(dsm_cols * dsm_rows, pXCoord, pYCoord, dsmGeoTransform);
    printf("Finish converting to geo coordinates %.3fs\n", omp_get_wtime() - time_start);
    // to latlon coordinates
    time_start = omp_get_wtime();
    pCoordTransform->Transform(dsm_cols * dsm_rows, pXCoord, pYCoord, pDSMBuffer);
    printf("Finish converting to lat/long in %.3fs\n", omp_get_wtime() - time_start);
    // RPC projection to sat image coordinates
    time_start = omp_get_wtime();
    int * pSuccess = (int *) CPLMalloc(sizeof(int) * dsm_cols * dsm_rows);
    GDALRPCTransform(pRPCTransform, true, dsm_cols * dsm_rows, pXCoord, pYCoord, pDSMBuffer, pSuccess);
    printf("Finish RPC transform in %.3fs\n", omp_get_wtime() - time_start);

    // Occlussion handling
    time_start = omp_get_wtime();
    // cv::Mat satElevImg(sat_rows, sat_cols, CV_32FC1, cv::Scalar(0));
    float * pSatElevImg;
    pSatElevImg = (float *) CPLMalloc(sizeof(float) * sat_cols * sat_rows);
    std::fill(pSatElevImg, pSatElevImg + sat_cols * sat_rows, dsmMinVal);
    for (int row_d = 0; row_d < dsm_rows - 1; ++row_d) {
        for (int col_d = 0; col_d < dsm_cols - 1; ++col_d) {
            int idx = row_d * dsm_cols + col_d;
            // triangular face **
            //                 *
            // if ((pDSMBuffer[idx] != dsmIgnoreValue) and
            //         (pDSMBuffer[idx + 1] != dsmIgnoreValue) and
            //         (pDSMBuffer[idx + dsm_cols] != dsmIgnoreValue)) {
            if (std::isnormal(pDSMBuffer[idx]) and
                    std::isnormal(pDSMBuffer[idx + 1]) and
                    std::isnormal(pDSMBuffer[idx + dsm_cols])) {
                draw_triangle(pXCoord[idx], pYCoord[idx],
                              pDSMBuffer[idx],
                              pXCoord[idx + 1], pYCoord[idx + 1],
                              pDSMBuffer[idx + 1],
                              pXCoord[idx + dsm_cols], pYCoord[idx + dsm_cols],
                              pDSMBuffer[idx + dsm_cols],
                              sat_cols, sat_rows,
                              pSatElevImg);
            }

            // triangular face  *
            //                 **
            // if ((pDSMBuffer[idx + dsm_cols + 1] != dsmIgnoreValue) and
            //         (pDSMBuffer[idx + 1] != dsmIgnoreValue) and
            //         (pDSMBuffer[idx + dsm_cols] != dsmIgnoreValue)) {
            if (std::isnormal(pDSMBuffer[idx + dsm_cols + 1]) and
                    std::isnormal(pDSMBuffer[idx + 1]) and
                    std::isnormal(pDSMBuffer[idx + dsm_cols])) {
                draw_triangle(pXCoord[idx + dsm_cols + 1], pYCoord[idx + dsm_cols + 1],
                              pDSMBuffer[idx + dsm_cols + 1],
                              pXCoord[idx + 1], pYCoord[idx + 1],
                              pDSMBuffer[idx + 1],
                              pXCoord[idx + dsm_cols], pYCoord[idx + dsm_cols],
                              pDSMBuffer[idx + dsm_cols],
                              sat_cols, sat_rows,
                              pSatElevImg);
            }
        }
    }

    const char *pszFormat = "GTiff";
    GDALDriver *pDriver;
    char **papszMetadata;
    pDriver = GetGDALDriverManager()->GetDriverByName(pszFormat);
    papszMetadata = pDriver->GetMetadata();

    GDALDataset *pSatElevationDS;
    char **papszOptions = NULL;
    const char *pszDstFilename = "sat_elevation.tif";
    pSatElevationDS = pDriver->Create(pszDstFilename, sat_cols, sat_rows,
                                      1, GDT_Float32, papszOptions);

    GDALRasterBand *poBand;
    poBand = pSatElevationDS->GetRasterBand(1);
    poBand->RasterIO(GF_Write, 0, 0, sat_cols, sat_rows,
                     pSatElevImg, sat_cols, sat_rows, GDT_Float32, 0, 0);

    GDALClose((GDALDatasetH) pSatElevationDS);

    printf("Finish sattelite elevation rendering in %.3fs\n", omp_get_wtime() - time_start);

    // Image re-collection
    time_start = omp_get_wtime();
    cv::Mat ortho_img(dsm_rows, dsm_cols, CV_8UC1);
    for (int row_d = 0; row_d < dsm_rows; ++row_d) {
        for (int col_d = 0; col_d < dsm_cols; ++col_d) {
            int idx = row_d * dsm_cols + col_d;
            int col_s = static_cast<int>(pXCoord[idx]);
            int row_s = static_cast<int>(pYCoord[idx]);
            int idx_s = row_s * sat_cols + col_s;
            if (std::abs(pSatElevImg[idx_s] - pDSMBuffer[idx]) > 1.) {
                continue;
            }
            if ((col_s >= 0) && (col_s < sat_cols) &&
                    (row_s >= 0) && (row_s < sat_cols)) {
                ortho_img.data[idx] = img.data[idx_s];
            } else {
                ortho_img.data[idx] = 0;
            }
        }
    }
    cv::imwrite("ortho2.png", ortho_img);
    printf("Finish image re-collection in %.3fs\n", omp_get_wtime() - time_start);

    GDALDestroyRPCTransformer(pRPCTransform);

}

int main() {
    double time_total = omp_get_wtime();
    // std::string sat_filepath = "data/15APR10184641-P1BS-500647760060_01_P001.tif";
    // std::string dsm_filepath = "data/D3_AOI_DSM.tif";
    std::string sat_filepath = "data/15JAN21161243-P1BS-500648061050_01_P001.tif";
    std::string dsm_filepath = "data/D4_AOI_DSM.tif";
    readImage(sat_filepath, dsm_filepath);

    printf("Total execution time: %.3fs\n", omp_get_wtime() - time_total);
    return 0;
}

