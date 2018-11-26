/*
A utility for fast ortho-rectification

Misha Shvets <mshvets@cs.unc.edu>
*/


#include "gdal_alg.h"
#include "gdal_priv.h"
#include "cpl_conv.h"
#include "ogr_spatialref.h"

#include "opencv2/opencv.hpp"

void pixel2latlon(double Xpixel, double Yline,
                  double & Xgeo, double & Ygeo, double * transform) {
    Xgeo = transform[0] + Xpixel * transform[1] + Yline * transform[2];
    Ygeo = transform[3] + Xpixel * transform[4] + Yline * transform[5];
}

void readImage(const std::string sat_filepath, std::string dsm_filepath) {
    GDALAllRegister();

    // Read satellite image
    GDALDataset * dataset;
    dataset = (GDALDataset *) GDALOpen(sat_filepath.c_str(), GA_ReadOnly);
    int cols = dataset->GetRasterXSize();
    int rows = dataset->GetRasterYSize();
    int channels = dataset->GetRasterCount();

    GDALRasterBand * pBand;
    pBand = dataset->GetRasterBand(1);
    int bGotMin, bGotMax;
    double adfMinMax[2];
    adfMinMax[0] = GDALGetRasterMinimum(pBand, &bGotMin);
    adfMinMax[1] = GDALGetRasterMaximum(pBand, &bGotMax);
    if( ! (bGotMin && bGotMax) )
        GDALComputeRasterMinMax( pBand, TRUE, adfMinMax );
    printf("Min=%.3f, Max=%.3f\n", adfMinMax[0], adfMinMax[1]);

    float *pBuffer;
    pBuffer = (float *) CPLMalloc(sizeof(float) * cols * rows);
    pBand->RasterIO(GF_Read,           // GDALRWFlag eRWFlag
                    0, 0,              // (xOff, yOff)
                    cols, rows,        // (xSize, ySize)
                    pBuffer,           // buffer
                    cols, rows,        // buffer (xSize, ySize)
                    GDT_Float32,       // GDALDataType eBufType
                    0, 0);             // nPixelSpace, nLineSpace

    // Transform to OpenCV matrix
    cv::Mat image(rows, cols, CV_8UC1);
    for (int i = 0; i < rows * cols; ++i) {
        image.data[i] = static_cast<uchar>(std::round(
            (pBuffer[i] - adfMinMax[0]) / (adfMinMax[1] - adfMinMax[0]) * 255.
        ));
    }
    cv::imwrite("tmp.png", image);

    // Read DSM image
    GDALDataset * dsm_dataset;
    dsm_dataset = (GDALDataset *) GDALOpen(dsm_filepath.c_str(), GA_ReadOnly);
    int dsm_cols = dsm_dataset->GetRasterXSize();
    int dsm_rows = dsm_dataset->GetRasterYSize();
    int dsm_channels = dsm_dataset->GetRasterCount();
    double dsmGeoTransform[6];
    dsm_dataset->GetGeoTransform(dsmGeoTransform);

    GDALRasterBand * pDSMBand;
    pDSMBand = dsm_dataset->GetRasterBand(1);
    float *pDSMBuffer;
    pDSMBuffer = (float *) CPLMalloc(sizeof(float) * dsm_cols * dsm_rows);
    pDSMBand->RasterIO(GF_Read,           // GDALRWFlag eRWFlag
                       0, 0,              // (xOff, yOff)
                       dsm_cols, dsm_rows,        // (xSize, ySize)
                       pDSMBuffer,           // buffer
                       dsm_cols, dsm_rows,        // buffer (xSize, ySize)
                       GDT_Float32,       // GDALDataType eBufType
                       0, 0);             // nPixelSpace, nLineSpace


    // TODO (mshvets): check if GDAL reads RPC info from RPB or from metadata
    // RPC based transformer ... src is pixel/line/elev, dst is long/lat/elev
    GDALRPCInfo rpc;
    char ** rpc_meta;
    rpc_meta = dataset->GetMetadata("RPC");
    GDALExtractRPCInfo(rpc_meta, &rpc);

    void * pTransform;
    pTransform = GDALCreateRPCTransformer(&rpc, false, 0.1, nullptr);

    // dsm pixel + elevation
    int col_d = 100;
    int row_d = 200;
    double elev = static_cast<double>(pDSMBuffer[row_d * dsm_cols + col_d]);
    printf("col_d=%.3f, row_d=%.3f, elev=%.3f\n", float(col_d), float(row_d), float(elev));
    // dsm geo + elevation
    double geoX_d, geoY_d;
    pixel2latlon(static_cast<double>(col_d),
                 static_cast<double>(row_d),
                 geoX_d, geoY_d, dsmGeoTransform);
    printf("geoX_d=%.3f, geoY_d=%.3f, elev=%.3f\n", float(geoX_d), float(geoY_d), float(elev));
    // dsm latlon
    const char * projection = dsm_dataset->GetProjectionRef();
    OGRSpatialReference * src = new OGRSpatialReference(projection);
    OGRSpatialReference * dst = new OGRSpatialReference();
    dst->SetWellKnownGeogCS("WGS84");
    OGRCoordinateTransformation * ct = OGRCreateCoordinateTransformation(src, dst);
    double lon_d, lat_d;
    lon_d = geoX_d;
    lat_d = geoY_d;
    double elev_d = elev;
    ct->Transform(1, &lon_d, &lat_d, &elev_d);
    printf("lon_d=%.3f, lat_d=%.3f, elev=%.3f\n", float(lon_d), float(lat_d), float(elev));
    // sat pixels
    double lng_to_pixel = lon_d;
    double lat_to_line = lat_d;
    double z = elev;
    int success = 0;
    GDALRPCTransform(pTransform, true, 1, &lng_to_pixel, &lat_to_line, &z, &success);
    printf("col_s=%.3f, row_s=%.3f, elev=%.3f\n", float(lng_to_pixel), float(lat_to_line), float(z));

    GDALDestroyRPCTransformer(pTransform);

}


int main() {
    std::string sat_filepath = "data/15APR10184641-P1BS-500647760060_01_P001.tif";
    std::string dsm_filepath = "data/D3_AOI_DSM.tif";
    readImage(sat_filepath, dsm_filepath);
    return 0;
}

