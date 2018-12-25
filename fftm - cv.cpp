#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "parameter.h"

#include <VX/vx.h>
#include <VX/vxu.h>
#include <VX/vx_api.h>
#include <VX/vx_khr_cnn.h>
#include <VX/vx_lib_extras.h>


using namespace std;
using namespace cv;

Mat LogPolarFFTTemplateMatch(Mat im0, Mat im1, double canny_threshold1, double canny_threshold2, int idx)
{
    //im0 ==== before
    //im1 ==== now
    Canny(im1, im1, canny_threshold2, canny_threshold1, 3, 1);
    if(im0.type()!=CV_32FC1)
    im0.convertTo(im0, CV_32FC1, 1.0 / 255.0);
    if(im1.type()!=CV_32FC1)
    im1.convertTo(im1, CV_32FC1, 1.0 / 255.0);

    Mat im1_ROI = im1(Rect(2,26,256,128));
	Mat im0_ROI = im0(Rect(2,26,256,128));

 
    clock_t a = clock();
    Point2d tr = cv::phaseCorrelate(im1_ROI, im0_ROI);
    clock_t b = clock();
    if(DEBUG_MSG)
        cout<< "PhaseCorrelate time  is: " << static_cast<double>(b - a) / CLOCKS_PER_SEC * 1000 << "ms" << endl;   
    
	Mat mov_mat = Mat::zeros(Size(3, 2), CV_64FC1);

	mov_mat.at<double>(0, 0) = 1.0;
	mov_mat.at<double>(0, 1) = 0.0;
	mov_mat.at<double>(1, 0) = 0.0;
	mov_mat.at<double>(1, 1) = 1.0;

	mov_mat.at<double>(0, 2) = -tr.x;
	mov_mat.at<double>(1, 2) = -tr.y;

	return mov_mat;

}

static vx_status  status = VX_SUCCESS;
static vx_context context = NULL;
static vx_int32 m = 0, n = 0;
static vx_int32   i = 0;
static vx_int32   j = 0;
static Size insize = Size(256,128);
static Mat C_planes[] = { Mat::zeros(insize,CV_32F) , Mat::zeros(insize,CV_32F) }; 
static Mat past_im;
static Mat imx(Size(256,128),CV_32FC1,Scalar::all(0));
static bool flag = true;

static vx_uint32 vxcGetTypeSize(vx_enum format)
{
    switch(format)
    {
    case VX_TYPE_INT8:
    case VX_TYPE_UINT8:
        return 1;
    case VX_TYPE_INT16:
    case VX_TYPE_UINT16:
        return 2;
    case VX_TYPE_INT32:
    case VX_TYPE_UINT32:
        return 4;
    case VX_TYPE_INT64:
    case VX_TYPE_UINT64:
        return 8;
    case VX_TYPE_FLOAT32:
        return 4;
    case VX_TYPE_FLOAT64:
        return 8;
    case VX_TYPE_ENUM:
        return 4;
    case VX_TYPE_FLOAT16:
        return 2;
    }

    return 4;
}

static int calculate_M(int len)
{
    int i;
    int k;

    i = 0;
    k = 1;
    while(k < len)
    {
        k = k*2;
        i++;
    }

    return i;
}

static Point2d weightedCentroid(InputArray _src, cv::Point peakLocation, cv::Size weightBoxSize, double* response)
{
    Mat src = _src.getMat();

    int type = src.type();
    CV_Assert( type == CV_32FC1 || type == CV_64FC1 );

    int minr = peakLocation.y - (weightBoxSize.height >> 1);
    int maxr = peakLocation.y + (weightBoxSize.height >> 1);
    int minc = peakLocation.x - (weightBoxSize.width  >> 1);
    int maxc = peakLocation.x + (weightBoxSize.width  >> 1);

    Point2d centroid;
    double sumIntensity = 0.0;

    // clamp the values to min and max if needed.
    if(minr < 0)
    {
        minr = 0;
    }

    if(minc < 0)
    {
        minc = 0;
    }

    if(maxr > src.rows - 1)
    {
        maxr = src.rows - 1;
    }

    if(maxc > src.cols - 1)
    {
        maxc = src.cols - 1;
    }

    if(type == CV_32FC1)
    {
        const float* dataIn = src.ptr<float>();
        dataIn += minr*src.cols;
        for(int y = minr; y <= maxr; y++)
        {
            for(int x = minc; x <= maxc; x++)
            {
                centroid.x   += (double)x*dataIn[x];
                centroid.y   += (double)y*dataIn[x];
                sumIntensity += (double)dataIn[x];
            }

            dataIn += src.cols;
        }
    }
    else
    {
        const double* dataIn = src.ptr<double>();
        dataIn += minr*src.cols;
        for(int y = minr; y <= maxr; y++)
        {
            for(int x = minc; x <= maxc; x++)
            {
                centroid.x   += (double)x*dataIn[x];
                centroid.y   += (double)y*dataIn[x];
                sumIntensity += dataIn[x];
            }

            dataIn += src.cols;
        }
    }

    if(response)
        *response = sumIntensity;

    sumIntensity += DBL_EPSILON; // prevent div0 problems...

    centroid.x /= sumIntensity;
    centroid.y /= sumIntensity;

    return centroid;
}

static void magSpectrums( InputArray _src, OutputArray _dst)
{
    Mat src = _src.getMat();
    int depth = src.depth(), cn = src.channels(), type = src.type();
    int rows = src.rows, cols = src.cols;
    int j, k;

    CV_Assert( type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2 );

    if(src.depth() == CV_32F)
        _dst.create( src.rows, src.cols, CV_32FC1 );
    else
        _dst.create( src.rows, src.cols, CV_64FC1 );

    Mat dst = _dst.getMat();
    dst.setTo(0);//Mat elements are not equal to zero by default!

    bool is_1d = (rows == 1 || (cols == 1 && src.isContinuous() && dst.isContinuous()));

    if( is_1d )
        cols = cols + rows - 1, rows = 1;

    int ncols = cols*cn;
    int j0 = cn == 1;
    int j1 = ncols - (cols % 2 == 0 && cn == 1);

    if( depth == CV_32F )
    {
        const float* dataSrc = src.ptr<float>();
        float* dataDst = dst.ptr<float>();

        size_t stepSrc = src.step/sizeof(dataSrc[0]);
        size_t stepDst = dst.step/sizeof(dataDst[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataSrc += cols - 1, dataDst += cols - 1;
                dataDst[0] = dataSrc[0]*dataSrc[0];
                if( rows % 2 == 0 )
                    dataDst[(rows-1)*stepDst] = dataSrc[(rows-1)*stepSrc]*dataSrc[(rows-1)*stepSrc];

                for( j = 1; j <= rows - 2; j += 2 )
                {
                    dataDst[j*stepDst] = (float)std::sqrt((double)dataSrc[j*stepSrc]*dataSrc[j*stepSrc] +
                                                          (double)dataSrc[(j+1)*stepSrc]*dataSrc[(j+1)*stepSrc]);
                }

                if( k == 1 )
                    dataSrc -= cols - 1, dataDst -= cols - 1;
            }
        }

        for( ; rows--; dataSrc += stepSrc, dataDst += stepDst )
        {
            if( is_1d && cn == 1 )
            {
                dataDst[0] = dataSrc[0]*dataSrc[0];
                if( cols % 2 == 0 )
                    dataDst[j1] = dataSrc[j1]*dataSrc[j1];
            }

            for( j = j0; j < j1; j += 2 )
            {
                dataDst[j] = (float)std::sqrt((double)dataSrc[j]*dataSrc[j] + (double)dataSrc[j+1]*dataSrc[j+1]);
            }
        }
    }
    else
    {
        const double* dataSrc = src.ptr<double>();
        double* dataDst = dst.ptr<double>();

        size_t stepSrc = src.step/sizeof(dataSrc[0]);
        size_t stepDst = dst.step/sizeof(dataDst[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataSrc += cols - 1, dataDst += cols - 1;
                dataDst[0] = dataSrc[0]*dataSrc[0];
                if( rows % 2 == 0 )
                    dataDst[(rows-1)*stepDst] = dataSrc[(rows-1)*stepSrc]*dataSrc[(rows-1)*stepSrc];

                for( j = 1; j <= rows - 2; j += 2 )
                {
                    dataDst[j*stepDst] = std::sqrt(dataSrc[j*stepSrc]*dataSrc[j*stepSrc] +
                                                   dataSrc[(j+1)*stepSrc]*dataSrc[(j+1)*stepSrc]);
                }

                if( k == 1 )
                    dataSrc -= cols - 1, dataDst -= cols - 1;
            }
        }

        for( ; rows--; dataSrc += stepSrc, dataDst += stepDst )
        {
            if( is_1d && cn == 1 )
            {
                dataDst[0] = dataSrc[0]*dataSrc[0];
                if( cols % 2 == 0 )
                    dataDst[j1] = dataSrc[j1]*dataSrc[j1];
            }

            for( j = j0; j < j1; j += 2 )
            {
                dataDst[j] = std::sqrt(dataSrc[j]*dataSrc[j] + dataSrc[j+1]*dataSrc[j+1]);
            }
        }
    }
}

static void fftShift(InputOutputArray _out)
{
    Mat out = _out.getMat();

    if(out.rows == 1 && out.cols == 1)
    {
        // trivially shifted.
        return;
    }

    std::vector<Mat> planes;
    split(out, planes);

    int xMid = out.cols >> 1;
    int yMid = out.rows >> 1;

    bool is_1d = xMid == 0 || yMid == 0;

    if(is_1d)
    {
        int is_odd = (xMid > 0 && out.cols % 2 == 1) || (yMid > 0 && out.rows % 2 == 1);
        xMid = xMid + yMid;

        for(size_t i = 0; i < planes.size(); i++)
        {
            Mat tmp;
            Mat half0(planes[i], Rect(0, 0, xMid + is_odd, 1));
            Mat half1(planes[i], Rect(xMid + is_odd, 0, xMid, 1));

            half0.copyTo(tmp);
            half1.copyTo(planes[i](Rect(0, 0, xMid, 1)));
            tmp.copyTo(planes[i](Rect(xMid, 0, xMid + is_odd, 1)));
        }
    }
    else
    {
        int isXodd = out.cols % 2 == 1;
        int isYodd = out.rows % 2 == 1;
        for(size_t i = 0; i < planes.size(); i++)
        {
            // perform quadrant swaps...
            Mat q0(planes[i], Rect(0,    0,    xMid + isXodd, yMid + isYodd));
            Mat q1(planes[i], Rect(xMid + isXodd, 0,    xMid, yMid + isYodd));
            Mat q2(planes[i], Rect(0,    yMid + isYodd, xMid + isXodd, yMid));
            Mat q3(planes[i], Rect(xMid + isXodd, yMid + isYodd, xMid, yMid));

            if(!(isXodd || isYodd))
            {
                Mat tmp;
                q0.copyTo(tmp);
                q3.copyTo(q0);
                tmp.copyTo(q3);

                q1.copyTo(tmp);
                q2.copyTo(q1);
                tmp.copyTo(q2);
            }
            else
            {
                Mat tmp0, tmp1, tmp2 ,tmp3;
                q0.copyTo(tmp0);
                q1.copyTo(tmp1);
                q2.copyTo(tmp2);
                q3.copyTo(tmp3);

                tmp0.copyTo(planes[i](Rect(xMid, yMid, xMid + isXodd, yMid + isYodd)));
                tmp3.copyTo(planes[i](Rect(0, 0, xMid, yMid)));

                tmp1.copyTo(planes[i](Rect(0, yMid, xMid, yMid + isYodd)));
                tmp2.copyTo(planes[i](Rect(xMid, 0, xMid + isXodd, yMid)));
            }
        }
    }

    merge(planes, out);
}

static void divSpectrums( InputArray _srcA, InputArray _srcB, OutputArray _dst, int flags, bool conjB)
{
    Mat srcA = _srcA.getMat(), srcB = _srcB.getMat();
    int depth = srcA.depth(), cn = srcA.channels(), type = srcA.type();
    int rows = srcA.rows, cols = srcA.cols;
    int j, k;

    CV_Assert( type == srcB.type() && srcA.size() == srcB.size() );
    CV_Assert( type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2 );

    _dst.create( srcA.rows, srcA.cols, type );
    Mat dst = _dst.getMat();

    CV_Assert(dst.data != srcA.data); // non-inplace check
    CV_Assert(dst.data != srcB.data); // non-inplace check

    bool is_1d = (flags & DFT_ROWS) || (rows == 1 || (cols == 1 &&
             srcA.isContinuous() && srcB.isContinuous() && dst.isContinuous()));

    if( is_1d && !(flags & DFT_ROWS) )
        cols = cols + rows - 1, rows = 1;

    int ncols = cols*cn;
    int j0 = cn == 1;
    int j1 = ncols - (cols % 2 == 0 && cn == 1);

    if( depth == CV_32F )
    {
        const float* dataA = srcA.ptr<float>();
        const float* dataB = srcB.ptr<float>();
        float* dataC = dst.ptr<float>();
        float eps = FLT_EPSILON; // prevent div0 problems

        size_t stepA = srcA.step/sizeof(dataA[0]);
        size_t stepB = srcB.step/sizeof(dataB[0]);
        size_t stepC = dst.step/sizeof(dataC[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( rows % 2 == 0 )
                    dataC[(rows-1)*stepC] = dataA[(rows-1)*stepA] / (dataB[(rows-1)*stepB] + eps);
                if( !conjB )
                    for( j = 1; j <= rows - 2; j += 2 )
                    {
                        double denom = (double)dataB[j*stepB]*dataB[j*stepB] +
                                       (double)dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + (double)eps;

                        double re = (double)dataA[j*stepA]*dataB[j*stepB] +
                                    (double)dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                        double im = (double)dataA[(j+1)*stepA]*dataB[j*stepB] -
                                    (double)dataA[j*stepA]*dataB[(j+1)*stepB];

                        dataC[j*stepC] = (float)(re / denom);
                        dataC[(j+1)*stepC] = (float)(im / denom);
                    }
                else
                    for( j = 1; j <= rows - 2; j += 2 )
                    {

                        double denom = (double)dataB[j*stepB]*dataB[j*stepB] +
                                       (double)dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + (double)eps;

                        double re = (double)dataA[j*stepA]*dataB[j*stepB] -
                                    (double)dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                        double im = (double)dataA[(j+1)*stepA]*dataB[j*stepB] +
                                    (double)dataA[j*stepA]*dataB[(j+1)*stepB];

                        dataC[j*stepC] = (float)(re / denom);
                        dataC[(j+1)*stepC] = (float)(im / denom);
                    }
                if( k == 1 )
                    dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
            }
        }

        for( ; rows--; dataA += stepA, dataB += stepB, dataC += stepC )
        {
            if( is_1d && cn == 1 )
            {
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( cols % 2 == 0 )
                    dataC[j1] = dataA[j1] / (dataB[j1] + eps);
            }

            if( !conjB )
                for( j = j0; j < j1; j += 2 )
                {
                    double denom = (double)(dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps);
                    double re = (double)(dataA[j]*dataB[j] + dataA[j+1]*dataB[j+1]);
                    double im = (double)(dataA[j+1]*dataB[j] - dataA[j]*dataB[j+1]);
                    dataC[j] = (float)(re / denom);
                    dataC[j+1] = (float)(im / denom);
                }
            else
                for( j = j0; j < j1; j += 2 )
                {
                    double denom = (double)(dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps);
                    double re = (double)(dataA[j]*dataB[j] - dataA[j+1]*dataB[j+1]);
                    double im = (double)(dataA[j+1]*dataB[j] + dataA[j]*dataB[j+1]);
                    dataC[j] = (float)(re / denom);
                    dataC[j+1] = (float)(im / denom);
                }
        }
    }
    else
    {
        const double* dataA = srcA.ptr<double>();
        const double* dataB = srcB.ptr<double>();
        double* dataC = dst.ptr<double>();
        double eps = DBL_EPSILON; // prevent div0 problems

        size_t stepA = srcA.step/sizeof(dataA[0]);
        size_t stepB = srcB.step/sizeof(dataB[0]);
        size_t stepC = dst.step/sizeof(dataC[0]);

        if( !is_1d && cn == 1 )
        {
            for( k = 0; k < (cols % 2 ? 1 : 2); k++ )
            {
                if( k == 1 )
                    dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( rows % 2 == 0 )
                    dataC[(rows-1)*stepC] = dataA[(rows-1)*stepA] / (dataB[(rows-1)*stepB] + eps);
                if( !conjB )
                    for( j = 1; j <= rows - 2; j += 2 )
                    {
                        double denom = dataB[j*stepB]*dataB[j*stepB] +
                                       dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + eps;

                        double re = dataA[j*stepA]*dataB[j*stepB] +
                                    dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                        double im = dataA[(j+1)*stepA]*dataB[j*stepB] -
                                    dataA[j*stepA]*dataB[(j+1)*stepB];

                        dataC[j*stepC] = re / denom;
                        dataC[(j+1)*stepC] = im / denom;
                    }
                else
                    for( j = 1; j <= rows - 2; j += 2 )
                    {
                        double denom = dataB[j*stepB]*dataB[j*stepB] +
                                       dataB[(j+1)*stepB]*dataB[(j+1)*stepB] + eps;

                        double re = dataA[j*stepA]*dataB[j*stepB] -
                                    dataA[(j+1)*stepA]*dataB[(j+1)*stepB];

                        double im = dataA[(j+1)*stepA]*dataB[j*stepB] +
                                    dataA[j*stepA]*dataB[(j+1)*stepB];

                        dataC[j*stepC] = re / denom;
                        dataC[(j+1)*stepC] = im / denom;
                    }
                if( k == 1 )
                    dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
            }
        }

        for( ; rows--; dataA += stepA, dataB += stepB, dataC += stepC )
        {
            if( is_1d && cn == 1 )
            {
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if( cols % 2 == 0 )
                    dataC[j1] = dataA[j1] / (dataB[j1] + eps);
            }

            if( !conjB )
                for( j = j0; j < j1; j += 2 )
                {
                    double denom = dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps;
                    double re = dataA[j]*dataB[j] + dataA[j+1]*dataB[j+1];
                    double im = dataA[j+1]*dataB[j] - dataA[j]*dataB[j+1];
                    dataC[j] = re / denom;
                    dataC[j+1] = im / denom;
                }
            else
                for( j = j0; j < j1; j += 2 )
                {
                    double denom = dataB[j]*dataB[j] + dataB[j+1]*dataB[j+1] + eps;
                    double re = dataA[j]*dataB[j] - dataA[j+1]*dataB[j+1];
                    double im = dataA[j+1]*dataB[j] + dataA[j]*dataB[j+1];
                    dataC[j] = re / denom;
                    dataC[j+1] = im / denom;
                }
        }
    }

}

typedef struct 
{
	vx_graph	graph;
	vx_node		node;
	vx_kernel	kernel;
	vx_border_t border;
	int			m;
	int			n;
	vx_scalar			mLen;
	vx_scalar			nLen;
	vx_scalar			nAligned;
	vx_scalar			mAligned;
	int			height;
	int			width;
	vx_array	arrayTmp;
	vx_array   	arrayIn;
	vx_array	arrayOut;
	void * 		mapPtrOut;
	void * 		mapPtrIn;
}VXFFTOBJS;

vx_status vxcFFTLayer(VXFFTOBJS fftobj, vx_array input, vx_array tmpOut, vx_array output, int nLen, int n, int mLen, int m,char *kernelName)
{
	vx_int32	index		 = 0;

    status        = vxLoadKernels(context, "fft");
    if(VX_SUCCESS == status)
    {
        //char kernelName[1024];
        //sprintf(kernelName, "com.vivantecorp.extension.fft_256x128");

        if(fftobj.kernel = vxGetKernelByName(context, kernelName))// match with VXC code
        {
            if(fftobj.node = vxCreateGenericNode(fftobj.graph, fftobj.kernel))
            {
                status |= vxSetParameterByIndex(fftobj.node, index++, (vx_reference)input);
                status |= vxSetParameterByIndex(fftobj.node, index++, (vx_reference)output);
                status |= vxSetParameterByIndex(fftobj.node, index++, (vx_reference)fftobj.nLen);
                status |= vxSetParameterByIndex(fftobj.node, index++, (vx_reference)fftobj.nAligned);
                status |= vxSetParameterByIndex(fftobj.node, index++, (vx_reference)fftobj.mLen);
                status |= vxSetParameterByIndex(fftobj.node, index++, (vx_reference)fftobj.mAligned);

                status |= vxSetNodeAttribute(fftobj.node, VX_NODE_BORDER, &fftobj.border, sizeof(fftobj.border));

                if(status != VX_SUCCESS)
                {
                    status = VX_ERROR_INVALID_PARAMETERS;
                    vxReleaseNode(&fftobj.node);
                    vxReleaseKernel(&fftobj.kernel);
                    return status;
                }
            }
            else
            {
                vxReleaseKernel(&fftobj.kernel);
                return status;
            }
        }
    }
    return status;
}

void init_vx_FFT(VXFFTOBJS & fftobj, int width, int height, int dst_format, bool bFlag)
{
	unsigned long arraySize = 0;
	vx_size itemNum = 0;
	vx_size itemSize = 0;
	vx_map_id mapId = 0;
	char *kernelname[2]={"com.vivantecorp.extension.fft_256x128","com.vivantecorp.extension.ifft_256x128"};
	if(context == NULL)

		context = vxCreateContext();
    fftobj.graph = vxCreateGraph(context);

	fftobj.m = calculate_M(height);    
    fftobj.n = calculate_M(width);	
	fftobj.nLen = vxCreateScalar(context, VX_TYPE_INT32, &width);
	fftobj.mLen = vxCreateScalar(context, VX_TYPE_INT32, &height);
	fftobj.nAligned	= vxCreateScalar(context, VX_TYPE_INT32, &fftobj.n);
	fftobj.mAligned	= vxCreateScalar(context, VX_TYPE_INT32, &fftobj.m);

    arraySize =width * height *2;

	fftobj.arrayIn = vxCreateArray(context, dst_format, arraySize * vxcGetTypeSize(dst_format));
	fftobj.arrayTmp = vxCreateArray(context, dst_format, arraySize * vxcGetTypeSize(dst_format));
	fftobj.arrayOut = vxCreateArray(context, dst_format, arraySize * vxcGetTypeSize(dst_format));

	float val; 
	for( i = 0;i<arraySize;i++)

		vxAddArrayItems(fftobj.arrayIn, 1, &val, 0);

	vx_size capacity ;
	vxQueryArray(fftobj.arrayIn, VX_ARRAY_CAPACITY, &capacity, sizeof(vx_size));
	vxQueryArray(fftobj.arrayIn, VX_ARRAY_NUMITEMS, &itemNum, sizeof(vx_size));
	vxQueryArray(fftobj.arrayIn, VX_ARRAY_ITEMSIZE, &itemSize, sizeof(vx_size));
	
	status = vxMapArrayRange(fftobj.arrayIn, 0, arraySize/*capacity*/, &mapId, &itemSize, (void**)&fftobj.mapPtrIn, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);
	
    vxcFFTLayer(fftobj, fftobj.arrayIn, fftobj.arrayTmp, fftobj.arrayOut, fftobj.width, fftobj.n, fftobj.height, fftobj.m,kernelname[bFlag]); // 0-dft   // 1-idft
	vxVerifyGraph(fftobj.graph);
	
    vxQueryArray(fftobj.arrayOut, VX_ARRAY_NUMITEMS, &itemNum, sizeof(itemNum));
    vxQueryArray(fftobj.arrayOut, VX_ARRAY_ITEMSIZE, &itemSize, sizeof(itemSize));
    status = vxMapArrayRange(fftobj.arrayOut, 0, itemNum/*capacity*/, &mapId, &itemSize, (void**)&fftobj.mapPtrOut, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);	
	
}


void vxFFT(VXFFTOBJS & fftobj, Mat src, Mat &dst)
{
	int nSize = src.rows * src.cols * sizeof(Vec2f);
	
	//@Warnning 替换输入buffer与输出buffer到graph？
	memcpy((Vec2f *)fftobj.mapPtrIn, (Vec2f *)src.data, nSize);
	
    vxProcessGraph(fftobj.graph); 
    
    memcpy((Vec2f *)dst.data, (Vec2f*)fftobj.mapPtrOut, nSize);

}

Point2d PhaseCorrelation2D(Mat src1,Mat &preSrc)
{
	static VXFFTOBJS vxFFTObj[2] = {0};
	Mat srcplanes[] = { Mat_<float>(src1), Mat::zeros(src1.size(),CV_32F) };
	Mat srcimg ;
	merge(srcplanes, 2, srcimg);

	static Mat dstplanes[] = { Mat::zeros(src1.size(),CV_32F) , Mat::zeros(src1.size(),CV_32F) };
	Mat dstimg ;
	merge(dstplanes, 2, dstimg);

	Point 		peakLoc;
	Point2d 	t={0};

	if(vxFFTObj[0].graph == NULL)

		init_vx_FFT(vxFFTObj[0],src1.cols,src1.rows,VX_TYPE_FLOAT32, 0); //DFT;

	if(vxFFTObj[1].graph == NULL)

		init_vx_FFT(vxFFTObj[1],src1.cols,src1.rows,VX_TYPE_FLOAT32, 1); //iDFT;

	vxFFT(vxFFTObj[0], srcimg, dstimg);//vxFFT(padded1,dst_signal_real,dst_signal_imag);

	float tmp;
	clock_t c1 = clock();
	Mat A(src1.rows, src1.cols, CV_32FC2,Scalar(0));  
	Mat C(src1.rows, src1.cols,CV_32FC2,Scalar(0));//必须先创建，然后提供给vxFFT作为输出的buffer；

	Vec2f * pSrc, * pPre, *pDst;

	for (int i=0; i<src1.rows; i++)
	{
		pSrc = dstimg.ptr<Vec2f>(i);
		pPre = preSrc.ptr<Vec2f>(i);
		pDst = A.ptr<Vec2f>(i);
		for (int j=0; j<src1.cols; j++)
		{

			pDst[j][0] = pPre[j][0] * pSrc[j][0] + pPre[j][1] * pSrc[j][1];	//A.at<Vec2f>(i, j)[0] = preSrc.at<Vec2f>(i, j)[0] * dstimg.at<Vec2f>(i, j)[0] - preSrc.at<Vec2f>(i, j)[1] * (-dstimg.at<Vec2f>(i, j)[1]);
			pDst[j][1] = pPre[j][1] * pSrc[j][0] - pPre[j][0] * pSrc[j][1];	//A.at<Vec2f>(i, j)[1] = preSrc.at<Vec2f>(i, j)[0] * (-dstimg.at<Vec2f>(i, j)[1]) + preSrc.at<Vec2f>(i, j)[1] * dstimg.at<Vec2f>(i, j)[0];

			tmp = sqrt(pow(pDst[j][0], 2.0) + pow(pDst[j][1],2.0)) + 0.001;
			pDst[j][0] /= (float)tmp;
			pDst[j][1] /= (float)tmp;
		}
	}

	vxFFT(vxFFTObj[1], A,C);//	C = vxFFT(reall,imagee);

	split(C, C_planes);
	fftShift(C_planes[0]); // shift the energy to the center of the frame.	   
	minMaxLoc(C_planes[0], NULL, NULL, NULL, &peakLoc);
	
	t = weightedCentroid(C_planes[0], peakLoc, Size(5, 5), 0);
	Point2d center((double)src1.cols / 2.0, (double)src1.rows / 2.0);
	double width_offset = center.x - t.x;
	double height_offset = center.y - t.y;
	t.x = width_offset;
	t.y = height_offset;
	preSrc = dstimg;
    return (t);  

}


Mat vx_LogPolarFFTTemplateMatch(Mat im0, Mat im1, double canny_threshold1, double canny_threshold2, int idx)
{
    //im0 ==== before
    //im1 ==== now
    
    Canny(im1, im1, canny_threshold2, canny_threshold1, 3, 1);
    if(im0.type()!=CV_32FC1)
    im0.convertTo(im0, CV_32FC1, 1.0 / 255.0);
    if(im1.type()!=CV_32FC1)
    im1.convertTo(im1, CV_32FC1, 1.0 / 255.0);
	
	Mat im1_ROI = im1(Rect(2,26,256,128));
	if(flag == true)
	{
		Mat im0_ROI = im0(Rect(2,26,256,128));
		PhaseCorrelation2D(im0_ROI, imx); //vx
		flag = false;
	}
	
	clock_t c3 = clock(); 
	Point2d tr = PhaseCorrelation2D(im1_ROI, imx); //vx	
	clock_t c4 = clock(); 
    if(DEBUG_MSG)
	    cout<< "vxFFT Running time	is: " << static_cast<double>(c4 - c3) / CLOCKS_PER_SEC * 1000 << "ms" << endl;

    //Point2d tr = phaseCorrelate(im1, im0);  //cv封装 
    
	Mat mov_mat = Mat::zeros(Size(3, 2), CV_64FC1);

	mov_mat.at<double>(0, 0) = 1.0;
	mov_mat.at<double>(0, 1) = 0.0;
	mov_mat.at<double>(1, 0) = 0.0;
	mov_mat.at<double>(1, 1) = 1.0;

	mov_mat.at<double>(0, 2) = -tr.x;
	mov_mat.at<double>(1, 2) = tr.y;

	return mov_mat;
}


