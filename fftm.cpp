#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "parameter.h"

#include <VX/vx.h>
#include <VX/vxu.h>
#include <VX/vx_api.h>
#include <VX/vx_khr_cnn.h>
#include <VX/vx_lib_extras.h>
#include <HAL/gc_hal.h>

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

    Point2d tr = cv::phaseCorrelate(im1_ROI, im0_ROI);
 
	Mat mov_mat = Mat::zeros(Size(3, 2), CV_64FC1);

	mov_mat.at<double>(0, 0) = 1.0;
	mov_mat.at<double>(0, 1) = 0.0;
	mov_mat.at<double>(1, 0) = 0.0;
	mov_mat.at<double>(1, 1) = 1.0;

	mov_mat.at<double>(0, 2) = -tr.x;
	mov_mat.at<double>(1, 2) = -tr.y;

	return mov_mat;

}

/****************************************************************************************/
/******************************** OPENVX_PHASECORRECT ************************************/
/****************************************************************************************/

static vx_status  status = VX_SUCCESS;
static vx_context context = NULL;

typedef struct{
    float real;
    float image;
}complex_fft;

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

typedef struct 
{
	vx_graph	graph[2];
	vx_node		node[5];
	vx_kernel	kernel[5];
	vx_border_t border;
	unsigned long	nGraphIdx;
	int			m;
	int			n;
	vx_scalar			mLen;
	vx_scalar			nLen;
	vx_scalar			nAligned;
	vx_scalar			mAligned;
	int			height;
	int			width;
    vx_image    imageIn;
	vx_array	arrayTmp;
	vx_array   	arrayIn;
	vx_array	arrayOut;
	vx_array   arrayFftOut;
	vx_array   arrayPreFftOut;
	vx_array   arraymagOut;
    vx_array   arrayFftHorOut;
    vx_array   arrayIfftHorOut;
    vx_array   arrayIfftOut;
    vx_array   ifftImgOut;
	void * 		mapPtrOut;
	void * 		mapPtrIn;
	void*      mapPtr          = NULL;
    void*      mapPtr_ir          = NULL;
    void*      mapPtr_ii          = NULL;
    vx_scalar   offsetX_s    = NULL;
    vx_scalar   offsetY_s    = NULL;

}VXFFTOBJS;

vx_status vxcFFTHorLayer(VXFFTOBJS fftobj, vx_image input, vx_array imgIn, vx_array output, int nLen, int n, int mLen, int m, int offsetX, int offsetY)
{
	vx_int32    index        = 0;
    vx_border_t border;
    vx_enum     dataFormat = VX_TYPE_FLOAT32;

    border.constant_value.U8 = 0;
    border.mode = VX_BORDER_CONSTANT;

    if (dataFormat == VX_TYPE_FLOAT16)
    {
        border.constant_value.U16 = 0;
    }
    else if (dataFormat == VX_TYPE_UINT8)
    {
        border.constant_value.U8 = 0;
    }

    status        = vxLoadKernels(context, "fft");

    if(VX_SUCCESS == status)
    {
        char * pKernelName = NULL;


		switch(dataFormat)
		{
		case VX_TYPE_UINT8:
			pKernelName = "com.vivantecorp.extension.rnn_uint8";
			break;
		case VX_TYPE_FLOAT32:
			pKernelName = "com.vivantecorp.extension.fft_h_256x128";
			break;
		case VX_TYPE_FLOAT16:
		default:
			return -4;
		}
        if(fftobj.kernel[0] = vxGetKernelByName(context, pKernelName))// match with VXC code
        {
			status |= vxGetStatus((vx_reference)fftobj.kernel[0] );

            if(fftobj.node[0] = vxCreateGenericNode(fftobj.graph[fftobj.nGraphIdx&0x01], fftobj.kernel[0]))
            {
				status |= vxGetStatus((vx_reference)fftobj.node[0] );

                status |= vxSetParameterByIndex(fftobj.node[0], index++, (vx_reference)input);
                status |= vxSetParameterByIndex(fftobj.node[0], index++, (vx_reference)imgIn);
                status |= vxSetParameterByIndex(fftobj.node[0], index++, (vx_reference)output);
                status |= vxSetParameterByIndex(fftobj.node[0], index++, (vx_reference)fftobj.nLen);	//(vx_reference)nLen_s);
                status |= vxSetParameterByIndex(fftobj.node[0], index++, (vx_reference)fftobj.nAligned);//(vx_reference)n_s);
                status |= vxSetParameterByIndex(fftobj.node[0], index++, (vx_reference)fftobj.mLen);	//(vx_reference)mLen_s);
                status |= vxSetParameterByIndex(fftobj.node[0], index++, (vx_reference)fftobj.mAligned);//(vx_reference)m_s);
                status |= vxSetParameterByIndex(fftobj.node[0], index++, (vx_reference)fftobj.offsetX_s);
                status |= vxSetParameterByIndex(fftobj.node[0], index++, (vx_reference)fftobj.offsetY_s);
                
                status |= vxSetNodeAttribute(fftobj.node[0], VX_NODE_BORDER, &border, sizeof(border));

                if(status != VX_SUCCESS)
                {
                    status = VX_ERROR_INVALID_PARAMETERS;
                    vxReleaseNode(&fftobj.node[0]);
                    vxReleaseKernel(&fftobj.kernel[0]);
                    return status;
                }
            }
            else
            {
                vxReleaseKernel(&fftobj.kernel[0]);
                return status;
            }
        }
    }
    return status;
}

vx_status vxcFFTVerLayer(VXFFTOBJS fftobj, vx_array input, vx_array output, int nLen, int n, int mLen, int m)
{
	vx_int32    index        = 0;
    vx_uint32   input_ZP   = 0;
    vx_border_t border;
    vx_enum     dataFormat = VX_TYPE_FLOAT32;

    border.constant_value.U8 = 0;
    border.mode = VX_BORDER_CONSTANT;

    if (dataFormat == VX_TYPE_FLOAT16)
    {
        border.constant_value.U16 = 0;
    }
    else if (dataFormat == VX_TYPE_UINT8)
    {
        border.constant_value.U8 = input_ZP;
    }

    status        = vxLoadKernels(context, "fft");
    if(VX_SUCCESS == status)
    {
		char * pKernelName = NULL;


		switch(dataFormat)
		{
		case VX_TYPE_UINT8:
			pKernelName = "com.vivantecorp.extension.rnn_uint8";
			break;
		case VX_TYPE_FLOAT32:
			pKernelName = "com.vivantecorp.extension.fft_v_256x128";
			break;
		case VX_TYPE_FLOAT16:
		default:
			return -4;
		}		
        if(fftobj.kernel[1] = vxGetKernelByName(context, pKernelName))// match with VXC code
        {
            if(fftobj.node[1] = vxCreateGenericNode(fftobj.graph[fftobj.nGraphIdx&0x01], fftobj.kernel[1]))
            {
                status |= vxSetParameterByIndex(fftobj.node[1], index++, (vx_reference)input);
                status |= vxSetParameterByIndex(fftobj.node[1], index++, (vx_reference)output);
                status |= vxSetParameterByIndex(fftobj.node[1], index++,(vx_reference)fftobj.nLen); //(vx_reference)nLen_s);
                status |= vxSetParameterByIndex(fftobj.node[1], index++,(vx_reference)fftobj.nAligned); //(vx_reference)n_s);
                status |= vxSetParameterByIndex(fftobj.node[1], index++,(vx_reference)fftobj.mLen); //(vx_reference)mLen_s);
                status |= vxSetParameterByIndex(fftobj.node[1], index++,(vx_reference)fftobj.mAligned); //(vx_reference)m_s);

                status |= vxSetNodeAttribute(fftobj.node[1], VX_NODE_BORDER, &border, sizeof(border));

                if(status != VX_SUCCESS)
                {
                    status = VX_ERROR_INVALID_PARAMETERS;
                    vxReleaseNode(&fftobj.node[1]);
                    vxReleaseKernel(&fftobj.kernel[1]);
                    return status;
                }
            }
            else
            {
                vxReleaseKernel(&fftobj.kernel[1]);
                return status;
            }
        }
    }



    return status;
}

vx_status vxcSpectrumLayer(VXFFTOBJS fftobj, vx_array input, vx_array input1, vx_array output, int width, int height)
{
	vx_int32    index        = 0;
    vx_uint32   input_ZP   = 0;
    vx_border_t border;
    vx_enum     dataFormat = VX_TYPE_FLOAT32;

    border.constant_value.U8 = 0;
    border.mode = VX_BORDER_CONSTANT;

    if (dataFormat == VX_TYPE_FLOAT16)
    {
        border.constant_value.U16 = 0;
    }
    else if (dataFormat == VX_TYPE_UINT8)
    {
        border.constant_value.U8 = input_ZP;
    }

    status        = vxLoadKernels(context, "fft");
    if(VX_SUCCESS == status)
    {
		char * pKernelName = NULL;


		switch(dataFormat)
		{
		case VX_TYPE_UINT8:
			pKernelName = "com.vivantecorp.extension.rnn_uint8";
			break;
		case VX_TYPE_FLOAT32:
			pKernelName = "com.vivantecorp.extension.powerSpectrum";
			break;
		case VX_TYPE_FLOAT16:
		default:
			return -4;
		}			
        if(fftobj.kernel[2] = vxGetKernelByName(context, pKernelName))// match with VXC code
        {
            if(fftobj.node[2] = vxCreateGenericNode(fftobj.graph[fftobj.nGraphIdx&0x01], fftobj.kernel[2]))
            {
                status |= vxSetParameterByIndex(fftobj.node[2], index++, (vx_reference)input);
                status |= vxSetParameterByIndex(fftobj.node[2], index++, (vx_reference)input1);
                status |= vxSetParameterByIndex(fftobj.node[2], index++, (vx_reference)output);
                status |= vxSetParameterByIndex(fftobj.node[2], index++, (vx_reference)fftobj.nLen);
                status |= vxSetParameterByIndex(fftobj.node[2], index++, (vx_reference)fftobj.mLen);

                status |= vxSetNodeAttribute(fftobj.node[2], VX_NODE_BORDER, &border, sizeof(border));

                if(status != VX_SUCCESS)
                {
                    status = VX_ERROR_INVALID_PARAMETERS;
                    vxReleaseNode(&fftobj.node[2]);
                    vxReleaseKernel(&fftobj.kernel[2]);
                    return status;
                }
            }
            else
            {
                vxReleaseKernel(&fftobj.kernel[2]);
                return status;
            }
        }
    }
    return status;
}

vx_status vxcIFFTHorLayer(VXFFTOBJS fftobj, vx_array input, vx_array output, int nLen, int n, int mLen, int m)
{
	vx_int32    index        = 0;
    vx_uint32   input_ZP   = 0;
    vx_border_t border;
    vx_enum     dataFormat = VX_TYPE_FLOAT32;

    border.constant_value.U8 = 0;
    border.mode = VX_BORDER_CONSTANT;

    if (dataFormat == VX_TYPE_FLOAT16)
    {
        border.constant_value.U16 = 0;
    }
    else if (dataFormat == VX_TYPE_UINT8)
    {
        border.constant_value.U8 = input_ZP;
    }

    status        = vxLoadKernels(context, "fft");
    if(VX_SUCCESS == status)
    {

		char * pKernelName = NULL;


		switch(dataFormat)
		{
		case VX_TYPE_UINT8:
			pKernelName = "com.vivantecorp.extension.rnn_uint8";
			break;
		case VX_TYPE_FLOAT32:
			pKernelName = "com.vivantecorp.extension.ifft_h_256x128";
			break;
		case VX_TYPE_FLOAT16:
		default:
			return -4;
		}			
		
        if(fftobj.kernel[3] = vxGetKernelByName(context, pKernelName))// match with VXC code
        {
            if(fftobj.node[3] = vxCreateGenericNode(fftobj.graph[fftobj.nGraphIdx&0x01], fftobj.kernel[3]))
            {
                status |= vxSetParameterByIndex(fftobj.node[3], index++, (vx_reference)input);
                status |= vxSetParameterByIndex(fftobj.node[3], index++, (vx_reference)output);
                status |= vxSetParameterByIndex(fftobj.node[3], index++, (vx_reference)fftobj.nLen);//(vx_reference)nLen_s);
                status |= vxSetParameterByIndex(fftobj.node[3], index++, (vx_reference)fftobj.nAligned);//(vx_reference)n_s);
                status |= vxSetParameterByIndex(fftobj.node[3], index++, (vx_reference)fftobj.mLen);//(vx_reference)mLen_s);
                status |= vxSetParameterByIndex(fftobj.node[3], index++, (vx_reference)fftobj.mAligned);//(vx_reference)m_s);

                status |= vxSetNodeAttribute(fftobj.node[3], VX_NODE_BORDER, &border, sizeof(border));

                if(status != VX_SUCCESS)
                {
                    status = VX_ERROR_INVALID_PARAMETERS;
                    vxReleaseNode(&fftobj.node[3]);
                    vxReleaseKernel(&fftobj.kernel[3]);
                    return status;
                }
            }
            else
            {
                vxReleaseKernel(&fftobj.kernel[3]);
                return status;
            }
        }
    }
    return status;
}

vx_status vxcIFFTVerLayer(VXFFTOBJS fftobj, vx_array input, vx_array output, vx_array imgOut, int nLen, int n, int mLen, int m)
{
    vx_int32    index        = 0;
    vx_uint32   input_ZP   = 0;
    vx_border_t border;
    vx_enum     dataFormat = VX_TYPE_FLOAT32;

    border.constant_value.U8 = 0;
    border.mode = VX_BORDER_CONSTANT;

    if (dataFormat == VX_TYPE_FLOAT16)
    {
        border.constant_value.U16 = 0;
    }
    else if (dataFormat == VX_TYPE_UINT8)
    {
        border.constant_value.U8 = input_ZP;
    }

    status        = vxLoadKernels(context, "fft");
    if(VX_SUCCESS == status)
    {
		char * pKernelName = NULL;


		switch(dataFormat)
		{
		case VX_TYPE_UINT8:
			pKernelName = "com.vivantecorp.extension.rnn_uint8";
			break;
		case VX_TYPE_FLOAT32:
			pKernelName =  "com.vivantecorp.extension.ifft_v_256x128";
			break;
		case VX_TYPE_FLOAT16:
		default:
			return -4;
		}			
				
        if(fftobj.kernel[4] = vxGetKernelByName(context, pKernelName))// match with VXC code
        {
            if(fftobj.node[4] = vxCreateGenericNode(fftobj.graph[fftobj.nGraphIdx&0x01], fftobj.kernel[4]))
            {
                status |= vxSetParameterByIndex(fftobj.node[4], index++, (vx_reference)input);
                status |= vxSetParameterByIndex(fftobj.node[4], index++, (vx_reference)output);
                status |= vxSetParameterByIndex(fftobj.node[4], index++, (vx_reference)imgOut);
                status |= vxSetParameterByIndex(fftobj.node[4], index++, (vx_reference)fftobj.nLen);//(vx_reference)nLen_s);
                status |= vxSetParameterByIndex(fftobj.node[4], index++, (vx_reference)fftobj.nAligned);//(vx_reference)n_s);
                status |= vxSetParameterByIndex(fftobj.node[4], index++, (vx_reference)fftobj.mLen);//(vx_reference)mLen_s);
                status |= vxSetParameterByIndex(fftobj.node[4], index++, (vx_reference)fftobj.mAligned);//(vx_reference)m_s);

                status |= vxSetNodeAttribute(fftobj.node[4], VX_NODE_BORDER, &border, sizeof(border));

                if(status != VX_SUCCESS)
                {
                    status = VX_ERROR_INVALID_PARAMETERS;
                    vxReleaseNode(&fftobj.node[4]);
                    vxReleaseKernel(&fftobj.kernel[4]);
                    return status;
                }
            }
            else
            {
                vxReleaseKernel(&fftobj.kernel[4]);
                return status;
            }
        }
    }
    return status;
}

void init_vx(VXFFTOBJS & fftobj, int width, int height, int src_format, int bIndex, int offsetX, int offsetY)
{
	unsigned long 	arraySize = 0;
	vx_size 		itemNum = 0;
	vx_size 		itemSize = 0;
	vx_map_id 		mapId = 0;
	vx_size 		capacity  = 0;
    vx_enum    		dst_format = VX_TYPE_FLOAT32;
    vx_map_id 		ifftRealmapId = 0;
    vx_map_id 		ifftImgmapId = 0;
    vx_size 		itemNumIr = 0;
    vx_size 		itemSizeIr = 0;
    vx_size 		itemNumIi = 0;
    vx_size 		itemSizeIi = 0;

	float 		val = 0.0; 
	char 		x	= 0;	
	int			i = 0;
	if(context == NULL)

	{
		context = vxCreateContext();

        
		fftobj.m = calculate_M(height);    
		fftobj.n = calculate_M(width);	
		fftobj.nLen = vxCreateScalar(context, VX_TYPE_INT32, &width);
		fftobj.mLen = vxCreateScalar(context, VX_TYPE_INT32, &height);
		fftobj.nAligned = vxCreateScalar(context, VX_TYPE_INT32, &fftobj.n);
		fftobj.mAligned = vxCreateScalar(context, VX_TYPE_INT32, &fftobj.m);
        fftobj.offsetX_s     = vxCreateScalar(context, VX_TYPE_INT32, &offsetX);
        fftobj.offsetY_s     = vxCreateScalar(context, VX_TYPE_INT32, &offsetY);
        fftobj.width = width;
        fftobj.height= height;
		arraySize =width * height *2;
		//fftobj.input_ptr = (complex_fft*)malloc(arraySize * vxcGetTypeSize(dst_format));

        fftobj.imageIn = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
		fftobj.arrayIn = vxCreateArray(context, src_format, arraySize * vxcGetTypeSize(src_format)/2);
		fftobj.arrayTmp = vxCreateArray(context, src_format, arraySize * vxcGetTypeSize(src_format)/2);
		//fftobj.arrayOut = vxCreateArray(context, dst_format, arraySize * vxcGetTypeSize(dst_format));
		fftobj.arrayFftOut = vxCreateArray(context, dst_format, arraySize * vxcGetTypeSize(dst_format));
		fftobj.arraymagOut = vxCreateArray(context, dst_format, arraySize * vxcGetTypeSize(dst_format));
		fftobj.arrayPreFftOut = vxCreateArray(context, dst_format, arraySize * vxcGetTypeSize(dst_format));
		fftobj.arrayFftHorOut = vxCreateArray(context, dst_format, arraySize * vxcGetTypeSize(dst_format));
		fftobj.arrayIfftHorOut = vxCreateArray(context, dst_format, arraySize * vxcGetTypeSize(dst_format));
		fftobj.arrayIfftOut = vxCreateArray(context, dst_format, arraySize * vxcGetTypeSize(dst_format) / 2);
		fftobj.ifftImgOut = vxCreateArray(context, dst_format, arraySize * vxcGetTypeSize(dst_format) / 2);
		
		
		for( i = 0;i<arraySize;i++)

		{
			vxAddArrayItems(fftobj.arrayFftOut, 1, &val, 0);	
		}
		
		for( i = 0;i<(arraySize>>1);i++)

		{
			vxAddArrayItems(fftobj.arrayIfftOut, 1, &val, 0);
			vxAddArrayItems(fftobj.arrayIn, 1, &x, 0);
			vxAddArrayItems(fftobj.arrayTmp,1, &x, 0);
		}

        
        
		vxQueryArray(fftobj.arrayIn, VX_ARRAY_CAPACITY, &capacity, sizeof(vx_size));
		vxQueryArray(fftobj.arrayIn, VX_ARRAY_NUMITEMS, &itemNum, sizeof(vx_size));
		vxQueryArray(fftobj.arrayIn, VX_ARRAY_ITEMSIZE, &itemSize, sizeof(vx_size));
        vx_rectangle_t rect_remap = {0,0,width,height};
        vx_imagepatch_addressing_t imgInfo = VX_IMAGEPATCH_ADDR_INIT;
        status = vxMapImagePatch(fftobj.imageIn,&rect_remap,0,&mapId,&imgInfo,(void**)&fftobj.mapPtrIn,VX_READ_AND_WRITE,VX_MEMORY_TYPE_HOST,0);
//		status = vxMapArrayRange(fftobj.arrayIn, 0, arraySize>>1, &mapId, &itemSize, (void**)&fftobj.mapPtrIn, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);
		status = vxQueryArray(fftobj.arrayFftOut, VX_ARRAY_NUMITEMS, &itemNum, sizeof(vx_size));
		status = vxQueryArray(fftobj.arrayFftOut, VX_ARRAY_ITEMSIZE, &itemSize, sizeof(vx_size));
		status = vxMapArrayRange(fftobj.arrayFftOut, 0, itemNum/*capacity*/, &mapId, &itemSize, (void**)&fftobj.mapPtr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
		status = vxQueryArray(fftobj.arrayIfftOut, VX_ARRAY_NUMITEMS, &itemNumIr, sizeof(vx_size));
		status = vxQueryArray(fftobj.arrayIfftOut, VX_ARRAY_ITEMSIZE, &itemSizeIr, sizeof(vx_size));
		status = vxMapArrayRange(fftobj.arrayIfftOut, 0, itemNumIr/*capacity*/, &ifftRealmapId, &itemSizeIr, (void**)&fftobj.mapPtr_ir, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);

	}

	if(fftobj.graph[bIndex&0x01] == NULL)
	{
		fftobj.graph[bIndex&0x01] = vxCreateGraph(context);

		status = vxcFFTHorLayer(fftobj, fftobj.imageIn, fftobj.arrayTmp, fftobj.arrayFftHorOut, fftobj.width, fftobj.n, fftobj.height, fftobj.m, offsetX, offsetY);

		if((bIndex & 0x01) == 0)
		{
			vxcFFTVerLayer(fftobj, fftobj.arrayFftHorOut, fftobj.arrayFftOut, fftobj.width, fftobj.n, fftobj.height, fftobj.m);
			vxcSpectrumLayer(fftobj, fftobj.arrayFftOut, fftobj.arrayPreFftOut ,fftobj.arraymagOut,fftobj.width, fftobj.height);
		}
		else
		{
			vxcFFTVerLayer(fftobj, fftobj.arrayFftHorOut, fftobj.arrayPreFftOut, fftobj.width, fftobj.n, fftobj.height, fftobj.m);
			vxcSpectrumLayer(fftobj, fftobj.arrayPreFftOut ,fftobj.arrayFftOut,fftobj.arraymagOut,fftobj.width, fftobj.height);
		}
		vxcIFFTHorLayer(fftobj,  fftobj.arraymagOut, fftobj.arrayIfftHorOut,fftobj.width, fftobj.n, fftobj.height, fftobj.m);
		vxcIFFTVerLayer(fftobj, fftobj.arrayIfftHorOut, fftobj.arrayIfftOut, fftobj.ifftImgOut, fftobj.width, fftobj.n, fftobj.height, fftobj.m);
        
		status = vxVerifyGraph(fftobj.graph[fftobj.nGraphIdx&0x01]);
	}	
	
}

void vxFFT(VXFFTOBJS & fftobj, Mat src, Mat &dst)
{
	int srcnSize = src.rows * src.cols * sizeof(char);
	int dstnSize = src.rows * src.cols * sizeof(float);
    
	memcpy((char *)fftobj.mapPtrIn, (char *)src.data, srcnSize);
//	gcoOS_CacheFlush( NULL, NULL,fftobj.mapPtrIn, srcnSize);
	
    status = vxProcessGraph(fftobj.graph[fftobj.nGraphIdx&0x01]); 
    fftobj.nGraphIdx++;
	memcpy((float *)dst.data,(float *)fftobj.mapPtr_ir, dstnSize);

}

static VXFFTOBJS vxFFTObj;


Point2d vx_PhaseCorrelation2D(Mat src1)
{
    
	Point 		peakLoc;
	Point2d 	t={0};
	static Mat dstimg(Size(256,128),CV_32FC1,Scalar::all(0));
	static bool vxFFTObjInit = false;
	if(vxFFTObjInit == false)
	{
		memset(&vxFFTObj, 0x00, sizeof(VXFFTOBJS));
		init_vx(vxFFTObj,src1.cols,src1.rows,VX_TYPE_CHAR,0, 2, 26);	
		vxFFTObj.nGraphIdx++;
		init_vx(vxFFTObj,src1.cols,src1.rows,VX_TYPE_CHAR,1, 2, 26);	
		vxFFTObj.nGraphIdx++;
		vxFFTObjInit =  true;
	}
    
	vxFFT(vxFFTObj, src1, dstimg);

    
	fftShift(dstimg); // shift the energy to the center of the frame.	   
	minMaxLoc(dstimg, NULL, NULL, NULL, &peakLoc);
	t = weightedCentroid(dstimg, peakLoc, Size(5, 5), 0);

	Point2d center((double)src1.cols / 2.0, (double)src1.rows / 2.0);
	
	t.x  = center.x - t.x;
	t.y = center.y - t.y;
	
    return (t);  

}

static bool first_fft = true;

Mat vx_LogPolarFFTTemplateMatch(Mat im0, Mat im1, double canny_threshold1, double canny_threshold2, int idx)
{
    Canny(im1, im1, canny_threshold2, canny_threshold1, 3, 1);
    if(im0.type() != CV_8UC1)
    im0.convertTo(im0, CV_8UC1);
    if(im1.type()!= CV_8UC1)
    im1.convertTo(im1, CV_8UC1);
     
	Mat im1_ROI_t = im1;//(Rect(2,26,256,128));
    Mat im1_ROI = im1_ROI_t.clone();

    if(first_fft == true)
	{
		first_fft = false;
        Mat im0_ROI_t = im0;//(Rect(2,26,256,128));
        Mat im0_ROI = im0_ROI_t.clone();
        vx_PhaseCorrelation2D(im0_ROI); 
	}

	clock_t c3 = clock(); 
	Point2d tr = vx_PhaseCorrelation2D(im1_ROI); //vx	
	clock_t c4 = clock(); 
    if(DEBUG_MSG)
	    cout<< "vxFFT Running time	is: " << static_cast<double>(c4 - c3) / CLOCKS_PER_SEC * 1000 << "ms" << endl;


	Mat mov_mat = Mat::zeros(Size(3, 2), CV_64FC1);

	mov_mat.at<double>(0, 0) = 1.0;
	mov_mat.at<double>(0, 1) = 0.0;
	mov_mat.at<double>(1, 0) = 0.0;
	mov_mat.at<double>(1, 1) = 1.0;

	mov_mat.at<double>(0, 2) = -tr.x;
	mov_mat.at<double>(1, 2) = tr.y;
    if(DEBUG_MSG)
        cout << mov_mat << endl;
	return mov_mat;

}
