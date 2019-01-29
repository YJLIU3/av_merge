#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "parameter.h" 
using namespace std;
using namespace cv;

#if 0
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
#endif

/****************************************************************************************/
/******************************** OPENVX_PHASECORRECT ************************************/
/****************************************************************************************/
Point2d phaseCorrelateRes(InputArray _src1, InputArray _src2, InputArray _window = noArray(), double* response = 0);

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



Point2d phaseCorrelateRes(InputArray _src1, InputArray _src2, InputArray _window , double* response)
{
    static Mat OLD_FFT;
    Mat src1 = _src1.getMat();
    Mat src2 = _src2.getMat();
    Mat window = _window.getMat();
    CV_Assert( src1.type() == src2.type());
    CV_Assert( src1.type() == CV_32FC1 || src1.type() == CV_64FC1 );
    CV_Assert( src1.size == src2.size);
    if(!window.empty())
    {
        CV_Assert( src1.type() == window.type());
        CV_Assert( src1.size == window.size);
    }
    int M = getOptimalDFTSize(src1.rows);
    int N = getOptimalDFTSize(src1.cols);
 
    Mat padded1, padded2, paddedWin;
    if(M != src1.rows || N != src1.cols)
    {
        copyMakeBorder(src1, padded1, 0, M - src1.rows, 0, N - src1.cols, BORDER_CONSTANT, Scalar::all(0));
        copyMakeBorder(src2, padded2, 0, M - src2.rows, 0, N - src2.cols, BORDER_CONSTANT, Scalar::all(0));
 
        if(!window.empty())
        {
            copyMakeBorder(window, paddedWin, 0, M - window.rows, 0, N - window.cols, BORDER_CONSTANT, Scalar::all(0));
        }
    }
    else
    {
        padded1 = src1;
        padded2 = src2;
        paddedWin = window;
    }
 
    Mat FFT1, FFT2, P, Pm, C;
 
    if(!paddedWin.empty())
    {
        // apply window to both images before proceeding...
        multiply(paddedWin, padded1, padded1);
        multiply(paddedWin, padded2, padded2);
    }
    static bool fir_fps = 1;
    if(fir_fps)
    {
        fir_fps = 0;
        dft(padded1, FFT1, DFT_REAL_OUTPUT);
        dft(padded2, FFT2, DFT_REAL_OUTPUT);
    }
    else
    {
        dft(padded2, FFT2, DFT_REAL_OUTPUT);
        FFT1 = OLD_FFT;
    }
   
    mulSpectrums(FFT1, FFT2, P, 0, true);

    magSpectrums(P, Pm);

    divSpectrums(P, Pm, C, 0, false); // FF* / |FF*| (phase correlation equation completed here...)

    idft(C, C);
    fftShift(C);
    OLD_FFT = FFT2;
    Point peakLoc;
    minMaxLoc(C, NULL, NULL, NULL, &peakLoc);
    Point2d t;
    t = weightedCentroid(C, peakLoc, Size(5, 5), response);
 
    if(response)
        *response /= M*N;
 
    Point2d center((double)padded1.cols / 2.0, (double)padded1.rows / 2.0);
 
    return (center - t);

}

float logpolar(Mat& src, Mat& dst)
{

	float radii = src.cols;
	float angles = src.rows;
	Point2f center(src.cols / 2, src.rows / 2);
	float d = norm(Vec2f(src.cols - center.x, src.rows - center.y));
	float log_base = pow(10.0, log10(d) / radii);
#if 1
	float d_theta = CV_PI / (float)angles;
	float theta = CV_PI / 2.0;
	float radius = 0;
	Mat map_x(src.size(), CV_32FC1);
	Mat map_y(src.size(), CV_32FC1);
	for (int i = 0; i < angles; ++i)
	{
		for (int j = 0; j < radii; ++j)
		{
			radius = pow(log_base, float(j));
			float x = radius * sin(theta) + center.x;
			float y = radius * cos(theta) + center.y;
			map_x.at<float>(i, j) = x;
			map_y.at<float>(i, j) = y;
		}
		theta += d_theta;
	}
	remap(src, dst, map_x, map_y, CV_INTER_NN, BORDER_CONSTANT, Scalar(0, 0, 0));
#else
	//logPolar(src, dst, center);
#endif
	return log_base;
}


void Recomb(Mat &src, Mat &dst)
{
	int cx = src.cols >> 1;
	int cy = src.rows >> 1;
	Mat tmp;
	tmp.create(src.size(), src.type());
	src(Rect(0, 0, cx, cy)).copyTo(tmp(Rect(cx, cy, cx, cy)));
	src(Rect(cx, cy, cx, cy)).copyTo(tmp(Rect(0, 0, cx, cy)));
	src(Rect(cx, 0, cx, cy)).copyTo(tmp(Rect(0, cy, cx, cy)));
	src(Rect(0, cy, cx, cy)).copyTo(tmp(Rect(cx, 0, cx, cy)));
	dst = tmp;
}

void ForwardFFT(Mat &Src, Mat *FImg, bool do_recomb = true)
{
	int M = getOptimalDFTSize(Src.rows);
	int N = getOptimalDFTSize(Src.cols);
	Mat padded;
	copyMakeBorder(Src, padded, 0, M - Src.rows, 0, N - Src.cols, BORDER_CONSTANT, Scalar::all(0));
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);
	dft(complexImg, complexImg);
	split(complexImg, planes);
	planes[0] = planes[0](Rect(0, 0, planes[0].cols & -2, planes[0].rows & -2));
	planes[1] = planes[1](Rect(0, 0, planes[1].cols & -2, planes[1].rows & -2));
	if (do_recomb)
	{
		Recomb(planes[0], planes[0]);
		Recomb(planes[1], planes[1]);
	}
	planes[0] /= float(M*N);
	planes[1] /= float(M*N);
	FImg[0] = planes[0].clone();
	FImg[1] = planes[1].clone();
}
void highpass(Size sz, Mat& dst)
{
	Mat a = Mat(sz.height, 1, CV_32FC1);
	Mat b = Mat(1, sz.width, CV_32FC1);

	float step_y = CV_PI / sz.height;
	float val = -CV_PI * 0.5;

	for (int i = 0; i < sz.height; ++i)
	{
		a.at<float>(i) = cos(val);
		val += step_y;
	}

	val = -CV_PI * 0.5;
	float step_x = CV_PI / sz.width;
	for (int i = 0; i < sz.width; ++i)
	{
		b.at<float>(i) = cos(val);
		val += step_x;
	}

	Mat tmp = a * b;
	dst = (1.0 - tmp).mul(2.0 - tmp);
}

Mat test_LogPolarFFTTemplateMatch(Mat im0, Mat im1, double canny_threshold1, double canny_threshold2, int idx)
{
    clock_t mat_st = clock();

    Mat im1_ROI = im1(Rect(2,0,256,128));
    Mat im0_ROI = im0(Rect(2,0,256,128));
    clock_t mat_en = clock();

    Point2d tr = phaseCorrelateRes(im0_ROI, im1_ROI);

   
	Mat mov_mat = Mat::zeros(Size(3, 2), CV_64FC1);

	mov_mat.at<double>(0, 0) = 1.0;
	mov_mat.at<double>(0, 1) = 0.0;
	mov_mat.at<double>(1, 0) = 0.0;
	mov_mat.at<double>(1, 1) = 1.0;
	mov_mat.at<double>(0, 2) = 0;
	mov_mat.at<double>(1, 2) = tr.y;

    
    if(DEBUG_MSG)
    cout<< "##### End Phase time = " << static_cast<double>(mat_en - mat_st) / CLOCKS_PER_SEC * 1000 << "ms #####" << endl;

	return mov_mat;

}
Mat LogPolarFFTTemplateMatch(Mat im0_1, Mat im1_1, double canny_threshold1, double canny_threshold2, int idx)
{
	Mat im0 = im0_1.clone();
	Mat im1 = im1_1.clone();
	CV_Assert((im0.type() == CV_8UC1) || (im0.type() == CV_8UC3) ||
		(im0.type() == CV_32FC1) || (im0.type() == CV_32FC3) ||
		(im0.type() == CV_64FC1) || (im0.type() == CV_64FC3));

	CV_Assert(im0.rows == im1.rows && im0.cols == im1.cols);

	CV_Assert(im0.channels() == 1 || im0.channels() == 3 || im0.channels() == 4);

	CV_Assert(im1.channels() == 1 || im1.channels() == 3 || im1.channels() == 4);

	Mat im0_tmp = im0.clone();
	Mat im1_tmp = im1.clone();
	if (im0.channels() == 3)
	{
		cvtColor(im0, im0, cv::COLOR_BGR2GRAY);
	}

	if (im0.channels() == 4)
	{
		cvtColor(im0, im0, cv::COLOR_BGRA2GRAY);
	}

	if (im1.channels() == 3)
	{
		cvtColor(im1, im1, cv::COLOR_BGR2GRAY);
	}

	if (im1.channels() == 4)
	{
		cvtColor(im1, im1, cv::COLOR_BGRA2GRAY);
	}

	if (im0.type() == CV_32FC1)
	{
		im0.convertTo(im0, CV_8UC1, 255.0);
	}

	if (im1.type() == CV_32FC1)
	{
		im1.convertTo(im1, CV_8UC1, 255.0);
	}

	if (im0.type() == CV_64FC1)
	{
		im0.convertTo(im0, CV_8UC1, 255.0);
	}

	if (im1.type() == CV_64FC1)
	{
		im1.convertTo(im1, CV_8UC1, 255.0);
	}


	Canny(im0, im0, canny_threshold1, canny_threshold2); // you can change this
	Canny(im1, im1, canny_threshold1, canny_threshold2);

	// Ensure both images are of CV_32FC1 type
	im0.convertTo(im0, CV_32FC1, 1.0 / 255.0);
	im1.convertTo(im1, CV_32FC1, 1.0 / 255.0);

	Mat F0[2], F1[2];
	Mat f0, f1;
	ForwardFFT(im0, F0);
	ForwardFFT(im1, F1);
	magnitude(F0[0], F0[1], f0);
	magnitude(F1[0], F1[1], f1);

	// Create filter 
	Mat h;
	highpass(f0.size(), h);

	// Apply it in freq domain
	f0 = f0.mul(h);
	f1 = f1.mul(h);

	float log_base;
	Mat f0lp, f1lp;

	log_base = logpolar(f0, f0lp);
	log_base = logpolar(f1, f1lp);

	// Find rotation and scale
	Point2d rotation_and_scale = cv::phaseCorrelate(f1lp, f0lp);

	float angle = 180.0 * rotation_and_scale.y / f0lp.rows;
	float scale = pow(log_base, rotation_and_scale.x);
	// --------------
	if (scale > 1.8)
	{
		rotation_and_scale = cv::phaseCorrelate(f1lp, f0lp);
		angle = -180.0 * rotation_and_scale.y / f0lp.rows;
		scale = 1.0 / pow(log_base, rotation_and_scale.x);
		if (scale > 1.8)
		{
			cout << "Images are not compatible. Scale change > 1.8" << endl;
		}
	}
	// --------------
	if (angle < -90.0)
	{
		angle += 180.0;
}
	else if (angle > 90.0)
	{
		angle -= 180.0;
	}

	// Now rotate and scale fragment back, then find translation
	Mat rot_mat = getRotationMatrix2D(Point(im1.cols / 2, im1.rows / 2), angle, 1.0 / scale);

	// rotate and scale
	Mat im1_rs;
	warpAffine(im1, im1_rs, rot_mat, im1.size());
	warpAffine(im1_tmp, im1_tmp, rot_mat, im1.size());

	im0_tmp.convertTo(im0_tmp, CV_32FC1);
	im1_tmp.convertTo(im1_tmp, CV_32FC1);

	Point2d tr = cv::phaseCorrelate(im0, im1_rs);

	// compute rotated rectangle parameters
	RotatedRect rr;
	rr.center = tr + Point2d(im0.cols / 2, im0.rows / 2);
	rr.angle = -angle;
	rr.size.width = im1.cols / scale;
	rr.size.height = im1.rows / scale;

	Mat mov_mat = Mat::zeros(Size(3, 3), CV_64FC1);
	Mat art_mat = Mat::zeros(Size(3, 3), CV_64FC1);
	mov_mat.at<double>(0, 0) = 1.0;
	mov_mat.at<double>(1, 1) = 1.0;
	mov_mat.at<double>(0, 2) = -tr.x;
	mov_mat.at<double>(1, 2) = -tr.y;
	mov_mat.at<double>(2, 2) = 1.0;
	art_mat.at<double>(2, 2) = 1.0;
	memcpy(art_mat.data, rot_mat.data, sizeof(double) * 6);
	Mat t_mat = Mat::zeros(Size(3, 2), CV_64FC1);
	memcpy(t_mat.data, mov_mat.data, sizeof(double) * 6);
	mov_mat = mov_mat * art_mat;
	Mat ret_mat = Mat::zeros(Size(3, 2), CV_64FC1);
	memcpy(ret_mat.data, mov_mat.data, sizeof(double) * 6);

	warpAffine(im1_tmp, im1_tmp, t_mat, Size(280, 200));

	im0_tmp.convertTo(im0_tmp, CV_8UC1);
	im1_tmp.convertTo(im1_tmp, CV_8UC1);

	im0 = im0_tmp.clone();
	im1 = im1_tmp.clone();

	return ret_mat;
}


