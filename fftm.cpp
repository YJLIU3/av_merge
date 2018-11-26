#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "parameter.h"

using namespace std;
using namespace cv;

static Mat past_im;
//----------------------------------------------------------
// Recombinate image quaters
//----------------------------------------------------------
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
//----------------------------------------------------------
// 2D Forward FFT
//----------------------------------------------------------
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
//----------------------------------------------------------
// 2D inverse FFT
//----------------------------------------------------------
void InverseFFT(Mat *FImg, Mat &Dst, bool do_recomb = true)
{
	if (do_recomb)
	{
		Recomb(FImg[0], FImg[0]);
		Recomb(FImg[1], FImg[1]);
	}
	Mat complexImg;
	merge(FImg, 2, complexImg);
	idft(complexImg, complexImg);
	split(complexImg, FImg);
	Dst = FImg[0].clone();
}
//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
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
//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
float logpolar(Mat& src, Mat& dst)
{
	float radii = src.cols;
	float angles = src.rows;
	Point2f center(src.cols / 2, src.rows / 2);
	float d = norm(Vec2f(src.cols - center.x, src.rows - center.y));
	float log_base = pow(10.0, log10(d) / radii);
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
	remap(src, dst, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
	return log_base;
}
//-----------------------------------------------------------------------------------------------------
// As input we need equal sized images, with the same aspect ratio,
// scale difference should not exceed 1.8 times.
//-----------------------------------------------------------------------------------------------------
Mat LogPolarFFTTemplateMatch(Mat im0, Mat im1, double canny_threshold1, double canny_threshold2, int idx)
{
    //im0 ==== before
    //im1 ==== now

    Canny(im1, im1, canny_threshold2, canny_threshold1, 3, 1);
    if(im0.type()!=CV_32FC1)
    im0.convertTo(im0, CV_32FC1, 1.0 / 255.0);
    if(im1.type()!=CV_32FC1)
    im1.convertTo(im1, CV_32FC1, 1.0 / 255.0);

    Mat im1_t = Mat::zeros(im1.size(),im1.type());
    Mat im1_Rec = Mat(im1, Rect( 0, 0, im1.cols*0.9, im1.rows*0.7));
    Mat im1_t_Rec = Mat(im1_t, Rect( 0, 0, im1_t.cols*0.9, im1_t.rows*0.7));
    im1_Rec.copyTo(im1_t_Rec);

    clock_t a = clock();
    Point2d tr = cv::phaseCorrelate(im1_t, im0);
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

    if(DEBUG_MSG)
        cout << mov_mat << endl;
	return mov_mat;
}
