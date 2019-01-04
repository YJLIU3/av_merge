#ifndef _OPENCV_fftm_HPP_
#define _OPENCV_fftm_HPP_
#ifdef __cplusplus

#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "iostream"
using namespace std;
//-----------------------------------------------------------------------------------------------------
// As input we need equal sized images, with the same aspect ratio,
// scale difference should not exceed 1.8 times.
//-----------------------------------------------------------------------------------------------------
cv::Mat LogPolarFFTTemplateMatch(cv::Mat im0, cv::Mat im1, double canny_threshold1=200, double canny_threshold2=100, int idx = 0);
cv::Mat vx_LogPolarFFTTemplateMatch(cv::Mat im0, cv::Mat im1, double canny_threshold1=200, double canny_threshold2=100, int idx = 0);
Mat test_LogPolarFFTTemplateMatch(Mat im0, Mat im1, double canny_threshold1=200, double canny_threshold2=100, int idx=0);
#endif
#endif