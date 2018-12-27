#ifndef CL_API_H
#define CL_API_H

#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <unistd.h>
using namespace cv;
using namespace std;

Mat cl_exc_remap_r(Mat input, Mat map_x, Mat map_y);
Mat cl_exc_remap_f(Mat input, Mat map_x, Mat map_y);

#endif