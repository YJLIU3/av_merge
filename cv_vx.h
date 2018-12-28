#ifndef CV_VX_H
#define CV_VX_H

#include <VX/vx.h>
#include <VX/vxu.h>
#include <VX/vx_lib_extras.h>
#include <VX/vx_api.h>
#include <VX/vx_khr_cnn.h>

#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

using namespace cv;
Mat vx_Affine_RGB(Mat input, Mat matrix);
Mat vx_Remap_RGB(Mat input, bool Revers = false);
void init_vx(vx_context& context, vx_graph& graph);
void init_vx_remap( Mat  map_x, Mat map_y);


#endif
