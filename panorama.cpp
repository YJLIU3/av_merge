#include <fstream>
#include "panorama.h"
#include "parameter.h"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include<time.h> 
#include "fftm.hpp" 
#include "cl_api.h"



static Mat matrix_bypast[25];
int bypast_cont = 0;
extern void *remap_gray_ptr[1];

#define grid_size 20
#define min_match 5
#define pi 3.1415926
static Mat imTime;
static Mat ims(Size_BG, CV_8UC4 ,Scalar::all(0));
static Mat imTime2;
static Mat imMask2;
//static Mat im2;
static Mat matrix;
static Mat expand_roi;
static Mat merge_roi;   
extern Mat frontimage, rearimage;
static Mat output(Size_BG, CV_8UC4);
static Mat Highlander;


static Mat front(image_size,  CV_8UC4, Scalar::all(0));
static Mat rear(image_size, CV_8UC4, Scalar::all(0));

static Mat im1;
static Mat im2;
static void * cl_affine_Ptr[1];


Panorama::Panorama()
{

}

Panorama::~Panorama()
{

}

void Panorama::compute_merge_matrix(Mat frontChessboard, Mat rearChessboard,
	Size board_size, int offset_x, int offset_y)
{
}
void Panorama::mergeFrontRearMat(Mat frontMat, Mat rearMat, Mat& out)
 {
    Mat ROI_rear = Mat(out, Rect((output.cols - rearMat.cols) * 0.5, out.rows - rearMat.rows, rearMat.cols, rearMat.rows));
    rearMat.copyTo(ROI_rear);
    merge_roi = Mat(out, Rect((out.cols - frontMat.cols) * 0.5, 0, frontMat.cols, frontMat.rows));
    frontMat.copyTo(merge_roi);
 }


void Panorama::mergeFrontMat(Mat frontMat, Mat& out)
{
	merge_roi = Mat(out, Rect((out.cols - frontMat.cols) * 0.5, 0, frontMat.cols, frontMat.rows));
	frontMat.copyTo(merge_roi);
}
void Panorama::mergeRearMat(Mat rearMat, Mat& out)
{
    Mat ROI_rear = Mat(out, Rect((output.cols - rearMat.cols) * 0.5, out.rows - rearMat.rows, rearMat.cols, rearMat.rows));
	rearMat.copyTo(ROI_rear);
}


void create_timeImg_from_mask(Mat mask, Mat& imTime, int curIdx)
{
	imTime.create(mask.size(), CV_8UC1);
	threshold(mask, imTime, 0, 1, CV_THRESH_BINARY);
	imTime.convertTo(imTime, CV_32FC1);
	imTime = imTime * curIdx;
}


Mat compute_alpha(Mat mask1, Mat mask2, Mat time1, Mat time2, float timeRatioThrd)
{
	Mat alpha(mask1.size(), CV_32FC1, Scalar::all(0));
	for (int i = 0; i < alpha.rows; i++)
	{
		for (int j = 0; j < alpha.cols; j++)
		{
			double ratioTime = 1.0 - (IMAGE_ELEM(time2, float, i, j) - IMAGE_ELEM(time1, float, i, j)) / timeRatioThrd;
			if (ratioTime > 1.0)
				ratioTime = 1.0;
			if (IMAGE_ELEM(mask1, float, i, j) + IMAGE_ELEM(mask2, float, i, j) > 0)
			{
				IMAGE_ELEM(alpha, float, i, j) = IMAGE_ELEM(mask1, float, i, j) * ratioTime /
					(IMAGE_ELEM(mask1, float, i, j) + IMAGE_ELEM(mask2, float, i, j));
			}
			else
			{
				IMAGE_ELEM(alpha, float, i, j) = 1.0;
			}
		}
	}
	return alpha;

}

void mix_image_front(Mat image1, Mat image2, Mat alpha, Mat alpha_1, Mat& output)
{
#if 1	
    memcpy((cl_affine_Ptr[0]), (image2.data ), sizeof(char)* 4 * 180*320);

	output = image1;
#endif
}

void * mix_image_rear(Mat image1, Mat image2, Mat alpha, Mat alpha_1)
{
#if 1	

//    memcpy(cl_affine_Ptr[0], image1.data, sizeof(char)* 4 * 480*320);
	memcpy((cl_affine_Ptr[0] + 4*Size_BG.width*(Size_BG.height - front.size().height)), (image2.data + 4*Size_BG.width*(Size_BG.height - front.size().height)), sizeof(char)* 4 * 180*320);

    return cl_affine_Ptr[0];
#endif
}




void Panorama::expand(Mat input, Mat& output)
{
	output = Mat(Size_BG, input.type(), Scalar::all(0));
}



Mat Panorama::front_process(Mat front, Mat rear)
{
    static Mat front_before;
    static Mat front_now = Mat::zeros(image_size, CV_32FC1);

    if (bypast_cont > 23)
		bypast_cont = 0;

	if (!im1.data)
	{
		idx = 1;


		expand(front, im1);
		mergeFrontMat(front, im1);

		cvtColor(front, front, CV_RGB2GRAY);
        
		preImg = im1;

		front_before = front;
        front_before.convertTo(front_before, CV_32FC1, 1/255.0);
		output = im1;
        
        init_cl_Affine(cl_affine_Ptr);
        memcpy(cl_affine_Ptr[0], im1.data, im1.cols * im1.rows * 4 * sizeof(char));
	}
	else
	{
		idx++;
        clock_t a = clock();
        if(idx == 2)
        {
            imMask2 = weight.clone();
		
		    create_timeImg_from_mask(imMask2, imTime2, idx);
            front_now.convertTo(front_now, CV_32FC1);
            expand(front, im2);
        }

        front_now.data = (uchar *)remap_gray_ptr[0];
        
        clock_t b = clock();
        if(DEBUG_MSG)
        cout<< "##### Merge pic time = " << static_cast<double>(b - a) / CLOCKS_PER_SEC * 1000 << "ms #####" << endl;

        clock_t warp_st1 = clock();
        Mat matrix;

        if(!VIP7K);
//          matrix = vx_LogPolarFFTTemplateMatch(front_before, front_now, 200, 100, idx);
        else
 		    matrix = test_LogPolarFFTTemplateMatch(front_before, front_now, 200, 100, idx);
        clock_t warp_st2 = clock();
        if(DEBUG_MSG)
        cout<< "Compute_matrix Running time  is: " << static_cast<double>(warp_st2 - warp_st1) / CLOCKS_PER_SEC * 1000 << "ms" << endl;   

        clock_t warp_st3 = clock();


        bool deltaX = abs((int)matrix.at<double>(0, 2)) > 5;
        bool deltaY = abs((int)matrix.at<double>(1, 2)) > 20;  
        
          if (deltaX || deltaY)
         {
            matrix = matrix_back;
         }
        
        
         matrix_bypast[bypast_cont++] = matrix;
         matrix = Mat(2, 3, CV_64FC1, Scalar(0.0));
        
         if (idx < 25)
         {
             for (size_t i = 0; i < bypast_cont; i++)
             {
            
                matrix += (matrix_bypast[i] / (bypast_cont ));
        
             }
         }
         
         else
         {
            for (size_t i = 0; i < 24; i++)
            {
                matrix += matrix_bypast[i] / 24.0;
            }
         }
        
        matrix_back = matrix;


        clock_t warp_st4 = clock();
        if(DEBUG_MSG)
        cout<< "Process_matrix Running time  is: " << static_cast<double>(warp_st4 - warp_st3) / CLOCKS_PER_SEC * 1000 << "ms" << endl;   

        clock_t warp_st = clock();
        
        if(VIP7K)
              im1t = cl_exc_affine(im1, matrix, 0);
        else
            warpAffine(im1, im1t, matrix, WEIGHT_BIGSIZE, INTER_NEAREST);


        clock_t warp_en = clock();
        if(DEBUG_MSG)
        cout<< "warpAffine Running time  is: " << static_cast<double>(warp_en - warp_st) / CLOCKS_PER_SEC * 1000 << "ms" << endl;   

        clock_t warp_st5= clock();
 
		front_before = front_now;
        

        output.data = im1t.data;
        if(DEBUG_MSG_IMG)
            imwrite("debug/output.png",output);

	}

	return output;
}

Mat Panorama::rear_process(Mat front, Mat rear)
{
    static Mat rear_before;
    static Mat rear_now = Mat::zeros(image_size, CV_32FC1);
    if (bypast_cont > 23)
		bypast_cont = 0;

	if (!im1.data)
	{
		idx = 1;

		expand(front, im1);
		mergeRearMat(rear, im1);

		imMask1 = weight.clone();
		create_timeImg_from_mask(imMask1, imTime1, 1);

        cvtColor(rear, rear, CV_BGR2GRAY);
        
		preImg = im1;
        rear_before = rear;
        rear_before.convertTo(rear_before, CV_32FC1, 1/255.0);
        init_cl_Affine(cl_affine_Ptr);
        cout << cl_affine_Ptr[0] << "ptr" << endl;
		output = im1;
        memcpy(cl_affine_Ptr[0], im1.data, im1.cols * im1.rows * 4 * sizeof(char));
	}
	else
	{
		idx++;
        clock_t a = clock();
        if(idx == 2)
        {
            imMask2 = weight.clone();
		
		    create_timeImg_from_mask(imMask2, imTime2, idx);
            
            expand(front, im2);
        }

        rear_now.data = (uchar *)remap_gray_ptr[0];
//        Mat hhh = 255.0 * rear_now;
//imwrite("debug/front_thre.jpg", hhh);        
        clock_t b = clock();
        
        if(DEBUG_MSG)
        cout<< "##### Merge pic time = " << static_cast<double>(b - a) / CLOCKS_PER_SEC * 1000 << "ms #####" << endl;   

        clock_t warp_st1 = clock();
        Mat matrix;

        
        if(!VIP7K);
//          matrix = vx_LogPolarFFTTemplateMatch(rear_before, rear_now,  200, 100, idx);
        else
 		    matrix = test_LogPolarFFTTemplateMatch(rear_before, rear_now, 200, 100, idx);

        Mat temp_mat;
        
        clock_t warp_st2 = clock();
        if(DEBUG_MSG)
        cout<< "##### Compute_matrix Running time = " << static_cast<double>(warp_st2 - warp_st1) / CLOCKS_PER_SEC * 1000 << "ms #####" << endl;
        clock_t warp_st3 = clock();


        bool deltaX = abs((int)matrix.at<double>(0, 2)) > 5;
        bool deltaY = abs((int)matrix.at<double>(1, 2)) > 20;
        

        
          if (deltaX || deltaY)
         {
            matrix = matrix_back;
         }
        
        Mat test = matrix;
         matrix_bypast[bypast_cont++] = matrix;
         matrix = Mat(2, 3, CV_64FC1, Scalar(0.0));
        
         if (idx < 25)
         {
             for (size_t i = 0; i < bypast_cont; i++)
             {
            
                matrix += (matrix_bypast[i] / (bypast_cont ));
        
             }
         }
         
         else
         {
            for (size_t i = 0; i < 24; i++)
            {
                matrix += matrix_bypast[i] / 24.0;
            }
         }
        if(abs(matrix.at<double>(0, 2)) < 0.1 && abs(matrix.at<double>(1, 2)) < 0.1)
        {
            matrix.at<double>(0, 2) = 0;
            matrix.at<double>(1, 2) = 0;
        }
        else
        {
            matrix.at<double>(0, 2) = 0;
            matrix.at<double>(1, 2) = matrix.at<double>(1, 2) + 0.0;
        }
        matrix_back = matrix;

        test.at<double>(1, 2) = test.at<double>(1, 2) - 2.5;

        clock_t warp_st4 = clock();
        if(DEBUG_MSG)
        cout<< "##### Process_matrix time = " << static_cast<double>(warp_st4 - warp_st3) / CLOCKS_PER_SEC * 1000 << "ms #####" << endl;   

        clock_t warp_st = clock();
        if(VIP7K)
            im1t = cl_exc_affine(im1, test, 1);

        else
            warpAffine(im1, im1t, matrix, WEIGHT_BIGSIZE, INTER_NEAREST);

        clock_t warp_en = clock();

        if(DEBUG_MSG)
        cout<< "##### warpAffine time = " << static_cast<double>(warp_en - warp_st) / CLOCKS_PER_SEC * 1000 << "ms #####" << endl;

        clock_t warp_st5= clock();

        rear_before = rear_now;
    
        output.data = im1t.data;
        
        if(DEBUG_MSG_IMG)
            imwrite("debug/output.png",output);
        clock_t warp_st6 = clock();
        
        if(DEBUG_MSG)
            cout<< "##### Blending Process = " << static_cast<double>(warp_st6 - warp_st5) / CLOCKS_PER_SEC * 1000 << "ms #####" << endl; 

	}

	return output;
}


void Panorama::preProcess(Mat front_mask, Mat rear_mask)
{
	for (int i = 0; i < front_mask.rows; i++)
	{
		for (int j = 0; j < front_mask.cols; j++)
		{
			int val = (255.0 / front_mask.rows) * (i + 1);


			if (front_mask.ptr<uchar>(i)[j])
			{
				front_mask.ptr<uchar>(i)[j] = val;
			}
			if (rear_mask.ptr<uchar>(i)[j])
			{
				rear_mask.ptr<uchar>(i)[j] = val;
			}
		}
	}
	expand(front_mask, weight);

	mergeFrontRearMat(front_mask, rear_mask, weight);

	weight.convertTo(weight, CV_32FC1);
	weight /= 255.0;

}
