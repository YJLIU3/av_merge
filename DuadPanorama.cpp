#include "panorama.h"
#include <opencv2/opencv.hpp>
#include "parameter.h"
#include <time.h> 
#include <cv_vx.h>
#include <cl_api.h>
#include <sstream>

using namespace std;
using namespace cv;



#define Size_ARGB_F 1280*720*4
#define Size_ARGB_R 1280*720*4

#define Size_Out_AGRB 640*320*4

#define av_inSize_F Size(1280, 720)
#define av_inSize_R Size(1280, 720)

#define av_outSize Size(320, 640)

static char *ptr = NULL;
static char *pData=NULL;
static char * out_buf = (char *)malloc(sizeof(char) * Size_Out_AGRB);
static Mat Map_Fx = Mat(image_size, CV_32FC1);
static Mat Map_Fy = Mat(image_size, CV_32FC1);
static Mat Map_Rx = Mat(image_size, CV_32FC1);
static Mat Map_Ry = Mat(image_size, CV_32FC1);

Mat front_mask, rear_mask;
Mat front_mask1, rear_mask1;
Mat frontMat,rearMat;

static bool init_ = true;
static Panorama pa;

static Mat front_image = Mat::zeros(av_inSize_F,CV_8UC4);
static Mat rear_image = Mat::zeros(av_inSize_R,CV_8UC4);

Mat out = Mat::ones(av_outSize,CV_8UC4);
Mat show_img = Mat::ones(av_outSize,CV_8UC4);

Mat front_trs(image_size, CV_8UC4, Scalar::all(0));
Mat rear_trs(image_size, CV_8UC4, Scalar::all(0));

void GetMapForRemap(Mat matrix[(grid_rows - 1)*(grid_cols - 1)],Mat Map_Fx, Mat Map_Fy)
{

	Mat output;

	for (size_t i = 0; i < (grid_rows - 1)*(grid_cols - 1); i++)
	{
		matrix[i] = matrix[i].inv();
	}

	Point2f po;

	for (size_t p = 0; p < grid_cols - 1; p++)
	{
		for (size_t q = 0; q < grid_rows - 1; q++)
		{
			vector<Point2f> Map_FP;
			vector<Point2f> SRC;
			for (float i = (grid_size) * p; i < (grid_size)*(p + 1); i++)
			{
				for (float j = (grid_size) * q; j < (grid_size)*(q + 1); j++)
				{
					po.x = j;
					po.y = i;
					SRC.push_back(po);
				}
			}
			perspectiveTransform(SRC, Map_FP, matrix[p*(grid_rows - 1) + q]);
			int idpix = 0;
			for (float i = grid_size * p; i < grid_size*(p + 1); i++)
			{
				for (float j = grid_size * q; j < grid_size*(q + 1); j++)
				{
					idpix = (i - grid_size * p) * grid_size + (j - grid_size * q);
					Map_Fx.at<float>(i, j) = Map_FP[idpix].x;
					Map_Fy.at<float>(i, j) = Map_FP[idpix].y;
				}
			}
			Map_FP.clear();
		}
	}	
}


Mat Mask[grid_rows * grid_cols];
Mat matrix_affine[grid_rows*grid_cols];
Mat matrix_affine_r[grid_rows*grid_cols];
Mat result[grid_rows*grid_cols];
Mat result_r[grid_rows*grid_cols];

void get_Univariate_matrix(void)
{
#if 1
		Mat img = imread("F.bmp",0);
		Mat img_r = imread("B.bmp",0);

		vector<Point2f> corners;
		vector<Point2f> corners_r;
        vector<Point2f> corner_tmp;
		TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 40, 0.1);

		cout << findChessboardCorners(img, CALIBRATOR_BOARD_SIZE, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE) << endl;
		
		cout << findChessboardCorners(img_r, CALIBRATOR_BOARD_SIZE, corners_r, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE) << endl;
		
		cornerSubPix(img, corners, Size(5,5), Size(-1,-1), criteria);
		cornerSubPix(img_r, corners_r, Size(5,5), Size(-1, -1), criteria);

        for(int i = corners_r.size() - 1; i > -1; i--)
        {
            corner_tmp.push_back(corners_r[i]);
        }
        corners_r = corner_tmp;
#endif		

	
		Point2f Src[grid_cols*grid_rows], Dst[grid_cols*grid_rows], Src_r[grid_cols*grid_rows];
		for (int i = 0; i < grid_cols*grid_rows; i++)
		{
			Src[i] = corners[i];
			Src_r[i] = corners_r[i];
		}
		for (int i = 0; i < grid_rows; i++)
		{
			if (i == 0)
			{
				for(int j = 0; j < grid_cols; j++)
				{
					Dst[i + grid_rows * j].x = 0;
				}
			}
			else
			{
				for(int j = 0; j < grid_cols; j++)
				{
					Dst[i + grid_rows * j].x = i * grid_size - 1;
				}
			}
			Dst[i].y = 0;
			for (int j = 1; j < grid_cols; j++)
			{
				Dst[i + grid_rows * j].y = grid_size * j - 1;;
			}
		}
		vector<Point2f> Dsst, Test;
		for (int i = 0; i < grid_cols*grid_rows; i++)
		{
			Dsst.push_back(Dst[i]);
		}
		for (int i = 0; i < grid_rows-1; i++)
		{
			Point2f m[4], n[4], m_r[4];
			for(int j = 0; j < grid_cols - 1; j++)
			{
				m[0] = corners[grid_rows * j + i];
				m[1] = corners[grid_rows * j + i + 1];
				m[2] = corners[grid_rows * (j + 1) + i];
				m[3] = corners[grid_rows * (j + 1) + 1 + i];
				m_r[0] = corners_r[grid_rows * j + i];
				m_r[1] = corners_r[grid_rows * j + i + 1];
				m_r[2] = corners_r[grid_rows * (j + 1) + i];
				m_r[3] = corners_r[grid_rows * (j + 1) + 1 + i];
				n[0] = Dst[grid_rows * j + i];
				n[1] = Dst[grid_rows * j + i + 1];
				n[2] = Dst[grid_rows * (j + 1) + i];
				n[3] = Dst[grid_rows * (j + 1) + 1 + i];
				matrix_affine_r[i + (grid_rows - 1) * j] = getPerspectiveTransform(m_r, n);
				matrix_affine[i + (grid_rows - 1) * j] = getPerspectiveTransform(m, n);	
			}

		}

}



static bool _init_vx_remap_F = true;
static bool _init_vx_remap_R = true;

Mat av_merge(Mat front_image, Mat rear_image, bool Reversing)
{
	Mat out;

    if(!Reversing)
    {
        clock_t end_remap = clock();
        if(VIP7K)
            front_trs = cl_exc_remap(front_image, Map_Rx, Map_Ry);
        else
            remap(front_image, front_trs, Map_Rx, Map_Ry, INTER_NEAREST, BORDER_CONSTANT);

        if(front_trs.size() != image_size)
    	{
            if(DEBUG_MSG)
    		    cout << "#################resize####################"<< endl;
    		resize(front_trs, front_trs, image_size);
    		resize(rear_trs, rear_trs, image_size);
    	}
        imwrite("debug/front_trs.jpg", front_trs);
        clock_t end_process = clock();
        if(!DEBUG_MSG)
            cout<< "###### front process Running time  is: " << static_cast<double>(end_process - end_remap) / CLOCKS_PER_SEC * 1000 << "ms #####" << endl;
        out = pa.front_process(front_trs, rear_trs);

    }
    else
    {
        clock_t end_remap = clock();
        if(VIP7K)
            rear_trs = cl_exc_remap(rear_image, Map_Rx, Map_Ry);
        else
            remap(rear_image, rear_trs, Map_Rx, Map_Ry, INTER_NEAREST, BORDER_CONSTANT);
        
        if(DEBUG_MSG_IMG)
        imwrite("debug/rear_trs.png",rear_trs);
        if(front_trs.size() != image_size)
    	{
            if(DEBUG_MSG)
    		    cout << "#################resize####################"<< endl;
    		resize(front_trs, front_trs, image_size);
    		resize(rear_trs, rear_trs, image_size);
    	}
        clock_t end_process = clock();
        if(DEBUG_MSG)
            cout<< "##### Remap time  = " << static_cast<double>(end_process - end_remap) / CLOCKS_PER_SEC * 1000 << "ms #####" << endl;        
        out = pa.rear_process(front_trs, rear_trs);

        
    }	
	return out;
}

extern "C" char * av_merge_image(char * front_buf, char * rear_buf, bool Reversing);
char* av_merge_image(char * front_buf, char * rear_buf, bool Reversing)
{
	if (init_ == true) 
	{
#if 0

        /*************  biao ding ***************/
		cout << "*********start**********"<<endl;
		get_Univariate_matrix();
     
		cout << "*********start-1**********"<<endl;
		GetMapForRemap(matrix_affine, Map_Fx, Map_Fy);
    
		GetMapForRemap(matrix_affine_r, Map_Rx, Map_Ry);
        /******************************************/


        ofstream output( "Map_Rx.txt", ios::out | ios::binary );
        if( ! output )
        {
            cerr << "Open output file error!" << endl;
            exit( -1 );
        }
        output.write ((char *) Map_Rx.data, sizeof( float ) * 260 * 180);

        ofstream output1( "Map_Ry.txt", ios::out | ios::binary );
        if( ! output1 )
        {
            cerr << "Open output file error!" << endl;
            exit( -1 );
        }
        output1.write ((char *) Map_Ry.data, sizeof( float ) * 260 * 180);
        
        ofstream output2( "Map_Fy.txt", ios::out | ios::binary );
        if( ! output2 )
        {
            cerr << "Open output file error!" << endl;
            exit( -1 );
        }
        output2.write ((char *) Map_Fy.data, sizeof( float ) * 260 * 180);

        ofstream output3( "Map_Fx.txt", ios::out | ios::binary );
        if( ! output3 )
        {
            cerr << "Open output file error!" << endl;
            exit( -1 );
        }
        output3.write ((char *) Map_Fx.data, sizeof( float ) * 260 * 180);        

#endif

        ifstream input( "Map_Rx.txt", ios::in | ios::binary );
        if( ! input )
        {
            cerr << "Open input file error!" << endl;
            exit( -1 );
        }
        input.read( ( char * )Map_Rx.data , sizeof( float ) * 260 * 180);


        ifstream input1( "Map_Ry.txt", ios::in | ios::binary );
        if( ! input1 )
        {
            cerr << "Open input file error!" << endl;
            exit( -1 );
        }
        input1.read( ( char * )Map_Ry.data , sizeof( float ) * 260 * 180);

        Mat front_chess = imread("F.bmp");
		Mat rear_chess = imread("B.bmp");
		remap(front_chess, front_chess, Map_Fx, Map_Fy, INTER_LINEAR, BORDER_CONSTANT);
		remap(rear_chess, rear_chess, Map_Rx, Map_Ry, INTER_LINEAR, BORDER_CONSTANT);

        if(!DEBUG_MSG_IMG)
        {
            imwrite("debug/F_chess.jpg", front_chess);
		    imwrite("debug/B_chess.jpg", rear_chess);
        }
		front_mask1 = Mat::ones(image_size, CV_8UC1);

		rear_mask1 = Mat::ones(image_size, CV_8UC1);

		pa.preProcess(front_mask1, rear_mask1);
        if(DEBUG_MSG)
            cout << "##################end Init_parameter###########" <<endl;
		init_ = false;
	}

    clock_t st_b = clock();
    if(!Reversing)
    {
        front_image.data = (unsigned char *)front_buf;
        if(DEBUG_MSG_IMG)
            imwrite("debug/front_input.png", front_image);
        clock_t en_b = clock();
        if(DEBUG_MSG)

        cout<< "###############################bef Running time  is: " << static_cast<double>(en_b - st_b) / CLOCKS_PER_SEC * 1000 << "ms#####################" << endl;
    
        out =  av_merge(front_image, rear_image, Reversing);
    }
    else
    {
        rear_image.data = (unsigned char *)rear_buf;

        if(DEBUG_MSG_IMG)
            imwrite("debug/rear_input.png", rear_image);
        
        clock_t en_c = clock();
        if(DEBUG_MSG)
            cout<< "##### bef Running time  is: " << static_cast<double>(en_c - st_b) / CLOCKS_PER_SEC * 1000 << "ms #####" << endl;
        out =  av_merge(front_image, rear_image, Reversing); 
    }

    clock_t st_up = clock();

    if(DEBUG_MSG_IMG)
        imwrite("debug/up_flash.png", out);

    
    clock_t en_up = clock();
    if(DEBUG_MSG)
    cout<< "##### Up Running time  is: " << static_cast<double>(en_up - st_up) / CLOCKS_PER_SEC * 1000 << "ms#####" << endl;

	return (char *)out.data;
}

