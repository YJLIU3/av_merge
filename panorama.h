#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

#define IMAGE_ELEM(image,pix_type,r,c) (image.ptr<pix_type>(r)[c])


class Panorama
{
public:
	Panorama();
	void compute_merge_matrix(Mat frontChessboard, Mat rearChessboard,
		Size board_size, int offset_x, int offset_y);
	void mergeFrontMat(Mat frontMat, Mat& output);
    void mergeRearMat(Mat frontMat, Mat& output);
	void mergeFrontRearMat(Mat frontMat, Mat rearMat, Mat& output);
	Mat front_process(Mat front, Mat rear);
    Mat rear_process(Mat front, Mat rear);

	void preProcess(Mat front_mask, Mat rear_mask);
	void expand(Mat input, Mat& output);
	~Panorama();


	Mat imTime1;
	Mat imTime2;

	Mat imMask1;
	Mat imMask2;

	int idx;

	Mat alpha;
	Mat alpha_1 = Mat::ones(alpha.size(),CV_32FC1);
	

	vector<Point2f> corners;
	
	//WEIGHT_BIGSIZE = Size(480 + offsize_a, 785 + offsize_b);
 	Mat im1t = Mat(Size(480, 905), CV_8UC4, Scalar::all(0));
 	Mat imMask1t = Mat(Size(480, 905), CV_8UC4, Scalar::all(0));
 	Mat imTime1t = Mat(Size(480, 905), CV_8UC4, Scalar::all(0));

public:


	Mat weight;

	Mat preImg;
	Mat merge_matrix;

	double K[3][3] = { {1, 0, 0}, {0, 1, 0} };   
	Mat matrix_back = Mat(2, 3, CV_64FC1, K);
    Mat matrix_zero = Mat(2, 3, CV_64FC1, K);



};
void init_system(class Panorama pa);

void  ShowImageToLCD(Mat img);

void TestProc();

void init_LCD(void);

void GetMapForRemap(Mat matrix,Mat Map_Fx, Mat Map_Fy);

void get_Univariate_matrix(void);

Mat av_merge_image(Mat front_image, Mat rear_image, bool Resersing);