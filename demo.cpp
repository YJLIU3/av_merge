#include <opencv2/opencv.hpp>
#include <panorama.h>
#include <iostream>
#include <cl_api.h>

using namespace std;
using namespace cv;
extern "C" uchar * av_merge_image(uchar * front_buf, uchar * rear_buf, bool Reversing);

Mat test_a = Mat::zeros(Size(1280,720),CV_8UC4);
Mat test_b = Mat::zeros(Size(1280,720),CV_8UC4);

int main()
{
    char * buff_in_a = (char *)malloc(sizeof(char) * 1280*720*4);
    char * buff_in_b = (char *)malloc(sizeof(char) * 1280*720*4);
	VideoCapture front_cap("3_F.mkv");
	VideoCapture rear_cap("3_B.mkv"); 
    Mat front_image,rear_image;
	clock_t total_start = clock();
    if (front_cap.isOpened() && rear_cap.isOpened())
    {
            
        front_cap >> front_image;
        rear_cap >> rear_image;
        cvtColor(front_image, front_image, CV_BGR2BGRA);
        cvtColor(rear_image, rear_image, CV_BGR2BGRA);

         test_a.data = front_image.data;
         test_b.data = rear_image.data;


        
        memcpy(buff_in_a, front_image.data, sizeof(char) * 1280*720*4);
        memcpy(buff_in_b, rear_image.data, sizeof(char) * 1280*720*4);
        cout << front_image.type() << endl;
        Mat output_mat = Mat::zeros(Size(320, 640), CV_8UC4);
        while (front_image.data && rear_image.data)
        {
            output_mat.data = (uchar *)av_merge_image((uchar *)front_image.data, (uchar *)rear_image.data, 1);
            
            front_cap >> front_image;
            rear_cap >> rear_image;
                
        }
        front_cap.release();
        rear_cap.release();
    }
    return 0;
}
