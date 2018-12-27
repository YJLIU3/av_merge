#include <cl_api.h>

#define SUCCESS 0
#define FAILURE 1


#define CHECK_ERROR(actual, msg) \
if (actual != 0) \
{ \
    std::cout << "Error: " << msg << " .error code: " << actual << std::endl; \
    std::cout << "Location: " << __FILE__ << " line: " << __LINE__ << std::endl; \
    return actual; \
}

typedef struct 
{
    cl_context          context;
    cl_command_queue    CmdQue;
    cl_kernel           imgKernel;
    int                 width;
    int                 height;
    int                 channel;
    int                 total_size;
    cl_device_id        *devices;
    cl_program          program; 
    cl_mem              inputImgMap;     
    cl_mem              inputImgMap_x;
    cl_mem              inputImgMap_y;
    cl_mem              outputImgMap;
    void*               mapWriteImgPtr;
    void*               mapReadImgPtr_t;
    void*               mapWriteImgPtr_x;
    void*               mapWriteImgPtr_y;
    size_t              globaworksize[2];
    
}CLremapOBJS;

int init_cl_remap(CLremapOBJS &clremapobj, Mat map_x, Mat map_y);

/* convert the kernel file into a string */
int convertToString(const char *filename, std::string& s)
{
    size_t size;
    char*  str;
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));

    if(f.is_open())
    {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);
        str = new char[size+1];
        if(!str)
        {
            f.close();
            return 0;
        }

        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
        s = str;
        delete[] str;
        return 0;
    }
    cout<<"Error: failed to open file\n:"<<filename<<endl;
    return FAILURE;
}

#if 0
int main(int argc, char* argv[])
{
	struct timeval tv, tv1;

    Mat map_x = Mat::zeros(Size(260, 180), CV_32FC1);
    Mat map_y = Mat::zeros(Size(260, 180), CV_32FC1);
    for(int i = 0; i < 180; i++)
        for(int j =0; j < 260; j++)
        {
            map_x.at<float>(i, j) = (float)(j+50);
            map_y.at<float>(i, j) = (float)(i+50);
        }
    Mat input_mat = imread("input.png");
    int i=0;
    float f;
    while(1){        
    Mat input_mat = imread("input.png");
    cvtColor(input_mat, input_mat, CV_BGR2BGRA);


    gettimeofday(&tv, NULL);
    Mat output = cl_exc_remap(input_mat, map_x,  map_y);
    gettimeofday(&tv1, NULL);

    f = tv1.tv_sec*1000000.0 + tv1.tv_usec;
    f -= tv.tv_sec*1000000.0 + tv.tv_usec;
    f = f/1000.0;
	cout << "FrmNo."<<(i+1)<<"\tinterval "<< f <<endl;


    Mat ROI = output(Rect(0, 0, 260, 180));

    imwrite("output.png", ROI);
    }

    
    return 0;
}

#endif

int  init_cl_remap(CLremapOBJS &clremapobj, Mat map_x, Mat map_y)
{
    if(clremapobj.context != NULL)
        clReleaseContext(clremapobj.context);  
    clremapobj.width = 260;
    clremapobj.height = 180;
    clremapobj.channel = 4;
    clremapobj.total_size = clremapobj.height * clremapobj.width;
    cl_uint numPlatforms;
    cl_platform_id platform = NULL;
    cl_int    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS)
    {
        cout << "Error: Getting platforms!" << endl;
        return FAILURE;
    }
    if(numPlatforms > 0)
    {
        cl_platform_id* platforms = (cl_platform_id* )malloc(numPlatforms* sizeof(cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        platform = platforms[0];
        free(platforms);
    }
    cout << "1" << endl;
    cl_uint             numDevices = 0;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if (numDevices == 0)    //no GPU available.
    {
        cout << "No GPU device available." << endl;
        cout << "Choose CPU as default device." << endl;
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
        clremapobj.devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, clremapobj.devices, NULL);
    }
    else
    {
        clremapobj.devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, clremapobj.devices, NULL);
    }
    clremapobj.context = clCreateContext(NULL,1, clremapobj.devices,NULL,NULL,NULL);
    if(clremapobj.context == NULL)
        cout << "err" << endl;
    /*Step 4: Creating command queue associate with the context.*/
    cout << "2" << endl;

    clremapobj.CmdQue = clCreateCommandQueue(clremapobj.context, clremapobj.devices[0], 0, NULL);
    cout << "3" << endl;

    const char *filename = "rotate.cl";
    string sourceStr;
    status = convertToString(filename, sourceStr);
    const char *source = sourceStr.c_str();
    size_t sourceSize[] = {strlen(source)};
    clremapobj.program = clCreateProgramWithSource(clremapobj.context, 1, &source, sourceSize, &status);

    status=clBuildProgram(clremapobj.program, 1, clremapobj.devices, NULL, NULL, NULL);
    CHECK_ERROR(status, "clCreateProgarm failed.");
    cout << "4" << endl;

    cl_image_format img_format;
    img_format.image_channel_data_type = CL_UNSIGNED_INT8;
    img_format.image_channel_order = CL_RGBA;

    cl_image_desc pixelDesc;
    pixelDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    pixelDesc.image_width = 1280;
    pixelDesc.image_height = 720;
    pixelDesc.image_depth = 0;
    pixelDesc.num_mip_levels = 0;
    pixelDesc.num_samples = 0;
    pixelDesc.image_row_pitch = 0;
    pixelDesc.image_array_size = 0;
    pixelDesc.image_slice_pitch = 0;

    cout << "5" << endl;

    clremapobj.inputImgMap_x = clCreateBuffer(clremapobj.context,CL_MEM_USE_HOST_PTR, clremapobj.total_size * sizeof(float),map_x.data,NULL);
    clremapobj.inputImgMap_y = clCreateBuffer(clremapobj.context,CL_MEM_USE_HOST_PTR, clremapobj.total_size * sizeof(float),map_y.data,NULL);

    cout << "6" << endl;


    clremapobj.inputImgMap = clCreateImage(clremapobj.context, CL_MEM_READ_ONLY || CL_MEM_ALLOC_HOST_PTR, &img_format, &pixelDesc, NULL, &status);
    CHECK_ERROR(status, "clCreateImage2D failed. (inputImgMap)");
    clremapobj.outputImgMap = clCreateImage(clremapobj.context, CL_MEM_WRITE_ONLY || CL_MEM_ALLOC_HOST_PTR, &img_format, &pixelDesc, NULL, &status);
    CHECK_ERROR(status, "clCreateImage2D failed. (outputImage)");
    
    size_t imageOrigin[3];
    size_t imageRegion[3];

    imageOrigin[0] = 0;
    imageOrigin[1] = 0;
    imageOrigin[2] = 0;

    imageRegion[0] = 1280;
    imageRegion[1] = 720;
    imageRegion[2] = 1;

    
    size_t imageOrigin_out[3];
    size_t imageRegion_out[3];

    imageOrigin_out[0] = 0;
    imageOrigin_out[1] = 0;
    imageOrigin_out[2] = 0;

    imageRegion_out[0] = 260;
    imageRegion_out[1] = 180;
    imageRegion_out[2] = 1;

    size_t imageRowPitch_in = 1280 * sizeof(char) * clremapobj.channel;

    clremapobj.mapReadImgPtr_t = clEnqueueMapImage( clremapobj.CmdQue, clremapobj.outputImgMap, CL_TRUE, CL_MAP_READ, 
        imageOrigin, imageRegion,&imageRowPitch_in, NULL, 0, NULL, NULL, &status);
    CHECK_ERROR(status, "clEnqueueMapBuffer failed. (resultBuf)");

    clremapobj.globaworksize[0] = clremapobj.width;
    clremapobj.globaworksize[1] = clremapobj.height;

    clremapobj.imgKernel = clCreateKernel(clremapobj.program, "imageRemap", &status);
    CHECK_ERROR(status, "clCreateKernel failed.");
    
    
    int argIdx = 0;
    status |= clSetKernelArg(clremapobj.imgKernel, argIdx++, sizeof(cl_mem), &clremapobj.outputImgMap);
    status |= clSetKernelArg(clremapobj.imgKernel, argIdx++, sizeof(cl_mem), &clremapobj.inputImgMap);
    status |= clSetKernelArg(clremapobj.imgKernel, argIdx++, sizeof(cl_mem), &clremapobj.inputImgMap_x);
    status |= clSetKernelArg(clremapobj.imgKernel, argIdx++, sizeof(cl_mem), &clremapobj.inputImgMap_y);

}


    
Mat cl_exc_remap_r(Mat input, Mat map_x, Mat map_y)
{
    static bool _int = true;
    static CLremapOBJS clremapobj;    
    static Mat output = Mat::zeros(Size(1280, 720), CV_8UC4);
    static Mat ROI = Mat::zeros(Size(260, 180), CV_8UC4);
    if(_int)
    {
        cout << "init" << endl;
        _int = false;
        memset(&clremapobj, 0x00, sizeof(CLremapOBJS));
        init_cl_remap(clremapobj, map_x, map_y);
            static size_t imageOrigin[3];
    static size_t imageRegion[3];

    imageOrigin[0] = 0;
    imageOrigin[1] = 0;
    imageOrigin[2] = 0;

    imageRegion[0] = 1280;
    imageRegion[1] = 720;
    imageRegion[2] = 1;
    
    size_t imageRowPitch_in = 1280 * sizeof(char) * clremapobj.channel;
    clEnqueueWriteImage( clremapobj.CmdQue, clremapobj.inputImgMap, CL_TRUE,
            imageOrigin, imageRegion, imageRowPitch_in, NULL, input.data, NULL, NULL, NULL);
 
    }

    struct timeval tv, tv1;
    long double f;
    gettimeofday(&tv, NULL);


    cl_event ndrEvt;
    int status = clEnqueueNDRangeKernel(clremapobj.CmdQue, clremapobj.imgKernel, 2, NULL,
                    clremapobj.globaworksize, NULL, 0, NULL, &ndrEvt);

    clFinish(clremapobj.CmdQue);

    gettimeofday(&tv1, NULL);

    f = (tv1.tv_sec-tv.tv_sec)*1000.0 + (tv1.tv_usec - tv.tv_usec)/1000.0;
	cout << "FrmNo."<<"_"<<"\tinterval "<< f <<endl;
    
    output.data = (uchar *)clremapobj.mapReadImgPtr_t;
    //output(Rect(0, 0, 260, 180)).copyTo(ROI);
    ROI = output(Rect(0, 0, 260, 180));
    return ROI;
}

Mat cl_exc_remap_f(Mat input, Mat map_x, Mat map_y)
{
    static bool _int = true;
    static CLremapOBJS clremapobj;    
    static Mat output = Mat::zeros(Size(1280, 720), CV_8UC4);
    static Mat ROI = Mat::zeros(Size(260, 180), CV_8UC4);
    if(_int)
    {
        cout << "init" << endl;
        _int = false;
        memset(&clremapobj, 0x00, sizeof(CLremapOBJS));
        init_cl_remap(clremapobj, map_x, map_y);
    }

    struct timeval tv, tv1;
    float f;

    gettimeofday(&tv, NULL);
    
    memcpy(clremapobj.mapWriteImgPtr, (char *)input.data, input.cols * input.rows * sizeof(CV_8UC4));

    cl_event ndrEvt;
    int status = clEnqueueNDRangeKernel(clremapobj.CmdQue, clremapobj.imgKernel, 2, NULL,
                    clremapobj.globaworksize, NULL, 0, NULL, &ndrEvt);
    cout << status << endl;
    clFinish(clremapobj.CmdQue);

    gettimeofday(&tv1, NULL);

    f = tv1.tv_sec*1000000.0 + tv1.tv_usec;
    f -= tv.tv_sec*1000000.0 + tv.tv_usec;
    f = f/1000.0;
	cout << "FrmNo."<<"_"<<"\tinterval "<< f <<endl;
    
    output.data = (uchar *)clremapobj.mapReadImgPtr_t;
    output(Rect(0, 0, 260, 180)).copyTo(ROI);
    return ROI;
}

