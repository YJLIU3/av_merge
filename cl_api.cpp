#include <cl_api.h>

#define SUCCESS 0
#define FAILURE 1
#define Cmd_Que_Num 4
#define Map_tab_wid 260
#define Map_tab_hei 180
#define input_size Size(1280, 720)
#define map_tab_size Size(260, 180)

static void *remap_ptr_table_f[Cmd_Que_Num];
static void *remap_ptr_table_r[Cmd_Que_Num];

#define Affine_W 320
#define Affine_H 640

#define Map_ch 4

#define CHECK_ERROR(actual, msg) \
if (actual != 0) \
{ \
    std::cout << "Error: " << msg << " .error code: " << actual << std::endl; \
    std::cout << "Location: " << __FILE__ << " line: " << __LINE__ << std::endl; \
    return actual; \
}

typedef struct 
{
    cl_command_queue    CmdQue[2];
    cl_kernel           imgKernel[2];
    cl_program          program[2]; 
    cl_mem              inputImgMap;     
    cl_mem              affine_matrix;
    cl_mem              outputImgMap;
    void*               mapPtr_in;
    void*               mapPtr_out;
    void*               affine_matrix_Ptr;
    
}CLaffineOBJ;


typedef struct 
{
    cl_command_queue    CmdQue[Cmd_Que_Num];
    cl_kernel           imgKernel[Cmd_Que_Num];
    cl_program          program[Cmd_Que_Num]; 
    cl_mem              inputImgMap[Cmd_Que_Num];     
    cl_mem              inputImgMap_x;
    cl_mem              inputImgMap_y;
    cl_mem              outputImgMap;
    void*               mapPtr_in[Cmd_Que_Num];
    void*               mapPtr_out;
    void*               mapWriteImgPtr_x;
    void*               mapWriteImgPtr_y;
    
}CLmemOBJ;

typedef struct 
{
    cl_context          context;
    cl_device_id        *devices;
    size_t              globaworksize[2];
    
}CLpltOBJ;

static CLpltOBJ clpltobj;
static CLmemOBJ clmemobj;
static CLmemOBJ clmemobj_r;
static CLaffineOBJ affineobj;



int init_cl_remap(CLpltOBJ &pltobj, CLmemOBJ &memobj, Mat map_x, Mat map_y);

/*================== convert the kernel file into a string ==================*/
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

int init_cl_plt(CLpltOBJ &pltobj)
{
    if(pltobj.context != NULL)
        clReleaseContext(pltobj.context);
    cl_uint numPlatforms;
    cl_platform_id platform = NULL;
    int    status = clGetPlatformIDs(0, NULL, &numPlatforms);
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
    cl_uint             numDevices = 0;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if (numDevices == 0)
    {
        cout << "No GPU device available." << endl;
        cout << "Choose CPU as default device." << endl;
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
        pltobj.devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, pltobj.devices, NULL);
    }
    else
    {
        pltobj.devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, pltobj.devices, NULL);
    }
    pltobj.context = clCreateContext(NULL,1, pltobj.devices,NULL,NULL,NULL);
    if(pltobj.context == NULL)
        cout << "Create GPU context false!" << endl;

    pltobj.globaworksize[0] = Map_tab_wid;
    pltobj.globaworksize[1] = Map_tab_hei;

    return status;
}

int init_cl_mem_obj(CLpltOBJ &pltobj, CLmemOBJ &memobj, Mat map_x, Mat map_y, void *GC50UT_ptr[])
{
    int status;
    int data_size = Map_tab_wid * Map_tab_hei;
    const char *filename = "rotate.cl";
    string sourceStr;
    status = convertToString(filename, sourceStr);
    const char *source = sourceStr.c_str();
    size_t sourceSize[] = {strlen(source)};
    for(int i = 0; i < Cmd_Que_Num; i++)
    {
        memobj.CmdQue[i] = clCreateCommandQueue(pltobj.context, pltobj.devices[0], 0, &status);
        CHECK_ERROR(status, "clCreateCommamdQueue failed.");
        memobj.program[i] = clCreateProgramWithSource(pltobj.context, 1, &source, sourceSize, &status);
        status = clBuildProgram(memobj.program[i], 1, pltobj.devices, NULL, NULL, NULL);
        CHECK_ERROR(status, "clCreateProgarm failed.");
        memobj.imgKernel[i] = clCreateKernel(memobj.program[i], "imageRemap", &status);
        CHECK_ERROR(status, "clCreateKernel failed.");
    }
    
    memobj.inputImgMap_x = clCreateBuffer(pltobj.context,CL_MEM_USE_HOST_PTR, data_size * sizeof(float),map_x.data,&status);
    CHECK_ERROR(status, "clCreateMapTable failed.");
    memobj.inputImgMap_y = clCreateBuffer(pltobj.context,CL_MEM_USE_HOST_PTR, data_size * sizeof(float),map_y.data,&status);
    CHECK_ERROR(status, "clCreateMapTable failed.");

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

    size_t imageOrigin[3];
    size_t imageRegion[3];

    imageOrigin[0] = 0;
    imageOrigin[1] = 0;
    imageOrigin[2] = 0;

    imageRegion[0] = 1280;
    imageRegion[1] = 720;
    imageRegion[2] = 1;
    size_t imageRowPitch_in = 1280 * sizeof(char) * Map_ch;
    
    for(int i = 0; i < Cmd_Que_Num; i++)
    {
        memobj.inputImgMap[i] = clCreateImage(pltobj.context, CL_MEM_READ_ONLY || CL_MEM_ALLOC_HOST_PTR, &img_format, &pixelDesc, NULL, &status);
        CHECK_ERROR(status, "clCreateImage2D failed. (inputImgMap)");

        memobj.mapPtr_in[i] = clEnqueueMapImage( memobj.CmdQue[i], memobj.inputImgMap[i], CL_TRUE, CL_MAP_WRITE, 
        imageOrigin, imageRegion, &imageRowPitch_in, NULL, 0, NULL, NULL, &status);
        CHECK_ERROR(status, "clEnqueueMapBuffer failed. (resultBuf)");
        GC50UT_ptr[i] = memobj.mapPtr_in[i];
        cout << GC50UT_ptr[i] << endl;
    }
    memobj.outputImgMap = clCreateImage(pltobj.context, CL_MEM_WRITE_ONLY || CL_MEM_ALLOC_HOST_PTR, &img_format, &pixelDesc, NULL, &status);
    CHECK_ERROR(status, "clCreateImage2D failed. (outputImage)");

    for(int i = 0; i < Cmd_Que_Num; i++)
    {
        int argIdx = 0;
        status |= clSetKernelArg(memobj.imgKernel[i], argIdx++, sizeof(cl_mem), &memobj.outputImgMap);
        status |= clSetKernelArg(memobj.imgKernel[i], argIdx++, sizeof(cl_mem), &memobj.inputImgMap[i]);
        status |= clSetKernelArg(memobj.imgKernel[i], argIdx++, sizeof(cl_mem), &memobj.inputImgMap_x);
        status |= clSetKernelArg(memobj.imgKernel[i], argIdx++, sizeof(cl_mem), &memobj.inputImgMap_y);
    }
    clmemobj.mapPtr_out = clEnqueueMapImage( clmemobj.CmdQue[3], clmemobj.outputImgMap, CL_TRUE, CL_MAP_READ, 
         imageOrigin, imageRegion,&imageRowPitch_in, NULL, 0, NULL, NULL, NULL); 
    cout << " 5 " << endl;
    return status;
}
extern "C" int gpu7k_get_viraddr(void* addrarray_f[], void* addrarray_r[], int size, int addr_cnt);

int gpu7k_get_viraddr(void* addrarray_f[], void* addrarray_r[], int size, int addr_cnt)
{
    int status;
    static Mat Map_Fx = Mat(map_tab_size, CV_32FC1);
    static Mat Map_Fy = Mat(map_tab_size, CV_32FC1);
    static Mat Map_Rx = Mat(map_tab_size, CV_32FC1);
    static Mat Map_Ry = Mat(map_tab_size, CV_32FC1);

    ifstream input( "Map_Rx.txt", ios::in | ios::binary );
    if( ! input )
    {
        cerr << "Open input file error!" << endl;
        exit( -1 );
    }
    input.read( ( char * )Map_Rx.data , sizeof( float ) * Map_tab_wid * Map_tab_hei);
    
    
    ifstream input1( "Map_Ry.txt", ios::in | ios::binary );
    if( ! input1 )
    {
        cerr << "Open input file error!" << endl;
        exit( -1 );
    }
    input1.read( ( char * )Map_Ry.data , sizeof( float ) * Map_tab_wid * Map_tab_hei);

    Map_Fx = Map_Rx.clone();
    Map_Fy = Map_Ry.clone();
    
    memset(&clpltobj, 0x00, sizeof(CLpltOBJ));
    memset(&clmemobj, 0x00, sizeof(CLmemOBJ));
    memset(&clmemobj_r, 0x00, sizeof(CLmemOBJ));
    status = init_cl_plt(clpltobj);
    status |= init_cl_mem_obj(clpltobj, clmemobj, Map_Fx, Map_Fy, addrarray_f);
    status |= init_cl_mem_obj(clpltobj, clmemobj_r, Map_Rx, Map_Ry, addrarray_r);

    for(int i = 0; i < Cmd_Que_Num; i++)
    {
        remap_ptr_table_f[i] = addrarray_f[i];
        remap_ptr_table_r[i] = addrarray_r[i];        
    }

}
    
Mat cl_exc_remap(Mat input, Mat map_x, Mat map_y)
{
    static Mat output = Mat::zeros(input_size, CV_8UC4);
    static Mat ROI = Mat::zeros(Size(260, 180), CV_8UC4);

    cl_event ndrEvt;
    for(int i = 0; i < Cmd_Que_Num; i++)
    {
        if(input.data == remap_ptr_table_f[i])
        {
            int status = clEnqueueNDRangeKernel(clmemobj.CmdQue[i], clmemobj.imgKernel[i], 2, NULL,
                    clpltobj.globaworksize, NULL, 0, NULL, &ndrEvt);
            clFinish(clmemobj.CmdQue[i]);
        }
        if(input.data == remap_ptr_table_r[i])
        {
            int status = clEnqueueNDRangeKernel(clmemobj_r.CmdQue[i], clmemobj_r.imgKernel[i], 2, NULL,
                    clpltobj.globaworksize, NULL, 0, NULL, &ndrEvt);
            clFinish(clmemobj_r.CmdQue[i]);
        }
    }    
    output.data = (uchar *)clmemobj.mapPtr_out;
    output(Rect(0, 0, 260, 180)).copyTo(ROI);
    return ROI;
}
/************************************************************* cl warpAffine ******************************************************************************************/



int init_cl_affine_obj(CLpltOBJ &pltobj, CLaffineOBJ &memobj, void * ptr[])
{
    int status;
    int data_size = Affine_H * Affine_W;
    const char *filename = "rotate.cl";
    string sourceStr;
    status = convertToString(filename, sourceStr);
    const char *source = sourceStr.c_str();
    size_t sourceSize[] = {strlen(source)};
    
    memobj.CmdQue[0] = clCreateCommandQueue(pltobj.context, pltobj.devices[0], 0, &status);
    CHECK_ERROR(status, "clCreateCommamdQueue failed.");
    memobj.program[0] = clCreateProgramWithSource(pltobj.context, 1, &source, sourceSize, &status);
    status = clBuildProgram(memobj.program[0], 1, pltobj.devices, NULL, NULL, NULL);
    CHECK_ERROR(status, "clCreateProgarm failed.");
    memobj.imgKernel[0] = clCreateKernel(memobj.program[0], "imageAffine", &status);
    CHECK_ERROR(status, "clCreateKernel failed.");


    memobj.CmdQue[1] = clCreateCommandQueue(pltobj.context, pltobj.devices[0], 0, &status);
    CHECK_ERROR(status, "clCreateCommamdQueue failed.");
    memobj.program[1] = clCreateProgramWithSource(pltobj.context, 1, &source, sourceSize, &status);
    status = clBuildProgram(memobj.program[1], 1, pltobj.devices, NULL, NULL, NULL);
    CHECK_ERROR(status, "clCreateProgarm failed.");
    memobj.imgKernel[1] = clCreateKernel(memobj.program[1], "imageCopy", &status);
    CHECK_ERROR(status, "clCreateKernel failed.");

    
    memobj.affine_matrix = clCreateBuffer(pltobj.context,CL_MEM_READ_ONLY || CL_MEM_ALLOC_HOST_PTR, 6 * sizeof(float), NULL,&status);
    CHECK_ERROR(status, "clCreateMapTable failed.");
    memobj.affine_matrix_Ptr = clEnqueueMapBuffer(memobj.CmdQue[0], memobj.affine_matrix, CL_TRUE, CL_MAP_WRITE,
                        0, 6 * sizeof(float), 0, NULL, NULL, NULL);

    cl_image_format img_format;
    img_format.image_channel_data_type = CL_UNSIGNED_INT8;
    img_format.image_channel_order = CL_RGBA;

    cl_image_desc pixelDesc;
    pixelDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    pixelDesc.image_width = Affine_W;
    pixelDesc.image_height = Affine_H;
    pixelDesc.image_depth = 0;
    pixelDesc.num_mip_levels = 0;
    pixelDesc.num_samples = 0;
    pixelDesc.image_row_pitch = 0;
    pixelDesc.image_array_size = 0;
    pixelDesc.image_slice_pitch = 0;

    size_t imageOrigin[3];
    size_t imageRegion[3];

    imageOrigin[0] = 0;
    imageOrigin[1] = 0;
    imageOrigin[2] = 0;

    imageRegion[0] = Affine_W;
    imageRegion[1] = Affine_H;
    imageRegion[2] = 1;
    size_t imageRowPitch_in = Affine_W * sizeof(char) * Map_ch;
    
    memobj.inputImgMap = clCreateImage(pltobj.context, CL_MEM_READ_WRITE || CL_MEM_ALLOC_HOST_PTR, &img_format, &pixelDesc, NULL, &status);
    CHECK_ERROR(status, "clCreateImage2D failed. (inputImgMap)");
    memobj.mapPtr_in = clEnqueueMapImage( memobj.CmdQue[0], memobj.inputImgMap, CL_TRUE, CL_MAP_WRITE, 
    imageOrigin, imageRegion, &imageRowPitch_in, NULL, 0, NULL, NULL, &status);
    CHECK_ERROR(status, "clEnqueueMapBuffer failed. (resultBuf)");



    memobj.outputImgMap = clCreateImage(pltobj.context, CL_MEM_READ_WRITE || CL_MEM_ALLOC_HOST_PTR, &img_format, &pixelDesc, NULL, &status);
    CHECK_ERROR(status, "clCreateImage2D failed. (outputImage)");
    memobj.mapPtr_out = clEnqueueMapImage( memobj.CmdQue[0], memobj.outputImgMap, CL_TRUE, CL_MAP_READ, 
        imageOrigin, imageRegion,&imageRowPitch_in, NULL, 0, NULL, NULL, &status);
    CHECK_ERROR(status, "clEnqueueMapBuffer failed. (resultBuf)");
    
    ptr[0] = memobj.mapPtr_in;
    int argIdx = 0;
    status |= clSetKernelArg(memobj.imgKernel[0], argIdx++, sizeof(cl_mem), &memobj.outputImgMap);
    status |= clSetKernelArg(memobj.imgKernel[0], argIdx++, sizeof(cl_mem), &memobj.inputImgMap);
    status |= clSetKernelArg(memobj.imgKernel[0], argIdx++, sizeof(cl_mem), &memobj.affine_matrix);

    argIdx = 0;
    status |= clSetKernelArg(memobj.imgKernel[1], argIdx++, sizeof(cl_mem), &memobj.outputImgMap);
    status |= clSetKernelArg(memobj.imgKernel[1], argIdx++, sizeof(cl_mem), &memobj.inputImgMap);

    
    return status;
}

int init_cl_Affine(void * affinePtr[])
{
    init_cl_affine_obj(clpltobj, affineobj, affinePtr);
}



Mat cl_exc_affine(Mat input,  Mat matrix)
{

    double _rotate[2][2] = {
        {matrix.at<double>(0, 0), matrix.at<double>(0, 1)},
        {matrix.at<double>(1, 0), matrix.at<double>(1, 1)}
    };
    Mat rotate_mat = Mat(2, 2, CV_64FC1, _rotate); 
    Mat affine_parameter = rotate_mat.inv();
    float mat[2][3] = {
        {(float)affine_parameter.at<double>(0, 0), (float)affine_parameter.at<double>(0, 1),(float)(-matrix.at<double>(0, 2))},
        {(float)affine_parameter.at<double>(1, 0), (float)affine_parameter.at<double>(1, 1),(float)(-matrix.at<double>(1, 2))},
    };


    static size_t   globaworksize[2];
    globaworksize[0] = input.cols;
    globaworksize[1] = input.rows;
    static bool init = 1;

    static Mat output = Mat::zeros(input.size(), CV_8UC4);
    memcpy(affineobj.affine_matrix_Ptr, mat, sizeof(float)*6);

    cl_event ndrEvt;
    
    int status = clEnqueueNDRangeKernel(affineobj.CmdQue[0], affineobj.imgKernel[0], 2, NULL,
            globaworksize, NULL, 0, NULL, &ndrEvt);

    status = clEnqueueNDRangeKernel(affineobj.CmdQue[1], affineobj.imgKernel[1], 2, NULL,
            globaworksize, NULL, 0, NULL, &ndrEvt);

    output.data = (uchar *)affineobj.mapPtr_out;
    return output;
}





