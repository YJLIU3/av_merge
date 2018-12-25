#include <VX/vx.h>
#include <VX/vx_helper.h>
#include <VX/vx_ext_program.h>
#include <VX/vx_khr_cnn.h>
#include <stdio.h>
#include <math.h>
#include <malloc.h>



#define  VX_KERNEL_ENUM_RNN  100
//#define  VX_KERNEL_NAME_RNN  "com.vivantecorp.extension.rnn_uint8"
//#define  VX_KERNEL_NAME_RNN  "com.vivantecorp.extension.svdf_fp16"
#define  VX_KERNEL_NAME_RNN  "com.vivantecorp.extension.fft_64x64"
#define  VX_KERNEL_NAME_FFT_256_128  "com.vivantecorp.extension.fft_256x128"
#define  VX_KERNEL_NAME_IFFT_256_128  "com.vivantecorp.extension.ifft_256x128"
//#define  VX_KERNEL_NAME_RNN  "com.vivantecorp.extension.rnn_fp16"
//#define  VX_KERNEL_NAME_RNN  "com.vivantecorp.extension.rnn_fp16_row4"

#define USE_RNN_VXC    

vx_status VX_CALLBACK vxcRnnInitializer(vx_node nodObj, const vx_reference *paramObj, vx_uint32 paraNum);
vx_status VX_CALLBACK vxcRnnDeinitializer(vx_node nodObj, const vx_reference *paraObj, vx_uint32 paraNum);
vx_status VX_CALLBACK vxcRnnValidateInput(vx_node node, vx_uint32 index);
vx_status VX_CALLBACK vxcRnnValidateOutput(vx_node nodre, vx_uint32 index, vx_meta_format  meta);
vx_status VX_CALLBACK vxcRnnValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[]);
vx_status VX_CALLBACK vxcRnnInternalKernel(vx_node node, const vx_reference *parameters, vx_uint32 num);

vx_param_description_t basekernel_fullyconnected_params[] = {
    {VX_INPUT,  VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED},
    {VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED},
    {VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED}
    /*, {VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED}*/
};



vx_kernel_description_t internalkernel_vxcL2Pool = {
    VX_KERNEL_ENUM_RNN,
    VX_KERNEL_NAME_RNN,
#ifdef USE_RNN_VXC
    NULL,
#else
    vxcRnnInternalKernel,
#endif
    basekernel_fullyconnected_params,
    (sizeof(basekernel_fullyconnected_params)/sizeof(basekernel_fullyconnected_params[0])),
    vxcRnnValidator,
    NULL,
    NULL,
    vxcRnnInitializer,
    vxcRnnDeinitializer
};

vx_kernel_description_t internalkernel_fft_256x128 = {
    VX_KERNEL_ENUM_RNN,
    VX_KERNEL_NAME_FFT_256_128,
#ifdef USE_RNN_VXC
    NULL,
#else
    vxcRnnInternalKernel,
#endif
    basekernel_fullyconnected_params,
    (sizeof(basekernel_fullyconnected_params)/sizeof(basekernel_fullyconnected_params[0])),
    vxcRnnValidator,
    NULL,
    NULL,
    vxcRnnInitializer,
    vxcRnnDeinitializer
};

vx_kernel_description_t internalkernel_ifft_256x128 = {
    VX_KERNEL_ENUM_RNN,
    VX_KERNEL_NAME_IFFT_256_128,
#ifdef USE_RNN_VXC
    NULL,
#else
    vxcRnnInternalKernel,
#endif
    basekernel_fullyconnected_params,
    (sizeof(basekernel_fullyconnected_params)/sizeof(basekernel_fullyconnected_params[0])),
    vxcRnnValidator,
    NULL,
    NULL,
    vxcRnnInitializer,
    vxcRnnDeinitializer
};

static vx_kernel_description_t* kernels[] = 
{
    &internalkernel_vxcL2Pool,
    &internalkernel_fft_256x128,
    &internalkernel_ifft_256x128
};



vx_status VX_CALLBACK vxcRnnValidateInput(vx_node node, vx_uint32 index)
{
    return VX_SUCCESS;
}

vx_status VX_CALLBACK vxcRnnValidateOutput(vx_node node, vx_uint32 index, vx_meta_format  metas)
{
    return VX_SUCCESS;
}  

vx_status VX_CALLBACK vxcRnnValidator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    vx_status status = VX_SUCCESS;
    vx_uint32 index = 0;
    for(index = 0; index < num; index++)
    {
        if(index == 0)
        {
            status |= vxcRnnValidateInput(node,index);
        }
#if 1
        else if(index == 1) //tensor
        {
            vx_array array = (vx_array)parameters[index];
            if(array != NULL)
            {
                // VX_ARRAY_ITEMTYPE
                vx_enum type = 0;
                vx_size capacity = 0;
                vx_size itemNum = 0;


                {
                    vx_enum item_type = 0;
                    vx_size capacity = 0;
                    vxQueryArray(array, VX_ARRAY_ITEMTYPE, &item_type, sizeof(vx_enum));
                    vxQueryArray(array, VX_ARRAY_CAPACITY, &capacity, sizeof(vx_size));


                    status = vxSetMetaFormatAttribute(metas[index], VX_ARRAY_ITEMTYPE, &item_type, sizeof(item_type));
                    status |= vxSetMetaFormatAttribute(metas[index], VX_ARRAY_CAPACITY, &capacity, sizeof(capacity));
                    vxSetArrayAttribute(array, VX_ARRAY_NUMITEMS, &capacity, sizeof(capacity));
                    //vxReleaseArray(&array);
                }

                status |= vxQueryArray(array, VX_ARRAY_NUMITEMS, &itemNum, sizeof(itemNum));

                status |= vxQueryArray(array, VX_ARRAY_ITEMTYPE, &type, sizeof(type));
                //if (type != /*VX_TYPE_INT32*/VX_TYPE_FLOAT32)                        
                //if (type != VX_TYPE_INT16)
                if (type != VX_TYPE_FLOAT32)
                    status |= VX_ERROR_INVALID_TYPE;

                // VX_ARRAY_CAPACITY
                status |= vxQueryArray(array, VX_ARRAY_CAPACITY, &capacity, sizeof(capacity));
                if (capacity == 0)
                    status |= VX_ERROR_INVALID_TYPE;

            }
            else
            {
                printf("[%s : %d] Warning parameter %d is NULL \n", __FILE__,__LINE__,index);
            }    
        }
#endif
        else
        {
            status |= vxcRnnValidateOutput(node, index, metas[index]);
        }

    }
    return status;
}

vx_status VX_CALLBACK vxcRnnInitializer(vx_node nodObj, const vx_reference *paramObj, vx_uint32 paraNum)
{
    vx_status status = VX_SUCCESS;

#ifdef USE_RNN_VXC
#define gcmALIGN(n, align) ((n) + ((align) - 1)) & ~((align) - 1)
    vx_kernel_execution_parameters_t shaderParam = {
        2,          // workdim
        {0, 0, 0},  // globalWorkOffset: control the start location be processed in the image
        {0, 0, 0},  // globalWorkScale: how many pixels could be processed by a single thread
        {0, 0, 0},  // localWorkSize: local group size in threads
        {0, 0, 0}}; // globalWorkSize: image size in threads
    vx_scalar     scalar[2];
    uint32_t width, height;
    scalar[0]            = (vx_scalar)paramObj[3];
    scalar[1]            = (vx_scalar)paramObj[5];

    status = vxCopyScalar(scalar[0], &width, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    status |= vxCopyScalar(scalar[1], &height, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    int localWorkSize[2]   = {16, 1};     // 256x128

    if(width == 64 && height == 64)
        localWorkSize[0] = 64;

    shaderParam.globalWorkOffset[0] = 0;
    shaderParam.globalWorkOffset[1] = 0;
    shaderParam.globalWorkScale[0]    = 1;
    shaderParam.globalWorkScale[1]    = 1;

    /*if(srcFormat == VX_TYPE_FLOAT32 || srcFormat == VX_TYPE_FLOAT16)
    shaderParam.globalWorkScale[0] = 4;*/
    shaderParam.localWorkSize[0]    = localWorkSize[0];
    shaderParam.localWorkSize[1]    = 1;
    shaderParam.globalWorkSize[0]    = localWorkSize[0];
    shaderParam.globalWorkSize[1]    = 1;


    status = vxSetNodeAttribute(nodObj, VX_NODE_ATTRIBUTE_KERNEL_EXECUTION_PARAMETERS, &shaderParam, sizeof(vx_kernel_execution_parameters_t));

    if(status != VX_SUCCESS)
        printf("initialize error.\n");
#endif

    return VX_SUCCESS;
}

vx_status VX_CALLBACK vxcRnnDeinitializer(vx_node nodObj, const vx_reference *paraObj, vx_uint32 paraNum)
{
    return VX_SUCCESS;
}

#ifdef USE_RNN_VXC
vx_char *programSource = NULL;
static vx_char* loadSources(const vx_char *filename, vx_size *programSize)
{
    FILE *pFile = NULL;


    pFile = fopen(filename, "rb");
    if (pFile != NULL && programSize)
    {
        vx_int32 size = 0;
        /* obtain file size:*/
        fseek(pFile, 0, SEEK_END);
        *programSize = ftell(pFile);
        rewind(pFile);

        size = (int)(*programSize + 1);
        programSource = (char*)malloc(sizeof(char)*(size));
        if (programSource == NULL)
        {
            fclose(pFile);
            free(programSource);
            return NULL;
        }

        fread(programSource, sizeof(char), *programSize, pFile);
        programSource[*programSize] = '\0';
        fclose(pFile);
    }
    else
    {
        printf("[%s line %d] Open %s failed.\n", __FILE__, __LINE__, filename);
    }

    return programSource;
}
#endif

extern "C" vx_status VX_API_CALL vxPublishKernels(vx_context ContextVX)
{
    vx_status status = VX_FAILURE;
    vx_kernel kernelObj = NULL;

#ifdef USE_RNN_VXC
    vx_program programObj = NULL;
    const vx_char* programSrc[1];
    vx_size programLen = 0;
    programSrc[0] = loadSources("fft_kernel.cpp",&programLen) ;

    programObj = vxCreateProgramWithSource(ContextVX, 1, programSrc, &programLen);
    status = vxBuildProgram(programObj, "-cl-viv-vx-extension");
    if(status < 0)
        printf("vxBuildProgram fail\n");

    for(int i=0; i<(sizeof(kernels)/sizeof(kernels[0])); i++)
    {
        kernelObj = vxAddKernelInProgram(programObj,
            kernels[i]->name,
            kernels[i]->enumeration,
            kernels[i]->numParams,
            kernels[i]->validate,
            kernels[i]->initialize,
            kernels[i]->deinitialize
            );
        if(kernelObj)
        {
            status = VX_SUCCESS; 
            for(int j=0; j < kernels[i]->numParams; j++)
            {
                status = vxAddParameterToKernel(kernelObj, 
                    j,
                    kernels[i]->parameters[j].direction,
                    kernels[i]->parameters[j].data_type,
                    kernels[i]->parameters[j].state
                    );
                if(status!=VX_SUCCESS)
                {
                    printf("Failed to add parameter %d to kernel %s.\n", j, kernels[i]->name);
                    break;
                }
            }

            if(VX_SUCCESS==status)
            {
                status = vxFinalizeKernel(kernelObj);
                if (status!=VX_SUCCESS)
                {
                    printf("Failed to finalize kernel[%u]=%s\n",i, kernels[i]->name);

                    if(VX_SUCCESS != vxRemoveKernel(kernelObj))
                        printf("Failed to remove kernel[%u]=%s\n",i, kernels[i]->name);
                }
            }
            else
            {
                if(VX_SUCCESS != vxRemoveKernel(kernelObj))
                    printf("Failed to remove kernel[%u]=%s\n",i, kernels[i]->name);
            }
        }
        else
        {
            printf("Failed to add kernel %s\n", kernels[i]->name);
        }
    }
#else
    for(int i=0; i<(sizeof(kernels)/sizeof(kernels[0])); i++)  
    {
        kernelObj = vxAddUserKernel(ContextVX,
            kernels[i]->name,
            kernels[i]->enumeration,
            kernels[i]->function,
            kernels[i]->numParams,
            kernels[i]->validate,
            kernels[i]->initialize,
            kernels[i]->deinitialize
            );
        if(kernelObj)
        {
            status = VX_SUCCESS; 
            for(int j=0; j<kernels[i]->numParams; j++)
            {
                status = vxAddParameterToKernel(kernelObj, 
                    j,
                    kernels[i]->parameters[j].direction,
                    kernels[i]->parameters[j].data_type,
                    kernels[i]->parameters[j].state
                    );
                if(status!=VX_SUCCESS)
                {
                    printf("Failed to add parameter %d to kernel %s.\n", j, kernels[i]->name);
                    break;
                }
            }

            if(VX_SUCCESS==status)
            {
                status = vxFinalizeKernel(kernelObj);
                if (status!=VX_SUCCESS)
                {
                    printf("Failed to finalize kernel[%u]=%s\n",i, kernels[i]->name);

                    if(VX_SUCCESS != vxRemoveKernel(kernelObj))
                        printf("Failed to remove kernel[%u]=%s\n",i, kernels[i]->name);
                }
            }
            else
            {
                if(VX_SUCCESS != vxRemoveKernel(kernelObj))
                    printf("Failed to remove kernel[%u]=%s\n",i, kernels[i]->name);
            }
        }
        else
        {
            printf("Failed to add kernel %s\n", kernels[i]->name);
        }
    }
#endif
    return status;
}  


#ifndef USE_RNN_VXC    

vx_float32 Uint8toFp32(vx_uint8 val, vx_int32 zeroPoint, vx_float32 scale)
{
    vx_float32 result = 0.0f;

    result = (val - (vx_uint8)zeroPoint) * scale;

    return result;
}


vx_float32 Fp16toFp32(const vx_uint16 in)
{
    vx_int32 t1;
    vx_int32 t2;
    vx_int32 t3;
    vx_float32 out;

    t1 = in & 0x7fff;                       // Non-sign bits
    t2 = in & 0x8000;                       // Sign bit
    t3 = in & 0x7c00;                       // Exponent

    t1 <<= 13;                              // Align mantissa on MSB
    t2 <<= 16;                              // Shift sign bit into position

    t1 += 0x38000000;                       // Adjust bias

    t1 = (t3 == 0 ? 0 : t1);                // Denormals-as-zero

    t1 |= t2;                               // Re-insert sign bit

    *((uint32_t*)&out) = t1;

    return out;
}

vx_uint16 Fp32toFp16(vx_float32 val)
{
#define F16_EXPONENT_BITS 0x1F
#define F16_EXPONENT_SHIFT 10
#define F16_EXPONENT_BIAS 15
#define F16_MANTISSA_BITS 0x3ff
#define F16_MANTISSA_SHIFT (23 - F16_EXPONENT_SHIFT)
#define F16_MAX_EXPONENT (F16_EXPONENT_BITS << F16_EXPONENT_SHIFT)
    vx_uint32 f32 = (*(vx_uint32 *) &val);
    vx_int16 f16 = 0;
    /* Decode IEEE 754 little-endian 32-bit floating-point value */
    int sign = (f32 >> 16) & 0x8000;
    /* Map exponent to the range [-127,128] */
    int exponent = ((f32 >> 23) & 0xff) - 127;
    int mantissa = f32 & 0x007fffff;
    if (exponent == 128) 
    { /* Infinity or NaN */
        f16 = (vx_int16)(sign | F16_MAX_EXPONENT);
        if (mantissa) f16 |= (mantissa & F16_MANTISSA_BITS);

    } 
    else if (exponent > 15) 
    { /* Overflow - flush to Infinity */
        f16 = (vx_int16)(sign | F16_MAX_EXPONENT);
    } 
    else if (exponent > -15) 
    { /* Representable value */
        exponent += F16_EXPONENT_BIAS;
        mantissa >>= F16_MANTISSA_SHIFT;
        f16 = (vx_int16)(sign | exponent << F16_EXPONENT_SHIFT | mantissa);
    }
    else 
    {
        f16 = (vx_int16)sign;
    }
    return f16;
}

static vx_int32 rint_c(vx_float32 val)
{
    vx_float32 r0x,r0y,r0z;
    vx_int32 sign;
    r0y = fabsf(val) + 0.5f;
    r0y = floor(r0y);
    r0z = floor(fabsf(val));
    r0x = fabsf(val) - r0z;
    if(r0x == 0.5)
    {
        vx_int32 R0x = r0y;
        R0x = R0x & 1;
        r0y = r0y - R0x;
        sign = val > 0 ? 1 : (val < 0 ? -1 : 0);
        r0y = r0y * sign;
        return (vx_int32)r0y;
    }
    else
    {
        sign = val > 0 ? 1 : (val < 0 ? -1 : 0);
        r0y = r0y * sign;
        return (vx_int32)r0y;
    }
}

static vx_uint32 vxcGetTypeSize(vx_enum format)
{
    switch(format)
    {
    case VX_TYPE_INT8:
    case VX_TYPE_UINT8:
        return 1;
    case VX_TYPE_INT16:
    case VX_TYPE_UINT16:
        return 2;
    case VX_TYPE_INT32:
    case VX_TYPE_UINT32:
        return 4;
    case VX_TYPE_INT64:
    case VX_TYPE_UINT64:
        return 8;
    case VX_TYPE_FLOAT32:
        return 4;
    case VX_TYPE_FLOAT64:
        return 8;
    case VX_TYPE_ENUM:
        return 4;
    case VX_TYPE_FLOAT16:
        return 2;
    }

    return 4;
}

vx_status tensorRead(vx_context context, vx_tensor ts, void *buf)
{
    vx_uint32  ts_size[4];
    vx_uint32 input_stride_size[4];
    vx_uint32 output_num,num_of_dim;
    vx_tensor_addressing input_user_addr = NULL;
    vx_status status = VX_FAILURE;
    vx_uint32 i = 0;
    void *dataBuf = (vx_uint16 *)buf;
    vx_uint32 dataFormat;

    status = vxQueryTensor(ts, VX_TENSOR_NUM_OF_DIMS, &num_of_dim, sizeof(num_of_dim));
    status |= vxQueryTensor(ts, VX_TENSOR_DIMS, ts_size, sizeof(ts_size));
    status |= vxQueryTensor(ts, VX_TENSOR_DATA_TYPE, &dataFormat, sizeof(dataFormat));

    input_stride_size[0] = vxcGetTypeSize(dataFormat);
    output_num = ts_size[0];
    for (i=1; i< num_of_dim; i++)
    {
        input_stride_size[i] = input_stride_size[i-1] * ts_size[i];
        output_num *= ts_size[i];
    }

    if(dataBuf == NULL)
    {
        printf("TensorRead fail! input empty \n");
        return VX_FAILURE;
    }

    input_user_addr = vxCreateTensorAddressing(
        context,
        ts_size,
        input_stride_size,
        num_of_dim
        );
    status = vxCopyTensorPatch(
        ts,
        NULL,
        input_user_addr,
        dataBuf,
        VX_READ_ONLY,
        0
        );
    vxReleaseTensorAddressing(&input_user_addr);
    if(status < 0)
    {
        free(dataBuf);
        dataBuf = NULL;
        printf("TensorRead fail! status = %d\n",status);
        return status;
    }

    return VX_SUCCESS;

}

#define gcmMAX(x, y)            (((x) >= (y)) ?  (x) :  (y))
void SoftmaxFunc(vx_int8* pIn, vx_float32* pOut, vx_uint32 width, vx_uint32 height, float scaleIn)
{
    vx_uint16 dst = 0;
    vx_float32 fMax = 0.0f;
    vx_float32 *pfProbFP32 = (vx_float32 *)malloc(width * sizeof(vx_float32)); 

    vx_uint32 singleItemCount = width;

    for (vx_uint32 i = 0; i < singleItemCount; i ++)
    {
        fMax = gcmMAX(fMax, pIn[i] * scaleIn);
    }


    float fProbSum = 0.0f;
    for (vx_uint32 i = 0; i < singleItemCount; i++)
    {
        vx_uint16 tmp0 = Fp32toFp16(scaleIn);
        vx_uint16 tmp1 = Fp32toFp16(fMax / scaleIn);
        vx_uint16 tmp2 = Fp32toFp16(pIn[i] * scaleIn - fMax);
        vx_float32 tmp3 = pIn[i] * scaleIn - fMax;
        vx_uint32 tmp4 = *(vx_uint32 *)&tmp3;
        pfProbFP32[i] = expf(pIn[i] * scaleIn - fMax);

        fProbSum += pfProbFP32[i];
    }

    for (vx_uint32 i = 0; i < singleItemCount; i++)
    {
        pOut[i] = pfProbFP32[i] / fProbSum;
    }

}

vx_status VX_CALLBACK vxcRnnInternalKernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    vx_status    status        = VX_SUCCESS;
    vx_context    context        = vxGetContext((vx_reference)node);
    vx_tensor    input        = (vx_tensor)parameters[0];
    vx_tensor    output        = (vx_tensor)parameters[1];

    void * inputBuf            = NULL;
    void * outputBuf        = NULL;
    vx_uint32 size[4]        = {0};
    vx_uint32 num_of_dims    = 0;
    vx_enum src_format;
    vx_enum dst_format;
    vx_tensor_create_params_t src_tensor_create_params;
    vx_tensor_create_params_t dst_tensor_create_params;

    status = vxQueryTensor(input, VX_TENSOR_DATA_TYPE, &src_format, sizeof(src_format));
    status |= vxQueryTensor(output, VX_TENSOR_DATA_TYPE, &dst_format, sizeof(dst_format));
    status |= vxQueryTensor(input, VX_TENSOR_NUM_OF_DIMS, &num_of_dims, sizeof(num_of_dims));
    status |= vxQueryTensor(input, VX_TENSOR_DIMS, size, sizeof(size[0]));


    tensorRead(context, input, inputBuf);


    //SoftmaxFunc(inputBuf, outputBuf, size, num_of_dims, src_format, dst_format);



    return status;
}    

#endif