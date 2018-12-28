__kernel void imageRemap(__write_only image2d_t outImg,
                         __read_only image2d_t inImg, 
                         __global float* map_x, 
                         __global float* map_y)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    
    int2 coord = (int2)(gidx, gidy);

    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                                    CLK_ADDRESS_CLAMP     |
                                    CLK_FILTER_NEAREST;
    float x_loc = map_x[gidy*260 + gidx];
    float y_loc = map_y[gidy*260 + gidx];
    
    float2 coord1 = (float2)(x_loc, y_loc);

    //int offset = gidy * width * channel;
    float4 data = read_imagef(inImg, sampler, coord1);
    
    write_imagef(outImg, coord, data);
}

  __kernel void imageAffine(__write_only image2d_t outImg,
                                __read_only image2d_t inImg, 
                                __global float* matrix)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
                             
    int2 coord = (int2)(gidx, gidy);
                         
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP     |
                              CLK_FILTER_NEAREST;
    float a = (float)matrix[0];
    float b = (float)matrix[1];
    float c = (float)matrix[2];
    float d = (float)matrix[3];
    float e = (float)matrix[4];
    float f = (float)matrix[5];
    
    float x_loc = a*gidx + b*gidy + c;
    float y_loc = d*gidx + e*gidy + f;
    
    float2 coord1 = (float2)(x_loc, y_loc);
                         
    //int offset = gidy * width * channel;
    float4 data = read_imagef(inImg, sampler, coord1);
                             
    write_imagef(outImg, coord, data);
}

__kernel void imageCopy(__read_only image2d_t outImg,
                           __write_only image2d_t inImg)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
                                                             
    int2 coord = (int2)(gidx, gidy);
                                                         
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP     |
                              CLK_FILTER_NEAREST;
                                                         
                                    //int offset = gidy * width * channel;
    int4 data = read_imagei(outImg, sampler, coord);
                                                             
    write_imagei(inImg, coord, data);
}

                         
