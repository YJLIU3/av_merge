__kernel void imageRemap(__write_only image2d_t outImg,
                         __read_only image2d_t inImg, 
                         __global float* map_x, 
                         __global float* map_y,
                         __global char* gray)
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
    int4 data = read_imagei(inImg, sampler, coord1);

    gray[gidy * 260 + gidx] = (data.x*0.299f + data.y*0.587f+data.z*0.114f);
    
    write_imagei(outImg, coord, data);
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

__kernel void imageMergef(__read_only image2d_t inImg,
                            __write_only image2d_t outImg)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    int2 coord_src = (int2)(gidx, gidy);
    int2 coord_dst = (int2)(gidx + 30, gidy);
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                                    CLK_ADDRESS_CLAMP     |
                                    CLK_FILTER_NEAREST;

    int4 data = read_imagei(inImg, sampler, coord_src);

    write_imagei(outImg, coord_dst, data);
}
__kernel void imageMerger(__read_only image2d_t inImg,
                             __write_only image2d_t outImg)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    int2 coord_src = (int2)(gidx, gidy);
    int2 coord_dst = (int2)(gidx + 30, gidy + 460);
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                                    CLK_ADDRESS_CLAMP     |
                                    CLK_FILTER_NEAREST;

    int4 data = read_imagei(inImg, sampler, coord_src);
                            
    write_imagei(outImg, coord_dst, data);
}

