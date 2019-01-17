arm-linux-g++ test_ddr_gup.cpp cl_test.cpp -o test -I./ `pkg-config --cflags --libs opencv` -lOpenCL
