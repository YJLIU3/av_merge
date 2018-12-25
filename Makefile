cc = arm-linux-g++ -I./ -w -O3 -std=c++11 `pkg-config --cflags --libs opencv`
DEP_INC_DIR =$(addprefix  -I, $(shell find $(ROOTDIR)inc -type d))
DEP_LIB_DIR =$(addprefix  -L, $(shell find $(ROOTDIR)lib -type d))

LIBS = -L$(DEP_LIB_DIR) -L/opt/sgks/arm-linux/arm-buildroot-linux-gnueabi/sysroot/usr/lib  -lpthread -lstdc++ -mcpu=cortex-a7 -mtune=cortex-a7 -mfpu=vfpv4-d16 -mfloat-abi=softfp -lz -lrt -lm -ldl -lCLC -lLLVM_viv -lOpenVX -lOpenVXU -lVSC -lGAL -fopenmp
CFLAGS = -w -O3 -g -pipe -Wl,-rpath=. -mcpu=cortex-a7 -mtune=cortex-a7 -mfpu=vfpv4-d16 -mfloat-abi=softfp -D__LINUX__ -D__cplusplus -DUSE_GETOUTLINES_VXC -DUSE_GETLINES_VXC -DUSE_NONEZEROPOS_VXC -DLINUX
OPT = -O3
objects = DuadPanorama.o panorama.o fftm.o cv_vx.o
out_obj = DuadPanorama.o panorama.o fftm.o cv_vx.o demo.o
edit:$(out_obj)
	$(cc)  $(LIBS) $(out_obj) -o edit

DuadPanorama.o: DuadPanorama.cpp panorama.h parameter.h 
	$(cc) -c  DuadPanorama.cpp 
panorama.o: panorama.cpp panorama.h parameter.h 
	$(cc) -c panorama.cpp
fftm.o: fftm.cpp fftm.hpp
	$(cc) -c fftm.cpp
cv_vx.o: cv_vx.cpp cv_vx.h
	$(cc) -c cv_vx.cpp
demo.o: demo.cpp
	$(cc) -c demo.cpp
	
	arm-linux-ar rcs libav_merge.a  $(objects)
	$(cc) DuadPanorama.cpp panorama.cpp fftm.cpp cv_vx.cpp -fPIC -shared -o libav_merge.so
#	cp edit ../NfsRoot/target/home/sgks_green_board/
clean:
	rm -rf *.o *.a *.so
