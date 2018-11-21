cc = arm-linux-g++ -I./ -w -O3 -std=c++11 `pkg-config --cflags --libs opencv`
DEP_INC_DIR =$(addprefix  -I, $(shell find $(ROOTDIR)inc -type d))
DEP_LIB_DIR =$(addprefix  -L, $(shell find $(ROOTDIR)lib -type d))

LIBS = -L$(DEP_LIB_DIR) -L/opt/sgks/arm-linux/arm-buildroot-linux-gnueabi/sysroot/usr/lib  -lpthread -lstdc++ -mcpu=cortex-a7 -mtune=cortex-a7 -mfpu=vfpv4-d16 -mfloat-abi=softfp -lz -lrt -lm -ldl -lCLC -lLLVM_viv -lOpenVX -lOpenVXU -lVSC -lGAL -fopenmp
CFLAGS = -w -O3 -g -pipe -Wl,-rpath=. -mcpu=cortex-a7 -mtune=cortex-a7 -mfpu=vfpv4-d16 -mfloat-abi=softfp -D__LINUX__ -D__cplusplus -DUSE_GETOUTLINES_VXC -DUSE_GETLINES_VXC -DUSE_NONEZEROPOS_VXC -DLINUX
OPT = -O3
objects = DuadPanorama.o panorama.o fftm.o

edit:$(objects)
DuadPanorama.o: DuadPanorama.cpp panorama.h parameter.h 
	$(cc) -c  DuadPanorama.cpp 
panorama.o: panorama.cpp panorama.h parameter.h 
	$(cc) -c panorama.cpp
fftm.o: fftm.cpp fftm.hpp
	$(cc) -c fftm.cpp

#gpu2d.o: gpu2d.cpp gpu2d.h 
#	$(cc) $(CFLAGS) -c $(DEP_INC_DIR) gpu2d.cpp
	
#render.o: render.cpp galUtil.h ringbuffer.hpp gpu2d.h
#	$(cc) $(CFLAGS) -c $(DEP_INC_DIR) render.cpp

	arm-linux-ar rcs libav_merge.a  $(objects)

clean:
	rm -rf *.o *.a
