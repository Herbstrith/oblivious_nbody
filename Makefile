MKL =1


#initial definitions (library paths et al.)
CUDA_PATH=/usr/local/cuda-7+0
BUILD_DIR=./

####################
#includes
#################### 
#Cuda includes
CUDA_INCLUDE_DIR=-I. -I$(CUDA_PATH)/include 




####################
#library search paths
####################
CUDA_LIB_DIR=-L$(CUDA_PATH)/lib64 



####################
#libraries
####################
CUDALIBS=-lcudadevrt -lcudart 

#utilS=  -lpthread  -lm 


####################
#other compilation flags
####################
CFLAGS= -Wwrite-strings
CUDAFLAGS= -arch=sm_35 
#LINKERFLAGS= -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/../compiler/lib/intel64/libiomp5.a -Wl,--end-group

nbody_kernel.o: src/nbody_kernel.cu
	nvcc $(CUDAFLAGS) $(CUDA_INCLUDE_DIR) -c -O3 -dc src/nbody_kernel.cu -o $(BUILD_DIR)/nbody_kernel.o

main.o : src/main.cpp
	gcc $(CFLAGS) -c -O3 src/main.cpp -o $(BUILD_DIR)/main.o -L/usr/lib64 -lstdc++

Nbody: main.o nbody_kernel.o
	nvcc $(CUDAFLAGS) $(CUDA_INCLUDE_DIR) -dlink $(BUILD_DIR)/main.o $(BUILD_DIR)/nbody_kernel.o -o $(BUILD_DIR)/link.o
	gcc $(CFLAGS) $(BUILD_DIR)/main.o $(BUILD_DIR)/nbody_kernel.o $(BUILD_DIR)/link.o -lcudadevrt $(CUDA_LIB_DIR)  $(CUDALIBS) -o Nbody -lcudart -L/usr/lib64 -lstdc++

#compiling on lesser than 3.5 cuda capables gpu's

nbody_kernel2.o: src/nbody_kernel.cu
	nvcc -arch=sm_30  $(CUDA_INCLUDE_DIR) -c -O3  src/nbody_kernel.cu -o $(BUILD_DIR)/nbody_kernel2.o

Nbody2: main.o nbody_kernel2.o
	nvcc -arch=sm_30 $(CUDA_INCLUDE_DIR) -dlink $(BUILD_DIR)/main.o $(BUILD_DIR)/nbody_kernel2.o  -o $(BUILD_DIR)/link.o
	gcc $(CFLAGS) $(BUILD_DIR)/main.o $(BUILD_DIR)/nbody_kernel2.o $(BUILD_DIR)/link.o  $(CUDA_LIB_DIR)  $(CUDALIBS) -o Nbody -lcudart -L/usr/lib64 -lstdc++


####################
#misc
####################
clean:
	rm -rf $(BUILD_DIR)/*.o ./Nbody
