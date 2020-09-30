CC=g++
NVCC=nvcc
CXXFLAGS= -fopenmp -O3 -Wextra -std=c++14
CUDAFLAGS= -std=c++11 -lineinfo -o
#LIBS= -lopenblas -lpthread -lcudart -lcublas
main.o: b1_multi.cu
	$(NVCC) b1_multi.cu $(CUDAFLAGS) main
all: main.o
clean:
	rm -rf *.o