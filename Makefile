EXECUTABLE := main

CU_FILES   := cudaRenderer.cu

CU_DEPS    :=

CC_FILES   := main.cpp

all: $(EXECUTABLE) $(REFERENCE)

LOGS	   := logs

###########################################################

OBJDIR=objs
CXX=g++ -std=c++11
CXXFLAGS=-Wall `gdal-config --cflags` -Ofast -fopenmp
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart `gdal-config --libs` -lgdal
NVCC=nvcc
NVCCFLAGS=-O3 --gpu-architecture compute_35


OBJS=$(OBJDIR)/main.o  $(OBJDIR)/cudaRenderer.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE) $(LOGS)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
