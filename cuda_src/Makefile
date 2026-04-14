all:
	nvcc -ccbin /usr/bin/clang++ -x cu blur.cpp -Xcompiler -fopenmp -o blur.x -arch=sm_75 -std=c++11

run_blur:
	./blur.x 20

clean:
	-rm -f *.x *.nsys-rep *.sqlite