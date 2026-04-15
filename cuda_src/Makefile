# Set these to match your test image dimensions.
IMG_WIDTH  ?= 321
IMG_HEIGHT ?= 481

all:
	nvcc -ccbin /usr/bin/clang++ -x cu blur.cpp -o blur.x -arch=sm_75 -std=c++17

run_blur: blur.x
	python3 deconstruct_image.py
	./blur.x $(IMG_WIDTH) $(IMG_HEIGHT)

clean:
	-rm -f *.x *.nsys-rep *.sqlite out/output_gray.txt out/0001_gray.txt
