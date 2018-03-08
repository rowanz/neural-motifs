export PATH := /usr/local/cuda-9.1/bin:$(PATH)

all: draw_rectangles box_intersections nms roi_align lstm

draw_rectangles:
	cd lib/draw_rectangles; python setup.py build_ext --inplace
box_intersections:
	cd lib/fpn/box_intersections_cpu; python setup.py build_ext --inplace
nms:
	cd lib/fpn/nms; make
roi_align:
	cd lib/fpn/roi_align; make
lstm:
	cd lib/lstm/highway_lstm_cuda; ./make.sh