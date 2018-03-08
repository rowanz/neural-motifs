#!/usr/bin/env bash

cd anchors
python setup.py build_ext --inplace
cd ..

cd box_intersections_cpu
python setup.py build_ext --inplace
cd ..

cd cpu_nms
python build.py
cd ..

cd roi_align
python build.py -C src/cuda clean
python build.py -C src/cuda clean
cd ..

echo "Done compiling hopefully"
