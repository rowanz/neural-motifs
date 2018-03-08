# neural-motifs
Code for Neural Motifs: Scene Graph Parsing with Global Context (CVPR 2018)

This repository contains data and code for the paper [Neural Motifs: Scene Graph Parsing with Global Context](https://arxiv.org/abs/1711.06640v1). NOTE: this ArXiv link is not the camera ready version - I'm working on that now! There are a couple minor details I'm working on changing, but they're not that important, so feel free to read it now. If the paper significantly inspires you, we request that you cite our work:

### Bibtex

```
@inproceedings{zellers2018scenegraphs,
  title={Neural Motifs: Scene Graph Parsing with Global Context},
  author={Zellers, Rowan and Yatskar, Mark and Thomson, Sam and Choi, Yejin},
  booktitle = "Conference on Computer Vision and Pattern Recognition",  
  year={2018}
}
```
# Setup


0. Install python3.6 and pytorch 3. I recommend the [Anaconda distribution](https://repo.continuum.io/archive/). To install PyTorch if you haven't already, use
 ```conda install pytorch torchvision cuda90 -c pytorch```.
1. Update the config file with the dataset paths. Specifically:
    - Visual Genome (the VG_100K folder, image_data.json, VG-SGG.h5, and VG-SGG-dicts.json). See data/stanford_filtered/README.md for the steps I used to download these.
    - You'll also need to fix your PYTHONPATH: ```export PYTHONPATH=/home/rowan/code/scene-graph``` 

2. Compile everything. run ```make``` in the main directory: this compiles the Bilinear Interpolation operation for the RoIs as well as the Highway LSTM.

3. Pretrain VG detection. The old version involved pretraining COCO as well, but we got rid of that! Run ./scripts/pretrain_detector.sh

4. Train VG scene graph classification: run ./scripts/train_models_sgcls.sh 2 (will run on GPU 2)

5. Refine for detection: run ./scripts/refine_for_detection.sh 2

# help

Feel free to ping me if you encounter trouble getting it to work!