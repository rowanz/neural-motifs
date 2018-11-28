# neural-motifs

### Like this work, or scene understanding in general? You might be interested in checking out my brand new dataset VCR: Visual Commonsense Reasoning, at [visualcommonsense.com](https://visualcommonsense.com)!

This repository contains data and code for the paper [Neural Motifs: Scene Graph Parsing with Global Context (CVPR 2018)](https://arxiv.org/abs/1711.06640v2) For the project page (as well as links to the baseline checkpoints), check out [rowanzellers.com/neuralmotifs](https://rowanzellers.com/neuralmotifs). If the paper significantly inspires you, we request that you cite our work:

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
 ```conda install pytorch=0.3.0 torchvision=0.2.0 cuda90 -c pytorch```.
 
1. Update the config file with the dataset paths. Specifically:
    - Visual Genome (the VG_100K folder, image_data.json, VG-SGG.h5, and VG-SGG-dicts.json). See data/stanford_filtered/README.md for the steps I used to download these.
    - You'll also need to fix your PYTHONPATH: ```export PYTHONPATH=/home/rowan/code/scene-graph``` 

2. Compile everything. run ```make``` in the main directory: this compiles the Bilinear Interpolation operation for the RoIs as well as the Highway LSTM.

3. Pretrain VG detection. The old version involved pretraining COCO as well, but we got rid of that for simplicity. Run ./scripts/pretrain_detector.sh
Note: You might have to modify the learning rate and batch size, particularly if you don't have 3 Titan X GPUs (which is what I used). [You can also download the pretrained detector checkpoint here.](https://drive.google.com/open?id=11zKRr2OF5oclFL47kjFYBOxScotQzArX)

4. Train VG scene graph classification: run ./scripts/train_models_sgcls.sh 2 (will run on GPU 2). OR, download the MotifNet-cls checkpoint here: [Motifnet-SGCls/PredCls](https://drive.google.com/open?id=12qziGKYjFD3LAnoy4zDT3bcg5QLC0qN6).
5. Refine for detection: run ./scripts/refine_for_detection.sh 2 or download the [Motifnet-SGDet](https://drive.google.com/open?id=1thd_5uSamJQaXAPVGVOUZGAOfGCYZYmb) checkpoint.
6. Evaluate: Refer to the scripts ./scripts/eval_models_sg[cls/det].sh.

# help

Feel free to open an issue if you encounter trouble getting it to work!
