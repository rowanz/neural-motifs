---
permalink: /
title: Neural Motifs
author: Rowan Zellers
description: Scene Graph Parsing with Global Context (CVPR 2018)
google_analytics_id: UA-84290243-3
---
# Neural Motifs: Scene Graph Parsing with Global Context (CVPR 2018)

### by [Rowan Zellers](https://rowanzellers.com), [Mark Yatskar](https://homes.cs.washington.edu/~my89/), [Sam Thomson](https://http://samthomson.com/), [Yejin Choi](https://homes.cs.washington.edu/~yejin/)


{% include image.html url="teaser.png" description="teaser" %} 

# Overview

* In this work, we investigate the problem of producing structured graph representations of visual scenes. Similar to object detection, we must predict a box around each object. Here, we also need to predict an edge (with one of several labels, possibly `background`) between every ordered pair of boxes, producing a directed graph where the edges hopefully represent the semantics and interactions present in the scene.
* We present an analysis of the [Visual Genome Scene Graphs dataset](http://visualgenome.org/). In particular:
    * Object labels (e.g. person, shirt) are highly predictive of edge labels (e.g. wearing), but **not vice versa**.
    * Over 90% of the edges in the dataset are non-semantic.
    * There is a significant amount of structure in the dataset, in the form of graph motifs (regularly appearing substructures). 
* Motivated by our analysis, we present a simple baseline that outperforms previous approaches.
* We introduce Stacked Motif Networks (MotifNet), which is a novel architecture that is designed to capture higher order motifs in scene graphs. In doing so, it achieves a sizeable performance gain over prior state-of-the-art.

# Read the paper!
The old version of the paper is available at [arxiv link](https://arxiv.org/abs/1711.06640) - camera ready version coming soon!

# Bibtex
```
@inproceedings{zellers2018scenegraphs,
  title={Neural Motifs: Scene Graph Parsing with Global Context},
  author={Zellers, Rowan and Yatskar, Mark and Thomson, Sam and Choi, Yejin},
  booktitle = "Conference on Computer Vision and Pattern Recognition",  
  year={2018}
}
```

# View some examples!

Check out [this tool](https://rowanzellers.com/scenegraph2/) I made to visualize the scene graph predictions. Disclaimer: the predictions are from an earlier version of the model, but hopefully they're still helpful!

# Code

Visit the [`neural-motifs` GitHub repository](https://github.com/rowanz/neural-motifs) for our reference implementation and instructions for running our code.

It is released under the MIT license.

# Checkpoints available for download
* [Pretrained Detector](https://drive.google.com/open?id=11zKRr2OF5oclFL47kjFYBOxScotQzArX)
* [Motifnet-SGDet](https://drive.google.com/open?id=1thd_5uSamJQaXAPVGVOUZGAOfGCYZYmb)
* [Motifnet-SGCls/PredCls](https://drive.google.com/open?id=12qziGKYjFD3LAnoy4zDT3bcg5QLC0qN6)

# questions?

Feel free to get in touch! My main website is at [rowanzellers.com](https://rowanzellers.com)
