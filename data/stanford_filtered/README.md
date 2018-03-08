# Filtered data
Adapted from [Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md).

Follow the folling steps to get the dataset set up.
1. Download the VG images [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to a file and link to them in `config.py` (eg. currently I have `VG_IMAGES=data/visual_genome/VG_100K`). 
2. Download the [VG metadata](http://cvgl.stanford.edu/scene-graph/VG/image_data.json). I recommend extracting it to this directory (e.g. `data/stanford_filtered/image_data.json`), or you can edit the path in `config.py`.
3. Download the [scene graphs](http://cvgl.stanford.edu/scene-graph/dataset/VG-SGG.h5) and extract them to `data/stanford_filtered/VG-SGG.h5`
4. Download the [scene graph dataset metadata](http://cvgl.stanford.edu/scene-graph/dataset/VG-SGG-dicts.json) and extract it to `data/stanford_filtered/VG-SGG-dicts.json`
