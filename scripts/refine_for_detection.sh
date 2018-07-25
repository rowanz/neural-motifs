#!/usr/bin/env bash

# Refine Motifnet for detection


export CUDA_VISIBLE_DEVICES=$1

if [ $1 == "0" ]; then
     echo "TRAINING THE BASELINE"
    python models/train_rels.py -m sgdet -model motifnet -nl_obj 0 -nl_edge 0 -b 6 \
    -clip 5 -p 100 -pooling_dim 4096 -lr 1e-4 -ngpu 1 -ckpt checkpoints/baseline-sgcls/vgrel-11.tar  -save_dir checkpoints/baseline-sgdet \
    -nepoch 50 -use_bias
elif [ $1 == "1" ]; then
    echo "TRAINING STANFORD"
    python models/train_rels.py -m sgdet -model stanford -b 6 -p 100 -lr 1e-4 -ngpu 1 -clip 5 \
    -ckpt checkpoints/stanford-sgcls/vgrel-11.tar -save_dir checkpoints/stanford-sgdet
elif [ $1 == "2" ]; then
    echo "Refining Motifnet for detection!"
    python models/train_rels.py -m sgdet -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-4 -ngpu 1 -ckpt checkpoints/motifnet-sgcls/vgrel-7.tar \
        -save_dir checkpoints/motifnet-sgdet -nepoch 10 -use_bias
fi