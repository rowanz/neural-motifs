#!/usr/bin/env bash

# Train Motifnet using different orderings

export CUDA_VISIBLE_DEVICES=$1

if [ $1 == "0" ]; then
    echo "TRAINING MOTIFNET V1"
    python models/train_rels.py -m sgcls -model motifnet -order size -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/vgdet/vg-24.tar \
        -save_dir checkpoints/motifnet-size-sgcls -nepoch 50 -use_bias
elif [ $1 == "1" ]; then
    echo "TRAINING MOTIFNET V2"
    python models/train_rels.py -m sgcls -model motifnet -order random -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/vgdet/vg-24.tar \
        -save_dir checkpoints/motifnet-random-sgcls -nepoch 50 -use_bias
elif [ $1 == "2" ]; then
    echo "TRAINING MOTIFNET V3"
    python models/train_rels.py -m sgcls -model motifnet -order confidence -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -ckpt checkpoints/vgdet/vg-24.tar \
        -save_dir checkpoints/motifnet-conf-sgcls -nepoch 50 -use_bias
fi