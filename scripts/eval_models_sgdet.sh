#!/usr/bin/env bash

# This is a script that will evaluate all the models for SGDET
export CUDA_VISIBLE_DEVICES=$1

if [ $1 == "0" ]; then
    echo "EVALING THE BASELINE"
    python models/eval_rels.py -m sgdet -model motifnet -nl_obj 0 -nl_edge 0 -b 6 \
    -clip 5 -p 100 -pooling_dim 4096 -ngpu 1 -ckpt checkpoints/baseline-sgdet/vgrel-17.tar \
    -nepoch 50 -use_bias -cache baseline_sgdet.pkl -test
elif [ $1 == "1" ]; then
    echo "EVALING MESSAGE PASSING"

    python models/eval_rels.py -m sgdet -model stanford -b 6 -p 100 -lr 1e-3 -ngpu 1 -clip 5 \
    -ckpt checkpoints/stanford-sgdet/vgrel-18.tar -cache stanford_sgdet.pkl -test
elif [ $1 == "2" ]; then
    echo "EVALING MOTIFNET"
    python models/eval_rels.py -m sgdet -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 6 -clip 5 \
        -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt checkpoints/motifnet-sgdet/vgrel-14.tar -nepoch 50 -cache motifnet_sgdet.pkl -use_bias
fi



