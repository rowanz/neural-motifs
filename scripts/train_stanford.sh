#!/usr/bin/env bash

python models/train_rels.py -m sgcls -model stanford -b 4 -p 400 -lr 1e-4 -ngpu 1 -ckpt checkpoints/vgdet/vg-24.tar -save_dir checkpoints/stanford -adam

# To test you can run this command
# python models/eval_rels.py -m sgcls -model stanford -ngpu 1 -ckpt checkpoints/stanford/vgrel-28.tar -test
