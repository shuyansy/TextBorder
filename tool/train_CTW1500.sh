#!/bin/bash
cd ../
CUDA_VISIBLE_DEVICES="3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node 8 train_TextGraph.py
