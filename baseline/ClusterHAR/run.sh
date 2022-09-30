#!/bin/bash

# python main.py --store 'original' -g 3

python transfer.py --pretrained 'original' -g 3 -version '50_200'  --store 'three_layer'