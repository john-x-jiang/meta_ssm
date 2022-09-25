#!/bin/bash

# Get z_c
python main.py --config meta09 --stage 4
python main.py --config meta10 --stage 4

# TSNE plot
# python visualization_tsne.py --config meta09 --test 0
# python visualization_tsne.py --config meta09 --test 1
python visualization_tsne.py --config meta09 --test 2
# python visualization_tsne.py --config meta09 --test 3
# python visualization_tsne.py --config meta10 --test 0
# python visualization_tsne.py --config meta10 --test 1
python visualization_tsne.py --config meta10 --test 2
# python visualization_tsne.py --config meta10 --test 3

# Run again for generation
python main.py --config meta09 --stage 4
python main.py --config meta10 --stage 4
