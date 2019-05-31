#!/bin/bash

for class in /penn-crop/frames/*; do
    echo "$(basename " $class ")"
    echo "python demo.py --demo $class --gpus -1 --load_model ../models/fusion_3d_var.pth"
    python demo.py --demo $class --gpus -1 --load_model ../models/fusion_3d_var.pth
done
echo "python lstm_classifier.py"
python lstm_classifier.py
echo "python pca.py"
python pca.py

