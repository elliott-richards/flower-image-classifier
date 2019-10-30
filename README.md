# Introduction
An image classifier able to identify 102 types of flowers. The train.py file will train a neural network on a set 
of flower images in a given directory. The predict.py file will then predict the type of flower in a submitted image.

# Requirements 
Both files rely on Pytorch in order to run
For training the network, the directory should be structured:
`folder/labels/images`

# How To Run
The application runs from the command line, examples of how to run each file can be found below
## train.py
`python path/to/train.py path/to/image_dir --save_dir path/to/save_dir --arch vgg16 --learning_rate 0.001 --hidden_units 4096
--epochs 5 --gpu True`
## predict.py
`python path/to/predict.py path/to/image path/to/checkpoint --top_k 5 --category_names path/to/idx_to_name_map --gpu True`
