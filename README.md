# Faster-RCNN-from-scratch

This is the implementation of the classic Faster RCNN paper. This repo contains mainly 2 folders. one of the folders "utils" contains all the basic functions like knn classifier, backword pass, forward pass, softmax etc. The main code for constructing the module is in the core directory. 


## Dataset 

We will use the props dataset provided by the University of Michigan's PROGRESS lab led by Professor Chad Jenkins. Here is a snapshot of the dataset and what it should look like once it's passed through the dataloader. 
<p align="center">
    <img src="images/Screenshot from 2025-02-02 17-32-02.png" alt="alt text">
</p>

## Models

All models trained for each stage is stored inside the core directory and should either be retrained according to nature or be used as is.

## Results

There a lot of intermediatory images for anchors on the datset and bounding box based gt visualizers found in the jupiter notebooks but here I will provide the final result from the completed 2 stage detector network written overall. 
<p align="center">
    <img src="images/Screenshot from 2025-02-02 18-07-58.png" alt="alt text" width="45%">
    <img src="images/Screenshot from 2025-02-02 18-08-19.png" alt="alt text" width="45%">
</p>
