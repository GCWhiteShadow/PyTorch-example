# AutoEncoder
Autoencoder is an unsupervised learning neural network. 
It's philosophy is using trained NN encoder to compress an image into lower latent space and used trained NN decoder to reconstruct it. 

## Overview
![](https://i.imgur.com/4VmiO0f.png)

## Dataset
This example is using MNIST handwritten digits as dataset. 
The dataset contains 60,000 examples for training and 10,000 examples for testing. 
These digits have been size-normalized and centered in a fixed-size image (28x28 pixels) with values in [0, 1]. 
For simplicity, each image has been flattened and converted to a 1-D numpy ndarray of 784 features (28*28).

![](https://i.imgur.com/8mVRgQg.png)

More info: ([http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/))

## Usage
```
pip3 install -r requirements.txt
python3 Autoencoder.py
```

## Result
![](https://i.imgur.com/Ps2yOE7.png)

upper row is the original images and the lower ones are the reconstructed ones

