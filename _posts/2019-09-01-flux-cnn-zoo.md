---
layout: single
title: Julia Flux Convolutional Neural Network Explained
date: 2019-09-01
categories: [Deep Learning]
comments: true
tags: [Flux, CNN]
excerpt: Taming the CNN vision example in the Flux Model Zoo
header:
  image: "/images/ai-s.jpeg"
---

In this blog post we’ll breakdown the convolutional neural network (CNN) demo given in the [Flux Model Zoo](https://github.com/FluxML/model-zoo/blob/master/vision/mnist/conv.jl).  We'll pay most attention to the CNN model build-up and will skip over some of the data preparation and training code. 

The objective is to train a CNN to recognize hand-written digits using the famous MNIST dataset.

If you are new to CNN's I recommend watching all these videos to gain the concepts needed to understand this post. Note, some of the videos dive into Kera’s coding but it’s actually very comparable to Flux.

[Convolutional Neural Networks (CNNs) explained](https://www.youtube.com/watch?v=YRhxdVk_sIs)

[Zero Padding in Convolutional Neural Networks explained](https://www.youtube.com/watch?v=qSTv_m-KFk0&t=611s)

[Max Pooling in Convolutional Neural Networks explained](https://www.youtube.com/watch?v=ZjM_XQa5s6s&t=407s)

[Batch Size in a Neural Network  explained](https://www.youtube.com/watch?v=U4WB9p6ODjM)

OK, we’ve got the concepts let’s dive into the Flux example.  The first block of code prepares the data for training.


```julia
# Classifies MNIST digits with a convolutional network.
# Writes out saved model to the file "mnist_conv.bson".
# Demonstrates basic model construction, training, saving,
# conditional early-exit, and learning rate scheduling.
#
# This model, while simple, should hit around 99% test
# accuracy after training for approximately 20 epochs.

using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated, partition
using Printf, BSON

# Load labels and images from Flux.Data.MNIST
@info("Loading data set")
train_labels = MNIST.labels()
train_imgs = MNIST.images()

# Bundle images together with labels and group into minibatchess
function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end
batch_size = 128
mb_idxs = partition(1:length(train_imgs), batch_size)
train_set = [make_minibatch(train_imgs, train_labels, i) for i in mb_idxs]

# Prepare test set as one giant minibatch:
test_imgs = MNIST.images(:test)
test_labels = MNIST.labels(:test)
test_set = make_minibatch(test_imgs, test_labels, 1:length(test_imgs))
```

Let’s pause here to look at how the training and test data has been arranged.  As usual in Flux the training data is arranged as a tuple of x training data and y labels.  Let’s verify.


```julia
typeof(train_set)
```

    Array{Tuple{Array{Float32,4},Flux.OneHotMatrix{Array{Flux.OneHotVector,1}}},1}

We see that the x part of the tuple is a 4 dimensional Float32 array and the y part is a Flux.OneHotVector.

Let’s take a look at the size of first training batch.

```julia
size(train_set[1][1]) # training data Float32
```

    (28, 28, 1, 128)

It is important to note these dimensions are arranged in WHCN order standing for Width, Height, Channels and Number (of batches).  

So as expected for MNIST, each image is W=28 pixels x H=28 pixels.

C = 1 as there is only one channel for the grey scale intensity.

N=128 as the batch size.

Now let's have a look at the size of the first batch of y labels.

```julia
size(train_set[1][2]) # OneHotVector labels
```


    (10, 128)

Each OneHotVector in the batch encodes the labelled digit; i.e. whether it is 1 through to 10.  You can see the first OneHotVector in the first batch with the following code.

```julia
train_set[1][2][:,1]
```




    10-element Flux.OneHotVector:
     false
     false
     false
     false
     false
      true
     false
     false
     false
     false


## Flux CNN Model Explained

Here's the next block of code from the model zoo that we're mostly interested in: -


```julia
# Define our model.  We will use a simple convolutional architecture with
# three iterations of Conv -> ReLU -> MaxPool, followed by a final Dense
# layer that feeds into a softmax probability output.
@info("Constructing model...")
model = Chain(
    # First convolution, operating upon a 28x28 image
    Conv((3, 3), 1=>16, pad=(1,1), relu),
    x -> maxpool(x, (2,2)),

    # Second convolution, operating upon a 14x14 image
    Conv((3, 3), 16=>32, pad=(1,1), relu),
    x -> maxpool(x, (2,2)),

    # Third convolution, operating upon a 7x7 image
    Conv((3, 3), 32=>32, pad=(1,1), relu),
    x -> maxpool(x, (2,2)),

    # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
    # which is where we get the 288 in the `Dense` layer below:
    x -> reshape(x, :, size(x, 4)),
    Dense(288, 10),

    # Finally, softmax to get nice probabilities
    softmax,
)

```

### Layer 1

`` Conv((3, 3), 1=>16, pad=(1,1), relu),`` 

The first layer can be broken down as follows: -

``(3,3)`` is the convolution filter size (3x3) that will slide over the image detecting new features.

 ``1=>16`` is the network input and output size.  The input size is 1 recalling that one batch is of size 28x28x1x128.  The output size is 16 meaning we’ll create 16 new feature matrices for every training digit in the batch.  

``pad=(1,1)`` This pads a single layer of zeros around the images meaning that the dimensions of the convolution output can remain at 28x28.

``relu`` is our activation function.

The output from this layer only can be viewed with ``model[1](train_set[1][1])`` and has the dimensions 28×28×16×128.


### Layer 2 

``x -> maxpool(x, (2,2)),``

Convolutional layers are generally followed by a maxpool layer.  In our case the parameter ``(2,2)`` is the window size that slides over x reducing it to half the size whilst retaining the most important feature information for learning.

The output from this layer only can be viewed with ``model[1:2](train_set[1][1])`` and has the output dimensions 14×14×16×128.

### Layer 3

``Conv((3, 3), 16=>32, pad=(1,1), relu),``

This is the second convolution operating on the output from layer 2.

``Conv((3, 3),`` is the same filter size as before.

``16=>32``  This time the input is 16 (from layer 2).  The output size of the layer will be 32.

The padding, filter size and activation remains the same as before.

The output from this layer only can be viewed with `` model[1:3](train_set[1][1])`` and has the output dimensions 14×14×32×128.

### Layer 4

``x -> maxpool(x, (2,2)),``

Maxpool reduces the dimensionality in half again whilst retaining the most important feature information for learning.

The output from this layer only can be viewed with ``model[1:4](train_set[1][1])`` and has the output dimensions 7×7×32×128.

### Layers 5 & 6

``Conv((3, 3), 32=>32, pad=(1,1), relu),``

``x -> maxpool(x, (2,2)),``

Perform a final convolution and maxpool.  The output from layer 6 is 3×3×32×128

### Layer 7 

``x -> reshape(x, :, size(x, 4)),``

The reshape layer effectively flattens the data from 4-dimensions to 2-dimensions suitable for the dense layer and training.

The output from this layer only can be viewed with ``model[1:7](train_set[1][1])`` and has the output dimensions 288×128.  If you’re wondering where 288 comes from, it is determined by multiplying the output of layer 6; i.e. 3x3x32.

### Layer 8

``Dense(288, 10),``

Our final training layer takes the input of 288 and outputs a size of 10x128. 

(10 for 10 digits 0-9)

### Layer 9

``softmax,``

Outputs probabilities between 0 and 1 of which digit the model has predicted.

The remainder of the code is pasted below for completeness.


```julia
# Load model and datasets onto GPU, if enabled
train_set = gpu.(train_set)
test_set = gpu.(test_set)
model = gpu(model)

# Make sure our model is nicely precompiled before starting our training loop
model(train_set[1][1])

# `loss()` calculates the crossentropy loss between our prediction `y_hat`
# (calculated from `model(x)`) and the ground truth `y`.  We augment the data
# a bit, adding gaussian random noise to our image to make it more robust.
function loss(x, y)
    # We augment `x` a little bit here, adding in random noise
    x_aug = x .+ 0.1f0*gpu(randn(eltype(x), size(x)))

    y_hat = model(x_aug)
    return crossentropy(y_hat, y)
end
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

# Train our model with the given training set using the ADAM optimizer and
# printing out performance against the test set as we go.
opt = ADAM(0.001)

@info("Beginning training loop...")
best_acc = 0.0
last_improvement = 0
for epoch_idx in 1:100
    global best_acc, last_improvement
    # Train for a single epoch
    Flux.train!(loss, params(model), train_set, opt)

    # Calculate accuracy:
    acc = accuracy(test_set...)
    @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))
    
    # If our accuracy is good enough, quit out.
    if acc >= 0.999
        @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
        break
    end

    # If this is the best accuracy we've seen so far, save the model out
    if acc >= best_acc
        @info(" -> New best accuracy! Saving model out to mnist_conv.bson")
        BSON.@save "mnist_conv.bson" model epoch_idx acc
        best_acc = acc
        last_improvement = epoch_idx
    end

    # If we haven't seen improvement in 5 epochs, drop our learning rate:
    if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
        opt.eta /= 10.0
        @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")

        # After dropping learning rate, give it a few epochs to improve
        last_improvement = epoch_idx
    end

    if epoch_idx - last_improvement >= 10
        @warn(" -> We're calling this converged.")
        break
    end
end
```

[Need more help?, try this article by Mike Gold](https://www.linkedin.com/pulse/creating-deep-neural-network-model-learn-handwritten-digits-mike-gold/)

