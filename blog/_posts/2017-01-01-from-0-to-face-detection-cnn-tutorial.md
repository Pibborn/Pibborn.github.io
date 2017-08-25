---
layout: post
title: ! 'From 0 to Face Detection: Convolutional Neural Networks Tutorial with Python'
permalink: cnn-tutorial/
mathjax: true
categories: neural-nets machine-learning tutorial
---
# Introduction
How hard is it for a complete newbie to learn how to do some interesting machine learning? It turns out, with the current state of tooling - not very. In this post, I will discuss a complete deep learning solution for the computer vision task of face detection. Face detection is an interesting task which boils down to:

- detect how many faces are in a given image;
- detect where they are as accurately as possible. 
 
This has obvious surveillance and security applications (...yay?), but it can also be used in complex multimedia information retrieval. But more about this later :) 

# Convolutional Neural Networks

A Convolutional Neural Network (CNN from here onwards) is a neural network with a "deep" architecture, i.e. with more than one hidden layer. CNNs are inspired by biological research on animals' visual system, which reveals that the cells in the visual cortex are arranged in a way so that each cell is sensible to a section of the animal's visible field [1]. For this reason they are expecially used in Computer Vision tasks. 

## Neural Networks Overview

The term "neural network" refers to a broad family of models that are inspired by the human neuron. They are often used to undertake supervised machine learning problems, but there are a number of models that also manage learning with a unsupervised dataset.

There are at least three elements that are needed to define a Neural Network model:

-  An **architecture**: defining how many neurons are in the model and how they are connected. Usually, the connections between neurons are weighted, and those weights are the parameters of a neural network.
- An **activation rule**: a function that determines how the neuron outputs values. In a Perceptron architecture a neuron can either be activated (output value 1) or not (output value 0), but that's not the case in a Convolutional Neural Network.
- An **update rule**: a function that modifies the weights of the connections over time by looking at the final performance of the network. 

But even before this, the basic building block of a neural network is the artificial neuron:

![neuron](http://www.theprojectspot.com/images/post-assets/an.jpg)
(from http://www.theprojectspot.com/)

Here, the neuron (grey in the center) takes 4 values as input; it weights them by vector multiplication $x^T w$; it sums them up; finally, it outputs a value $y$ depending on its activation function, $y = f(x^T w)$. This is a 4-input, single-output Perceptron architecture.

 All neural network architectures have a similar structure, with the main differences being the connection schema and the number of layers. An example of a multi-layer architecture may be:

![MLP](https://upload.wikimedia.org/wikipedia/commons/4/46/Colored_neural_network.svg)
(from [Wikipedia](https://en.wikipedia.org/wiki/Artificial_neural_network# /media/File:Colored_neural_network.svg))

This is usually called a Multi-Layer Perceptron. All layers are fully connected, meaning that all neurons of any layer are connected to all neurons of the following layer, and the activation functions are usually non-linear. 

For a long time, this architecture was thought to be the best Neural Network possible: it is possible to prove that the Multi-Layer Perceptron is an universal approximator, meaning that it can learn any function. However, more recent research has brought to the development of many "deep" architectures, that have more than one hidden layer and often have a lot of them. 

## Architecture Overview

As it often the case when talking about "deep learning", there is no specific architecture that perfectly defines all CNNs; however, some common characteristics can be found in all the work dealing with CNNs. 

- A **convolutional layer**: at least one of the hidden layers implements the signal processing operation of convolution. A convolutional layer is not densely connected and typically has a nonlinear activation function for its neurons. The neurons in a convolutional layer are arranged in a 3D volume, reflecting the structure of the input. 
- A **pooling layer**: a pooling layer usually comes immediately after a convolutional layer: its purpose is to reduce the dimensionality of the output of the convolutional layer. This layer's activation function is a block function that takes a matrix as input and returns a single value, for example $max(matrix)$ or $avg(matrix)$. This can be thought of as a non-linear spatial filter.
- A **fully-connected layer**: after convolving and downsampling the input image the desired amount of times (by stacking layers), the neural net should have learned high level features. However, we still seek to make the final inference using these features; we could be interested in outputting class scores for a classification task [2], or pixel positions for regression tasks [3]. This is done by stacking one or more fully-connected layers on top of the existing architecture. 

Literature on CNNs also defines the neurons responsible for receiving the input as the *input layer* and the neurons on the deepest level as the *output layer*.

##  The Image Domain

Before diving into the details of how CNNs work, it makes sense to introduce some image processing concepts to better understand why they are a good fit for Computer Vision tasks. 

### A Color Image is a 3D Matrix: the RGB model
An image can be thought of as a map of pixels: each pixel has a location $(x, y)$ and a color $c$. In the real world there are infinitely possible colors, but for digital images the accepted standard is the RGB color model. In this model, each pixel is colored by combining its Red, Green and Blue component: each component can have a value from 0 to 255 (8 bit for each "color channel", so 24 bits of *color depth* total).
This way, a color image can be thought of as being made up of three stacked matrices: one stores the Red component for each pixel, one stores the Green component and the last one stores the Blue component.

A greyscale image is a 2D matrix with each pixel having a value from 0 to 255; a black-and-white image is again 2D, but its pixel's color values can only be $0$ (black) or $1$ (white).

### A Color Image is a Discrete Function: Convolution

Convolution is a signal processing operation originally defined on continuous functions: it outputs a third function which shape depends on the correlation between the two input functions. Usually, the first input function $f$ is the original "signal" we are interested in processing; the second input function $g$ is a "convolution function", a special function that has no meaning on its own but is useful to modify the first function via convolution. This process is sometimes called *filtering*, as some convolution functions are useful to filter out unwanted characteristics of the image, or to enhance desirable ones.

An image can be thought of as a two-variable function $i(x, y) = p_n$, where c is the pixel color that can be computed as described in the preceding paragraph. $x, y$ and $c$ are discrete, finite values: they can be represented as a 2D matrix $h$ where $I(i, y)  . When computing convolution between an image matrix and a *convolution matrix*, the following computation is performed.

![convolution-image](http://i.imgur.com/aD2Cuiv.png)
*(from MACHINE VISION, Ramesh Jain, Rangachar Kasturi, Brian G. Schunck. McGraw-Hill 1995)*

As we can see in the figure, the convolution matrix $[A \dots F]$ is multiplied elementwise with a block of the image $[p1 \dots p9]$. These values are then summed to give the final convolution result for output image pixel $h[i, j]$. Then, the convolution matrix "slides" throughout the original image and the computation is performed again. In the end, the output will be another matrix that can be interpreted as an image.


![](http://cse19-iiith.vlabs.ac.in/neigh/convolution.jpg)
*(From http://cse19-iiith.vlabs.ac.in/theory.php?exp=neigh)*


Convolution can be thought of as "superimposing" the convolution matrix over the image matrix. Note that each pixel of the original image gets its turn in being the "center" of the convolution operation. This way, the output image size is the same as the input image size. 

Some convolution matrices examples ($\star$ is the convolution operator):

![orig](https://upload.wikimedia.org/wikipedia/commons/5/50/Vd-Orig.png) $ \star $ ![id-matrix](http://i.imgur.com/uc1rr1T.png) $ = $
![orig](https://upload.wikimedia.org/wikipedia/commons/5/50/Vd-Orig.png) (identity)


![orig](https://upload.wikimedia.org/wikipedia/commons/5/50/Vd-Orig.png) $ \star $ ![blur-matrix](http://i.imgur.com/e8g8Hax.png) $ * 1/9 = $ ![blur](https://upload.wikimedia.org/wikipedia/commons/0/04/Vd-Blur2.png)(blurring)

![orig](https://upload.wikimedia.org/wikipedia/commons/5/50/Vd-Orig.png) $ \star $ ![edge-matrix](http://i.imgur.com/LjHIndq.png) $ = $ ![blur](https://upload.wikimedia.org/wikipedia/commons/6/6d/Vd-Edge3.png) (edge detection)

*(From [Wikipedia](https://en.wikipedia.org/wiki/Kernel_%28image_processing%29)*).

##  The Convolutional Layer
So, we can see that convolution can be useful to extract features from an image: a sharpening filter can also be thought of as an edge detector. This is the fundamental idea of the convolutional layer: the data coming in from the input layer should be convolved with a matrix that extracts meaningful features from it. So, the weights $w$ we want to learn are the values of the convolution matrix.

###  Depth Columns: Local Connectivity
As convolution is a local operation, it is unnecessary for the convolution layer to be fully connected with the preceding layer. This helps in reducing the number of parameters that the net has to learn. 

![prova](http://cs231n.github.io/assets/cnn/depthcol.jpeg)

On the left, there is the *input layer*, taking in 32x32 pixels RGB color pictures (so a 32x32x3 matrix); on the right, there is the *convolutional layer*, with a *depth column* of five neurons all looking at the same area in the picture. A convolutional layer is a *volume* of neurons: depth columns are useful to learn different features in the same region of the picture. The extent of the input area these neurons are connected with can be defined as a parameter of the layer, but is usually a square region. 

###  Depth Slices: Parameter Sharing 
In the convolution volume/3D matrix $C$, a *depth slice* is the 2D matrix of neurons at a certain depth $d$: $C[:,:,d]$. To further reduce the number of parameters that the convolutional layer has to learn, we can have all the neurons at depth $d$ *share* the learned weights. 

Since we defined the weights as the values of the convolution matrix, this also makes sense from an image processing point of view: when we compute the convolution operation between an image and a convolution matrix, we keep the same convolution matrix while sliding throughout the image. This way, the input image gets processed the same way in all its areas, just as it is in classical image processing.

Summarizing depth columns and depth slices:

- Neurons in the same depth column ($C[x, y, :]$) look at the same area of the picture, but filter (convolute) differently. Therefore, the neurons in a depth column extract *different features from the same area*;
- Neurons in a depth slice ($C[:,:,d]$) implement the same filtering operation in all areas of the picture. Therefore, they extract *the same feature from the whole picture*.

### Activation Function
While any nonlinear activation function can be used, some work has shown the rectifier function $f(x)=max(0, x)$ to work better than $tanh$ or the sigmoid[[2]]. Another argument that could be made in favor of the ReLU function is that it better preserves the filtering work made by the convolution operation. 

### Output Volume
As we discussed, the convolution layer learns a series of filters, the weights in a depth slice. The image will be convolved with each one of these filters; therefore on top of being a volume the convolutional layer also outputs a volume. Its depth is equal to the depth of the convolutional layer, while its width and height depend on some parameters.

### Convolutional Layer Parameters

#### Filter Size
Recall that each depth column is connected to a small area of the input layer. This can be changed to have a bigger or smaller convolution matrix, for any reason.

#### Stride
Instead of sliding the convolution matrix pixel-by-pixel over the original image, we can make bigger "jumps": as a result, the output image gets smaller, and the filters we are learning have less overlap. If the stride is greater than half the filter size, the convolution layer does not cover the whole input area. We will not be learning any feature from these areas.

#### Number of Filters
Which is also the size of a depth column. We may interested in learning a lot of features from the image, or just one.

## Pooling Layer
The pooling layer implements a downsampling operation, which could be useful if the output volume of the convolutional layer is too big. 

![max-pooling](http://cs231n.github.io/assets/cnn/maxpool.jpeg)

Here, max-pooling is implemented: from each 2x2 block of each depth slice of the output volume the maximum value is taken. There are alternatives such as average pooling, but max-pooling is more used in practice.

Note that a pooling layer does not reduce the depth of the volume.

## Fully-connected layer
The fully-connected layer is a "classic" neural network layer. One interesting thing to note is that the last layer has to be one-dimensional, so it is responsible for "squashing" the three-dimensional convolution and pooling results into the final class prediction or regression score. 

## Training a CNN
All the work done on CNNs cites "gradient descent" [4] or backpropagation [5] with some stochastic elements (a temperature or momentum factor). Otherwise, there is little reason to believe that the training algorithms are nothing but "stock" backpropagation.

# A Task: Finding Faces
One classical computer vision task is to parse images to find human faces. This problem predates CNNs popolarity, and some robust approaches do not use neural networks at all [6]. However, some work has shown that CNNs that have been trained for a different purpose (such as classifying images with broad labels such as "cat" or "house", as in [2]) can be adapted to face detection with some fine-tuning and further training [7]. 

It could be interesting to train a CNN from scratch to specifically detect faces.

## A Tool: Lasagne + Nolearn
[Lasagne](https://github.com/Lasagne/Lasagne) is a Python library that has high-level abstractions in place to train neural networks. Its [layers](http://lasagne.readthedocs.org/en/latest/modules/layers.html) module has implementation available for all the layers we need in a CNN (hence the library's name).

Lasagne is meant to be used alongside [Theano](http://deeplearning.net/software/theano/), another Python library for high-performance computing on arrays. Theano's applications are not limited to neural networks, but it is particularly efficient when [computing backpropagation](http://deeplearning.net/software/theano/tutorial/symbolic_graphs.html), as it uses [symbolic differentiation](http://deeplearning.net/software/theano/tutorial/gradients.html# tutcomputinggrads). Also, Theano supports the compiling of mathematical expressions to highly-parallel C code that can be run on the GPU. 

[Nolearn](https://github.com/dnouri/nolearn) bridges Lasagne to the [scikit-learn](http://scikit-learn.org/stable/) environment, which makes some generic machine learning tools available. It also puts in place further abstractions for training a neural network. 

Defining a CNN layer becomes very easy when using nolearn and Lasagne:

```
(Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3)})
```

Which creates a convolutional layer with depth 32 (recall that the number of filters and the depth of the convolutional layer are the same thing) and (3, 3) filter size. The width and height of the layer depend on the filter size and the stride, which can be defined explicitely but defaults to 1. A similar sintax can be used for Pooling layers and Fully Connected layers. 


## A Dataset: FDDB
The [Face Detection Data Set and Benchmark](http://vis-www.cs.umass.edu/fddb/) (FDDB) [8] is a dataset of 5171 face regions annotations over the [Faces in the Wild](http://tamaraberg.com/faceDataset/index.html) dataset, which comes from news photographs. The annotations describe elliptical regions.

FDDB is organized in 10 folds which are already randomized in order, and also comes with code to test performance of any custom face detection algorithm one would like to try. Each fold is a .txt file that is organized this way:

>2002/08/11/big/img_591 [Faces in the Wild picture path]

>1 [number of detected faces]

>123.583300 85.549500 1.265839 269.693400 161.781200  1

>[major_axis_radius minor_axis_radius angle center_x center_y 1]

However, this is a bit unconvienent for usage with a CNN: we do wish to learn the parameters of an elliptical face region, but to find out whether a square patch of pixels contains a human face. So what we would like to have is a number of RGB pixel values.  Also, this does not follow the standard .csv formatting that Lasagne can load with little issue. So the whole dataset gets processed in the following way:

1. *Squaring*: the elliptical regions are approximated by square regions with the same center;
2. *Cleaning(1)*: square regions smaller than 30x30 pixels are removed from the dataset;
3. *Cleaning(2)*: square regions which boundaries get out of the bounds of the image are also removed from the dataset;
4. *Pixel Value Extraction*: the square regions become defined by the RGB pixel values which are contained in their boundaries, obtaining a square 3D matrix; 
5. *Downsampling*: the RGB pixel values matrices get resized to a 30x30 RGB pixel matrix, via a downsampling algorithm.

###  Squaring
To the end of obtaining square patches, I obtained 4 points from the ellipse parameters:

   #  point arrangement:
   #  A B
   #  D C
    A_x = center_x - major_axis
    A_y = center_y + major_axis
    B_x = center_x + major_axis
    B_y = center_y + major_axis
    C_x = B_x
    C_y = center_y - major_axis
    D_x = A_x
    D_y = C_y


Always using the major axis of the ellipse guarantees that the square covers the ellipse's area as well as possible.

###  Cleaning
The two cleaning steps reduce the size of the dataset from 5171 to 4538. This is about a 12% loss on the training set, which is sizable. One could seek to mitigate this via data augmentation: I discuss a data augmentation technique later on.

# Codebase Overview
You can find my git repository [on GitHub](https://github.com/Pibborn/CNN-FaceDetection).
The code is organized in three main python modules: `train.py`, `find.py` and `data/preprocess.py`.

## Dependencies
Please skip this paragraph if you don't wish to run the code. Assuming pip and virtualenv are not installed on your system, open a terminal window and run:

    easy_install pip 
    pip install virtualenv
    cd [path-to-project-directory]
    virtualenv venv 
    source venv/bin/activate
    pip install lasagne
    pip install nolearn
    pip install matplotlib
    

## data/preprocess.py
This module computes all the dataset pre-processing: it takes the .csv files which contain the 4 square patch parameters, extracts RGB pixel values from that area and saves it in another .csv file which is the final dataset. It creates both true and false examples.

###  False example creation
At this point, I have implemented a simple function that takes a 150x150 pixel area, rescales it to 30x30 and calls it a false example. This is done enough times to maintain a 50/50 split between positive and negative examples in the final dataset.

###  Data augmentation
To improve on the ~4500 faces we have, I implemented a simple data augmentation procedure that offsets the square patch by a value between $0$ and $1$. $0$ means no offset, while $1$ moves the square patch by its length, missing the face completely. 

It is also possible to specify the dimension along the square patch will be slided.  

In my second experiment (described below), I experimented with a $0.5$ overlap along the $x$ axis.

## train.py
This program has two parameters: a `.csv` file and an output path where the neural network object will be dumped to. 

The `.csv` formatting it expects is:
`pixel1r, ...,pixel900r, pixel1g, ..., pixel900g, pixel1b ,...,pixel900b, class`

Where `class` is 1 if the picture is a face, and 0 if it is not. Also, different examples should be put in different lines.

It can then be run this way:

`python train.py [path-to-csv] [output-path]`

I provide an example dataset at ```data/training-set.csv```.

The file first loads the examples in two numpy arrays, `X` and `y`:


    with open(path, 'rb') as f:
        next(f) #  skip header
        for line in f:
            xi = line.split(',')
            xi[-1] = xi[-1].strip()
            yi = xi[-1]
            X.append(xi[0:-1])
            y.append(yi)

    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.int32)
    
By removing the last element from `xi`, we removed the class information from `X`: scikit-learn wants the data in a `X` numpy array and the labels in a separate `y` array. This is remindful of the $y = x^T w$ notation.

`X` is then normalized and reshaped:


    X -= X.mean()
    X /= X.std()

    X = X.reshape(
        -1, #  number of samples, -1 makes it so that this number is determined automatically
        3,  #  3 color channel
        30, #  first image dimension (vertical)
        30 #  second image dimension (horizontal)
    )

Then, what is left is defining the Neural Net layers:

    layers1 = [
    (InputLayer, {'shape': (None, X.shape[1], X.shape[2], X.shape[3])}),

    (Conv2DLayer, {'num_filters': 10, 'filter_size': (3, 3)}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (Conv2DLayer, {'num_filters': 10, 'filter_size': (3, 3)}),
    (Conv2DLayer, {'num_filters': 10, 'filter_size': (3, 3)}),
    (MaxPool2DLayer, {'pool_size': (2, 2)}),

    (DenseLayer, {'num_units': 30}),

    (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
    ]

    net0 = NeuralNet(
        layers=layers1,
        update_learning_rate=0.01,
        max_epochs=15,
        verbose=1,
        train_split=TrainSplit(eval_size=0.2)
    )

At this time, the Neural Net is not very deep and the single convolutional layers do not learn as many filters as they could be. This was necessary to keep the training time reasonable even on a GPU-less machine like mine. This way, an epoch of training took about 10 seconds. 

To fit the model to the data we loaded in `X` and `y`, the scikit-learn notation is:

`net0.fit(X, y)`

I performed two separate experiments, with some differences.

- 1: architecture and neural network parameters as described above; no data augmentation; 15 epochs of training.
- 2: architecture as described above; learning rate $0.0002$; data augmentation with overlap $0.5$; 100 epochs of training.

### Experiment 1
Running `train.py` prints the following output on the terminal:

     #  Neural Network with 9692 learnable parameters
  
      Layer information
  
     #     name        size
      ---  ----------  --------
      0    input0      3x30x30
      1    conv2d1     10x28x28
      2    maxpool2d2  10x14x14
      3    conv2d3     10x12x12
      4    conv2d4     10x10x10
      5    maxpool2d5  10x5x5
      6    dense6      30
      7    dense7      2
  
      epoch    train loss    valid loss    train/val    valid acc   dur
      -------  ------------  ------------  -----------  -----------  ------
        1       0.58346       0.43057      1.35507      0.81009    13.37s
        2       0.34838       0.29651      1.17492      0.88698    13.66s
        3       0.27342       0.27000      1.01266      0.89176    13.41s
        4       0.24866       0.25595      0.97149      0.89541    13.23s
        5       0.23632       0.24883      0.94975      0.89905    13.37s
        6       0.22545       0.23743      0.94951      0.90521    13.76s
        7       0.21979       0.23358      0.94095      0.90938    14.28s
        8       0.21186       0.22517      0.94091      0.91302    15.36s
        9       0.20499       0.22179      0.92425      0.91563    14.29s
       10       0.19565       0.22192      0.88164      0.91771    14.63s
       11       0.18991       0.21620      0.87840      0.92031    12.81s
       12       0.18438       0.21434      0.86022      0.92135    13.35s
       13       0.17837       0.21520      0.82887      0.92344    13.29s
       14       0.17338       0.21111      0.82128      0.92812    12.68s
       15       0.16778       0.21503      0.78029      0.92396    12.89s

This already features OK accuracy, but the loss function starts to get very low after only a few iterations, so the learning algorithm has an hard time doing better. This situation is what inspired me to implement the data augmentation procedure: a more challenging dataset may fix this situation.

###  Experiment 2
    epoch    train loss    valid loss    train/val    valid acc  dur
    -------  ------------  ------------  -----------  -----------  ------
      1       0.70398       1.40156      0.50228      0.36257  20.77s
      2       1.01768       0.90145      1.12893      0.50675  20.65s
      3       0.86900       0.83468      1.04111      0.71733  20.22s
      4       0.81641       0.80285      1.01688      0.73509  20.76s
      5       0.77764       0.77165      1.00776      0.76918  20.67s
      6       0.73962       0.74351      0.99477      0.80930  21.01s
      7       0.70768       0.71469      0.99019      0.83665  20.36s
      8       0.67796       0.69015      0.98233      0.84482  20.31s
      9       0.65296       0.66576      0.98077      0.84553  21.20s
     10       0.63012       0.64700      0.97391      0.84091  20.48s
     11       0.61101       0.62894      0.97149      0.84233  21.04s
     12       0.59497       0.61376      0.96939      0.84339  20.17s
     13       0.58152       0.59874      0.97125      0.84375  20.32s
     14       0.57090       0.58267      0.97980      0.84588  20.21s
     15       0.56058       0.57044      0.98272      0.84943  20.29s
     [...]
     95       0.39397       0.39997      0.98499      0.91726  19.57s
     96       0.39312       0.39920      0.98476      0.91726  20.42s
     97       0.39227       0.39854      0.98427      0.91797  20.29s
     98       0.39140       0.39786      0.98376      0.91797  20.88s
     99       0.39060       0.39706      0.98374      0.91868  21.03s
    100       0.38970       0.39639      0.98313      0.91939  20.80s

While the intuition about providing a more challenging dataset was correct, as the loss functions are much higher than in the first experiment even after 100 epochs of training, the validation set accuracy is lower. 

However, as the validation sets are not the same between the two experiments, it is hard to argue that the second experiment was less succesful than the first one: I would not be surprised if using another dataset entirely for validation would show that the second CNN outperforms the first.

## find.py

This program loads a small dataset (13 pictures) that was not included in the default training set ```data/training-set.csv```. It also takes as input the file the neural net was dumped to. If ```train.py``` has not been run, I provide two pre-trained neural nets: please run either

 ```python train.py data/validation-set.csv cnn-exp-one.pickle``` 

or 

 ```python train.py data/validation-set.csv cnn-exp-two.pickle``` 

to test the matching procedure. ```find.py```loads the validation data in the same way the training data was loaded, then it loads the neural net: 

    with open(argList[2], 'r') as f:
          net = pickle.load(f)
    X, y = loadExamples(argList[1])
    labels = net.predict(X)

and uses the scikit-learn notation ```predict(X)``` to try and predict labels for the loaded examples. Note that ```predict``` only takes ```X``` as input, as it has no knowledge of the actual labels of the data.  ```find.py``` will then print the following string on the terminal window:

    predicted labels:
    [1 0 1 0 1 0 0 1 1 1 1 0 1]
    actual lables:
    [1 0 1 0 1 0 0 1 1 0 1 0 1] 

In which we notice that the neural net only makes one mistake out of the 13 examples we used as a (toy) validation set. This seems coherent with the 92.3% accuracy we noticed during training. 


#  Results Discussion
While Lasagne and Nolearn make it easy to train a CNN and the results seem reasonable, there are some issues with the current architecture:

1. It is not very deep, and could stand to have more layers and filters in a single layer, as already discussed. It is easy enough to use cloud computing services to "borrow" a GPU, and this is something I will try to do in the next iteration;
2. There is no procedure in place to take a whole picture and *find the faces that are in it*. This is because the neural net wants a 30x30x3 matrix as input, but faces come in all sizes.
3. I'd also like to make the training procedure more flexible: one should just point the program to a folder full of pictures, and the program should be able to either train the network with those pictures or try to predict where the faces are.  
  

# Bibliography
[1] Hubel, D. and Wiesel, T.. Receptive fields and functional architecture of monkey striate cortex. Journal of Physiology (London), 195, 215–243. 1968.  
[2] Krizhevsky, Sutskever, Hinton. ImageNet Classification with Deep Convolutional Neural Networks. 2012.
[3] Nouri. [Using convolutional neural nets to detect facial keypoints tutorial](http://nbviewer.jupyter.org/github/dnouri/nolearn/blob/master/docs/notebooks/CNN_tutorial.ipynb). 2014. 
[4] Krizhevsky, Sutskever, Hinton. ImageNet Classification with Deep Convolutional Neural Networks. 2012. Page 3.
[5] Delakis, Garcia. Convolutional Face Finder: A Neural Architecture for Fast and Robust Face Detection. 2004. Page 7.
[6] Viola, Jones. Robust Real-Time Face Detection. 2001.
[7] Farfade, Saberian, Li. Multi-view Face Detection Using Deep Convolutional Neural Networks. 2015.
[8] Vidit Jain and Erik Learned-Miller. FDDB: A Benchmark for Face Detection in Unconstrained Settings. Technical Report UM-CS-2010-009, Dept. of Computer Science, University of Massachusetts, Amherst. 2010. 

Much of the discourse about CNNs and the Convolutional Layer was inspired by [the Stanford University class on CNNs](http://cs231n.github.io/).
