# Retinex For Underwater Image Enhancement
***
## Introduction
This project is an implementation of MSRCR (Multi-Scale Retinex with Color Restoration) aiming underwater image enhancement task.
The MSRCR algorithm is proposed by Daniel J. Jobson in paper [***A Multiscale Retinex for Bridging the Gap Between Color Images and the Human Observation of Scenes***](https://ieeexplore.ieee.org/abstract/document/597272).

The program provided contains following algorithms:
+ SSR (Single Scale Retinex), the basic form of retinex;
+ MSR (Multi-Scale Retinex), combines SSR with different scales;
+ MSRCR, the major algorithm of the project.

## Configuration
One disadvantage of MSRCR is that it works with empirical parameters. Therefore, choosing appropriate parameters could significantly improve the performance of our algorithm. Here we list some of the parameters used in our version of MSRCR.
+ `sigma`: For Gaussian filtering in basic Retinex. As MSR blend SSR under different scales of Gaussian kernel, `sigma` list is required for constructing those Gaussian kernels.
+ `weights`: Weights array determines at what weight different scales of SSR results are superimposed on the final result of MSR. `weights` MUST keep the same length as `sigma` list.
+ `alpha` and `beta`: Empirical parameters to build CRF (Color Restoration Function) factor. CRF is the core of MSRCR algorithm. We construct the CRF based on the formulation proposed in Daniel's paper: **C(x, y) = beta * (log(alpha * I(x, y)) - log(sigmaI))** where **sigmaI** is the sum of **I(x, y)** along RGB channel.
+ `high_clip` and `low_clip`: For Retinex, pixel matrix is converted to logarithmic domain for computation. The output matrix of Retinex therefore becomes a logarithmic representation of the output pixel matrix on spatial domain. The transformation from logarithmic domain to spatial domain is called *quantization*. We proposed a quantitative method as followed:
    1. Go through a grayscale histogram of each channel in RGB.
    2. Sorted elements in each channel ascendingly into an array called `flatten`, we truncate lowest `len(flatten) * low_clip` elements and highest `len(flatten) * high_clip` elements into 0 and 255, respectively.
    3. Use linear quantization to quantify other elements to a range of [0,255].
    4. After the linear quantization, we impose a gamma correction to the pixel matrix. Experiments shows that gamma correction we used in our quantization performs well with underwater images. Here we leverage a gamma lookup table to minimize time consumption.
+ `gamma`: As we mentioned, this parameter is used in quantization process. Default value is 0.6.

Default values of above parameters work well on underwater images, so it's unnecessary to change them. However, to run the project correctly, you still need to set `path` for you own environment. Specifically, these options should be set:
+ `img_path`: Default as "./tests". All images in this path will be processed;
+ `output_path`: Default as "./results". If `output_result == true`, output processed images to this direction.

## Run
To start your image processing, you should run `MSRCR.py`. Note that the configuration file `config.json` must be checked before running the program.
