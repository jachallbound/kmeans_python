# kmeans_python
Rudimentary implementation of KMeans classification in Python

Able to classify any number K means for any number N dimensions.
Creates plot for 2D and 3D data.

Done just for fun. I am quite happy with `generate_gaussian_data` in GaussianDistributionGenerator.py.
(Although, I have since learned it is a worse re-implementation of SciPy's `scipy.stats.multivariate_normal`. Oh, well.)

`kmeans_driver.py` demonstrates how to use classes and functions in src/

## Plots
When working with 2D or 3D data, the program will plot the data with color coordinated labelling for the different classes.

It will also plot a histogram showing the distribution of true and predicted samples.

## Issues
+ K Means can achieve good classification, but report it as poor in the figures due to the fact that the order of the labels can change between truth and prediction. (i.e. KMeans calls label 0 label 1 instead.)
  + Coordinating this system seems like too much work.
+ You can choose to create a different amount of actual distributions than what you try to classify, but it's currently broken.
Thus, you must classify the same amount of distributions you create.

## To Do
+ Write docstrings for all functions in `src/` (I'm probably not going to do this)

## Requirements
+ Python 3
+ numpy
+ matplotlib
