# kmeans_python
Rudimentary implementation of KMeans classification in Python

Able to classify any number K means for any number N dimensions.
Creates plot for 2D and 3D data.

Done just for fun. I am quite happy with `generate_gaussian_data` in GaussianDistributionGenerator.py.
(Although, I have since learned it is a re-implementation of SciPy's `scipy.stats.multivariate_normal`. Oh, well.)

`kmeans_driver.py` demonstrates how to use classes and functions in src/

You can choose to create a different amount of actual distributions than what you try to classify, but it's currently broken.
Thus, you must classify the same amount of distributions you create.

## Requirements
+ Python 3
+ numpy
+ matplotlib
