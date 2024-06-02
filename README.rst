===================
ImageClassification
===================

A short project on training classfication model on the various images of numbers.

Used the mnist data often referred as the "Hello World" of machine learning.  
There are 70,000 images and each images has 784 features, as its a 28X28 pixels each feature represents a pixel's intensity from 0 to 255. 
An illustration of how we can train a classfication model and challenges of evaluation process that changes as per the requirment of end goals. 

Evaluating the trained model is bit different when it comes to classification models compared to regression model. 
There are few metrics that i have mentioned in this repository like, ROC curve, precision and recall, plotting graphs.
Based on the result or goal you can select the metrics the suits the situation. For instance comparing accuracy is mostly not gonna be much helpful when the 
proportion of class in your training set if skewed. In situation like that looking up the precision, recall or f1 score is more sensible.  

* Free software: MIT license
* Documentation: https://ClassificationModel.readthedocs.io.


Features
--------


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
