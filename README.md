# Using a multi-strain infectious disease model with physical information neural networks to study the time dependence  of SARS-CoV-2 variants of concern
## VOCs-INN Code
### Python Dependencies
We use the Python 3.7.4 and `VOCs-INN` mainly depends on the following Python packages.

```
tensorflow-gpu==1.14.0
matplotlib==3.3.3
numpy==1.19.5
pandas==1.2.1
pyDOE==0.3.8
scipy==1.6.0
```

## Running the Codes


In the folder, there are four Python codes: the "Training" code runs the corresponding model to infer parameters and fit data, and saves the results. The "PostProcess" code calculates the mean and standard deviation of the training results. The "Prediction" code is used for short-term forecasting. The "Plot_con" code is used to draw results.
