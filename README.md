# Covid-Death-Prediction ML Algorithm

## Disclaimer
This project is a learning experience and emerged as a solution to the [COVID 19 Death Prediction](https://www.kaggle.com/competitions/Covid19-Death-Predictions/overview) Kaggle challange. This is not an ideal solution, no significant amount of data analysis went into this project, for insights into data analysis please see other entries to the challange with the link above.

## The Problem
We are looking for a Machine Learning Algorithm that is able to predict the number of people that are likely going to die this week, based on some given disease information from last week. 

This Information (set of samples) consists of:
- The given samples' Location (string/object)
- Total Weekly Cases (number/float)
- The year that the sample was created (number/int)
- Weekly Cases per 1.000.000 people (number/float)
- Weekly Deaths for this week (number/float)
- Weekly Deaths per 1.000.000 people (number/float)
- Total number of administered Vaccinations (number/float)
- Total number of people who recieved at least one Vaccine dose (number/float)
- Total number of people who recieved all doses prescribed by the initial vaccination protocol (number/float)
- <i>Next weeks Covid deaths (number/float)</i>

We are looking for:
- Next weeks Covid deaths (number/float)

## How the notebook works
The notebook consists of three parts:

### "Import and Prep",
which does some basic data preparation (null eviction/ string to number transfer with dictionary/ ...) and prepares all the dataframes, 

### "Train and Test", 
which uses the dataframes to train a [HistogramGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) from scikit learn (or any other model) and displays the model test score and training time, 

### "Import Model and Reuse", 
which imports a pickled (exported) model and 

### "Dump and Export", 

which dumps the last trained algorithm into a pickle file, uses the model to predict the challenges test.csv and stores both in the project directory.

## Produce your own model
To produce your own model with the given dataset, it should suffice to clone the repo, install the necessary modules using pip (see imports in the notebook) and then execute the first and second cell in that order. <i>This is an older, unmaintaned notebook, correctness or functionality are neither guaranteed nor looked after.</i>

## What could be improved?

### Data Analysis
The missing further data analysis was most likely hindering the models performance, if we assume that the data set was not already checked for relevant corellation patterns, which it usually is not. Even just a simple analysis of the data correlation might have helped to mitigate datapoints that do not show any significance to the problem at hand, therefore improving the overall performance of the resulting model. Finding hidden data links or discovering new corellating data points from within the data set <i>(eg: name: "Mr. ..." -> reveals classifiable gender)</i> is unlikely due to the bare bones nature of all of the data points.

### Deeper Model Analysis
The choice for the HistogramGradientBoostingRegressor was mostly a result of its quick training times and promising result and less of an educated decision. With more time and effort, one might be able to construct an ideal Machine Learning Model with either less or more complexity, or find a better fitting model within the scikit learn module.
