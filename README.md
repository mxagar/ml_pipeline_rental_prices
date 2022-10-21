# Reproducible Machine Learning Pipeline for Short-Term Rental Price Prediction

This repository contains a Machine Learning (ML) pipeline which is able to predict short-term rental property prices in New York City. The pipeline is designed to the retrained easily with new data that comes frequently in bulk, assuming that prices (and thus, the model) vary constantly. The pipeline is divided into the typical steps or components in an ML training pipeline, carried out in order, and explained in the section [Introduction](#introduction).

The following tools are used:

- [MLflow](https://www.mlflow.org) for reproduction and management of pipeline processes.
- [Weights and Biases](https://wandb.ai/site) for artifact and execution tracking.
- [Hydra](https://hydra.cc) for configuration management.
- [Conda](https://docs.conda.io/en/latest/) for environment management.
- [Pandas](https://pandas.pydata.org) for data analysis.
- [Scikit-Learn](https://scikit-learn.org/stable/) for data modeling.

The starter code of the repository was originally forked from a project in the Udacity repository [build-ml-pipeline-for-short-term-rental-prices](https://github.com/udacity/build-ml-pipeline-for-short-term-rental-prices). The instructions of that source project can be found in the file [`Instructions.md`](Instructions.md). If you would like to know more about *why* reproducible ML pipelines matter and *how* the tools used here interact, you can have a look at my [ML pipeline project boilerplate](https://github.com/mxagar/music_genre_classification).

The used dataset is a ...

Table of contents:

- [Reproducible Machine Learning Pipeline for Short-Term Rental Price Prediction](#reproducible-machine-learning-pipeline-for-short-term-rental-price-prediction)
  - [Introduction](#introduction)
  - [How to Use This Project](#how-to-use-this-project)
  - [Dependencies](#dependencies)
  - [Notes](#notes)
  - [Possible Improvements](#possible-improvements)
  - [Interesting Links](#interesting-links)
  - [Authorship](#authorship)

## Introduction

## How to Use This Project

- [`Instructions.md`](Instructions.md)
- [ML pipeline project boilerplate](https://github.com/mxagar/music_genre_classification).

## Dependencies

## Notes

## Possible Improvements

## Interesting Links

- [ML pipeline project boilerplate](https://github.com/mxagar/music_genre_classification).
- This repository doesn't focus on the techniques for data processing and modeling; if you are interested in those topics, you can visit my  [Guide on EDA, Data Cleaning and Feature Engineering](https://github.com/mxagar/eda_fe_summary).
- This project creates an inference pipeline managed with [MLflow](https://www.mlflow.org) and tracked with [Weights and Biases](https://wandb.ai/site); however, it is possible to define a production inference pipeline in a more simple way without the exposure to those 3rd party tools. In [this blog post](https://mikelsagardia.io/blog/machine-learning-production-level.html) I describe how to perform that transformation from research code to production-level code; the associated repository is [customer_churn_production](https://github.com/mxagar/customer_churn_production).
- If you are interested in more MLOps-related content, you can visit my notes on the [Udacity Machine Learning DevOps Engineering Nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821): [mlops_udacity](https://github.com/mxagar/mlops_udacity).
- [Weights and Biases tutorials](https://wandb.ai/site/tutorials).
- [Weights and Biases documentation](https://docs.wandb.ai/).

## Authorship

Mikel Sagardia, 2022.  
No guarantees.

If you find this repository helpful and use it, please link to the original source.
