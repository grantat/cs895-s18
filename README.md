# CS 895 - Big Data and Recommender Systems

This repository contains source code for the assignments completed, slides, and papers read for the course CS 895 at ODU Spring 2018.

## Goals

The goals of this class is for students to be introduced to well known recommendation system models.
Each student was assigned a model and was required to present a recommender system paper that was based on their assigned model.
I was assigned the Autoencoder model.

The source code in this repository is meant to look at different implementations of autoencoders but also apply it a Kaggle competition.

## Installing requirements

All the python requirements are in [requirements.txt](./requirements.txt).
First create a python virtualenv:

```
$ virtualenv -p python3 venv
$ . ./venv/bin/activate
```

Then install the requirements in a python virtualenv:

```
$ pip3 install -r requirements.txt
```

GPU code run on CS servers.

## Data

Data not included in this repo, but can be downloaded below:

- [Movielense 1 million ratings](https://grouplens.org/datasets/movielens/)
- [Netflix 100 Million ratings](http://academictorrents.com/details/9b13183dc4d60676b773c9e2cd6de5e5542cee9a)
- [TalkingData 240 million click records](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data)

Visit their respective README files for information on data fields provided.

## References

Papers:

- [AutoRec: Autoencoders Meet Collaborative Filtering](http://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf)
- [Training Deep AutoEncoders for Collaborative Filtering](https://arxiv.org/pdf/1708.01715.pdf)

Repositories:

- Autorec - https://github.com/mesuvash/NNRec
- DeepRecommender - https://github.com/NVIDIA/DeepRecommender

Competitions:

- [TalkingData AdTracking Fraud Detection Challenge](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection)

Slides:

- [Taken from Dr. Yaohang Li](http://www.cs.odu.edu/~yaohang/cs895/)
