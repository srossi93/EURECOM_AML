# Algorithmic Machine Learning
This repository contains the Jupyter Notebooks for the Algorithmic Machine Learning Course at EURECOM (Spring 2017). All the work has been jointly performed by [Simone Rossi](https://github.com/srossi93) and [Matteo Romiti](https://github.com/MatteoRomiti).

## Objectives of the course
The goal of this course is mainly to offer students to gain hands-on experience on Data Science projects. It nicely merges the theoretical concepts students can learn in our courses on machine learning and statistical inference, and systems concepts we teach in distributed systems.

Notebooks require to address several challenges, that can be roughly classified in:

* Data preparation and cleaning
* Building descriptive statistics of the data
* Working on a selected algorithm, e.g., for building a statistical model, for running parallel Monte Carlo simulations and so on...
* Working on experimental validation

## Technical notes
Students will use the EURECOM cloud computing platform to work on Notebooks. Our cluster is managed by [Zoe](http://zoe-analytics.eu/), which is a container-based analytics-as-a-service system we have built. Notebooks front-end run in a user-facing container, whereas Notebooks kernel run in clusters of containers. For this course, we focus on Apache Spark kernel.

Our Notebooks are configured to use Spark Python API. This choice is motivated by the heterogeneity we expect from our students' background.
