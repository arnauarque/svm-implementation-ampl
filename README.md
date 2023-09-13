# Optimizing SVM Classification: Implementation and Validation in AMPL

This task involves implementing the primal and dual quadratic formulations of Support Vector Machines (SVM) in AMPL, with a focus on a Soft-Margin SVM. The project includes theoretical background, code implementation, validation with generated data, and experimentation with an additional dataset.

## Authors and context

This project has been developed by the users [@arnauarque](https://github.com/arnauarque) and [@danielesquina](https://github.com/danielesquina) as part of the [Optimization Techniques for Data Mining](https://www.fib.upc.edu/en/studies/masters/master-data-science/curriculum/syllabus/OTDM-MDS) course in the [Master's in Data Science](https://www.fib.upc.edu/en/studies/masters/master-data-science) program at the [Faculty of Informatics of Barcelona](https://www.fib.upc.edu/en) (Universitat Politècnica de Catalunya). In this file, you can find an introduction to the project and its objectives. Additionally, you will find a detailed description of the repository's organization.

## Summary of the requirements

This project complies with the following requirements:

1. Implement the primal and dual quadratic formulation of the Support Vector Machine in AMPL.
2. Apply to a dataset obtained with the accompanying generator. Validate the SVM with data different from that of the training set.
3. Optionally (but highly recommended) apply it to other datasets.
4. Compute the separation hyperplane from the dual model and check that it coincides with that of the primal model.

## Overall description

We start this project with a theoretical introduction to the primal and dual formulations of SVMs, outlining their key similarities and differences.

Next, we provide a detailed description of the AMPL implementations we propose for both formulations.

To avoid the need for manual executions using the AMPL terminal, we have described and generated a Python script that allows us to perform executions with different parameterizations and datasets more conveniently and intuitively.

Subsequently, we conducted experiments using the generated implementations with both a synthetic dataset and a real-world Sonar dataset. This enabled us to conclude that the proposed implementations appear to be correct and exhibit coherent behavior.

## Repository organization

This repository is organized as follows: 

- The [code](code/) directory contains both the [data](code/data/) used and the code ([primal](./code/primal/) and [dual](./code/dual/) formulations) generated in this project.
 - The [report.pdf](report.pdf) file contains a comprehensive introduction, objectives, results and conclusions of the project.

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository is licensed under the MIT License.



