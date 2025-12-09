# Optimizing GANs with Quantum Computation
## Authors: Adam Haile & Drake Cofta
### CSC 4631-121 Final Project

## Project Overview
Our project aims to generate realistic handwritten digits by learning the distribution of the UCI Handwritten Digits 
dataset and comparing a classical GAN (Generative Adversarial Network) with a conditional quantum-based generative model. 

The motivation is to explore whether quantum circuit can reduce the computational cost 
and complexity required for generative modeling (as they can represent high-dimensional data with fewer parameters). 

This problem is relevant as modern generative AI demands heavy compute, 
and quantum approaches may offer a more efficient alternative. 

A Standard GAN provides the classical benchmark and necessary context, since its adversarial structure is the 
current state-of-the-art method for modeling complex pixel distributions.

Overall, we were able to successfully implement both a classical GAN and a quantum GAN, but our findings show that 
Standard GANs still outperform Quantum GANs in terms of image quality and training stability for this specific task. 
However, our results indicate that Quantum GANs have potential for future success, especially as quantum hardware 
improves. What a time to be in the Machine Learning field!

## Repository Structure
- `Standard_GAN.ipynb`: Contains code and resources for the classical GAN implementation and its evaluation metrics.
- `Quantum GAN.ipynb`: Contains code and resources for the quantum GAN implementation and its evaluation metrics. 
  - This also includes combined results and comparisons with the classical GAN.
- `qcircuit.py`: Python module defining the quantum circuit architecture used in the Conditional Quantum GAN.
- `qgenerator.py`: Python module implementing the quantum generator model for the Conditional Quantum GAN.
- `optdigits.tra`: Dataset file containing the UCI Handwritten Digits data used for training and evaluation for the CQ-GAN.

The UCI Handwritten Digits problem was implemented both in `Standard_GAN.ipynb` and `Quantum GAN.ipynb`, allowing for direct comparison of the two approaches.

## Instructions to Run the Code
1. Ensure you have Python 3.x installed along with the required libraries. Run: 
    ```
    pip install math numpy torch ucimlrepo torchmetrics matplotlib scikit-learn scipy pennylane sys time torchvision torchmetrics[image] pennylane-lightning
    ```
2. Run the Jupyter notebooks in order:
    - Start with `Standard_GAN.ipynb` to see the classical GAN implementation.
    - Follow with `Quantum GAN.ipynb` to explore the quantum GAN implementation and comparisons.

## Interpreting the Results
- `Standard_GAN.ipynb` will provide insights into the performance of the classical GAN, including generated images and evaluation metrics.
- `Quantum GAN.ipynb` will showcase the quantum GAN's performance, generated images, and a comparative analysis with the classical GAN.
- 'Loss Curves' sections in both notebooks will help visualize the quality of training.
- 'Mode-Collapse Check' and 'Judgement Test' sections in both notebooks will help assess the diversity and realism of generated images.
- 'Latent Space Interpolation' sections in both notebooks will illustrate how smoothly the models can transition between different digit representations.
- 'Inception Score' (quality & diversity index; higher is better) and 'Fréchet Inception Distance' (distance between real and generated images; lower is better) metrics sections in both notebooks will provide quantitative measures of image quality.
- Standard GAN quality vs. # of parameters, quality vs. training time, and speed of generation are found in `Standard_GAN.ipynb` for further analysis.
- Combined results and comparisons between the two models (quality vs. size, training time, and speed of generation) are found in `Quantum GAN.ipynb`.

Across the board, we saw that the Standard GAN outperformed the Quantum GAN in terms of image quality and training stability for this specific task.
- This is highlighted by Standard GANs having a stable loss curve, higher Inception Scores, and lower Fréchet Inception Distances compared to the Quantum GAN.
- We also observed that the Standard GAN generated images that were more realistic and diverse, as evidenced by the mode-collapse checks and judgement tests.
- In addition, the training time for the Standard GAN was generally shorter, and it produced images much faster than the Quantum GAN.
- However, the Quantum GAN showed potential for future success, especially as quantum hardware and algorithms improve as they may eventually offer advantages in representing high-dimensional data with fewer parameters.