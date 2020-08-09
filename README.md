# Master Thesis Project
This repository contains the code of the project of the master thesis whose title is "Comparative study between the generative capacities of GAN, VAEand VAE-GAN".

The dataset CELEBA should be placed as a zip file of images with shape 48x48x3 in the Data folder

The sources folder has the following subfolders:
  - Common: Contains file python with methods used by all models
  - Evaluation: Contains FID, IS and P&R python files
  - GAN: Contains the implementation of GAN for the MNIST and CELEBA datasets
  - VAE: Contains the implementation of VAE for the MNIST and CELEBA datasets
  - VAEGAN: Contains the implementation of VAEGAN for the MNIST and CELEBA datasets

For the Precision & Recall metric, the implementation of the authors is used directly from the repository titled "Assessing Generative Models via Precision and Recall
". The pre-trained inception network given is this repository should be placed in Sources/Evaluation

The jupyter notebook projectLuncher allows to lunch the scripts in google collab which works directly with google drive. The output results of meaningful cells are shown in this file.
