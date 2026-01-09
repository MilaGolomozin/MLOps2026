Diffusion models are a class of generative models that progressively introduce noise, typically Gaussian, to data and learn the reverse-time dynamics of a Markov chain. This framework provides significant flexibility allowing diffusion models to capture diverse and intricate data patterns. [1] Their capacity to model complex data distributions supports their ability to generate high-quality images and audio, contributing to their rapid adoption and popularity in generative AI research. [2] Throughout the course of this project, the authors have chosen to employ Variational Diffusion Models (VDMs). VDMs differ from the original diffusion model architecture by generalizing the whole diffusion process to the continuous time domain. Moreover, VDMs have extended beyond the fixed linear noise schedule used in DDPMs by employing a dynamic noise schedule, which is optimized jointly with the rest of the model. [3]


Throughout the project the authors will work on training a VDM  on the Pokemon dataset outlined below in order to ultimately be able to generate new Pokemon characters using the trained model. 


The initial dataset for this project will be the Pokemon Sprites Images dataset from Kaggle. It contains 10,437 Pokemon sprite images in 96x96 resolution. The set includes many variations of different Pokemon species in different versions, such as shiny or normal. The key characteristics of this dataset are the image type, resolution, and coverage. The image type is a static 2D sprite image in RGB format, while the resolution is 96x96 pixels per image. The dataset is split into 1070 directories for each Pokemon type. Nearly 898 different Pokemon sprites are included in standard and alternate versions. The dataset will be split into training, validation, and test subsets to support model development and assessment.


PyTorch will be the deep learning framework used for the implementation of the model. The framework will be used for data handling and preprocessing, definition of the model’s architecture as well as its training and evaluation. PyTorch Profiling is going to assist with detecting and visualizing bottlenecks in the code. Docker will be used for the deployment part in order to ensure the producibility of the project. Weights and Biases (wandb) will track experiments, monitor metrics, and will help with sharing the results and visualizations among members. We will use the implementation provided in the GitHub repository below as a starting point for our project, to ensure a baseline model. Over the following weeks we will gradually be building upon it to establish a strong and robust pipeline.  






References:

[1] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli, “Deep unsupervised learning using nonequilibrium thermodynamics”, 2015

[2] Dave Bermann, “What are diffusion models?,” IBM, 2025

[3] Diederik P. Kingma, Tim Salimans, Ben Poole, and Jonathan Ho, “Variational diffusion models”, 2021
