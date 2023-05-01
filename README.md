# DataScience_Programming
Repository featuring my completed data science projects for academic, self-improvement, and recreational objectives.

## Software Requirment
1) Python(3.2 or above)
2) VScode
3) Jupyter Notebook
4) python packages:
   1) Numpy
   2) Pandas
   3) Matplotlib
   4) Tensorflow
   5) SKlearn 
   6) keras
   7) Flask
   
## Content

### Machine Learning 

1) Regression
2) Logistic and Svm
3) Decision Trees
4) Ensemble Techniques

### Deep Learning

1) Neural Networks
2) Deep Neural Networks
3) Convolution neural network
4) Recurrent neural network

### Text Mining

1) CountVectorizer
2) TfidVectorizer
3) SVD

## Projects
### 1) Water Quality Prediction(using ML and NN)

This data is Sourced by Department of the Interior, Water quality data for the Refuge collected by volunteers collected once every two weeks: Turbidity, pH, Dissolved oxygen (DO),Salinity & Temperature. Sampling will occur at designated locations in the following water bodies: the Bay, D-Pool (fishing pond), C-Pool, B-Pool and A-Pool.

Problem Statment: to categorize areas into Bay,A, B, C , D pool( which are differentiated by the water Quality). which will further detect changes in water quality early on, indicating potential contamination. This can help authorities take timely action to prevent the spread of contaminants and ensure the safety of the water supply. 

How will building this model will help: model can be trained to predict water quality based on various parameters such as pH, temperature, and turbidity.to understand the contamination of the water source.This can help in targeting resources such as testing, treatment, and monitoring efforts, to the areas that need them the most. This can save time, money, and effort in managing water quality.This can also help in reducing the incidence of water-borne diseases and ensure safe drinking water for all.

### 2) Stock Price Predicition using RNN

In this project, we used daily stock price data from Yahoo Finance to predict the 10-day closing stock price of a publicly traded company of our choice. We experimented with four different deep learning techniques: RNN, LSTM, GRU, and Conv1D.

Our results showed that all four models were able to make reasonably accurate predictions, with mean absolute errors (MAEs) ranging from around 0.5 to 1.5 (depending on the specific company and model). However, the LSTM and GRU models consistently outperformed the RNN and Conv1D models, achieving lower MAEs on average.

In particular, the LSTM model seemed to perform the best overall, consistently producing the lowest MAEs across different companies and time periods. This suggests that the LSTM architecture may be particularly well-suited for this type of time series prediction task, perhaps due to its ability to remember long-term dependencies and handle vanishing gradients.

Overall, this project demonstrates the power of deep learning techniques for predicting stock prices and highlights the importance of choosing the right model architecture for the task at hand. While further research is needed to determine the most effective techniques for different companies and time periods, our results suggest that LSTM and GRU models may be a good starting point for future exploration.

### 3) Traning a pretrained Encoder Model to predict a new alphabet

In this project, we trained a pre-trained autoencoder model on a dataset of our choice, with the goal of improving its performance on a specific task. Autoencoders are powerful unsupervised learning models that are commonly used for dimensionality reduction, data denoising, and feature extraction.

Our results showed that fine-tuning a pre-trained autoencoder can lead to significant improvements in performance on the task of interest, compared to training the model from scratch. By leveraging the pre-existing knowledge encoded in the weights of the autoencoder, we were able to achieve better accuracy and faster convergence with fewer training epochs.

Additionally, we explored different techniques for fine-tuning the autoencoder, such as changing the learning rate and freezing certain layers. We found that adjusting the learning rate can have a significant impact on the performance of the model, and that freezing the lower layers of the autoencoder can help prevent overfitting and improve generalization to new data.

Overall, this project demonstrates the potential of pre-trained autoencoder models for improving the performance of machine learning models on specific tasks. By leveraging the pre-existing knowledge encoded in the weights of the autoencoder, we can save time and resources, while achieving better accuracy and faster convergence. This opens up exciting possibilities for a wide range of applications, from image recognition to natural language processing.

### Contact Me

If you've enjoyed browsing through my portfolio or would like to discuss potential work opportunities or collaborations, please don't hesitate to reach out to me. You can view my GitHub repository for more examples of my work and skills, and you can contact me directly at my email address himarohith@gmail.com . I'm always excited to connect with new people in the industry and explore potential partnerships or projects together. Looking forward to hearing from you soon!
