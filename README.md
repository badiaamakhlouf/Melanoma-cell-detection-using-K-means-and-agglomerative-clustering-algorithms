# Clustering-with-ML 
## Objectives: 
This script is composed by two parts: 

-The first part consists on performing the agglomerative clustering on the data of chronic kidney disease and comment the results. 

-The second part aims to help medical doctors in moles analysis by analyzing the border of each mole presented in a given image using the K-means clustering algorithm of Scikit learn. 

Knowing that, doctors to diagnose melanoma moles consider not only borders but also four other features: asymmetry, color, diameter and evolution. 

After performing the K-means algorithm on each of the given image and extracting the mole area, it is required to find the corresponding contour. Also, evaluate the perimeter of a circle that has an area equal to the mole area then evaluate the ratio between the perimeter of the mole and the perimeter of the circle. 
## Dataset description: 
The provided dataset for this lab comports 54 images classified into three classes:
### Class 1: 11 images have the name low_risk_n.jpg for moles and they have a low probability of being melanoma (i.e. tumors).
### Class 2:  16 images named medium_risk_n.jpg for moles that have a low probability of be-ing melanoma.
### Class 3: 27 images of name_melanoma_n.jpg for moles that have a high probability of being melanoma.
In all the mentioned cases n was an integer. 
## The K-means algorithm
It is one of the iterative clustering algorithms, which consists on splitting the original da-taset into K clusters. Samples are assigned to each cluster based on the minimum distance between each sample and the cluster’s centroid. The centroid is updated for each iteration till a stop condition is verified. In this work, the cluster number was chosen 3 and the obser-vations or samples were the pixels of the image. 

For each iteration, the K-means tests hundreds of different initial vectors to select the best clustering among the obtained results. The best clustering is when the initial vector has pre-sented the minimum moment of inertia. The performance of K-means or the best solution strictly depends on the target features. 
