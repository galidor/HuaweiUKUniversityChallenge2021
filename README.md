# HuaweiUKUniversityChallenge2021
***Team MatchLab submission to Huawei UK University Challenge 2021***

## Authors
<b>Team Name:</b> MatchLab

<b>Team members</b>:
* Axel Barroso Laguna, Imperial College London, Email: axel.barroso17@imperial.ac.uk
* Mikolaj Jankowski, Imperial College London, Email: mikolaj.jankowski17@imperial.ac.uk
* Michal Nazarczuk, Imperial College London, Email: michal.nazarczuk17@imperial.ac.uk
## Method
### Task 1
* MAC address vendor lookup and removing invalid APs (probably mobile hotspots)
* Removing rare MAC addresses (less than 10 occurrences)
* Removing outlier samples in the training set (>100m between fingerprints)
* Input to the model: pairs of signal powers for common MAC addresses (up to 10) for both fingerprints, MAC similarity index indicating the total number of common MAC addresses, absolute difference between fingerprint indices
* Model: 5-layer NN with linear layers, PReLU activations and batch normalization layers
* Trained for 10 epochs with batch size equal to 1024, SGD optimizer and sum of L1 and MSE as a loss function
### Task 2
* Initialize clusters from elevations (all trajectories between fingerprints specified in elevations considered as one of the initial clusters)
* Merge clusters based on threshold of estimated WiFi distance between closest points in two clusters
## Replicating the results
To replicate our results it is advised to use the newest versions of PyTorch, Pandas, Numpy, and tqdm.
For task 1 simply run ```python train_task1.py``` and it will produce a submission file called my_submission.csv. For task 2, run ```python elevation_clustering_task2.py```, which produces file submission.csv.
