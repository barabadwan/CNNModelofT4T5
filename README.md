# Modeling Drosophila visual pathway with Deep Neural Nets
A number of deep neural net architectures to represent the visual pathway of Drosophila Melanogaster.

## Dataset
Training is conducted on database of natural scenes that have been rotated at different velocities 
See load_data_rr function in utils for details on train/test split 

Train data consists of pixel values for the images in four dimensional (Samples, Time, Space, 1)\
Test data is consists of the set velocities the images were rotated with at each time point (Samples, Time, 1)

## How to run:
1) Relevant scripts found in src folder 
2) Open ModelSetup.py and make sure they data paths are configured correctly
3) Choose model types and plotting options 


