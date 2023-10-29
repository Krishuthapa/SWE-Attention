# Attention-based Models for Snow-Water Equivalent Prediction
## Krishu K Thapa^1, Bhupinderjeet Singh^1, Supriya Savalkar^1, Alan Fern^2, Kirti Rajagopalan^1, Ananth Kalyanaraman^1
### ^1 Washington State University, Pullman, WA. ^2 Oregon State University, Corvallis, OR.
####{krishu.thapa, bhupinderjeet.singh, supriya.savalkar}@wsu.edu, alan.fern@oregonstate.edu, {kirtir, ananth}@wsu.edu

# SWE_Prediction_Attention
Predicting the SWE value for multiple SNOTEL locations in Western US using the Attentnion Models


## Model Folder

- Spatial_Attention.py - This file has the code for the spatial attention implementation along with training and testing.
                          The data is loaded from the ```SDL.py``` file inside the ```DataLoader``` folder.

- Temporal_Attention.py - This file has the code for the temporal attention implementation along with training and testing.
                          The data is loaded from the ```TDL.py``` file inside the ```DataLoader``` folder.


## DataLoader

- Data: This has all the data we have used in our model implementation for the ```SNOTEL``` locations.
- SDL.py: This is the data loader file for the spatial model. It returns the training and testing data for the Spatial Attention model.
- TDL.py: This is the data loader file for the temporal model. It returns the training and testing data for the Temporal Attention model.
- feature_prep.py: This file processes all the raw data and generates the processed csv files of data which are used by the data loaders for
                    spatial and attention model.
