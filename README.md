# Attention-based Models for Snow-Water Equivalent Prediction
## Krishu K Thapa<sup>1</sup>, Bhupinderjeet Singh<sup>1</sup>, Supriya Savalkar<sup>1</sup>, Alan Fern<sup>2</sup>, Kirti Rajagopalan<sup>1</sup>, Ananth Kalyanaraman<sup>1</sup>
### <sup>1</sup> Washington State University, Pullman, WA. <sup>2</sup> Oregon State University, Corvallis, OR.

Predicting the SWE value for multiple SNOTEL locations in the Western US using the Attention Models


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


## Citation

If you use our idea in your research, please cite:

Thapa, Krishu & Singh, Bhupinderjeet & Savalkar, Supriya & Fern, Alan & Rajagopalan, Kirti & Kalyanaraman, Ananth. (2024). Attention-Based Models for Snow-Water Equivalent Prediction. Proceedings of the AAAI Conference on Artificial Intelligence. 38. 22969-22975. 10.1609/aaai.v38i21.30337. https://doi.org/10.1609/aaai.v38i21.30337
