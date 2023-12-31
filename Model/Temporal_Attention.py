#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

import random

import pandas as pd
import numpy as np

import os
import sys
sys.path.append(os.getcwd())

import hydroeval as he

import math

import time

from DataLoader.TDL import *


current_directory = os.getcwd()
data_import_path = current_directory + "/DataLoader/Data/"

snotel_locations_info = pd.read_csv('{}Snotel_Locations_Filtered_v3.csv'.format(data_import_path))
snotel_locations_info['Southness'] = [ math.sin(value['Slope_tif1_x']) * math.cos(value['Aspect_tif_x']) for index,value in snotel_locations_info.iterrows()]


tp_dataLoader = TemporalDataLoader()

tp_dataLoader.cleanDataJunk()
tp_dataLoader.prepareInputsAndOutputs()

tp_training_input , tp_training_output = tp_dataLoader.getTrainingData()
tp_testing_data_1,tp_testing_data_2, tp_testing_data_3 , tp_testing_data_4,tp_testing_data_5 = tp_dataLoader.getTestingData()


tp_testing_input_1 , tp_testing_output_1 = tp_testing_data_1
tp_testing_input_2 , tp_testing_output_2 = tp_testing_data_2
tp_testing_input_3 , tp_testing_output_3 = tp_testing_data_3
tp_testing_input_4 , tp_testing_output_4 = tp_testing_data_4
tp_testing_input_5 , tp_testing_output_5 = tp_testing_data_5


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        batch_size, seq_len , embedding_dim = x.shape
        
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        
        seq_len , batch_size, embedding_dim = x.shape
        
        return x.reshape(batch_size,seq_len,embedding_dim)

    
class SWETransformer(nn.Module):
    def __init__(self, encoder_input_dim = 20, model_dim = 64, n_output_heads = 1,window = 3,
                  seq_length = 10):
        super().__init__()
        
        # Storing the passed argument on the class definition.
        
        self.model_dim = model_dim
        self.encoder_input_dim = encoder_input_dim
        self.n_output_heads = n_output_heads
        self.seq_length = seq_length
        
        # Linear Layers for the input.
        
        self.input_embed_1 = torch.nn.Linear(self.encoder_input_dim , int(self.model_dim/2))
        self.input_embed_2 = torch.nn.Linear(int(self.model_dim/2) , self.model_dim)

        self.input_dropout = torch.nn.Dropout(p = 0.10)
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(self.model_dim,0.2)
           
        # Transformer model definition.
        
        self.encoder_layer_1 = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=16)
        self.transformers_1 = torch.nn.TransformerEncoder(self.encoder_layer_1 , num_layers = 24)
        
        # Dimension Reduction.
        
        initial_dim  = self.model_dim
        
        self.dim_red_1 = torch.nn.Linear(initial_dim * 2 , int(initial_dim/2))
        self.dim_red_2 = torch.nn.Linear(int(initial_dim/2) , int(initial_dim/2))
        self.dim_red_3 = torch.nn.Linear(int(initial_dim/2) , int(initial_dim/4))
        self.dim_red_4 = torch.nn.Linear(int(initial_dim/4) , int(initial_dim/8))
        
        self.dim_red_dropout = torch.nn.Dropout(p = 0.05)
        
        # Final output layer for the model.
        
        self.decoder_layer_1 = torch.nn.Linear(self.seq_length * int(self.model_dim/8) ,self.seq_length * int(self.model_dim/16))    
        self.decoder_layer_2 = torch.nn.Linear(self.seq_length * int(self.model_dim/16), self.seq_length)
        
        # Activation Functions
        
        self.activation_relu = torch.nn.ReLU()
        self.activation_identity = torch.nn.Identity()
        self.activation_gelu = torch.nn.GELU()
        self.activation_tanh = torch.nn.Tanh()
        self.activation_sigmoid = torch.nn.Sigmoid()
        
        # Dropout Functions 
        
        self.dropout_5 = torch.nn.Dropout(p = 0.05)
        self.dropout_10 = torch.nn.Dropout(p = 0.10)
        self.dropout_15 = torch.nn.Dropout(p = 0.15)
        self.dropout_20 = torch.nn.Dropout(p = 0.20)
        
        
    def forward(self,encoder_inputs):
        
        # Converting to the torch array.
        
        encoder_inputs = torch.from_numpy(encoder_inputs).to(dtype= torch.float32)
                
        # Getting the configuration of the passed data.
        
        encoder_batch_size, encoder_sequence_length , encoder_input_dim = encoder_inputs.shape
                
        # Embedding the daily data passed to the model for the locations.
        
        
        embed_input_x = encoder_inputs.reshape(-1,self.encoder_input_dim)
        
        embed_input_x = self.input_embed_1(embed_input_x)
        embed_input_x = self.activation_gelu(embed_input_x)
        
        embed_input_x = self.input_embed_2(embed_input_x)
        embed_input_x = self.activation_gelu(embed_input_x)
        
        embed_input_x = embed_input_x.reshape(encoder_batch_size, encoder_sequence_length, self.model_dim)
        
        # Applying positional encoding.
        
        x = self.positional_encoding(embed_input_x)
        
        # Applying the transformer layer.

        x = self.transformers_1(x)
        
        x = x.reshape(-1, self.model_dim)
        embed_input_x = embed_input_x.reshape(-1,self.model_dim)
        
        x = torch.cat((x , embed_input_x),1)
        
        # Dim reduction layer.
        
        x = self.dim_red_1(x) 
        x= self.dropout_20(x)
        
        x = self.dim_red_2(x)
        x = self.activation_relu(x)
        x= self.dropout_20(x)

        x = self.dim_red_3(x)
        x= self.dropout_20(x)
        
        x= self.dim_red_4(x)
        x = self.activation_relu(x)
        x= self.dropout_20(x)
        
        
        
        # Final layer for the output.
        
        x = x.reshape(-1, encoder_sequence_length * int(self.model_dim/8))
        
        x= self.decoder_layer_1(x)
        x= self.activation_gelu(x)
        x = self.dropout_10(x)
        
        x = self.decoder_layer_2(x)
        x = self.activation_identity(x)
        
        x= x.reshape(encoder_batch_size , encoder_sequence_length , self.n_output_heads)
        
        return x
    


swe_model_tp = SWETransformer(encoder_input_dim = 19, model_dim = 512, n_output_heads = 1, seq_length = 270)

# Loss Function 
mean_squared_error_tp = nn.MSELoss()

# Optimizer

optimizer_tp = torch.optim.AdamW(swe_model_tp.parameters(), lr= 0.0001, weight_decay = 0.0001)
scheduler_tp = torch.optim.lr_scheduler.StepLR(optimizer_tp, step_size = 3 ,gamma = 0.6, last_epoch= -1, verbose=False)


def TrainModelTP(train_inputs, train_outputs, epoch_number, data_type= "Temporal"):
    loss_value = 0
    total_loss = 0
    total_batches = 0
    
    for input_index , batch_input in enumerate(train_inputs):
        
        total_batches +=1
        
        optimizer_tp.zero_grad()
                
        batch_size , sequence_length , feature_dim = train_outputs[input_index].shape
            
        output = swe_model_tp(batch_input)
                    
        loss = mean_squared_error_tp(output, torch.from_numpy(train_outputs[input_index]).to(dtype=torch.float32))
      
        loss_in_batch = loss.item() * batch_input.shape[0]
        
        total_loss+=loss_in_batch
        
        loss.backward()
        optimizer_tp.step()
        
        if (input_index + 1) % 1000 == 0:
            print("Obtained Output:")
            print(output )
        
            
        if  (input_index + 1) % 100 == 0:
            print('Type: {} , Epoch Number: {} => Batch number :{} , loss value : {} '.format(data_type,epoch_number,input_index+1, loss.item()))
            print("=================================================")
        
            
    print('Type:{}, Epoch Number: {} => Avg loss value : {} '.format(data_type,epoch_number, total_loss / (total_batches * 1 )))
    
    return total_loss / (total_batches * 1 )


# Training Section

swe_model_tp.train(True)

train_epoch_avg_losses_tp = []
start_time = time.time()

for index in range(12):
        
    temp_holder_tp = list(zip(tp_training_input, tp_training_output))
    random.shuffle(temp_holder_tp)
    
    train_input_batches_tp, train_output_batches_tp = zip(*temp_holder_tp)
    
    epoch_loss_tp = TrainModelTP(train_input_batches_tp, train_output_batches_tp, epoch_number = index+ 1)
    
    train_epoch_avg_losses_tp.append(epoch_loss_tp)
    
    scheduler_tp.step()

end_time = time.time()

print("Time Elapsed", end_time - start_time)


# Testing section

swe_model_tp.eval()
def TestModelTP(test_inputs, test_outputs):
    loss_value = 0
    losses = []
    
    print("========================")
    print('Testing the model.')
    print("========================")
    
    outputs = []
    outputs_loss = []
    nse_values = []
    
    mse_tp = nn.MSELoss()
    
    with torch.no_grad():
        for input_index , batch_input in enumerate(test_inputs):
            
            batch_size, sequence_length , output_feature_dim = test_outputs[input_index].shape
            
            #SWE_Model.transformers_1.apply(lambda module: setattr(module, "training", True))
            
            output = swe_model_tp(batch_input)
                        
            loss = mse_tp(output[0], torch.from_numpy(test_outputs[input_index]).to(dtype=torch.float32))
            
            outputs.append(output[0].detach().numpy())
            outputs_loss.append(loss.item())
            
            
            nse = he.evaluator(he.nse, output[0].reshape(sequence_length,1), np.array(test_outputs[input_index].reshape(sequence_length, output_feature_dim)))
            
            nse_values.append(nse)
            
            print('Location :{} , loss value : {}, nse:{} '.format(snotel_locations_info['Station Name'].values[input_index], loss.item(),nse))
            
            losses.append(loss.item())
    
    return (np.array(outputs), np.array(outputs_loss))
            

test_outputs_1_tp, test_losses_1_tp = TestModelTP(tp_testing_input_1,tp_testing_output_1)
test_outputs_2_tp, test_losses_2_tp = TestModelTP(tp_testing_input_2,tp_testing_output_2)
test_outputs_3_tp, test_losses_3_tp = TestModelTP(tp_testing_input_3,tp_testing_output_3)
test_outputs_4_tp, test_losses_4_tp = TestModelTP(tp_testing_input_4,tp_testing_output_4)
test_outputs_5_tp, test_losses_5_tp = TestModelTP(tp_testing_input_5,tp_testing_output_5)
