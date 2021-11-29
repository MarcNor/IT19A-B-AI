# Manage Imports
from argparse import Namespace
from collections import Counter
import json
import os
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook

# Create the Vocabulary Class to process text and extract vocabulary for mapping
class Vocabulary(object):
    # Create the init function to instantiate class
    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        
    # serialize dictionary 
    def to_serializable(self):

    # instantiates the Vocabulary from a serialized dictionary
    @classmethod
    def from_serializable(cls, contents):

    # Update mapping dicts based on the token and give it an index
    def add_token(self, token):
    
    # Add a list of tokens into the Vocabulary
    def add_many(self, tokens):

    # Retrieve the index associated with the token or the UNK index if token isn't present.
    def lookup_token(self, token):

    # Return the token associated with the index
    def lookup_index(self, index):

    # Return the size of the Vocabulary as string
    def __str__(self):

    # return the length of token_to_idx
    def __len__(self):

# Create the Surname Vectorizer Class
class SurnameVectorizer(object):
    # Create the init function to instantiate class
    def __init__(self, surname_vocab, nationality_vocab, max_surname_length):

    # Create One Hot Matrix
    def vectorize(self, surname):

    # Instantiate the vectorizer from the dataset dataframe
    @classmethod
    def from_dataframe(cls, surname_df):

    # Get the data from the from_serializable dictionary
    @classmethod
    def from_serializable(cls, contents):

    # serialize dictionary
    def to_serializable(self):

# Load the data via the Surname Dataset class
class SurnameDataset(Dataset):
    # Create the init function to instantiate class
    def __init__(self, surname_df, vectorizer):

    # Load dataset and make a new vectorizer from scratch
    @classmethod
    def load_dataset_and_make_vectorizer(cls, surname_csv):

    # Load dataset and the corresponding vectorizer. Used in the case in the vectorizer has been cached for re-use
    @classmethod
    def load_dataset_and_load_vectorizer(cls, surname_csv, vectorizer_filepath):

    # A static method for loading the vectorizer from file
    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):

    # Saves the vectorizer to disk using json
    def save_vectorizer(self, vectorizer_filepath):

    # Returns the vectorizer 
    def get_vectorizer(self):

    # Selects the splits in the dataset using a column in the dataframe
    def set_split(self, split="train"):

    # Returns the target_size
    def __len__(self):

    # The primary entry point method for PyTorch datasets to get the items
    def __getitem__(self, index):

    # Given a batch size, return the number of batches in the dataset
    def get_num_batches(self, batch_size):

# Create the Surname Classifier class with a CNN
class SurnameClassifier(nn.Module):
    # Create the init function to instantiate class
    def __init__(self, initial_num_channels, num_classes, num_channels):

    # The forward pass of the classifier
    def forward(self, x_surname, apply_softmax=False):

# A generator function which wraps the PyTorch DataLoader. It will ensure each tensor is on the write device location.   
def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):

# Define the training parameter
def make_train_state(args):

# Handle training state updates
def update_train_state(args, model, train_state):
    # Save one model at least

    # Save model if performance improved

        # If loss worsened
            # Update step
        # Loss decreased
            # Save the best model

            # Reset early stopping step

        # Stop early ?

# Calculate Accuracy
def compute_accuracy(y_pred, y_target):

# Predict the nationality from a new surname

# Predict the top K nationalities from a new surname
def predict_topk_nationality(surname, classifier, vectorizer, k=5):    

# Start Main path of the Training
def main():
    # Some more Parameters to declare upfront

    # Create the File Paths for saving
        
    # Check CUDA

    # Set the seed for the model
    def set_seed_everywhere(seed, cuda):

    # Create dir paths if necessary      
    def handle_dirs(dirpath):
            
    # Set seed for reproducibility

    # handle dirs

    # training from a checkpoint
    # create dataset and vectorizer

    # Get the vectorizer from the dataset    

    # Initalize Classifier

    # Create Loss Function, Optimizer and Scheduler

    # Split between train and validation

    # Start training
                # the training routine is these 5 steps:

                # --------------------------------------
                # step 1. zero the gradients
                # step 2. compute the output
                # step 3. compute the loss
                # step 4. use loss to produce gradients
                # step 5. use optimizer to take gradient step
                # -----------------------------------------
                # compute the accuracy

            # Iterate over val dataset

                # compute the output

                # step 3. compute the loss

                # compute the accuracy

    # Load the saved classifier

    # Use the test dataset to validate

        # compute the output
        # compute the loss
        # compute the accuracy

    # Try out the model

if __name__ == "__main__":
    main()
