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
        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}
        
        self._add_unk = add_unk
        self._unk_token = unk_token
        
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token) 
        
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
        return "<Vocabulary(size=%d)>" % len(self)

    # return the length of token_to_idx
    def __len__(self):
        return len(self._token_to_idx)

# Create the Surname Vectorizer Class
class SurnameVectorizer(object):
    # Create the init function to instantiate class
    def __init__(self, surname_vocab, nationality_vocab, max_surname_length):
        self.surname_vocab = surname_vocab
        self.nationality_vocab = nationality_vocab
        self._max_surname_length = max_surname_length

    # Create One Hot Matrix
    def vectorize(self, surname):

    # Instantiate the vectorizer from the dataset dataframe
    @classmethod
    def from_dataframe(cls, surname_df):
        surname_vocab = Vocabulary(unk_token="@")
        nationality_vocab = Vocabulary(add_unk=False)
        max_surname_length = 0

    # Get the data from the from_serializable dictionary
    @classmethod
    def from_serializable(cls, contents):

    # serialize dictionary
    def to_serializable(self):

# Load the data via the Surname Dataset class
class SurnameDataset(Dataset):
    # Create the init function to instantiate class
    def __init__(self, surname_df, vectorizer):
        self.surname_df = surname_df
        self._vectorizer = vectorizer
        self.train_df = self.surname_df[self.surname_df.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.surname_df[self.surname_df.split=='val']
        self.validation_size = len(self.val_df)

        self.test_df = self.surname_df[self.surname_df.split=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')
        
        # Class weights
        class_counts = surname_df.nationality.value_counts().to_dict()
        def sort_key(item):
            return self._vectorizer.nationality_vocab.lookup_token(item[0])
        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

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
        row = self._target_df.iloc[index]

        surname_matrix = \
            self._vectorizer.vectorize(row.surname)

        nationality_index = \
            self._vectorizer.nationality_vocab.lookup_token(row.nationality)

        return {'x_surname': surname_matrix,
                'y_nationality': nationality_index}

    # Given a batch size, return the number of batches in the dataset
    def get_num_batches(self, batch_size):

# Create the Surname Classifier class with a CNN
class SurnameClassifier(nn.Module):
    # Create the init function to instantiate class
    def __init__(self, initial_num_channels, num_classes, num_channels):
        super(SurnameClassifier, self).__init__()
        
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=initial_num_channels, 
                      out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, 
                      kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, 
                      kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels, 
                      kernel_size=3),
            nn.ELU()
        )
        self.fc = nn.Linear(num_channels, num_classes)

    # The forward pass of the classifier
    def forward(self, x_surname, apply_softmax=False):

# A generator function which wraps the PyTorch DataLoader. It will ensure each tensor is on the write device location.   
def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):

# Define the training parameter
def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': args.model_state_file}

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
def predict_nationality(surname, classifier, vectorizer):
    vectorized_surname = vectorizer.vectorize(surname)
    vectorized_surname = torch.tensor(vectorized_surname).unsqueeze(0)
    result = classifier(vectorized_surname, apply_softmax=True)

# Predict the top K nationalities from a new surname
def predict_topk_nationality(surname, classifier, vectorizer, k=5):    

# Start Main path of the Training
def main():
    # Some more Parameters to declare upfront
    args = Namespace(
        # Data and Path information
        surname_csv="data/surnames/surnames_with_splits.csv",
        vectorizer_file="vectorizer.json",
        model_state_file="model.pth",
        save_dir="model_storage/ch4/cnn",
        # Model hyper parameters
        hidden_dim=100,
        num_channels=256,
        # Training hyper parameters
        seed=1337,
        learning_rate=0.001,
        batch_size=128,
        num_epochs=100,
        early_stopping_criteria=5,
        dropout_p=0.1,
        # Runtime options
        cuda=False,
        reload_from_files=False,
        expand_filepaths_to_save_dir=True,
        catch_keyboard_interrupt=True
        )


    # Create the File Paths for saving
    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)

        args.model_state_file = os.path.join(args.save_dir, args.model_state_file)
        
        print("Expanded filepaths: ")
        print("\t{}".format(args.vectorizer_file))
        print("\t{}".format(args.model_state_file))
        
    # Check CUDA
    if not torch.cuda.is_available():
        args.cuda = False

    args.device = torch.device("cuda" if args.cuda else "cpu")
    print("Using CUDA: {}".format(args.cuda))

    # Set the seed for the model
    def set_seed_everywhere(seed, cuda):

    # Create dir paths if necessary      
    def handle_dirs(dirpath):
            
    # Set seed for reproducibility

    # handle dirs

    # training from a checkpoint
    if args.reload_from_files:
        dataset = SurnameDataset.load_dataset_and_load_vectorizer(args.surname_csv, args.vectorizer_file)
    # create dataset and vectorizer
    else:
        dataset = SurnameDataset.load_dataset_and_make_vectorizer(args.surname_csv)
        dataset.save_vectorizer(args.vectorizer_file)

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
    classifier.load_state_dict(torch.load(train_state['model_filename']))

    classifier = classifier.to(args.device)
    dataset.class_weights = dataset.class_weights.to(args.device)
    loss_func = nn.CrossEntropyLoss(dataset.class_weights)

    # Use the test dataset to validate

        # compute the output
        # compute the loss
        # compute the accuracy

    # Try out the model

if __name__ == "__main__":
    main()
