# Manage Imports

# Create the Vocabulary Class to process text and extract vocabulary for mapping
    # Create the init function to instantiate class
        
    # serialize dictionary 

    # instantiates the Vocabulary from a serialized dictionary

    # Update mapping dicts based on the token and give it an index
    
    # Add a list of tokens into the Vocabulary

    # Retrieve the index associated with the token or the UNK index if token isn't present.

    # Return the token associated with the index

    # Return the size of the Vocabulary as string

    # return the length of token_to_idx


# Create the Surname Vectorizer Class
    # Create the init function to instantiate class

    # Create One Hot Matrix

    # Instantiate the vectorizer from the dataset dataframe

    # Get the data from the from_serializable dictionary

    # serialize dictionary


# Load the data via the Surname Dataset class
    # Create the init function to instantiate class

    # Load dataset and make a new vectorizer from scratch

    # Load dataset and the corresponding vectorizer. Used in the case in the vectorizer has been cached for re-use

    # A static method for loading the vectorizer from file

    # Saves the vectorizer to disk using json

    # Returns the vectorizer 

    # Selects the splits in the dataset using a column in the dataframe

    # Returns the target_size

    # The primary entry point method for PyTorch datasets to get the items

    # Given a batch size, return the number of batches in the dataset


# Create the Surname Classifier class with a CNN
    # Create the init function to instantiate class

    # The forward pass of the classifier


# A generator function which wraps the PyTorch DataLoader. It will ensure each tensor is on the write device location.   

# Define the training parameter

# Handle training state updates

# Calculate Accuracy

# Predict the nationality from a new surname

# Predict the top K nationalities from a new surname  

# Start Main path of the Training

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

    # Use the test dataset to validate

        # compute the output
        # compute the loss
        # compute the accuracy

    # Try out the model
