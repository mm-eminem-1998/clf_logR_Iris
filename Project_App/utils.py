
import pandas as pd
import numpy as np

# To load model trained in jupyter notebook. 
import pickle

# To load json file.  
import json

# To avoid warning 
import warnings
warnings.filterwarnings("ignore")

# To dynamicaly load pickle & json file. 
import config

class IrisDataset():

    # Don't forget to pass self in function, it is mandetory
    # Function for variable which we will taken from user input should pass as parameter. 
    def __init__(self, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm):
        
        # Need to create instance variable. 
        # Instance variable is created to use this variable in any function
        self.SepalLengthCm = SepalLengthCm
        self.SepalWidthCm  = SepalWidthCm
        self.PetalLengthCm = PetalLengthCm
        self.PetalWidthCm  = PetalWidthCm
        
    # Fuction to load Model & json_data.
    def load_models(self):
        with open(config.MODEL_FILE_PATH, "rb") as f:    
            self.model = pickle.load(f)  # Need to create instance variable.

        # json_data file stored column name & encoding data in dictionary format.
        with open(config.JSON_FILE_PATH, "r") as f:
            self.json_data = json.load(f) # Need to create instance variable.           
    
    # Function to define Prediction.
    # Need repeat coding related user input.
    def get_prediction(self):
        # We can't use model & json_data without creating instance of load_models function
        self.load_models() # Creating instance of model & json_data

        # Here columns names needs to fetch from json_data
        # Need to create instance variable.   
        test_array = np.zeros(len(self.json_data['Columns'])) 

        test_array[0] = self.SepalLengthCm
        test_array[1] = self.SepalWidthCm
        test_array[2] = self.PetalLengthCm
        test_array[3] = self.PetalWidthCm

        print("Array : ---> ", test_array)
        
        flower_species = self.model.predict([test_array])[0]
        
        # Prediction value
        return flower_species
       
if __name__ == "__main__":
    # User inputs
    SepalLengthCm = 5.7
    SepalWidthCm  = 3.8
    PetalLengthCm = 1.7
    PetalWidthCm  = 0.5

    # IrisDataset class with parameter.
    iris = IrisDataset(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
    flower_species = iris.get_prediction()
    

    print("Predicted Flower Species is : -----> ", flower_species)
        








