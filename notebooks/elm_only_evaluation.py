import sys
import argparse
sys.path.insert(0,'/content/drive/MyDrive/elm_checkpoints')
sys.path.insert(0,'/content/drive/MyDrive/')
sys.path.insert(0,'/content/kcg-ml-elm/notebooks')
#from str2bool import str2bool
from typing import List
from elm_model_upd import TrainAndEvaluate 
from Image_Dataset_Processor import ImageDatasetProcessor
import patoolib
import matplotlib.pyplot as plt
from elm_training_helper_functions import *
from elm_model_upd import ELMClassifier
#import elm_model
import elm_model_upd
import shutil
import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score
import json
import torch.nn.functional as F
import os
global dict1

def test_data_prepare(
                    tag_all_emb_list: List, 
                    other_all_emb_list: List,
                    ):
    """takes embeding list of tag/class and other images,
    converts them into arrays ready for the training/testing.

    :param tag_all_emb_list: list of embeddings for the tag images.
    :type tag_all_emb_list: List
    :param other_all_emb_list: list of embeddings for the other images.
    :type other_all_emb_list: List
    :param test_per: percentage of the test images from the embeddings list.
    :type test_per: float
    :returns: tuple of the train_embds , train_labels , test_embs , test_labels \
              number of tags test images , number of other test images 
    :rtype: tuple
    """
    # get the test embeds from both classes (tag class and other)
    train_emb_list   = []
    train_label_list = []
    test_emb_list    = []
    test_label_list  = []

    # size of the number of the test set of the tag/class 
    tag_n_test = len(tag_all_emb_list) if len(tag_all_emb_list)  > 0 else 1 
    test_emb_list.extend(tag_all_emb_list)
    test_tag_label_list  = [0] * len(tag_all_emb_list)   # test labels for tag/class embeddings
    test_label_list.extend(test_tag_label_list) 
   
    # size of the number of the test set of the tag/class 
    other_n_test = len(other_all_emb_list) if  len(other_all_emb_list)  > 0 else 1 
       # test other embeddings.
    test_emb_list.extend(other_all_emb_list)
    test_other_label_list  = [1] * len(other_all_emb_list)       # test labels for other embeddings.        
    test_label_list.extend(test_other_label_list) 
   

    # convert all of these lists into numpy arrays and returns them. 
    return np.array(test_emb_list), np.array(test_label_list), \
           tag_n_test, other_n_test, test_tag_label_list[0],test_other_label_list[0] 


def test_data(metadata_json,tag_to_hash_json,output_dir,checkpoint,tag):
        #hidden_size =  # number of hidden neurons, you can adjust this parameter
        metadata_dict    = load_json(metadata_json)
        tag_to_hash_json = load_json(tag_to_hash_json)
        if metadata_dict is None or tag_to_hash_json is None : # Problem happened with the json file loading.
                raise('problem in data')

            # Get training start time
        
            # get the two output folder paths (models and reports) with respect to \
            # the output directory provided in the script.
       # report_out_folder , models_out_folder = check_out_folder(output_dir) 

            # other training and other validation embeddings lists.
        other_all_emb_list     = [metadata_dict[hash_id]["embeddings_vector"] for hash_id in tag_to_hash_json['other-training']]
        other_val_all_emb_list = [metadata_dict[hash_id]["embeddings_vector"] for hash_id in tag_to_hash_json['other-validation']]

            # get embeddings from tag_emb_dict and make it reay for training the classifier 


        # get embedding list of tag images.
        tag_all_emb_list = [metadata_dict[hash_id]["embeddings_vector"] for hash_id in tag_to_hash_json[tag]]

        # get train test embeddings and labels.
        test_emb, test_labels, t_n , o_n, tg_lb, otr_lb = test_data_prepare(tag_all_emb_list, other_all_emb_list)
      #  print(len(test_emb))
      #  print(len(test_labels))
      #  print(tag)
        dict1={'Model':'ELM','Number of neurons':checkpoint['hidden_size'],
                'tag':tag, 'Total test samples':len(test_labels),
                "tag_test_samples": t_n,
                "others_test_samples": o_n,
                "tag_label_name": tg_lb,
                "others_label_name": otr_lb}
# Assuming your computed embeddings are stored in a numpy array called 'embeddings'
        x_test = torch.Tensor(test_emb)
        y_test = torch.Tensor(test_labels) # assuming you have labels for your data
        
#        input_size =  x_train.shape[1] # assuming the shape of the embeddings is (num_samples, embedding_size)

        output_size = len(set(test_labels)) # number of unique labels
        return   x_test, y_test,dict1

def select_device():
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        # Move model to CUDA device
        device = torch.device('cuda:0')
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
    else:
        # Use CPU
        device = torch.device('cpu')
        print("Warning: CUDA not available, using CPU.")
        
def evaluation(compute_clip_features,dataset_path,metadata_json,tag_to_hash_json,output_dir,clip_model,tag,checkpoint_path,device,j):
    # Load datase
    torch.manual_seed(42)
    np.random.seed(42)  
    if compute_clip_features:
            print("Feature extraction started ...")
            ImageDatasetProcessor.process_dataset(dataset_path,output_dir, clip_model)
    # make generator for unseen speaker identification
            print("Feature extraction completed")
    
    
    checkpoint = torch.load(checkpoint_path)
    print('Checkpoint loaded successfully...')   
    data_test, test_labels, dict1=test_data(metadata_json,tag_to_hash_json,output_dir,checkpoint,tag)

# Create a new ELMClassifier with the same hyperparameters as the trained model
    model = ELMClassifier(input_size=checkpoint['input_size'], hidden_size=checkpoint['hidden_size'], 
                        output_size=checkpoint['output_size'],batch_size=data_test.shape[0], use_gpu=True)

    model.to(device)  
    # Load the weights and biases from the checkpoint into the new model
    

    # Put the model in evaluation mode
    model.eval()
    # Use the model to make predictions on some test data
    print('model predictions started')
    predicted_labels,tag_prob = model.predict(data_test)
    #print(predicted_labels)
    evaluate = TrainAndEvaluate(metadata_json, tag_to_hash_json, output_dir,checkpoint_path)
    acc,report=evaluate.result_stats(test_labels,predicted_labels,tag,checkpoint['hidden_size'],output_dir,dict1,j)
    folder_plots=evaluate.save_classification_report(test_labels, predicted_labels, tag, acc, output_dir,checkpoint['hidden_size'],j)
    # Release memory
    #evaluate.scatter_plot(checkpoint['hidden_size'],tag_prob)
    
    model.clear_memory()
    #print("Testing completed check output folder for results")
    return tag_prob,checkpoint['hidden_size'],evaluate,folder_plots
