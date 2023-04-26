# -*- coding: utf-8 -*-

import sys
import os
import itertools
sys.path.append('./')
#sys.path.insert(0,'/content/drive/MyDrive/')
sys.path.insert(0,'/content/kcg-ml-elm/notebooks')
import argparse
import warnings
import numpy as np
from datetime import datetime
from elm_training_helper_functions import *
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score
import torch
import numpy as np
import json
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import torch
from sklearn.model_selection import KFold
import numpy as np
import torch.nn as nn
import open_clip
from PIL import Image
import time
import matplotlib.pyplot as plt

class ELMClassifier(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size,batch_size=1, l1_weight=0.001, l1_bias=0.001, use_gpu=True):
        super(ELMClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size=batch_size
        self.l1_weight = l1_weight
        self.l1_bias = l1_bias
        self.use_gpu = use_gpu
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.weight = torch.randn(input_size, hidden_size, requires_grad=True, device=self.device)
        self.bias = torch.randn(hidden_size, requires_grad=True, device=self.device)
        self.beta = torch.randn(hidden_size, output_size, requires_grad=True, device=self.device)
        self.beta.requires_grad = True
        self.activation_hidden = nn.ReLU()
        self.activation_output = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.to(self.weight.device)
        h = torch.relu(x @ self.weight + self.bias)       
        y_pred = torch.softmax(h @ self.beta, dim=1)  # apply softmax to output
        return y_pred

    def print_model_details(self,model):
          print("================ ELMClassifier Model Architecture ================")

          print("Input Layer (size {}):".format(model.input_size))
          print("  Name: input layer")
          print("  Shape: ({},)".format(model.input_size))
          print("  Batch_size: ({},)".format(model.batch_size))
          print("\n================  Hidden layer ================")

          print("\nHidden Layer (size {}):".format(model.hidden_size))
          print("  Name: hidden layer")
          print("  Shape: ({},)".format(model.hidden_size))
          print("  Activation Function: {}".format(self.activation_hidden))
          print("  Input to Hidden Layer Size: ({}, {})".format(model.input_size, model.hidden_size))
          print("  Weight Matrix Size: ({}, {})".format(model.input_size, model.hidden_size))
          print("  Bias Matrix Size: ({},)".format(model.hidden_size))
          print("  Input to Hidden Layer: ({}, {})".format(model.input_size, model.hidden_size))
          print("  Output of Hidden Layer: ({}, {})".format(model.input_size, model.hidden_size))
          print("\n================  Output layer ================")

          print("\nOutput Layer (size {}):".format(model.output_size))
          print("  Name: output layer")
          print("  Shape: ({},)".format(model.output_size))
          print("  Activation Function: {}".format(self.activation_output))
          print("  Output of Output Layer: ({}, {})".format(model.batch_size, model.output_size))
          print("\n===============================================")
    
    def fit(self, x_train, y_train, n_neurons,batch_size, l1_reg_weight=0.001, num_iterations=100):
        elm = ELMClassifier(self.input_size, n_neurons, self.output_size,self.batch_size, self.use_gpu)
        elm.to(self.device)
        H = torch.relu(torch.matmul(x_train, elm.weight) + elm.bias)
        H_inv = torch.pinverse(H)
        elm.beta = torch.matmul(H_inv, y_train)
        return elm
    
    def predict(self, x_test):
        y_pred = self.forward(x_test)
        #print('yr',y_pred)
        predicted_labels = torch.argmax(y_pred, dim=1).cpu().numpy()
        class_probs = torch.softmax(y_pred, dim=1).detach().cpu().numpy().squeeze()
        class0_prob, class1_prob = class_probs[:, 0], class_probs[:, 1]
        return predicted_labels,class0_prob

    def clear_memory(self):
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()

class TrainAndEvaluate:

    def __init__(self,metadata_json, tag_to_hash_json, output_dir,checkpoint_path, n_samples_train=100, n_power=4, sparsity=0.001, num_iterations=100,evaluate_on_test=False):
        self.metadata_json = metadata_json
        self.tag_to_hash_json = tag_to_hash_json
        self.output_dir = output_dir
        self.n_samples_train = n_samples_train
        self.n_power = n_power
        self.sparsity = sparsity
        self.num_iterations = num_iterations
        self.checkpoint_path = checkpoint_path
        self.evaluate_on_test=evaluate_on_test
    '''
    def train_elm_with_kfold(self,x, y, n_power, k_fold, l1_reg_weight, num_iterations,use_gpu=True, random_seed=42):
        global device
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        neuron_powers_of_2 = [2**i for i in range(1, n_power)]
        k_fold_results = []
        best_model = None
        best_accuracy = 0
        l1_bias=0.000
        k_fold_splitter = KFold(n_splits=k_fold, shuffle=True, random_state=random_seed)
        for n_neurons in neuron_powers_of_2:
            k_fold_accuracy = []
            for train_index, test_index in k_fold_splitter.split(x):
                x_train, y_train = x[train_index], y[train_index]
            #    print(  'x.shape[1]', x.shape[1])
                x_train = torch.Tensor(x_train)
                x_train = x_train.to(device)
                encoder = OneHotEncoder(sparse=False)
                onehot_labels = encoder.fit_transform(y_train.reshape(-1, 1))
                y_train = torch.Tensor(onehot_labels) # assuming you have labels for your data
                y_train=  y_train.to(device) 
                x_test, y_test = x[test_index], y[test_index]
                elm = ELMClassifier(x.shape[1], n_neurons, len(np.unique(y)), x.shape[0],l1_reg_weight,l1_bias, use_gpu=use_gpu)
                trained_elm = elm.fit(x_train, y_train, n_neurons,x.shape[0],l1_reg_weight=0.000, num_iterations=100)

                #trained_elm .summary(input_size=x_train,device=device)
                x_test = torch.Tensor(x_test)
                x_test = x_test.to(device)
                y_pred = trained_elm.predict(x_test)
            #   print( y_pred)
                accuracy  = accuracy_score(y_test,y_pred)
            #   print('axxc',accuracy)
              # accuracy = np.mean(y_pred == y_test)
                k_fold_accuracy.append(accuracy)
            average_accuracy = np.mean(k_fold_accuracy)
        #   print( 'average_accuracy', average_accuracy)
            k_fold_results.append((n_neurons, average_accuracy))
            if average_accuracy > best_accuracy:
                best_accuracy = average_accuracy
                best_model = trained_elm
        best_neuron_count = max(k_fold_results, key=lambda x: x[1])[0]
        print(f"Best neuron count: {best_neuron_count}")
        print('Best average accuracy:', best_accuracy )
       
        return best_model,best_neuron_count
        
    '''            
    def train_test_classifier(self):
            import matplotlib.pyplot as plt
            global class_names,best_neuron_count,device
            n_neurons=50 #default
            class_names = []
            accuracies = []
            precisions = []
            recalls = []
            f1_scores = []
            metadata_dict    = load_json(self.metadata_json)
            tag_to_hash_json = load_json(self.tag_to_hash_json)
            use_cuda = torch.cuda.is_available()
            if use_cuda:
                # Move tensors to CUDA device
                device = torch.device('cuda:0')
                print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
            else:
                # Use CPU
                device = torch.device('cpu')
                print("Warning: CUDA not available, using CPU.")
            if metadata_dict is None or tag_to_hash_json is None : # Problem happened with the json file loading.
                    raise('problem in data')

                # Get training start time
            t_start = datetime.now()
                # get the two output folder paths (models and reports) with respect to \
                # the output directory provided in the script.
            report_out_folder , models_out_folder = check_out_folder(self.output_dir) 

                # other training and other validation embeddings lists.
        #   other_all_emb_list     = [metadata_dict[hash_id]["embeddings_vector"] for hash_id in tag_to_hash_json['Others']]

            other_all_emb_list     = [metadata_dict[hash_id]["embeddings_vector"] for hash_id in tag_to_hash_json['other-training']]
            other_val_all_emb_list = [metadata_dict[hash_id]["embeddings_vector"] for hash_id in tag_to_hash_json['other-validation']]

                # get embeddings from tag_emb_dict and make it reay for training the classifier 
            start_time = time.time()
                       
            for tag in tag_to_hash_json:
                    neurons_list = []
                    tag_predictions_list = []
                    # make sure that it's a pixel art class tag. 
                    if tag in ['other-training' ,'other-validation','Others']:
                        continue
                    class_names.append(tag)
                    # get embedding list of tag images.
                    tag_all_emb_list = [metadata_dict[hash_id]["embeddings_vector"] for hash_id in tag_to_hash_json[tag]]
                    if len(tag_all_emb_list)<=self.n_samples_train or len( other_all_emb_list)<=self.n_samples_train:
                        continue
                    # get train test embeddings and labels.
                    train_emb, train_labels, test_emb, test_labels , t_n , o_n ,tr_tg,tr_otr,tg_lb,otr_lb= get_train_test(tag_all_emb_list, other_all_emb_list , self.n_samples_train)
            #        print(test_emb.shape)
               #     print(len(test_labels))
               #     print(tag)
                    # Create test tensor
                    x_test = torch.Tensor(test_emb)
                    x_test = x_test.to(device)
                #    print(x_test.shape)
                #   print(x_train.shape)
                    input_size =  train_emb.shape[1] # assuming the shape of the embeddings is (num_samples, embedding_size)
                    x_train = torch.Tensor( train_emb)
                    x_train = x_train.to(device)
                    encoder = OneHotEncoder(sparse=False)
                    onehot_labels = encoder.fit_transform(train_labels.reshape(-1, 1))
                    y_train = torch.Tensor(onehot_labels) # assuming you have labels for your data
                    y_train=  y_train.to(device) 
                    output_size = len(set(train_labels)) # number of unique labels
                    if  output_size !=2:
                        raise ValueError(" output_size should be equal to 2")
              #      print( 'output_size', output_size)
              #      print(input_size)
                    neuron_powers_of_2 = [2**i for i in range(1, self.n_power)]  # e.g. [2, 4, 8, 16, 32, 64, 128]
    # Train and save ELM models with different numbers of neurons
                    checkpoint_folder=os.path.join(self.checkpoint_path,tag)
                    if not os.path.exists(checkpoint_folder):
                          os.makedirs(checkpoint_folder,exist_ok=True)    
                    for n_neurons in neuron_powers_of_2:
                        elm = ELMClassifier(x_train.shape[1], n_neurons, len(np.unique(y_train)), x_train.shape[0],l1_weight=0.000,l1_bias=0.000, use_gpu=use_cuda)
                        trained_elm = elm.fit(x_train, y_train, n_neurons,x_train.shape[0])
                # Assuming your computed embeddings for the test data are stored in a numpy array called 'test_embeddings'
                        dict_args_list = [n_neurons, tag, len(test_labels), t_n, o_n, tr_tg, tr_otr, tg_lb, otr_lb, input_size, output_size, trained_elm]
                        dict1, checkpoint = self.create_dicts(dict_args_list)
                      #  predicted_labels = best_model.predict(x_test)    
                        torch.save(checkpoint, checkpoint_folder+'/'+'elm_classifier'+'_'+str(n_neurons)+'_'+tag+'.pth')
                        if self.evaluate_on_test==True:
                            predicted_labels,tag_prob = trained_elm.predict(x_test)
                            acc,class_specific_accuracies=self.result_stats(test_labels,predicted_labels,tag,n_neurons,self.output_dir,dict1)
                            folder_plots=self.save_classification_report(test_labels, predicted_labels, tag, acc, self.output_dir,n_neurons)
                            tag_predictions_list.append(tag_prob)
                            neurons_list.append(n_neurons)    
                    if self.evaluate_on_test==True:
                          self.scatter_plot(neurons_list,tag_predictions_list,tag,folder_plots) 
            end_time=time.time()
            total_time = end_time - start_time
            print("Total training time: {:.2f} seconds".format(total_time))
          

    def create_dicts(self,args_list):
        dict1 = {
            'Model': 'ELM',
            'Number of neurons': args_list[0],
            'tag': args_list[1],
            'Total test samples': args_list[2],
            "tag_test_samples": args_list[3],
            "others_test_samples": args_list[4],
            "tag_train_samples": args_list[5],
            "other_train_samples": args_list[6],
            "tag_label_name": args_list[7],
            "others_label_name": args_list[8]
        }
        checkpoint = {
            'input_size': args_list[9],
            'hidden_size': args_list[0],
            'output_size': args_list[10],
            'state_dict': args_list[11].state_dict()
        }
        return dict1, checkpoint


    def convert_ndarray_to_list(self,d):
        for k, v in d.items():
            if isinstance(v, dict):
                self.convert_ndarray_to_list(v)
            elif isinstance(v, np.ndarray):
                d[k] = v.tolist()
             
    def result_stats(self,test_labels,predicted_labels,tag,best_neuron_count,output_dir,dict1):
            
            if test_labels is not None:
                acc = accuracy_score(test_labels,predicted_labels)
                report = classification_report(test_labels,  predicted_labels,output_dict=True)        
                confusion_matrix = metrics.confusion_matrix(test_labels, predicted_labels)
                FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
                FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
                TP = np.diag(confusion_matrix)
                TN = confusion_matrix.sum() - (FP + FN + TP)
                ALL_SUM = FP + FN + TP + TN
        # Sensitivity, hit rate, recall, or true positive rate
                TPR = TP/(TP+FN)
        # Specificity or true negative rate
                TNR = TN/(TN+FP) 
        # Fall out or false positive rate
                FPR = FP/(FP+TN)
        # False negative rate
                FNR = FN/(TP+FN)
                print(type(report))
                class_specific_accuracy = TP / (TP + FP + FN)
                compile_results={'Correct predictions':np.sum(np.diag(confusion_matrix)),"Accuracy":np.sum(np.diag(confusion_matrix))*100/len(test_labels), 
                'Sensitivity':TPR,'Specificity':TNR, 'Classification report': report}
                complete_dict={**dict1,**compile_results}
                for key, value in  complete_dict.items():
                    if isinstance(value, np.int64):
                          complete_dict[key] = int(value)
                #resultList = [(key, value) for key, value in compile_results.items()]
                self.convert_ndarray_to_list(complete_dict) 
              #   print(type( resultList))
                #compile_results.tolist()         
                with open(os.path.join(output_dir,'reports',tag)+'_'+str(best_neuron_count)+ '.json', 'w') as f:
                      json.dump(complete_dict, f,indent=4)
                complete_dict.clear()      
                return  acc,class_specific_accuracy  

    def save_classification_report(self,test_labels, predicted_labels, tag, acc, output_dir,n_neurons):
        report = classification_report(test_labels, predicted_labels, digits=4, output_dict=True)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1_score = report['weighted avg']['f1-score']
        folder_plots = os.path.join(output_dir, 'report', tag)
        print(folder_plots)
        if not os.path.exists(folder_plots):
            os.makedirs(folder_plots, exist_ok=True)
        x_labels = ["Accuracy", "F1 Score", "Precision", "Recall"]
        y_values = [acc, f1_score, precision, recall]
        x_pos = np.arange(len(x_labels))
        
        plt.bar(x_pos, y_values)
        plt.ylim(0, 1)
        plt.xticks(x_pos, x_labels)
        plt.title('Elm model for ' + tag)
        plt.savefig(os.path.join(folder_plots, 'performance_merics of' + ' ' + tag +'for' +str(n_neurons)+  '.png'), dpi=300, bbox_inches='tight')
        plt.clf()
        return folder_plots
    
    def scatter_plot(self,neurons_list,tag_predictions_list,tag,folder_plots):
       
        arrays=list(tag_predictions_list)
        n_new=[]
        # create a list of n_number values
        for i in neurons_list:
            upd=[]
            for j in range(len(arrays[0])):
                upd.append(i)
            n_new.append(upd)
        # create a scatter plot of the arrays against their corresponding n_number index
        for i, arr in enumerate(arrays):
            plt.scatter(n_new[i], arr)
        
        # set plot title and axis labels
        plt.xlabel('Number of Neurons')
        plt.ylabel('Predicted Probability')
        plt.title('Different checkpoints predictions')
        # Set y-axis limits
        plt.ylim(0, 1)
      #  plt.savefig(os.path.join(folder_plots, 'Predicted Probabilities for' + ' ' + tag + '.png'), dpi=300, bbox_inches='tight')
      
        plt.legend(list(map(str, neurons_list)),  ncol = 1,loc='center left', bbox_to_anchor=(1, 0.5))
      #  plt.legend(bbox_to_anchor=(0.4, 0.8), loc="upper right")
        plt.savefig(os.path.join(folder_plots, 'Predicted Probabilities for' + ' ' + tag + '.png'), dpi=300, bbox_inches='tight')

        # plt.legend(['2', '4', '8', '16', '32', '64'])
        # plt.legend(['128', '256', '512'])
        # show the plot
        plt.show()
        plt.clf()
      
def process_single_image(
        image: str,
        clip_model: str = 'ViT-L-14', 
    ) -> None: 
       
        pretrained = 'openai'        
        #detect the device to be used in calculating the embeddings. 
        if torch.cuda.is_available():
           device = "cuda"
        else:
           device ="cpu" 
           print('warning running on cpu, no cuda device is found')

        
        #load the CLIP model. 
        model, _, preprocess = open_clip.create_model_and_transforms(clip_model,pretrained = pretrained)
        
        model = model.to(device)
    
        #preprocess the image
        image=Image.open(image)
        print(image.size)
        preprocessed_chunk = torch.stack((preprocess(image),))

        #compute the CLIP embeddings of the current image. 
        image_embeddings = model.encode_image(preprocessed_chunk.to(device))
        return image_embeddings        
