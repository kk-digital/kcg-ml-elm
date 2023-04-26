import os
import shutil
import random

class PrepareDataset:
    def __init__(self, input_folder_path, train_ratio=0.8):
        self.input_folder_path = input_folder_path
        self.train_ratio = train_ratio

    def split_dataset(self, output_folder_path):
        train_folder_path = os.path.join(output_folder_path, "train")
        test_folder_path = os.path.join(output_folder_path, "test")
        os.makedirs(train_folder_path, exist_ok=True)
        os.makedirs(test_folder_path, exist_ok=True)

        for root, dirs, files in os.walk(self.input_folder_path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".gif") :
                    source_file_path = os.path.join(root, file)
                    # get the label of the file by going up one directory level
                    label = os.path.basename(os.path.dirname(source_file_path))
                    if random.random() < self.train_ratio:
                        dest_folder_path = os.path.join(train_folder_path, label)
                    else:
                        dest_folder_path = os.path.join(test_folder_path, label)
                    os.makedirs(dest_folder_path, exist_ok=True)
                    shutil.copy2(source_file_path, os.path.join(dest_folder_path, file))
