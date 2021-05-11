import os
import random
import shutil

data_root = "data"
data_raw = os.path.join(data_root, "raw")
data_train = os.path.join(data_root, "train")
data_val  = os.path.join(data_root, "val")

for folder in os.listdir(data_raw):
    if folder[0]!=".":
        print("Folder ", folder)
        file_list = []
        full_folder = os.path.join(data_raw, folder, folder)
        for file in os.listdir(full_folder):
            full_file = os.path.join(full_folder, file)
            file_list.append(full_file)

        total_files = len(file_list)
        total_train_files = int(0.8*total_files)
        train_files = random.choices(file_list, k=total_train_files)
        print("Số file = ",len(train_files))

        train_folder  = os.path.join(data_train, folder)
        if not os.path.exists(train_folder):
            os.makedirs(train_folder)

        for train_file in train_files:
            print("Copy to train ", train_file)
            shutil.copyfile(train_file, os.path.join(train_folder, os.path.basename(train_file)))

        val_folder = os.path.join(data_val, folder)
        if not os.path.exists(val_folder):
            os.makedirs(val_folder)

        for val_file in file_list:
            if not (val_file in train_file):
                print("Copy to val ", val_file)
                shutil.copyfile(val_file, os.path.join(val_folder, os.path.basename(val_file)))

