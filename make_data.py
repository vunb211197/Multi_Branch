import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


data_path = 'C:\\Users\\nguyenbavu\\Desktop\\Multitask_Branch\\data'

def make_data():
    X_data=[]
    age_labels=[]
    gender_labels=[]
    race_labels=[]
    
    for dir in os.listdir(data_path):
        try :
            all = dir.split('_')

            class_dir = os.path.join(data_path,dir)
            img = cv2.imread(class_dir)
            img = cv2.resize(img,(96,96))
            # để model hội tụ nhanh hơn thì chia toàn bộ các pixel
            img = img/255.0
            age = int(all[0])
            gender = int(all[1])
            race = int(all[2])

            age_labels.append(age)
            gender_labels.append(gender)
            race_labels.append(race)
            X_data.append(img)
        except : 
            continue
    
    
    #chuyển sang numpy array
    X_data = np.asarray(X_data)
    X_data = X_data / 255.0
    age_labels = np.reshape(np.asarray(age_labels),(9778,1))
    gender_labels = np.reshape(np.asarray(age_labels),(9778,1))
    race_labels = np_utils.to_categorical(np.asarray(race_labels))
    


    X_data,age_labels,gender_labels,race_labels = shuffle(X_data,age_labels,gender_labels,race_labels)

    return X_data, age_labels,gender_labels,race_labels




