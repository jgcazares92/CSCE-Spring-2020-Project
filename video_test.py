from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.applications.vgg16 import VGG16
import cv2
import math
import os, getopt, sys
import argparse
from glob import glob
from scipy import stats as s
import json
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--video_name")
args = parser.parse_args()
if args.video_name:
    print("Output : %s" % args.video_name)

modsteep_dict = {'ModifiedSteeple': None}
pointing_dict = {'Pointing': None}
steeple_dict = {'Steeple': None}
data = {'ModifiedSteeple': None,
        'Pointing': None,
        'Steeple': None}
tracker_mod, tracker_point, tracker_steep = [], [], []
start_time = time.time()

base_model = VGG16(weights='imagenet', include_top=False)

# Model architecture
model = Sequential()
model.add(LSTM(2048, return_sequences=False,
               input_shape = (1, 25088), dropout=0.5))
#model.add(Dense(1024, activation='relu', input_shape=(25088,)))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))

# loading the trained weights
model.load_weights("weight.hdf5")

# compiling the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

#Show model summary
model.summary()

# getting the test list
temp = "{}".format(args.video_name)
videos =  temp.split('\n')

# creating the dataframe
test = pd.DataFrame()
test['video_name'] = videos
test_videos = test['video_name']
test.head()

# creating the tags
train = pd.read_csv('./CSCE-636-Videos/train_new.csv')
y = train['class']
y = pd.get_dummies(y)

# creating two lists to store predicted and actual tags
predict = []
actual = []

# Track Time
time_tracker = []

# for loop to extract frames from each test video
for i in tqdm(range(test_videos.shape[0])):
    count = 0
    videoFile = test_videos[i]
    cap = cv2.VideoCapture('./CSCE-636-Videos/'+videoFile.split(' ')[0])   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate 
    x=1
    # removing all other files from the temp folder
    files = glob('temp/*')
    for f in files:
        os.remove(f)
        tracker = []
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            # storing the frames of this particular video in temp folder
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            filename ='temp/' + "_frame%d.jpg" % count;count+=1
            time_tracker.append(time.time()-start_time)
            cv2.imwrite(filename, frame)
    cap.release()
    
    # reading all the frames from temp folder
    images = glob("temp/*.jpg")
    
    prediction_images = []
    for i in range(len(images)):
        img = image.load_img(images[i], target_size=(224,224,3))
        img = image.img_to_array(img)
        img = img/255
        prediction_images.append(img)
        

    # converting all the frames for a test video into numpy array
    prediction_images = np.array(prediction_images)
    # extracting features using pre-trained model
    prediction_images = base_model.predict(prediction_images)
    # converting features in one dimensional array
    prediction_images = prediction_images.reshape(prediction_images.shape[0], 7*7*512)
    prediction_images = np.reshape(prediction_images, (prediction_images.shape[0], 1, prediction_images.shape[1]))
    # predicting tags for each array
    prediction = model.predict_classes(prediction_images)
    # print("prediction: ", prediction) #DEBUG
    pred_probabilities = model.predict_proba(prediction_images)
    #print(pred_probabilities) # DEBUG
    # appending the mode of predictions in predict list to assign the tag to the video
    predict.append(y.columns.values[s.mode(prediction)[0][0]])

    #Makeshift method for combining classes and their probablities in the desired format
    for i in range(len(prediction)):
        end_time = time.time()
        if prediction[i]==0:
            tracker_mod.append([time_tracker[i], pred_probabilities[i][0]])
        elif prediction[i]==1:
            tracker_point.append([time_tracker[i], pred_probabilities[i][1]])
        else:
            tracker_steep.append([time_tracker[i], pred_probabilities[i][2]])

    #print(tracker_point) # DEBUG
    data.update(ModifiedSteeple = tracker_mod)
    data.update(Pointing = tracker_point)
    data.update(Steeple = tracker_steep)

    # appending the actual tag of the video
    actual.append(videoFile.split('/')[1].split('_')[1])

# checking the accuracy of the predicted tags
print("Check accuracy of the predicted tags...")
from sklearn.metrics import accuracy_score
print(accuracy_score(predict, actual)*100)
print("predicted: ", predict)
print("actual: ", actual)

fig = plt.figure()
x_val = []
y_val = []
for i in range(len(tracker_mod)):
    x_val.append(time_tracker[i])
    y_val.append(tracker_mod[i][1])
    plt.plot(x_val, y_val, "ro", c='blue')

x_val = []
y_val = []
for i in range(len(tracker_point)):
    x_val.append(time_tracker[i])
    y_val.append(tracker_point[i][1])
    plt.plot(x_val, y_val, "ro", c='maroon')

x_val = []
y_val = []
for i in range(len(tracker_steep)):
    x_val.append(time_tracker[i])
    y_val.append(tracker_steep[i][1])
    plt.plot(x_val, y_val, "ro", c='green')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel(r'$Label$', fontsize=20)
plt.xlabel(r'$Time (s)$', fontsize=20)
plt.legend(loc='best')
fig.set_size_inches(10.5, 7.5)
plt.savefig('data.jpg')


with open('data.json', 'w') as outfile:
    json.dump(str(data), outfile)
