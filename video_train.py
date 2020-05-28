import keras
from keras.models import Sequential, load_model
from keras.applications.vgg16 import VGG16
from keras.layers.recurrent import LSTM
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.preprocessing import image
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

train = pd.read_csv('CSCE-636-Videos/train_new.csv')
train.head()

# creating an empty list
train_image = []

# for loop to read and store frames
for i in tqdm(range(train.shape[0])):
    # loading the image and keeping the target size as (224,224,3)
    img = image.load_img(r'C:/Users/admin-taleb/Documents/Videos/train_1/'+train['image'][i], target_size=(224,224,3))
    # converting it to array
    img = image.img_to_array(img)
    # normalizing the pixel value
    img = img/255
    # appending the image to the train_image list
    train_image.append(img)
    
# converting the list to numpy array
X = np.array(train_image)

# shape of the array
X.shape


# separating the target
y = train['class']

# creating the training and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify = y)

# creating dummies of target variable for train and validation set
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

# creating the base model of pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

# extracting features for training frames
X_train = base_model.predict(X_train)
X_train.shape
print("X_TRAIN SHAPE: ", X_train.shape)
print("y_train shape: ", y_train.shape)

# extracting features for validation frames
X_test = base_model.predict(X_test)
X_test.shape
print("X_TEST SHAPE: ", X_test.shape)
print("y_test shape: ", y_test.shape)

# reshaping the training as well as validation frames in single dimension
# len(X_train) is number of training images 
X_train = X_train.reshape(len(X_train), 7*7*512)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
# len(X_test) is number of test images
X_test = X_test.reshape(len(X_test), 7*7*512)
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# normalizing the pixel values
max = X_train.max()
X_train = X_train/max
X_test = X_test/max

# shape of images
X_train.shape

# Model architecture
input_shape = (30,2084)

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
#model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))


# defining a function to save the weights of best model
from keras.callbacks import ModelCheckpoint
mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')

# compiling the model
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# training the model
history = model.fit(X_train, y_train, 
                    epochs=10, batch_size=64,
                    validation_data=(X_test, y_test), 
                    callbacks=[mcp_save])

save_dir = "./results"
model.save(save_dir)
print("Saved trained model")

test_model = load_model("results")
loss_and_metrics = test_model.evaluate(X_test, y_test, verbose=2)

print("Test Loss ", loss_and_metrics[0])
print("Test Accuracy " , loss_and_metrics[1])

fig = plt.figure(figsize=(10,5))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
mng = plt.get_current_fig_manager()
fig.set_size_inches(9.59, 7.135)
plt.savefig('Accuracy.pdf', bbox_inches='tight', transparent=True, dpi=300)

fig = plt.figure(figsize=(10,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
mng = plt.get_current_fig_manager()
fig.set_size_inches(9.59, 7.135)
plt.savefig('Loss.pdf', bbox_inches='tight', transparent=True, dpi=300)

