from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy

train_data_path = 'flower_dataset/train'
test_data_path = 'flower_dataset/test'

model = Sequential()
model.add(Conv2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())

model.add(Dense(5, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_data_path,
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('flower_dataset/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

"""
# Fit the model just once 
model.fit_generator(training_set,
                         samples_per_epoch = 3418,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 905)"""

# Saving the model  as model.h5 to my root dir
#model.save('model.h5') 
                
from keras.models import load_model

# Loading the saved fit model

flower_model = load_model('model.h5')

flower_test = test_datagen.flow_from_directory('flower_dataset/test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')
# Retrieving the labels from test data into labels variable

x, labels = flower_test .next()

labels = labels.tolist()
actual_labels = []
for l in labels:
    index = l.index(max(l))
    actual_labels.append(index)

#flower_model.summary()

predictions = flower_model.predict_generator(flower_test, steps = 1)
predictions = numpy.array(predictions)
predictions = predictions.tolist()
predicted_labels = []
for p in predictions:
    index = p.index(max(p))
    predicted_labels.append(index)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(actual_labels, predicted_labels)

from sklearn.metrics import accuracy_score
acc = accuracy_score(actual_labels, predicted_labels)

