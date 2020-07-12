import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


if tf.test.gpu_device_name(): 
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")


# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print("Number of train data - " + str(len(x_train)))
print("Number of test data - " + str(len(x_test)))

# Split data into train / validation sets 
(x_train, x_valid) = x_train[5000:], x_train[:5000] 
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# Reshape data to (28, 28, 1)
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Print training set shape
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Print the number of training, validation, and test datasets
print(x_train.shape[0], 'train set')
print(x_valid.shape[0], 'validation set')
print(x_test.shape[0], 'test set')

# First model: simple CNN

model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Take a look at the model summary
model.summary()

model.compile(loss='categorical_crossentropy',
             optimizer="adam",
             metrics=['accuracy'])

 
# Create checkpoint to save only model whose validation loss has improved
# Create also an early stopper

checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)
earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001,patience=20)

model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=100,
         validation_data=(x_valid, y_valid),
         callbacks=[checkpointer, earlystop_callback])

# Load Model with the best validation accuracy
# Load the weights with the best validation accuracy
model.load_weights('model.weights.best.hdf5')

#Save the complete model with the loaded weights
model.save('model.best.h5') 

# Test Accuracy
# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])


# Second model: simple CNN tuned with KerasTuner

from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
from tensorflow.keras.optimizers import Adam


def build_model(hp):
    
    # Specify model
    model = tf.keras.Sequential()

    # Must define the input shape in the first layer of the neural network
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout_1', 0, 0.4, step=0.1, default=0)))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout_2', 0, 0.4, step=0.1, default=0)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(hp.Float('dropout_3', 0, 0.4, step=0.1, default=0)))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # Compile the constructed model and return it
    model.compile(
        optimizer=Adam(
            hp.Choice('learning_rate',
                        values=[0.01, 0.005, 0.001])),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model
 
# Construct the RandomSearch tuner
random_tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=30,
    executions_per_trial = 1,
    seed=10, 
    project_name='fashion_mnist_keras',
    directory="DIRECTORY\fashion-mnist")


# Search for the best parameters of the neural network using the contructed random search tuner
random_tuner.search(x_train, y_train,
             epochs=30,
             validation_data=(x_valid, y_valid),
             )

#get the best model

random_params = random_tuner.get_best_hyperparameters()[0]
best_model = random_tuner.get_best_models(1)[0]

#Save the complete model with the loaded weights
best_model.save('model.best.kerastuned.h5') 

#Evaluate it on the validation test

print("Evalutation of best performing model:")
print(best_model.evaluate(x_valid, y_valid, verbose=0)[1])


# Evaluate the model on test set
score_tuned = best_model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score_tuned[1])


# Third model: CNN trained using Data Augmentation
 
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                               height_shift_range=0.08, zoom_range=0.08)
batches = gen.flow(x_train, y_train, batch_size=64)
val_batches = gen.flow(x_valid, y_valid, batch_size=64)


checkpointer_aug = ModelCheckpoint(filepath='model.weights.best.aug.hdf5', verbose = 1, save_best_only=True)

model.fit(batches, steps_per_epoch = len(x_train)//64, epochs=100,
                    validation_data=val_batches, validation_steps=len(x_valid)//64, use_multiprocessing=False, callbacks=[checkpointer_aug, earlystop_callback])

# Load the weights with the best validation accuracy
model.load_weights('model.weights.best.aug.hdf5')
 
#Save the complete model with the loaded weights
model.save('model.best.aug.h5') 
 
# Evaluate the model on test set

score_aug = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score_aug[0])
print('Test accuracy:', score_aug[1])


# Load a model and visualize predictions

# Define the labels
fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Sneaker",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9

# Load model from kerastuner
model = best_model

y_hat = model.predict(x_test)

# Plot a random sample of 10 test images, their predicted labels and ground truth
figure = plt.figure(figsize=(20, 20), facecolor='black')
for i, index in enumerate(np.random.choice(x_test.shape[0], size=100, replace=False)):
    ax = figure.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
    # Display each image
    ax.imshow(np.squeeze(x_test[index]),cmap = plt.cm.gray)
    predict_index = np.argmax(y_hat[index])
    true_index = np.argmax(y_test[index])
    # Set the title for each image
    ax.set_title("{} ({})".format(fashion_mnist_labels[predict_index], 
                                  fashion_mnist_labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))


 


