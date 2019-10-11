import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata

class CNN: 

    def __init__(self, batch_size, epochs, height, width):
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = Sequential()
        self.height = height
        self.width = width
        self.directories = {
            "training": "./data/training",
            "validation": "./data/validation"
        }
        
        self.customize_layers()

    def setup_image_generator(self):
        self.training_image_gen = ImageDataGenerator(
                                    rescale=1./255,  
                                    rotation_range=45,
                                    width_shift_range=.15,
                                    height_shift_range=.15,
                                    horizontal_flip=True,
                                    zoom_range=0.5,
                                    brightness_range=[0.2,1.0])
        self.validation_image_gen = ImageDataGenerator(rescale=1./255)
    
    def data_generator(self):
        self.training_data_generator = self.training_image_gen.flow_from_directory(
                                                         batch_size=self.batch_size,
                                                         directory=self.directories["training"],
                                                         shuffle=True,
                                                         target_size=(self.height, self.width),
                                                         class_mode='binary')
        self.validation_data_generator = self.validation_image_gen.flow_from_directory(
                                                        batch_size=self.batch_size,
                                                        directory=self.directories["validation"],
                                                        target_size=(self.height, self.width),
                                                        class_mode='binary')

    def customize_layers(self):
        # Convolution layer
        self.add_input_layer(32)
        self.add_convolution_layer(32, (3,3))
        self.add_convolution_layer(64, (3,3))

        # Dense layers   
        self.add_flatten_layer()
        self.add_dense_layer(128, 'relu', dropout=0.5)
        self.add_dense_layer(64, 'relu', dropout=0.2)
        self.add_dense_layer(1, 'sigmoid')
    
    def add_input_layer(self, output_size):
        self.model.add(Conv2D(
                        filters=output_size, 
                        kernel_size=(3, 3), 
                        padding='same', 
                        input_shape=(self.height, self.width, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D())

    def add_convolution_layer(self, output_size, filter, regularizer=None, dropout=0):
        self.model.add(Conv2D(
                        filters=output_size, 
                        kernel_size=filter, 
                        padding='same',
                        kernel_regularizer=regularizer))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D())
        if dropout:
            self.model.add(Dropout(dropout))

    def add_dense_layer(self, output_size, activation, regularizer=None, dropout=0):
        self.model.add(Dense(
                units=output_size,
                kernel_regularizer=regularizer))
        self.model.add(Activation(activation))
        if dropout:
            self.model.add(Dropout(dropout))
        

    def add_flatten_layer(self):
        self.model.add(Flatten())

    
    def compile_model(self, learning_rate, momentum, decay, optimizer=None):
        if not optimizer:
            optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'mse']
        )
        self.model.summary()


    def train_model(self, steps):
        return self.model.fit_generator(
                    self.training_data_generator,
                    steps_per_epoch=steps,
                    epochs=self.epochs,
                    validation_data=self.validation_data_generator,
                    validation_steps=steps
        )
    
    def save(self):
        model = self.model.to_json()
        with open('model.json', 'w') as json_file:
            json_file.write(model)
        self.model.save_weights("model.h5")



def display_result(history, model):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(cnn.epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.show()

if __name__ == "__main__":
    results = np.zeros(5)
    optimizers = ['rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax']
    o = ['rmsprop']
    top_model = None
    top_optimizer = None
    accuracy = 0
    for optimizer in o:
        cnn = CNN(128, 50, 64, 64)
        cnn.setup_image_generator()
        cnn.data_generator()
        cnn.compile_model(0.01, 0.8, 0.2, optimizer)
        history = cnn.train_model(15)
        cnn.save()
        display_result(history, cnn)



    