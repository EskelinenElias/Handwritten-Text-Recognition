# A neural network classifier for recognizing handwritten digits
# Sources: 
# - Tensorflow docs: https://www.tensorflow.org/tutorials/quickstart/beginner
# - Tensorflow optimizer: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers

import os
import tensorflow as tf
from keras import layers, models, optimizers, losses
from keras.datasets import mnist
from keras.utils import to_categorical
import plotly.graph_objects as go

if __name__ == "__main__": 

    # define model parameters
    loss_function = 'categorical_crossentropy'
    optimizer_function = optimizers.Adam(learning_rate=0.001)
    training_epochs = 10
    model_layers = [
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ]
    model_path = "./trained_models/trained_model.keras"
    figures_dir = "./figures"

    # load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255   
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # build a neural network model
    model = models.Sequential(model_layers)

    # compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # train the model
    history = model.fit(train_images, train_labels, epochs=training_epochs, batch_size=64, validation_data=(test_images, test_labels))

    # save the trained model
    models.save_model(model, model_path)


### PLOTS ###########################################################################################################

    # plot training and validation accuracy over epochs
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=list(range(1, len(history.history['accuracy']) + 1)),
                                y=history.history['accuracy'],
                                mode='lines',
                                name='Training accuracy'))
    figure.add_trace(go.Scatter(x=list(range(1, len(history.history['val_accuracy']) + 1)),
                                y=history.history['val_accuracy'],
                                mode='lines',
                                name='Validation accuracy'))
    figure.update_layout(title='Training and validation accuracy over epochs',
                         xaxis=dict(title='Epoch'),
                         yaxis=dict(title='Accuracy'),
                         legend=dict(x=0, y=1, traceorder='normal'))
    figure.write_html(os.path.join(figures_dir, 'accuracy.html'))
    figure.write_image(os.path.join(figures_dir, 'accuracy.png'))

    # plot training and testing loss over epochs
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=list(range(1, len(history.history['loss']) + 1)),
                                y=history.history['loss'],
                                mode='lines',
                                name='Training loss'))
    figure.add_trace(go.Scatter(x=list(range(1, len(history.history['val_loss']) + 1)),
                                y=history.history['val_loss'],
                                mode='lines',
                                name='Validation loss'))
    figure.update_layout(title='Training and validation loss over epochs',
                         xaxis=dict(title='Epoch'),
                         yaxis=dict(title='Loss'),
                         legend=dict(x=0, y=1, traceorder='normal'))
    figure.write_html(os.path.join(figures_dir, 'loss.html'))
    figure.write_image(os.path.join(figures_dir, 'loss.png'))
