import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

def build_and_train_model_transfer_learning_model(x_train, y_train, x_val, y_val, x_test, y_test, num_classes, epochs=30, batch_size=128):
    print(" -- Aplicando Transfer Learning com VGG16 --")

    vgg = VGG16(weights='imagenet', include_top=True)
    vgg.summary()

    inp = vgg.input

    vgg_features = vgg.layers[-2].output

    new_classification_layer = Dense(num_classes, activation='softmax', name='spiderman_classifier')
    out = new_classification_layer(vgg_features)

    model = Model(inp, out)

    for layer in model.layers[:-1]:
        layer.trainable = False
    model.layers[1].trainable = True

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',  
                  metrics=['accuracy'])  
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,  
                        validation_data=(x_val, y_val))
    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f'\nResultado do Transfer Learning:')
    print(f'Loss no teste: {loss:.4f}')
    print(f'Acurácia no teste: {accuracy:.4f}')

    return model, history
