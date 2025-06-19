import keras
from keras.model import Sequential
from keras.layers import Dense, Flatten, Activation, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

def build_and_train_baseline_model(x_train, y_train, x_val, y_val, x_test, y_test, num_classes, epochs=30, batch_size=128):
    print(" -- Rede Neural Convolucional --")

    model = Sequential()
    print("Dimensões de entrada esperada: ", x_train.shape[1:])

    model.add(Conv2D(32, (3,3), input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossemtropy',
                        optmizer='adam',
                        metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_val,y_val))
    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f'\nResultados: ')
    print(f'Loss no teste: {loss:.4f}')
    print(f'Acurácia no teste: {accuracy:.4f}')

    plt.figure(figsize=(12,4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss Treino')
    plt.plot(history.history['val_loss'], label='Loss Validação')
    plt.title('Loss ao longo das épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Acurácia Treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
    plt.title('Acurácia ao longo das épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model, history


