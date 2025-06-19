
import os
import matplotlib.pyplot as plt

from src.data_loader import load_and_preprocess_data, get_image
from src.model_baseline import build_and_train_baseline_model
from src.model_transfer_learning import build_and_train_model_transfer_learning_model
from src.predictor import make_prediction

DATASET_ROOT_DIR = 'spiderman_dataset'

def main():
    print("Iniciando o carregamento e pré-processamento dos dados...")
    try:
        x_train, y_train, x_val, y_val, x_test, y_test, categories, num_classes = \
            load_and_preprocess_data(DATASET_ROOT_DIR)
        print("Dados carregados e pré-processados com sucesso!")
    except ValueError as e:
        print(f"Erro ao carregar dados: {e}")
        print("Certifique-se de que o diretório DATASET_ROOT_DIR está correto e contém subpastas com imagens.")
        return 

    # model_baseline, history_baseline = build_and_train_baseline_model(
    #     x_train, y_train, x_val, y_val, x_test, y_test, num_classes,
    #     epochs=30, # Ajuste o número de épocas aqui
    #     batch_size=128
    # )
    # print("\nModelo de Linha de Base treinado e avaliado.")


    model_transfer, history_transfer = build_and_train_model_transfer_learning_model(
        x_train, y_train, x_val, y_val, x_test, y_test, num_classes,
        epochs=15, 
        batch_size=128
    )
    print("\nModelo de Transfer Learning treinado e avaliado.")

    print("\nRealizando uma previsão de exemplo...")

    example_image_path = os.path.join(DATASET_ROOT_DIR, 'Tom_Holland', 'tom_01.jpg') 
    if not os.path.exists(example_image_path):
        print(f"ATENÇÃO: A imagem de exemplo '{example_image_path}' não foi encontrada. "
              "Por favor, atualize 'example_image_path' no 'main.py' para um caminho de imagem válido.")
    else:
        make_prediction(model_transfer, example_image_path, categories)

    # if 'history_baseline' in locals() and 'history_transfer' in locals():
    #     plt.figure(figsize=(16, 4))
    #     plt.subplot(1, 2, 1)
    #     plt.plot(history_baseline.history['val_loss'], label='CNN do Zero Val Loss')
    #     plt.plot(history_transfer.history['val_loss'], label='VGG16 Transfer Val Loss')
    #     plt.title('Validation Loss Comparativa')
    #     plt.xlabel('Épocas')
    #     plt.legend()

    #     plt.subplot(1, 2, 2)
    #     plt.plot(history_baseline.history['val_accuracy'], label='CNN do Zero Val Accuracy')
    #     plt.plot(history_transfer.history['val_accuracy'], label='VGG16 Transfer Val Accuracy')
    #     plt.title('Validation Accuracy Comparativa')
    #     plt.xlabel('Épocas')
    #     plt.ylim(0, 1)
    #     plt.legend()
    #     plt.show()

if __name__ == "__main__":
    main()