
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

    model_transfer, history_transfer = build_and_train_model_transfer_learning_model(
        x_train, y_train, x_val, y_val, x_test, y_test, num_classes,
        epochs=15, 
        batch_size=128
    )
    print("\nModelo de Transfer Learning treinado e avaliado.")

    print("\nRealizando uma previsão de exemplo...")

    example_image_path = os.path.join(DATASET_ROOT_DIR, 'ex.jpg') 
    if not os.path.exists(example_image_path):
        print(f"ATENÇÃO: A imagem de exemplo '{example_image_path}' não foi encontrada. "
              "Por favor, atualize 'example_image_path' no 'main.py' para um caminho de imagem válido.")
    else:
        make_prediction(model_transfer, example_image_path, categories)


if __name__ == "__main__":
    main()