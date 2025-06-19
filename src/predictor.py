import os
import numpy as np
import matplotlib.pyplot as plt
from .data_loader import get_image

def make_prediction(model, image_path, categories):

    print("\n--- Fazendo Previsões ---")

    try:
        img_display, x_predict = get_image(image_path)

        probabilities = model.predict(np.expand_dims(x_predict, axis=0))

        predicted_class_idx = np.argmax(probabilities[0])
        predicted_class_label = os.path.basename(categories[predicted_class_idx]) 
        plt.imshow(img_display)
        plt.title(f"Previsão: {predicted_class_label} (Probabilidade: {probabilities[0][predicted_class_idx]*100:.2f}%)")
        plt.axis('off')
        plt.show()

        print(f"Probabilidades para cada Homem-Aranha:")
        for i, prob in enumerate(probabilities[0]):
            print(f"  {os.path.basename(categories[i])}: {prob*100:.2f}%")

    except FileNotFoundError:
        print(f"Erro: Imagem não encontrada em '{image_path}'. Por favor, verifique o caminho.")
    except Exception as e:
        print(f"Ocorreu um erro ao processar a imagem '{image_path}': {e}")