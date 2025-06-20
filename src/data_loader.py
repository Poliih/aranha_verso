import os
import random
import numpy as np
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

def get_image(path, target_size=(224, 224)):
    img = image.load_img(path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x[0]

def load_and_preprocess_data(root_dir, train_split=0.7, val_split=0.15):
    categories = [x[0] for x in os.walk(root_dir) if x[0]][1:]
    categories = sorted([c for c in categories if os.path.basename(c) not in ['temp_folder', '.ipynb_checkpoints']])

    if not categories:
        raise ValueError(f"Opah que não achou '{root_dir}'")
    
    print(f"Encontrou -> {[os.path.basename(c) for c in categories]}")

    data = []
    for c_idx, category_path in enumerate(categories):
        images_in_category = [os.path.join(dp, f) for dp, dn, filenames in os.walk(category_path) 
                              for f in filenames if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']]

        print(f"Carregando {len(images_in_category)} para a categoria {os.path.basename(category_path)} hehe ")

        for img_path in images_in_category:
            try:
                _, x = get_image(img_path)
                data.append({'x': x, 'y': c_idx})
            except Exception as e:
                print(f"\nDeu ruim e foi aqui {img_path}: {e}")
                continue

    if not data:
        raise ValueError("Cadê imagem? Num tem kkk")

    num_classes = len(categories)
    print(f"\nTotal de imagens: {len(data)} de {num_classes} categorias")

    random.shuffle(data)

    idx_val = int(train_split * len(data))
    idx_test = int((train_split + val_split) * len(data))

    train = data[:idx_val]
    val = data[idx_val:idx_test]
    teste = data[idx_test:]

    x_train = np.array([t["x"] for t in train])
    y_train = [t["y"] for t in train]
    x_val = np.array([t["x"] for t in val])
    y_val = [t["y"] for t in val]
    x_test = np.array([t["x"] for t in teste])
    y_test = [t["y"] for t in teste]

    print(f'\nDivisão dos dados:')
    print(f'Treino: {len(x_train)} imagens')
    print(f'Validação: {len(x_val)} imagens')
    print(f'Teste: {len(x_test)} imagens')

    x_train = x_train.astype('float32') / 255.
    x_val = x_val.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print(f'\nFormato dos dados do treino: Imagens {x_train.shape}, Rótulos {y_train.shape}')

    return x_train, y_train, x_val, y_val, x_test, y_test, categories, num_classes
