# 🕷️ Homens-Aranha em Foco: Classificador de Versões

Este projeto de Visão Computacional utiliza **Transfer Learning** com o modelo VGG16 para classificar imagens de diferentes versões do Homem-Aranha, distinguindo entre:

  * **Tobey Maguire** (Trilogia de Sam Raimi)
  * **Andrew Garfield** (Duologia *O Espetacular Homem-Aranha*)
  * **Tom Holland** (Universo Cinematográfico Marvel - MCU)

A ideia é demonstrar como modelos pré-treinados em grandes datasets podem ser adaptados e "ajustados" para tarefas específicas com datasets menores, obtendo alta precisão.

-----

## 🚀 Como Funciona

O projeto segue os seguintes passos:

1.  **Coleta e Pré-processamento de Dados:** As imagens dos diferentes Homens-Aranha são coletadas, redimensionadas para o formato esperado pelo VGG16 (224x224 pixels) e normalizadas.
2.  **Divisão do Dataset:** As imagens são divididas em conjuntos de treino, validação e teste para garantir uma avaliação justa do modelo.
3.  **Transfer Learning (Feature Extraction):** O modelo VGG16, pré-treinado no vasto dataset ImageNet, é carregado. Suas camadas convolucionais (que já aprenderam a identificar características genéricas em imagens) são "congeladas". Apenas uma nova camada de classificação é adicionada e treinada para distinguir os três Homens-Aranha. Isso permite que o modelo aproveite o conhecimento pré-existente do VGG16, mesmo com um dataset menor.
4.  **Treinamento:** A nova camada de classificação é treinada usando os dados preparados.
5.  **Previsão:** O modelo treinado é capaz de receber uma nova imagem e prever a qual Homem-Aranha ela pertence.

-----

## 🛠️ Tecnologias Utilizadas

  * **Python 3.x**
  * **Keras** (com TensorFlow como backend)
  * **TensorFlow**
  * **NumPy**
  * **Matplotlib**
  * **Pillow (PIL)**

-----

## 📋 Pré-requisitos

Antes de começar, certifique-se de ter o Python 3.x instalado em sua máquina.

É **altamente recomendado** usar um ambiente virtual para gerenciar as dependências do projeto.

1.  **Crie um ambiente virtual:**

    ```bash
    python -m venv venv
    ```

2.  **Ative o ambiente virtual:**

      * No Windows (PowerShell):
        ```bash
        .\venv\Scripts\Activate.ps1
        ```
      * No Windows (CMD):
        ```bash
        venv\Scripts\activate.bat
        ```
      * No Linux/macOS:
        ```bash
        source venv/bin/activate
        ```

    Você verá `(venv)` no início da sua linha de comando, indicando que o ambiente está ativo.

3.  **Instale as dependências:**
    Com o ambiente ativado, instale as bibliotecas necessárias:

    ```bash
    pip install tensorflow==2.15.0 numpy matplotlib pillow
    ```

    *Nota: `keras` já está incluído no `tensorflow` a partir das versões mais recentes.*

-----

## 🎬 Como Rodar o Projeto

Siga os passos abaixo para colocar o projeto em funcionamento:

### 1\. Obtenha o Dataset do Homem-Aranha

Você precisa de um conjunto de imagens para cada ator. Crie a pasta `spiderman_dataset/` na raiz do projeto e, dentro dela, crie as subpastas `Tobey_Maguire/`, `Andrew_Garfield/` e `Tom_Holland/`.

  * **Organização:**
    ```
    aranha_verso/
    └── spiderman_dataset/
        ├── Tobey_Maguire/
        │   ├── tobey_1.jpg
        │   └── ... (suas imagens do Tobey)
        ├── Andrew_Garfield/
        │   ├── andrew_1.jpg
        │   └── ... (suas imagens do Andrew)
        └── Tom_Holland/
            ├── tom_1.jpg
            └── ... (suas imagens do Tom)
    ```
  * **Dicas de Coleta:** Tente obter uma boa variedade de poses, ângulos e iluminação para cada ator (pelo menos 50-100 imagens por ator para um bom começo).

### 2\. Configure o Projeto

Abra o arquivo `main.py` e **atualize os caminhos** conforme necessário:

```python
# main.py

# ... (outras importações) ...

# Define o caminho para o seu dataset de imagens
DATASET_ROOT_DIR = 'spiderman_dataset' # <--- Verifique se este caminho está correto em relação a main.py!

# ... (resto do código) ...

# ATENÇÃO: Altere este caminho para uma imagem que você queira testar!
# Pode ser uma imagem do seu dataset de teste ou uma nova imagem.
example_image_path = os.path.join(DATASET_ROOT_DIR, 'Tom_Holland', 'tom_1.jpg') # <-- Atualize este caminho!

# ... (resto do código) ...
```

### 3\. Execute o Script Principal

Com o ambiente virtual ativado e os caminhos configurados, navegue até a pasta `projeto_homem_aranha/` no seu terminal e execute:

```bash
python main.py
```

O script fará o seguinte:

  * Carregará e pré-processará suas imagens.
  * Treinará o modelo de Transfer Learning.
  * Avaliará o modelo no conjunto de teste.
  * Fará uma previsão em uma imagem de exemplo (a que você definiu em `example_image_path`).

-----

## 📈 Resultados Esperados

Você deverá observar que o modelo de **Transfer Learning** (utilizando VGG16) atinge uma **acurácia significativamente maior** e treina mais rapidamente do que uma CNN simples construída do zero (se você descomentar e rodar a seção `model_baseline` em `main.py`). Isso demonstra a eficácia do aproveitamento de conhecimento de modelos pré-treinados.

-----

## 🤝 Contribuições

Sinta-se à vontade para contribuir com este projeto\! Ideias incluem:

  * Adicionar mais imagens para expandir o dataset.
  * Implementar **fine-tuning** em vez de apenas feature extraction.
  * Experimentar outros modelos pré-treinados (ResNet, Inception, MobileNet).
  * Adicionar aumento de dados para melhorar a robustez do modelo.
  * Criar uma interface gráfica simples para as previsões.

