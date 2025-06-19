import os
from PIL import Image

pasta_origem = "C:\\Users\\polia\Downloads\\aranha_verso\\spiderman_dataset\\Tom_Holland_color"
pasta_destino = "C:\\Users\\polia\Downloads\\aranha_verso\\spiderman_dataset\\Tom_Holland"


for nome_arquivo in os.listdir(pasta_origem):
    if nome_arquivo.lower().endswith(('.jpg', '.png', '.jpg', '.jpeg')):

        caminho_origem = os.path.join(pasta_origem,nome_arquivo)
        caminho_destino = os.path.join(pasta_destino,nome_arquivo)

        try:
            imagem = Image.open(caminho_origem)
            imagem_cinza = imagem.convert('L')
            imagem_cinza.save(caminho_destino)
            print(f"Convertido -> Gloria a Deus kkkk {nome_arquivo}")
        except Exception as e:
            print(f"deuruim oremos: {nome_arquivo}: {e}")

print(f'\nProcesso deu bom <3')