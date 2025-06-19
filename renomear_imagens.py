import os

def renomear_imagens():

    caminho = "C:\\Users\\polia\\Downloads\\aranha_verso\\spiderman_dataset\\Tom_Holland"

    new_name_base = "tom"

    extensoes = ['.jpg', '.png', '.jpg', '.jpeg']

    contador = 1

    print(f"Procurando ->  {caminho}")

    try:

        lista_arquivos = os.listdir(caminho)

        for nome_arquivo in lista_arquivos:
            extensao = os.path.splitext(nome_arquivo)[1].lower()

            if extensao in extensoes:
                caminho_old = os.path.join(caminho, nome_arquivo)

                new_name = f"{new_name_base}_{contador}{extensao}"

                caminho_new = os.path.join(caminho, new_name)

                os.rename(caminho_old, caminho_new)

                print(f"Renomeado: '{nome_arquivo}' para '{new_name}'")

                contador += 1
            
            print("\nDeu tudo certo queridaaaaaa!")

    except FileNotFoundError:
        print(f"\nNão achouuuuu '{caminho}' ")
    except Exception as e:
        print(f"\nDeu ruim e foi aqui {e} ")

renomear_imagens ()
        

