import cv2
from PIL import Image
import os

def processar_imagem_desmatamento(caminho_imagem):
    #Função para processar uma imagem e identificar áreas de desmatamento.

    nome_arquivo = os.path.basename(caminho_imagem)
    nome_mata = os.path.splitext(nome_arquivo)[0]
    
    # Redimensionar a imagem
    img = Image.open(caminho_imagem)
    img_resized = img.resize((800, 500))
    img_resized.save('img/matapequena.png')
    
    imagem_cv2 = cv2.imread('img/matapequena.png')
    imagem_cinza = cv2.cvtColor(imagem_cv2, cv2.COLOR_BGR2GRAY)
    
    bordas = cv2.Canny(imagem_cinza, 20, 200)
    
    # Encontrar os contornos
    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Desenhar os contornos na imagem original
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 0.5:
            cv2.drawContours(imagem_cv2, [contorno], -1, (0, 0, 255), 2)
    
    # Mostrar a imagem resultante e salvar
    cv2.imshow('Áreas Desmatadas', imagem_cv2)
    cv2.imwrite(f'img/{nome_mata}_contornos.png', imagem_cv2)
    
    # Fechar a janela de exibição
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    caminho_imagem = 'Path/to/image/to/analyze'
    processar_imagem_desmatamento(caminho_imagem)
