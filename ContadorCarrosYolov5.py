import cv2
import torch
import time
import numpy
from PIL import ImageFont, ImageDraw, Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5n') # carrega o modelo
model.conf = 0.4

video = cv2.VideoCapture('video/cars.mp4')

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Dimensoes: {width}, {height}") # exibe as dimensões do vídeo

qtCars = 0
qtFrames = 0 # enumera cada frame em execução
nframeLeft, nframeRight = 0, 0 #

while True:
    sucess, frame = video.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # uso de uma fonte externa para exibir a quantidade de carros detectados (opcional fontes do próprio opencv)
    img_pil = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("font/BebasNeue-Regular.ttf", 36)
    draw.text((50, 50), f"Carros: {qtCars}", fill=(47, 82, 224), font=font)
    frame = cv2.cvtColor(numpy.array(img_pil), cv2.COLOR_RGB2BGR)

    results = model(frame_rgb)
    predictions = results.pred[0]

    for inf in predictions:

        x1, y1, x2, y2 = inf[:4] # coordenadas do ponto inicial e final do bounding box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # converte essas coordenadas para valores inteiros
        xc, yc = 0, 0 # coordendas do ponto central do objeto detectado
        classe = int(inf[5]) # pega o número da classe do objeto detectado

        if classe == 2: # desenha o bounding box apenas nos objetos da classe carro
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            name = results.names[classe] # pega o nome do objeto no array names de acordo com o número da classe

            cv2.putText(frame, name, (x1+20, y1-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1)

            xc, yc = int((x1+x2)/2), int((y1+y2)/2) # cálculo para obter as coordenadas do ponto central de acordo com as dimensões do bouding box

            cv2.circle(frame, (xc, yc), 2, (0, 0, 255), 2) # desenha o ponto central


        if classe == 2 and 450 < xc < 900 and 601 < yc < 610 and (qtFrames - nframeLeft) > 5: # houve a detecção de um automóvel na pista esquerda, pois o ponto central cruzou um certo trecho da tela, o qual contabiliza que mais um carro passou por essa rodovia

            qtCars += 1
            nframeLeft = qtFrames # captura o número do frame quando foi identificado um carro na pista esquerda, para termos um gerenciamento e inibir a detecção em frames muito próximos a esse, porque muito provavelmente será o mesmo carro, e consequentemente contabilizará falhamente um carro mais de uma vez
            # após a identificação de um automóvel a detecção só volta a funcionar depois de 5 frames a frente, para evitar contagens incorretas de um mesmo carro. Esse atraso é insignificante para gerar perdas na detecção de todos os automóveis

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # o bouding box muda de cor por alguns milisegundos para mostrar que ele foi identificado
            time.sleep(0.1)

        if classe == 2 and 910 < xc < 1280 and 601 < yc < 610 and (qtFrames - nframeRight) > 5: # houve a detecção de um automóvel na pista direita, pois o ponto central cruzou um certo trecho da tela, o qual contabiliza que mais um carro passou por essa rodovia

            qtCars += 1
            nframeRight = qtFrames # captura o número do frame quando foi identificado um carro na pista direita, para termos um gerenciamento e inibir a detecção em frames muito próximos a esse, porque muito provavelmente será o mesmo carro, e consequentemente contabilizará falhamente um carro mais de uma vez
            # após a identificação de um automóvel a detecção só volta a funcionar depois de 5 frames a frente, para evitar contagens incorretas de um mesmo carro. Esse atraso é insignificante para gerar perdas na detecção de todos os automóveis

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # o bouding box muda de cor por alguns milisegundos para mostrar que ele foi identificado
            time.sleep(0.1)


        # desenha a faixa na tela onde há a detecção dos carros
        # cv2.rectangle(frame, (450, 600), (900, 601), (0, 0, 255), 1)
        # cv2.rectangle(frame, (450, 610), (900, 611), (0, 0, 255), 1)
        #
        # cv2.rectangle(frame, (910, 600), (1280, 601), (0, 0, 255), 1)
        # cv2.rectangle(frame, (910, 610), (1280, 611), (0, 0, 255), 1)


    cv2.imshow('live', frame)

    qtFrames += 1 # mais um frame foi exibido

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()




