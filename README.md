
Medição de área de couro utilizando OpenCV

O vídeo foi capturado por camera de celular comum.
A primeira parte do código define pontos de referência no primeiro frame, para rotacionar e estabilizar os frames seguintes.
Cria-se uma máscara de comparação com 1 pixel de altura em Y e o comprimento da régua de medição, sendo a máscara aplicada aos frames seguintes.
A cada frame selecionam-se os pixels coincidentes com a cor do couro wet-blue que passam pela máscara. Estes pixels formam o contorno do couro. 
O fator [área / pixels] é calculado por amostragem de peças com áreas conhecidas.
A velocidade de deslocamento do couro foi considerada constante no trabalho. Para implantação final a velocidade do couro deve ser calculada via software.

