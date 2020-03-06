import cv2
import numpy as np
from PIL import Image
import math

def rotate(image, angle, scale = 1.):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

font = cv2.FONT_HERSHEY_SIMPLEX 

cap = cv2.VideoCapture('leather.mp4')
ret,old = cap.read()
ang = 180*math.atan(22/310)/math.pi
old = rotate(old,-ang,1.)
old_gray = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
old_gray_roi = old_gray[50:180,0:620]

black = np.zeros(old.shape[:2], dtype = "uint8")
leather = np.zeros((352,640,3), dtype = "uint8")
board = leather.copy()

yellow = np.zeros((352,640,3), dtype = "uint8")
#yellow[np.where((yellow==[0,0,0]).all(axis=2))] = [110,255,255]
#yellow[np.where((yellow==[0,0,0]).all(axis=2))] = [110,110,255]
#yellow[np.where((yellow==[0,0,0]).all(axis=2))] = [80,255,80]
yellow[np.where((yellow==[0,0,0]).all(axis=2))] = [255,180,150]
#mask_line = cv2.line(black, (90,200), (400,178), (255,255,255), 1)
mask_line = cv2.line(black, (90,178), (400,178), (255,255,255), 2)

fourcc = cv2.VideoWriter_fourcc(*'MPEG') # Cria o objeto para gravar vídeo
out = cv2.VideoWriter('leather_estabilizado.mp4', fourcc, 31.0, (600 , 300))  # Determina o nome do arquivo de saída, sua taxa de FPS e sua resolução.
#out = cv2.VideoWriter('leather_estabilizado.mp4', fourcc, 31.0, (640 , 352))  # Determina o nome do arquivo de saída, sua taxa de FPS e sua resolução.

transforms = [np.identity(3)]
pontos_chave_old = cv2.goodFeaturesToTrack(old_gray_roi, 100, 0.0001, 10)
pontos_chave_old_b = np.int0(pontos_chave_old)

count = 0
while (ret):
    
    ret,frame = cap.read()
    if not(frame is None):
        oframe = frame.copy()
        frame = rotate(frame,-ang,1.)
        atual = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        atual_roi = atual[50:180,0:620]
        copy = frame.copy()

        for i in pontos_chave_old_b:
            x,y = i.ravel()
            cv2.circle(copy,(x+00,y+50),3,(0,255,255),-1)

        pontos_chave_atual, status, _ = cv2.calcOpticalFlowPyrLK(old_gray_roi, atual_roi, pontos_chave_old, np.array([]))
        pontos_chave_old, pontos_chave_atual = map(lambda pontos_chave_old_b: pontos_chave_old_b[status.ravel().astype(bool)], [pontos_chave_old, pontos_chave_atual])

        transform,_ = cv2.estimateAffinePartial2D(pontos_chave_old, pontos_chave_atual, True)
        height, width = frame.shape[0], frame.shape[1]
        last_transform = np.identity(3)
        transform = transform.dot(last_transform)
        transformado = cv2.warpAffine(frame, transform, (width, height))
        inverse_transform = cv2.invertAffineTransform(transform[:2])
        estabilizado = cv2.warpAffine(frame, inverse_transform, (width, height))
        #estabilizado = cv2.line(estabilizado, (90,195), (400,173), (255,100,100), 1) 
        
        oframe = estabilizado.copy()
        final_blur = cv2.blur(oframe, (13,13)) 
        oframe[0:75,0:640]=final_blur[0:75,0:640]
        
        # ************** PROCESSAMENTO DA ÁREA *************************
        
        img_line = cv2.bitwise_and(estabilizado, estabilizado, mask = mask_line)
        lower = np.array([110,110,110], dtype=np.uint8)
        upper = np.array([250, 250, 250], dtype=np.uint8)
        mask_leather = cv2.inRange(img_line, lower, upper)
        invert_mask = np.invert(mask_leather)
        
        y_leather = cv2.bitwise_and(yellow, yellow, mask = mask_leather)
        
        if(cv2.countNonZero(mask_leather)>0):
            if (count == 0):
                area = 0
                leather = np.zeros((352,640,3), dtype = "uint8")
            #leather[320-count:321-count,320:630] = y_leather[178:179,90:400]
            leather[50+count:51+count,320:630] = y_leather[178:179,90:400]
            count += 1
            area += cv2.countNonZero(mask_leather)
        else:
            count = 0
            #print(area)
        
        cv2.rectangle(board, (450, 80), (620, 300),(100, 100, 100), -1)
        cv2.addWeighted(board, -0.85, oframe, 1, 0, oframe)     
        
        nleather = cv2.resize(leather,(320,176))
        oframe[110:270,460:615] = nleather[10:170,160:315]
        
        oframe = cv2.bitwise_and(oframe, oframe, mask = invert_mask)
        oframe = cv2.addWeighted(oframe,1,y_leather,1,0)
        
        varea = area / 2000
        cv2.putText(oframe, 'area: '+str(round(varea, 2))+'ft2', (475,100), font,.5, [255,255,255], 1, cv2.LINE_AA) # Imprime o texto das coordenadas
                
        # *************** FIM PROCESSAMENTO ÁREA **********************
        
        estabilizado_crop = oframe[32:332,20:620]
        out.write(estabilizado_crop)
        last_transform = transform

        final1 = cv2.hconcat([frame,oframe])
        final2 = cv2.hconcat([copy, estabilizado])
        final = cv2.vconcat([final1,final2])
    
    cv2.imshow('resultado',final)
    k = cv2.waitKey(1)
    if k == ord('q'):
        exit()
        out.release()

cap.release()
out.release()
print('fim processamento')
