from __future__ import print_function
#import potrebnih biblioteka
import cv2
from scipy.misc import imread, imresize
import numpy as np
from numpy import arccos, array, dot, pi, cross
from numpy.linalg import norm
import matplotlib.pyplot as plt
import collections
import math

# keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD

import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 16, 12 # za prikaz većih slika i plotova, zakomentarisati ako nije potrebno

class Broj:
    def __init__(self,vrednost, x,y,w,h,uracunatP, uracunatZ, pronadjen, broj_prop_frejmova):
        self.vrednost = vrednost
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.uracunatP = uracunatP
        self.uracunatZ = uracunatZ
        self.pronadjen = pronadjen
        self.broj_prop_frejmova = broj_prop_frejmova

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def invert(image):
    return 255-image

def display_image(image):
    cv2.imshow("result", image)
    cv2.waitKey(0)

def dilate(image):
    kernel = np.ones((2,2)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((2,2)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def distance_numpy(A, B, P):
    if all(A == P) or all(B == P):
        return 0
    if arccos(dot((P - A) / norm(P - A), (B - A) / norm(B - A))) > pi / 2:
        return norm(P - A)
    if arccos(dot((P - B) / norm(P - B), (A - B) / norm(A - B))) > pi / 2:
        return norm(P - B)
    return norm(cross(A-B, A-P))/norm(B-A)

def transform_img(img):
    h, w = img.shape
    ret = np.ones((28,28), dtype=np.uint8)
    ret.fill(255)
    if h > 26 or w > 26:
        tmp = 28
        if h > w:
            tmp = h
        else:
            tmp = w
        odnos = 28/tmp
        img = cv2.resize(img, (int(w*odnos), int(h*odnos)), interpolation = cv2.INTER_NEAREST)
        h,w = img.shape
        ret[0:h, 0:w] = img
    else:
        ret[2:h+2, 2:w+2] = img
    #cv2.imwrite(r'C:\Users\MardzaPC\Desktop\soft-project\slika.png', ret)
    return ret

def select_roi(image_orig, image_bin):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = [] # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)

        if area > 40 and area < 500 and h > 15 and h < 40 and w < 30:
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
            global niz_brojeva
            global frame_num

            for obj in niz_brojeva:
                obj.pronadjen = False

            pronadjen_postojeci = False
            for obj in niz_brojeva:
                if abs(obj.x - x) <= 15 and abs(obj.y - y) <= 15:
                    obj.x = x
                    obj.y = y
                    obj.pronadjen = True
                    pronadjen_postojeci = True

            for obj in niz_brojeva:
                if obj.pronadjen == False:
                    obj.broj_prop_frejmova += 1

            if pronadjen_postojeci == False and not (x > 615 or y > 440):
                lista_reg = []
                region = image_bin[y:y+h+1,x:x+w+1]
                ts = transform_img(region)
                lista_reg.append([ts, (x,y,w,h)])
                regija = np.copy(ts)
                regija = np.invert(regija)
                regija = regija.reshape(1,28,28,1)
                regija = regija.astype('float32')
                regija /= 255

                out = model.predict(regija)
                vrednost_broja = np.argmax(out)
                #print(vrednost_broja)
                niz_brojeva.append(Broj(vrednost_broja,x,y,w,h,False,False,False,0))

    for obj in niz_brojeva:
        cv2.putText(image_orig, str(obj.vrednost),
                            (obj.x, obj.y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255))
        if obj.y > 440 or obj.x > 615:
            niz_brojeva.remove(obj)
            break

    for obj in niz_brojeva:
        global suma
        p1 = np.array([(koordinate_plave[0])[0], (koordinate_plave[0])[1]])
        p2 = np.array([(koordinate_plave[0])[2], (koordinate_plave[0])[3]])
        z1 = np.array([(koordinate_zelene[0])[0], (koordinate_zelene[0])[1]])
        z2 = np.array([(koordinate_zelene[0])[2], (koordinate_zelene[0])[3]])
        p = np.array([obj.x, obj.y])
        d1 = distance_numpy(p1,p2,p)
        d2 = distance_numpy(z1,z2,p)

        objX = obj.x + obj.w
        objY = obj.y + obj.h

        if obj.uracunatP == False:
            x1 = (koordinate_plave[0])[0]
            x2 = (koordinate_plave[0])[2]
            y1 = (koordinate_plave[0])[1]
            y2 = (koordinate_plave[0])[3]

            if y1 < y2:
                tmpY = y2
                y2 = y1
                y1 = tmpY
            if x1 > x2:
                tmpX = x2
                x2 = x1
                x1 = tmpX
            if d1 < 20 and objX >= x1 and objX <= x2 and objY <= y1 and objY >= y2:
                suma += obj.vrednost
                obj.uracunatP = True
        if obj.uracunatZ == False:
            x1 = (koordinate_zelene[0])[0]
            x2 = (koordinate_zelene[0])[2]
            y1 = (koordinate_zelene[0])[1]
            y2 = (koordinate_zelene[0])[3]

            if y1 < y2:
                tmpY = y2
                y2 = y1
                y1 = tmpY
            if x1 > x2:
                tmpX = x2
                x2 = x1
                x1 = tmpX
            if d2 < 20 and objX >= x1 and objX <= x2 and objY <= y1 and objY >= y2:
                suma -= obj.vrednost
                #print(suma)
                obj.uracunatZ = True

    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]

    return image_orig, sorted_regions

def scale_to_range(image):
    return image/255

def matrix_to_vector(image):
    return image.flatten()

def pripremi_sliku(frame):
    image_color = frame
    img = invert(image_bin(image_gray(image_color)))
    img_bin = erode(img)
    selected_regions, numbers = select_roi(image_color.copy(), img_bin)
    return selected_regions, numbers

def average_lines(image, lines):
    avg_line = np.average(lines, axis = 0)
    (avg_line[0])[0] -= 5
    (avg_line[0])[1] -= 8
    (avg_line[0])[2] -= 5
    (avg_line[0])[3] -= 8
    return avg_line

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cv2.line(line_image, (x1,y1), (x2,y2), (0, 0, 255), 2)
    return line_image

def pronadji_linije(copy, flag):
    gray = cv2.cvtColor(copy, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(canny, 2, np.pi/180, 100, np.array([]), minLineLength=180, maxLineGap=5)
    avr_lines = average_lines(frame, lines)
    if flag:
        global koordinate_plave
        koordinate_plave = avr_lines
    else:
        global koordinate_zelene
        koordinate_zelene = avr_lines
    #print("Plave" ,koordinate_plave)
    #print("Zelene" ,koordinate_zelene)
    line_image = display_lines(frame, avr_lines)
    return line_image

def izoluj_liniju(img, color):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([])
    upper = np.array([])
    if color == 'blue':
        lower = np.array([100,50,50])
        upper = np.array([255,255,255])
    if color == 'green':
        lower = np.array([50,100,50])
        upper = np.array([115,255,255])

    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    return res

#Ucitavanje obucene NM
from keras.models import load_model
model = load_model('cnn.h5')

sume = []

for i in range(0,10):
    suma = 0
    frame_num = 0
    flag = True
    #Koordinate linija
    koordinate_plave = []
    koordinate_zelene = []

    #Niz objekata bqrojeva
    niz_brojeva = []

    cap = cv2.VideoCapture('video/video-'+str(i)+'.avi')
    if(cap.isOpened() == False):
        print('Error opening video file')

    while(cap.isOpened()):
        frame_num += 1
        ret, frame = cap.read()
        if ret == True:
            copy = np.copy(frame)
            height, width, c = copy.shape
            if flag:
                plava_linija = izoluj_liniju(copy, 'blue')
                zelena_linija = izoluj_liniju(copy, 'green')
                zelena = pronadji_linije(zelena_linija, False)
                plava = pronadji_linije(plava_linija, True)
                flag = False

            combo_lines = cv2.addWeighted(zelena, 1, plava, 1, 1)
            combo_image = cv2.addWeighted(frame, 0.8, combo_lines, 1, 1)
            img, n = pripremi_sliku(combo_image)
            #cv2.putText(img, 'Suma:  ' + str(suma), (410, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)
            #cv2.imshow('Video-' + str(i), img)
            cv2.putText(frame, 'Suma:  ' + str(suma), (410, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 255), 2)
            cv2.imshow('Video-' + str(i), frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    sume.append(suma)
    #print(suma)
    cap.release()
    cv2.destroyAllWindows()


file = open("out.txt","w")
file.write("RA 156/2015 Marijana Negru\r")
file.write("file	sum\r")
file.write('video-0.avi\t' + str(sume[0]) +'\r')
file.write('video-1.avi\t' + str(sume[1]) +'\r')
file.write('video-2.avi\t' + str(sume[2]) +'\r')
file.write('video-3.avi\t' + str(sume[3]) +'\r')
file.write('video-4.avi\t' + str(sume[4]) +'\r')
file.write('video-5.avi\t' + str(sume[5]) +'\r')
file.write('video-6.avi\t' + str(sume[6]) +'\r')
file.write('video-7.avi\t' + str(sume[7]) +'\r')
file.write('video-8.avi\t' + str(sume[8]) +'\r')
file.write('video-9.avi\t' + str(sume[9]) +'\r')
file.close()
