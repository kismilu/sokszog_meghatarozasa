import numpy as np
import cv2
import random

#kepek
KEPEK = ["teszt1.jpg","teszt2.jpg","teszt3.jpg","teszt4.jpg","teszt5.jpg","teszt6.jpg"]

#szinek
COLORS = [(255, 255, 0), (255, 200, 0), (255, 150, 0), (255, 100, 0), (255, 50, 0), (200, 0, 255),
          (150, 0, 255), (100, 0, 255), (50, 0, 255), (0, 200, 255), (0, 150, 255), (0, 100, 255),
          (200, 255, 0), (150, 255, 0), (100, 255, 0), (50, 255, 0),]

#random szinek kiválasztasa
def random_color():
    return random.choice(COLORS)

#kepek bejarasa
for i in KEPEK:
    im = cv2.imread('.\\tesztkepek\\'+i)       #kep beolvasasa
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)     #kep atalakitasa szurkere

    noise = np.zeros(gray.shape, np.int16)          #mesterseges zaj letrehozasa szruke kephez
    cv2.randn(noise, 0.0, 20.0)                     #mesterseges zaj random generalasa szruke kephez

    imnoise = cv2.add(gray, noise, dtype=cv2.CV_8UC1)               #mesterseges zaj hozzaadasa
    denoise = cv2.fastNlMeansDenoising(imnoise, gray, 20, 21, 20)   #mesterseges zaj szurese

    ret, thresh = cv2.threshold(denoise, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)        #konturok, hiearachia megkeresese

    print('\n')
    print('Vizsgált kép:',i)

    for n, cnt in enumerate(contours):
        if(n < len(contours)-1):
            approx = cv2.approxPolyDP(
                cnt, 0.02 * cv2.arcLength(cnt, True), True      #gorbe csucsainak meghatarozása
            )
            print('Az alakzat csúcsainak száma:', len(approx))

            for x in range(len(approx)):                        #csucsok bejarasa
                szin = random_color()                           #random szin generalasa
                print('(x,y)= (',approx[x][0][0],',',approx[x][0][1], ') RGB:',szin)        #koordinatak kiiratasa
                cv2.circle(im, (approx[x][0][0], approx[x][0][1]), 5, szin, -1, 20, 0)      #pontok rajzolasa

    cv2.imshow('vegeredmeny', im)       #vegeredmeny kiiratasa
    cv2.waitKey(0)

cv2.destroyAllWindows()
