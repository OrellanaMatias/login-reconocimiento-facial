# Librerias
from tkinter import *
import cv2
import numpy as np
from PIL import Image, ImageTk
import imutils
import mediapipe as mp
import math
import os
import face_recognition as fr
from ultralytics import YOLO

def Code_Face(images):
    listacod = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cod = fr.face_encodings(img)[0]
        listacod.append(cod)

    return listacod

def Close_Windows():
    global step, conteo
    conteo = 0
    step = 0
    pantalla2.destroy()

def Close_Windows2():
    global step, conteo
    conteo = 0
    step = 0
    pantalla3.destroy()

def Object_Detection(img):
    global glass, capHat
    glass = False
    capHat = False

    frame = img

    clsNameCap = ['Gafas', 'Sombrero', 'Abrigo', 'Camisa', 'Pantalones', 'Shorts', 'Falda', 'Vestido', 'Maleta','Zapato']
    clsNameGlass = ['Gafas']

    resultsCap = modelCap(frame, stream=True, imgsz=640)
    resultsGlass = modelGlass(frame, stream=True, imgsz=640)

    for resCap in resultsCap:
        boxesCap = resCap.boxes
        for boxCap in boxesCap:
            xi1, yi1, xf1, yf1 = boxCap.xyxy[0]
            xi1, yi1, xf1, yf1 = int(xi1), int(yi1), int(xf1), int(yf1)

            if xi1 < 0: xi1 = 0
            if yi1 < 0: yi1 = 0
            if xf1 < 0: xf1 = 0
            if yf1 < 0: yf1 = 0

            clsCap = int(boxCap.cls[0])

            confCap = math.ceil(boxCap.conf[0])

            if clsCap == 1:
                capHat = True
                cv2.rectangle(frame, (xi1, yi1), (xf1, yf1), (255, 255, 0), 2)
                cv2.putText(frame, f"{clsNameCap[clsCap]} {int(confCap * 100)}%", (xi1, yi1 - 20),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
    for resGlass in resultsGlass:
        boxesGlass = resGlass.boxes
        for boxGlass in boxesGlass:
            xi2, yi2, xf2, yf2 = boxGlass.xyxy[0]
            xi2, yi2, xf2, yf2 = int(xi2), int(yi2), int(xf2), int(yf2)

            if xi2 < 0: xi2 = 0
            if yi2 < 0: yi2 = 0
            if xf2 < 0: xf2 = 0
            if yf2 < 0: yf2 = 0

            clsGlass = int(boxGlass.cls[0])

            confGlass = math.ceil(boxGlass.conf[0])

            if clsGlass == 0:
                glass = True
                cv2.rectangle(frame, (xi2, yi2), (xf2, yf2), (255, 0, 255), 2)
                cv2.putText(frame, f"{clsNameGlass[clsGlass]} {int(confGlass * 100)}%", (xi2, yi2 - 20),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

    return frame


def Profile():
    global step, conteo, UserName, OutFolderPathUser
    conteo = 0
    step = 0

    pantalla4 = Toplevel(pantalla)
    pantalla4.title("BIOMETRIC SIGN")
    pantalla4.geometry("1280x720")

    back = Label(pantalla4, image=imagenB, text="Back")
    back.place(x=0, y=0, relwidth=1, relheight=1)

    UserFile = open(f"{OutFolderPathUser}/{UserName}.txt", 'r')
    InfoUser = UserFile.read().split(',')
    Name = InfoUser[0]
    User = InfoUser[1]
    Pass = InfoUser[2]
    UserFile.close()

    if User in clases:
        texto1 = Label(pantalla4, text=f"BIENVENIDO {Name}")
        texto1.place(x=580, y=50)
        lblImgUser = Label(pantalla4)
        lblImgUser.place(x=490, y=80)

        PosUserImg = clases.index(User)
        UserImg = images[PosUserImg]

        ImgUser = Image.fromarray(UserImg)
        ImgUser = cv2.imread(f"{OutFolderPathFace}/{User}.png")
        ImgUser = cv2.cvtColor(ImgUser, cv2.COLOR_RGB2BGR)
        ImgUser = Image.fromarray(ImgUser)
        IMG = ImageTk.PhotoImage(image=ImgUser)

        lblImgUser.configure(image=IMG)
        lblImgUser.image = IMG


def Log_Biometric():
    global pantalla, pantalla2, conteo, parpadeo, img_info, step, glass, capHat

    if cap is not None:
        ret, frame = cap.read()
        frameSave = frame.copy()

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frameObject = frame.copy()

        if ret == True:
            res = FaceMesh.process(frameRGB)
            frame = Object_Detection(frameObject)
            px = []
            py = []
            lista = []
            r = 5
            t = 3

            if res.multi_face_landmarks:
                for rostros in res.multi_face_landmarks:

                    mpDraw.draw_landmarks(frame, rostros, FacemeshObject.FACE_CONNECTIONS, ConfigDraw, ConfigDraw)

                    for id, puntos in enumerate(rostros.landmark):

                        al, an, c = frame.shape
                        x, y = int(puntos.x * an), int(puntos.y * al)
                        px.append(x)
                        py.append(y)
                        lista.append([id, x, y])

                        if len(lista) == 468:
                            x1, y1 = lista[145][1:]
                            x2, y2 = lista[159][1:]
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            longitud1 = math.hypot(x2 - x1, y2 - y1)

                            x3, y3 = lista[374][1:]
                            x4, y4 = lista[386][1:]
                            cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
                            longitud2 = math.hypot(x4 - x3, y4 - y3)

                            x5, y5 = lista[139][1:]
                            x6, y6 = lista[368][1:]

                            x7, y7 = lista[70][1:]
                            x8, y8 = lista[300][1:]

                            faces = detector.process(frameRGB)

                            if faces.detections is not None:
                                for face in faces.detections:

                                    score = face.score
                                    score = score[0]
                                    bbox = face.location_data.relative_bounding_box

                                    if score > confThreshold:
                                        alimg, animg, c = frame.shape

                                        xi, yi, an, al = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                                        xi, yi, an, al = int(xi * animg), int(yi * alimg), int(
                                            an * animg), int(al * alimg)

                                        offsetan = (offsetx / 100) * an
                                        xi = int(xi - int(offsetan/2))
                                        an = int(an + offsetan)
                                        xf = xi + an

                                        offsetal = (offsety / 100) * al
                                        yi = int(yi - offsetal)
                                        al = int(al + offsetal)
                                        yf = yi + al

                                        if xi < 0: xi = 0
                                        if yi < 0: yi = 0
                                        if an < 0: an = 0
                                        if al < 0: al = 0

                                    if step == 0 and glass == False and capHat == False:
                                        cv2.rectangle(frame, (xi, yi, an, al), (255, 0, 255), 2)
                                        alis0, anis0, c = img_step0.shape
                                        frame[50:50 + alis0, 50:50 + anis0] = img_step0

                                        alis1, anis1, c = img_step1.shape
                                        frame[50:50 + alis1, 1030:1030 + anis1] = img_step1

                                        alis2, anis2, c = img_step2.shape
                                        frame[270:270 + alis2, 1030:1030 + anis2] = img_step2

                                        if x7 > x5 and x8 < x6:

                                            if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False:  
                                                conteo = conteo + 1
                                                parpadeo = True

                                            elif longitud1 > 10 and longitud2 > 10 and parpadeo == True:  
                                                parpadeo = False

                                            alich, anich, c = img_check.shape
                                            frame[165:165 + alich, 1105:1105 + anich] = img_check

                                            cv2.putText(frame, f'Parpadeos: {int(conteo)}', (1070, 375), cv2.FONT_HERSHEY_COMPLEX,0.5,
                                                        (255, 255, 255), 1)


                                            if conteo >= 3:
                                                alich, anich, c = img_check.shape
                                                frame[385:385 + alich, 1105:1105 + anich] = img_check

                                                if longitud1 > 14 and longitud2 > 14:
                                                    cut = frameSave[yi:yf, xi:xf]
                                                    cv2.imwrite(f"{OutFolderPathFace}/{RegUser}.png", cut)
                                                    step = 1
                                        else:
                                            conteo = 0

                                    if step == 1 and glass == False and capHat == False:
                                        cv2.rectangle(frame, (xi, yi, an, al), (0, 255, 0), 2)
                                        allich, anlich, c = img_liche.shape
                                        frame[50:50 + allich, 50:50 + anlich] = img_liche

                                    if glass == True:
                                        algla, angla, c = img_glass.shape
                                        frame[50:50 + algla, 50:50 + angla] = img_glass

                                    if capHat == True:
                                        alcap, ancap, c = img_cap.shape
                                        frame[50:50 + alcap, 50:50 + ancap] = img_cap

                            close = pantalla2.protocol("WM_DELETE_WINDOW", Close_Windows)

            frame = imutils.resize(frame, width=1280)

            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)

            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, Log_Biometric)

        else:
            cap.release()

def Sign_Biometric():
    global pantalla, pantalla3, conteo, parpadeo, img_info, step, UserName, prueba

    if cap is not None:
        ret, frame = cap.read()

        frameCopy = frame.copy()

        frameFR = cv2.resize(frameCopy, (0, 0), None, 0.25, 0.25)

        rgb = cv2.cvtColor(frameFR, cv2.COLOR_BGR2RGB)

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frameObject = frame.copy()

        if ret == True:

            res = FaceMesh.process(frameRGB)
            frame = Object_Detection(frameObject)

            px = []
            py = []
            lista = []
            r = 5
            t = 3

            if res.multi_face_landmarks:
                for rostros in res.multi_face_landmarks:

                    mpDraw.draw_landmarks(frame, rostros, FacemeshObject.FACE_CONNECTIONS, ConfigDraw, ConfigDraw)

                    for id, puntos in enumerate(rostros.landmark):

                        al, an, c = frame.shape
                        x, y = int(puntos.x * an), int(puntos.y * al)
                        px.append(x)
                        py.append(y)
                        lista.append([id, x, y])

                        if len(lista) == 468:
                            x1, y1 = lista[145][1:]
                            x2, y2 = lista[159][1:]
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            longitud1 = math.hypot(x2 - x1, y2 - y1)

                            x3, y3 = lista[374][1:]
                            x4, y4 = lista[386][1:]
                            cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
                            longitud2 = math.hypot(x4 - x3, y4 - y3)

                            x5, y5 = lista[139][1:]
                            x6, y6 = lista[368][1:]

                            x7, y7 = lista[70][1:]
                            x8, y8 = lista[300][1:]

                            faces = detector.process(frameRGB)

                            if faces.detections is not None:
                                for face in faces.detections:

                                    score = face.score
                                    score = score[0]
                                    bbox = face.location_data.relative_bounding_box

                                    if score > confThreshold:
                                        alimg, animg, c = frame.shape

                                        xi, yi, an, al = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                                        xi, yi, an, al = int(xi * animg), int(yi * alimg), int(
                                            an * animg), int(al * alimg)

                                        offsetan = (offsetx / 100) * an
                                        xi = int(xi - int(offsetan/2))
                                        an = int(an + offsetan)
                                        xf = xi + an

                                        offsetal = (offsety / 100) * al
                                        yi = int(yi - offsetal)
                                        al = int(al + offsetal)
                                        yf = yi + al

                                        if xi < 0: xi = 0
                                        if yi < 0: yi = 0
                                        if an < 0: an = 0
                                        if al < 0: al = 0

                                        if step == 0 and glass == False and capHat == False:
                                            cv2.rectangle(frame, (xi, yi, an, al), (255, 0, 255), 2)
                                            alis0, anis0, c = img_step0.shape
                                            frame[50:50 + alis0, 50:50 + anis0] = img_step0

                                            alis1, anis1, c = img_step1.shape
                                            frame[50:50 + alis1, 1030:1030 + anis1] = img_step1

                                            alis2, anis2, c = img_step2.shape
                                            frame[270:270 + alis2, 1030:1030 + anis2] = img_step2

                                            if x7 > x5 and x8 < x6:
                                                if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False:  # Parpadeo
                                                    conteo = conteo + 1
                                                    parpadeo = True

                                                elif longitud2 > 10 and longitud2 > 10 and parpadeo == True:  # Seguridad parpadeo
                                                    parpadeo = False

                                                alich, anich, c = img_check.shape
                                                frame[165:165 + alich, 1105:1105 + anich] = img_check

                                                cv2.putText(frame, f'Parpadeos: {int(conteo)}', (1070, 375),
                                                            cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                                            (255, 255, 255), 1)

                                                if conteo >= 3:
                                                    alich, anich, c = img_check.shape
                                                    frame[385:385 + alich, 1105:1105 + anich] = img_check

                                                    if longitud1 > 14 and longitud2 > 14:
                                                        step = 1
                                            else:
                                                conteo = 0

                                        if step == 1 and glass == False and capHat == False:
                                            cv2.rectangle(frame, (xi, yi, an, al), (0, 255, 0), 2)
                                            allich, anlich, c = img_liche.shape
                                            frame[50:50 + allich, 50:50 + anlich] = img_liche

                                            faces = fr.face_locations(rgb)
                                            facescod = fr.face_encodings(rgb, faces)

                                            for facecod, faceloc in zip(facescod, faces):

                                                Match = fr.compare_faces(FaceCode, facecod)

                                                simi = fr.face_distance(FaceCode, facecod)

                                                min = np.argmin(simi)

                                                if Match[min]:
                                                    UserName = clases[min].upper()

                                                    Profile()
                                        if glass == True:
                                            algla, angla, c = img_glass.shape
                                            frame[50:50 + algla, 50:50 + angla] = img_glass

                                        if capHat == True:
                                            alcap, ancap, c = img_cap.shape
                                            frame[50:50 + alcap, 50:50 + ancap] = img_cap


                            close = pantalla3.protocol("WM_DELETE_WINDOW", Close_Windows2)

            frame = imutils.resize(frame, width=1280)

            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)

            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, Sign_Biometric)

        else:
            cap.release()

def Sign():
    global LogUser, LogPass, OutFolderPath, cap, lblVideo, pantalla3, FaceCode, clases, images

    images = []
    clases = []
    lista = os.listdir(OutFolderPathFace)

    for lis in lista:
        imgdb = cv2.imread(f'{OutFolderPathFace}/{lis}')
        images.append(imgdb)
        clases.append(os.path.splitext(lis)[0])

    FaceCode = Code_Face(images)

    pantalla3 = Toplevel(pantalla)
    pantalla3.title("BIOMETRIC SIGN")
    pantalla3.geometry("1280x720")

    back2 = Label(pantalla3, image=imagenB, text="Back")
    back2.place(x=0, y=0, relwidth=1, relheight=1)

    lblVideo = Label(pantalla3)
    lblVideo.place(x=0, y=0)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)
    Sign_Biometric()


def Log():
    global RegName, RegUser, RegPass, InputNameReg, InputUserReg, InputPassReg, cap, lblVideo, pantalla2
    RegName, RegUser, RegPass = InputNameReg.get(), InputUserReg.get(), InputPassReg.get()

    if len(RegName) == 0 or len(RegUser) == 0 or len(RegPass) == 0:
        print(" FORMULARIO INCOMPLETO ")

    else:
        UserList = os.listdir(PathUserCheck)
        UserName = []
        for lis in UserList:
            User = lis
            User = User.split('.')
            UserName.append(User[0])

        if RegUser in UserName:
            print("USUARIO REGISTRADO ANTERIORMENTE")

        else:
            info.append(RegName)
            info.append(RegUser)
            info.append(RegPass)

            f = open(f"{OutFolderPathUser}/{RegUser}.txt", 'w')
            f.writelines(RegName + ',')
            f.writelines(RegUser + ',')
            f.writelines(RegPass + ',')
            f.close()

            InputNameReg.delete(0, END)
            InputUserReg.delete(0, END)
            InputPassReg.delete(0, END)

            pantalla2 = Toplevel(pantalla)
            pantalla2.title("BIOMETRIC REGISTER")
            pantalla2.geometry("1280x720")

            back = Label(pantalla2, image=imagenB, text="Back")
            back.place(x=0, y=0, relwidth=1, relheight=1)

            lblVideo = Label(pantalla2)
            lblVideo.place(x=0, y=0)

            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(3, 1280)
            cap.set(4, 720)
            Log_Biometric()

confidenceCap = 0.5
confidenceGlass = 0.5
confThresholdCap = 0.5
confThresholdGlass = 0.5

modelGlass = YOLO("Modelos/Gafas.pt")
modelCap = YOLO("Modelos/Gorras.pt")

#xddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd#
OutFolderPathUser = '/FaceRecognitionLivenessSystem/DataBase/Users'
PathUserCheck = '/FaceRecognitionLivenessSystem/DataBase/Users/'
OutFolderPathFace = '/FaceRecognitionLivenessSystem/DataBase/Faces'

info = []

parpadeo = False
conteo = 0
muestra = 0
step = 0

offsety = 30
offsetx = 20

confThreshold = 0.5
blurThreshold = 15

mpDraw = mp.solutions.drawing_utils
ConfigDraw = mpDraw.DrawingSpec(thickness=1, circle_radius=1) 

FacemeshObject = mp.solutions.face_mesh
FaceMesh = FacemeshObject.FaceMesh(max_num_faces=1)

FaceObject = mp.solutions.face_detection
detector = FaceObject.FaceDetection(min_detection_confidence= 0.5, model_selection=1)

#xddddddddddddddddddddddddddddddddddddddddddd#
img_cap = cv2.imread("/FaceRecognitionLivenessSystem/SetUp/cap.png")
img_glass = cv2.imread("/FaceRecognitionLivenessSystem/SetUp/glass.png")
img_check = cv2.imread("/FaceRecognitionLivenessSystem/SetUp/check.png")
img_step0 = cv2.imread("/FaceRecognitionLivenessSystem/SetUp/Step0.png")
img_step1 = cv2.imread("/FaceRecognitionLivenessSystem/SetUp/Step1.png")
img_step2 = cv2.imread("/FaceRecognitionLivenessSystem/SetUp/Step2.png")
img_liche = cv2.imread("/FaceRecognitionLivenessSystem/SetUp/LivenessCheck.png")

pantalla = Tk()
pantalla.title("FACE RECOGNITION SYSTEM")
pantalla.geometry("1280x720")

imagenF = PhotoImage(file="FaceRecognitionLivenessSystem/SetUp/Inicio.png")
background = Label(image = imagenF, text = "Inicio")
background.place(x = 0, y = 0, relwidth = 1, relheight = 1)


imagenB = PhotoImage(file="/FaceRecognitionLivenessSystem/SetUp/Back2.png")

InputNameReg = Entry(pantalla)
InputNameReg.place(x= 110, y = 320)

InputUserReg = Entry(pantalla)
InputUserReg.place(x= 110, y = 430)

InputPassReg = Entry(pantalla)
InputPassReg.place(x= 110, y = 540)


imagenBR = PhotoImage(file="/FaceRecognitionLivenessSystem/SetUp/BtSign.png")
BtReg = Button(pantalla, text="Registro", image=imagenBR, height="40", width="200", command=Log)
BtReg.place(x = 300, y = 580)


imagenBL = PhotoImage(file="/FaceRecognitionLivenessSystem/SetUp/BtLogin.png")
BtSign = Button(pantalla, text="Sign", image=imagenBL, height="40", width="200", command=Sign)
BtSign.place(x = 850, y = 580)

pantalla.mainloop()