#import subprocess 

#################################Preparando el ambiente de trabajo#######################################

# Instalar numpy 
#subprocess.run(["pip3", "install", "numpy"])

# Instalar opencv-python 
#subprocess.run(["pip3", "install", "opencv-python"])

# Instalar tflite_runtime 
#subprocess.run(["pip3", "install", "tflite_runtime"])

#########################################################################################################
import subprocess
import numpy as np
import cv2
import datetime
import time
import tflite_runtime.interpreter as tflite
import git
import os

def commit_and_push(repo_path, file_path):
    try:
        repo = git.Repo(repo_path)
        repo.git.add(file_path)

        # Realizar el commit incluso si no hay cambios
        repo.git.commit('-m', 'Agregar emotions_detected.csv')
        repo.git.push()
        print("Archivo emotions_detected.csv agregado al repositorio.")

        # Eliminar el archivo localmente después del commit y push
        local_file_path = os.path.join(repo_path, file_path)
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
            print(f"Archivo {file_path} eliminado localmente.")
        else:
            print(f"El archivo {file_path} no existe localmente.")

    except Exception as e:
        print("Error al realizar el commit y push:", str(e))

Interpreter = tflite.Interpreter(model_path="model.tflite")
Interpreter.allocate_tensors()

input_details = Interpreter.get_input_details()
output_details = Interpreter.get_output_details()

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Enojado", 1: "Disgustado", 2: "Temeroso", 3: "Feliz", 4: "Neutral", 5: "Triste", 6: "Sorprendido"}

cap = cv2.VideoCapture(0)
Emotions_File = open("emotions_detected.csv", "a")

frame = None
emotion_start_time = time.time()
show_video = True  # Mostrar el video

while True:
    # Leer el contenido del archivo "data.txt"
    with open("data.txt", "r") as data_file:
        data_content = data_file.read().strip()

    # Tomar decisiones basadas en el contenido del archivo
    if data_content == "iniciar":
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('cascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Resto del código para el procesamiento de emociones
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            cropped_img = np.array(cropped_img, dtype='f')
            Interpreter.set_tensor(input_details[0]['index'], cropped_img)
            Interpreter.invoke()
            output_data = Interpreter.get_tensor(output_details[0]['index'])
            maxindex = int(np.argmax(output_data))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0), 2, cv2.LINE_AA)

            if time.time() - emotion_start_time >= 1.0:
                emocion = emotion_dict[maxindex]
                tc = datetime.datetime.now()
                Emotions_File.write(str((emocion)) + ";" + str(tc) + "\n")
                emotion_start_time = time.time()

        if frame is not None:
            cv2.imshow('Video', cv2.resize(frame, (800, 480), interpolation=cv2.INTER_CUBIC))

    elif data_content == "detener":
        # Detener el código
        # Realizar commit y push al repositorio
        repo_path = "/home/joham/Desktop/proyecto_2/datos_proyecto_2"  # Coloca tu ruta de repositorio
        file_path = "emotions_detected.csv"
        commit_and_push(repo_path, file_path)
        break

    key = cv2.waitKey(1)
    if key == 27:  # Si se presiona la tecla Esc (27 en ASCII)
        break

Emotions_File.close()
cap.release()
cv2.destroyAllWindows()

