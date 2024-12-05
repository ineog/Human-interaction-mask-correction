#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:56:12 2024

@author: cota
"""

import cv2
import matplotlib.pyplot as plt
import os
import nibabel as nib
import numpy as np
#%% funciones
def mask_extractor(data):
    lk = len(data[0,0,:])
    lj = len(data[:,0,0])
    li = lj
    cumul = []
    for k in range(lk):
        counter = []
        cant = 0
        for j in range(lj):
            for i in range(li):
                if data[i,j,k] == 1:
                    counter.append(1)
                else:
                    counter.append(0)
        cant = sum(counter)
        cumul.append(cant)
    index = cumul.index(max(cumul))
    return data[:,:,index], index
def ct_extractor(data,index):
    return data[:,:,index]
#%% Extracción de datos
# ruta_target = "/home/cota/EMC-Click/datasets/Task06_Lung/labelsTr/"
# ruta = "/home/cota/EMC-Click/datasets/Task06_Lung/imagesTr/"

def save_data(ruta_target, ruta, organo):
    # archivos_target = sorted(os.listdir(ruta_target))
    archivos = sorted(os.listdir(ruta))
    # c= -1
    for i in range(len(archivos)):
        # archivo_target = archivos_target[i]
        archivo  = archivos[i]
        print(archivo, f" {i+1}/{len(archivos)}")
        # if archivo[-4:] == ".nii":
            # c += 1
            # print(archivo_target)
            # print(archivo)
            # print(c)
                # print(archivo[-4:])
                # print(f"{ruta_target}/{archivo_target}")
        imagedir = f'/home/cota/datasets/{organo}/images/'
        # print(imagedir)
        maskdir = f'/home/cota/datasets/{organo}/masks/'
        os.makedirs(imagedir, exist_ok=True)
        os.makedirs(maskdir, exist_ok=True)
        imagen_target = nib.load(f"{ruta_target}/{archivo}")
        imagen = nib.load(f"{ruta}/{archivo}")
        # print(f"Procesando {archivo}")
        datos_target = imagen_target.get_fdata()
        datos = imagen.get_fdata()
        y_target,index = mask_extractor(datos_target)
        y = ct_extractor(datos,index)
        y = (y-np.min(y))/(np.max(y)-np.min(y))*255
        y_ = (y_target-np.min(y_target))/(np.max(y_target)-np.min(y_target))*255
        # nombre = f"{archivo[:-7]}_{c:05}"
        # imagedir = f'/home/cota/EMC-Click/datasets/{organo}_ventaneo/images/{nombre}.jpg'
        # print(imagedir)
        # maskdir = f'/home/cota/EMC-Click/datasets/{organo}_ventaneo/masks/{nombre}.jpg' 
        # print(nombre)
        dirim = f"{imagedir}/{organo}_{i:05}.jpg"
        dirmask = f"{maskdir}/{organo}_{i:05}.jpg"
        cv2.imwrite(dirim,(y[:]).astype(np.uint8))
        cv2.imwrite(dirmask,y_.astype(np.uint8))
            # print(f"Imagenes en directorios '/home/cota/EMC-Click/datasets/{organo}'")
            # return(y.shape)
            # break
# save_data(ruta_target,ruta,'lung')
#%% Guardar datos spleen
# ruta_target = "/home/cota/EMC-Click/datasets/Task02_Heart/labelsTr/"
# ruta = "/home/cota/EMC-Click/ventaneo/heart/"
import sys


def main():
    # Verifica que se hayan pasado exactamente 2 argumentos
    if len(sys.argv) != 4:
        print("Uso incorrecto.")
        sys.exit(1)  # Salir del script con un código de error

    # Los argumentos se obtienen desde sys.argv
    ruta_mask = sys.argv[1]
    ruta = sys.argv[2]  
    organ = sys.argv[3]
    # Llamar a la función con los argumentos
    save_data(ruta_mask,ruta, organ)

if __name__ == "__main__":
    main()
# # datos liver
# ruta_target = "/home/cota/EMC-Click/datasets/Task03_Liver/labelsTr/"
# ruta = "/home/cota/EMC-Click/datasets/Task03_Liver/imagesTr/"
# save_data(ruta_target,ruta,'liver')
# datos prostate
# ruta = ruta_target = "/home/cota/EMC-Click/datasets/Task05_Prostate/labelsTr/"
# ruta = "/home/cota/EMC-Click/datasets/Task05_Prostate/imagesTr/"
# archivos_target = sorted(os.listdir(ruta_target))
# archivos = sorted(os.listdir(ruta))
# a = save_data(ruta_target,ruta,'prostate')
#%% Escalar y guardar imagenes
# iiinnn = type(index)
# plt.imshow(y, cmap='gray')
# plt.axis('off')  # Ocultar ejes
# plt.show()#  
