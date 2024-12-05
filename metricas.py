#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 01:34:27 2024

@author: cota
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#%%
# kk = imagen_truth > 0.5
def coeficiente_dice(matriz1, matriz2):
    # Asegúrate de que las matrices tengan el mismo tamaño
    assert matriz1.shape == matriz2.shape, "Las matrices deben tener el mismo tamaño"

    # Convierte las matrices a booleanas (1 si el pixel está presente, 0 si no)
    a = matriz1.astype(bool)
    b = matriz2.astype(bool)

    # Calcula la intersección y la suma de los elementos
    interseccion = np.sum(a & b)
    suma = np.sum(a) + np.sum(b)

    # Calcula el coeficiente de Dice
    if suma == 0:
        return 1.0  # Si ambas matrices están vacías, se puede considerar Dice como 1
    else:
        return 2. * interseccion / suma

def TP_FP_TN_FN(mask, truth):
    mask_binary = mask.astype(bool)
    ground_truth_binary = truth.astype(bool)

# Calcular verdaderos positivos (TP) y falsos positivos (FP)
    TP = np.sum((mask_binary == 1) & (ground_truth_binary == 1))
    FP = np.sum((mask_binary == 1) & (ground_truth_binary == 0))
    TN = np.sum((mask_binary == 0) & (ground_truth_binary == 0))
    FN = np.sum((mask_binary == 0) & (ground_truth_binary == 1))
    
    return TP, FP, TN, FN

#%%
ruta_pred = '/home/cota/EMC-Click/experiments/evaluation_logs/others/hr32/predictions_vis/liver_ventaneo_mediana/matrices'
ruta_truth = '/home/cota/datasets/liver_ventaneo_mediana/masks'
organo = "liver_ventaneo_mediana"
# def calculate_metrics(ruta_pred, ruta_truth, organo):   
archivos_pred = sorted((os.listdir(ruta_pred) ))
archivos_truth = sorted(os.listdir(ruta_truth))
names = []
noc = []
for i in range(len(archivos_truth)):
    # print(i)
    for j in archivos_pred:
         # print(j)
          #print(archivos_truth[-7:-4])
         # print(j[:4], 555)
         # print(archivos_truth[i][-8:-4])
        if j[:4]==archivos_truth[i][-8:-4]:
              # print(j)
              # print(j[:4] , archivos_truth[i][-8:-4])
              # print(archivos_truth[i][-7:-4])
              file = j
    names.append(file)
    noc.append(int(file[-7:-4]))
dice = []
beetwen = []
P = []
# S = []
R = []
 # print("Dice:", coeficiente_dice(mask, kk[:,:,2]))
for i in range(len(names)):
     
     
     
    mask_pred = np.loadtxt(f"{ruta_pred}/{names[i]}") > 0.5
    mask_truth = cv2.imread(f"{ruta_truth}/{archivos_truth[i]}")[:,:,0]
    # print(f"Metricas entre {names[i]} y {archivos_truth[i]}")
    dc = coeficiente_dice(mask_pred, mask_truth)
    dice.append(dc)
     # print(dc)
    beetwen.append(f"Metricas entre {names[i]} y {archivos_truth[i]}")
     
    TP, FP, TN, FN = TP_FP_TN_FN(mask_pred, mask_truth)
    mc = np.array([[TP, FP],
                   [FN, TN]])
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
     # specifity = TN /(FP + TN) if(FP + TN) > 0 else 0
 # Calcular recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    P.append(precision)
     # S.append(specifity)
    R.append(recall)
     
     
    # plt.imshow(mask_truth, cmap='gray')
    # plt.axis('off')  # No mostrar ejes
    # plt.title(f"Mask Truth {i}")
    # plt.show()
 
    # plt.imshow(mask_pred, cmap='gray')
    # plt.axis('off')  # No mostrar ejes
    # plt.title(f"Mask Pred {i}")
    # plt.show() 
    # dice_coeficients, precision, recall, NoC, beetwen = calculate_metrics(ruta_pred, ruta_truth)
data = {'Dice coeficient': dice,
         'Precision': P,
         'Recall': R,
                 }
 # import pandas as pd
data = pd.DataFrame(data)
 # pd.Da
  # import seaborn as sns
sns.lineplot(data, marker= 'o', linestyle='--')
organ = organo
print(f"Medias {organ}: \n -Dice={np.mean(dice)} \n -Precision={np.mean(P)} \n -Recall={np.mean(R)} \n -NoC={np.mean(noc)}" )
print(f"Desviaciones estandar {organ}: \n -Dice={np.std(dice)} \n -Precision={np.std(P)} \n -Reecall={np.std(R)} \n -NoC={np.std(noc)}")
print(f"Matriz de confusión:\n TP FP \n FN TN \n {mc}")
# media_dice = np.mean(dice)
 #print("Media dice:", media_dice)ArithmeticErrorimport matplotlib.pyplot as plt
plt.plot(data, marker= 'o', linestyle= '--')
os.makedirs('plots', exist_ok=True)
plt.title(f"{organ} metrics")
plt.xlabel('Id')
plt.ylabel('Value')
plt.legend()
#plt.show()
plt.savefig(f"plots/{organ}_metrics.png")  # Guardamos la primera imagen
plt.close()
 
plt.plot(noc, marker='o', linestyle = '-')
plt.title(f"NoC ({organ})")
plt.xlabel('Id')
plt.ylabel('Clicks')
plt.legend()
#plt.show()
plt.savefig(f"plots/{organ}_NoC.png")  # Guardamos la primera imagen
plt.close()
 # text = f"TP FP\nFN Tn= {mc}"
 # np.savetxt("matriz_simple.txt", text, fmt="%d", delimiter="\t")
  
#print("La matriz se ha guardado en 'matriz_simple.txt'")
    # print(f"Medias {organ}: \n -Dice={np.mean(dice)} \n -Precision={np.mean(P)} \n -Recall={np.mean(R)} \n -NoC={np.mean(noc)}" )
    # print(f"Desviaciones estandar {organ}: \n -Dice={np.std(dice)} \n -Precision={np.std(P)} \n -Reecall={np.std(R)} \n -NoC={np.std(noc)}")
    # print(f"Matriz de confusión TP FP\nMatriz de confusion FN TN \n {mc}")
    # return dice, P, R, mc, noc
# ruta_pred = '/home/cota/EMC-Click/experiments/evaluation_logs/others/hr32/predictions_vis/lung/matrices'
# ruta_truth = '/home/cota/EMC-Click/datasets/Lung/masks'
# dice_coeficients, precision, recall, _, beetwen = calculate_metrics(ruta_pred, ruta_truth)

#%%

# ruta_truth = '/home/cota/EMC-Click/datasets/spleen/masks'
# ruta_pred = '/home/cota/EMC-Click/experiments/evaluation_logs/others/hr32/predictions_vis/spleen/matrices'
# organ = 'spleen'
# dice_coeficients, precision, recall, confusion_matrix, NoC = calculate_metrics(ruta_pred, ruta_truth, organ)
#%%
import sys


def main():
    # Verifica que se hayan pasado exactamente 2 argumentos
    if len(sys.argv) != 4:
        print("Uso incorrecto.")
        sys.exit(1)  # Salir del script con un código de error

    # Los argumentos se obtienen desde sys.argv
    ruta_truth = sys.argv[1]
    ruta_pred = sys.argv[2]  # Convertimos la edad a un entero
    organ = sys.argv[3]
    # Llamar a la función con los argumentos
    calculate_metrics(ruta_pred, ruta_truth, organ)

if __name__ == "__main__":
    main()

# ruta_truth = '/home/cota/EMC-Click/datasets/Berkeley/masks'
# ruta_pred = '/home/cota/EMC-Click/experiments/evaluation_logs/others/hr32/predictions_vis/Berkeley/matrices'
# organ = 'liver'

# plt.plot(NoC, marker='o', linestyle = '-')
# plt.title(f"NoC ({organ})")
# plt.xlabel('Id')
# plt.ylabel('Clicks')
# plt.legend()
# plt.show()
# data = {'Dice coeficient': dice_coeficients,
#         'Precision': precision,
#         'Recall': recall,
#                 }
# import pandas as pd
# data = pd.DataFrame(data)
# # pd.Da
# import seaborn as sns
# sns.lineplot(data, marker= 'o', linestyle='--')

# # import matplotlib.pyplot as plt
# plt.plot(data, marker= 'o', linestyle= '-')
# plt.title("Liver metrics")
# plt.xlabel('Id')
# plt.ylabel('Percentage')
# plt.legend()
# plt.show()

# plt.title("Liver metrics")
# plt.xlabel('Id')
# plt.ylabel('Percentage')
# plt.legend()
# plt.show()