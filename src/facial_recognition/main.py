import tkinter as tk
import threading
from src.facial_recognition import face_recognition
import sys
from io import StringIO


def run(etiqueta2):
    thread = threading.Thread(target=face_recognition.run, args=(etiqueta2,))
    thread.start()

def ejecutar(etiqueta2):
    etiqueta.config(text="Para finalizar programa presionar la letra q")
    run(etiqueta2)

def cambiar_nombre():
    global nombre_cambiado
    if nombre_cambiado:
        boton2.config(text="NOBRE CAMBIADO")
    else:
        boton2.config(text="Cambiar nombre")
    nombre_cambiado = not nombre_cambiado

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Ejemplo de Tkinter")
ventana.geometry("700x700")

# Crear una etiqueta
etiqueta = tk.Label(ventana, text="Reconocimiento Facial", font=("Arial", 14))
etiqueta.pack(pady=20)

etiqueta2 = tk.Label(ventana, text="ACA VA EL TEXTO DE LA TERMINAL", font=("Arial", 14))
etiqueta2.pack(pady=20)

# Crear un botón
boton = tk.Button(ventana, text="Correr programa", command=lambda: ejecutar(etiqueta2))
boton.pack(pady=10)

# Crear un botón
nombre_cambiado=False
boton2 = tk.Button(ventana, text="Cambiar nombre", command=cambiar_nombre)
boton2.pack(pady=10)

# Ejecutar el loop principal de la ventana
ventana.mainloop()

