import tkinter as tk

def cambiar_texto():
    etiqueta.config(text="¡Texto cambiado!")

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Ejemplo de Tkinter")
ventana.geometry("300x200")

# Crear una etiqueta
etiqueta = tk.Label(ventana, text="Texto inicial", font=("Arial", 14))
etiqueta.pack(pady=20)

# Crear un botón
boton = tk.Button(ventana, text="Cambiar texto", command=cambiar_texto)
boton.pack(pady=10)

# Ejecutar el loop principal de la ventana
ventana.mainloop()
