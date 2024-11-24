import torch
import tensorflow as tf

# Verificar si CUDA está disponible
'''print("CUDA disponible:", torch.cuda.is_available())
print("Número de GPUs:", torch.cuda.device_count())
print("Nombre de la GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No hay GPU disponible")
'''
print("TensorFlow versión:", tf.__version__)

print("GPUs disponibles:", len(tf.config.list_physical_devices('GPU')))
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

print("CUDA soportada:", tf.test.is_built_with_cuda())
print("GPU disponible:", tf.config.list_physical_devices('GPU'))