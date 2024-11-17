"""
    ESTO ES UN MODULO HDP RECONOCELO
"""
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from scipy.spatial.distance import cosine, euclidean
import os
import shutil


############# -------------------------------- get_embedding_of_image -------------------------------- #############
def get_embedding_of_image(image_path):
    # Cargar el modelo preentrenado ResNet-50
    model = models.resnet50(pretrained=True)
    # Eliminar la última capa para obtener el embedding
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    
    # Definir las transformaciones de preprocesamiento de la imagen
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Cargar y preprocesar la imagen
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Añadir una dimensión extra para el batch

    # Verificar si hay GPU disponible y mover el modelo y la imagen
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_batch = input_batch.to(device)
    
    # Obtener el embedding
    with torch.no_grad():
        embedding = model(input_batch)
    
    # El resultado es un tensor de 4D (batch_size, channels, height, width)
    # Convertimos a 2D y extraemos como vector de características (embedding)
    embedding = embedding.squeeze().cpu().numpy()
    
    return embedding



############# -------------------------------- get_id_of_embedding -------------------------------- #############

def compare_two_embeddings(embedding1, embedding2):
    """Retorna la similitud entre dos embeddings como un valor entre 0 y 1."""
    euclidean_sim = 1 / (1 + euclidean(embedding1, embedding2))
    cosine_sim = 1 - cosine(embedding1, embedding2)
    return min(euclidean_sim, cosine_sim)

def average_embedding(file):
    """
    Calcula el embedding promedio dado la cantidad de embeddings previos, 
    
    Parámetros:
        - file: El txt en donde se tiene la cantidad y la suma de los embedding.
    
    Retorno:
        - list[float]: Embedding promedio actualizado.
    """
    
    lines = file.readlines()
    count = int(lines[0].strip())
    stored_sum_embedding = [float(x) for x in lines[1].split()]
    stored_average = [x / count for x in stored_sum_embedding]
    return stored_average

def get_existing_embedding_id(stored_path, embedding, similarity_threshold):
    """
    Busca un embedding similar en las carpetas existentes.
    
    Parámetros:
        - stored_path (str): Ruta base donde se buscan los embeddings.
        - embedding (list[float]): Embedding a comparar.
        
    Retorno:
        - int: ID de la carpeta que contiene un embedding similar o -1 si no se encuentra.
    """
    best_match_id = -1
    highest_similarity = similarity_threshold

    for folder_name in os.listdir(stored_path):
        folder_path = os.path.join(stored_path, folder_name)
        embedding_file = os.path.join(folder_path, "embedding.txt")
        
        if os.path.isfile(embedding_file):
            with open(embedding_file, "r") as file:
                stored_average = average_embedding(file)
                
                similarity = compare_two_embeddings(stored_average, embedding)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match_id = int(folder_name)
                    
    return best_match_id

def save_image_and_update_embedding(stored_path, image_path, image_id, embedding, cant_max_imagenes):
    """
    Guarda la imagen en la carpeta correspondiente y actualiza el archivo de embedding promedio.
    
    Parámetros:
        - stored_path (str): Ruta base donde se guardarán las imágenes.
        - image_path (str): Ruta de la imagen a guardar.
        - image_id (int): ID de la carpeta donde se guardará la imagen.
        - embedding (list[float]): Embedding de la imagen a guardar.
    """
    folder_path = os.path.join(stored_path, str(image_id))
    os.makedirs(folder_path, exist_ok=True)
    
    # Guarda la imagen con un número autoincremental + timestamp
    timestamp = time.time()
    image_count = len(os.listdir(folder_path))-1 #TODO: REVISAR QUE HAGA LO MISMO QUE ESTO: len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    if not image_count > cant_max_imagenes:
        new_image_path = os.path.join(folder_path, f"{image_count + 1} - {timestamp}.jpg")
        shutil.copy(image_path, new_image_path)
    
    # Actualiza el archivo de embedding promedio
    embedding_file = os.path.join(folder_path, "embedding.txt")
    
    if os.path.isfile(embedding_file):
        with open(embedding_file, "r") as file:
            lines = file.readlines()
            count = int(lines[0].strip())
            stored_sum_embedding = [float(x) for x in lines[1].split()]
    else:
        count = 0
        stored_sum_embedding = [0.0] * len(embedding)

    # Actualizar la suma
    updated_sum = [s + e for s, e in zip(stored_sum_embedding, embedding)]
    
    with open(embedding_file, "w") as file:
        file.write(f"{count + 1}\n")
        file.write(" ".join(map(str, updated_sum)))

def get_id_of_embedding(image_path, embedding, stored_path, similarity_threshold, cant_max_imagenes):
    """
    Retorna el ID de un embedding dado, y guarda la imagen en 'stored-images/id'.
    
    Parámetros:
        - image_path (str): Ruta de la imagen a guardar.
        - embedding (list[float]): Embedding a comparar.
        - stored_path (str): Ruta base donde se guardarán los datos.
        
    Retorno:
        - int: ID de la carpeta donde se guardó la imagen.
    """

    image_id = get_existing_embedding_id(stored_path, embedding, similarity_threshold)
    
    if image_id == -1:
        # Usamos la cantidad de archivos en stored_path para asignar un nuevo ID
        num_files = len(os.listdir(stored_path))
        image_id = num_files
    
    save_image_and_update_embedding(stored_path, image_path, image_id, embedding, cant_max_imagenes)
    return image_id

############# -------------------- FUNCION QUE SE LLAMA DESDE AFUERA ("api") -------------------- #############

def get_id_of_image(image_path, stored_path="src/facial_recognition/stored-images", clear=False, similarity_threshold=0.5, cant_max_imagenes = 15):
    '''
    Esta funcion recibe la ruta de una imagen y retorna el id del objeto reconocido
    '''
    
    # Crear carpeta "stored_path" si no existe, y si existe vaciarla
    if os.path.exists(stored_path):
        if clear:
            shutil.rmtree(stored_path)
            os.makedirs(stored_path)
    else:
        os.makedirs(stored_path)

    embedding = get_embedding_of_image(image_path)
    id = get_id_of_embedding(image_path, embedding, stored_path, similarity_threshold, cant_max_imagenes)
    return id
