import numpy as np
import cv2
from ultralytics import YOLO
from sort import Sort
import random

def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    model = YOLO("yolo11n.pt")
    tracker = Sort()
    
    # Diccionario para mantener colores consistentes por clase
    class_colors = {}

    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break

        results = model(frame, stream=True)

        for res in results:
            filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.5)[0]
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
            
            # Obtener clasificaciones y confianzas
            classes = res.boxes.cls.cpu().numpy()[filtered_indices].astype(int)
            confidences = res.boxes.conf.cpu().numpy()[filtered_indices]
            
            # Actualizar tracking
            tracks = tracker.update(boxes)
            tracks = tracks.astype(int)
            
            for i, (xmin, ymin, xmax, ymax, track_id) in enumerate(tracks):
                # Obtener el nombre de la clase
                class_id = classes[i]
                class_name = model.names[class_id]
                confidence = confidences[i]
                
                # Asignar color consistente por clase
                if class_name not in class_colors:
                    class_colors[class_name] = get_random_color()
                color = class_colors[class_name]
                
                # Dibujar rectángulo
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(xmax, ymax), 
                            color=color, thickness=2)
                
                # Crear texto con toda la información
                text = f"{class_name} ({confidence:.2f}) ID:{track_id}"
                
                # Crear fondo para el texto
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (xmin, ymin-30), (xmin + text_width, ymin), color, -1)
                
                # Agregar texto
                cv2.putText(img=frame, text=text, org=(xmin, ymin-10),
                           fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, 
                           color=(255, 255, 255), thickness=2)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()