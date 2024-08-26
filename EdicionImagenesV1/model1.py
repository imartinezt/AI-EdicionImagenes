import asyncio
import os

import numpy as np
import torch
from PIL import Image, ExifTags
from skimage.transform import resize
from skimage.util import img_as_ubyte
from transformers import DetrImageProcessor, DetrForObjectDetection


def correct_orientation(imagen):
    orientation_key = next((k for k, v in ExifTags.TAGS.items() if v == 'Orientation'), None)
    if not orientation_key:
        return imagen
    try:
        exif = imagen.getexif()
        if exif is not None:
            orientation = exif.get(orientation_key)
            rotations = {3: 180, 6: 270, 8: 90}
            if orientation in rotations:
                imagen = imagen.rotate(rotations[orientation], expand=True)
    except Exception as e:
        print(f"Error correcting orientation: {e}")
    return imagen


# mejor version
def process_single_image(image_path, output_path, margin=25, desired_width=940, desired_height=1215):
    try:
        image = Image.open(image_path)
        image = correct_orientation(image)
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

        # Convertir a formato numpy para usar con OpenCV
        image_np = np.array(image)

        # Detectar objetos en la imagen
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
        boxes = results["boxes"].detach().numpy()
        scores = results["scores"].detach().numpy()

        # Obtener la caja delimitadora más grande
        largest_box = boxes[np.argmax(scores)]
        x1, y1, x2, y2 = largest_box

        # Calcular la altura y el ancho del objeto detectado
        detected_width = x2 - x1
        detected_height = y2 - y1

        # Regla 1: Si el objeto cubre completamente el eje Y de la imagen o más del 90%
        image_height, image_width = image_np.shape[:2]
        if (y1 <= 0 and y2 >= image_height) or (detected_height >= 0.9 * image_height):
            print("La caja delimitadora cubre más del 90% del eje Y, aplicando recorte centrado y proporcional.")
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            aspect_ratio = desired_width / desired_height

            # Mantener la proporción, ajustando el recorte de forma centrada
            if detected_width / detected_height > aspect_ratio:
                new_crop_height = detected_height
                new_crop_width = new_crop_height * aspect_ratio
            else:
                new_crop_width = detected_width
                new_crop_height = new_crop_width / aspect_ratio

            # Calcular las coordenadas del recorte centrado
            new_x1 = max(0, center_x - new_crop_width / 2)
            new_x2 = min(image_width, center_x + new_crop_width / 2)
            new_y1 = max(0, center_y - new_crop_height / 2)
            new_y2 = min(image_height, center_y + new_crop_height / 2)

            # Recortar la imagen
            cropped_image = image_np[int(new_y1):int(new_y2), int(new_x1):int(new_x2)]

        # Regla 2: Agregar margen solo en la parte superior o inferior según la disponibilidad de espacio
        elif y1 - margin >= 0 or y2 + margin <= image.height:
            if y1 - margin >= 0 and y2 + margin > image.height:
                print("Aplicando margen solo en la parte superior.")
                new_y1 = max(0, y1 - margin)
                new_y2 = y2
            elif y2 + margin <= image_height and y1 - margin < 0:
                print("Aplicando margen solo en la parte inferior.")
                new_y1 = y1
                new_y2 = min(image_height, y2 + margin)
            else:
                new_y1 = max(0, y1 - margin)
                new_y2 = min(image_height, y2 + margin)

            crop_width = x2 - x1
            crop_height = new_y2 - new_y1
            aspect_ratio = desired_width / desired_height
            current_aspect_ratio = crop_width / crop_height

            if current_aspect_ratio > aspect_ratio:
                crop_height = crop_width / aspect_ratio
                new_y1 = max(0, (y1 + y2) / 2 - crop_height / 2)
                new_y2 = min(image_height, new_y1 + crop_height)
            else:
                crop_width = crop_height * aspect_ratio
                new_x1 = max(0, (x1 + x2) / 2 - crop_width / 2)
                new_x2 = min(image_width, new_x1 + crop_width)

            cropped_image = image_np[int(new_y1):int(new_y2), int(new_x1):int(new_x2)]

        # Regla 3: Si hay espacio tanto en la parte superior como en la inferior
        else:
            print("Aplicando lógica de cuerpo completo.")
            width_center = (x1 + x2) / 2
            new_y1 = max(0, y1 - margin)
            new_y2 = min(image_height, y2 + margin)

            crop_width = x2 - x1
            crop_height = new_y2 - new_y1
            aspect_ratio = desired_width / desired_height
            current_aspect_ratio = crop_width / crop_height

            if current_aspect_ratio > aspect_ratio:
                crop_height = crop_width / aspect_ratio
                new_y1 = max(0, (y1 + y2) / 2 - crop_height / 2)
                new_y2 = min(image_height, new_y1 + crop_height)
            else:
                crop_width = crop_height * aspect_ratio
                new_x1 = max(0, width_center - crop_width / 2)
                new_x2 = min(image_width, width_center + crop_width / 2)

            cropped_image = image_np[int(new_y1):int(new_y2), int(new_x1):int(new_x2)]

        # Convertir la imagen recortada a formato numpy para skimage
        cropped_image_skimage = np.array(cropped_image)

        # Redimensionar la imagen con skimage sin distorsión
        final_image = resize(cropped_image_skimage, (desired_height, desired_width), anti_aliasing=True)

        # Convertir de vuelta a un formato de imagen PIL
        final_image_pil = Image.fromarray(img_as_ubyte(final_image))

        final_image_pil.save(output_path)
        print(f"Imagen procesada y guardada en: {output_path}")

    except Exception as e:
        print(f"Error procesando la imagen {image_path}: {e}")


async def process_image(image_file, input_folder, output_folder, margin=25, desired_width=940, desired_height=1215):
    input_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, image_file)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, process_single_image, input_path, output_path, margin, desired_width,
                               desired_height)


async def process_images_in_folder(input_folder, salida_folder, margin=25, desired_width=940, desired_height=1215):
    if not os.path.exists(salida_folder):
        os.makedirs(salida_folder)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in image_extensions]

    tasks = []
    for image_file in image_files:
        tasks.append(process_image(image_file, input_folder, salida_folder, margin, desired_width, desired_height))

    await asyncio.gather(*tasks)
