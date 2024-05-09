import asyncio
import io
import os
import time

from PIL import Image, ExifTags
from google.cloud import vision_v1

"""
@Autor: Iván Martínez Trejo.
Contacto: imartinezt@liverpool.com.mx
 -- Edicion de Imagenes Modelo 1 | Foro Fotografico | v1.2
        - Recorte centrado
             - Lienzo a 940 px ancho X 1215 px de alto.
             - Resolución 72 dpis.
             - Formatos de imagen y peso: JPG yMB
"""


async def calculate_cropped_size(original_width, original_height, target_width, target_height):
    target_ratio = target_width / target_height
    original_ratio = original_width / original_height

    if original_ratio > target_ratio:
        new_width = int(original_height * target_ratio)
        new_height = original_height
    else:
        new_width = original_width
        new_height = int(original_width / target_ratio)

    return new_width, new_height


async def detect_faces(image_content, client):
    response = client.face_detection(image={'content': image_content})
    faces = response.face_annotations

    return await asyncio.gather(*faces)  # update


async def compress_image(image_path, quality=70):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="JPEG", quality=quality)
        return output_buffer.getvalue()


async def process_image(image_path, client):
    compressed_image_content = await compress_image(image_path)

    response = client.face_detection(image={'content': compressed_image_content})
    faces = response.face_annotations

    with open(image_path, 'rb') as image_file:
        pil_image = Image.open(image_file)

        # Corregir orientación
        try:
            orientation_tag = [tag for tag, description in ExifTags.TAGS.items() if description == 'Orientation'][0]
            exif = pil_image.getexif()

            if exif is not None and orientation_tag in exif:
                orientation_value = exif[orientation_tag]
                orientation_dict = {3: 180, 6: 270, 8: 90}
                rotation_angle = orientation_dict.get(orientation_value, 0)

                if rotation_angle:
                    pil_image = pil_image.rotate(rotation_angle)

        except (AttributeError, KeyError, IndexError):
            pass

        original_width, original_height = pil_image.size
        new_width, new_height = await calculate_cropped_size(original_width, original_height, 940, 1215)

        # centro de la imagen
        center_x = original_width / 2
        center_y = original_height / 2

        if faces:
            # Sí se detectan rostros, utilizar las coordenadas del primer rostro detectado
            face = faces[0]  # Tomar solo el primer rostro detectado

            # Calcular las coordenadas del rectángulo que rodea el rostro
            min_x = face.bounding_poly.vertices[0].x
            min_y = face.bounding_poly.vertices[0].y
            max_x = face.bounding_poly.vertices[2].x
            max_y = face.bounding_poly.vertices[2].y

            # centro del rectángulo
            face_center_x = (min_x + max_x) / 2
            face_center_y = (min_y + max_y) / 2

            # recorte centrado en el rostro
            new_left = max(int(face_center_x - new_width / 2), 0)
            new_top = max(int(face_center_y - new_height / 2), 0)
            new_right = min(new_left + new_width, original_width)
            new_bottom = min(new_top + new_height, original_height)

        else:
            # Si no se detectan rostros, centrar el recorte en el centro de la imagen
            new_left = max(int(center_x - new_width / 2), 0)
            new_top = max(int(center_y - new_height / 2), 0)
            new_right = min(new_left + new_width, original_width)
            new_bottom = min(new_top + new_height, original_height)

        # Verificar si las coordenadas son válidas antes de recortar
        if new_left < new_right and new_top < new_bottom:
            cropped_image = pil_image.crop((new_left, new_top, new_right, new_bottom))
            adjusted_image = cropped_image.resize((940, 1215))

            return pil_image, adjusted_image

    return None, None


async def process_images_async(image_paths, output_folder='pilotoAI', output_dpi=72, batch_size=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    client = vision_v1.ImageAnnotatorClient()

    # lotes
    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]

    for batch in batches:
        tasks = []
        for image_path in batch:
            print("Procesando:", image_path)
            task = asyncio.create_task(process_image(image_path, client))
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        for i, result in enumerate(results):
            original_image, adjusted_image = result
            if original_image and adjusted_image:
                original_width, _ = original_image.size
                new_width, _ = adjusted_image.size
                adjusted_path = os.path.join(output_folder, f'imagenAI{i + 1}.jpg')
                adjusted_image.save(adjusted_path)
                await adjust_image_resolution(adjusted_path, output_dpi)
            else:
                print("No se pudo procesar la imagen.")


async def adjust_image_resolution(image_path, dpi):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img.save(image_path, dpi=(dpi, dpi))


async def list_image_paths(folder_paths):
    image_paths = []
    for folder_path in folder_paths:
        if os.path.exists(folder_path):
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(root, file))
        else:
            print(f"La carpeta {folder_path} no existe.")
    return image_paths


async def main():
    imgs_folders = [
        "[DIRECTORIO]"
    ]

    image_paths = await list_image_paths(imgs_folders)

    await asyncio.gather(process_images_async(image_paths))


if __name__ == "__main__":
    start = time.time()

    asyncio.run(main())

    end = time.time()
    print(f"Se ha tardado {end - start} segundos")