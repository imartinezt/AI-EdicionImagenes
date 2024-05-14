import asyncio
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ExifTags
from google.cloud import vision_v1
from google.oauth2 import service_account

"""
@Autor: Iván Martínez Trejo.
Contacto: imartinezt@liverpool.com.mx
 -- Edicion de Imagenes Modelo 1 | Foro Fotografico | v1.2
        - Recorte centrado
             - Lienzo a 940 px ancho X 1215 px de alto.
             - Resolución 72 dpis.
             - Formatos de imagen y peso: JPG y MB
"""


def get_vision_client():
    sa_path = "keys.json"
    with open(sa_path) as source:
        info = json.load(source)
    creds = service_account.Credentials.from_service_account_info(info)
    return vision_v1.ImageAnnotatorClient(credentials=creds)


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


def detect_faces(image_content, client):
    image = vision_v1.Image(content=image_content)
    response = client.face_detection(image=image)
    faces = response.face_annotations
    return faces


def compress_image(image_path, quality=70):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="JPEG", quality=quality)
        return output_buffer.getvalue()


async def process_image(image_path, client):
    loop = asyncio.get_event_loop()
    compressed_image_content = await loop.run_in_executor(None, compress_image, image_path)

    faces = await loop.run_in_executor(None, detect_faces, compressed_image_content, client)

    with open(image_path, 'rb') as image_file:
        pil_image = Image.open(image_file)

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

        center_x = original_width / 2
        center_y = original_height / 2

        if faces:
            face = faces[0]
            min_x = face.bounding_poly.vertices[0].x
            min_y = face.bounding_poly.vertices[0].y
            max_x = face.bounding_poly.vertices[2].x
            max_y = face.bounding_poly.vertices[2].y

            face_center_x = (min_x + max_x) / 2
            face_center_y = (min_y + max_y) / 2

            new_left = max(int(face_center_x - new_width / 2), 0)
            new_top = max(int(face_center_y - new_height / 2), 0)
            new_right = min(new_left + new_width, original_width)
            new_bottom = min(new_top + new_height, original_height)

        else:
            new_left = max(int(center_x - new_width / 2), 0)
            new_top = max(int(center_y - new_height / 2), 0)
            new_right = min(new_left + new_width, original_width)
            new_bottom = min(new_top + new_height, original_height)

        if new_left < new_right and new_top < new_bottom:
            cropped_image = pil_image.crop((new_left, new_top, new_right, new_bottom))
            adjusted_image = cropped_image.resize((940, 1215))

            return pil_image, adjusted_image

    return None, None


async def process_images_async(image_paths, output_folder='Piloto1', output_dpi=72, batch_size=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    client = get_vision_client()

    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]

    image_counter = 0

    with ThreadPoolExecutor(max_workers=5):
        for batch in batches:
            tasks = [process_image(image_path, client) for image_path in batch]
            results = await asyncio.gather(*tasks)

            for i, result in enumerate(results):
                original_image, adjusted_image = result
                if original_image and adjusted_image:
                    image_counter += 1
                    adjusted_path = os.path.join(output_folder, f'AIModelOne-{image_counter}.jpg')
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
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(root, file))
        else:
            print(f"La carpeta {folder_path} no existe.")
    return image_paths


async def main():
    imgs_folders = [
        "[/PATH/]"
    ]

    image_paths = await list_image_paths(imgs_folders)

    await asyncio.gather(process_images_async(image_paths))


if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f"Se ha tardado {end - start} segundos")
