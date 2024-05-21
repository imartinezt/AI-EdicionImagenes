import asyncio
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import ProcessPoolExecutor

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


async def detect_faces(image_content, client, thread_executor):
    loop = asyncio.get_event_loop()
    image = vision_v1.Image(content=image_content)
    response = await loop.run_in_executor(thread_executor, client.face_detection, image)
    faces = response.face_annotations
    return faces


def compress_image_sync(image_path, quality):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        output_buffer = io.BytesIO()
        img.save(output_buffer, "JPEG", quality=quality)
        return output_buffer.getvalue()


async def compress_image(image_path, quality, process_executor):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(process_executor, compress_image_sync, image_path, quality)


async def process_image(image_path, client, thread_executor, process_executor):
    compressed_image_content = await compress_image(image_path, 70, process_executor)
    faces = await detect_faces(compressed_image_content, client, thread_executor)

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
                    pil_image = pil_image.rotate(rotation_angle, expand=True)

        except (AttributeError, KeyError, IndexError):
            pass

        original_width, original_height = pil_image.size
        new_width, new_height = await calculate_cropped_size(original_width, original_height, 940, 1215)

        center_x = original_width / 2
        center_y = original_height / 2

        if faces:
            face = faces[0]
            min_x = min(vertex.x for vertex in face.bounding_poly.vertices)
            min_y = min(vertex.y for vertex in face.bounding_poly.vertices)
            max_x = max(vertex.x for vertex in face.bounding_poly.vertices)
            max_y = max(vertex.y for vertex in face.bounding_poly.vertices)

            face_center_x = (min_x + max_x) / 2
            face_center_y = (min_y + max_y) / 2

            new_left = max(int(face_center_x - new_width / 2), 0)
            new_top = max(int(face_center_y - new_height / 2), 0)
            new_right = min(new_left + new_width, original_width)
            new_bottom = min(new_top + new_height, original_height)

            if new_right - new_left < new_width:
                new_left = max(original_width - new_width, 0)
                new_right = original_width

            if new_bottom - new_top < new_height:
                new_top = max(original_height - new_height, 0)
                new_bottom = original_height

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


def save_image_sync(image, path, dpi):
    image.save(path, dpi=(dpi, dpi))


async def adjust_image_resolution(image_path, dpi, thread_executor):
    loop = asyncio.get_event_loop()
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        await loop.run_in_executor(thread_executor, save_image_sync, img, image_path, dpi)


async def process_images_async(image_paths, output_folder='Modelo-V2', output_dpi=72, batch_size=5, thread_executor=None,
                               process_executor=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    client = get_vision_client()

    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]

    image_counter = 0  # Add a counter to keep track of the global image index

    for batch in batches:
        tasks = [process_image(image_path, client, thread_executor, process_executor) for image_path in batch]
        results = await asyncio.gather(*tasks)

        for i, result in enumerate(results):
            original_image, adjusted_image = result
            if original_image and adjusted_image:
                image_counter += 1  # Increment the counter for each processed image
                adjusted_path = os.path.join(output_folder, f'ShootingV3-{image_counter}.jpg')
                await asyncio.get_event_loop().run_in_executor(thread_executor, adjusted_image.save, adjusted_path)
                await adjust_image_resolution(adjusted_path, output_dpi, thread_executor)
            else:
                print("No se pudo procesar la imagen.")


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
        "/Path/"
    ]

    image_paths = await list_image_paths(imgs_folders)

    thread_executor = ThreadPoolExecutor(max_workers=10)
    process_executor = ProcessPoolExecutor(max_workers=4)

    await process_images_async(image_paths, thread_executor=thread_executor, process_executor=process_executor)

    thread_executor.shutdown(wait=True)
    process_executor.shutdown(wait=True)


if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f"Se ha tardado {end - start} segundos")
