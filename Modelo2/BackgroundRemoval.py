import asyncio
import json
import os
import time
import io
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

import aiofiles
from PIL import Image, ExifTags
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
from google.oauth2 import service_account
from rembg import remove
from collections import OrderedDict

"""
@Autor: Iván Martínez Trejo.
Contacto: imartinezt@liverpool.com.mx
 -- Edicion de Imagenes Modelo 2 | Foro Fotografico | v1.2
        - Recorte centrado.
              - Lienzo a 940px ancho X 1215 px de alto.
              - Resolución 72 dpis.
              - Formatos de imagen y peso: JPG y MB
              - Remueve el background con rembg
              + Rellenar con fondo blanco (#FFFFFF ) el fondo (Este es el único cambio)
"""


def get_vision_client():
    sa_path = "keys.json"
    with open(sa_path) as source:
        info = json.load(source)
    creds = service_account.Credentials.from_service_account_info(info)
    return vision_v1.ImageAnnotatorClient(credentials=creds)


async def detect_objects(image_content, client):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, client.object_localization, types.Image(content=image_content))


async def remove_background_async(input_buffer):
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as executor:
        return await loop.run_in_executor(executor, partial(remove, input_buffer, alpha_matting=True,
                                                            alpha_matting_foreground_threshold=240,
                                                            alpha_matting_background_threshold=10,
                                                            alpha_matting_erode_structure_size=15,
                                                            alpha_matting_erode_size=10))


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


async def correct_image_orientation_async(img):
    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor() as executor:
        return await loop.run_in_executor(executor, correct_orientation, img)


async def process_image(image_path, client, output_folder, filename_prefix, i, cache):
    async with aiofiles.open(image_path, "rb") as image_file:
        content = await image_file.read()

    if content in cache:
        pil_image, response = cache[content]
    else:
        pil_image = Image.open(io.BytesIO(content))

        orientation_task = correct_image_orientation_async(pil_image)
        detection_task = detect_objects(content, client)

        pil_image, response = await asyncio.gather(orientation_task, detection_task)
        cache[content] = (pil_image, response)

    if response.localized_object_annotations:
        detected_objects = response.localized_object_annotations
        object_vertices = [vertex for obj in detected_objects for vertex in obj.bounding_poly.normalized_vertices]
        if object_vertices:
            x_coords, y_coords = zip(
                *[(vertex.x * pil_image.width, vertex.y * pil_image.height) for vertex in object_vertices])
            left, top, right, bottom = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

            padding = max(pil_image.width, pil_image.height) * 0.13
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(pil_image.width, right + padding)
            bottom = min(pil_image.height, bottom + padding)

            cropped_image = pil_image.crop((left, top, right, bottom))
            original_size = cropped_image.size
            cropped_image.thumbnail((800, 800), Image.LANCZOS)

            image_buffer = io.BytesIO()
            cropped_image.save(image_buffer, format="PNG")
            image_buffer.seek(0)
            transparent_image_buffer = await remove_background_async(image_buffer.read())
            transparent_image = Image.open(io.BytesIO(transparent_image_buffer))

            transparent_image = transparent_image.resize(original_size, Image.LANCZOS)

            final_image = Image.new("RGB", (940, 1215), "white")
            transparent_image.thumbnail((890, 1165), Image.LANCZOS)

            x_center = (940 - transparent_image.width) // 2
            y_center = (1215 - transparent_image.height) // 2

            final_image.paste(transparent_image, (x_center, y_center), transparent_image)

            await save_adjusted_image_async(final_image, output_folder, filename_prefix, i)


async def save_adjusted_image_async(image, output_folder, filename_prefix, i):
    filename = f"{filename_prefix}{i}.jpeg"
    output_path = os.path.join(output_folder, filename)
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        await loop.run_in_executor(executor, partial(image.save, output_path, format='JPEG', dpi=(72, 72)))


async def worker(queue, client, output_folder, filename_prefix, cache):
    while True:
        image_path, index = await queue.get()
        if image_path is None:
            break
        await process_image(image_path, client, output_folder, filename_prefix, index, cache)
        queue.task_done()


async def process_images(image_folder, output_folder="Reloj", max_tasks=10):
    client = get_vision_client()
    os.makedirs(output_folder, exist_ok=True)
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if
                   filename.lower().endswith((".jpg", ".jpeg", ".png"))]

    queue = asyncio.Queue()
    cache = OrderedDict()

    for index, path in enumerate(image_paths):
        await queue.put((path, index + 1))

    workers = []
    for i in range(max_tasks):
        worker_task = asyncio.create_task(worker(queue, client, output_folder, "lentes-", cache))
        workers.append(worker_task)

    await queue.join()

    for i in range(max_tasks):
        await queue.put((None, None))

    await asyncio.gather(*workers)


async def main():
    image_folder = "/Path/"
    await asyncio.gather(process_images(image_folder))


if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f"Se ha tardado {end - start:.2f} segundos")
