import asyncio
import os
import time
import io

from PIL import Image
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
from rembg import remove

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
    return vision_v1.ImageAnnotatorClient()


async def detect_objects(image_content, client):
    return await asyncio.get_event_loop().run_in_executor(None, client.object_localization,
                                                          types.Image(content=image_content))


async def remove_background_async(input_buffer):
    return await asyncio.get_event_loop().run_in_executor(None, remove, input_buffer)


def calculate_aspect_ratio(left, top, right, bottom):
    width = right - left
    height = bottom - top
    aspect_ratio = width / height
    return aspect_ratio


async def process_image(image_path, client, output_folder, filename_prefix, i):
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    pil_image = Image.open(image_path)
    response = await detect_objects(content, client)

    if response.localized_object_annotations:
        detected_objects = response.localized_object_annotations

        object_vertices = []
        for obj in detected_objects:
            vertices = [(vertex.x * pil_image.width, vertex.y * pil_image.height) for vertex in
                        obj.bounding_poly.normalized_vertices]
            object_vertices.extend(vertices)

        if object_vertices:
            x_coords, y_coords = zip(*object_vertices)
            left = min(x_coords)
            top = min(y_coords)
            right = max(x_coords)
            bottom = max(y_coords)

            aspect_ratio = calculate_aspect_ratio(left, top, right, bottom)
            desired_width = 940
            desired_height = int(desired_width / aspect_ratio)

            crop_left = max(left - 100, 0)
            crop_top = max(top - 100, 0)
            crop_right = min(right + 100, pil_image.width)
            crop_bottom = min(bottom + 100, pil_image.height)

            cropped_image = pil_image.crop((crop_left, crop_top, crop_right, crop_bottom))
            adjusted_image = cropped_image.copy()
            adjusted_image.thumbnail((desired_width, desired_height))

            with io.BytesIO() as input_buffer:
                adjusted_image.save(input_buffer, format='PNG')
                input_buffer.seek(0)
                background_removed = await remove_background_async(input_buffer.read())

            adjusted_image = Image.open(io.BytesIO(background_removed))

            save_adjusted_image(adjusted_image, output_folder, filename_prefix, i)


def save_adjusted_image(image, output_folder, filename_prefix, i):
    filename = f"{filename_prefix}{i}.jpeg"
    output_path = os.path.join(output_folder, filename)
    rgb_image = image.convert('RGB')
    rgb_image.save(output_path, dpi=(72, 72))


async def process_images(image_folder, output_folder="prueba", batch_size=10):
    client = get_vision_client()
    os.makedirs(output_folder, exist_ok=True)

    tasks = []
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)
                   if filename.endswith((".jpg", ".jpeg", ".png"))]

    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]

    for batch in batches:
        for image_path in batch:
            task = asyncio.create_task(process_image(image_path, client, output_folder, "DetectAI", len(tasks) + 1))
            tasks.append(task)
        await asyncio.gather(*tasks)
        tasks.clear()


async def main():
    image_folder = "[DIRECTORIO]"
    await asyncio.gather(process_images(image_folder))


if __name__ == "__main__":
    start = time.time()

    asyncio.run(main())

    end = time.time()
    print(f"Se ha tardado {end - start} segundos")
