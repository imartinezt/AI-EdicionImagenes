import asyncio
import os
import time
import io
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from PIL import Image, ExifTags
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


async def correct_image_orientation_async(img, executor):
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

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, correct_orientation, img)


async def process_image(image_path, client, output_folder, filename_prefix, i, executor):
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    pil_image = Image.open(io.BytesIO(content))

    orientation_task = correct_image_orientation_async(pil_image, executor)
    detection_task = detect_objects(content, client)

    pil_image, response = await asyncio.gather(orientation_task, detection_task)

    if response.localized_object_annotations:
        detected_objects = response.localized_object_annotations
        object_vertices = [vertex for obj in detected_objects for vertex in obj.bounding_poly.normalized_vertices]
        if object_vertices:
            x_coords, y_coords = zip(
                *[(vertex.x * pil_image.width, vertex.y * pil_image.height) for vertex in object_vertices])
            left, top, right, bottom = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

            padding = max(pil_image.width, pil_image.height) * 0.12
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
            transparent_image.thumbnail((940, 1215), Image.LANCZOS)
            x_center = (940 - transparent_image.width) // 2
            y_center = (1215 - transparent_image.height) // 2
            final_image.paste(transparent_image, (x_center, y_center), transparent_image)

            await save_adjusted_image_async(final_image, output_folder, filename_prefix, i, executor)


async def save_adjusted_image_async(image, output_folder, filename_prefix, i, executor):
    filename = f"{filename_prefix}{i}.jpeg"
    output_path = os.path.join(output_folder, filename)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, partial(image.save, output_path, format='JPEG', dpi=(72, 72)))


async def process_images(image_folder, output_folder="prueba", batch_size=10):
    client = get_vision_client()
    os.makedirs(output_folder, exist_ok=True)
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if
                   filename.endswith((".jpg", ".jpeg", ".png"))]

    tasks = []
    with ThreadPoolExecutor() as executor:
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            for index, path in enumerate(batch):
                task = process_image(path, client, output_folder, "DetectAI", i + index + 1, executor)
                tasks.append(task)
        await asyncio.gather(*tasks)


async def main():
    image_folder = "[RUTA]"
    await asyncio.gather(process_images(image_folder))


if __name__ == "__main__":
    start = time.time()
    asyncio.run(main())
    end = time.time()
    print(f"Se ha tardado {end - start:.2f} segundos")
