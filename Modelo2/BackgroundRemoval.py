import asyncio
import os
import time
import io
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


def correct_image_orientation(img):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif is not None:
            exif = dict(exif.items())
            if exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img = img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img = img.rotate(90, expand=True)
    except Exception as e:
        print(f"Error correcting orientation: {e}")
    return img


async def process_image(image_path, client, output_folder, filename_prefix, i):
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    pil_image = Image.open(io.BytesIO(content))
    pil_image = correct_image_orientation(pil_image)

    response = await detect_objects(content, client)
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
            image_buffer = io.BytesIO()
            cropped_image.save(image_buffer, format="PNG")
            image_buffer.seek(0)
            transparent_image_buffer = await remove_background_async(image_buffer.read())
            transparent_image = Image.open(io.BytesIO(transparent_image_buffer))

            final_image = Image.new("RGB", (940, 1215), "white")
            transparent_image.thumbnail((940, 1215), Image.LANCZOS)
            x_center = (940 - transparent_image.width) // 2
            y_center = (1215 - transparent_image.height) // 2
            final_image.paste(transparent_image, (x_center, y_center), transparent_image)

            save_adjusted_image(final_image, output_folder, filename_prefix, i)


def save_adjusted_image(image, output_folder, filename_prefix, i):
    filename = f"{filename_prefix}{i}.jpeg"
    output_path = os.path.join(output_folder, filename)
    image.save(output_path, format="JPEG", dpi=(72, 72))


async def process_images(image_folder, output_folder="prueba", batch_size=10):
    client = get_vision_client()
    os.makedirs(output_folder, exist_ok=True)
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if
                   filename.endswith((".jpg", ".jpeg", ".png"))]
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        tasks = [asyncio.create_task(process_image(path, client, output_folder, "DetectAI", i + index + 1)) for
                 index, path in enumerate(batch)]
        await asyncio.gather(*tasks)


async def main():
    image_folder = "[DIRECTORIO]"
    await asyncio.gather(process_images(image_folder))


if __name__ == "__main__":
    start = time.time()

    asyncio.run(main())

    end = time.time()
    print(f"Se ha tardado {end - start:.2f} segundos")
