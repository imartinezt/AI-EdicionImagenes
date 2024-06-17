import asyncio
import io
import json
import os
from functools import partial

import aiofiles
from PIL import Image, ExifTags
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
from google.oauth2 import service_account
from rembg import remove


def get_vision_client():
    sa_path = "keys.json"
    with open(sa_path) as source:
        info = json.load(source)
    creds = service_account.Credentials.from_service_account_info(info)
    return vision_v1.ImageAnnotatorClient(credentials=creds)


async def detect_objects(image_content, client):
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, client.object_localization, types.Image(content=image_content))
    return response.localized_object_annotations


async def remove_background_async(input_buffer):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(remove, input_buffer, alpha_matting=True,
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
    return await loop.run_in_executor(None, correct_orientation, img)


async def process_image(image_path, client, salida_folder, filename_prefix):
    async with aiofiles.open(image_path, "rb") as image_file:
        content = await image_file.read()

    pil_image = Image.open(io.BytesIO(content))

    orientation_task = correct_image_orientation_async(pil_image)
    detection_task = detect_objects(content, client)

    pil_image, detected_objects = await asyncio.gather(orientation_task, detection_task)

    if detected_objects:
        object_vertices = [vertex for obj in detected_objects for vertex in obj.bounding_poly.normalized_vertices]
        if object_vertices:
            x_coords, y_coords = zip(
                *[(vertex.x * pil_image.width, vertex.y * pil_image.height) for vertex in object_vertices])
            left, top, right, bottom = min(x_coords), min(y_coords), max(x_coords), max(y_coords)

            object_width = right - left
            object_height = bottom - top
            margin_x = object_width * 0.1
            margin_y = object_height * 0.1

            left = max(left - margin_x, 1)
            top = max(top - margin_y, 1)
            right = min(right + margin_x, pil_image.width)
            bottom = min(bottom + margin_y, pil_image.height)

            cropped_image = pil_image.crop((left, top, right, bottom))
            image_buffer = io.BytesIO()
            cropped_image.save(image_buffer, format="PNG")
            image_buffer.seek(0)

            transparent_image_buffer = await remove_background_async(image_buffer.read())
            transparent_image = Image.open(io.BytesIO(transparent_image_buffer))

            final_width, final_height = 940, 1215
            final_image = Image.new("RGB", (final_width, final_height), "white")

            object_width, object_height = transparent_image.size
            aspect_ratio = object_width / object_height

            if aspect_ratio > final_width / final_height:
                new_width = final_width - 50
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = final_height - 50
                new_width = int(new_height * aspect_ratio)

            resized_image = transparent_image.resize((new_width, new_height), Image.LANCZOS)
            x_center = (final_width - new_width) // 2
            y_center = (final_height - new_height) // 2
            final_image.paste(resized_image, (x_center, y_center), resized_image)
            await save_adjusted_image_async(final_image, salida_folder, filename_prefix, os.path.basename(image_path))


async def save_adjusted_image_async(image, salida_folder, filename_prefix, original_filename):
    filename, ext = os.path.splitext(original_filename)
    filename = f"{filename_prefix}{filename}{ext}"
    output_path = os.path.join(salida_folder, filename)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, partial(image.save, output_path, format='JPEG', dpi=(72, 72)))


output_folder = os.path.expanduser("~/Desktop/SalidaModel2AI")
os.makedirs(output_folder, exist_ok=True)


async def process_images(image_folder, salida_folder=output_folder, max_tasks=10):
    client = get_vision_client()
    os.makedirs(salida_folder, exist_ok=True)
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if
                   filename.lower().endswith((".jpg", ".jpeg", ".png"))]

    queue = asyncio.Queue()
    for index, path in enumerate(image_paths):
        await queue.put((path, index + 1))

    workers = []
    for i in range(max_tasks):
        worker_task = asyncio.create_task(worker(queue, client, salida_folder, "AI-"))
        workers.append(worker_task)

    await queue.join()

    for i in range(max_tasks):
        await queue.put((None, None))

    await asyncio.gather(*workers)


async def worker(queue, client, salida_folder, filename_prefix):
    while True:
        image_path, index = await queue.get()
        if image_path is None:
            break
        await process_image(image_path, client, salida_folder, filename_prefix)
        queue.task_done()
