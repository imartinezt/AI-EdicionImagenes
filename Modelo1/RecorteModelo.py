import asyncio
import io
import os
import cv2
import numpy as np
import rawpy
import torch
from PIL import Image, ExifTags
from transformers import DetrImageProcessor, DetrForObjectDetection


async def detect_objects(image_content, model, processor, thread_executor, threshold=0.1, target_classes=None):
    if target_classes is None:
        target_classes = ['person']

    loop = asyncio.get_event_loop()
    try:
        image = Image.open(io.BytesIO(image_content))
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = await loop.run_in_executor(thread_executor, lambda: model(**inputs))

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []
        for box, score, label in zip(results['boxes'], results['scores'], results['labels']):
            if model.config.id2label[label.item()] in target_classes:
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_labels.append(label)

        filtered_results = {
            'boxes': torch.stack(filtered_boxes) if filtered_boxes else torch.tensor([]),
            'scores': torch.stack(filtered_scores) if filtered_scores else torch.tensor([]),
            'labels': torch.stack(filtered_labels) if filtered_labels else torch.tensor([])
        }

        return filtered_results
    except Exception as e:
        print(f"Error processing image content: {e}")
        return None


def compress_image_sync(image_path, quality):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        output_buffer = io.BytesIO()
        img.save(output_buffer, "JPEG", quality=quality)
        return output_buffer.getvalue()


async def compress_image(image_path, quality, process_executor):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(process_executor, compress_image_sync, image_path, quality)


def crop_and_center_image_sync(image_array, object_polygon, target_size, margin=55):
    # Obtener el rectángulo delimitador del polígono
    x, y, w, h = cv2.boundingRect(object_polygon)

    object_width = w
    object_height = h

    # Calcular el centro del objeto detectado
    center_x = x + w // 2
    center_y = y + h // 2

    target_width, target_height = target_size
    aspect_ratio = target_width / target_height

    # Calcular el ancho y alto del recorte manteniendo el aspecto
    if object_width / object_height > aspect_ratio:
        crop_width = object_width + 2 * margin
        crop_height = crop_width / aspect_ratio
    else:
        crop_height = object_height + 2 * margin
        crop_width = crop_height * aspect_ratio

    # Calcular las coordenadas del recorte centrado
    new_left = max(int(center_x - crop_width // 2), 0)
    new_top = max(int(center_y - crop_height // 2), 0)
    new_right = min(new_left + int(crop_width), image_array.shape[1])
    new_bottom = min(new_top + int(crop_height), image_array.shape[0])

    # Ajustar el recorte si es necesario
    if new_right - new_left < crop_width:
        new_left = max(int(new_right - crop_width), 0)
    if new_bottom - new_top < crop_height:
        new_top = max(int(new_bottom - crop_height), 0)

    cropped_image = image_array[new_top:new_bottom, new_left:new_right]

    # Ajustar el recorte para que sea exactamente del tamaño objetivo y centrado
    cropped_height, cropped_width = cropped_image.shape[:2]
    delta_w = target_width - cropped_width
    delta_h = target_height - cropped_height

    top, bottom = max(delta_h // 2, 0), max(delta_h - (delta_h // 2), 0)
    left, right = max(delta_w // 2, 0), max(delta_w - (delta_w // 2), 0)

    color = [0, 0, 0]  # Negro
    new_image = cv2.copyMakeBorder(cropped_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    new_image = cv2.resize(new_image, target_size)

    pil_cropped_image = Image.fromarray(new_image)

    return pil_cropped_image


async def crop_and_center_image(image_array, object_polygon, target_size, margin, process_executor):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(process_executor, crop_and_center_image_sync, image_array,
                                      object_polygon, target_size, margin)


def save_image_sync(image, path, dpi):
    image.save(path, dpi=(dpi, dpi))


async def save_image(image, path, dpi, thread_executor):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(thread_executor, save_image_sync, image, path, dpi)


async def process_image(image_path, model, processor, thread_executor, process_executor):
    compressed_image_content = await compress_image(image_path, 70, process_executor)
    objects = await detect_objects(compressed_image_content, model, processor, thread_executor)

    pil_image = await load_image(image_path, thread_executor)

    if pil_image:
        pil_image = await adjust_image_orientation(pil_image, thread_executor)
        image_array = np.array(pil_image)

        if objects and 'boxes' in objects and len(objects['boxes']) > 0:
            first_object = objects['boxes'][0]
            first_polygon = np.array([[int(first_object[0]), int(first_object[1])],
                                      [int(first_object[2]), int(first_object[1])],
                                      [int(first_object[2]), int(first_object[3])],
                                      [int(first_object[0]), int(first_object[3])]])

            adjusted_image = await crop_and_center_image(image_array, first_polygon, (940, 1215), margin=200,
                                                         process_executor=process_executor)
            return pil_image, adjusted_image

    return None, None


async def load_image(image_path, thread_executor):
    loop = asyncio.get_event_loop()
    try:
        if image_path.lower().endswith(('.nef', '.cr2', '.arw', '.dng', '.rw2', '.orf', '.srw', '.raw')):
            with rawpy.imread(image_path) as raw:
                rgb_image = raw.postprocess()
                pil_image = Image.fromarray(rgb_image)
        else:
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
            pil_image = await loop.run_in_executor(thread_executor, Image.open, io.BytesIO(image_data))
        return pil_image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


async def adjust_image_orientation(pil_image, thread_executor):
    loop = asyncio.get_event_loop()
    try:
        orientation_tag = [tag for tag, description in ExifTags.TAGS.items() if description == 'Orientation'][0]
        exif = await loop.run_in_executor(thread_executor, pil_image.getexif)
        if exif is not None and orientation_tag in exif:
            orientation_value = exif[orientation_tag]
            orientation_dict = {3: 180, 6: 270, 8: 90}
            rotation_angle = orientation_dict.get(orientation_value, 0)

            if rotation_angle:
                pil_image = await loop.run_in_executor(thread_executor,
                                                       lambda: pil_image.rotate(rotation_angle, expand=True))
    except (AttributeError, KeyError, IndexError, Exception) as e:
        print(f"Error adjusting image orientation: {e}")

    return pil_image


async def adjust_image_resolution(image_path, dpi, thread_executor):
    loop = asyncio.get_event_loop()
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        await loop.run_in_executor(thread_executor, save_image_sync, img, image_path, dpi)


output_folder = os.path.expanduser("~/Desktop/SalidaModel1AI")
os.makedirs(output_folder, exist_ok=True)


async def process_images_async(image_paths, salida_folder=output_folder, output_dpi=72, batch_size=5,
                               thread_executor=None, process_executor=None):
    if not os.path.exists(salida_folder):
        os.makedirs(salida_folder)

    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]

    for batch in batches:
        tasks = [process_image(image_path, model, processor, thread_executor, process_executor) for image_path in batch]
        results = await asyncio.gather(*tasks)

        for result, image_path in zip(results, batch):
            original_image, adjusted_image = result
            if original_image and adjusted_image:
                output_path = os.path.join(salida_folder, os.path.basename(image_path))
                await save_image(adjusted_image, output_path, output_dpi, thread_executor)
            else:
                print(f"No se pudo procesar la imagen {os.path.basename(image_path)}.")


async def list_image_paths(folder_paths):
    image_paths = []
    for folder_path in folder_paths:
        if os.path.exists(folder_path):
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(
                            ('.png', '.jpg', '.jpeg', '.nef', '.cr2', '.arw', '.dng', '.rw2', '.orf', '.srw', '.raw')):
                        image_paths.append(os.path.join(root, file))
        else:
            print(f"La carpeta {folder_path} no existe.")
    return image_paths

