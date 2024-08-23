import asyncio
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ExifTags
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


def process_single_image(image_path, output_path, margin=25, desired_width=940, desired_height=1215):
    try:
        image = Image.open(image_path)
        image = correct_orientation(image)

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
        boxes = results["boxes"].detach().numpy()
        scores = results["scores"].detach().numpy()


        largest_box = boxes[np.argmax(scores)]
        x1, y1, x2, y2 = largest_box

        # Ajuste de la caja delimitadora para añadir margen solo en el eje Y
        new_y1 = max(0, y1 - margin)
        new_y2 = min(image.height, y2 + margin)

        # proprociones
        crop_height = new_y2 - new_y1
        crop_width = crop_height * (desired_width / desired_height)
        width_center = (x1 + x2) / 2

        # valid para que no exceda bordes
        new_x1 = max(0, width_center - crop_width / 2)
        new_x2 = min(image.width, width_center + crop_width / 2)

        # Recortar la imagen
        cropped_image = image.crop((new_x1, new_y1, new_x2, new_y2))

        # Redimensionar proporcionalmente al lado más largo
        cropped_image.thumbnail((desired_width, desired_height), Image.LANCZOS)

        # check de  las dimensiones finales y ajustar mediante recorte
        final_image = cropped_image

        if final_image.width > desired_width:
            excess_width = (final_image.width - desired_width) // 2
            final_image = final_image.crop((excess_width, 0, final_image.width - excess_width, final_image.height))

        if final_image.height > desired_height:
            excess_height = (final_image.height - desired_height) // 2
            final_image = final_image.crop((0, excess_height, final_image.width, final_image.height - excess_height))

        # Guardar la imagen final
        final_image.save(output_path, format='JPEG', dpi=(72, 72))
        print(f"Imagen procesada y guardada en: {output_path}")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")


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


def main():
    input_folder = "/Users/imartinezt/Downloads/EDICION_AI/pruebas"
    output_folder = "/Users/imartinezt/Desktop/SalidaModel1AI"
    asyncio.run(process_images_in_folder(input_folder, output_folder))


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Se ha tardado {end - start} segundos")
