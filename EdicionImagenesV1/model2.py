import os
from io import BytesIO
from PIL import Image, ExifTags
from rembg import remove
from multiprocessing import Pool, cpu_count


def correct_orientation(image):
    orientation_key = next((k for k, v in ExifTags.TAGS.items() if v == 'Orientation'), None)
    if not orientation_key:
        return image
    try:
        exif = image.getexif()
        if exif is not None:
            orientation = exif.get(orientation_key)
            rotations = {3: 180, 6: 270, 8: 90}
            if orientation in rotations:
                image = image.rotate(rotations[orientation], expand=True)
    except Exception as e:
        print(f"Error correcting orientation: {e}")
    return image


def remove_background(input_image):
    try:
        input_buffer = BytesIO()
        input_image.save(input_buffer, format='PNG')
        input_buffer.seek(0)
        result = remove(input_buffer.getvalue(), alpha_matting=True,
                        alpha_matting_foreground_threshold=220,
                        alpha_matting_background_threshold=10,
                        alpha_matting_erode_structure_size=5,
                        alpha_matting_erode_size=5)
        return Image.open(BytesIO(result))
    except Exception as e:
        print(f"Error removing background: {e}")
        return input_image


def process_single_image(args):
    image_path, output_path, margin, desired_width, desired_height = args
    try:
        with open(image_path, 'rb') as file:
            image_data = file.read()
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image = correct_orientation(image)

        # Eliminar el fondo utilizando rembg
        image = remove_background(image)

        # Convertir la imagen a RGBA para manejar la transparencia
        image = image.convert("RGBA")

        # Obtener la caja delimitadora del objeto
        bbox = image.getbbox()

        if bbox:
            x1, y1, x2, y2 = bbox
            box_width = x2 - x1
            box_height = y2 - y1

            if box_height > box_width:  # Caja vertical, márgenes en Y
                new_y1 = max(0, y1 - margin)
                new_y2 = min(image.height, y2 + margin)
                crop_height = new_y2 - new_y1
                crop_width = crop_height * (desired_width / desired_height)
                width_center = (x1 + x2) / 2
                new_x1 = max(0, width_center - crop_width / 2)
                new_x2 = min(image.width, width_center + crop_width / 2)
            else:  # Caja horizontal, márgenes en X
                new_x1 = max(0, x1 - margin)
                new_x2 = min(image.width, x2 + margin)
                crop_width = new_x2 - new_x1
                crop_height = crop_width * (desired_height / desired_width)
                height_center = (y1 + y2) / 2
                new_y1 = max(0, height_center - crop_height / 2)
                new_y2 = min(image.height, height_center + crop_height / 2)

            # Realizar el recorte
            cropped_image = image.crop((new_x1, new_y1, new_x2, new_y2))

            # Redimensionar a las dimensiones deseadas sin distorsión
            scale = min(desired_width / cropped_image.width, desired_height / cropped_image.height)
            new_size = (int(cropped_image.width * scale), int(cropped_image.height * scale))
            resized_image = cropped_image.resize(new_size, Image.LANCZOS)

            # Crear una imagen con fondo blanco
            final_image_with_white_bg = Image.new("RGB", (desired_width, desired_height), (255, 255, 255))
            top_left_x = (desired_width - resized_image.width) // 2
            top_left_y = (desired_height - resized_image.height) // 2
            final_image_with_white_bg.paste(resized_image, (top_left_x, top_left_y), resized_image)

            # Guardar la imagen final con 72 DPI
            final_image_with_white_bg.save(output_path, format='JPEG', dpi=(72, 72))
            print(f"Imagen procesada y guardada en: {output_path}")
        else:
            print(f"No se pudo obtener la caja delimitadora para la imagen: {image_path}")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")


def process_images_in_folder(input_folder, output_folder, margin=25, desired_width=940, desired_height=1215):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in image_extensions]

    # Preparar los argumentos para cada proceso
    tasks = [(os.path.join(input_folder, image_file), os.path.join(output_folder, image_file),
              margin, desired_width, desired_height) for image_file in image_files]

    # Usar multiprocessing para procesar en paralelo
    with Pool(cpu_count()) as pool:
        pool.map(process_single_image, tasks)
