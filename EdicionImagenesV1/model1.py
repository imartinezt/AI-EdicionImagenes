import asyncio
import os
from io import BytesIO

import numpy as np
import torch
from PIL import Image, ExifTags
from skimage.transform import resize
from skimage.util import img_as_ubyte
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


class Model1Processor:
    def __init__(self):
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    def process_model(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
        boxes = results["boxes"].detach().numpy()
        scores = results["scores"].detach().numpy()
        return boxes, scores

    def process_single_image(self, image, margin=25, desired_width=940, desired_height=1215, dpi=(72, 72)):
        try:
            image = correct_orientation(image)
            image_np = np.array(image)

            boxes, scores = self.process_model(image)
            largest_box = boxes[np.argmax(scores)]
            x1, y1, x2, y2 = largest_box

            detected_height = y2 - y1
            image_height, image_width = image_np.shape[:2]
            top_margin = y1
            bottom_margin = image_height - y2
            coverage_percentage = detected_height / image_height

            if (y1 <= 0 and y2 >= image_height) or (coverage_percentage >= 0.9):
                final_image = self.rule1(image_np, x1, y1, x2, y2, desired_width, desired_height)
            elif self.is_aligned_towards_bottom(top_margin, bottom_margin):
                print("Espacio dominante arriba, aplicando Regla 2")
                final_image = self.rule2(image_np, x1, y1, x2, y2, margin, desired_width, desired_height)
            elif top_margin > margin and bottom_margin > margin:
                final_image = self.rule3(image_np, x1, y1, x2, y2, margin, desired_width, desired_height)
            else:
                # Fallback para cualquier otro caso, si es necesario
                print("fallabck")
                final_image = self.rule2(image_np, x1, y1, x2, y2, margin, desired_width, desired_height)

            output_buffer = self.buffer_final(dpi, final_image)
            return output_buffer

        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    @staticmethod
    def is_aligned_towards_bottom(top_margin, bottom_margin, tolerance=0.15):
        """
        Verifica si el objeto está alineado hacia abajo con espacio en la parte superior.
        """
        if top_margin > bottom_margin and bottom_margin / (top_margin + 1e-5) < tolerance:
            return True
        return False

    @staticmethod
    def rule1(image_np, x1, y1, x2, y2, desired_width, desired_height):
        print("Regla 1.")
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        aspect_ratio = desired_width / desired_height

        detected_width = x2 - x1
        detected_height = y2 - y1

        if detected_width / detected_height > aspect_ratio:
            new_crop_height = detected_height
            new_crop_width = new_crop_height * aspect_ratio
        else:
            new_crop_width = detected_width
            new_crop_height = new_crop_width / aspect_ratio

        new_x1 = max(0, center_x - new_crop_width / 2)
        new_x2 = min(image_np.shape[1], center_x + new_crop_width / 2)
        new_y1 = max(0, center_y - new_crop_height / 2)
        new_y2 = min(image_np.shape[0], center_y + new_crop_height / 2)

        cropped_image = image_np[int(new_y1):int(new_y2), int(new_x1):int(new_x2)]
        return Model1Processor.resize_with_skimage(cropped_image, desired_width, desired_height)

    @staticmethod
    def rule2(image_np, x1, y1, x2, y2, margin, desired_width, desired_height):
        print("Regla 2")

        image_height, image_width = image_np.shape[:2]

        if y1 - margin >= 0:
            y1 -= margin
            y2 = y1 + (y2 - y1)
            recorte_medio = True

        elif y2 + margin <= image_height:
            y2 += margin
            y1 = y2 - (y2 - y1)
            recorte_medio = True
        else:
            recorte_medio = False
        crop_height = y2 - y1
        crop_width = crop_height * (desired_width / desired_height)
        x_center = (x1 + x2) / 2
        new_x1 = max(0, x_center - crop_width / 2)
        new_x2 = min(image_width, x_center + crop_width / 2)
        cropped_image = image_np[int(y1):int(y2), int(new_x1):int(new_x2)]

        img = Image.fromarray(img_as_ubyte(cropped_image))

        img_ratio = img.width / img.height
        target_ratio = desired_width / desired_height

        if img_ratio > target_ratio:
            new_height = desired_height
            new_width = int(new_height * img_ratio)
        else:
            new_width = desired_width
            new_height = int(new_width / img_ratio)

        resized_image = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        if resized_image.width > desired_width or resized_image.height > desired_height:
            left = max(0, (resized_image.width - desired_width) // 2)
            top = max(0, (resized_image.height - desired_height) // 2)

            if recorte_medio:
                if y1 < image_height // 2:
                    top = 0
                elif y2 > image_height // 2:
                    top = resized_image.height - desired_height

            right = left + desired_width
            bottom = top + desired_height

            final_image = resized_image.crop((left, top, right, bottom))
        else:
            final_image = resized_image

        return final_image

    @staticmethod
    def rule3(image_np, x1, y1, x2, y2, margin, desired_width, desired_height):
        print("Regla 3.")
        image_height = image_np.shape[0]
        center_x = (x1 + x2) / 2

        new_y1 = max(0, y1 - margin)
        new_y2 = min(image_height, y2 + margin)
        crop_width = x2 - x1
        crop_height = new_y2 - new_y1
        aspect_ratio = desired_width / desired_height

        if crop_width / crop_height > aspect_ratio:
            crop_height = crop_width / aspect_ratio
            new_y1 = max(0, (y1 + y2) / 2 - crop_height / 2)
            new_y2 = min(image_height, new_y1 + crop_height)
        else:
            crop_width = crop_height * aspect_ratio
            new_x1 = max(0, center_x - crop_width / 2)
            new_x2 = min(image_np.shape[1], center_x + crop_width / 2)

        cropped_image = image_np[int(new_y1):int(new_y2), int(new_x1):int(new_x2)]
        return Model1Processor.resize_with_skimage(cropped_image, desired_width, desired_height)

    @staticmethod
    def resize_with_skimage(image_np, desired_width, desired_height):
        resized_image = resize(image_np, (desired_height, desired_width), anti_aliasing=True)
        return Image.fromarray(img_as_ubyte(resized_image))

    @staticmethod
    def buffer_final(dpi, final_image):
        output_buffer = BytesIO()
        final_image.save(output_buffer, format='JPEG', dpi=dpi)
        output_buffer.seek(0)
        return output_buffer


async def process_image(image_file, input_folder, output_folder, margin=25, desired_width=940, desired_height=1215):
    input_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, image_file)
    loop = asyncio.get_running_loop()
    processor = Model1Processor()
    await loop.run_in_executor(None, processor.process_single_image, input_path, output_path, margin, desired_width,
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
