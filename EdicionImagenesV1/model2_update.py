import asyncio
import os
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session


@dataclass
class ProcessingSettings:
    """Configuración de procesamiento"""
    target_width: int = 940
    target_height: int = 1215
    margin_pixels: int = 25
    background_color: Tuple[int, int, int] = (255, 255, 255)
    output_dpi: Tuple[int, int] = (72, 72)
    jpeg_quality: int = 95
    max_workers: int = 4
    backend: str = 'u2netp'  # 'u2net', 'u2netp', 'u2net_human_seg', 'silueta'
    quality_mode: str = 'balanced'  # 'fast', 'balanced', 'high'
    output_dir: str = 'processed_products'  # TODO --> Luis aqui lo cambias por el que ya esta en la logica main <---

class ObjectOrientation(Enum):
    """Orientación del objeto detectado"""
    VERTICAL = "vertical"  # Altura > Ancho → márgenes arriba/abajo
    HORIZONTAL = "horizontal"  # Ancho > Altura → márgenes izquierda/derecha
    SQUARE = "square"  # Casi cuadrado → márgenes uniformes
    NO_OBJECT = "no_object"  # Sin detección → centrado simple


class FastBackgroundRemover:
    """Gestión optimizada de remoción de fondo con múltiples backends"""

    def __init__(self, backend='u2netp', quality_mode='balanced'):
        """
        Backends disponibles, [SISTEMA HIBRIDO] --> Next Update seria usar uina combinacion de u2net_cloth_seg y u2netp
        - u2netp: Más rápido que u2net, buena calidad
        - u2net_human_seg: Optimizado para personas
        - u2net_cloth_seg: Optimizado para ropa
        - silueta: Más rápido, menor calidad

        Modos de calidad:
        - 'fast': Sin alpha matting, más rápido
        - 'balanced': Alpha matting optimizado
        - 'high': Alpha matting completo, más lento
        """
        self.backend = backend
        self.quality_mode = quality_mode
        self.session = None
        self._init_session()

    def _init_session(self):
        """Inicializar sesión según backend"""
        try:
            self.session = new_session(self.backend)
            print(f"✅ Backend '{self.backend}' inicializado")
        except:
            print(f"⚠️ Backend '{self.backend}' no disponible, usando u2netp")
            self.backend = 'u2netp'
            self.session = new_session('u2netp')

    def remove_background(self, image: Image.Image) -> Image.Image:
        """Remover fondo con configuración optimizada según modo de calidad"""
        original_size = image.size
        max_size = 1536 if self.quality_mode == 'high' else 1024
        was_resized = False

        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            working_image = image.resize(new_size, Image.Resampling.LANCZOS)
            was_resized = True
        else:
            working_image = image

        input_buffer = BytesIO()
        working_image.save(input_buffer, format='PNG')
        input_buffer.seek(0)

        if self.quality_mode == 'fast':
            # Modo rápido sin alpha matting
            result = remove(
                input_buffer.getvalue(),
                session=self.session,
                alpha_matting=False,
                only_mask=False
            )

        elif self.quality_mode == 'balanced':
            # Modo balanceado - alpha matting optimizado
            result = remove(
                input_buffer.getvalue(),
                session=self.session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,  # Más alto = más agresivo
                alpha_matting_background_threshold=15,  # Más bajo = mejor detección
                alpha_matting_erode_size=10  # Suavizado de bordes
            )

        else:  # high quality
            # Modo alta calidad - alpha matting completo
            result = remove(
                input_buffer.getvalue(),
                session=self.session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=220,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_structure_size=10,
                alpha_matting_erode_size=10
            )

        result_image = Image.open(BytesIO(result)).convert("RGBA")
        if was_resized:
            result_image = result_image.resize(original_size, Image.Resampling.LANCZOS)

        if self.quality_mode in ['balanced', 'high']:
            result_image = self._enhance_edges(result_image)

        return result_image

    def _enhance_edges(self, image: Image.Image) -> Image.Image:
        """Post-procesamiento para mejorar bordes y suavidad"""
        if image.mode != 'RGBA':
            return image

        r, g, b, a = image.split()
        a_array = np.array(a)
        kernel = np.ones((3, 3), np.uint8)
        a_smoothed = cv2.morphologyEx(a_array, cv2.MORPH_OPEN, kernel, iterations=1)
        a_smoothed = cv2.GaussianBlur(a_smoothed, (3, 3), 0)
        # Los valores entre 50-200 se mapean suavemente
        a_smoothed = np.where(a_smoothed > 50,
                              np.minimum(255, a_smoothed * 1.2),
                              a_smoothed * 0.5)
        a_enhanced = Image.fromarray(a_smoothed.astype(np.uint8))
        image.putalpha(a_enhanced)

        return image


class OrientationCorrector:
    """Corrección de orientación EXIF"""

    @staticmethod
    def correct_orientation(image: Image.Image) -> Image.Image:
        """Corregir orientación basada en EXIF"""
        try:
            exif = image.getexif()
            if not exif:
                return image

            orientation = exif.get(0x0112)  # Orientation tag
            if not orientation:
                return image

            rotations = {3: 180, 6: 270, 8: 90}
            if orientation in rotations:
                return image.rotate(rotations[orientation], expand=True)

        except Exception:
            pass

        return image


class ObjectAnalyzer:
    """Análisis de objetos optimizado"""

    @staticmethod
    def get_object_bounds_fast(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """Versión rápida de detección de límites"""
        if image.mode != 'RGBA':
            return image.getbbox()
        bbox = image.getbbox()

        if not bbox:
            alpha = np.array(image.split()[-1])
            if not alpha.any():
                return None

            rows = np.any(alpha > 10, axis=1)
            cols = np.any(alpha > 10, axis=0)

            if not (rows.any() and cols.any()):
                return None

            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]

            return (x_indices[0], y_indices[0], x_indices[-1] + 1, y_indices[-1] + 1)

        return bbox

    @staticmethod
    def determine_orientation(bbox: Tuple[int, int, int, int]) -> ObjectOrientation:
        """Determinar orientación del objeto"""
        if not bbox:
            return ObjectOrientation.NO_OBJECT

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        ratio = width / height if height > 0 else 1

        # Tolerancia del 15% para considerar cuadrado
        if 0.85 <= ratio <= 1.15:
            return ObjectOrientation.SQUARE
        elif ratio < 1:
            return ObjectOrientation.VERTICAL
        else:
            return ObjectOrientation.HORIZONTAL

    @staticmethod
    def detect_edge_touching(image: Image.Image, bbox: Tuple[int, int, int, int],
                             threshold: int = 5) -> dict:
        """
        Detectar qué bordes está tocando el objeto
        threshold: píxeles de tolerancia para considerar que toca el borde
        """
        if not bbox:
            return {'top': False, 'bottom': False, 'left': False, 'right': False}

        x1, y1, x2, y2 = bbox
        img_width, img_height = image.size

        return {
            'top': y1 <= threshold,
            'bottom': y2 >= (img_height - threshold),
            'left': x1 <= threshold,
            'right': x2 >= (img_width - threshold)
        }


class Model2Processor:
    """Procesador principal optimizado"""

    def __init__(self, settings: ProcessingSettings = ProcessingSettings()):
        self.settings = settings
        self.bg_remover = FastBackgroundRemover(settings.backend, settings.quality_mode)
        self.orientation_corrector = OrientationCorrector()
        self.analyzer = ObjectAnalyzer()

        os.makedirs(settings.output_dir, exist_ok=True)

    def process_single_image(self, image: Image.Image) -> Optional[Image.Image]:
        """Procesar una imagen individual"""
        try:
            image = self.orientation_corrector.correct_orientation(image)
            image_no_bg = self.bg_remover.remove_background(image)
            bbox = self.analyzer.get_object_bounds_fast(image_no_bg)

            if not bbox:
                return self._apply_no_object_rule(image_no_bg)
            edges_touching = self.analyzer.detect_edge_touching(image_no_bg, bbox)

            orientation = self.analyzer.determine_orientation(bbox)
            if orientation == ObjectOrientation.VERTICAL:
                return self._apply_vertical_rule_smart(image_no_bg, bbox, edges_touching)
            elif orientation == ObjectOrientation.HORIZONTAL:
                return self._apply_horizontal_rule_smart(image_no_bg, bbox, edges_touching)
            elif orientation == ObjectOrientation.SQUARE:
                return self._apply_square_rule_smart(image_no_bg, bbox, edges_touching)
            else:
                return self._apply_no_object_rule(image_no_bg)

        except Exception as e:
            print(f"Error procesando imagen: {e}")
            return None

    def _apply_vertical_rule_smart(self, image: Image.Image, bbox: Tuple[int, int, int, int],
                                   edges_touching: dict) -> Image.Image:
        """Regla 1 inteligente: Objeto vertical - márgenes solo donde no toca bordes"""
        x1, y1, x2, y2 = bbox
        obj_width = x2 - x1
        obj_height = y2 - y1
        margin_top = 0 if edges_touching['top'] else self.settings.margin_pixels
        margin_bottom = 0 if edges_touching['bottom'] else self.settings.margin_pixels
        scale_with_margins = min(
            self.settings.target_width / obj_width,
            (self.settings.target_height - margin_top - margin_bottom) / obj_height
        )

        final_obj_width = int(obj_width * scale_with_margins)
        final_obj_height = int(obj_height * scale_with_margins)

        final_image = Image.new('RGB',
                                (self.settings.target_width, self.settings.target_height),
                                self.settings.background_color)
        obj_image = image.crop(bbox)
        obj_scaled = obj_image.resize((final_obj_width, final_obj_height), Image.Resampling.LANCZOS)
        x_pos = (self.settings.target_width - final_obj_width) // 2

        if edges_touching['top']:
            y_pos = 0
        elif edges_touching['bottom']:
            y_pos = self.settings.target_height - final_obj_height
        else:
            y_pos = (self.settings.target_height - final_obj_height) // 2

        final_image.paste(obj_scaled, (x_pos, y_pos), obj_scaled)

        return final_image

    def _apply_horizontal_rule_smart(self, image: Image.Image, bbox: Tuple[int, int, int, int],
                                     edges_touching: dict) -> Image.Image:
        """Regla 2 inteligente: Objeto horizontal - márgenes solo donde no toca bordes"""
        x1, y1, x2, y2 = bbox
        obj_width = x2 - x1
        obj_height = y2 - y1
        margin_left = 0 if edges_touching['left'] else self.settings.margin_pixels
        margin_right = 0 if edges_touching['right'] else self.settings.margin_pixels
        scale_with_margins = min(
            (self.settings.target_width - margin_left - margin_right) / obj_width,
            self.settings.target_height / obj_height
        )

        final_obj_width = int(obj_width * scale_with_margins)
        final_obj_height = int(obj_height * scale_with_margins)
        final_image = Image.new('RGB',
                                (self.settings.target_width, self.settings.target_height),
                                self.settings.background_color)
        obj_image = image.crop(bbox)
        obj_scaled = obj_image.resize((final_obj_width, final_obj_height), Image.Resampling.LANCZOS)
        if edges_touching['left']:
            x_pos = 0
        elif edges_touching['right']:
            x_pos = self.settings.target_width - final_obj_width
        else:
            x_pos = (self.settings.target_width - final_obj_width) // 2

        y_pos = (self.settings.target_height - final_obj_height) // 2

        final_image.paste(obj_scaled, (x_pos, y_pos), obj_scaled)

        return final_image

    def _apply_square_rule_smart(self, image: Image.Image, bbox: Tuple[int, int, int, int],
                                 edges_touching: dict) -> Image.Image:
        """Regla 3 inteligente: Objeto cuadrado - márgenes solo donde no toca bordes"""
        x1, y1, x2, y2 = bbox
        obj_width = x2 - x1
        obj_height = y2 - y1

        margin_top = 0 if edges_touching['top'] else self.settings.margin_pixels
        margin_bottom = 0 if edges_touching['bottom'] else self.settings.margin_pixels
        margin_left = 0 if edges_touching['left'] else self.settings.margin_pixels
        margin_right = 0 if edges_touching['right'] else self.settings.margin_pixels

        obj_size = max(obj_width, obj_height)
        scale_h = (self.settings.target_width - margin_left - margin_right) / obj_size
        scale_v = (self.settings.target_height - margin_top - margin_bottom) / obj_size
        scale = min(scale_h, scale_v)
        final_size = int(obj_size * scale)
        final_image = Image.new('RGB',
                                (self.settings.target_width, self.settings.target_height),
                                self.settings.background_color)

        obj_image = image.crop(bbox)
        square_img = Image.new('RGBA', (obj_size, obj_size), (0, 0, 0, 0))
        x_offset = (obj_size - obj_width) // 2
        y_offset = (obj_size - obj_height) // 2
        square_img.paste(obj_image, (x_offset, y_offset))

        obj_scaled = square_img.resize((final_size, final_size), Image.Resampling.LANCZOS)
        if edges_touching['left']:
            x_pos = 0
        elif edges_touching['right']:
            x_pos = self.settings.target_width - final_size
        else:
            x_pos = (self.settings.target_width - final_size) // 2

        if edges_touching['top']:
            y_pos = 0
        elif edges_touching['bottom']:
            y_pos = self.settings.target_height - final_size
        else:
            y_pos = (self.settings.target_height - final_size) // 2

        final_image.paste(obj_scaled, (x_pos, y_pos), obj_scaled)

        return final_image

    def _apply_no_object_rule(self, image: Image.Image) -> Image.Image:
        """Regla 4: Sin objeto detectado - centrar imagen completa"""
        img_width, img_height = image.size
        scale = min(
            self.settings.target_width / img_width,
            self.settings.target_height / img_height
        ) * 0.9  # 90% para margen

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        final_image = Image.new('RGB',
                                (self.settings.target_width, self.settings.target_height),
                                self.settings.background_color)
        scaled = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Centrar
        x_pos = (self.settings.target_width - new_width) // 2
        y_pos = (self.settings.target_height - new_height) // 2

        final_image.paste(scaled, (x_pos, y_pos), scaled if scaled.mode == 'RGBA' else None)

        return final_image

    def process_images_batch(self, image_paths: List[str]) -> List[Tuple[str, bool]]:
        """Procesar lote de imágenes y guardar resultados"""
        results = []

        for i, path in enumerate(image_paths):
            try:
                print(f"📸 Procesando {i + 1}/{len(image_paths)}: {os.path.basename(path)}")

                with Image.open(path) as img:
                    result = self.process_single_image(img)

                    if result:
                        filename = os.path.basename(path)
                        name, _ = os.path.splitext(filename)
                        output_path = os.path.join(
                            self.settings.output_dir,
                            f"{name}.jpg"
                        )
                        result.save(
                            output_path,
                            format='JPEG',
                            quality=self.settings.jpeg_quality,
                            dpi=self.settings.output_dpi,
                            optimize=True
                        )

                        print(f"   ✅ Guardado: {output_path}")
                        results.append((output_path, True))
                    else:
                        print(f"   ❌ Error procesando")
                        results.append((path, False))

            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append((path, False))

        successful = sum(1 for _, success in results if success)
        print(f"\n📊 Resumen: {successful}/{len(image_paths)} procesadas exitosamente")
        print(f"📁 Resultados en: {self.settings.output_dir}/")

        return results

    async def process_images_async(self, images: List[Image.Image]) -> List[Optional[Image.Image]]:
        """Versión asíncrona para procesamiento"""
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self.process_single_image, img)
            for img in images
        ]

        return await asyncio.gather(*tasks)


def process_products_fast(image_paths: List[str],
                          backend: str = 'u2netp',
                          quality_mode: str = 'balanced',
                          output_dir: str = 'processed_products'):
    """
    Procesar productos con diferentes modos de calidad

    quality_mode:
    - 'fast': ~2s por imagen, bordes básicos
    - 'balanced': ~3-4s por imagen, buenos bordes (recomendado)
    - 'high': ~5-6s por imagen, máxima calidad
    """

    settings = ProcessingSettings(
        backend=backend,
        quality_mode=quality_mode,
        output_dir=output_dir
    )

    processor = Model2Processor(settings)

    print(f"🚀 Procesando {len(image_paths)} imágenes...")
    print(f"⚙️  Backend: {backend}")
    print(f"🎨 Calidad: {quality_mode}")
    print(f"📁 Salida: {output_dir}/")
    print("-" * 50)

    results = processor.process_images_batch(image_paths)

    return results


def process_product_image(image_path: str, output_path: str = "output_product.jpg"):
    """Procesar una imagen de producto"""
    processor = Model2Processor()

    with Image.open(image_path) as img:
        result = processor.process_single_image(img)

        if result:
            result.save(
                output_path,
                format='JPEG',
                quality=processor.settings.jpeg_quality,
                dpi=processor.settings.output_dpi,
                optimize=True
            )
            print(f"✅ Imagen procesada: {output_path}")
            return result
        else:
            print("❌ Error procesando imagen")
            return None


def process_images_in_folder(input_folder: str, output_folder: str):
    """
    Método llamado en la interfaz gráfica

    :param input_folder: El directorio que contiene todas las imágenes a procesar
    :param output_folder: El directorio donde se guardaran las imágenes procesadas
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(input_folder) if os.path.splitext(f)[1].lower() in image_extensions]

    image_paths = []
    for image_file in image_files:

        # Generamos el path completo (considerando image file e input path)
        image_input_path = os.path.join(input_folder, image_file)

        # Agregamos la ruta actual a la lista de rutas
        image_paths.append(image_input_path)

    # Mandamos a llamar la nueva lógica del modelo

    results = process_products_fast(
        image_paths= image_paths,
        backend='u2net',
        quality_mode='balanced',  # fast, high, balanced
        output_dir=output_folder)


# if __name__ == "__main__":
#     image_paths = [
#         "products/1114342919_2p.jpg",
#         "products/1132714947_1p.jpg",
#         "products/1133550590.jpg",
#         "products/1142090534_2p.JPG",
#         "products/1142325612_2p.JPG",
#         "products/1142511416.jpg",
#         "products/1149005095.JPG",
#         "products/1155721835_3p.JPG",
#         "products/1155721835_4p.JPG",
#         "products/mes_ev_TEST_00083.JPG",
#         "products/mes_ref_TEST_00033.JPG",
#         "products/mes_TEST_00004.JPG",
#         "products/mes_TEST_00012.JPG"
#     ]
#     results = process_products_fast(
#         image_paths,
#         backend='u2netp', #u2netp
#         quality_mode='balanced',  # fast, high, balanced
#         output_dir='productos_procesados'
#     )