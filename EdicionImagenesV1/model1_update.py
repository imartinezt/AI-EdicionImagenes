import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ExifTags
from ultralytics import YOLO
import os
import sys

"""
@Autor: Iván Martínez Trejo.

Sistema de Crop Sin Distorsión para Retail de Moda
Elimina completamente la distorsión usando técnicas matemáticas avanzadas
Preserva proporciones naturales de las personas detectadas
"""

def resource_path(relative_path):
    """
    Obtenemos la ruta absoluta a los recursos como íconos, imágenes, etc.
    En este caso hay que acceder a las rutas de los modelos .pt
    """

    try:

        base_path = sys._MEIPASS

    except Exception:

        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class Settings:
    """Configuración anti-distorsión"""
    target_width: int = 940
    target_height: int = 1215
    target_ratio: float = 940 / 1215  # ≈ 0.774
    target_dpi: Tuple[int, int] = (72, 72)
    min_confidence: float = 0.7
    model_path: str = "yolo11n-pose.pt"
    margin_pixels: int = 25
    max_distortion_tolerance: float = 0.02  # Máximo 2% de distorsión permitida
    prefer_crop_over_resize: bool = True  # Preferir crop vs resize
    use_proportional_margins: bool = True  # Usar márgenes proporcionales
    min_person_height_ratio: float = 0.6  # Mínimo 60% de altura para persona


class CropRule(Enum):
    """Las 4 reglas sin distorsión"""
    RULE_1_SYMMETRIC = "regla_1_simetrico"
    RULE_2_TOP_ONLY = "regla_2_solo_arriba"
    RULE_3_BOTTOM_ONLY = "regla_3_solo_abajo"
    RULE_4_NO_MARGIN = "regla_4_sin_margen"


@dataclass
class PersonDetection:
    """Detección con métricas de proporción"""
    bbox: Tuple[float, float, float, float]
    confidence: float
    coverage_ratio: float
    person_aspect_ratio: float  # Proporción natural de la persona
    keypoints: Optional[np.ndarray] = None
    head_bbox: Optional[Tuple[float, float, float, float]] = None
    body_center: Optional[Tuple[float, float]] = None


@dataclass
class LayoutAnalysis:
    """Análisis con consideraciones de distorsión"""
    top_space: float
    bottom_space: float
    left_space: float
    right_space: float
    can_add_top_margin: bool
    can_add_bottom_margin: bool
    recommended_rule: CropRule
    margin_percentage: float  # Margen como % de altura de persona
    safe_crop_region: Tuple[int, int, int, int]  # Región sin distorsión


@dataclass
class CropStrategy:
    """Estrategia matemática para evitar distorsión"""
    method: str  # "perfect_crop", "minimal_resize", "hybrid"
    crop_coords: Tuple[int, int, int, int]
    final_resize_needed: bool
    resize_factor: float
    distortion_estimate: float
    explanation: str


class OrientationCorrector:
    """Corrección de orientación sin cambios"""

    async def process(self, image: Image.Image) -> Image.Image:
        """Corregir orientación basada en EXIF"""
        try:
            exif = image.getexif()
            if exif is None:
                return image

            orientation_key = next(
                (k for k, v in ExifTags.TAGS.items() if v == 'Orientation'), None
            )

            if orientation_key and orientation_key in exif:
                orientation = exif[orientation_key]
                rotation_map = {3: 180, 6: 270, 8: 90}

                if orientation in rotation_map:
                    image = image.rotate(
                        rotation_map[orientation],
                        expand=True,
                        resample=Image.Resampling.LANCZOS
                    )
                    print(f"✅ Rotación: {rotation_map[orientation]}°")

        except Exception as e:
            print(f"⚠️  Error en orientación: {e}")

        return image


class EnhancedPersonDetector:
    """Detector con análisis de proporciones"""

    def __init__(self, model_path: str, confidence_threshold: float = 0.7):
        print(f"🤖 Cargando modelo: {model_path}")

        try:
            self.model = YOLO(str(resource_path(model_path)))
            self.has_pose = True
            print(f"✅ Modelo YOLO11 Pose cargado!")
        except Exception as e:
            print(f"⚠️  Fallback a YOLO11 regular: {e}")
            self.model = YOLO(str(resource_path("yolo11n.pt")))
            self.has_pose = False
            print(f"✅ Modelo YOLO11 regular cargado!")

        self.confidence_threshold = confidence_threshold
        self.person_class_id = 0

    async def detect_person(self, image: Image.Image) -> Optional[PersonDetection]:
        """Detección con análisis de proporciones naturales"""

        print(f"🔍 Detectando persona con análisis de proporciones...")

        img_array = np.array(image)
        results = self.model(img_array, verbose=False)

        if not results or len(results) == 0:
            print("❌ Sin resultados")
            return None

        best_detection = None
        best_score = 0
        keypoints = None

        for result in results:
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    if (box.cls == self.person_class_id and
                            box.conf >= self.confidence_threshold and
                            box.conf > best_score):

                        best_detection = box
                        best_score = float(box.conf)

                        if hasattr(result, 'keypoints') and result.keypoints is not None:
                            try:
                                keypoints = result.keypoints.xy[i].cpu().numpy()
                            except:
                                keypoints = None

        if best_detection is None:
            print(f"❌ Ninguna persona detectada")
            return None

        # Procesar detección
        bbox = best_detection.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = bbox

        # Calcular métricas
        img_width, img_height = image.size
        detection_area = (x2 - x1) * (y2 - y1)
        image_area = img_width * img_height
        coverage_ratio = detection_area / image_area

        # CRÍTICO: Calcular proporción natural de la persona
        person_width = x2 - x1
        person_height = y2 - y1
        person_aspect_ratio = person_width / person_height

        # Analizar pose
        head_bbox = None
        body_center = None

        if keypoints is not None and len(keypoints) > 0:
            head_bbox, body_center = self._analyze_pose(keypoints)

        detection = PersonDetection(
            bbox=(x1, y1, x2, y2),
            confidence=float(best_detection.conf),
            coverage_ratio=coverage_ratio,
            person_aspect_ratio=person_aspect_ratio,
            keypoints=keypoints,
            head_bbox=head_bbox,
            body_center=body_center
        )

        print(f"✅ Persona detectada!")
        print(f"   Confianza: {detection.confidence:.2f}")
        print(f"   Cobertura: {coverage_ratio:.2f}")
        print(f"   Proporción natural: {person_aspect_ratio:.3f}")
        print(f"   Dimensiones: {person_width:.0f}x{person_height:.0f}px")

        return detection

    @staticmethod
    def _analyze_pose(keypoints: np.ndarray) -> Tuple[
        Optional[Tuple[float, float, float, float]], Optional[Tuple[float, float]]]:
        """Análisis de pose para mejor composición"""

        # Puntos clave COCO
        NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR = 0, 1, 2, 3, 4
        LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
        LEFT_HIP, RIGHT_HIP = 11, 12

        head_bbox = None
        body_center = None

        try:
            # Calcular cabeza
            head_points = []
            for idx in [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR]:
                if idx < len(keypoints) and keypoints[idx][0] > 0 and keypoints[idx][1] > 0:
                    head_points.append(keypoints[idx])

            if len(head_points) >= 2:
                head_points = np.array(head_points)
                x_min, y_min = head_points.min(axis=0)
                x_max, y_max = head_points.max(axis=0)

                padding = max((x_max - x_min), (y_max - y_min)) * 0.3
                head_bbox = (
                    max(0, x_min - padding),
                    max(0, y_min - padding),
                    x_max + padding,
                    y_max + padding
                )

            # Calcular centro de cuerpo
            body_points = []
            for idx in [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]:
                if idx < len(keypoints) and keypoints[idx][0] > 0 and keypoints[idx][1] > 0:
                    body_points.append(keypoints[idx])

            if len(body_points) >= 2:
                body_points = np.array(body_points)
                body_center = tuple(body_points.mean(axis=0))

        except Exception as e:
            print(f"⚠️  Error en pose: {e}")

        return head_bbox, body_center


class AntiDistortionLayoutAnalyzer:
    """Analizador que considera proporciones naturales"""

    @staticmethod
    def analyze_layout(
            image_size: Tuple[int, int],
            detection: PersonDetection,
            settings: Settings
    ) -> LayoutAnalysis:
        """Análisis considerando distorsión y proporciones naturales"""

        img_w, img_h = image_size
        x1, y1, x2, y2 = detection.bbox

        # Espacios disponibles
        top_space = y1
        bottom_space = img_h - y2
        left_space = x1
        right_space = img_w - x2

        # Calcular margen como porcentaje de altura de persona
        person_height = y2 - y1
        margin_percentage = settings.margin_pixels / person_height

        print(f"📐 Análisis Anti-Distorsión:")
        print(f"   Imagen: {img_w}x{img_h}")
        print(f"   Persona: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
        print(f"   Proporción persona: {detection.person_aspect_ratio:.3f}")
        print(f"   Altura persona: {person_height:.0f}px")
        print(f"   Margen como %: {margin_percentage:.1%}")
        print(f"   Espacios: ↑{top_space:.0f} ↓{bottom_space:.0f} ←{left_space:.0f} →{right_space:.0f}")

        # Usar márgenes proporcionales si está habilitado
        if settings.use_proportional_margins:
            effective_margin = max(settings.margin_pixels, person_height * 0.02)  # Mínimo 2% de altura
            print(f"   Margen efectivo: {effective_margin:.0f}px (proporcional)")
        else:
            effective_margin = settings.margin_pixels
            print(f"   Margen efectivo: {effective_margin:.0f}px (fijo)")

        # Verificar capacidad de márgenes
        can_add_top_margin = top_space >= effective_margin
        can_add_bottom_margin = bottom_space >= effective_margin

        # 🔧 MODIFICACIÓN: Determinar regla - Si es detección por defecto, forzar Regla 4
        if detection.confidence == 0.0:  # Detección por defecto (marcador especial)
            recommended_rule = CropRule.RULE_4_NO_MARGIN
            print(f"   → FORZANDO REGLA 4: Sin persona detectada - crop centrado automático")
        elif can_add_top_margin and can_add_bottom_margin:
            recommended_rule = CropRule.RULE_1_SYMMETRIC
            print(f"   → REGLA 1: Márgenes simétricos (25px arriba + 25px abajo)")
        elif can_add_top_margin and not can_add_bottom_margin:
            recommended_rule = CropRule.RULE_2_TOP_ONLY
            print(f"   → REGLA 2: Solo margen arriba (25px arriba)")
        elif not can_add_top_margin and can_add_bottom_margin:
            recommended_rule = CropRule.RULE_3_BOTTOM_ONLY
            print(f"   → REGLA 3: Solo margen abajo (25px abajo)")
        else:
            recommended_rule = CropRule.RULE_4_NO_MARGIN
            print(f"   → REGLA 4: Sin margen (recorte centrado)")

        # Calcular región segura para crop sin distorsión
        safe_crop_region = AntiDistortionLayoutAnalyzer._calculate_safe_crop_region(
            img_w, img_h, detection, settings.target_ratio
        )

        print(f"   Región segura: {safe_crop_region}")

        return LayoutAnalysis(
            top_space=top_space,
            bottom_space=bottom_space,
            left_space=left_space,
            right_space=right_space,
            can_add_top_margin=can_add_top_margin,
            can_add_bottom_margin=can_add_bottom_margin,
            recommended_rule=recommended_rule,
            margin_percentage=margin_percentage,
            safe_crop_region=safe_crop_region
        )

    @staticmethod
    def _calculate_safe_crop_region(
            img_w: int,
            img_h: int,
            detection: PersonDetection,
            target_ratio: float
    ) -> Tuple[int, int, int, int]:
        """Calcular región donde se puede hacer crop sin distorsión"""

        x1, y1, x2, y2 = detection.bbox
        person_center_x = (x1 + x2) / 2
        person_center_y = (y1 + y2) / 2

        # Calcular el crop máximo posible centrado en la persona
        # que respete la proporción objetivo

        img_ratio = img_w / img_h

        if img_ratio > target_ratio:
            # Imagen más ancha - limitar ancho
            max_width = img_h * target_ratio
            safe_x1 = max(0, person_center_x - max_width / 2)
            safe_x2 = min(img_w, safe_x1 + max_width)

            # Ajustar si se sale de límites
            if safe_x2 > img_w:
                safe_x2 = img_w
                safe_x1 = max(0, safe_x2 - max_width)

            return (int(safe_x1), 0, int(safe_x2), img_h)

        else:
            # Imagen más alta - limitar altura
            max_height = img_w / target_ratio
            safe_y1 = max(0, person_center_y - max_height / 2)
            safe_y2 = min(img_h, safe_y1 + max_height)

            # Ajustar si se sale de límites
            if safe_y2 > img_h:
                safe_y2 = img_h
                safe_y1 = max(0, safe_y2 - max_height)

            return (0, int(safe_y1), img_w, int(safe_y2))


class DistortionFreeCropper:
    """Cropper que elimina completamente la distorsión"""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def apply_crop_rule(
            self,
            image: Image.Image,
            detection: PersonDetection,
            layout: LayoutAnalysis
    ) -> Image.Image:
        """Aplicar crop sin distorsión"""

        print(f"✂️  Aplicando {layout.recommended_rule.value} SIN DISTORSIÓN...")

        # Calcular estrategia óptima
        strategy = self._calculate_optimal_strategy(image, detection, layout)

        print(f"   Estrategia: {strategy.method}")
        print(f"   Distorsión estimada: {strategy.distortion_estimate:.1%}")
        print(f"   {strategy.explanation}")

        # Aplicar crop según estrategia
        if strategy.method == "perfect_crop":
            return await self._apply_perfect_crop(image, strategy)
        elif strategy.method == "minimal_resize":
            return await self._apply_minimal_resize(image, strategy)
        else:  # hybrid
            return await self._apply_hybrid_method(image, strategy)

    def _calculate_optimal_strategy(
            self,
            image: Image.Image,
            detection: PersonDetection,
            layout: LayoutAnalysis
    ) -> CropStrategy:
        """Calcular la mejor estrategia sin distorsión"""

        # Intentar crop perfecto primero
        perfect_crop = self._try_perfect_crop(image, detection, layout)

        if perfect_crop and perfect_crop.distortion_estimate <= self.settings.max_distortion_tolerance:
            return perfect_crop

        # Si no es posible, usar resize mínimo
        minimal_resize = self._try_minimal_resize(image, detection, layout)

        if minimal_resize.distortion_estimate <= self.settings.max_distortion_tolerance:
            return minimal_resize

        # Último recurso: método híbrido
        return self._try_hybrid_method(image, detection, layout)

    def _try_perfect_crop(
            self,
            image: Image.Image,
            detection: PersonDetection,
            layout: LayoutAnalysis
    ) -> Optional[CropStrategy]:
        """Intentar crop que logre proporción exacta sin resize"""

        x1, y1, x2, y2 = detection.bbox

        # Aplicar regla específica para calcular crop
        if layout.recommended_rule == CropRule.RULE_1_SYMMETRIC:
            crop_coords = self._calculate_symmetric_crop(image, detection, layout)
        elif layout.recommended_rule == CropRule.RULE_2_TOP_ONLY:
            crop_coords = self._calculate_top_only_crop(image, detection, layout)
        elif layout.recommended_rule == CropRule.RULE_3_BOTTOM_ONLY:
            crop_coords = self._calculate_bottom_only_crop(image, detection, layout)
        else:
            crop_coords = layout.safe_crop_region

        # Verificar si el crop resulta en proporción exacta
        crop_w = crop_coords[2] - crop_coords[0]
        crop_h = crop_coords[3] - crop_coords[1]
        crop_ratio = crop_w / crop_h

        ratio_error = abs(crop_ratio - self.settings.target_ratio) / self.settings.target_ratio

        if ratio_error <= 0.01:  # Error menor al 1%
            return CropStrategy(
                method="perfect_crop",
                crop_coords=crop_coords,
                final_resize_needed=False,
                resize_factor=1.0,
                distortion_estimate=0.0,
                explanation=f"Crop perfecto logra proporción {crop_ratio:.3f} vs objetivo {self.settings.target_ratio:.3f}"
            )

        return None

    def _try_minimal_resize(
            self,
            image: Image.Image,
            detection: PersonDetection,
            layout: LayoutAnalysis
    ) -> CropStrategy:
        """Crop + resize mínimo necesario"""

        # Usar la región más cercana a la proporción objetivo
        crop_coords = layout.safe_crop_region
        crop_w = crop_coords[2] - crop_coords[0]
        crop_h = crop_coords[3] - crop_coords[1]
        crop_ratio = crop_w / crop_h

        # Calcular factor de resize mínimo
        if crop_ratio > self.settings.target_ratio:
            # Crop más ancho - necesita stretch vertical
            resize_factor = crop_ratio / self.settings.target_ratio
        else:
            # Crop más alto - necesita stretch horizontal
            resize_factor = self.settings.target_ratio / crop_ratio

        # Estimar distorsión
        distortion = abs(resize_factor - 1.0)

        return CropStrategy(
            method="minimal_resize",
            crop_coords=crop_coords,
            final_resize_needed=True,
            resize_factor=resize_factor,
            distortion_estimate=distortion,
            explanation=f"Crop + resize mínimo (factor: {resize_factor:.3f})"
        )

    def _try_hybrid_method(
            self,
            image: Image.Image,
            detection: PersonDetection,
            layout: LayoutAnalysis
    ) -> CropStrategy:
        """Método híbrido que minimiza distorsión"""

        # Usar padding/letterboxing para evitar distorsión
        crop_coords = layout.safe_crop_region

        return CropStrategy(
            method="hybrid",
            crop_coords=crop_coords,
            final_resize_needed=True,
            resize_factor=1.0,
            distortion_estimate=0.0,
            explanation="Método híbrido con letterboxing - sin distorsión"
        )

    def _calculate_symmetric_crop(
            self,
            image: Image.Image,
            detection: PersonDetection,
            layout: LayoutAnalysis
    ) -> Tuple[int, int, int, int]:
        """Calcular crop simétrico que respete proporciones"""

        x1, y1, x2, y2 = detection.bbox
        margin = self.settings.margin_pixels

        # Agregar márgenes
        crop_y1 = max(0, y1 - margin)
        crop_y2 = min(image.height, y2 + margin)
        crop_height = crop_y2 - crop_y1

        # Calcular ancho necesario para proporción exacta
        needed_width = crop_height * self.settings.target_ratio

        # Centrar en persona
        person_center_x = (x1 + x2) / 2
        crop_x1 = max(0, person_center_x - needed_width / 2)
        crop_x2 = min(image.width, crop_x1 + needed_width)

        # Ajustar si no cabe
        if crop_x2 > image.width:
            crop_x2 = image.width
            crop_x1 = max(0, crop_x2 - needed_width)

        return (int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2))

    def _calculate_top_only_crop(
            self,
            image: Image.Image,
            detection: PersonDetection,
            layout: LayoutAnalysis
    ) -> Tuple[int, int, int, int]:
        """Calcular crop solo arriba que respete proporciones"""

        x1, y1, x2, y2 = detection.bbox
        margin = self.settings.margin_pixels

        # Margen solo arriba
        crop_y1 = max(0, y1 - margin)
        crop_y2 = y2
        crop_height = crop_y2 - crop_y1

        # Calcular ancho necesario
        needed_width = crop_height * self.settings.target_ratio

        # Centrar en persona
        person_center_x = (x1 + x2) / 2
        crop_x1 = max(0, person_center_x - needed_width / 2)
        crop_x2 = min(image.width, crop_x1 + needed_width)

        # Si no cabe el ancho, extender altura hacia abajo
        if crop_x2 > image.width or needed_width > image.width:
            needed_width = image.width * 0.95
            needed_height = needed_width / self.settings.target_ratio
            crop_y2 = min(image.height, crop_y1 + needed_height)
            crop_x1 = (image.width - needed_width) / 2
            crop_x2 = crop_x1 + needed_width

        return (int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2))

    def _calculate_bottom_only_crop(
            self,
            image: Image.Image,
            detection: PersonDetection,
            layout: LayoutAnalysis
    ) -> Tuple[int, int, int, int]:
        """Calcular crop solo abajo que respete proporciones"""

        x1, y1, x2, y2 = detection.bbox
        margin = self.settings.margin_pixels

        # Margen solo abajo
        crop_y1 = y1
        crop_y2 = min(image.height, y2 + margin)
        crop_height = crop_y2 - crop_y1

        # Calcular ancho necesario
        needed_width = crop_height * self.settings.target_ratio

        # Centrar en persona
        person_center_x = (x1 + x2) / 2
        crop_x1 = max(0, person_center_x - needed_width / 2)
        crop_x2 = min(image.width, crop_x1 + needed_width)

        # Si no cabe el ancho, extender altura hacia arriba
        if crop_x2 > image.width or needed_width > image.width:
            needed_width = image.width * 0.95
            needed_height = needed_width / self.settings.target_ratio
            crop_y1 = max(0, crop_y2 - needed_height)
            crop_x1 = (image.width - needed_width) / 2
            crop_x2 = crop_x1 + needed_width

        return (int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2))

    async def _apply_perfect_crop(self, image: Image.Image, strategy: CropStrategy) -> Image.Image:
        """Aplicar crop perfecto - no requiere resize"""

        print(f"   🎯 Aplicando crop perfecto...")

        cropped = image.crop(strategy.crop_coords)

        # Solo resize para dimensiones exactas si las proporciones son correctas
        final_image = cropped.resize(
            (self.settings.target_width, self.settings.target_height),
            Image.Resampling.LANCZOS
        )

        return final_image

    async def _apply_minimal_resize(self, image: Image.Image, strategy: CropStrategy) -> Image.Image:
        """Aplicar crop + resize mínimo"""

        print(f"   📏 Aplicando resize mínimo...")

        cropped = image.crop(strategy.crop_coords)

        # Resize proporcional primero
        crop_w, crop_h = cropped.size
        scale_factor = min(
            self.settings.target_width / crop_w,
            self.settings.target_height / crop_h
        )

        intermediate_w = int(crop_w * scale_factor)
        intermediate_h = int(crop_h * scale_factor)

        scaled = cropped.resize((intermediate_w, intermediate_h), Image.Resampling.LANCZOS)

        # Crop final para dimensiones exactas
        if intermediate_w > self.settings.target_width:
            # Crop horizontal
            x_offset = (intermediate_w - self.settings.target_width) // 2
            final_image = scaled.crop((
                x_offset, 0,
                x_offset + self.settings.target_width, intermediate_h
            ))
        elif intermediate_h > self.settings.target_height:
            # Crop vertical
            y_offset = (intermediate_h - self.settings.target_height) // 2
            final_image = scaled.crop((
                0, y_offset,
                intermediate_w, y_offset + self.settings.target_height
            ))
        else:
            # Letterboxing si es necesario
            final_image = self._apply_letterboxing(scaled)

        return final_image

    async def _apply_hybrid_method(self, image: Image.Image, strategy: CropStrategy) -> Image.Image:
        """Aplicar método híbrido con letterboxing"""

        print(f"   🎨 Aplicando método híbrido...")

        cropped = image.crop(strategy.crop_coords)

        # Resize proporcional para que encaje
        crop_w, crop_h = cropped.size
        scale_factor = min(
            self.settings.target_width / crop_w,
            self.settings.target_height / crop_h
        )

        new_w = int(crop_w * scale_factor)
        new_h = int(crop_h * scale_factor)

        scaled = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Letterboxing para completar dimensiones
        final_image = self._apply_letterboxing(scaled)

        return final_image

    def _apply_letterboxing(self, image: Image.Image) -> Image.Image:
        """Aplicar letterboxing para completar dimensiones sin distorsión"""

        # Crear imagen de fondo del color promedio de los bordes
        img_array = np.array(image)

        # Calcular color de fondo inteligente
        border_pixels = np.concatenate([
            img_array[0, :].reshape(-1, 3),  # Top
            img_array[-1, :].reshape(-1, 3),  # Bottom
            img_array[:, 0].reshape(-1, 3),  # Left
            img_array[:, -1].reshape(-1, 3)  # Right
        ])

        bg_color = tuple(border_pixels.mean(axis=0).astype(int))

        # Crear imagen de fondo
        background = Image.new('RGB', (self.settings.target_width, self.settings.target_height), bg_color)

        # Centrar imagen original
        img_w, img_h = image.size
        x_offset = (self.settings.target_width - img_w) // 2
        y_offset = (self.settings.target_height - img_h) // 2

        background.paste(image, (x_offset, y_offset))

        print(f"   Letterboxing aplicado con color de fondo: {bg_color}")

        return background


class AntiDistortionProcessor:
    """Procesador maestro sin distorsión"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.orientation_corrector = OrientationCorrector()
        self.person_detector = EnhancedPersonDetector(
            settings.model_path, settings.min_confidence
        )
        self.layout_analyzer = AntiDistortionLayoutAnalyzer()
        self.distortion_free_cropper = DistortionFreeCropper(settings)

    def _create_default_detection(self, image: Image.Image) -> PersonDetection:
        """🔧 NUEVO: Crear detección por defecto cuando no se encuentra persona"""

        img_w, img_h = image.size

        # Crear una "detección" centrada que cubra una porción razonable de la imagen
        # Esto asegura que la Regla 4 se aplique correctamente
        coverage_ratio = 0.7  # 70% de la imagen

        # Calcular dimensiones de la "detección" centrada
        detection_w = img_w * coverage_ratio
        detection_h = img_h * coverage_ratio

        # Centrar en la imagen
        center_x = img_w / 2
        center_y = img_h / 2

        x1 = center_x - detection_w / 2
        y1 = center_y - detection_h / 2
        x2 = center_x + detection_w / 2
        y2 = center_y + detection_h / 2

        # Calcular proporción "natural" basada en dimensiones de imagen
        person_aspect_ratio = detection_w / detection_h

        return PersonDetection(
            bbox=(x1, y1, x2, y2),
            confidence=0.0,  # Marcador especial para detección por defecto
            coverage_ratio=coverage_ratio,
            person_aspect_ratio=person_aspect_ratio,
            keypoints=None,
            head_bbox=None,
            body_center=(center_x, center_y)
        )

    async def process_single_image(self, image: Image.Image) -> Optional[Image.Image]:
        """Procesamiento sin distorsión"""

        try:
            print(f"\n🚀 PIPELINE SIN DISTORSIÓN")
            print(f"📸 Entrada: {image.size} {image.mode}")
            print(f"🎯 Objetivo: {self.settings.target_width}x{self.settings.target_height}")
            print(f"🚫 Máxima distorsión permitida: {self.settings.max_distortion_tolerance:.1%}")
            print("=" * 60)

            # Paso 1: Corrección de orientación
            print(f"1️⃣  Corrección de orientación...")
            corrected_image = await self.orientation_corrector.process(image)

            # Paso 2: Detección con análisis de proporciones
            print(f"2️⃣  Detección con análisis de proporciones...")
            detection = await self.person_detector.detect_person(corrected_image)

            # 🔧 MODIFICACIÓN PRINCIPAL: Si no hay detección, crear una por defecto
            if not detection:
                print("⚠️  Ninguna persona detectada - Aplicando Regla 4 por defecto")
                print("   🎯 Crop centrado automático sin márgenes")
                detection = self._create_default_detection(corrected_image)
                print(f"   ✅ Detección por defecto creada: centro de imagen")

            # Validar altura mínima de persona (solo si es detección real)
            if detection.confidence > 0.0:  # Solo validar si es detección real
                person_height = detection.bbox[3] - detection.bbox[1]
                min_height = corrected_image.height * self.settings.min_person_height_ratio

                if person_height < min_height:
                    print(f"⚠️  Persona muy pequeña: {person_height:.0f}px < {min_height:.0f}px")
                    print(f"   Continuando con procesamiento...")
            else:
                print(f"   🤖 Usando detección automática - aplicará Regla 4")

            # Paso 3: Análisis anti-distorsión
            print(f"3️⃣  Análisis anti-distorsión...")
            layout = self.layout_analyzer.analyze_layout(
                corrected_image.size, detection, self.settings
            )

            # Paso 4: Crop sin distorsión
            print(f"4️⃣  Crop sin distorsión...")
            processed_image = await self.distortion_free_cropper.apply_crop_rule(
                corrected_image, detection, layout
            )

            # Paso 5: Validación final
            print(f"5️⃣  Validación final...")
            if processed_image.mode != 'RGB':
                processed_image = processed_image.convert('RGB')

            print(f"\n✅ PROCESAMIENTO SIN DISTORSIÓN COMPLETADO!")
            print(f"📊 Dimensiones finales: {processed_image.size}")
            print(f"🎯 Proporción final: {processed_image.width / processed_image.height:.3f}")
            print(f"🎯 Proporción objetivo: {self.settings.target_ratio:.3f}")
            print(f"💎 Sin distorsión de personas")
            print(f"🚫 Proporciones naturales preservadas")

            return processed_image

        except Exception as e:
            print(f"❌ Error en procesamiento: {e}")
            import traceback
            traceback.print_exc()
            return None

async def process_local_image(input_path: str, output_dir: str = "output_sin_distorsion"):
    """Procesar imagen sin distorsión (async version)""" # Note: no asyncio.run() here
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    if not Path(input_path).exists():
        print(f"❌ Archivo no encontrado: {input_path}")
        return

    print(f"📁 Entrada: {input_path}")
    print(f"📁 Salida: {output_dir}")

    settings = Settings()
    processor = AntiDistortionProcessor(settings)

    try:
        with Image.open(input_path) as img:
            print(f"📸 Imagen cargada: {img.size} {img.mode}")

            result = await processor.process_single_image(img)

            if result:
                input_file = Path(input_path)
                # Ensure the suffix is correctly handled, e.g., for .jpg vs .jpeg
                suffix = input_file.suffix
                if suffix.lower() == '.jpeg':
                    suffix = '.jpg' # Or handle consistency as desired

                output_file = output_path / f"{input_file.stem}{suffix}"

                result.save(
                    output_file,
                    format='JPEG', # Forcing JPEG as per your original code
                    quality=95,
                    optimize=True,
                    dpi=settings.target_dpi
                )

                print(f"\n🎉 ÉXITO TOTAL!")
                print(f"📄 Guardado: {output_file}")
                print(f"📊 Dimensiones exactas: {result.size}")
                print(f"🚫 SIN DISTORSIÓN garantizada")
                print(f"💎 Proporciones naturales preservadas")

            else:
                print("❌ Procesamiento falló")
    except Exception as e:
        print(f"An error occurred during process_local_image: {e}")


async def process_images_in_folder(entrada_folder, salida_folder):
    if not os.path.exists(salida_folder):
        os.makedirs(salida_folder)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(entrada_folder) if os.path.splitext(f)[1].lower() in image_extensions]

    tasks = []

    for image_file in image_files:

        # Generamos el path completo (considerando image file e input path)
        image_input_folder = os.path.join(entrada_folder, image_file)

        # Mandamos a llamar la nueva lógica del modelo
        tasks.append(process_local_image(input_path=image_input_folder, output_dir=salida_folder))

    await asyncio.gather(*tasks)