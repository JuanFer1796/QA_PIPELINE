import cv2
import numpy as np
from skimage import filters
from PIL import Image

class RetinaProcessor:
    def __init__(self, target_size=(800, 800), method='default'):
        """
        Initialize the RetinaProcessor.
        
        Args:
            target_size (tuple): Target size for output image (default: (800, 800))
            method (str): Preprocessing method to use ('default' or 'mdao')
        """
        self.target_size = target_size
        self.method = method
        
    def process_image(self, img):
        """
        Process image using the selected method.
        
        Args:
            img: Input image in BGR format
            
        Returns:
            tuple: (processed_image, mask, metrics) based on selected method
        """
        if self.method == 'mdao':
            return self._process_mdao(img)
        else:
            return self._process_default(img)
            
    def _process_default(self, img):
        """Original RetinaProcessor method"""
        # Obtener máscara y métricas iniciales
        mask, bbox, center, radius = self.get_mask(img)
        initial_metrics = self.compute_image_metrics(img, mask)

        # Aplicar máscara
        r_img = self.mask_image(img.copy(), mask)

        # Recortar área de interés
        r_img, r_border = self.remove_back_area(r_img, bbox=bbox)
        mask, _ = self.remove_back_area(mask, border=r_border)

        # Rellenar con negro y redimensionar
        r_img, sup_border = self.supplemental_black_area(r_img)
        mask, _ = self.supplemental_black_area(mask, border=sup_border)

        # Mejorar canal verde
        enhanced_img = self.enhance_green_channel(r_img)

        # Calcular métricas finales
        final_metrics = self.compute_image_metrics(enhanced_img, mask)

        # Redimensionar al tamaño objetivo
        if enhanced_img.shape[:2] != self.target_size:
            enhanced_img = cv2.resize(enhanced_img, self.target_size)
            mask = cv2.resize(mask, self.target_size)

        return enhanced_img, mask, {
            'initial_metrics': initial_metrics,
            'final_metrics': final_metrics,
            'center': center,
            'radius': radius,
        }
    
    def _process_mdao(self, img, scale=600):
        """MDAO preprocessing method"""
        if img is None:
            return None, None, {}
            
        try:
            # Escalar la imagen
            img = self._scale_radius_mdao(img, scale)
            
            # Métricas iniciales (usando toda la imagen como máscara temporal)
            temp_mask = np.ones(img.shape[:2], dtype=np.uint8)
            initial_metrics = self.compute_image_metrics(img, temp_mask)
            
            # Preprocesar la imagen
            processed = self._preprocess_image_mdao(img, scale)
            
            # Crear máscara circular
            mask = np.zeros(processed.shape[:2], dtype=np.uint8)
            center = (processed.shape[1] // 2, processed.shape[0] // 2)
            radius = int(scale * 0.9)
            cv2.circle(mask, center, radius, 1, -1, 8, 0)
            
            # Aplicar máscara
            processed = processed * mask[..., np.newaxis] + 128 * (1 - mask[..., np.newaxis])
            
            # Métricas finales
            final_metrics = self.compute_image_metrics(processed, mask)
            
            # Redimensionar si es necesario
            if processed.shape[:2] != self.target_size:
                processed = cv2.resize(processed, self.target_size)
                mask = cv2.resize(mask, self.target_size)
            
            return processed, mask, {
                'initial_metrics': initial_metrics,
                'final_metrics': final_metrics,
                'center': center,
                'radius': radius,
            }
            
        except Exception as e:
            print(f"Error in MDAO preprocessing: {str(e)}")
            return None, None, {}
    
    def _scale_radius_mdao(self, img, scale):
        """Scale image based on radius detection (MDAO method)"""
        x = img[img.shape[0] // 2, :, :].sum(1)
        r = (x > x.mean() / 10).sum() / 2
        s = scale * 1.0 / r
        return cv2.resize(img, (0, 0), fx=s, fy=s)
    
    def _preprocess_image_mdao(self, img, scale):
        """MDAO specific preprocessing steps"""
        # Seleccionar el canal verde
        img_green = img[:, :, 1]
        
        # Aplicar CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_green.astype(np.uint8))
        
        # Restar el color promedio local
        img_clahe = cv2.addWeighted(img_clahe, 4, 
                                   cv2.GaussianBlur(img_clahe, (0, 0), scale / 30), 
                                   -4, 128)
        
        # Convertir a RGB
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        
        return img_rgb

    # === Métodos originales de RetinaProcessor ===
    def get_mask_BZ(self, img):
        if img.ndim == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img
        threhold = np.mean(gray_img) / 3 - 5
        _, mask = cv2.threshold(gray_img, max(0, threhold), 1, cv2.THRESH_BINARY)
        nn_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
        new_mask = (1 - mask).astype(np.uint8)
        _, new_mask, _, _ = cv2.floodFill(new_mask, nn_mask, (0, 0), (0), cv2.FLOODFILL_MASK_ONLY)
        _, new_mask, _, _ = cv2.floodFill(new_mask, nn_mask, (new_mask.shape[1] - 1, new_mask.shape[0] - 1), (0),
                                          cv2.FLOODFILL_MASK_ONLY)
        mask = mask + new_mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        mask = cv2.erode(mask, kernel)
        mask = cv2.dilate(mask, kernel)
        return mask

    def _get_center_by_edge(self, mask):
        center = [0, 0]
        x = mask.sum(axis=1)
        center[0] = np.where(x > x.max() * 0.95)[0].mean()
        x = mask.sum(axis=0)
        center[1] = np.where(x > x.max() * 0.95)[0].mean()
        return center

    def _get_radius_by_mask_center(self, mask, center):
        mask = mask.astype(np.uint8)
        ksize = max(mask.shape[1] // 400 * 2 + 1, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        index = np.where(mask > 0)
        d_int = np.sqrt((index[0] - center[0]) ** 2 + (index[1] - center[1]) ** 2)
        b_count = np.bincount(np.ceil(d_int).astype(np.int32))
        radius = np.where(b_count > b_count.max() * 0.995)[0].max()
        return radius

    def get_mask(self, img):
        if img.ndim == 3:
            g_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.ndim == 2:
            g_img = img.copy()
        else:
            raise Exception('image dim is not 1 or 3')
        h, w = g_img.shape
        shape = g_img.shape[0:2]
        g_img = cv2.resize(g_img, (0, 0), fx=0.5, fy=0.5)
        tg_img = cv2.normalize(g_img, None, 0, 255, cv2.NORM_MINMAX)
        tmp_mask = self.get_mask_BZ(tg_img)
        center = self._get_center_by_edge(tmp_mask)
        radius = self._get_radius_by_mask_center(tmp_mask, center)
        center = [center[0] * 2, center[1] * 2]
        radius = int(radius * 2)
        s_h = max(0, int(center[0] - radius))
        s_w = max(0, int(center[1] - radius))
        bbox = (s_h, s_w, min(h - s_h, 2 * radius), min(w - s_w, 2 * radius))
        tmp_mask = self._get_circle_by_center_bbox(shape, center, bbox, radius)
        return tmp_mask, bbox, center, radius

    def _get_circle_by_center_bbox(self, shape, center, bbox, radius):
        center_mask = np.zeros(shape=shape).astype('uint8')
        center_tmp = (int(center[0]), int(center[1]))
        center_mask = cv2.circle(center_mask, center_tmp[::-1], int(radius), (1), -1)
        return center_mask

    def mask_image(self, img, mask):
        img[mask <= 0, ...] = 0
        return img

    def remove_back_area(self, img, bbox=None, border=None):
        image = img
        if border is None:
            border = np.array((bbox[0], bbox[0] + bbox[2], bbox[1], bbox[1] + bbox[3], img.shape[0], img.shape[1]),
                              dtype=np.int32)
        image = image[border[0]:border[1], border[2]:border[3], ...]
        return image, border

    def supplemental_black_area(self, img, border=None):
        image = img
        if border is None:
            h, v = img.shape[0:2]
            max_l = max(h, v)
            if image.ndim > 2:
                image = np.zeros(shape=[max_l, max_l, img.shape[2]], dtype=img.dtype)
            else:
                image = np.zeros(shape=[max_l, max_l], dtype=img.dtype)
            border = (
            int(max_l / 2 - h / 2), int(max_l / 2 - h / 2) + h, int(max_l / 2 - v / 2), int(max_l / 2 - v / 2) + v,
            max_l)
        else:
            max_l = border[4]
            if image.ndim > 2:
                image = np.zeros(shape=[max_l, max_l, img.shape[2]], dtype=img.dtype)
            else:
                image = np.zeros(shape=[max_l, max_l], dtype=img.dtype)
        image[border[0]:border[1], border[2]:border[3], ...] = img
        return image, border

    def compute_image_metrics(self, image, mask):
        """Calcula métricas de calidad de la imagen"""
        metrics = {}

        # Solo procesar área dentro de la máscara
        valid_area = mask > 0

        if image.ndim == 3:
            # Métricas por canal
            for i, channel_name in enumerate(['R', 'G', 'B']):
                channel = image[:, :, i][valid_area]
                metrics[f'{channel_name}_mean'] = np.mean(channel)
                metrics[f'{channel_name}_std'] = np.std(channel)

                # Contraste local del canal
                local_std = cv2.Laplacian(image[:, :, i], cv2.CV_64F).var()
                metrics[f'{channel_name}_contrast'] = local_std

            # Métricas específicas del canal verde
            green_channel = image[:, :, 1][valid_area]
            metrics['green_channel_entropy'] = entropy = filters.rank.entropy(
                image[:, :, 1].astype(np.uint8),
                np.ones((9, 9), np.uint8)
            )[valid_area].mean()

            # Detectar reflejos (áreas muy brillantes)
            highlights = np.sum(image > 250, axis=2)
            metrics['highlight_ratio'] = np.sum((highlights > 0) & valid_area) / np.sum(valid_area)

        return metrics

    def enhance_green_channel(self, image):
        """Mejora el canal verde de la imagen"""
        if image.ndim != 3:
            return image

        # Separar canales
        r, g, b = cv2.split(image)

        # Aplicar CLAHE al canal verde
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g_enhanced = clahe.apply(g)

        # Normalizar el canal verde mejorado
        g_enhanced = cv2.normalize(g_enhanced, None, 0, 255, cv2.NORM_MINMAX)

        # Recombinar canales
        enhanced = cv2.merge([r, g_enhanced, b])
        return enhanced
