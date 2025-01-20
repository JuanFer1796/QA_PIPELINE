import numpy as np
import tensorflow as tf
import cv2
from typing import Dict, Union
import os
class PredictionPipeline:
    def __init__(self, model_path: str, target_size: tuple = (380, 380)):
        """Initialize prediction pipeline"""
        # Inicializar procesadores
        self.processor_default = RetinaProcessor(target_size=target_size, method='default')
        self.processor_mdao = RetinaProcessor(target_size=target_size, method='mdao')
        
        # Inicializar modelo
        self.model = QualityAssessmentModel(
            input_shape=(target_size[0], target_size[1], 3),
            num_classes=3
        )
        self.model.model = tf.keras.models.load_model(model_path)
        self.target_size = target_size
        
        # Mapeo de clases
        self.class_names = {
            0: 'Buena',
            1: 'Usable',
            2: 'Rechazada'
        }

    def read_image(self, image_path: str) -> Union[np.ndarray, None]:
        """Lee una imagen desde un archivo"""
        if not os.path.exists(image_path):
            print(f"Error: No se encontró la imagen en {image_path}")
            return None
            
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: No se pudo leer la imagen {image_path}")
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            print(f"Error al leer la imagen {image_path}: {str(e)}")
            return None

    def predict(self, image_input: Union[str, np.ndarray]) -> Dict:
        """Procesa y predice la calidad de una imagen"""
        # Leer imagen si es necesario
        if isinstance(image_input, str):
            image = self.read_image(image_input)
            if image is None:
                return {'error': 'Failed to read image'}
        else:
            image = image_input.copy()

        # Predicción con preprocesamiento default
        results_default = self.preprocess_and_predict(image)
        
        # Si es clase 0, aplicar solo preprocesamiento MDAO
        if results_default['predicted_class'] == 0:
            print("Detectada clase 0, aplicando preprocesamiento MDAO...")
            mdao_processed_image, mdao_mask, mdao_metrics = self.processor_mdao.process_image(image)
            
            final_results = {
                'default_prediction': results_default,
                'mdao_processed': {
                    'processed_image': mdao_processed_image,
                    'preprocessing_metrics': mdao_metrics
                },
                'method_used': 'mdao',
                'original_image': image,
                'final_prediction': results_default['predicted_class']
            }
        else:
            final_results = {
                'default_prediction': results_default,
                'method_used': 'default',
                'original_image': image,
                'final_prediction': results_default['predicted_class']
            }
            
        return final_results

    def preprocess_and_predict(self, image: np.ndarray) -> Dict:
        """Preprocesa y predice la calidad de la imagen"""
        # Preprocess image
        processed_image, mask, preprocess_metrics = self.processor_default.process_image(image)
        
        # Resize if needed
        if processed_image.shape[:2] != self.target_size:
            processed_image = cv2.resize(processed_image, self.target_size)
        
        # Normalize
        input_image = processed_image.astype('float32')
        input_image = np.expand_dims(input_image, axis=0)
        
        # Predict
        predictions = self.model.model.predict(input_image)
        predicted_class = np.argmax(predictions[0])
        
        return {
            'predicted_class': int(predicted_class),
            'probabilities': predictions[0].tolist(),
            'processed_image': processed_image,
            'preprocessing_metrics': preprocess_metrics,
        }

    def plot_results(self, results: Dict, save_path: str = None):
        """
        Visualiza los resultados del pipeline
        
        Args:
            results: Diccionario con los resultados de la predicción
            save_path: Ruta donde guardar la imagen (opcional)
        """
        if 'error' in results:
            print(f"Error en resultados: {results['error']}")
            return

        # Configurar el plot
        if results['method_used'] == 'mdao':
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            plot_titles = ['Imagen Original', 'Procesamiento Default', 'Procesamiento MDAO']
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            plot_titles = ['Imagen Original', 'Procesamiento Default']
            axes = [axes[0], axes[1]]

        # Imagen original
        axes[0].imshow(results['original_image'])
        axes[0].set_title(plot_titles[0])
        axes[0].axis('off')

        # Resultados del procesamiento default
        default_pred = results['default_prediction']
        default_class = self.class_names[default_pred['predicted_class']]
        default_prob = max(default_pred['probabilities']) * 100
        
        axes[1].imshow(default_pred['processed_image'])
        axes[1].set_title(f'{plot_titles[1]}\n{default_class} ({default_prob:.1f}%)')
        axes[1].axis('off')

        # Si se usó MDAO, mostrar el preprocesamiento
        if results['method_used'] == 'mdao':
            mdao_processed = results['mdao_processed']['processed_image']
            axes[2].imshow(mdao_processed)
            axes[2].set_title(f'{plot_titles[2]}\n(Para segundo modelo)')
            axes[2].axis('off')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Resultados guardados en: {save_path}")
            
        plt.show()

        # Imprimir métricas de predicción
        print("\nPredicción del modelo:")
        for i, prob in enumerate(results['default_prediction']['probabilities']):
            print(f"{self.class_names[i]}: {prob*100:.1f}%")
