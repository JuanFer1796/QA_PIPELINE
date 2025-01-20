import pandas as pd
from typing import Dict, Tuple
import tensorflow as tf

class RetrainPipeline:
    def __init__(self, processor: RetinaProcessor, model_path: str):
        """
        Initialize retrain pipeline
        
        Args:
            processor (RetinaProcessor): Procesador de retina para preprocesamiento
            model_path (str): Ruta al modelo pre-entrenado
        """
        self.processor = processor
        self.model = QualityAssessmentModel(
            input_shape=(380, 380, 3),
            num_classes=3
        )
        self.model.model = tf.keras.models.load_model(model_path)

    def preprocess_images(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesa las imágenes del DataFrame usando el RetinaProcessor
        
        Args:
            df (pd.DataFrame): DataFrame con las imágenes a preprocesar
            
        Returns:
            pd.DataFrame: DataFrame con información de imágenes preprocesadas
        """
        processed_df = df.copy()
        # Aquí va la lógica de preprocesamiento usando self.processor
        return processed_df
        
    def retrain_model(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Reentrena el modelo con los nuevos datos
        
        Args:
            train_df (pd.DataFrame): DataFrame de entrenamiento
            test_df (pd.DataFrame): DataFrame de test
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrames procesados de train y test
        """
        # Usar el método retrain del QualityAssessmentModel
        results = self.model.retrain_model(
            new_data_df=train_df,
            images_dir=train_df['image_dir'].iloc[0],  # Asume que todas las imágenes están en el mismo directorio
            min_f1_score=0.80,
            epochs=50,
            batch_size=32,
            use_cross_validation=True,
            n_splits=5
        )
        return train_df, test_df

    def run(self) -> Dict:
        """
        Ejecuta el pipeline completo
        
        Returns:
            Dict: Resultados del reentrenamiento
        """
        try:
            # 1. Preprocesar imágenes de train y test
            processed_train = self.preprocess_images(self.train_df)
            processed_test = self.preprocess_images(self.test_df)

            # 2. Reentrenar modelo
            final_train, final_test = self.retrain_model(processed_train, processed_test)

            return {
                'status': 'success',
                'processed_train': final_train,
                'processed_test': final_test,
                'model': self.model
            }

        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e)
            }
