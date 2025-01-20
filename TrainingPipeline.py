import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class TrainingPipeline:
    def __init__(self, 
                 train_data_path: str,
                 test_data_path: str,
                 train_images_dir: str,
                 test_images_dir: str,
                 input_shape: tuple = (380, 380, 3),
                 num_classes: int = 3):
        """
        Inicializa el pipeline de entrenamiento
        
        Args:
            train_data_path: Ruta al CSV de entrenamiento
            test_data_path: Ruta al CSV de test
            train_images_dir: Directorio de imágenes de entrenamiento
            test_images_dir: Directorio de imágenes de test
            input_shape: Forma de las imágenes de entrada
            num_classes: Número de clases
        """
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.train_images_dir = train_images_dir
        self.test_images_dir = test_images_dir
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def _load_data(self) -> tuple:
        """Carga y prepara los datos"""
        logging.info("Cargando datos...")
        
        try:
            # Cargar datos de entrenamiento
            train_df = pd.read_csv(self.train_data_path)
            train_df['quality'] = train_df['quality'].astype(str)
            train_df['image'] = 'processed_' + train_df['image']
            
            # Cargar datos de test
            test_df = pd.read_csv(self.test_data_path)
            test_df['quality'] = test_df['quality'].astype(str)
            test_df['image'] = 'processed_' + test_df['image']
            
            logging.info(f"Datos cargados - Train: {len(train_df)}, Test: {len(test_df)} imágenes")
            logging.info("\nDistribución de clases en train:")
            logging.info(train_df['quality'].value_counts().sort_index())
            
            return train_df, test_df
            
        except Exception as e:
            logging.error(f"Error cargando datos: {e}")
            raise
            
    def _perform_cross_validation(self, train_df: pd.DataFrame) -> tuple:
        """Realiza cross-validation"""
        logging.info("Realizando cross-validation...")
        
        try:
            fold_histories, fold_metrics, best_fold = self.model.cross_validate(
                labels_df=train_df,
                images_dir=self.train_images_dir,
                n_splits=5,
                epochs=1,
                batch_size=32
            )
            
            logging.info(f"Cross-validation completada. Mejor fold: {best_fold}")
            logging.info("\nMétricas por fold:")
            logging.info(fold_metrics)
            
            return fold_histories, fold_metrics, best_fold
            
        except Exception as e:
            logging.error(f"Error en cross-validation: {e}")
            raise
            
    def _evaluate_cv_model(self, test_df: pd.DataFrame) -> dict:
        """Evalúa el mejor modelo de CV en test"""
        logging.info("Evaluando mejor modelo de cross-validation en test...")
        
        try:
            cv_test_results = self.model.evaluate_test_set(
                test_df=test_df,
                test_dir=self.test_images_dir,
                batch_size=32,
                save_dir='cv_test_evaluation'
            )
            return cv_test_results
            
        except Exception as e:
            logging.error(f"Error evaluando modelo CV: {e}")
            raise
            
    def _train_final_model(self, train_df: pd.DataFrame) -> tuple:
        """Entrena el modelo final"""
        logging.info("Realizando entrenamiento final...")
        
        try:
            history, metrics = self.model.train_model(
                labels_df=train_df,
                images_dir=self.train_images_dir,
                epochs=1,
                batch_size=32,
                validation_split=0.1,
                learning_rate=1e-4,
                trainable_fraction=0.1
            )
            
            logging.info("Entrenamiento final completado")
            return history, metrics
            
        except Exception as e:
            logging.error(f"Error en entrenamiento final: {e}")
            raise
            
    def _save_results(self, history, cv_test_results, final_test_results):
        """Guarda todos los resultados"""
        logging.info("Guardando información...")
        
        try:
            # Guardar modelo
            self.model.model.save('best_model.keras')
            
            # Guardar historia y métricas
            with open('training_history.json', 'w') as f:
                json.dump(history.history, f)
            
            # Guardar umbrales optimizados
            if self.model.optimal_thresholds is not None:
                with open('best_thresholds.json', 'w') as f:
                    json.dump(self.model.optimal_thresholds, f, indent=4)
            
            # Guardar información general
            self.model.save_model_info(save_dir='model_info')
            
            # Mostrar comparación de resultados
            self._print_comparison(cv_test_results, final_test_results)
            
        except Exception as e:
            logging.error(f"Error guardando resultados: {e}")
            raise
            
    def _print_comparison(self, cv_results, final_results):
        """Imprime comparación de resultados"""
        logging.info("\n=== Comparación de Resultados ===")
        
        if 'metrics' in cv_results:
            cr = cv_results['metrics']['classification_report']
            logging.info("\nMejor modelo de Cross-Validation en Test:")
            logging.info(f"Accuracy: {cr['accuracy']:.4f}")
            logging.info(f"Macro avg F1: {cr['macro avg']['f1-score']:.4f}")
            logging.info(f"Weighted avg F1: {cr['weighted avg']['f1-score']:.4f}")
        
        if 'metrics' in final_results:
            cr = final_results['metrics']['classification_report']
            logging.info("\nModelo Final en Test:")
            logging.info(f"Accuracy: {cr['accuracy']:.4f}")
            logging.info(f"Macro avg F1: {cr['macro avg']['f1-score']:.4f}")
            logging.info(f"Weighted avg F1: {cr['weighted avg']['f1-score']:.4f}")
            
    def train(self):
        """Ejecuta el pipeline completo de entrenamiento"""
        logging.info("=== Iniciando Pipeline de Entrenamiento ===")
        
        try:
            # 1. Inicializar modelo
            self.model = QualityAssessmentModel(
                input_shape=self.input_shape,
                num_classes=self.num_classes,
                weights='imagenet',
                seed=123
            )
            
            # 2. Cargar datos
            train_df, test_df = self._load_data()
            
            # 3. Cross-validation
            fold_histories, fold_metrics, best_fold = self._perform_cross_validation(train_df)
            fold_metrics.to_csv('cross_validation_metrics.csv', index=False)
            
            # 4. Evaluar mejor modelo CV
            cv_test_results = self._evaluate_cv_model(test_df)
            
            # 5. Entrenamiento final
            history, metrics = self._train_final_model(train_df)
            
            # 6. Evaluar modelo final
            final_test_results = self.model.evaluate_test_set(
                test_df=test_df,
                test_dir=self.test_images_dir,
                batch_size=32,
                save_dir='final_test_evaluation'
            )
            
            # 7. Visualizar y guardar resultados
            self.model.plot_training_history(save_dir='training_plots')
            self._save_results(history, cv_test_results, final_test_results)
            
            logging.info("=== Pipeline de entrenamiento completado exitosamente ===")
            return self.model
            
        except Exception as e:
            logging.error(f"Error en pipeline de entrenamiento: {e}")
            raise

# Ejemplo de uso:
if __name__ == "__main__":
    pipeline = TrainingPipeline(
        train_data_path='/kaggle/input/eqa-images-preprocessed/Label_EyeQ_train.csv',
        test_data_path='/kaggle/input/eqa-images-preprocessed/Label_EyeQ_test.csv',
        train_images_dir='/kaggle/input/eqa-images-preprocessed/train/train',
        test_images_dir='/kaggle/input/eqa-images-preprocessed/test',
    )
    
    trained_model = pipeline.train()
