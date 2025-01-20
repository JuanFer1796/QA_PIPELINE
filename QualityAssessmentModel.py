import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
import json
import os
from sklearn.model_selection import KFold

class QualityAssessmentModel:
    def __init__(self, input_shape=(380, 380, 3), num_classes=3, weights='imagenet', seed=123):
        self.seed = seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights = weights
        self.model = None
        self.history = None
        self.optimal_thresholds = None
        self._setup_gpu()

    def _setup_gpu(self):
        """Configura el uso de GPU y memoria"""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

    def build_model(self, trainable_fraction=0.1):
        """Construye el modelo EfficientNetB4 con capas adicionales para fine-tuning.
    
        Parameters:
        -----------
        trainable_fraction : float, opcional
            Fracción de capas del modelo base que serán entrenables. 
            Debe estar en el rango [0, 1], donde 0 significa todas las capas congeladas y 1 significa todas entrenables.
        """
        base_model = EfficientNetB4(weights=self.weights, include_top=False, input_shape=self.input_shape)
        
        # Determinar el número de capas entrenables según la fracción especificada
        total_layers = len(base_model.layers)
        trainable_layers = int(total_layers * trainable_fraction)
        for layer in base_model.layers[:total_layers - trainable_layers]:
            layer.trainable = False
        
        print(f"Total de capas en el modelo base: {total_layers}")
        print(f"Capas entrenables: {trainable_layers} (fracción: {trainable_fraction})")
        
        inputs = Input(shape=self.input_shape)
        x = base_model(inputs)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='swish')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='swish')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='swish')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)


    def compile_model(self, learning_rate=1e-4):
        """Compila el modelo con el optimizador Adam y una tasa de aprendizaje dada"""
        if self.model is None:
            self.build_model()
        print(f"\nCompilando modelo con learning rate: {learning_rate}")
        optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        print("Modelo compilado exitosamente")

    def get_data_generator(self, labels_df, images_dir, subset='training', batch_size=32, validation_split=0.2):
        """Crea generadores de datos de entrenamiento o validación según el subset especificado"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        labels_df['quality'] = labels_df['quality'].astype(str)  # Asegura que las etiquetas estén en formato de texto
        datagen = ImageDataGenerator(
            validation_split=validation_split,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        generator = datagen.flow_from_dataframe(
            labels_df,
            directory=images_dir,
            x_col='image',
            y_col='quality',
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            subset=subset,
            seed=self.seed,
            class_mode='categorical'
        )
        
        return generator

    def calculate_class_weights(self, train_gen):
        """Calcula pesos de clase balanceados para el entrenamiento"""
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_gen.classes),
            y=train_gen.classes
        )
        return dict(enumerate(class_weights))

    def get_callbacks(self, checkpoint_path='best_model.keras'):
        """Genera callbacks para el entrenamiento"""
        return [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(checkpoint_path, save_best_only=True),
            CSVLogger('training_log.csv'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
        ]

    def train_model(self, labels_df, images_dir, epochs=50, batch_size=32, validation_split=0.2, learning_rate=1e-4, trainable_fraction=0.1, save_model_path='modelo.keras'):
        """
        Entrena el modelo usando los datos y callbacks especificados, guarda el modelo y evalúa su rendimiento.
    
        Parameters:
        -----------
        labels_df : DataFrame
            DataFrame que contiene las rutas de imágenes y etiquetas.
        images_dir : str
            Directorio que contiene las imágenes.
        epochs : int, opcional
            Número de épocas de entrenamiento.
        batch_size : int, opcional
            Tamaño del batch.
        validation_split : float, opcional
            Proporción del conjunto de datos para validación.
        learning_rate : float, opcional
            Tasa de aprendizaje para el optimizador.
        trainable_fraction : float, opcional
            Fracción de capas del modelo base que serán entrenables.
        save_model_path : str, opcional
            Ruta donde se guardará el modelo entrenado.
    
        Returns:
        --------
        history : History
            Historial del entrenamiento.
        val_metrics : dict
            Diccionario con las métricas de validación finales.
        """
        # Construir y compilar el modelo
        print("\nConstruyendo y compilando el modelo...")
        self.build_model(trainable_fraction=trainable_fraction)
        self.compile_model(learning_rate=learning_rate)
        
        # Preparar generadores de datos
        print("\nPreparando generadores de datos de entrenamiento y validación...")
        train_generator = self.get_data_generator(labels_df, images_dir, subset='training', batch_size=batch_size, validation_split=validation_split)
        validation_generator = self.get_data_generator(labels_df, images_dir, subset='validation', batch_size=batch_size, validation_split=validation_split)
        
        # Calcular los pesos de clase
        print("\nCalculando pesos de clase...")
        class_weights = self.calculate_class_weights(train_generator)
        
        # Configurar los callbacks
        callbacks = self.get_callbacks()
        
        # Iniciar el entrenamiento
        print(f"\nIniciando entrenamiento por {epochs} épocas...")
        self.history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks
        )
        
        # Evaluación en el conjunto de validación
        print("\nEvaluando el modelo en el conjunto de validación...")
        val_metrics = self.model.evaluate(validation_generator, verbose=0)
        metrics_dict = {
            'val_loss': val_metrics[0],
            'val_accuracy': val_metrics[1]
        }
        print(f"Resultados de validación - Loss: {metrics_dict['val_loss']}, Accuracy: {metrics_dict['val_accuracy']}")
        
        # Análisis adicional de predicciones
        print("\nRealizando análisis de predicciones y optimización de umbrales...")
        val_predictions, optimal_thresholds = self.evaluate_and_optimize(
            validation_generator,
            save_prefix='final_validation'
        )
         # Convertir optimal_thresholds a tipo float para que sea JSON serializable
        optimal_thresholds = {k: float(v) for k, v in optimal_thresholds.items()}

        # Guardar el modelo entrenado
        print(f"\nGuardando modelo entrenado en {save_model_path}...")
        self.model.save(save_model_path)
        
        # Guardar historial de entrenamiento y métricas
        print("\nGuardando historial de entrenamiento, métricas y umbrales óptimos...")
        with open('training_history.json', 'w') as f:
            json.dump(self.history.history, f)
        with open('validation_metrics.json', 'w') as f:
            json.dump(metrics_dict, f)
        with open('optimal_thresholds.json', 'w') as f:
            json.dump(optimal_thresholds, f)
    
        return self.history, metrics_dict
    def evaluate_and_optimize(self, generator, save_prefix='evaluation', metric='f1', beta=1.0, is_test=False):
        """
        Evalúa el modelo en un conjunto de datos, calcula métricas y optimiza umbrales.

        Parameters:
        -----------
        generator : DataFrameIterator
            Generador de datos para evaluar el modelo.
        save_prefix : str, opcional
            Prefijo para guardar los archivos de resultados y métricas.
        metric : str, opcional
            Métrica para optimizar los umbrales ('f1' o 'fbeta').
        beta : float, opcional
            Parámetro beta para F-beta score (solo si metric='fbeta').
        is_test : bool, opcional
            Si es True, solo realiza predicciones sin optimizar umbrales.

        Returns:
        --------
        results_df : DataFrame
            DataFrame con las predicciones y métricas calculadas.
        optimal_thresholds : dict
            Diccionario con los umbrales óptimos para cada clase.
        """
        print(f"\nEvaluando predicciones para el conjunto con prefijo '{save_prefix}'...")
        predictions = self.model.predict(generator)

        # Crear DataFrame de resultados
        results_df = pd.DataFrame()
        results_df['image'] = generator.filenames
        if hasattr(generator, 'classes'):
            results_df['true_label'] = generator.classes

        for i in range(self.num_classes):
            results_df[f'prob_class_{i}'] = predictions[:, i]

        results_df['predicted_class'] = np.argmax(predictions, axis=1)

        # Si es test y tenemos umbrales optimizados, los aplicamos sin recalcular
        if is_test and self.optimal_thresholds is not None:
            results_df = self.apply_thresholds(results_df)
            results_df.to_csv(f'{save_prefix}_predictions.csv', index=False)
            return results_df, self.optimal_thresholds

        # Optimizar umbrales solo si no es test y tenemos clases
        if not is_test and hasattr(generator, 'classes'):
            print("\nOptimizando thresholds...")
            y_true = label_binarize(results_df['true_label'], classes=range(self.num_classes))
            optimal_thresholds = {}
            additional_metrics = {}

            for i in range(self.num_classes):
                probs = results_df[f'prob_class_{i}']

                # Calcular métricas originales
                precision, recall, thresholds = precision_recall_curve(y_true[:, i], probs)
                scores = [f1_score(y_true[:, i], (probs >= t).astype(int)) for t in thresholds] if metric == 'f1' else \
                         [fbeta_score(y_true[:, i], (probs >= t).astype(int), beta=beta) for t in thresholds]
                optimal_thresholds[str(i)] = thresholds[np.argmax(scores)] 

                # Calcular solo AUC-ROC
                fpr, tpr, _ = roc_curve(y_true[:, i], probs)
                auc_score = auc(fpr, tpr)

                # Almacenar solo AUC-ROC para esta clase
                additional_metrics[f'class_{i}'] = {
                    'auc_roc': float(auc_score)
                }

            self.optimal_thresholds = optimal_thresholds

            # Aplicar umbrales para crear columna de predicciones ajustadas
            results_df = self.apply_thresholds(results_df)

            # Guardar métricas y resultados
            metrics = {
                'classification_report': classification_report(results_df['true_label'], 
                                                            results_df['predicted_class_with_threshold'], 
                                                            output_dict=True),
                'confusion_matrix': confusion_matrix(results_df['true_label'], 
                                                  results_df['predicted_class_with_threshold']).tolist(),
                'additional_metrics': additional_metrics
            }
            with open(f'{save_prefix}_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
            results_df.to_csv(f'{save_prefix}_predictions.csv', index=False)

            # Imprimir solo AUC-ROC
            print("\nResumen de métricas adicionales:")
            for class_name, class_metrics in additional_metrics.items():
                print(f"\n{class_name}:")
                print(f"  AUC-ROC: {class_metrics['auc_roc']:.3f}")

        return results_df, self.optimal_thresholds

    def predict_single_image(self, image_path=None, image_array=None, return_probabilities=False):
        """
        Realiza predicción sobre una única imagen.
    
        Parameters:
        -----------
        image_path : str, opcional
            Ruta completa a la imagen a predecir. Debe ser proporcionada si image_array es None.
        image_array : np.array o tf.Tensor, opcional
            Imagen ya cargada y preprocesada. Si se proporciona, se omite image_path.
        return_probabilities : bool, opcional
            Si True, devuelve las probabilidades para cada clase.
    
        Returns:
        --------
        dict
            Diccionario con la predicción, confianza y, opcionalmente, probabilidades.
        """
        if self.model is None:
            raise ValueError("No hay modelo cargado. Usa build_model primero o entrena el modelo.")
        
        # Verificación de los argumentos
        if image_array is None and image_path is None:
            raise ValueError("Debes proporcionar image_path o image_array.")
        
        # Preprocesamiento de la imagen si se proporciona image_path
        if image_array is None:
            img = tf.keras.preprocessing.image.load_img(image_path)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
        else:
            img_array = image_array
        
        # Asegurar que la imagen tenga el tamaño correcto
        img_array = tf.image.resize_with_pad(img_array, self.input_shape[0], self.input_shape[1])
        img_array = tf.expand_dims(img_array, axis=0)  # Añadir dimensión batch
    
        # Realizar predicción
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
    
        result = {'predicted_class': int(predicted_class), 'confidence': float(confidence)}
    
        # Añadir probabilidades si se solicitan
        if return_probabilities:
            result['probabilities'] = {f'class_{i}': float(pred) for i, pred in enumerate(predictions[0])}
    
        # Aplicar umbrales optimizados si existen
        if self.optimal_thresholds:
            binary_predictions = [(predictions[0][i] >= self.optimal_thresholds[i]).astype(int) for i in range(self.num_classes)]
            result['predicted_class_with_threshold'] = int(np.argmax(binary_predictions))
    
        return result

    def retrain_model(self, new_data_df, images_dir, min_f1_score=0.80, epochs=50, batch_size=32, use_cross_validation=False, n_splits=5):
        """
        Evalúa el modelo actual en nuevos datos y reentrena si el rendimiento es insuficiente,
        usando los métodos existentes train_model o cross_validate.
    
        Parameters:
        -----------
        new_data_df : pandas.DataFrame
            DataFrame con los nuevos datos de entrenamiento.
        images_dir : str
            Directorio que contiene las imágenes.
        min_f1_score : float
            F1-score mínimo aceptable.
        epochs : int, opcional
            Número de épocas para el reentrenamiento.
        batch_size : int, opcional
            Tamaño del batch para el reentrenamiento.
        use_cross_validation : bool, opcional
            Si True, usa cross_validate para el reentrenamiento.
            Si False, usa train_model.
        n_splits : int, opcional
            Número de folds para cross-validation si use_cross_validation es True.
    
        Returns:
        --------
        dict
            Diccionario con los resultados del reentrenamiento incluyendo:
            - 'retraining_needed': bool
            - 'initial_f1': float
            - 'final_f1': float
            - 'improvement': float
            - 'training_history': dict o list (dependiendo del método usado)
        """
        print("\nEvaluando rendimiento actual en nuevos datos...")
        
        # Guardamos el estado actual del modelo
        current_weights = self.model.get_weights()
        
        # Evaluación inicial
        eval_generator = self.get_data_generator(
            new_data_df, 
            images_dir, 
            subset='validation', 
            batch_size=batch_size,
            validation_split=0.2
        )
        
        eval_predictions, _ = self.evaluate_and_optimize(
            eval_generator, 
            save_prefix='pre_retrain_evaluation'
        )
        
        initial_f1 = f1_score(
            eval_predictions['true_label'], 
            eval_predictions['predicted_class'], 
            average='weighted'
        )
        
        print(f"\nF1-score inicial en nuevos datos: {initial_f1:.4f}")
        print(f"F1-score mínimo requerido: {min_f1_score:.4f}")
        
        results = {
            'retraining_needed': initial_f1 < min_f1_score,
            'initial_f1': initial_f1,
            'final_f1': initial_f1,  # Por defecto, se mantiene igual
            'improvement': 0.0,
            'training_history': None
        }
        
        if initial_f1 < min_f1_score:
            print("\nReentrenando modelo debido a rendimiento insuficiente...")
            
            if use_cross_validation:
                print(f"\nUsando {n_splits}-fold cross validation para reentrenamiento...")
                fold_histories, fold_metrics = self.cross_validate(
                    new_data_df,
                    images_dir,
                    n_splits=n_splits,
                    epochs=epochs,
                    batch_size=batch_size
                )
                
                # Evaluamos el mejor fold
                best_fold = np.argmin([m['val_loss'] for m in fold_metrics]) + 1
                print(f"\nCargando mejor modelo del fold {best_fold}...")
                self.model.load_weights(f'best_model_fold_{best_fold}.keras')
                training_history = fold_histories[best_fold - 1]
                
            else:
                print("\nUsando train_model para reentrenamiento...")
                training_history = self.train_model(
                    new_data_df,
                    images_dir,
                    epochs=epochs,
                    batch_size=batch_size
                ).history
            
            # Evaluación final
            final_eval_predictions, _ = self.evaluate_and_optimize(
                eval_generator,
                save_prefix='post_retrain_evaluation'
            )
            
            final_f1 = f1_score(
                final_eval_predictions['true_label'],
                final_eval_predictions['predicted_class'],
                average='weighted'
            )
            
            print(f"\nF1-score después del reentrenamiento: {final_f1:.4f}")
            
            # Si el reentrenamiento no mejoró, volvemos al modelo original
            if final_f1 < initial_f1:
                print("\nAdvertencia: El reentrenamiento no mejoró el rendimiento. Restaurando modelo original...")
                self.model.set_weights(current_weights)
                final_f1 = initial_f1
            else:
                print("\nGuardando modelo reentrenado...")
                self.model.save('retrained_model.keras')
                
            results.update({
                'final_f1': final_f1,
                'improvement': final_f1 - initial_f1,
                'training_history': training_history
            })
        else:
            print("\nNo es necesario reentrenar el modelo.")
        
        return results

    def apply_thresholds(self, predictions_df):
        if self.optimal_thresholds is None:
            raise ValueError("No hay thresholds optimizados disponibles.")

        # Aplicar umbrales a las predicciones
        binary_predictions = np.zeros((len(predictions_df), self.num_classes))
        for i in range(self.num_classes):
            binary_predictions[:, i] = (predictions_df[f'prob_class_{i}'] >= self.optimal_thresholds[str(i)]).astype(int)  # <- SOLO ESTA LÍNEA CAMBIA

        # Si ninguna clase cumple el umbral, se asigna la clase con mayor probabilidad
        no_class = ~binary_predictions.any(axis=1)
        if no_class.any():
            binary_predictions[no_class] = np.eye(self.num_classes)[np.argmax(predictions_df.loc[no_class, [f'prob_class_{i}' for i in range(self.num_classes)]].values, axis=1)]

        # Si más de una clase cumple el umbral, se elige la de mayor probabilidad
        multiple_classes = binary_predictions.sum(axis=1) > 1
        if multiple_classes.any():
            binary_predictions[multiple_classes] = np.eye(self.num_classes)[np.argmax(predictions_df.loc[multiple_classes, [f'prob_class_{i}' for i in range(self.num_classes)]].values, axis=1)]

        # Agregar la columna de clase predicha ajustada con umbrales
        predictions_df['predicted_class_with_threshold'] = np.argmax(binary_predictions, axis=1)

        return predictions_df

   
    def cross_validate(self, labels_df, images_dir, n_splits=5, epochs=50, batch_size=32):
        """
        Realiza una validación cruzada K-fold optimizada, enfocándose en el log loss como criterio principal.
        La optimización de umbrales se realiza solo una vez al final con el mejor modelo.

        Parameters:
        -----------
        labels_df : DataFrame
            DataFrame que contiene las rutas de imágenes y etiquetas.
        images_dir : str
            Directorio que contiene las imágenes.
        n_splits : int, opcional
            Número de folds para cross-validation.
        epochs : int, opcional
            Número de épocas para cada fold.
        batch_size : int, opcional
            Tamaño del batch para cada fold.

        Returns:
        --------
        fold_histories : list
            Lista de historiales de entrenamiento para cada fold.
        fold_metrics : DataFrame
            DataFrame con las métricas de validación finales para cada fold.
        """
        print(f"\nIniciando {n_splits}-fold cross-validation...")
        df = labels_df.copy()
        df['quality'] = labels_df['quality'].astype(str)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        fold_histories = []
        fold_metrics = []
        best_fold = None
        best_val_loss = float('inf')

        # Preparar generadores base
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

        for fold, (train_idx, val_idx) in enumerate(kf.split(df), 1):
            print(f"\nFold {fold}/{n_splits}")
            print("-" * 50)

            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()

            # Crear generadores
            train_generator = train_datagen.flow_from_dataframe(
                dataframe=train_df,
                directory=images_dir,
                x_col='image',
                y_col='quality',
                target_size=(self.input_shape[0], self.input_shape[1]),
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=True,
                seed=self.seed
            )

            val_generator = val_datagen.flow_from_dataframe(
                dataframe=val_df,
                directory=images_dir,
                x_col='image',
                y_col='quality',
                target_size=(self.input_shape[0], self.input_shape[1]),
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False
            )

            # Reiniciar modelo para cada fold
            self.model = None
            self.build_model()
            self.compile_model()

            # Calcular class weights
            class_weights = self.calculate_class_weights(train_generator)

            # Entrenar el modelo
            print(f"\nEntrenando modelo en fold {fold}...")
            history = self.model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                class_weight=class_weights,
                callbacks=self.get_callbacks(checkpoint_path=f'best_model_fold_{fold}.keras')
            )

            # Evaluar usando solo log loss y accuracy
            val_metrics = self.model.evaluate(val_generator, verbose=0)
            val_loss, val_accuracy = val_metrics[0], val_metrics[1]
            print(f"Fold {fold} - Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

            # Guardar resultados del fold
            fold_histories.append(history.history)
            fold_metrics.append({
                'fold': fold,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })

            # Actualizar mejor modelo basado solo en val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_fold = fold
                self.model.save('best_model.keras')
                print(f"\nMejor modelo actualizado con fold {fold} - Validation Loss: {val_loss}")

            # Guardar resultados intermedios
            pd.DataFrame(fold_metrics).to_csv('cross_validation_results.csv', index=False)

        # Cargar el mejor modelo
        print(f"\nCargando mejor modelo del fold {best_fold}...")
        self.model.load_weights(f'best_model_fold_{best_fold}.keras')

        # Realizar una única optimización de umbrales con el mejor modelo
        if hasattr(df, 'quality'):  # Solo si tenemos etiquetas disponibles
            print("\nOptimizando umbrales con el mejor modelo...")
            # Crear un generador de validación con todos los datos
            final_val_generator = val_datagen.flow_from_dataframe(
                dataframe=df,
                directory=images_dir,
                x_col='image',
                y_col='quality',
                target_size=(self.input_shape[0], self.input_shape[1]),
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False
            )

            _, optimal_thresholds = self.evaluate_and_optimize(
                final_val_generator,
                save_prefix='final_validation'
            )

            # Guardar los thresholds optimizados
            optimal_thresholds = {k: float(v) for k, v in optimal_thresholds.items()}
            with open('best_thresholds.json', 'w') as f:
                json.dump(optimal_thresholds, f)

        print("\nValidación cruzada optimizada completada.")
        print(f"Mejor modelo guardado del fold {best_fold} con Validation Loss: {best_val_loss}")

        return fold_histories, pd.DataFrame(fold_metrics)
    def predict(self, generator, apply_thresholds=True, save_prefix=None):
        """
        Realiza predicciones en lote usando un generador de datos.

        Parameters:
        -----------
        generator : DataFrameIterator
            Generador de datos con las imágenes a predecir.
        apply_thresholds : bool, opcional
            Si True, aplica los umbrales optimizados a las predicciones.
        save_prefix : str, opcional
            Si se proporciona, guarda las predicciones en un archivo CSV.

        Returns:
        --------
        DataFrame
            DataFrame con las predicciones que incluye:
            - Nombre de la imagen
            - Probabilidades para cada clase
            - Clase predicha
            - Clase predicha con umbrales (si apply_thresholds=True)
        """
        if self.model is None:
            raise ValueError("No hay modelo cargado. Usa build_model primero o carga un modelo existente.")

        print("\nRealizando predicciones...")
        predictions = self.model.predict(generator, verbose=1)

        # Crear DataFrame con resultados
        results_df = pd.DataFrame()
        results_df['image'] = generator.filenames

        # Agregar probabilidades por clase
        for i in range(self.num_classes):
            results_df[f'prob_class_{i}'] = predictions[:, i]

        # Predicción basada en máxima probabilidad
        results_df['predicted_class'] = np.argmax(predictions, axis=1)

        # Aplicar umbrales si se solicita y están disponibles
        if apply_thresholds and self.optimal_thresholds is not None:
            results_df = self.apply_thresholds(results_df)
            print("\nUmbrales optimizados aplicados a las predicciones.")

        # Guardar resultados si se especifica un prefijo
        if save_prefix:
            save_path = f'{save_prefix}_predictions.csv'
            results_df.to_csv(save_path, index=False)
            print(f"\nPredicciones guardadas en: {save_path}")

        return results_df
