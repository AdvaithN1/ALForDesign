from collections.abc import Sequence
from typing import Callable, Any, List, Tuple, Union
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from .helper import classifier_predict_simple_uncertainty, fit
from autogluon.tabular import TabularPredictor
import pandas as pd

class GluonClassifier(TabularPredictor):
    def __init__(self, label, path=None):
        super().__init__(label=label, path=path, verbosity=1, problem_type='binary')
        self.lab = label
        self.pat = path
    
    def fit(self, X, y_train, **kwargs):
        super().__init__(label=self.lab, path=self.pat, verbosity=1, problem_type='binary')
        df_train = pd.DataFrame(X)
        df_train[self.label] = y_train
        
        super().fit(df_train, presets='medium_quality',**kwargs)
    
    def predict(self, X, **kwargs):
        df_test = pd.DataFrame(X)
        
        predictions_proba = super().predict_proba(df_test, **kwargs)
        
        return predictions_proba.iloc[:, 1].values  # Assuming positive class is the second column

class GluonRegressor(TabularPredictor):
    def __init__(self, label, path=None):
        super().__init__(label=label, path=path, problem_type='regression', verbosity=1)
        self.lab = label
        self.pat = path

    def fit(self, X, y_train, **kwargs):
        super().__init__(label=self.lab, path=self.pat, problem_type='regression', verbosity=1)
        df_train = pd.DataFrame(X)
        df_train[self.label] = y_train
        
        super().fit(df_train,  presets='medium_quality', **kwargs)
    
    def predict(self, X, **kwargs):
        df_test = pd.DataFrame(X)
        
        predictions = super().predict(df_test, **kwargs)
        
        return predictions.values


def weighted_binary_crossentropy(alpha, beta):
    """
    @param alpha is the loss weight for the positive (1) class
    
    @param beta is the loss weight for the negative (0) class
    """
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = - (alpha * y_true * K.log(y_pred) + beta * (1 - y_true) * K.log(1 - y_pred))
        return K.mean(loss)
    return loss

class TargetPerformance:
    def __init__(self, validity_checker:Callable[[np.ndarray], np.ndarray], label, estimator=None, classifier=None, target_perf_fit_func:Callable[[Any, np.ndarray, np.ndarray], None]=fit):
        self.estimator = estimator
        self.target_perf_fit_func = target_perf_fit_func
        self.label = label
        self.validity_checker = validity_checker
        self.classifier = classifier
class TargetPerformanceRegress:
    def __init__(self, validity_checker:Callable[[np.ndarray], np.ndarray], label, estimator=None, classifier=None, target_perf_fit_func:Callable[[Any, np.ndarray, np.ndarray], None]=fit):
        self.regressor = estimator
        self.classifier = classifier
        self.target_perf_fit_func = target_perf_fit_func
        self.label = label
        self.validity_checker = validity_checker

class TargetPerformanceHyperparam:
    def __init__(self, validity_checker:Callable[[np.ndarray], np.ndarray], label, target_perf_fit_func:Callable[[Any, np.ndarray, np.ndarray], None]=fit):
        self.target_perf_fit_func = target_perf_fit_func
        self.label = label
        self.validity_checker = validity_checker
        self.classifier = GluonClassifier(label)
        self.regressor = GluonRegressor(label)
        
class ContinuousDesignBound:
    def __init__(self, lower_bound:float, upper_bound:float, label):
        self.label = label
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.validate()
    def validate(self):
        if(self.lower_bound>=self.upper_bound):
            raise ValueError("ContinuousDesignBound lower bound must be less than upper bound.")
        if(self.label=="" or self.label is None):
            raise ValueError("ContinuousDesignBound must have a label.")

class CategoricalDesignBound:
    def __init__(self, categories:List[Any], label):
        self.categories = categories
        self.label = label
        self._validate()
    
    def _validate(self):
        if not (isinstance(self.categories, np.ndarray) or isinstance(self.categories, Sequence)):
            raise ValueError("CategoricalDesignBound categories must be a list or numpy array.")
        if(len(self.categories)<=1):
            raise ValueError("CategoricalDesignBound must have at least 2 categories.")
        if(self.label=="" or self.label is None):
            raise ValueError("CategoricalDesignBound must have a label.")

class DataSetup:
    def __init__(self, params:List[Union[ContinuousDesignBound,CategoricalDesignBound]], targetPerfs:List[TargetPerformance], PERFORMANCE_ESTIMATOR_DROPOUT_RATE:float=0.2):
        self.params = params
        self.target_perfs = targetPerfs
        total_input_len = 0
        for param in params:
            if isinstance(param, ContinuousDesignBound):
                total_input_len += 1
            else:
                total_input_len += len(param.categories)
        for perf in targetPerfs:
            if(perf.estimator is None):
                perf.estimator = tf.keras.Sequential([
                    tf.keras.layers.Dense(100, activation='relu', input_shape=(total_input_len,)),
                    tf.keras.layers.Dense(50, activation='relu'),
                    tf.keras.layers.Dropout(PERFORMANCE_ESTIMATOR_DROPOUT_RATE),
                    tf.keras.layers.Dense(25, activation='relu'),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                perf.estimator.compile(optimizer='adam',
                loss=weighted_binary_crossentropy(3, 1),
                metrics=['accuracy'])

class RegressionDataSetup:
    def __init__(self, params:List[Union[ContinuousDesignBound,CategoricalDesignBound]], targetPerfs:List[TargetPerformanceRegress], PERFORMANCE_ESTIMATOR_DROPOUT_RATE:float=0.2):
        self.params = params
        self.target_perfs = targetPerfs
        total_input_len = 0
        for param in params:
            if isinstance(param, ContinuousDesignBound):
                total_input_len += 1
            else:
                total_input_len += len(param.categories)
        for perf in targetPerfs:
            if perf.regressor is None:
                perf.regressor = tf.keras.Sequential([
                    tf.keras.layers.Dense(100, activation='relu', input_shape=(total_input_len,)),
                    tf.keras.layers.Dense(50, activation='relu'),
                    tf.keras.layers.Dropout(PERFORMANCE_ESTIMATOR_DROPOUT_RATE),
                    tf.keras.layers.Dense(25, activation='relu'),
                    tf.keras.layers.Dense(1)
                ])
                perf.regressor.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])
            if perf.classifier is None:
                perf.classifier = tf.keras.Sequential([
                    tf.keras.layers.Dense(100, activation='relu', input_shape=(total_input_len,)),
                    tf.keras.layers.Dense(50, activation='relu'),
                    tf.keras.layers.Dropout(PERFORMANCE_ESTIMATOR_DROPOUT_RATE),
                    tf.keras.layers.Dense(25, activation='relu'),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                perf.classifier.compile(optimizer='adam',
                    loss=weighted_binary_crossentropy(0, 1),
                    metrics=['accuracy'])
                
class HyperparamDataSetup:
    def __init__(self, params:List[Union[ContinuousDesignBound,CategoricalDesignBound]], targetPerfs:List[TargetPerformanceHyperparam]):
        self.params = params
        self.target_perfs = targetPerfs
        total_input_len = 0
        for param in params:
            if isinstance(param, ContinuousDesignBound):
                total_input_len += 1
            else:
                total_input_len += len(param.categories)