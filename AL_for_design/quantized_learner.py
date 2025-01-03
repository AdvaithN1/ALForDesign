import numpy as np
import sobol_seq
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from .datasetup import HyperparamDataSetup, ContinuousDesignBound, CategoricalDesignBound, TargetPerformanceHyperparam, GluonClassifier, GluonRegressor
import math
from .helper import classifier_predict_simple_uncertainty, rsquared, get_regressor_uncertainty
import tensorflow as tf
from tensorflow.keras import backend as K
import keras_tuner as kt
from typing import Callable, Any, Tuple
import matplotlib.pyplot as plt
import os
from scipy.stats import beta


MEANING_OF_LIFE = 42
REDUNDANCY_THRESHOLD = 0.95

os.environ['PYTHONHASHSEED'] = str(MEANING_OF_LIFE)
np.random.seed(MEANING_OF_LIFE)
tf.random.set_seed(MEANING_OF_LIFE)

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

class QuantizedActiveLearner:
    def get_valid(X):
        # Returns True for valid points
        return np.ones(len(X), dtype=bool)
    
    def get_redundancy_regressor_accuracy(model, X, Y):
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=MEANING_OF_LIFE)
        model.fit(X_train, Y_train, 
            epochs=10, verbose=0)
        y_pred = model.predict(X_val)
        return rsquared(Y_val, y_pred)
    
    def _convert_to_one_hot(self, X):
        X = np.array(X)
        one_hot = None
        for i in range(len(self.params)):
            if i == 0:
                if isinstance(self.params[i], ContinuousDesignBound):
                    one_hot = np.array([X[:,i]])
                else:
                    one_hot = np.array([np.eye(len(self.params[i].categories))[X[:,i].astype(int)]])
                continue
            if isinstance(self.params[i], ContinuousDesignBound):
                one_hot = np.append(one_hot, [X[:,i]], axis=0)
            else:
                eye = np.eye(len(self.params[i].categories))[X[:,i].astype(int)]
                for i in range(eye.shape[1]):
                    one_hot = np.append(one_hot, [eye[:,i]], axis=0)
                # one_hot = np.append(one_hot, np.eye(len(self.params[i].categories))[X[:,i]], axis=1)
        return np.transpose(one_hot)
    
    def _convert_to_reduced_one_hot(self, X):
        X = np.array(X)
        one_hot = None
        for i in range(len(self.params)):
            if i == 0:
                if isinstance(self.params[i], ContinuousDesignBound):
                    one_hot = np.array([X[:,i]])
                else:
                    one_hot = np.array([np.eye(len(self.params[i].categories)/math.sqrt(2))[X[:,i].astype(int)]])
                continue
            if isinstance(self.params[i], ContinuousDesignBound):
                one_hot = np.append(one_hot, [X[:,i]], axis=0)
            else:
                eye = np.eye(len(self.params[i].categories))[X[:,i].astype(int)]/math.sqrt(2)
                for i in range(eye.shape[1]):
                    one_hot = np.append(one_hot, [eye[:,i]], axis=0)
                # one_hot = np.append(one_hot, np.eye(len(self.params[i].categories))[X[:,i]], axis=1)
        return np.transpose(one_hot)

    def _convert_to_one_hot_hypercube(self, X):
        factor = math.sqrt(len(self.params))
        X = np.array(X)
        one_hot = None
        for i in range(len(self.params)):
            if i == 0:
                if isinstance(self.params[i], ContinuousDesignBound):
                    one_hot = np.array([(X[:,i]-self.params[i].lower_bound)/(self.params[i].upper_bound - self.params[i].lower_bound)/factor])
                else:
                    one_hot = np.array([np.eye(len(self.params[i].categories)/math.sqrt(2)/factor)[X[:,i].astype(int)]])
                continue
            if isinstance(self.params[i], ContinuousDesignBound):
                one_hot = np.append(one_hot, [(X[:,i]-self.params[i].lower_bound)/(self.params[i].upper_bound - self.params[i].lower_bound)/factor], axis=0)
            else:
                eye = np.eye(len(self.params[i].categories))[X[:,i].astype(int)]/math.sqrt(2)/factor
                for i in range(eye.shape[1]):
                    one_hot = np.append(one_hot, [eye[:,i]], axis=0)
                # one_hot = np.append(one_hot, np.eye(len(self.params[i].categories))[X[:,i]], axis=1)
        return np.transpose(one_hot)

    def __init__(self, data_pack:HyperparamDataSetup, fail_predictor=None, use_absolute_query_strategy=False, classifier_predict_uncertainty_func: Callable[[Any, np.ndarray], Tuple[np.ndarray, np.ndarray]]=classifier_predict_simple_uncertainty, redundancy_regressor=None, redundancy_regressor_accuracy:Callable[[Any, np.ndarray, np.ndarray],float]=get_redundancy_regressor_accuracy, get_valid_func:Callable[[np.ndarray],np.ndarray]=get_valid, X_train: np.ndarray=None, Y_train_success: np.ndarray=None, Y_train_perfs: np.ndarray=None, skip_redundancy = False, DESIGN_SPACE_DENSITY:int=100000, UNCERTAINTY_DROP_THRESHOLD:float=0.05, DROPOUT_RATE:float=0.2):
        """
        Initializes the Active Learner.

        Parameters:
        ----------
        data_pack: HyperparamDataSetup
            The data pack containing the hyperparameters and target performances.
        fail_predictor: default=GluonClassifier("Failure Prediction")
            The classifier that predicts whether a point will fail or not. Default is an AutoGluon classifier wrapper GluonClassifier with the fit and predict methods.
        use_absolute_query_strategy: bool: default=False
            Whether to use the absolute query strategy. Default is False. If true, the Active Learner will not dynamically update the proximity weight.
        classifier_predict_uncertainty_func: Callable[[Any, np.ndarray], Tuple[np.ndarray, np.ndarray]]: default=classifier_predict_simple_uncertainty
            The function that predicts the uncertainty of the classifier. Default is the classifier_predict_simple_uncertainty function. Takes in Regressor and X_pool, and returns the predictions and uncertainty for each point in X_pool.
        redundancy_regressor: default=None
            The regressor that predicts the redundancy of performance values. Default is None. If None, a neural network tf regressor will be used.
        redundancy_regressor_accuracy: Callable[[Any, np.ndarray, np.ndarray],float]: default=get_redundancy_regressor_accuracy
            The function that calculates the accuracy of the regressor. Default is the get_regressor_accuracy function, which uses the r^2 metric. Takes in the regressor, X, and Y, and returns the accuracy.
        get_valid_func: Callable[[np.ndarray],np.ndarray]: default=get_valid
            The function that returns a boolean array, where True denotes a valid design. Default is get_valid, which returns all True.
        X_train: np.ndarray: default=None
            The initial X_train data. Default is None. If None, the Active Learner will start no initial data.
        Y_train_success: np.ndarray: default=None
            The initial Y_train_success data. Default is None. If None, the Active Learner will start with no initial data.
        Y_train_perfs: np.ndarray: default=None
            The initial Y_train_perfs data. Default is None. If None, the Active Learner will start with no initial data.
        skip_redundancy: bool: default=False
            Whether to skip the redundancy check. Default is False. If True, the Active Learner will not check for redundant performance values.
        DESIGN_SPACE_DENSITY: int: default=100000
            The number of points in the design space. Default is 100000. If this is set to a greater number, more points in the design space will be scored for more precise queries. A greater number is recommended for higher dimensional design spaces.
        UNCERTAINTY_THRESHOLD: float: default=0.05
            The uncertainty threshold for invalidity. Default is 0.05. When calling the teach method, each point has a probability of being valid. If the probability is less than the threshold, the point is considered invalid and will be dropped from the pool and not queried in the future.
        DROPOUT_RATE: float: default=0.2
            The dropout rate for the redundancy neural networks. Default is 0.2. This can be experimented with and used with monte carlo dropout to get more accurate uncertainty estimates.
        """
        
        self.skip_redundancy = skip_redundancy
        self.use_absolute_query_strategy = use_absolute_query_strategy
        if(fail_predictor is None):
            total_input_len = 0
            for param in data_pack.params:
                if isinstance(param, ContinuousDesignBound):
                    total_input_len += 1
                else:
                    total_input_len += len(param.categories)
            self.fail_predictor = GluonClassifier("Failure Prediction")
        else:
            self.fail_predictor = fail_predictor
        self.target_perfs = data_pack.target_perfs
        self.DESIGN_SPACE_DENSITY = DESIGN_SPACE_DENSITY
        self.UNCERTAINTY_THRESHOLD = UNCERTAINTY_DROP_THRESHOLD
        self.params = data_pack.params
        self.classifier_predict_uncertainty_func = classifier_predict_uncertainty_func
        self.X_pool = np.array([])
        self.regressor_accuracy_func = redundancy_regressor_accuracy
        if redundancy_regressor is None:
            self.redundancy_regressor = tf.keras.Sequential([
                tf.keras.layers.Dense(100, activation='relu', input_shape=(len(data_pack.target_perfs)-1,)),
                tf.keras.layers.Dropout(DROPOUT_RATE),
                tf.keras.layers.Dense(50, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            self.redundancy_regressor.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
        else:
            self.redundancy_regressor = redundancy_regressor

        # Generate and scale sobol sequence vectors
        sobol_vec= sobol_seq.i4_sobol_generate(np.sum([isinstance(item, ContinuousDesignBound) for item in self.params]), self.DESIGN_SPACE_DENSITY)
        complete_vec = np.array([])
        sobol_ind = 0

        for i in range(len(data_pack.params)):
            if i == 0:
                if isinstance(data_pack.params[i],ContinuousDesignBound):
                    complete_vec = np.array([sobol_vec[:,0] * (data_pack.params[i].upper_bound - data_pack.params[i].lower_bound) + data_pack.params[i].lower_bound])
                    sobol_ind += 1
                else:
                    complete_vec = np.array([np.random.randint(0, len(data_pack.params[i].categories), (self.DESIGN_SPACE_DENSITY, 1))])
                continue
            if isinstance(data_pack.params[i],ContinuousDesignBound):
                complete_vec = np.append(complete_vec, [sobol_vec[:,sobol_ind] * (data_pack.params[i].upper_bound - data_pack.params[i].lower_bound) + data_pack.params[i].lower_bound], axis=0)
                sobol_ind += 1
            else:
                complete_vec = np.append(complete_vec, [np.random.randint(0, len(data_pack.params[i].categories), (self.DESIGN_SPACE_DENSITY, 1)).flatten()], axis=0)
        complete_vec = np.transpose(complete_vec)
        
        mask = get_valid_func(complete_vec)
        self.X_pool = complete_vec[mask]
        self.complete_vec = self.X_pool
        
        if(X_train is None or Y_train_success is None or Y_train_perfs is None):
            print("Initializing Active Learner with no initial data...")
            self.queryNum = 0
            self.X_train = np.array([])
            self.Y_train_success = np.array([])
            self.Y_train_perfs = np.array([])
        else:
            print("Initializing Active Learner with initial data...")
            # Validating X_train, Y_train_success, and Y_train_perfs
            if(len(X_train) != len(Y_train_success) or len(X_train) != len(Y_train_perfs) or len(Y_train_success) != len(Y_train_perfs)):
                raise ValueError("X_train, Y_train_success, and Y_train_perfs must have the same number of samples.")
            if(len(X_train) == 0 or len(Y_train_success) == 0 or len(Y_train_perfs) == 0):
                raise ValueError("X_train, Y_train_success, and Y_train_perfs must have at least one element.")
            if(len(X_train[0]) != len(self.params)):
                raise ValueError("X_train must have the same number of features as the number of parameters.")
            if(len(Y_train_perfs[0]) != len(self.target_perfs)):
                raise ValueError("Y_train_perfs must have the same number of columns as the number of target performances (", len(self.target_perfs), ").")
            if(np.all(np.isin(Y_train_success, [True, False, 1, 0]))):
                raise ValueError("Y_train_success must contain only boolean or binary values.")
            self.teach(np.array(X_train), np.array(Y_train_success), np.array(Y_train_perfs))
            self.queryNum = 1
            # self.X_train = np.array(X_train)
            # self.Y_train_success = np.array(Y_train_success)
            # self.Y_train_perfs = np.array(Y_train_perfs)
            # self.queryNum = 1
            # self.fail_predictor.fit(self._convert_to_one_hot(X_train), Y_train_success)
            # for i in range(len(self.target_perfs)):
            #     print("Fitting performance validity estimator for", self.target_perfs[i].label, "...")
            #     self.target_perfs[i].regressor.fit(self._convert_to_one_hot(self.X_train), self._convert_to_one_hot(self.Y_train_perfs[:,i]))

    def query(self, batchNum:int, proximity_weight:float=1) -> np.ndarray:
        """
        Queries the Active Learner for a batch of points.
        
        Parameters:
        ----------
        batchNum: int
            The number of points to query.
        proximity_weight: float: default=1
            The weight of the proximity score. Default is 1. If 0, the Active Learner densely sample points of high average uncertainty. If 1, the Active Learner will mostly uniformly sample the design space. If between 0 and 1, the Active Learner will consider a weighted average of the proximity score and the uncertainty of the performance values, and will sample points with a probability distribution accordingly.
        
        Returns:
            Numpy array of the queried points.
        """
        
        if len(self.X_pool) == 0:
            print("No more points in pool. Please reinitialize the ActiveLearner with a greater DESIGN_SPACE_DENSITY.")
            return np.array([])
        if(len(self.X_pool) < batchNum):
            print("Not enough points in pool. Returning all remaining points. To add more points to the pool, please reinitialize the ActiveLearner with a greater DESIGN_SPACE_DENSITY.")
            ret = self.X_pool
            self.X_pool = np.array([])
            return ret
        if(self.queryNum == 0):
            print("Generating initial batch...")
            self.all_stuff = self.X_pool
            ret = self.X_pool[:batchNum]
            self.X_pool = self.X_pool[batchNum:]
            return ret
        print("Fetching Queries for query #", self.queryNum, "...")
        # _, classifier_uncertainty = self.classifier_predict_uncertainty_func(self.fail_predictor, self._convert_to_one_hot(self.X_pool))
        if np.all(self.Y_train_success):
            fail_predictions = np.ones(len(self.X_pool))
        else:
            fail_predictions = self.fail_predictor.predict(self._convert_to_one_hot(self.X_pool)).flatten()
        total_ml_uncertainty = -fail_predictions * np.log2(fail_predictions) - (1 - fail_predictions) * np.log2(1 - fail_predictions)
        total_ml_uncertainty = np.nan_to_num(total_ml_uncertainty)
        total_residuals = 0
        # print("Fail preds: ", fail_predictions)
        # print("Success: ",self.Y_train_success)
        for i in range(len(self.target_perfs)):
            # _, uncertainty = perf.performance_predict_uncertainty_func(perf.estimator, self._convert_to_one_hot(self.X_pool))
            # classifier_uncertainty = np.add(uncertainty, classifier_uncertainty)
            unc, avg_residuals = get_regressor_uncertainty(self.target_perfs[i].regressor, self._convert_to_one_hot(self.X_pool), self._convert_to_one_hot(self.X_calib), self.Y_calib[:, i])
            total_residuals+=(1/avg_residuals)
            total_ml_uncertainty = np.add(unc, total_ml_uncertainty)
        avg_resids = len(self.target_perfs)/total_residuals

        # if self.queryNum == 1:
        #     self.initial_residuals = avg_resids
        # else:
        #     if not self.use_absolute_query_strategy:
        #         proximity_weight = min(proximity_weight * (avg_resids/self.initial_residuals), 1)
        if not self.use_absolute_query_strategy:
            proximity_weight = min(proximity_weight * avg_resids, 1)
        print("AVERAGE MAPE: ", avg_resids)
        
        print("USING PROXIMITY WEIGHT: ", proximity_weight)
        total_ml_uncertainty /= len(self.target_perfs) + 1
        distance_scores, similarity_scores = self.get_similarity_scores(self.X_pool, self.X_train)
        squared_dists = distance_scores**2
        hot_pool = self._convert_to_one_hot_hypercube(self.X_pool)
        

        scores = proximity_weight + (1 - proximity_weight) * total_ml_uncertainty

        scores = scores**(1/proximity_weight) # Change this
        chosen_index = np.random.choice(len(scores), p=scores/np.sum(scores))
        error = total_ml_uncertainty[chosen_index]
        
        if self.use_absolute_query_strategy:
            indexes = [index for index, value in enumerate(total_ml_uncertainty) if error-1 <= value <= error+1]
        else:
            lower_bound = error - 0.2
            upper_bound = error + 0.2
            if lower_bound < 0:
                shift = -lower_bound
                lower_bound += shift
                upper_bound += shift
            elif upper_bound > 1:
                shift = upper_bound - 1
                lower_bound -= shift
                upper_bound -= shift
            indexes = [index for index, value in enumerate(total_ml_uncertainty) if lower_bound <= value <= upper_bound]
        max_index = max(indexes, key=lambda i: squared_dists[i])
            
        
        hot_deleted = self._convert_to_one_hot_hypercube([self.X_pool[max_index]])[0]
        batch = np.array([self.X_pool[max_index]])
        self.X_pool = np.delete(self.X_pool, max_index, axis=0)
        total_ml_uncertainty = np.delete(total_ml_uncertainty, max_index)
        scores = np.delete(scores, max_index)
        # distance_scores = np.delete(distance_scores, max_index)
        similarity_scores = np.delete(similarity_scores, max_index)
        hot_pool = np.delete(hot_pool, max_index, axis=0)
        squared_dists = np.delete(squared_dists, max_index, axis=0)
        # print("hot pool now: ",hot_pool)
        scores = proximity_weight + (1 - proximity_weight) * total_ml_uncertainty

        scores = scores**(1/proximity_weight)

        for i in range(1, batchNum):
            print("Getting query number ",i+1, end="\r")
            dists = np.sum((hot_pool - hot_deleted)**2, axis=1)
            for j in range(len(similarity_scores)):
                # print("Hot Deleted: ",hot_deleted)
                # print("Hot Pool: ",hot_pool[j])
                # dist = self.get_distance_squared(hot_deleted, hot_pool[j])
                if dists[j] < squared_dists[j]:
                    # distance_scores[j] = dist
                    squared_dists[j] = dists[j]
                    similarity_scores[j] = 1 - math.sqrt(dists[j])
            
            chosen_index = np.random.choice(len(scores), p=scores/np.sum(scores))
            error = total_ml_uncertainty[chosen_index]
            
            if self.use_absolute_query_strategy:
                indexes = [index for index, value in enumerate(total_ml_uncertainty) if error-1 <= value <= error+1]
            else:
                lower_bound = error - 0.2
                upper_bound = error + 0.2
                if lower_bound < 0:
                    shift = -lower_bound
                    lower_bound += shift
                    upper_bound += shift
                elif upper_bound > 1:
                    shift = upper_bound - 1
                    lower_bound -= shift
                    upper_bound -= shift
                indexes = [index for index, value in enumerate(total_ml_uncertainty) if lower_bound <= value <= upper_bound]
            max_index = max(indexes, key=lambda i: squared_dists[i])
            
            hot_deleted = self._convert_to_one_hot_hypercube([self.X_pool[max_index]])[0]
            batch = np.append(batch, [self.X_pool[max_index]], axis=0)
            self.X_pool = np.delete(self.X_pool, max_index, axis=0)
            total_ml_uncertainty = np.delete(total_ml_uncertainty, max_index)
            scores = np.delete(scores, max_index)
            # distance_scores = np.delete(distance_scores, max_index)
            squared_dists = np.delete(squared_dists, max_index, axis=0)
            similarity_scores = np.delete(similarity_scores, max_index)
            hot_pool = np.delete(hot_pool, max_index, axis=0)
        print("\nDone querying. Generating visual model...")
        
        pca = PCA(n_components=2)
        pca.fit(X=self.complete_vec)
        # print("pool: ",self.X_pool)
        # print("train: ",self.X_train)
        # print("Batch: ",batch)
        transformed_pool = pca.transform(self.X_pool)
        transformed_training = pca.transform(self.X_train)
        transformed_batch = pca.transform(batch)

        transformed_calib = pca.transform(self.X_calib)
        # print(pd.Series(scores).describe())
        with plt.style.context(plt.style.available[7]):
            plt.figure(figsize=(8, 8))
            plt.scatter(transformed_pool[:, 0], transformed_pool[:, 1], c=total_ml_uncertainty, cmap='viridis', label='unlabeled')
            plt.scatter(transformed_training[:, 0], transformed_training[:, 1], c='r', s=70, label='Labeled for training')
            plt.scatter(transformed_calib[:, 0], transformed_calib[:, 1], s=70, label='Labeled for testing')
            plt.scatter(transformed_batch[:, 0], transformed_batch[:, 1], c='k', s=60, label='queried')
            
            plt.colorbar()
            plt.title('Scores of query #'+str(self.queryNum))
            plt.legend()
        plt.show()
        return batch
        
    def teach(self, X:np.ndarray, Y_success:np.ndarray, Y_perfs:np.ndarray, proximity_weight=0.1) -> np.ndarray:
        """
        Teaches the Active Learner with the queried points.

        Parameters:
        ----------
        X: np.ndarray
            The design points to learn
        Y_success: np.ndarray
            The success values of the design points. Must be a boolean or binary array.
        Y_perfs: np.ndarray
            The performance values of the design points. Must be a 2D array.
        proximity_weight: float: default=0.1
            Used in determining invalid points. Default is 0.1. If 0, the Active Learner will only consider the uncertainty of the performance value regressors. If 1, the Active Learner will only consider the proximity score. If between 0 and 1, the Active Learner will consider a weighted average of the proximity score and the uncertainty of the performance values.
        
        Returns:
            Numpy array of the deleted points.
        """
        if(len(self.X_train) == 0):
            self.X_train = np.array(X)
            self.Y_train_success = np.array(Y_success)
            self.Y_train_perfs = np.array(Y_perfs)
        else:
            self.X_train = np.append(self.X_train, X, axis=0)
            self.Y_train_success = np.append(self.Y_train_success, Y_success, axis=0)
            self.Y_train_perfs = np.append(self.Y_train_perfs, Y_perfs, axis=0)
        if not np.all(self.Y_train_success):
            print("Fitting failure classifier...")
            self.fail_predictor.fit(self._convert_to_one_hot(self.X_train), self.Y_train_success)
        else:
            print("Skipping Failure Classifier fitting due to no failures.")
        # ORIGINAL TEST SPLIT
        # X_train, X_calib, Y_train, Y_calib = train_test_split(self.X_train, self.Y_train_perfs, test_size=0.2, random_state=MEANING_OF_LIFE)
        
        # NEW TEST SPLIT METHOD
        num_test = len(self.X_train) // 5

        ind = np.random.choice(len(self.X_train), 1)
        X_calib = self.X_train[ind]
        X_train = np.delete(self.X_train, ind, axis=0)
        Y_calib = self.Y_train_perfs[ind]
        Y_train = np.delete(self.Y_train_perfs, ind, axis=0)
        # print("X_train: ",X_train)
        # print("X_calib: ",X_calib)
        for i in range(num_test-1):
            # print(i)
            dists, _ = self.get_similarity_scores(X_train, X_calib)
            ind = np.argmax(dists)
            # print("Ind deleted: ",ind)
            X_calib = np.append(X_calib, [X_train[ind]], axis=0)
            X_train = np.delete(X_train, ind, axis=0)
            Y_calib = np.append(Y_calib, [Y_train[ind]], axis=0)
            Y_train = np.delete(Y_train, ind, axis=0)
        # END NEW SPLIT METHOD


        # print("X_calib", X_calib)
        # print("Y_calib", Y_calib)
        # print("X_train", X_train)
        # print("Y_train", Y_train)

        self.X_calib = X_calib
        self.Y_calib = Y_calib
        X_train_hots = self._convert_to_one_hot(X_train)
        self_X_train_hots = self._convert_to_one_hot(self.X_train)
        for i in range(len(self.target_perfs)):
            print("Fitting performance regressor and validity classifier for", self.target_perfs[i].label, "...")
            # print("Valids: ",self.target_perfs[i].validity_checker(self.Y_train_perfs[:,i]))
            self.target_perfs[i].regressor.fit(X_train_hots, Y_train[:,i])
            # self.target_perfs[i].classifier.fit(self_X_train_hots, self.target_perfs[i].validity_checker(self.Y_train_perfs[:,i]))
            if not np.all(self.target_perfs[i].validity_checker(self.Y_train_perfs[:,i])):
                self.target_perfs[i].classifier.fit(self_X_train_hots, self.target_perfs[i].validity_checker(self.Y_train_perfs[:,i]))
            
            
            # p, _ = self.target_perfs[i].performance_predict_uncertainty_func(self.target_perfs[i].estimator, self._convert_to_one_hot(self.X_train))
            # print("truths:",self.target_perfs[i].validity_checker(self.Y_train_perfs[:,i]))
        print("-"*70)
        print("Getting uncertainities for invalidity classifier...")
        # Deleting points predicted to fail or have invalid performance values
        dissimilarity_scores, _ = self.get_similarity_scores(self.X_pool, self.X_train) # Dis Scores is the distances
        # fail_predictions, classifier_uncertainty = self.classifier_predict_uncertainty_func(self.fail_predictor, self._convert_to_one_hot(self.X_pool))
        if np.all(self.Y_train_success):
            fail_predictions = np.ones(len(self.X_pool))
        else:
            fail_predictions = self.fail_predictor.predict(self._convert_to_one_hot(self.X_pool)).flatten()
        # classifier_uncertainty = 1-abs(2*fail_predictions-1)
        # print("Similiarity Scores: ", similarity_scores)
        # uncertainty_scores = proximity_weight * (1 - similarity_scores) + (1 - proximity_weight) * classifier_uncertainty
        uncertainty_scores = proximity_weight * dissimilarity_scores
        # print("unc sc",uncertainty_scores)
        # print("sim sc",similarity_scores)
        # print("class unc",classifier_uncertainty)
        certainty_scores = 1 - uncertainty_scores
        # cutted = np.copy(fail_predictions)
        # cutted[cutted>0.5] = 1
        # total_invalidity = cutted.flatten()**certainty_scores
        total_invalidity = fail_predictions.flatten()**certainty_scores
        stack = [total_invalidity]
        # print("predictions: ",fail_predictions)
        for perf in self.target_perfs:
            print("Computing uncertainities for ", perf.label, "...")
            # _, uncertainty = perf.performance_predict_uncertainty_func(perf.estimator, self._convert_to_one_hot(self.X_pool))
            if not np.all(self.target_perfs[i].validity_checker(self.Y_train_perfs[:,i])):
                preds = perf.classifier.predict(self._convert_to_one_hot(self.X_pool)).flatten()
            else:
                preds = np.ones(len(self.X_pool))
            # uncertainty = 1-abs(2*preds-1)
            # No need for classifier uncertainty since it's already biased towards valid
            # print("Preds for ", perf.label, ": ", preds)
            # print("minimum pred: ", np.min(preds))
            # uncertainty_scores = proximity_weight * (1 - similarity_scores) + (1 - proximity_weight) * uncertainty
            # certainty_scores = 1 - uncertainty_scores
            # print("Fail Predictions: ", predictions)
            # print("Certainty Scores for ", perf.label, ": ", certainty_scores)
            
            # cutted = np.copy(fail_predictions)
            # cutted[cutted>0.5] = 1
            # invalidity = cutted.flatten()**certainty_scores

            # invalidity = cutted.flatten()**certainty_scores
            invalidity = preds.flatten()**certainty_scores
            # print(1)
            # print("Invalidity for ", perf.label, ": ", invalidity)
            total_invalidity = total_invalidity*(invalidity)
            # print(2, len(stack[0]))
            stack = np.append(stack, [invalidity], axis=0)
            # print(3)
        print("-"*70)
        # Mask is True for points that are predicted to fail or be outside thresholds and have low uncertainty
        mask = (total_invalidity<self.UNCERTAINTY_THRESHOLD)  # Assumes 0 is invalid
        # print("MIN CERTAINTY SCORE: ", np.min(total_invalidity))
        stack = np.transpose(stack)

        # print("Point with min validity score score: ", self.X_pool[np.argmin(total_invalidity)])
        # print("min invalidity score: ", np.min(total_invalidity))
        # print("stack: ",stack)
        
        initialLen = len(self.X_pool)
        deleted = stack[mask]
        deletedPoints = self.X_pool[mask]
        self.X_pool = self.X_pool[~mask]
        finalLen = len(self.X_pool)

        uncertain_perfs = np.array([np.argmin(arr) for arr in deleted])
        unique, counts = np.unique(uncertain_perfs, return_counts=True)
        perfsdict = dict(zip(unique, counts))
        # print("PerfsDict: ",perfsdict)
        print("Deleted", initialLen-finalLen, "points from pool due to certain invalidity.")
        if 0 in perfsdict:
            print("    Deleted", perfsdict[0], "points from pool due to certain invalidity for failure prediction.")
        for i in range(len(self.target_perfs)):
            if (i+1) in perfsdict:    
                print("    Deleted", perfsdict[i+1], "points from pool due to certain invalidity for", self.target_perfs[i].label,".")
        print("Total remining points in the pool/design space: ", len(self.X_pool))
        print('-'*70)
        if not self.skip_redundancy:
            for i in range(len(self.target_perfs)):
                print("Fitting redundancy regressor for", self.target_perfs[i].label, "...")
                accuracy = self.regressor_accuracy_func(self.redundancy_regressor, np.delete(self.Y_train_perfs, i, axis=1), self.Y_train_perfs[:, i])
                if(accuracy>REDUNDANCY_THRESHOLD):
                    print('\033[1m'+'DETECTED REDUNDANT PERFORMANCE VALUE'+'\033[0m'+': Accuracy for predicting ', self.target_perfs[i].label, " from other performance values is ", accuracy)
        self.queryNum += 1
        return deletedPoints

    def get_pool_size(self):
        return len(self.X_pool)

    def get_similarity_scores(self, X:np.ndarray, X_train:np.ndarray):
        # dists = pairwise_distances(self._convert_to_reduced_one_hot(X), self._convert_to_reduced_one_hot(X_train), metric='euclidean').min(axis=1)
        # return dists, 1 / (1 + pairwise_distances(self._convert_to_reduced_one_hot(X), self._convert_to_reduced_one_hot(X_train), metric='euclidean').min(axis=1))
        dists = pairwise_distances(self._convert_to_one_hot_hypercube(X), self._convert_to_one_hot_hypercube(X_train), metric='euclidean').min(axis=1)
        return dists, 1-dists

    def get_overall_accuracy(self, true_vals):
        """
        To be used only for testing the true accuracy of the ActiveLearner regressors.
        """
        mapes = []
        all = []
        for i in range(len(self.target_perfs)):
            # We get mean percentage error for each performance value
            print("Getting TRUE overall accuracy for ", self.target_perfs[i].label, "...")
            reg = GluonRegressor(self.target_perfs[i].label)
            reg.fit(self._convert_to_one_hot(self.X_train), self.Y_train_perfs[:, i])
            preds = reg.predict(self._convert_to_one_hot(self.all_stuff))
            mapes.append(np.mean(np.abs((true_vals[:, i] - preds)/(true_vals[:, i]+0.0000001))))
            all.append((self.target_perfs[i].label+" Mape", mapes[-1]))
        # Return the harmonic mean of the mapes
        all.append(("Average Mapes", len(mapes)/np.sum(1.0/np.array(mapes))))
        return all