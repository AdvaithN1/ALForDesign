import numpy as np
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def classifier_predict_simple_uncertainty(fail_predictor, X_pool:np.ndarray):
    """
    Returns validity predictions and uncertainty (between 0 and 1, where 0 is certain and 1 is uncertain)
    """
    predictions = fail_predictor.predict(X_pool)
    # return predictions.flatten(), 1-2*np.absolute(predictions.flatten()-0.5)
    return predictions, min(1,2*predictions)

def fit(model, X:np.ndarray, Y:np.ndarray):
    if(len(X)<32):
        model.fit(X, Y, epochs=10, batch_size=len(X), verbose=0)
    else:
        model.fit(X, Y, epochs=10, batch_size=32, verbose=0)        

def rsquared(y_true:np.ndarray, y_pred:np.ndarray):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / (ss_tot+0.0000001))

def get_regressor_uncertainty(regressor, X_pool:np.ndarray, X_calib:np.ndarray, y_calib:np.ndarray, alpha=0.05):
    """
    Returns uncertainty (between 0 and 1, where 0 is certain and 1 is uncertain), MAPE
    """
    calib_predictions = regressor.predict(X_calib)
    # print("PREDICTED:", calib_predictions)
    # print("ACTUAL:", y_calib)
    # print("Calib predictions:", calib_predictions)
    # print("Calib y:", y_calib)
    calib_residuals = np.abs(calib_predictions - y_calib)
    mape = np.mean(calib_residuals / (np.abs(y_calib)+0.0000001))
    resid_predictor = KNeighborsRegressor(n_neighbors=3, weights='distance')
    # resid_predictor = LinearRegression()

    # OLD METHOD OF NORMALIZATION
    # normalized_residuals = (calib_residuals - np.min(calib_residuals)) / (np.max(calib_residuals) - np.min(calib_residuals))
    

    normalized_residuals = (calib_residuals) / (np.max(calib_residuals))

    
    resid_predictor.fit(X_calib, normalized_residuals)

    return np.array(resid_predictor.predict(X_pool)).flatten(), mape

    # models = regressor.get_model_names()
    # totals = []
    # for model in models:
    #     totals.append(np.array(regressor.predict(X_pool, model=model)))
    # totals = np.array(totals)
    # totals = totals.transpose()

    # variances = np.var(totals, axis=1)
    # variances = variances-np.min(variances)
    # variances = variances/np.max(variances)
    # return variances

    # return np.zeros(len(X_pool)) # Temporary

def get_bayesian_uncertainty(X_pool:np.ndarray, X_train:np.ndarray, Y_train:np.ndarray):
    gp = GaussianProcessRegressor(n_restarts_optimizer=10, random_state=42)
    gp.fit(X_train, Y_train)
    _, stds = gp.predict(X_pool, return_std=True)
    normalized_stds = stds/np.max(stds)
    print("Normalized STDs:", normalized_stds)
    print("Mean STD:", np.mean(stds))
    return np.array(normalized_stds).flatten(), np.mean(normalized_stds)