import numpy as np

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

def get_regressor_uncertainty(regressor, X_pool:np.ndarray, X_test:np.ndarray, y_test:np.ndarray):
    """
    Returns uncertainty (between 0 and 1, where 0 is certain and 1 is uncertain)
    """
    models = regressor.get_model_names()
    totals = []
    for model in models:
        totals.append(np.array(regressor.predict(X_pool, model=model)))
    totals = np.array(totals)
    totals = totals.transpose()

    variances = np.var(totals, axis=1)
    variances = variances-np.min(variances)
    variances = variances/np.max(variances)
    return variances


    # return np.zeros(len(X_pool)) # Temporary