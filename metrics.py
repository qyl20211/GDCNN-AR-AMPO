import numpy as np
def tilted_loss_valid(y_true, y_pred, quantile):
    e = y_true - y_pred
    return np.mean(np.maximum(quantile * e, (quantile - 1) * e), axis=0)
def evaluate_PICP_WS_PL(y_pred, y_test, quantiles):
    PICP, WS, QS ,MPIW= 0.0, 0.0, 0.0,0.0
    upper_bound, lower_bound = quantiles[0], quantiles[1]
    confidence = 1 - (upper_bound - lower_bound)
    y_pred_upper = np.reshape(y_pred[:, 1], (y_pred.shape[0], 1))
    y_pred_lower = np.reshape(y_pred[:, 0], (y_pred.shape[0], 1))
    # Find out of bound indices for WS
    idx_oobl = np.where((y_test < y_pred_lower) > 0)
    #  print(idx_oobl[0])
    idx_oobu = np.where((y_test > y_pred_upper) > 0)
    # add to sum
    PICP += (np.sum((y_test > y_pred_lower) & (y_test <= y_pred_upper)) \
             / len(y_test) * 100)
    WS += (np.sum(np.sum(y_pred_upper - y_pred_lower) +
                  np.sum(2 * (y_pred_lower[idx_oobl[0]] -
                              y_test[idx_oobl[0]]) / confidence) +
                  np.sum(2 * (y_test[idx_oobu[0]] -
                              y_pred_upper[idx_oobu[0]]) / confidence)) \
           / len(y_test))
    QS += (tilted_loss_valid(y_test, y_pred_lower, lower_bound) +
           tilted_loss_valid(y_test, y_pred_upper, upper_bound))
    MPIW += np.sum(y_pred_upper - y_pred_lower) / len(y_pred_upper)
    return PICP, WS, QS,MPIW
