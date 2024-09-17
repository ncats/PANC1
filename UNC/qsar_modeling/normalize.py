import numpy as np
from scipy.stats import pearsonr

class Normalizer:

    def __init__(self):
        print("Normalizer made")
        pass

    def fit(self, data, scores, eps = 0.001, corr_threshold = 0.95):

        self.mean = np.mean(data, axis = 0)
        self.std = np.std(data, axis = 0)

        #remove low variance columns
        keep_columns = np.where(self.std > eps, True, False)

        #remove highly correlated columns
        _, _, removed = remove_correlated_columns(data)
        for val in removed:
            keep_columns[val] = False

        '''
        for i in range(data.shape[1]):
            if keep_columns[i] == False:
                continue
            for j in range(data.shape[1]):
                if keep_columns[j] == False:
                    continue
                if i != j:
                    a = data[:,i]
                    b = data[:,j]
                    corr = abs(pearsonr(a,b)[0])
                    if corr > corr_threshold:
                        keep_columns[j] = False
        '''

        #remove low correlations with predictor
        #TODO: vectorize
        for i in range(data.shape[1]):
            if keep_columns[i]:
                col = data[:,i]
                corr = abs(pearsonr(col, scores)[0])
                if corr < eps:
                    keep_columns[i] = False

        self.keep_columns = keep_columns
        self.mean = self.mean[keep_columns]
        self.std = self.std[keep_columns]

    def transform(self, data):

        print(dir(self))
        #data = data[:, self.passed]
        data = data[:, self.keep_columns]
        t = (data - self.mean) / self.std

        return t

def remove_constant_columns(matrix, threshold = 0.0001):

    std = np.std(matrix, axis = 0)
    passed = std > threshold
    removed = std <= threshold

    junk_matrix = matrix[:,list(removed)]
    matrix = matrix[:,passed]

    passed = list(np.argwhere(passed == True).flat)
    removed = list(np.argwhere(removed == True).flat)

    return matrix, passed, removed


def remove_correlated_columns(matrix, cc_threshold = 0.95):

    corr = np.corrcoef(matrix, rowvar = False)

    to_remove = set()
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if i == j:
                continue
            if i not in to_remove and j not in to_remove:
                if corr[i,j] > cc_threshold:
                    to_remove.add(i)

    to_keep = set(list(range(corr.shape[0]))) - to_remove
    to_keep = list(to_keep)

    matrix = matrix[:,to_keep]

    passed = to_keep
    removed = to_remove
    return matrix, passed, removed
