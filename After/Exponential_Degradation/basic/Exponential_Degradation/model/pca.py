import sklearn
import pandas as pd

from sklearn.decomposition import PCA


def pca_train_df(data, feats, num):

    pca = PCA(n_components= num)

    pca_data = pca.fit_transform(data[feats])
    pca_df = pd.DataFrame(pca_data, columns = ['pc1', 'pc2', 'pc3'])
    pca_df['UnitNumber'] = data.UnitNumber.values
    pca_df['cycle'] = pca_df.groupby('UnitNumber').cumcount()+1
    pca_df['RUL'] = pca_df.groupby('UnitNumber').cycle.transform('max') - pca_df.cycle

    return pca_df

def pca_test_df(data, feats, num):
    pca = PCA(n_components= num)

    pca_data = pca.fit_transform(data[feats])
    pca_df = pd.DataFrame(pca_data, columns = ['pc1', 'pc2', 'pc3'])
    pca_df['UnitNumber'] = data.UnitNumber.values
    pca_df['cycle'] = pca_df.groupby('UnitNumber').cumcount()+1

    return pca_df    

def get_threshold(pca_df):
    pcs = ['pc1', 'pc2', 'pc3']

    threshold = pca_df.pc1[pca_df.RUL == 0].mean()

    return threshold

def get_threshold_std(pca_df):

    threshold_std = pca_df.pc1[pca_df.RUL == 0].std()

    return threshold_std