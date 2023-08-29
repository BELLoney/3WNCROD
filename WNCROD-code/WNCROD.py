# Three-way neighborhood characteristic region-based outlier detection (3WIROD) algorithm
# Please refer to the following papers:
# Zhang xianyong,Yuan Zhong, and Miao Duoqian.Outlier Detection Using Three-Way
# Neighborhood Characteristic Regions and Corresponding Fusion Measurement[J].TKDE,2023.
# Uploaded by Yuan Zhong on August 29, 2023. E-mail:yuanzhong2799@foxmail.com.
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist


def WNCROD(data, X_tem, lammda):
    # input:
    # data is data matrix without decisions, where rows for samples and columns for attributes.
    # Numerical attributes should be normalized into [0,1].
    # Nominal attributes be replaced by different integer values.
    # X_tem denotes the selected condition subdata.
    # lammda is a given parameter for the radius adjustment.
    # output
    # Multiple neighborhood outlier factor (MNOF)
    n, m = data.shape
    X = np.zeros(n)
    X[X_tem] = 1

    D1 = m / 3
    D2 = m / 2
    D3 = 0.9 * m

    delta = np.zeros(m)
    ID = np.all(data <= 1, axis=0)
    delta[ID] = np.std(data[:, ID], axis=0) / lammda

    Lower = np.zeros((m, n))

    for col in range(m):
        RM_tem = cdist(data[:, [col]], data[:, [col]], metric='cityblock') <= delta[col]
        Lower_temp = np.min(np.maximum(1 - RM_tem, np.tile(X, (n, 1))), axis=1)
        Lower[col, :Lower_temp.shape[0]] = Lower_temp

    IB = np.tile(X, (m, 1)) - Lower
    NEB = np.min(IB, axis=0)
    NPB = IB - np.tile(NEB, (m, 1))

    n_X = int(sum(X))

    weight = np.zeros((n_X, m))

    for col in range(m):
        RM_tem = cdist(data[:, [col]], data[:, [col]], metric='cityblock') <= delta[col]
        weight_x = []

        for i in range(n_X):
            temp1 = RM_tem[X_tem[i], :]
            weight_temp = 1 - (np.sqrt((np.sum(np.minimum(temp1, X))) / n_X))
            weight_x.append(weight_temp)

        weight[:len(weight_x), col] = weight_x

    D_tem = np.zeros((n, n))

    for col in range(m):
        RM_tem = cdist(data[:, [col]], data[:, [col]], metric='cityblock') <= delta[col]
        D_tem += RM_tem

    NOM = m - D_tem
    X_OM = NOM[np.ix_(X_tem, X_tem)]

    NEB_num = np.zeros((n_X, m))
    Lower_num = np.zeros((n_X, m))
    NPB_num = np.zeros((n_X, m))

    for col in range(m):
        temp2 = Lower[col, :]
        temp3 = NPB[col, :]

        for i in range(n_X):
            temp1 = X_OM[i, :]
            NEB_num[i, col] = np.sum(np.minimum(NEB[X_tem], temp1 <= D1))
            NPB_num[i, col] = np.sum(np.minimum(temp3[X_tem], temp1 >= D2))
            Lower_num[i, col] = np.sum(np.minimum(temp2[X_tem], temp1 >= D3))

    MNOF = np.mean(((NEB_num + Lower_num + NPB_num) / n_X) * weight, axis=1)
    return MNOF


if __name__ == "__main__":
    load_data = loadmat('Example.mat')
    trandata = load_data['Example']
    scaler = MinMaxScaler()
    trandata[:, 1:] = scaler.fit_transform(trandata[:, 1:])

    X_tem = [0, 1, 4, 5]
    lammda = 1
    out_scores = WNCROD(trandata, X_tem, lammda)
    print(out_scores)
