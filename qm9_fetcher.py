import torch.utils.data as data
import pickle as p
import numpy as np

# for random rotation. cen_pcs is the point cloud
def Rot(mol):
    # create euclidean axes
    euc_ax = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])

    x = np.random.randn(3)
    x = x/np.linalg.norm(x)
    # create a random orthogonal y unit vector 
    y = np.random.randn(3)
    y = y - (y.dot(x) * x) 
    y = y/np.linalg.norm(y)
    # create the z unit vector from cross product
    z = np.cross(x,y)
    # stack it all to get a new set of axes to represent the molecule in
    ran_ax = np.vstack((x,y,z))
    #print(ran_ax[0].dot(ran_ax[1])) # this should be [0.,0.,0.] or very close
    #  create rotation matrix (https://math.stackexchange.com/questions/2004800/math-for-simple-3d-coordinate-rotation-python)
    rot_mx = np.array([[ran_ax[0].dot(euc_ax[0]),ran_ax[0].dot(euc_ax[1]),ran_ax[0].dot(euc_ax[2])],
                        [ran_ax[1].dot(euc_ax[0]),ran_ax[1].dot(euc_ax[1]),ran_ax[1].dot(euc_ax[2])],
                        [ran_ax[2].dot(euc_ax[0]),ran_ax[2].dot(euc_ax[1]),ran_ax[2].dot(euc_ax[2])]])
    # create rotated coordinates
    return np.matmul(mol, rot_mx.transpose())

class Qm9Dataset(data.Dataset):
    PATH = '/share/jolivaunc/data/qm9/class_qm9.p'
    def __init__(
        self, data_type='train', noise_std=0.01,
    ):
        dataset = p.load(open(Qm9Dataset.PATH, 'rb'))

        self.data_type = data_type
        self._data = dataset[self.data_type]
        self._labels = dataset[f'{self.data_type}_labels']
        self._noise_std = noise_std
        self._trans = lambda d: Rot(d) + self._noise_std * np.random.rand(*d.shape)

    def __getitem__(self, index):
        if self.data_type == 'train':
            return (
                self._trans(self._data[index]), self._labels[index]
            )
        else:
            return self._data[index], self._labels[index]

    def __len__(self):
        return len(self._data)

class Qm9RotInvDataset(data.Dataset):
    PATH = '/share/jolivaunc/data/qm9/class_qm9_rot_inv.pkl'
    def __init__(
        self, data_type='train', noise_std=0.01,
    ):
        with open(Qm9RotInvDataset.PATH, 'rb') as f:
            dataset = p.load(f)

        self.data_type = data_type
        self._data = dataset[self.data_type]
        self._labels = dataset[f'{self.data_type}_labels']

    def __getitem__(self, index):
        return self._data[index], self._labels[index]

    def __len__(self):
        return len(self._data)