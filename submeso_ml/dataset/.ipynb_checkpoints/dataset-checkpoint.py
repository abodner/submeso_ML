from torch.utils.data import Dataset
import numpy as np
import random

class SubmesoDataset(Dataset):
    def __init__(self,input_features = ['grad_B','FCOR', 'Nsquared', 'HML', 'TAU',
              'Q', 'HBL', 'div', 'vort', 'strain'],seed=123,train_split=0.9):
        super().__init__()
        self.seed=seed
        self.train_split=train_split
        self.input_features=input_features
        self.path= '/scratch/ab10313/pleiades/NN_data_smooth/'
        
        # load features for input
        in_features=[]
        for input_feature in self.input_features:
            in_features.append(np.load(self.path+'%s.npy' % input_feature))
        # x input
        self.x = np.stack(in_features,axis=1)
        
        # load output
        WB_sg = np.load(self.path+'WB_sg.npy')
        
        # y output
        self.y = np.tile(WB_sg,(1,1,1,1)).reshape(WB_sg.shape[0],1,WB_sg.shape[1],WB_sg.shape[2]) 
        
        self._get_split_indices()
        self._norm_factors()
        
    def _get_split_indices(self):
        """ obtain a set of train and test indices """
        
        # randomnly generate train, test and validation time indecies 
        time_ind = self.x.shape[0]
        rand_ind = np.arange(time_ind)
        rand_seed = self.seed
        random.Random(rand_seed).shuffle(rand_ind)
        self.train_ind, self.test_ind =  rand_ind[:round(self.train_split*time_ind)], rand_ind[round((self.train_split)*time_ind):]                                                                        

        # sort test_ind
        self.train_ind = np.sort(self.train_ind)
        
    def _norm_factors(self):
        """ keep track of region noramlization mean and std"""
        self.y_mean = np.load(self.path+'WB_sg_mean.npy')
        self.y_std = np.load(self.path+'WB_sg_std.npy')
        
    def __getitem__(self,idx):
        return (self.x[idx],self.y[idx])
    