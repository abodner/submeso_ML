import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


class RegressionSystem(LightningModule):
    """ Class to store core model methods """
    
    def __init__(self,network,lr=0.001,wd=0.01):
        super().__init__()
        self.network=network
        self.criterion=nn.MSELoss()
        self.lr=lr
        self.wd=wd
        
       # self.logger = WandbLogger()

    def forward(self,x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.lr,weight_decay=self.wd)
        return optimizer

    def step(self,batch,kind):
        x_data, y_data = batch
        
        #loss
        loss = self.criterion(self(x_data), y_data)
        self.log(f"loss", loss, on_step=False, on_epoch=True)
        #r2
        r2 = r2_score(self(x_data).detach().cpu().numpy(), y_data.detach().cpu().numpy())
        self.log(f"R-squared", r2, on_step=False, on_epoch=True)
        
        #correlation coefficient
        corr, _ = pearsonr(self(x_data).detach().cpu().numpy().flatten(), y_data.detach().cpu().numpy().flatten())
        self.log(f"correlation", corr, on_step=False, on_epoch=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch,"train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch,"valid")