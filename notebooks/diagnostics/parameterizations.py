{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pyqg\n",
    "import torch\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "class MLE():\n",
    "    \"\"\" Fox-Kemper et al 2008 MLE parameterization computed from data\"\"\"\n",
    "    def __init__(self,):\n",
    "        \n",
    "        WB_FK = np.load(PATH_NN+'WB_FK.npy')\n",
    "        WB_FK_test = WB_FK[test_ind]\n",
    "        \n",
    "        # compare with FK_Lf\n",
    "        normalization factors\n",
    "\n",
    "        # nrmaliation factors: mean and std \n",
    "        WB_sg_mean_test = WB_sg_mean[test_ind]\n",
    "        WB_sg_std_test = WB_sg_std[test_ind] \n",
    "\n",
    "    def FoxKemper2011():\n",
    "    \"\"\" Fox-Kemper et al 2011 MLE parameterization with Lf deformation radius computed from data\"\"\"\n",
    "        Lf_FK11 = np.load(PATH_NN+'Lf_FK11.npy')\n",
    "        Lf_FK11_test = Lf_FK11[test_ind]\n",
    "        \n",
    "    def Bodner2023():\n",
    "    \"\"\" Bodner et al 2023 MLE parameterization with Lf from TTW computed from data\"\"\"\n",
    "        # compare with Bod_Lf\n",
    "        Lf_BD23 = np.load(PATH_NN+'Lf_BD23.npy')\n",
    "        Lf_BD23_test = Lf_BD23[test_ind]\n",
    "   \n",
    "        \n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "my_env",
   "language": "python",
   "name": "my_env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
