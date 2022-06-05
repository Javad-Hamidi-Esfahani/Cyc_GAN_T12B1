import imp
import torch
from torch.nn import L1Loss, MSELoss
import nibabel as nib
import numpy as np


class L1_loss(torch.nn.Module):
    def __init__(self):
        super(L1_loss, self).__init__()
        self.instance = L1Loss()

    def forward(self, target, output):
        l = self.instance(output[:,0,:,:], target[:,0,:,:])
        return l


def save_input(inputs, comment='input'):
    inputs = inputs.detach().cpu().numpy()
    N1,N2,N3,N4,N5 = inputs.shape
    inputs = np.reshape(inputs[0], (N3,N4,N5,N2))
    affine = np.array([[ 8.49883914e-01,  7.44797662e-03, -1.22618685e-02,
        -8.95130615e+01],
       [-9.09765251e-03,  8.57123375e-01, -1.37016967e-01,
        -9.84052505e+01],
       [ 1.07044466e-02,  1.37129501e-01,  8.57086062e-01,
        -8.96880875e+01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]])

    img = nib.Nifti1Image(inputs, affine)
    nib.save(img, '/scratch/javadhe/Result_Test/Cyc_GAN/_10/{}.nii.gz'.format(comment))
    return
