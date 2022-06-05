import os
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import nibabel as nib
from scipy import ndimage


def pad_array(T1, B1, pad):
    pad_np = ((pad[0]//2, int(np.ceil(pad[0]/2))), (pad[1]//2, int(np.ceil(pad[1]/2))), (pad[2]//2, int(np.ceil(pad[2]/2))))
    T1 = np.pad(T1, pad_np)
    B1 = np.pad(B1, pad_np)
    return T1, B1


def crop_3D(input, mask, image_size):
    cropped_inputs_T1 = []
    cropped_masks = []
    
    # min0, min1, min2 = (np.ceil(image_size[0]/2), np.ceil(image_size[1]/2), np.ceil(image_size[2]/2))
    #N0, N1, N2 = ( np.array(input.shape) / np.array(image_size) ).astype('int')
    N0 = np.arange(0, input.shape[0], image_size[0])
    N1 = np.arange(0, input.shape[1], image_size[1])
    N2 = np.arange(0, input.shape[2], image_size[2])
    for i in range(len(N0) - 1):
        for j in range(len(N1) - 1):
            for k in range(len(N2) - 1):
                cropped_inputs_T1.append(input[N0[i]:N0[i+1], N1[j]:N1[j+1], N2[k]:N2[k+1]][np.newaxis])
                cropped_masks.append(mask[N0[i]:N0[i+1], N1[j]:N1[j+1], N2[k]:N2[k+1]][np.newaxis])

    return np.vstack(cropped_inputs_T1), np.vstack(cropped_masks)


def normalize_data(input, mask, mean=0, std=0.225):
    # input = np.vstack([input[np.newaxis], input[np.newaxis]])
    
    if len(input.shape)==3:
        input = input[np.newaxis]
        mask = mask[np.newaxis]

    for i in range(input.shape[0]):
        input[i,:,:,:] = (input[i,:,:,:] - np.mean(input[i,:,:,:])) / (np.max(input[i,:,:,:]) - np.mean(input[i,:,:,:]))
        mask[i,:,:,:] = (mask[i,:,:,:] - np.mean(mask[i,:,:,:])) / (np.max(mask[i,:,:,:]) - np.mean(mask[i,:,:,:]))

    input = torch.from_numpy(input)
    mask = torch.from_numpy(mask)

    mask = mask / (mask.max() + 1e-3)

    return input, mask


class Pair_T1B1_Dataset(data.Dataset):
    def __init__(self, dir_dataset, image_size, n_train, pad=[0,0,0], degrade=False, perspective=False, crop=False,
                mask_roi=True, zoom=[0,0,0]):
        super(Pair_T1B1_Dataset, self).__init__()

        self.degrade = degrade
        self.perspective = perspective
        self.crop = crop
        self.image_size = image_size
        self.pad = pad
        self.mask_roi = mask_roi
        self.zoom = zoom
        
        self.dir_T1_nifti = []
        self.dir_B1_nifti = []
        self.dir_csf_nifti = []
        self.dir_gm_nifti = []
        self.dir_wm_nifti = []
        self.dir_brain_nifti = []
        for n in n_train:
            name = 'Ab300_{:03d}'.format(n)
            if os.path.isfile(join(join(dir_dataset,name), 'T1_biascorr_brain.nii.gz')):
                self.dir_T1_nifti.append(join(join(dir_dataset,name), 'T1_biascorr_brain.nii.gz'))
            if os.path.isfile(join(join(dir_dataset,name), 'B1map_T1space_brain.nii.gz')):
                self.dir_B1_nifti.append(join(join(dir_dataset,name), 'B1map_T1space_brain.nii.gz'))
            if os.path.isfile(join(join(dir_dataset,name), 'T1_fast_pve_0.nii.gz')):
                self.dir_csf_nifti.append(join(join(dir_dataset,name), 'T1_fast_pve_0.nii.gz'))
            if os.path.isfile(join(join(dir_dataset,name), 'T1_fast_pve_1_0p9.nii.gz')):
                self.dir_gm_nifti.append(join(join(dir_dataset,name), 'T1_fast_pve_1_0p9.nii.gz'))
            if os.path.isfile(join(join(dir_dataset,name), 'T1_fast_pve_2_0p9.nii.gz')):
                self.dir_wm_nifti.append(join(join(dir_dataset,name), 'T1_fast_pve_2_0p9.nii.gz'))
            if os.path.isfile(join(join(dir_dataset,name), 'T1_biascorr_brain_mask2.nii.gz')):
                self.dir_brain_nifti.append(join(join(dir_dataset,name), 'T1_biascorr_brain_mask2.nii.gz'))

    def __getitem__(self, index):
        T1 = nib.load(self.dir_T1_nifti[index]).get_fdata()
        B1 = nib.load(self.dir_B1_nifti[index]).get_fdata()
        brain = nib.load(self.dir_brain_nifti[index]).get_fdata()
        if self.mask_roi:
            csf = nib.load(self.dir_csf_nifti[index]).get_fdata()
            wm = nib.load(self.dir_wm_nifti[index]).get_fdata()
            gm = nib.load(self.dir_gm_nifti[index]).get_fdata()

        if np.sum(self.pad)>0:
            T1, B1 = pad_array(T1, B1, self.pad)
            brain, _ = pad_array(brain, brain, self.pad)
            if self.mask_roi:
                csf, _ = pad_array(csf, csf, self.pad)
                wm, _ = pad_array(wm, wm, self.pad)
                gm, _ = pad_array(gm, gm, self.pad)

        if np.sum(self.zoom)>0:
            T1 = ndimage.zoom(T1, self.zoom)
            B1 = ndimage.zoom(B1, self.zoom)
            brain = ndimage.zoom(brain, self.zoom)
            if self.mask_roi:
                csf = ndimage.zoom(csf, self.zoom)
                wm = ndimage.zoom(wm, self.zoom)
                gm = ndimage.zoom(gm, self.zoom)

        if self.crop:
            T1, B1 = crop_3D(T1, B1, self.image_size)
            if self.mask_roi:
                csf, _ = crop_3D(csf, csf, self.image_size)
                wm, _ = crop_3D(wm, wm, self.image_size)
                gm, _ = crop_3D(gm, gm, self.image_size)


        if self.degrade:
            # T1 = degrade(T1)
            pass

        if self.perspective:
            k = np.random.choice(3,3, replace=False)
            T1 = np.transpose(T1, (k[0], k[1], k[2]))
            B1 = np.transpose(B1, (k[0], k[1], k[2]))
            if self.mask_roi:
                csf = np.transpose(csf, (k[0], k[1], k[2]))
                wm = np.transpose(wm, (k[0], k[1], k[2]))
                gm = np.transpose(gm, (k[0], k[1], k[2]))

        brain = torch.unsqueeze(torch.from_numpy(brain), 0)

        T1, B1 = normalize_data(T1, B1)
        T1 = T1 * brain
        B1 = B1 * brain

        if self.mask_roi:
            csf = torch.unsqueeze(torch.from_numpy(csf), 0)
            wm = torch.unsqueeze(torch.from_numpy(wm), 0)
            gm = torch.unsqueeze(torch.from_numpy(gm), 0)
            inputs_T1 = torch.cat((T1, csf, wm, gm), 0)
            outputs_B1 = torch.cat((B1, csf, wm, gm), 0)
        else:
            inputs_T1 = T1
            outputs_B1 = B1

        return inputs_T1.float(), outputs_B1.float(), brain.float()

    def __len__(self):
        return len(self.dir_T1_nifti)



def preview_dataloader(dir_folder, pad=[0,0,0], mask_roi=True, zoom=[0,0,0]):
    dir_T1 = join(dir_folder, 'T1_biascorr_brain.nii.gz')
    dir_B1 = join(dir_folder, 'B1map_T1space_brain.nii.gz')
    dir_brain = join(dir_folder, 'T1_biascorr_brain_mask2.nii.gz')
    if mask_roi:
        dir_csf = join(dir_folder, 'T1_fast_pve_0.nii.gz')
        dir_gm = join(dir_folder, 'T1_fast_pve_1_0p9.nii.gz')
        dir_wm = join(dir_folder, 'T1_fast_pve_2_0p9.nii.gz')

    T1_nii = nib.load(dir_T1)
    affine = T1_nii.affine
    T1 = T1_nii.get_fdata()
    B1 = nib.load(dir_B1).get_fdata()
    brain = nib.load(dir_brain).get_fdata()
    if mask_roi:
        csf = nib.load(dir_csf).get_fdata()
        wm = nib.load(dir_wm).get_fdata()
        gm = nib.load(dir_gm).get_fdata()

    mx_T1 = np.max(T1)

    if np.sum(pad)>0:
        T1, B1 = pad_array(T1, B1, pad)
        brain, _ = pad_array(brain, brain, pad)
        if mask_roi:
            csf, _ = pad_array(csf, csf, pad)
            wm, _ = pad_array(wm, wm, pad)
            gm, _ = pad_array(gm, gm, pad)

    if np.sum(zoom)>0:
        T1 = ndimage.zoom(T1, zoom)
        B1 = ndimage.zoom(B1, zoom)
        brain = ndimage.zoom(brain, zoom)
        if mask_roi:
            csf = ndimage.zoom(csf, zoom)
            wm = ndimage.zoom(wm, zoom)
            gm = ndimage.zoom(gm, zoom)

    brain = torch.unsqueeze(torch.from_numpy(brain), 0)

    T1, B1 = normalize_data(T1, B1)
    T1 = T1 * brain
    B1 = B1 * brain
    
    if mask_roi:
        csf = torch.unsqueeze(torch.from_numpy(csf), 0)
        wm = torch.unsqueeze(torch.from_numpy(wm), 0)
        gm = torch.unsqueeze(torch.from_numpy(gm), 0)
        inputs_T1 = torch.cat((T1, csf, wm, gm), 0)
        outputs_B1 = torch.cat((B1, csf, wm, gm), 0)
    else:
        inputs_T1 = T1
        outputs_B1 = B1

    return inputs_T1.float(), np.squeeze(B1.numpy()), affine, mx_T1

    