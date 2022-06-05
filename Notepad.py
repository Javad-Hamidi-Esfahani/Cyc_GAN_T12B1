import os
import shutil
import nibabel as nib

dir_base_input = r'X:\BeaulieuLab\AB300\PROCESSED_DATA\AB300_T2map'
dir_base_des = r'C:\Users\hamid\D_drive\University\Ryan\Data'

dir_B1 = '\B1map_T1space.nii.gz'
dir_T1 = '\T1_1mm.anat\T1_biascorr_brain.nii.gz'
dir_brain_mask = '\T1_1mm.anat\T1_biascorr_brain_mask2.nii.gz'
dir_csf_mask = '\T1_1mm.anat\T1_fast_pve_0.nii.gz'
dir_wm_mask = '\T1_1mm.anat\T1_fast_pve_1_0p9.nii.gz'
dir_gm_mask = '\T1_1mm.anat\T1_fast_pve_2_0p9.nii.gz'

list_move = [dir_B1, dir_T1, dir_brain_mask, dir_csf_mask, dir_wm_mask, dir_gm_mask]

# for i in range(390,391):
#     AB_fol = 'Ab300_{:03d}'.format(i)
#     AB_folder = AB_fol + '\_fsl'

#     dir_folder_input = os.path.join(dir_base_input, AB_folder)

#     if os.path.isdir(dir_folder_input):
#         dir_folder_des = os.path.join(dir_base_des, AB_fol)

#         if not os.path.isdir(dir_folder_des):
#             os.mkdir(dir_folder_des)

#         for n in list_move:
#             shutil.copy(dir_folder_input + n, dir_folder_des+'\\'+n.split('\\')[-1])

for i in range(391):
    AB_fol = 'Ab300_{:03d}'.format(i)

    dir_folder_des = os.path.join(dir_base_des, AB_fol)
    if os.path.isdir(dir_folder_des):
        dir_B1 = dir_folder_des + '\B1map_T1space.nii.gz'
        dir_mask = dir_folder_des + '\T1_biascorr_brain_mask2.nii.gz'

        B1_nii = nib.load(dir_B1)
        affine = B1_nii.affine
        B1 = B1_nii.get_fdata()
        mask = nib.load(dir_mask).get_fdata()

        B1_masked = B1 * mask

        img = nib.Nifti1Image(B1_masked, affine)

        nib.save(img, dir_folder_des + '\B1map_T1space_brain.nii.gz')