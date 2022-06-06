from datetime import datetime
import os
# /scratch/javadhe/Result_Test/Cyc_GAN/
from torch.utils.tensorboard import SummaryWriter


lambda_normal = 0
lambda_cyc = 100
mask_roi = True
comment = '_{}_brain'.format(lambda_cyc)
zoom = [1,1,1]
load_check = False
checkpoint = 100


image_size = ()
N_split = 315
batch = 1
num_workers = 0
epoch = 10000
lr = 1e-3

dir_dataset = r"C:\Users\hamid\D_drive\University\Ryan\Data"

# Model
features = 8
maxpool = False
avgpool = False

# Data Augmentation
crop = False
perspective = False
degrade = False
pad = [0,0]

# Preview Function
test_data_dir = r"C:\Users\hamid\D_drive\University\Ryan\Data\Ab300_390"
path_test_save = os.path.join(r"C:\Users\hamid\D_drive\University\Ryan\Misc", comment)

log_dir = os.path.join(r"C:\Users\hamid\D_drive\University\Ryan\Misc", comment, "Logs/")
model_path_save = os.path.join(r"C:\Users\hamid\D_drive\University\Ryan\Misc", comment)

if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
if not os.path.isdir(model_path_save):
    os.makedirs(model_path_save)
if not os.path.isdir(path_test_save):
    os.makedirs(path_test_save)

writer_train = SummaryWriter(log_dir=log_dir)
writer_test = SummaryWriter(log_dir=log_dir)