import argparse
import samscore
import cv2
import os
import torch


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p1','--path_source', type=str, default='./imgs/n02381460_20_real.png')
parser.add_argument('-p0','--path_generated', type=str, default='./imgs/n02381460_20_fake.png')
parser.add_argument('-v','--version', type=str, default='1.0')

opt = parser.parse_args()

## Initializing the model
SAMScore_Evaluation = samscore.SAMScore(model_type = "vit_b" ) #, model_weight_path = "D:\Code\SAMScore\pytorch-CycleGAN-and-pix2pix-master\sam_vit_l_0b3195.pth"

source_cv2 = cv2.imread(opt.path_source)
source = torch.from_numpy(source_cv2.transpose(2, 0, 1)).unsqueeze(0).float()
source = torch.cat((source,source,source),dim=0)

generated_cv2 = cv2.imread(opt.path_generated)
generated = torch.from_numpy(generated_cv2.transpose(2, 0, 1)).unsqueeze(0).float()
generated = torch.cat((generated,generated,generated),dim=0)

samscore_result = SAMScore_Evaluation.evaluation_from_torch(source,  generated)

print('SAMScore:',samscore_result)
