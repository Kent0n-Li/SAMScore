import argparse
import samscore

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p1','--path_source', type=str, default='./imgs/n02381460_20_real.png')
parser.add_argument('-p0','--path_generated', type=str, default='./imgs/n02381460_20_fake.png')
parser.add_argument('-v','--version', type=str, default='1.0')

opt = parser.parse_args()

## Initializing the model
SAMScore_Evaluation = samscore.SAMScore(model_type = "vit_l" ) #, model_weight_path = "D:\Code\SAMScore\pytorch-CycleGAN-and-pix2pix-master\sam_vit_l_0b3195.pth"
samscore_result = SAMScore_Evaluation(source_image_path=opt.path_source,  generated_image_path=opt.path_generated)

print('SAMScore: %.4f'%samscore_result)
