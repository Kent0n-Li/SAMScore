
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import torch.nn
import os
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch.nn.functional as F
import cv2
import requests



def rescale(X):
    X = 2 * X - 1.0
    return X


def inverse_rescale(X):
    X = (X + 1.0) / 2.0
    return torch.clamp(X, 0.0, 1.0)


def imageresize2tensor(path, image_size):
    img = Image.open(path)
    convert = transforms.Compose(
        [transforms.Resize(image_size, interpolation=Image.BICUBIC), transforms.ToTensor()]
    )
    return convert(img)


def image2tensor(path):
    img = Image.open(path)
    convert_tensor = transforms.ToTensor()
    return convert_tensor(img)


def calculate_l2_given_paths(path1, path2):
    file_name_old = os.listdir(path1)
    total = 0
    file_name = []
    for filename in file_name_old:
        if "fake" in str(filename):
            file_name.append(filename)

    for name in file_name:
        s = imageresize2tensor(os.path.join(path1, name.replace("fake", "real")), 256)
        name_i = name.split('.')[0]
        name = name_i + '.png'
        t = imageresize2tensor(os.path.join(path2, name), 256)
        l2_i = torch.norm(s - t, p=2)
        total += l2_i
    return total / len(file_name)


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def cosine_similarity(X, Y):
    '''
    compute cosine similarity for each pair of image
    Input shape: (batch,channel,H,W)
    Output shape: (batch,1)
    '''
    b, c, h, w = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    X = X.reshape(b, c, h * w)
    Y = Y.reshape(b, c, h * w)
    corr = norm(X) * norm(Y)  # (B,C,H*W)
    similarity = corr.sum(dim=1).mean(dim=1)
    return similarity


def sam_encode(sam_model, image, image_target,device):
    if sam_model is not None:
        sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        resize_img = sam_transform.apply_image(image)
        # resized_shapes.append(resize_img.shape[:2])
        resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(device)
        # model input: (1, 3, 1024, 1024)
        input_image = sam_model.preprocess(resize_img_tensor[None, :, :, :])  # (1, 3, 1024, 1024)
        assert input_image.shape == (1, 3, sam_model.image_encoder.img_size,
                                     sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'

        resize_img_target = sam_transform.apply_image(image_target)
        resize_img_target_tensor = torch.as_tensor(resize_img_target.transpose(2, 0, 1)).to(device)
        input_image_target = sam_model.preprocess(resize_img_target_tensor[None, :, :, :])
        assert input_image_target.shape == (1, 3, sam_model.image_encoder.img_size,
                                            sam_model.image_encoder.img_size), 'input image should be resized to 1024*1024'

        # input_imgs.append(input_image.cpu().numpy()[0])
        with torch.no_grad():
            embedding = sam_model.image_encoder(input_image)
            embedding_target = sam_model.image_encoder(input_image_target)
            samscore = cosine_similarity(embedding, embedding_target)
        return samscore

def download_model(url,destination):

    chunk_size = 8192  # Size of each chunk in bytes

    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(destination, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
        print("File downloaded successfully.")
    else:
        print("Failed to download file. Status code:", response.status_code)


class SAMScore(nn.Module):
    def __init__(self, model_type = "vit_l", model_weight_path = None ,version='1.0'):
        """ Initializes a perceptual loss torch.nn.Module

        Parameters (default listed first)
        ---------------------------------
        samscore : float
            The SAM score between the source image and the generated image
        model_type : str
            The type of model to use for the SAM score. Currently only supports 'vit_l,vit_b,vit_h'
        version : str
            The version of the SAM model to use. Currently only supports '1.0'
        source_image_path : str
            The path to the source image
        generated_image_path : str
            The path to the generated image
        
        """

        super(SAMScore, self).__init__()

        online_vit_l_model_path = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
        online_vit_b_model_path = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        online_vit_h_model_path = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

        if model_weight_path is None:
            if model_type == "vit_l":
                online_model_weight_path = online_vit_l_model_path
            elif model_type == "vit_b":
                online_model_weight_path = online_vit_b_model_path
            elif model_type == "vit_h":
                online_model_weight_path = online_vit_h_model_path
            else:
                raise ValueError("model_type must be one of 'vit_l','vit_b','vit_h'")
            
        # to download the model weights from online link
        if model_weight_path is None:
            model_weight_path = download_model(url = online_model_weight_path,destination= os.path.join("samscore","weights","sam_model.pth"))

                                               

        self.version = version
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam = sam_model_registry[self.model_type](checkpoint=model_weight_path) #"sam_vit_l_0b3195.pth"
        self.sam.to(device=self.device)


    def forward(self, source_image_path=None,  generated_image_path=None):

        source_cv2 = cv2.imread(source_image_path)
        source_cv2 = cv2.cvtColor(source_cv2, cv2.COLOR_BGR2RGB)

        generated_cv2 = cv2.imread(generated_image_path)
        generated_cv2 = cv2.cvtColor(generated_cv2, cv2.COLOR_BGR2RGB)

        samscore = sam_encode(self.sam, source_cv2, generated_cv2,device = self.device)

        return samscore
