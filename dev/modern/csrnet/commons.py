import io

from torchvision import models
from PIL import Image

import torchvision.transforms as transforms
import torch

from pythonModel import CSRNet

def get_model():
    model = CSRNet()
    model.load_state_dict(torch.load('./model.pt', map_location='cpu')) # Where we upload our model (Download model to local)
    model.eval()
    return model