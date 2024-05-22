import io
import torch
from torch import nn
import timm
from torchvision import transforms
import numpy as np
from PIL import Image
import os


def predict(image_data):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    t = transforms.Compose([transforms.Resize(255),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)])
    
    model = timm.create_model("efficientnet_b1")
    model.classifier = nn.Sequential(nn.Linear(model.classifier.in_features, 128, bias=False),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(),
                                 nn.Dropout(0.25),
                                 nn.Linear(128, 1))
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'app', 'model', 'model.pt'), map_location=torch.device('cpu')))
    model.eval()
    
    img = Image.open(io.BytesIO(image_data))
    img = t(img).unsqueeze(0) # type: ignore

    pred = "positive" if torch.sigmoid(model(img)) >= 0.6 else "negative"
    return pred
