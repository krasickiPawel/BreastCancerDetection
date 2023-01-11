import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


def set_parameter_requires_grad(model, fine_tuning):
    if not fine_tuning:
        for param in model.parameters():
            param.requires_grad = False


def create_resnet_model(num_classes, pretrained=True, fine_tuning=True):
    model = models.resnet18(pretrained=pretrained)
    set_parameter_requires_grad(model, fine_tuning)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # model.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
    return model
