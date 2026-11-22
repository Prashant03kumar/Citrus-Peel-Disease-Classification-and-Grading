import torch
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def create_efficientnet_b0(num_classes: int, pretrained: bool = False) -> torch.nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    classifier_input_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(classifier_input_features, num_classes)
    return model
