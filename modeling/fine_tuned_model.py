import torch
import torch.nn as nn

print('cuda', torch.cuda.is_available())


class FineTuneModel(nn.Module):
    def __init__(self, cfg, original_model):
        super(FineTuneModel, self).__init__()

        num_class = cfg.MODEL.NUM_CLASSES

        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(32, num_class)
        )
        self.modelName = 'fine_tune_model'
        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        y = self.classifier(f)
        return y
