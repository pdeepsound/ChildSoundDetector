import torch
import torch.nn as nn

class ExtractFeature(nn.Module):  # универсальная модель для любого времени
    def __init__(self, spectrogram_extractor, filters):     
        super(ExtractFeature, self).__init__()
        self.spectrogram_extractor = spectrogram_extractor
        self.filters = filters

    def forward(self, x):
        x, _, _ = self.spectrogram_extractor(x)    
        x = self.filters(x)
        return x
    