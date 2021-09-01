import torch
import torch.nn as nn


def init_weights(m, mode = 'xavier_uniform'):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        if mode is not None:
            if mode == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif mode == 'xavier_normal':
                nn.init.xavier_normal_(m.weight, gain=1.0)
            elif mode == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif mode == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif mode == 'tds_uniform':
                tds_uniform_(m.weight)
            elif mode == 'tds_normal':
                tds_normal_(m.weight)
            else:
                raise ValueError("Unknown Initialization mode: {0}".format(mode))
    elif isinstance(m, nn.BatchNorm1d):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


    
class LinearDecoder(nn.Module):
    def __init__(self, encoder, num_classes, init_mode="xavier_uniform", type_='sigmoid'):
        super().__init__()

        self._feat_in = encoder.channels_last
        self._num_classes = num_classes

        self.decoder_layers = torch.nn.Sequential(
            torch.nn.Linear(self._feat_in, self._num_classes)
        )
        
        if type_ == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=-1)
        
        self.apply(lambda x: init_weights(x, mode=init_mode))


    def forward(self, encoder_output):
        encoder_output = self.decoder_layers(encoder_output)
        return self.activation(encoder_output)
    
    
    