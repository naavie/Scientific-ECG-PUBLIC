import torch
from torch import nn

_ACTIVATION_DICT = {'relu': nn.ReLU,
                    'tanh': nn.Tanh,
                    'none': nn.Identity,
                    'leaky_relu': lambda: nn.LeakyReLU(negative_slope=0.2)}

class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 act='relu', bn=True, dropout=None,
                 maxpool=None, padding=None, stride=1):

        super().__init__()

        if padding is None or padding == 'same':
            padding = kernel_size // 2

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=not bn, stride=stride)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.act = _ACTIVATION_DICT[act]()
        self.dropout = None if dropout is None else nn.Dropout(dropout)
        self.maxpool = None if maxpool is None else nn.MaxPool1d(maxpool)

    def forward(self, x):
        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        x = self.act(x)

        if self.dropout is not None:
            x = self.dropout(x)

        if self.maxpool is not None:
            x = self.maxpool(x)

        return x

class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', bn=True, dropout=None):

        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels, bias=not bn)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.act = _ACTIVATION_DICT[act]()
        self.dropout = None if dropout is None else nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)

        if self.bn is not None:
            x = self.bn(x)

        x = self.act(x)

        if self.dropout is not None:
            x = self.dropout(x)
        return x

class ConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernels, bn=True, dropout=None, maxpool=2, padding=0, stride=None):
        super().__init__()

        num_layers = len(channels)
        if stride is None:
            stride = [1] * num_layers

        self.in_layer = Conv1dBlock(in_channels, channels[0], kernels[0], bn=bn, dropout=dropout, maxpool=maxpool, padding=padding, stride=stride[0])

        conv_layers = list()
        for i in range(1, num_layers):
            conv_layers.append(Conv1dBlock(channels[i - 1], channels[i], kernels[i], bn=bn, dropout=dropout, maxpool=maxpool, padding=padding, stride=stride[i]))
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, x):
        x = self.in_layer(x)
        for layer in self.conv_layers:
            x = layer(x)
        return x

class ECGEncoder(nn.Module):
    def __init__(self,
                 window=1280,
                 in_channels=12,
                 channels=(32, 32, 64, 64, 128, 128, 256, 256),
                 kernels=(7, 7, 5, 5, 3, 3, 3, 3),
                 linear=512,
                 output=512):

        super().__init__()

        self.conv_encoder = ConvEncoder(in_channels, channels,  kernels, bn=True)

        with torch.no_grad():
            inpt = torch.zeros((1, in_channels, window), dtype=torch.float32)
            outpt = self.conv_encoder(inpt)
            output_window = outpt.shape[2]

        self.flatten = nn.Flatten()
        self.conv_to_linear = nn.Linear(output_window * channels[-1], linear)
        self.act = nn.ReLU()
        self.out_layer = nn.Linear(linear, output)

    def forward(self, x):
        # print(x.shape)
        x = self.conv_encoder(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.conv_to_linear(x)
        x = self.act(x)
        x = self.out_layer(x)
        return x

class ImageEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = CONFIG
        self.encoder = ECGEncoder(output=CONFIG.image_embedding_size)

    def forward(self, x):
        x = self.encoder(x)
        return x