import torch
from torch import nn

# Dataframes for PTB-XL
# Train: https://github.com/naavie/Scientific-ECG-PUBLIC/blob/main/train_subset.csv
# Validation: https://github.com/naavie/Scientific-ECG-PUBLIC/blob/main/validation_subset.csv
# Test: https://github.com/naavie/Scientific-ECG-PUBLIC/blob/main/test_subset.csv (excluded classes only for CLIP evaluation)
# ECG Signal Data: https://drive.google.com/file/d/1wqHbp6_DJZkIseMAdtySNSZ_ug9DQwr2/view?usp=sharing

# For training and evaluation of Image Encoders only, split Train DataFrame into train and validation subsets and use the validation subset for evaluation..

# AlexNet
class AlexNetECG(nn.Module):
    def __init__(self, num_classes=30):
        super(AlexNetECG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, stride=1, padding=3),  # Conv1D: (None, 5000, 32)
            nn.ReLU(),
            nn.Conv1d(32, 128, kernel_size=5, stride=2, padding=2),  # Conv1D: (None, 2500, 128)
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=5, stride=2, padding=2),  # Conv1D: (None, 1250, 64)
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),  # Conv1D: (None, 625, 64)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Max pooling 1D: (None, 312, 64)
        )
        self.classifier = nn.Sequential(
            nn.Linear(312 * 64, 156),  # Flatten to (None, 19968), then Dense to (None, 156)
            nn.ReLU(),
            nn.Linear(156, 140),  # Dense: (None, 140)
            nn.ReLU(),
            nn.Linear(140, num_classes),  # Dense (Sigmoid): (None, 30)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

# RNN
class RNNEncoder(nn.Module):
    def __init__(self,
                 input_size=12,
                 hidden_size=256,
                 num_layers=2,
                 dropout=0.0,
                 bidirectional=True,
                 output_size=512):

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.output_size = output_size

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )

        rnn_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(rnn_output_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)

        # Pass the input through the RNN layers
        _, (hidden, _) = self.rnn(x)

        # Extract the last hidden state from the last RNN layer
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]

        # Pass the last hidden state through the fully connected layer
        output = self.fc(hidden)

        return output

class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = CONFIG
        self.encoder = RNNEncoder(
            input_size=CONFIG.in_channels,
            hidden_size=CONFIG.hidden_size,
            num_layers=CONFIG.num_layers,
            dropout=CONFIG.dropout,
            bidirectional=CONFIG.bidirectional,
            output_size=CONFIG.image_embedding_size
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

# 1D-CNN
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

# 1D-CNN with Attention
class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, V)
        return output

class ECGEncoder1D(nn.Module):
    def __init__(self, num_classes=80):
        super(ECGEncoder1D, self).__init__()

        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3)
        self.bn4 = nn.BatchNorm1d(256)

        self.attention = SelfAttention(256)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional layers with ReLU activation and max pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn4(self.conv4(x)))

        # Apply attention
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, seq_len, channels)
        x = self.attention(x)
        x = x.permute(0, 2, 1)  # Change shape back to (batch_size, channels, seq_len)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x