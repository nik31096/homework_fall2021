from typing import Union

import torch
from torch import nn
from cs285.infrastructure.dqn_utils import PreprocessAtari

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
):
    """
    `Builds a feedforward neural network
    arguments:
        input_placeholder: placeholder variable for the state (batch_size, input_size)
        scope: variable scope of the network
        n_layers: number of hidden layers
        size: dimension of each hidden layer
        activation: activation of each hidden layer
        input_size: size of the input layer
        output_size: size of the output layer
        output_activation: activation of the output layer
    returns:
    `    output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)


class CNNPolicy(nn.Module):
    def __init__(self, output_size: int, activation: Activation = 'relu', output_activation: Activation = 'identity'):
        super(CNNPolicy, self).__init__()
        self.output_size = output_size
        if isinstance(activation, str):
            self.activation = _str_to_activation[activation]
        if isinstance(output_activation, str):
            self.output_activation = _str_to_activation[output_activation]

        self.preprocess = PreprocessAtari()
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=(8, 8), stride=(4, 4)),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=(3, 3)),
            self.activation,
            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2)),
            self.activation,
            nn.Conv2d(128, 128, kernel_size=(3, 3)),
            self.activation
        )
        # self.conv1 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        # self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)
        # self.conv3 = nn.Conv2d(256, 128, kernel_size=(3, 3), padding=1)
        self.linear1 = nn.Linear(4608, 512)
        self.linear2 = nn.Linear(512, self.output_size)

    def forward(self, x):
        out = self.preprocess(x)
        out = self.backbone(out)
        # out1 = self.activation(self.conv1(out))
        # out2 = self.activation(self.conv2(out))
        # out = self.activation(self.conv3(torch.cat([out1, out2], dim=-1)))
        out = nn.Flatten()(out)
        out = self.activation(self.linear1(out))
        out = self.output_activation(self.linear2(out))

        return out


class CNNCritic(nn.Module):
    def __init__(self, output_size: int, activation: Activation = 'relu', output_activation: Activation = 'identity'):
        super(CNNCritic, self).__init__()
        self.output_size = output_size
        if isinstance(activation, str):
            self.activation = _str_to_activation[activation]
        if isinstance(output_activation, str):
            self.output_activation = _str_to_activation[output_activation]

        self.backbone = nn.Sequential(
            PreprocessAtari(),
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            self.activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            self.activation,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            self.activation,
            nn.Flatten(),
            nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
            self.activation,
            nn.Linear(512, self.output_size)
        )

    def forward(self, x):
        out = self.backbone(x)

        return out


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
