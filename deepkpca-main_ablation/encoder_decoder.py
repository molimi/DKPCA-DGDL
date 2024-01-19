from torch import nn


class Lin_View(nn.Module):
    """ Unflatten linear layer to be used in Convolution layer"""

    def __init__(self, a, b):
        super(Lin_View, self).__init__()
        self.a, self.b = a, b

    def forward(self, x):
        try:
            return x.view(x.size(0), self.a, self.b)
        except:
            return x.view(1, self.a, self.b)


class Net1(nn.Module):
    """ Encoder - network architecture """
    def __init__(self, nChannels, capacity, x_fdim1, x_fdim2, cnn_kwargs):
        super(Net1, self).__init__()  # inheritance used here.      # SNP = 2000
        self.encoder = nn.Sequential(
            nn.Conv1d(nChannels, capacity, **cnn_kwargs[0]),        # when capacity=1: 1, 1, k=4, s=2, p=1，Now capacity=2: 1, 2, k=4, s=2, p=1
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv1d(capacity, capacity * 2, **cnn_kwargs[0]),     # when capacity=1: 1, 2, 4, 2, 1, Now capacity=2: 2, 4, 4, 2, 1   
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv1d(capacity * 2, capacity * 4, **cnn_kwargs[1]), # when capacity=1: 2, 4, 3, 1, Now capacity=2: 4, 8, k=3, s=1
            nn.LeakyReLU(negative_slope=0.2),

            nn.Flatten(),
            nn.Linear(capacity * 4 * cnn_kwargs[2], x_fdim1),          # when capacity=1: 1 * 4 * 498, Now capacity=2: 2 * 4 * 498
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(x_fdim1, x_fdim2),
        )

    def forward(self, x):
        x=x.reshape(x.shape[0], 1, x.shape[1])          # 变成卷积的样子
        return self.encoder(x)


class Net3(nn.Module):
    """ Decoder - network architecture """
    def __init__(self, nChannels, capacity, x_fdim1, x_fdim2, cnn_kwargs):
        super(Net3, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(x_fdim2, x_fdim1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(x_fdim1, capacity * 4 * cnn_kwargs[2]),
            nn.LeakyReLU(negative_slope=0.2),
            Lin_View(capacity * 4, cnn_kwargs[2]),  # Unflatten

            nn.ConvTranspose1d(capacity * 4, capacity * 2, **cnn_kwargs[1]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose1d(capacity * 2, capacity, **cnn_kwargs[0]),
            nn.LeakyReLU(negative_slope=0.2),

            nn.ConvTranspose1d(capacity, nChannels, **cnn_kwargs[0]),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)