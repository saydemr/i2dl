"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

# TODO: Choose from either model and uncomment that line
class KeypointModel(nn.Module):
# class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
            
        NOTE: You could either choose between pytorch or pytorch lightning, 
            by switching the class name line.
        """
        super().__init__()
        self.hparams = hparams
        # self.save_hyperparameters(hparams) # uncomment when using pl.LightningModule
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        #                                                                      #
        # We would truly recommend to make your code generic, such as you      #
        # automate the calculation of the number of parameters at each layer.  #
        # You're going probably try different architecutres, and that will     #
        # allow you to be quick and flexible.                                  #
        ########################################################################
    
        self.pre_process_layer = ConvLayer(hparams["in_channels"], 32, 3, 1, 1)
        self.res_net = nn.Sequential(
            ResBlock(32, 32, 3, 1, 1),
            nn.MaxPool2d(2, 1), # 95 x 95
            ResBlock(32, 32, 3, 1, 1),
            nn.MaxPool2d(2, 1), # 94 x 94
            ResBlock(32, 32, 3, 1, 1),
            nn.MaxPool2d(2, 1), # 93 x 93
            ConvLayer(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 1), # 92 x 92
            ResBlock(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2), # 46 x 46
            ResBlock(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 1), # 45 x 45
            ResBlock(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 1), # 44 x 44
            ResBlock(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 1), # 43 x 43
            ResBlock(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 1), # 42 x 42
            ResBlock(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2), # 21 x 21 
            ResBlock(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 1), # 20 x 20
            ResBlock(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 1), # 19 x 19
            ResBlock(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 1), # 18 x 18
            ConvLayer(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 1), # 17 x 17
            ResBlock(128, 128, 3, 1, 1),
            nn.MaxPool2d(2, 1), # 16 x 16
            ResBlock(128, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2), # 8 x 8
            ResBlock(128, 128, 3, 1, 1),
            nn.MaxPool2d(2, 1), # 7 x 7
            ResBlock(128, 128, 3, 1, 1),
            nn.MaxPool2d(2, 1), # 6 x 6
            ResBlock(128, 128, 3, 1, 1),
            nn.MaxPool2d(2, 1), # 5 x 5
            ResBlock(128, 128, 3, 1, 1)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(128*5*5, 512+128),
            nn.ReLU(),
            nn.BatchNorm1d(512+128),
            nn.Linear(512+128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, hparams["out_channels"])
        )



    def forward(self, x):
        
        # check dimensions to use show_keypoint_predictions later
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints.                                   #
        # NOTE: what is the required output size?                              #
        ########################################################################

        x = self.pre_process_layer(x)
        x = self.res_net(x)
        x = self.flatten(x)
        x = self.classifier(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        x = x + out
        return x
