import torch
import torch.nn as nn


'''
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
'''

# Encoder_DecoderクラスのためにBasicBlockクラスを作成する
class BasicBlock(nn.Module):
    expansion = 1  # 出力のチャンネル数を入力のチャンネル数の何倍に拡大するか

    def __init__(self, inputDim, outputDim, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inputDim, outputDim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outputDim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inputDim, outputDim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outputDim)

        # 入力と出力のチャンネル数が異なる場合、x をダウンサンプリングする。
        if inputDim != outputDim * self.expansion:
            self.shortcut = nn.Sequential(nn.Conv2d(inputDim, outputDim*self.expansion, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(outputDim*self.expansion))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)

        out = self.relu(out)

        return out


# stress用のモデル
class Encoder_Decoder_stress(nn.Module):
    def __init__(self, inputDim, outputDim):
        super(Encoder_Decoder_stress,  self).__init__()

        # encoder
        self.conv1 = nn.Sequential(nn.Conv2d(inputDim, 32, kernel_size=9, stride=1, padding=4), nn.BatchNorm2d(32), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU())

        # resnet
        self.RN1 = BasicBlock(128, 128)
        self.RN2 = BasicBlock(128, 128)
        self.RN3 = BasicBlock(128, 128)
        self.RN4 = BasicBlock(128, 128)
        self.RN5 = BasicBlock(128, 128)

        # decoder
        self.conv4 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.LeakyReLU())
        self.conv5 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.LeakyReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(32, outputDim, kernel_size=9, stride=1, padding=4), nn.LeakyReLU())

    def forward(self, x):
        # encoder
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        # resnet
        out = self.RN1(out)
        out = self.RN2(out)
        out = self.RN3(out)
        out = self.RN4(out)
        out = self.RN5(out)

        # decoder
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)

        out *= x[:, 1:2, :, :]

        return out


# force用のモデル
class Encoder_Decoder_force(nn.Module):
    def __init__(self, inputDim, outputDim):
        super(Encoder_Decoder_force,  self).__init__()

        # encoder
        self.conv1 = nn.Sequential(nn.Conv2d(inputDim, 32, kernel_size=9, stride=1, padding=4), nn.BatchNorm2d(32), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU())

        # resnet
        self.RN1 = BasicBlock(128, 128)
        self.RN2 = BasicBlock(128, 128)
        self.RN3 = BasicBlock(128, 128)
        self.RN4 = BasicBlock(128, 128)
        self.RN5 = BasicBlock(128, 128)

        # decoder
        self.conv4 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(64), nn.LeakyReLU())
        self.conv5 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.LeakyReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(32, outputDim, kernel_size=9, stride=1, padding=4), nn.LeakyReLU())

    def forward(self, x):
        # encoder
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        # resnet
        out = self.RN1(out)
        out = self.RN2(out)
        out = self.RN3(out)
        out = self.RN4(out)
        out = self.RN5(out)

        # decoder
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)

        out *= x[:, 2:3, :, :]

        return out


# test用のモデル
class CNN(nn.Module):
    def __init__(self, inputDim, outputDim):
        super(CNN,  self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(inputDim, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(16, outputDim, kernel_size=5, stride=1, padding=2)

    def forward(self, x):

        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.conv4(out)

        out *= x[:,1:2,:,:]

        return out