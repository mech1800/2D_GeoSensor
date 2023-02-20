from PIL import Image
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('..')

from ImgGeo.Object.get_object import super_pixel, clustering,cluster_to_object, Extract_object
from ImgGeo.Person.model import Extract_person
from ImgGeo.SnapShot.get_fixed_contact import get_fixed_contact

from GeoSensor.code.model import Encoder_Decoder_stress, Encoder_Decoder_force, Encoder_Decoder_force_

'''
変数の説明
　img_snap：変形前の画像
　img：変形中の画像
　input_image：変形後の画像（imgとはrgbの順番が違う）
　points：get_fixed_contactの内部で使用される変数
　pre_geometry：変形前の物体ラベル(0は空間セル,1は物体セルを表す)
　fixed_contact：床との接触点のラベル(0は空間セル,1は接触セルを表す)
　person_label：変形中の人間のラベル(0は空間セル,1は人間セルを表す)
　geometry：変形中の物体ラベル(0は空間セル,1は物体セルを表す)
　unfixed_contact：人間との接触点のラベル(0は空間セル,1は接触セルを表す)
　contact：すべての接触点のラベル(0は空間セル,1は接触セルを表す)
　label：person_labelとgeometryを足したもの(0は空間セル,1は人間セル,2は物体ラベル,3は人間セルと物体セルを表す)
　model：形状変形特徴量から物理量を指定する学習済み深層学習モデル
　input：深層学習モデルへの入力[[変形前の物体ラベル],[変形中の物体ラベル],[接触点のラベル]]
　output：深層学習モデルの出力
'''

# 任意のnumpy画像を64×64にリサイズする(utilityディレクトリを作成して移動させたい)
def resize_64(image):
    # m×n(m>n)をn×nに切り取る
    if image.shape[0]<image.shape[1]:
        w = int((image.shape[1]-image.shape[0])/2)
        image = image[:,w:w+image.shape[0]]
    if image.shape[0]>image.shape[1]:
        h = int((image.shape[0]-image.shape[1])/2)
        image = image[h:h+image.shape[1],:]

    # n×nを64×64に圧縮（cv2で使用されるdtypeに変換する）
    image = image.astype(np.uint8)
    image = cv2.resize(image,dsize=(64,64))
    image = image.astype(np.int32)

    return image


# Featureを作成
img_snap = cv2.imread('snapshot.jpg')
img = cv2.imread('test.jpg')

# Snapshot
# ----------pre_geometry----------
pre_geometry = Extract_object(img_snap)
# plt.imshow(pre_geometry)
# plt.show()

# ----------unfixed_contact----------
# 選択した座標を保存するリスト
points = []
# 固定された接触位置を抽出
fixed_contact = get_fixed_contact(img_snap)
cv2.imshow('fixed_contact',fixed_contact)
cv2.waitKey(0)
# plt.imshow(fixed_contact)
# plt.show()


# No Snapshot
# ----------personラベル----------
# モデルを用意
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
model.eval()

input_image = Image.open('test.jpg')
person_label = Extract_person(input_image).byte().cpu().numpy()
# plt.imshow(person_label)
# plt.show()

# ----------geometry----------
geometry = Extract_object(img)
# plt.imshow(geometry)
# plt.show()

# personラベルとobjectラベルを重ね合わせて表示する
label = person_label + 2*geometry
# plt.imshow(label)
# plt.show()

# ----------contact----------
unfixed_contact = (label==3)
unfixed_contact.astype(np.uint8)
# plt.imshow(unfixed_contact)
# plt.show()

contact = (unfixed_contact + fixed_contact).astype(np.bool)
# plt.imshow(contact)
# plt.show()


# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 64×64にresize
pre_geometry = resize_64(pre_geometry)   # 実際はスナップショットの画像を入力する
geometry = resize_64(geometry)
contact = resize_64(contact)

plt.imshow(pre_geometry)
plt.show()
plt.imshow(geometry)
plt.show()
plt.imshow(contact)
plt.show()

pre_geometry = torch.from_numpy(pre_geometry.astype(np.float32)).to(device)   # 実際はスナップショットの形状を入力する
# pre_geometry = Extract_pre_geometry(input_image)
geometry = torch.from_numpy(geometry.astype(np.float32)).to(device)
contact = torch.from_numpy(contact.astype(np.float32)).to(device)

input = torch.stack([pre_geometry,geometry,contact], dim=0)
input = input.unsqueeze(dim=0)


# modelに入力
model = Encoder_Decoder_force_(inputDim=3, outputDim=1)
model.load_state_dict(torch.load('../GeoSensor/code/result/force/model_weight.pth'))
model = model.to(device)

# model = torch.load('../GeoSensor/code/result/force/model.pth')
# model = model.to(device)

output = model(input)
plt.imshow(output[0,0,:,:].to('cpu').detach().numpy())
plt.show()

'''
やること
手との接触位置でフィルタリングする方法を考える
リアルタイムで動画を入力および出力できるようにする（3月）
'''