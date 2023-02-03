'''
https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py
'''

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


# モデルをダウンロード
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)

# モデルを評価モードにする
model.eval()


def Extract_person(input_image):
    # 入力画像を標準化する
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # GPUが使える場合は入力とモデルをGPUに乗せる
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # モデルの出力をクラスのラベルに変換する
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # personクラスのラベルだけ抽出する
    label = (output_predictions == 15)

    '''
    # 表示する
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    r = Image.fromarray(label.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)
    plt.imshow(r)
    plt.show()
    '''

    return label

'''
input_image = Image.open('test.jpg')
label = Extract_person(input_image)
'''


'''
# モデルをダウンロード
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)

# モデルを評価モードにする
model.eval()

# 入力画像を標準化する
input_image = Image.open('test.jpg')
input_image = input_image.convert("RGB")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# GPUが使える場合は入力とモデルをGPUに乗せる
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

# モデルの出力をクラスのラベルに変換する
with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

# カラーパレットを作成して各クラスに色を割り当てる
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)

# 表示する
plt.imshow(r)
plt.show()
'''