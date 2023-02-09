import numpy as np
import cv2

# 選択した座標を保存するコールバック関数
def onMouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))

def get_fixed_contact(img):

    # 座標を選択する
    cv2.imshow('sample', img)
    cv2.setMouseCallback('sample', onMouse)
    cv2.waitKey(0)

    # 2点ずつ繋いで抽出する
    if len(points)%2 == 0:
        for i in range(int(len(points)/2)):
            cv2.line(img,                                      # 図形入力画像
                     points[2*i],                              # 開始点の座標（X,Y)
                     points[2*i+1],                            # 終了点の座標（X,Y)
                     (255,255,255),                            # カラーチャネル(B,G,R)
                     int(min(img.shape[0],img.shape[1])/64))   # 直線の太さ

        fixed_contact = cv2.inRange(img, (255,255,255), (255,255,255))
        cv2.imshow('fixed_contact',fixed_contact)
        cv2.waitKey(0)

    else:
        print('点数が奇数です')

    return fixed_contact

# 選択した座標を保存するリスト
points = []

# スナップショット画像
img = cv2.imread('test.jpg')

# 固定された接触位置を抽出
fixed_contact = get_fixed_contact(img)
cv2.imshow('fixed_contact',fixed_contact)
cv2.waitKey(0)

'''
# 選択した座標を保存するリスト
points = []

# 座標を選択する
img = cv2.imread('test.jpg')
cv2.imshow('sample', img)
cv2.setMouseCallback('sample', onMouse)
cv2.waitKey(0)

# 2点ずつ繋いで表示する
if len(points) % 2 == 0:
    for i in range(int(len(points) / 2)):
        cv2.line(img,  # 図形入力画像
                 points[2 * i],  # 開始点の座標（X,Y)
                 points[2 * i + 1],  # 終了点の座標（X,Y)
                 (255, 255, 255),  # カラーチャネル(B,G,R)
                 int(min(img.shape[0], img.shape[1]) / 64))  # 直線の太さ
    cv2.imshow('img', img)
    cv2.waitKey(0)

    img_mask = cv2.inRange(img, (255, 255, 255), (255, 255, 255))
    cv2.imshow('img_mask', img_mask)
    cv2.waitKey(0)

else:
    print('点数が奇数です')

cv2.destroyAllWindows()
'''