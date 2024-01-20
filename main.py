import random
import cv2
import numpy as np
import math

checkList = list(map(chr, list(range(65, 91)) + [33, 35, 36, 37, 38] + list(range(49, 58))))
LENGTH = 7
X = 230
Y = 100
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_PLAIN
FONT_THICKNESS = 1
FONT_SCALE = 3


def ziggle(cptimg, howMuchZiggle):
    height, width = cptimg.shape[:2]
    map2, map1 = np.indices((height, width), dtype=np.float32)

    map1 = map1 + width / howMuchZiggle * np.sin(map1)
    map2 = map2 + height / howMuchZiggle * np.cos(map2)

    dst = cv2.remap(captchaImg, map1, map2, cv2.INTER_CUBIC)

    return dst


def rotate(img, x, y, z):
    scale_percent = 300
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

    # 2d to 3d (projection)  , and -> rotation point - center point (origin point)
    proj2dto3d = np.array([[1, 0, -img.shape[1] / 2],
                           [0, 1, -img.shape[0] / 2],
                           [0, 0, 0],
                           [0, 0, 1]], np.float32)

    rx = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)  # 0

    ry = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)

    rz = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)  # 0

    trans = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 400],  # 400 to move the image in z axis
                      [0, 0, 0, 1]], np.float32)

    proj3dto2d = np.array([[200, 0, img.shape[1] / 2, 0],
                           [0, 200, img.shape[0] / 2, 0],
                           [0, 0, 1, 0]], np.float32)


    ax = float(x * (math.pi / 180.0))  # 0
    ay = float(y * (math.pi / 180.0))
    az = float(z * (math.pi / 180.0))  # 0

    rx[1, 1] = math.cos(ax)  # 0
    rx[1, 2] = -math.sin(ax)  # 0
    rx[2, 1] = math.sin(ax)  # 0
    rx[2, 2] = math.cos(ax)  # 0

    ry[0, 0] = math.cos(ay)
    ry[0, 2] = -math.sin(ay)
    ry[2, 0] = math.sin(ay)
    ry[2, 2] = math.cos(ay)

    rz[0, 0] = math.cos(az)  # 0
    rz[0, 1] = -math.sin(az)  # 0
    rz[1, 0] = math.sin(az)  # 0
    rz[1, 1] = math.cos(az)  # 0

    r = rx.dot(ry).dot(rz)  # if we remove the lines we put    r=ry
    final = proj3dto2d.dot(trans.dot(r.dot(proj2dto3d)))

    dst = cv2.warpPerspective(img, final, (img.shape[1], img.shape[0]), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, BLACK)

    return dst


def sincos(img, l, amp):

    rows, cols = img.shape[:2]

    # 초기 매핑 배열 생성 ---①
    mapy, mapx = np.indices((rows, cols), dtype=np.float32)

    # sin, cos 함수를 적용한 변형 매핑 연산 ---②
    sinx = mapx + amp * np.sin(mapy / l)
    cosy = mapy + amp * np.cos(mapx / l)

    # x,y 축 모두 sin, cos 곡선 적용 및 외곽 영역 보정
    img_both = cv2.remap(img, sinx, cosy, cv2.INTER_LINEAR, \
                         None, cv2.BORDER_REPLICATE)

    return img_both

def randxyz(rotval):
    x = random.randint(-rotval // 2, rotval // 2)
    rotval -= abs(x)
    y = random.randint(-rotval // 2, rotval // 2)
    rotval -= abs(y)
    z = random.randint(-rotval // 2, rotval // 2)
    rotval -= abs(y)

    return x, y, z


toCaptcha = list()
for _ in range(LENGTH):
    toCaptcha.append(random.choice(checkList))
toCaptcha = "".join(toCaptcha)


captchaImg = np.zeros((Y, X, FONT_SCALE), np.uint8)


captchaSize = cv2.getTextSize(toCaptcha, FONT, FONT_SCALE, FONT_THICKNESS)[0]
textX = (captchaImg.shape[1] - captchaSize[0]) // 2
textY = (captchaImg.shape[0] + captchaSize[1]) // 2


cv2.putText(captchaImg, toCaptcha, (textX, textY), FONT, FONT_SCALE, WHITE, FONT_THICKNESS, cv2.LINE_AA)

x, y, z = randxyz(90)
captchaImg = rotate(captchaImg, x, y, z)
captchaImg = ziggle(captchaImg, 800)
captchaImg = sincos(captchaImg, 35 ,20)
captchaImg = ziggle(captchaImg, 500)

cv2.imshow('captcha', captchaImg)
cv2.waitKey(0)
# ans = input("Solve the captcha: ")
# ans = ans.upper()
# if ans == toCaptcha:
#     print("Correct!")
# else:
#     print("Incorrect!")
# cv2.destroyAllWindows()

cv2.imwrite(f"./img/{toCaptcha}.png", captchaImg)
