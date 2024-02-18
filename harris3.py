import numpy as np
import cv2
import matplotlib.pyplot as plt

SOBEL_X = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int32")

SOBEL_Y = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]), dtype="int32")

GAUSS = np.array((
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]), dtype="float64")


def convolve(img, kernel):
    # verificare dimensiune kernel
    if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
        raise ValueError("Conv size error")

    # obținere dimensiuni imagine și kernel
    img_height = img.shape[0]
    img_width = img.shape[1]
    pad_height = kernel.shape[0] // 2
    pad_width = kernel.shape[1] // 2

    pad = ((pad_height, pad_height), (pad_height, pad_width))
    # se creează o imagine de aceleasi dim ca imag initiala, dar goala
    g = np.empty(img.shape, dtype=np.float64)
    img = np.pad(img, pad, mode='constant', constant_values=0)

    # se parcurge imaginea
    for i in np.arange(pad_height, img_height + pad_height):
        for j in np.arange(pad_width, img_width + pad_width):
            # se formează o matrice cu valorile din imagine corespunzatoare cu pozitiile kernelului
            roi = img[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1]
            # se aplica op de convolutie si se adaugă pe pozitia specifică
            g[i - pad_height, j - pad_width] = (roi * kernel).sum()

    # se normalizeaza daca este cazul
    if (g.dtype != np.float64):
        g = g + abs(np.amin(g))
        g = g / np.amax(g)
        g = (g * 255.0)

    return g

def harris(img, threshold, k = 0.04):
    # se realizează o copie a imaginii sursa
    img_cpy = img.copy()
    # convertire imagine în grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # se aplică Sobel
    dx = convolve(img_gray, SOBEL_X)
    dy = convolve(img_gray, SOBEL_Y)

    # se realizează Ix^2, Iy^2 și IxIy
    dx2 = np.square(dx)
    dy2 = np.square(dy)
    dxdy = dx*dy

    # filtrare gaussiana
    g_dx2 = convolve(dx2, GAUSS)
    g_dy2 = convolve(dy2, GAUSS)
    g_dxdy = convolve(dxdy, GAUSS)

    # matricea de raspuns + afisarea ei
    harris = g_dx2*g_dy2 - np.square(g_dxdy) - k*np.square(g_dx2 + g_dy2)
    cv2.imshow("Before Norm", harris)

    #  normalizare intre 0 si 255 a matricii de răspuns + convertire in grayscale
    cv2.normalize(harris, harris, 0, 255, cv2.NORM_MINMAX)
    harris2 = np.uint8(harris)
    # cv2.convertScaleAbs(harris, harris)
    cv2.imshow("After Norm", harris2)

    # calculare histograma + afisarea ei
    hist = cv2.calcHist([harris2], [0], None, [256], [0, 256])

    plt.plot(hist)
    plt.title('Histogram Harris Corners')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

    print(harris.mean())
    for i in range(harris.shape[0]):
        for j in range(harris.shape[1]):
            # filtrare prin thresholding
            if int(harris[i, j]) > threshold:
                OK = False
                # non-maximum supression
                for ii in range(-5, 6):
                    for jj in range(-5, 6):
                        # verificare să nu iasa din imagine punctele
                        if 0 < i + ii < harris.shape[0] and 0 < j + jj < harris.shape[1]:
                            # obtinere maxime locale
                            if harris[i, j] < harris[i + ii, j + jj]:
                                OK = True

                # punctele care trec de non-maximum supression
                # si de filtrarea prin thresholding
                # sunt puncte de colt si sunt marcate pe imagine
                if not OK:
                    cv2.circle(harris2, (j, i), 3, 179, -1, 8, 0)

    return harris2

def harris_rgb(img_rgb, threshold=0.48, k = 0.10):
    # se realizează o copie a imaginii sursa
    img_rgb_cpy = img_rgb.copy()
    # se imparte imaginea pe canale RGB
    b, g, r = cv2.split(img_rgb)

    # se aplică Sobel
    dxb = convolve(b, SOBEL_X)
    dyb = convolve(b, SOBEL_Y)

    dxg = convolve(g, SOBEL_X)
    dyg = convolve(g, SOBEL_Y)

    dxr = convolve(r, SOBEL_X)
    dyr = convolve(r, SOBEL_Y)

    # se realizează Rx^2 + Gx^2 + Bx^2,
    # Ry^2 + Gy^2 + By^2 și
    # RxRy + GxGy + BxBy
    mx = np.square(dxb) + np.square(dxg) + np.square(dxr)
    my = np.square(dyb) + np.square(dyg) + np.square(dyr)
    mxy = dxb*dyb + dxg*dyg + dxr*dyr

    # filtrare gaussiana
    gx = convolve(mx, GAUSS)
    gy = convolve(my, GAUSS)
    gxy = convolve(mxy, GAUSS)

    # matricea de raspuns + afisarea ei
    harris = gx * gy - np.square(gxy) - k * np.square(gx + gy)
    cv2.imshow("Before Norm", harris)

    # normalizare intre 0 si 255 a matricii de răspuns
    cv2.normalize(harris, harris, 0, 255, cv2.NORM_MINMAX)
    harris2 = np.uint8(harris)
    # cv2.convertScaleAbs(harris, harris)
    cv2.imshow("Normalized Harris Abs", harris2)

    # calculare histograma + afisarea ei
    hist = cv2.calcHist([harris2], [0], None, [256], [0, 256])

    plt.plot(hist)
    plt.title('Histogram of Harris Form')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

    print(harris.mean())
    for i in range(harris.shape[0]):
        for j in range(harris.shape[1]):
            # filtrare prin thresholding
            if int(harris[i, j]) > threshold:
                OK = False
                # non-maximum supression
                for ii in range(-5, 6):
                    for jj in range(-5, 6):
                        # verificare să nu iasa din imagine punctele
                        if 0 < i + ii < harris.shape[0] and 0 < j + jj < harris.shape[1]:
                            # obtinere maxime locale
                            if harris[i, j] < harris[i + ii, j + jj]:
                                OK = True

                # punctele care trec de non-maximum supression
                # si de filtrarea prin thresholding
                # sunt puncte de colt si sunt marcate pe imagine
                if not OK:
                    color = np.random.randint(0, 254)
                    cv2.circle(img_rgb_cpy, (j, i), 2, color, 1, 8, 0)
    return img_rgb_cpy

if __name__ == "__main__":
    # # grayscale
    # # citire si afisare imagine sursa
    # img1 = cv2.imread('D:\\Facultate\\ANUL 4\\Anul 4 - Semestrul 1\\IOC\\Proiect\\Harris_corner_gray_and_rgb\\01.bmp')
    # cv2.imshow("Src", img1)
    # # aplicare algoritm harris() pe imaginea sursa si afisare rezultat
    # gray_result = harris(img1, threshold=85)
    # cv2.namedWindow("Out_gray", cv2.WINDOW_NORMAL)
    # cv2.imshow("Out_gray", gray_result)
    # cv2.waitKey(0)

    # # citire si afisare imagine sursa
    # img2 = cv2.imread('D:\\Facultate\\ANUL 4\\Anul 4 - Semestrul 1\\IOC\\Proiect\\Harris_corner_gray_and_rgb\\A000002-1.bmp')
    # cv2.imshow("Src", img2)
    # # aplicare algoritm harris() pe imaginea sursa si afisare rezultat
    # gray_result = harris(img2, threshold=130)
    # cv2.namedWindow("Out_gray", cv2.WINDOW_NORMAL)
    # cv2.imshow("Out_gray", gray_result)
    # cv2.waitKey(0)

    # citire si afisare imagine sursa
    img3 = cv2.imread('D:\\Facultate\\ANUL 4\\Anul 4 - Semestrul 1\\IOC\\Proiect\\Harris_corner_gray_and_rgb\\A000020-1.bmp')
    cv2.imshow("Src", img3)
    # aplicare algoritm harris() pe imaginea sursa si afisare rezultat
    gray_result = harris(img3, threshold=150)
    cv2.namedWindow("Out_gray", cv2.WINDOW_NORMAL)
    cv2.imshow("Out_gray", gray_result)
    cv2.waitKey(0)

    # # color RGB
    # # citire si afisare imagine sursa
    # img_rgb = cv2.imread('D:\\Facultate\\ANUL 4\\Anul 4 - Semestrul 1\\IOC\\Proiect\\Harris_corner_gray_and_rgb\\rgb1.jpg')
    # cv2.imshow("Src", img_rgb)
    # # aplicare algoritm harris_rgb() pe imaginea sursa si afisare rezultat
    # rgb_result = harris_rgb(img_rgb, threshold=50, k=0.04)
    # cv2.namedWindow("Out_rgb", cv2.WINDOW_NORMAL)
    # cv2.imshow("Out_rgb", rgb_result)
    # cv2.waitKey(0)

    # # citire si afisare imagine sursa
    # img_rgb = cv2.imread('D:\\Facultate\\ANUL 4\\Anul 4 - Semestrul 1\\IOC\\Proiect\\Harris_corner_gray_and_rgb\\rgb2.jpg')
    # cv2.imshow("Src", img_rgb)
    # # aplicare algoritm harris_rgb() pe imaginea sursa si afisare rezultat
    # rgb_result = harris_rgb(img_rgb, threshold=65, k=0.04)
    # cv2.namedWindow("Out_rgb", cv2.WINDOW_NORMAL)
    # cv2.imshow("Out_rgb", rgb_result)
    # cv2.waitKey(0)

    # citire si afisare imagine sursa
    img_rgb = cv2.imread('D:\\Facultate\\ANUL 4\\Anul 4 - Semestrul 1\\IOC\\Proiect\\Harris_corner_gray_and_rgb\\rgb3.jpg')
    cv2.imshow("Src", img_rgb)
    # aplicare algoritm harris_rgb() pe imaginea sursa si afisare rezultat
    rgb_result = harris_rgb(img_rgb, threshold=45, k=0.04)
    cv2.namedWindow("Out_rgb", cv2.WINDOW_NORMAL)
    cv2.imshow("Out_rgb", rgb_result)
    cv2.waitKey(0)
