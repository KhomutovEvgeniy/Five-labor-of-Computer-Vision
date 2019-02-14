import cv2 as cv
import numpy as np

"""
Здесь рассматривается матрица размером 3х3
__________
|P9|P2|P3|
|P8|P1|P4|
|P7|P6|P5|
"""

""" Функция сравнения двух одноканальных изображений, возвращает реультат совпадения в процентах"""
def comparePict(img1, img2):
    equal = 0
    height, width = img1.shape
    diff = cv.subtract(img1, img2)
    for line in diff:
        for pixel in line:
            if pixel == 0:
                equal += 1
    return (equal / (height * width)) * 100


""" Функция скелетизации изображения """
def skeletization(img):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rows, cols = imgGray.shape  # Размер изображения
    image = cv.threshold(imgGray, 100, 1, type=cv.THRESH_BINARY)    # Бинаризация изображения
    imgBin = np.copy(image[1])  # Массив для промежуточного хранения картинки
    imgTemp = np.copy(imgBin)   # Временное изображение для текущей итерации скелетизации

    #   -1 т.к. обработке не подвергаются граничные пиксели изображения
    while True:
        for row in range(1, (rows - 1)):
            for col in range(1, (cols - 1)):
                if imgBin[row][col] == 1:
                    sequence = np.copy(
                        [imgBin[row - 1][col], imgBin[row - 1][col + 1], imgBin[row][col + 1], imgBin[row + 1][col + 1],
                         imgBin[row + 1][col], imgBin[row + 1][col - 1], imgBin[row][col - 1], imgBin[row - 1][col - 1]])

                    countWhitePix = sum(sequence)  # Количество белых пикселей вокруг центрального

                    if 2 <= countWhitePix <= 6:
                        """ количество найденных последовательностей 01 
                       в последовательности P2, P3, P4, P5, P6, P7, P8, P9, P2 """
                        countZeroToOne = 0  # Количество переходов 0->1 в последовательности из пикселей P2 P3 P4 P5 P6 P7 P8 P9 P2
                        for number in range(sequence.size):
                            prevPix = sequence[number - 1]
                            tempPix = sequence[number]
                            if prevPix == 0 and tempPix == 1:
                                countZeroToOne += 1
                        """ Должен существовать только один переход от нуля к единице """
                        if countZeroToOne == 1:
                            # Удаляются все пиксели на юго-восточной границе и северо-западные угловые пиксели
                            if (sequence[0] * sequence[2] * sequence[4]) == 0 and (
                                    sequence[2] * sequence[4] * sequence[6]) == 0:
                                imgTemp[row][col] = 0

                            # Удаляются все пиксели на северо-западной границе и юго-восточные угловые пиксели
                            elif (sequence[0] * sequence[2] * sequence[6]) == 0 and (
                                    sequence[0] * sequence[4] * sequence[6]) == 0:
                                imgTemp[row][col] = 0

        diff = comparePict(imgBin, imgTemp) # возвращает 100 - полное совпадение, 0 - нет совпадений

        """ Скелетизация выполняется до тех пор, пока в новой итерации не будет удален ни один пиксель """
        if diff == 100:
            # Домножение для того, чтоб можно было визуально представить результат
            imgTemp *= 255
            return imgTemp

        imgBin = imgTemp[:][:]


""" Функция поиска и отрисовки линий на скелете изображения """
def drawSkeletLines(skelet):
    # 	Ищем линии
    skeletGray= cv.cvtColor(skelet, cv.COLOR_BGR2GRAY)
    middleLines = cv.HoughLinesP(skeletGray, rho=0.1, theta=0.1, threshold=0, minLineLength=1)
    result = np.copy(skelet)
    # Рисуем полученные линии на изображении
    for line in middleLines:
        point1 = tuple(line[0][0:2])
        point2 = tuple(line[0][2:4])
        cv.line(result, point1, point2, (0, 0, 255), 2)
    return result

""" Поиск и отрисовка кругов на изображении """
def findAndDrawCircles(img):
    imgTemp = np.copy(img)
    imgGray = cv.cvtColor(imgTemp, cv.COLOR_BGR2GRAY)
    # Поиск кругов на изображении
    circles = cv.HoughCircles(imgGray, method=cv.HOUGH_GRADIENT, dp=1, minDist=20,
                              param1=100,
                              param2=60,
                              minRadius=3,
                              maxRadius=0)

    circles = np.uint16(np.around(circles))  # Округление чисел до целого
    for circle in circles[0, :]:
        cv.circle(imgTemp, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)

    return imgTemp


"""
# Первая часть задания
letterA = cv.imread('A.jpg', cv.IMREAD_COLOR)
letterAGray = cv.cvtColor(letterA, cv.COLOR_BGR2GRAY)

skelet = skeletization(letterA)
res = np.concatenate((skelet, letterAGray), axis=1)

cv.imshow('skelet', res)
cv.waitKey(0)
cv.destroyWindow('skelet')
"""

"""
# Вторая часть задания
skeletLines = drawSkeletLines(skelet)
cv.imshow('SkeletLines', skelet)
cv.waitKey(0)
cv.destroyWindow('SkeletLines')
"""

# Четвертая часть задания
coins = cv.imread('coins.jpg', cv.IMREAD_COLOR)

circles = findAndDrawCircles(coins)
cv.imshow('Circles', circles)
cv.waitKey(0)
cv.destroyWindow('Circles')