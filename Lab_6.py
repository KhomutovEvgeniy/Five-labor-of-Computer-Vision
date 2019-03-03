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
# На вход подается трехканальное изображение в пространстве BGR!
# Возвращает скелетизированное одноканальное изображение
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
                            # Удаляются все юго-восточные  и северо-западные угловые пиксели
                            if (sequence[0] * sequence[2] * sequence[4]) == 0 and (
                                    sequence[2] * sequence[4] * sequence[6]) == 0:
                                imgTemp[row][col] = 0

                            # Удаляются все пиксели на северо-западной границе и юго-восточные угловые пиксели (коммент неверен наверн, да не суть)
                            elif (sequence[0] * sequence[2] * sequence[6]) == 0 and (
                                    sequence[0] * sequence[4] * sequence[6]) == 0:
                                imgTemp[row][col] = 0

        diff = comparePict(imgBin, imgTemp) # возвращает 100 - полное совпадение, 0 - нет совпадений

        """ Скелетизация выполняется до тех пор, пока в новой итерации не будет удален ни один пиксель """
        if diff == 100:
            # Домножение для того, чтоб можно было визуально представить результат
            imgTemp *= 255
            imgTemp = cv.cvtColor(imgTemp, cv.COLOR_GRAY2BGR)
            return imgTemp

        imgBin = imgTemp[:][:]

""" Функция соединения линий """
def connectLines(skelet):
    # 	Ищем линии
    skeletGray = cv.cvtColor(skelet, cv.COLOR_BGR2GRAY)
    lineImage = np.copy(skelet) * 0

    skeletLines = cv.HoughLinesP(skeletGray, rho=0.1, theta=0.1, threshold=0, minLineLength=1)

    new_lines = []
    points = []
    max_dst = 50
    angle_max = 10  # angle threshold on degrees
    for line in skeletLines:
        if line[0, 1] < 250 or line[0, 3] < 250:
            line[:] = 0, 0, 0, 0
            print("pidor")
        if line[0, 0]:
            # c1x= (line[0,0]+line[0,2])//2
            # c1y= (line[0,1]+line[0,3])//2
            # line[0,0]=0
            for second_line in skeletLines:
                if second_line[0, 1] < 250 or second_line[0, 3] < 250:
                    second_line[:] = 0, 0, 0, 0
                    print("pidor")
                if (second_line[0, 0] or second_line[0, 1]) and not np.allclose(line, second_line):
                    dst1 = ((line[0, 0] - second_line[0, 0]) ** 2 + (
                                line[0, 1] - second_line[0, 1]) ** 2) ** 0.5  # p11 p21
                    dst2 = ((line[0, 0] - second_line[0, 2]) ** 2 + (
                                line[0, 1] - second_line[0, 3]) ** 2) ** 0.5  # p11 p22
                    dst3 = ((line[0, 2] - second_line[0, 0]) ** 2 + (
                                line[0, 3] - second_line[0, 1]) ** 2) ** 0.5  # p12 p21
                    dst4 = ((line[0, 2] - second_line[0, 2]) ** 2 + (
                                line[0, 3] - second_line[0, 3]) ** 2) ** 0.5  # p12 p22

                    if dst1 < max_dst:  #
                        points.append(
                            ((line[0, 0] + 0.0, line[0, 1] + 0.0), (second_line[0, 0] + 0.0, second_line[0, 1] + 0.0)))
                        print(" u1")
                        cv.line(lineImage, (line[0, 0], line[0, 1]), (second_line[0, 0], second_line[0, 1]),
                                 (255, 0, 0), 5)
                        break
                    if dst2 < max_dst:  #
                        points.append(
                            ((line[0, 0] + 0.0, line[0, 1] + 0.0), (second_line[0, 2] + 0.0, second_line[0, 3] + 0.0)))
                        print("u2")
                        cv.line(lineImage, (line[0, 0], line[0, 1]), (second_line[0, 2], second_line[0, 3]),
                                 (255, 0, 0), 5)
                        break
                    if dst3 < max_dst:  #
                        points.append(
                            ((line[0, 2] + 0.0, line[0, 3] + 0.0), (second_line[0, 0] + 0.0, second_line[0, 1] + 0.0)))
                        print(" u3")
                        cv.line(lineImage, (line[0, 2], line[0, 3]), (second_line[0, 0], second_line[0, 1]),
                                 (255, 0, 0), 5)
                        break
                    if dst4 < max_dst:  # line = (p12+p22)/2, (p11+p21)/2
                        points.append(
                            ((line[0, 2] + 0.0, line[0, 3] + 0.0), (second_line[0, 2] + 0.0, second_line[0, 3] + 0.0)))
                        print(" u4")
                        cv.line(lineImage, (line[0, 2], line[0, 3]), (second_line[0, 2], second_line[0, 3]),
                                 (255, 0, 0), 5)
                        break

    for line in skeletLines:
        if line[0, 0] or line[0, 1]:
            for x1, y1, x2, y2 in line:
                points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
                cv.line(lineImage, (x1, y1), (x2, y2), (255, 0, 0), 5)

    for p in points:
        x1 = int(p[0][0])
        y1 = int(p[0][1])
        x2 = int(p[1][0])
        y2 = int(p[1][1])

        cv.circle(lineImage, (x1, y1), 5, (100, 100, 100), -1)
        cv.circle(lineImage, (x2, y2), 5, (100, 100, 100), -1)

    lines_edges = cv.addWeighted(skelet, 0.8, lineImage, 1, 0)

    cv.imwrite('path_lines2.png', lines_edges)
    cv.imshow('tata', lines_edges)
    cv.waitKey(0)


""" Функция поиска и отрисовки линий на скелете изображения """
# На вход подается скелетизированное изображение
# Возвращается копия исходного изображения с отрисованными линиями
def drawSkeletLines(skelet):
    # 	Ищем линии
    skeletGray = cv.cvtColor(skelet, cv.COLOR_BGR2GRAY)
    skeletLines = cv.HoughLinesP(skeletGray, rho=0.1, theta=0.1, threshold=0, minLineLength=1)   # Поиск линий Хафом
    np.insert(skeletLines, 1, [1, 1, 1, 1, ]) # Уже не помню зачем эта команда здесь

    result = np.copy(skelet)
    # Рисуем полученные линии на изображении
    for line in skeletLines:
        point1 = tuple(line[0][0:2])
        point2 = tuple(line[0][2:4])
        cv.line(result, point1, point2, (0, 255, 0), 2)
    return result


""" Поиск и отрисовка кругов на изображении """
# На вход подается трехканальное BGR изображение
# Возвращает массив элементов, каждый элемент имеет структуру (х, y, r), х,у - центр круга, r - радиус
def findCircles(img):
    imgTemp = np.copy(img)
    imgGray = cv.cvtColor(imgTemp, cv.COLOR_BGR2GRAY)
    # Поиск кругов на изображении
    circles = cv.HoughCircles(imgGray, method=cv.HOUGH_GRADIENT, dp=1, minDist=20,
                              param1=100,
                              param2=60,
                              minRadius=3,
                              maxRadius=0)

    circles = np.uint16(np.around(circles))  # Округление чисел до целого

    return circles


# Отрисовка кругов на изображении, img - изображение в gray-пространстве, circles - массив найденных кругов,
# каждый элемент имеет формат (x, y, r), х,у - центр круга, r - радиус
def drawCircles(img, circles):
    imgTemp = np.copy(img)
    for circle in circles[0, :]:
        cv.circle(imgTemp, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)

    return imgTemp


# На вход принимается трехканальное изображение и считается среднее значение по изображению,
# например в пространстве HSV в формате average = (H + S + V) / 3
# Возвращает среднее значение
def averageValueOfImage(image):
    averageValue = 0
    count = 0
    for row in image:
        for pixel in row:
            if pixel[0] != 0:
                if pixel[1] != 0:
                    if pixel[2] != 0:
                        averageValue += (int(pixel[0]) + int(pixel[1]) + int(pixel[2])) / 3
                        count += 1

    averageValue /= count
    averageValue = round(averageValue)  # Среднее значение никелевой монеты, округление

    return averageValue


def coinsMarking(coin1, coin2, coins):
    coins_hsv = cv.cvtColor(coins, cv.COLOR_BGR2HSV)
    coin_1_hsv = cv.cvtColor(coin1, cv.COLOR_BGR2HSV)
    coin_2_hsv = cv.cvtColor(coin2, cv.COLOR_BGR2HSV)

    # Размеры шаблонов монет
    coin_1_rows, coin_1_cols = coin_1_hsv.shape[0:2]
    coin_2_rows, coin_2_cols = coin_2_hsv.shape[0:2]

    # Центральные пиксели шаблонов
    center_pixel_coin_1 = coin_1_hsv[round(coin_1_rows / 2)][round(coin_1_cols / 2)]
    center_pixel_coin_2 = coin_2_hsv[round(coin_2_rows / 2)][round(coin_2_cols / 2)]

    # Нахождение среднего значения HSV  монет-шаблонов

    averageCoin_1 = averageValueOfImage(coin_1_hsv)  # Среднее значение никелевой монеты
    averageCoin_2 = averageValueOfImage(coin_2_hsv)  # Среднее значение латунной монеты

    """ 
    Пробовал сначала таким образом, шоб не считать среднее значение всего шаблона, 
    а найти среднее значение только центрального пикселя. Результат в итоге одинаковый
    """
    # averageCoin_2 = (int(center_pixel_coin_2[0]) + int(center_pixel_coin_2[1]) + int(center_pixel_coin_2[2])) / 3

    circles = findCircles(coins)  # Поиск монет на изображении с монетами из методички

    """
    Берется центральный пиксель каждой монеты, считается его средне-канальное значение HSV. 
    Затем сравнивается с средними значениями шаблонов, к какому значению ближе - такая и монета. 
    """
    for circle in circles[0]:
        averageCircle = (int(coins_hsv[circle[1]][circle[0]][0]) + int(coins_hsv[circle[1]][circle[0]][1]) + int(
            coins_hsv[circle[1]][circle[0]][2])) / 3

        distToNikel = abs(averageCoin_1 - averageCircle)
        distToLatun = abs(averageCoin_2 - averageCircle)

        #  Если монетка ближе к никелевой - то помечается зеленой меткой, в обратном случае красной - латунная
        if (distToNikel < distToLatun + 10):  # +10 - это костыль
            cv.circle(coins, (circle[0], circle[1]), 3, (0, 255, 0), -1)
        else:
            cv.circle(coins, (circle[0], circle[1]), 3, (0, 0, 255), -1)

    drawCirclesTemp = drawCircles(coins, circles)  # Отрисовка монет на изображении

    return drawCirclesTemp



# Первая часть задания
letterA = cv.imread('A.jpg', cv.IMREAD_COLOR) # A.png - изображение буковки А из методички к лабе
letterAGray = cv.cvtColor(letterA, cv.COLOR_BGR2GRAY)

skelet = skeletization(letterA)
res = np.concatenate((skelet, letterA), axis=1)

""" Вывод скелетизированного изображения"""
# cv.imshow('skelet', res)
# cv.waitKey(0)
# cv.destroyWindow('skelet')


# Вторая часть задания
skeletLines = drawSkeletLines(skelet)

cv.imshow('SkeletLines', skeletLines)
cv.waitKey(0)
cv.destroyWindow('SkeletLines')

""" 
Это не сделал как раз еще
# Третья часть задания
connectLines(skelet)
skeletLines = drawSkeletLines(skelet)
cv.imshow('SkeletLines', skelet)
cv.waitKey(0)
cv.destroyWindow('SkeletLines')
"""

# Четвертая часть задания
coins_all = cv.imread('coins.jpg', cv.IMREAD_COLOR) # Загрузка изображения с монетами, взято из методички
coin_1 = cv.imread('nikel.jpg', cv.IMREAD_COLOR) # шаблон никелевой монеты из методички
coin_2 = cv.imread('latun.jpg', cv.IMREAD_COLOR) # шаблон латунной монеты из методички

markingCoins = coinsMarking(coin_1, coin_2, coins_all)

cv.imshow('markingCoins', markingCoins)
cv.waitKey(0)
cv.destroyWindow('markingCoins')
