import sys
sys.path.append("../..")

import cv2

from Utils.Config.app import setConfig


def addIcon(orgImg, upIcon, downIcon, numInIcon, upCounter, downCounter, numIn, textSize=1.4, iconSize=(64, 64)):
    orgImg = cv2.cvtColor(orgImg, cv2.COLOR_BGR2BGRA)

    imgH, imgW, channel = orgImg.shape

    paddingIcon = 20
    paddingText = 15

    startX  = int(imgW * 0.025)
    endX    = int(imgW * 0.025) + iconSize[1]

    startY_Up   = int(imgH * 0.025)
    endY_Up     = startY_Up + iconSize[0]

    startY_Down   = endY_Up + paddingIcon
    endY_Down     = startY_Down + iconSize[0]

    startY_NumIn   = endY_Down + paddingIcon
    endY_NumIn     = startY_NumIn + iconSize[0]

    # Draw Up, Down, NumIn Icon
    # Up Icon
    alphaBackground = orgImg[startY_Up:endY_Up, startX:endX, 3] / 255.0
    alphaForeground = upIcon[:, :, 3] / 255.0

    for color in range(0, 3):
        orgImg[startY_Up:endY_Up, startX:endX, color] = \
            alphaForeground * upIcon[:, :, color] + \
            alphaBackground * orgImg[startY_Up:endY_Up, startX:endX, color] \
            * (1 - alphaForeground)

    # Down Icon
    alphaBackground = orgImg[startY_Down:endY_Down, startX:endX, 3] / 255.0
    alphaForeground = downIcon[:, :, 3] / 255.0

    for color in range(0, 3):
        orgImg[startY_Down:endY_Down, startX:endX, color] = \
            alphaForeground * downIcon[:, :, color] + \
            alphaBackground * orgImg[startY_Down:endY_Down, startX:endX, color] \
            * (1 - alphaForeground)

    # NumIn Icon
    alphaBackground = orgImg[startY_Up:endY_Up, startX:endX, 3] / 255.0
    alphaForeground = numInIcon[:, :, 3] / 255.0

    for color in range(0, 3):
        orgImg[startY_NumIn:endY_NumIn, startX:endX, color] = \
            alphaForeground * numInIcon[:, :, color] + \
            alphaBackground * orgImg[startY_NumIn:endY_NumIn, startX:endX, color] \
            * (1 - alphaForeground)
    
    orgImg = cv2.cvtColor(orgImg, cv2.COLOR_BGRA2BGR)
    
    cv2.putText(orgImg, str(upCounter), (endX + paddingText, startY_Up+iconSize[0]//2+paddingText), cv2.FONT_HERSHEY_DUPLEX, textSize, (78,177,251), 2)
    cv2.putText(orgImg, str(downCounter), (endX + paddingText, startY_Down+iconSize[0]//2+paddingText), cv2.FONT_HERSHEY_DUPLEX, textSize, (78,177,251), 2)
    cv2.putText(orgImg, str(numIn), (endX + paddingText, startY_NumIn+iconSize[0]//2+paddingText), cv2.FONT_HERSHEY_DUPLEX, textSize, (78,177,251), 2)

    return orgImg


def selectZone(config, img):
    cv2.namedWindow("Select Gate Position",2)
    r = cv2.selectROI("Select Gate Position", img, fromCenter=True, showCrosshair=True)
    r = [r[0], r[1]]
    r = [str(i) for i in r]
    r = ", ".join(r)
    setConfig(config, "GATE_POSITION", r)
    setConfig(config, "SELECT_GATE_POSITION", "0")
    print("\nGate position saved! Please re-run application.")
    exit()

def isInZone(box, zone):
    xMin, yMin, xMax, yMax                 = box
    xMinZone, yMinZone, xMaxZone, yMaxZone = zone
    Xcentroid = (xMin + xMax)//2
    Ycentroid = (yMin + yMax)//2
    if (xMinZone <= Xcentroid <= xMaxZone) and (yMinZone <= Ycentroid <= yMaxZone):
        return True
    return False

def runNoiseFilter(topList, botList, historyDict, numFilter):

    # Modify history base on top list
    for obj in topList:
        id   = obj[0]
        if str(id) not in historyDict:
            historyDict[str(id)] = ["top"]
        else:
            historyDict[str(id)].append("top")
            if len(historyDict[str(id)]) > numFilter:
                historyDict[str(id)].pop(0)

    # Modify history base on bot list
    for obj in botList:
        id   = obj[0]
        if str(id) not in historyDict:
            historyDict[str(id)] = ["bot"]
        else:
            historyDict[str(id)].append("bot")
            if len(historyDict[str(id)]) > numFilter:
                historyDict[str(id)].pop(0)
        
    # Filter topList
    for obj in topList:
        id   = obj[0]
        numTop = historyDict[str(id)].count("top")
        numBot = historyDict[str(id)].count("bot")
        if numTop < numBot:
            topList.remove(obj)
            botList.append(obj)

    # Filter topList
    for obj in botList:
        id   = obj[0]
        numTop = historyDict[str(id)].count("top")
        numBot = historyDict[str(id)].count("bot")
        if numTop > numBot:
            botList.remove(obj)
            topList.append(obj)

    return topList, botList, historyDict

def countPeople(historyDict, preTopList, preBotList, upCounter, downCounter, numFilter, outputTrackingBboxes, doorPosition, imgW, imgH, topZone, botZone):
    topList = []
    botList = []
    # Decide box in whether list
    for person in outputTrackingBboxes:
        ID = person[4]
        x0, y0, x1, y1 = int(person[0]),int(person[1]),int(person[2]),int(person[3])
        # Ignore invalid bounding boxes
        if (y1-y0 < 1) or (x1-x0 < 1) or (x0<0) or (y0<0) or (x1>=imgW) or (y1>=imgH):
            continue
        # Check bouding boxes in topZone
        box = (x0, y0, x1, y1)
        if isInZone(box, topZone):
            topList.append([ID, box])
        elif isInZone(box, botZone):
            botList.append([ID, box])

    # Filter noise
    topList, botList, historyDict = runNoiseFilter(topList, botList, historyDict, numFilter)

    # Check bouding boxes in topZone
    for preBox in preTopList:
        preID  = preBox[0]
        for box in botList:
            id  = box[0]
            if id == preID:
                downCounter += 1

    # Check bouding boxes in botZone
    for preBox in preBotList:
        preID  = preBox[0]
        for box in topList:
            id  = box[0]
            if id == preID:
                upCounter += 1
    # Compute num people in
    numIn = 0
    if doorPosition == "TOP":
        numIn = downCounter - upCounter
    elif doorPosition == "BOTTOM":
        numIn = upCounter - downCounter
    preTopList = topList
    preBotList = botList
    return (numIn, upCounter, downCounter), (historyDict, preTopList, preBotList)