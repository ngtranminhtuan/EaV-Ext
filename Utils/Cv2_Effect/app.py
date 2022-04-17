import cv2


def rectangle(img, start_point, end_point, color, thickess, line_ratio=0.1):

    left, top       = start_point
    right, bottom   = end_point

    vertical_length     = bottom - top
    horizontal_length   = right - left

    box_horizontal_length = int(horizontal_length * line_ratio)
    box_vertical_length   = int(vertical_length   * line_ratio)

    img = cv2.line(img, (left,top), (left+box_horizontal_length,top), color, thickess)
    img = cv2.line(img, (left,top), (left,top+box_vertical_length), color, thickess)

    img = cv2.line(img, (left+horizontal_length,top), (left+horizontal_length-box_horizontal_length,top), color, thickess)
    img = cv2.line(img, (left+horizontal_length,top), (left+horizontal_length,top+box_vertical_length), color, thickess)
    
    img = cv2.line(img, (left,bottom), (left,bottom-box_vertical_length), color, thickess)
    img = cv2.line(img, (left,bottom), (left+box_horizontal_length,bottom), color, thickess)

    img = cv2.line(img, (right,bottom), (right-box_horizontal_length,bottom), color, thickess)
    img = cv2.line(img, (right,bottom), (right,bottom-box_vertical_length), color, thickess)
    
    return img

def putText(img, text, start_point, end_point, color):
    left, top       = start_point
    right, bottom   = end_point

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    text_width  = text_size[0]
    text_height = text_size[1]

    text_x = left + ((right-left - text_width) // 2)
    text_y = top  - 10

    # Fill text with rectangle
    # img = cv2.rectangle(img, (left,top-text_height-15), (right,top), (255,0,0), -1)

    img = cv2.putText(img, text, (text_x,text_y), font, font_scale, color, 1, cv2.LINE_AA)
    return img

def rectangle_with_text(img, text, start_point, end_point, color, thickess, line_ratio=0.1):
    img = rectangle(img, start_point, end_point, color, thickess, line_ratio)
    img = putText(img, text, start_point, end_point, color)
    return img

def rectangle_with_text_fade(img, text, start_point, end_point, color, thickess, line_ratio=0.1, steps=5):
    imgs = []
    for i in reversed(range(0, steps)):
        temp_img = img.copy()

        left, top       = start_point
        right, bottom   = end_point

        left   = left   - int(left * (i/10))
        top    = top    - int(top * (i/10))
        right  = right  + int(right * (i/10))
        bottom = bottom + int(bottom * (i/10))

        temp_img = rectangle(temp_img, (left,top), (right,bottom), color, thickess, line_ratio)
        temp_img = putText(temp_img, text, (left,top), (right,bottom), color) 
        
        imgs.append(temp_img)
    return imgs

def rectangle_fade(img, start_point, end_point, color, thickess, line_ratio=0.1, steps=5):

    imgs = []
    for i in reversed(range(0, steps)):
        temp_img = img.copy()

        left, top       = start_point
        right, bottom   = end_point

        left   = left   - int(left * (i/10))
        top    = top    - int(top * (i/10))
        right  = right  + int(right * (i/10))
        bottom = bottom + int(bottom * (i/10))

        temp_img = rectangle(temp_img, (left,top), (right,bottom), color, thickess, line_ratio)
        imgs.append(temp_img)
    return imgs

def rounded_rectangle(src, top_left, bottom_right, radius=1, color=255, thickness=1, line_type=cv2.LINE_AA):

    #  corners:
    #  p1 - p2
    #  |     |
    #  p4 - p3

    p1 = top_left
    p2 = (bottom_right[1], top_left[1])
    p3 = (bottom_right[1], bottom_right[0])
    p4 = (top_left[0], bottom_right[0])

    height = abs(bottom_right[0] - top_left[1])

    if radius > 1:
        radius = 1

    corner_radius = int(radius * (height/2))

    if thickness < 0:

        #big rect
        top_left_main_rect = (int(p1[0] + corner_radius), int(p1[1]))
        bottom_right_main_rect = (int(p3[0] - corner_radius), int(p3[1]))

        top_left_rect_left = (p1[0], p1[1] + corner_radius)
        bottom_right_rect_left = (p4[0] + corner_radius, p4[1] - corner_radius)

        top_left_rect_right = (p2[0] - corner_radius, p2[1] + corner_radius)
        bottom_right_rect_right = (p3[0], p3[1] - corner_radius)

        all_rects = [
        [top_left_main_rect, bottom_right_main_rect], 
        [top_left_rect_left, bottom_right_rect_left], 
        [top_left_rect_right, bottom_right_rect_right]]

        [cv2.rectangle(src, rect[0], rect[1], color, thickness) for rect in all_rects]

    # draw straight lines
    cv2.line(src, (p1[0] + corner_radius, p1[1]), (p2[0] - corner_radius, p2[1]), color, abs(thickness), line_type)
    cv2.line(src, (p2[0], p2[1] + corner_radius), (p3[0], p3[1] - corner_radius), color, abs(thickness), line_type)
    cv2.line(src, (p3[0] - corner_radius, p4[1]), (p4[0] + corner_radius, p3[1]), color, abs(thickness), line_type)
    cv2.line(src, (p4[0], p4[1] - corner_radius), (p1[0], p1[1] + corner_radius), color, abs(thickness), line_type)

    # draw arcs
    cv2.ellipse(src, (p1[0] + corner_radius, p1[1] + corner_radius), (corner_radius, corner_radius), 180.0, 0, 90, color ,thickness, line_type)
    cv2.ellipse(src, (p2[0] - corner_radius, p2[1] + corner_radius), (corner_radius, corner_radius), 270.0, 0, 90, color , thickness, line_type)
    cv2.ellipse(src, (p3[0] - corner_radius, p3[1] - corner_radius), (corner_radius, corner_radius), 0.0, 0, 90,   color , thickness, line_type)
    cv2.ellipse(src, (p4[0] + corner_radius, p4[1] - corner_radius), (corner_radius, corner_radius), 90.0, 0, 90,  color , thickness, line_type)

    return src

if __name__ == "__main__":
    
    img = cv2.imread("img.jpg")

    # Rectangle
    # img = rectangle(img, (100,100), (300,300), (255,0,0), 1)
    # cv2.imshow("Result", img)
    # cv2.waitKey(0)

    # Rectangle with text
    # img = rectangle_with_text(img,"Test Class" , (100,100), (300,300), (255,0,0), 1)
    # cv2.imshow("Result", img)
    # cv2.waitKey(0)

    # Fade
    # imgs = rectangle_fade(img, (100,100), (200,200), (255,0,0), 1, steps=5)
    # for img in imgs:
    #     cv2.imshow("Result", img)
    #     cv2.waitKey(1000)

    # Fade with text
    imgs = rectangle_with_text_fade(img, "Test Class", (100,100), (200,200), (255,0,0), 1, line_ratio=0.1, steps=5)
    for img in imgs:
        cv2.imshow("Result", img)
        cv2.waitKey(1000)
