import cv2

mouseX, mouseY = 0, 0

undistorted = cv2.imread("output/set_1_1.jpg", "output/set_1_2.jpg")

def get_coords(undistorted):
    global mouseX, mouseY
    cv2.namedWindow("window")

    for image in undistorted:
        current_image = image.copy()

        def draw_circle(event, x, y, flags, param):
            global mouseX, mouseY
            if event == cv2.EVENT_LBUTTONDBLCLK:
                cv2.circle(current_image, (x, y), 10, (255, 0, 0), -1)
                cv2.imshow("window", current_image)
                mouseX, mouseY = x, y

        cv2.setMouseCallback("window", draw_circle)
        cv2.imshow("window", current_image)

        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            break
        elif k == ord('a'):
            print(mouseX, mouseY)
            yield mouseX, mouseY

coords = list(get_coords(undistorted))
print(coords)