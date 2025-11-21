import task1_camera_calibration
import cv2

WINDOW_NAME = "Collecting points..."
BLUE = (255, 0, 0)

# Assuming you have your undistorted images
undistorted = [
    cv2.imread("output/set_1_1.jpg"),
    cv2.imread("output/set_1_2.jpg")
]


def get_coords(images):
    all_coords = []

    for idx, image in enumerate(images):
        print(f"\nImage {idx + 1}/{len(images)}")
        print("Double-click to mark points\n"
              "Press ENTER to move to next image\n"
              "Press ESC to quit")

        current_image = image.copy()
        feature_points = []

        def draw_circle(event, point_x, point_y, flags, param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                cv2.circle(current_image, (point_x, point_y), 5, BLUE, -1)
                cv2.putText(current_image, str(len(feature_points)), (point_x + 10, point_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow(WINDOW_NAME, current_image)
                feature_points.append((point_x, point_y))
                print(f"  Point {len(feature_points)}: ({point_x}, {point_y})")

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, draw_circle)
        cv2.imshow(WINDOW_NAME, current_image)

        k = cv2.waitKey(0) & 0xFF
        if k == 27:  # ESC
            break

        all_coords.append(feature_points)

    cv2.destroyAllWindows()
    return all_coords


print(get_coords(undistorted))
