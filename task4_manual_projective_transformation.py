import cv2
import numpy as np
import task2_3_projective_transformation as t2p

WINDOW_NAME = "Collecting points..."
BLUE = (255, 0, 0)

image_pair = [
    cv2.imread("output/set_3_1.jpg"),
    cv2.imread("output/set_3_2.jpg")
]


def get_coords(images: list) -> list[list[int]]:
    """
    Collect feature point coordinates from an image by clicking.
    :param images: list of images to mark points at.
    :return: a list of lists of points for each image.
    """
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

if __name__ == "__main__":
    coords = get_coords(image_pair)

    corners_from, corners_to = coords[0], coords[1]

    assert len(corners_from) == len(corners_to), "Please check the same amount of points on both images!"

    print(corners_from)
    print(corners_to)
    matrix = t2p.Transformer.find_homography(corners_from, corners_to)
    transformer = t2p.Transformer("./output/")
    transformer.check_projective_transformation("set_3_1.jpg", np.linalg.inv(matrix))