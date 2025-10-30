# Download all images from https://www.mimuw.edu.pl/~ciebie/rc25-26/images/
# and https://www.mimuw.edu.pl/~ciebie/rc25-26/calibration/

import requests
import os
from urllib.parse import quote

URL = "https://www.mimuw.edu.pl/~ciebie/rc25-26/images/"
IMAGE_NAMES = ["set_1_1.jpg", "set_1_2.jpg",
               "set_2_1.jpg", "set_2_2.jpg",
               "set_3_1.jpg", "set_2_2.jpg",]

def download_images(url, folder_name, image_names):
    os.makedirs(folder_name, exist_ok=True)
    for i, image_name in enumerate(image_names):
        response = requests.get(url + quote(image_name))
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(f"./{folder_name}/" + f"{i}.jpg", 'wb') as f:
            f.write(response.content)

        print(f"Image downloaded successfully: {image_name}")

CALIBRATION_URL = "https://www.mimuw.edu.pl/~ciebie/rc25-26/calibration/"
CALIBRATION_IMAGE_NAMES = [
# "Zdjęcie, 28.10.2025 o 17.57 #2.jpg",
# "Zdjęcie, 28.10.2025 o 17.57 #3.jpg",
# "Zdjęcie, 28.10.2025 o 17.57 #4.jpg",
# "Zdjęcie, 28.10.2025 o 17.57.jpg",
# "Zdjęcie, 28.10.2025 o 17.58 #2.jpg",
# "Zdjęcie, 28.10.2025 o 17.58 #3.jpg",
# "Zdjęcie, 28.10.2025 o 17.58.jpg",
# "Zdjęcie, 28.10.2025 o 17.59 #2.jpg",
# "Zdjęcie, 28.10.2025 o 17.59.jpg", - not ARUCO board
"Zdjęcie, 28.10.2025 o 18.03 #2.jpg",
"Zdjęcie, 28.10.2025 o 18.03 #3.jpg",
"Zdjęcie, 28.10.2025 o 18.03 #4.jpg",
"Zdjęcie, 28.10.2025 o 18.03 #5.jpg",
"Zdjęcie, 28.10.2025 o 18.03.jpg",
"Zdjęcie, 28.10.2025 o 18.04 #2.jpg",
"Zdjęcie, 28.10.2025 o 18.04 #3.jpg",
"Zdjęcie, 28.10.2025 o 18.04 #4.jpg",
"Zdjęcie, 28.10.2025 o 18.04.jpg",
"Zdjęcie, 28.10.2025 o 18.05 #2.jpg",
"Zdjęcie, 28.10.2025 o 18.05.jpg"]

if __name__ == "__main__":
    download_images(CALIBRATION_URL, "calibration", CALIBRATION_IMAGE_NAMES)

