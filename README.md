Implementation of [this lab](https://mim-ml-teaching.github.io/public-rc-2025-26/lab5-public/lab5.html)

Workflow:

* Run the helper `image_download.py` script.
This will automatically download
all images used in this homework and put them in appropriate
folders. The images are not included in the repository.

* In this lab, we are using images of a ChArUco board, so we
can use the appropriate OpenCV instruments for it, such as
the `interpolateCornersCharuco` functon.

* Create a `camera_calibrator` and detect markers.
We can use to `run_detection_check` to see the detected markers.
In our run, all markers get detected, except in one image that is
partially blurred and only half the markers have been detected:

![image with only half the markers](./nondetection.png "Non-detection")

That is however acceptable for our purposes.

* There are two possible approaches here: go directly off the list
of ArUco markers or, with potentially higher accuracy, use the detection
specific for CharUco. We can try both approaches to compare.

* As `calibrateCameraAruco` and `calibrateCameraCharuco` no longer
exist for OpenCV 5.12 which is used for this lab, we can use `GridBoard.matchImagePoints`
instead in order to create object and image points in the format necessary for
the calibration.