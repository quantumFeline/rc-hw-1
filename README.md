Implementation of [this lab](https://mim-ml-teaching.github.io/public-rc-2025-26/lab5-public/lab5.html)

Self-link: https://github.com/quantumFeline/rc-hw-1

Part 1: Camera calibration
---

`uv run task1_camera_calibration.py`

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

* Further improvement with marker detection on the blurry image is unlikely.
We checked whether relaxing parameters (cornerRefinementWinSize, cornerRefinementMaxIterations, minMarkerPerimeterRate)
or converting image to grayscale would result in a significant improvement, but no difference
was found; these adjustments are not included in the final version of the code.

* There are two possible approaches here: go directly off the list
of ArUco markers or, with potentially higher accuracy, use the detection
specific for CharUco. We can try both approaches to compare.

* As `calibrateCameraAruco` and `calibrateCameraCharuco` no longer
exist for OpenCV 5.12 which is used for this lab, we can use `GridBoard.matchImagePoints`
instead in order to create object and image points in the format necessary for
the calibration.

* Importantly, we need to set up `setLegacyPattern(True)` is order for the calibration
to function (you can see a detailed explanation [here](https://stackoverflow.com/questions/73829313/opencv-interpolatecornerscharuco-giving-bad-estimation-when-board-is-partially)).
This is due to ChArUco boards design being changed, and us using a newer library.

* An alternative is to use camera calibration based only on markers 
rather than board corners, but that proved to be ineffective, leading to near-zero camera matrix.

* A visual test of the undistortion check has been run with the following result:

![undistorted image](./undistorted.png "Non-detection")

Example undistorted image from the main set:

![undistorted image](./example_undistorted.png "Example of an undistorted image")

Parts 2-3: Projective transformation
---

`uv run task2_3_projective_transformation_test.py`

We need to calculate a projected image based on the source image and the matrix.
For that, we can simply iterate over each destination pixel and, using the inverse
of the matrix, calculate the source pixel to take the colour from.

To find the homography matrix, we can create a matrix representing
the system of linear equations to solve and then find the least squares solution.

To test, we create a random set of source points, a random matrix, and calculate
the destination points by applying the matrix to them. We normalize the z-value of the
coordinate vector at each step. Then we restore the matrix and ensure it
matches the original one.

We repeat this 100 times to exclude the possibility of flakiness.

Parts 4: Manual projective transformation
---

`uv run task4_manual_projective_transformation.py`

(Note: `ginput` is not supported for PyCharm, which was used for this homework.)

To make our lives easier, we can bind the mouse click on the OpenCV preview
to a callback function. This won't give you sub-pixel coordinates, however,
this way, we can get more pair of points relatively effortlessly.

For example, by clicking, we can find the following coordinates in the first image of the first pair:

`[(340, 638), (743, 633), (261, 695), (270, 793), (169, 800), (165, 696), (335, 421), (337, 615), (157, 623), (152, 428)]`

![points](./point_collection.png "Point collection")

For the second image, the same points have the coordinates:

`[(399, 788), (795, 797), (322, 847), (333, 937), (240, 941), (230, 850), (392, 585), (395, 765), (225, 763), (220, 581)]]`

For those two sets of points, we run the `find_homography` function from earlier.

An example result for the first pair - with original, our result, and the result
of using the standard OpenCV functions for comparison/ground truth:

![transformation](./Transformation.png "Projective transformation, manual")

Part 5: Image stitching 
---

`uv run task5_image_stitching.py`

We will use the naive blending by averaging the value where both points are present. The point values for the feature points of the images are hardcoded for this part.

The result of the stitching & blending:

![stitching](./Stitched.png "Stitched image (pair 3)")

We see that there are some artefacts at the edge of the overlaid image, however, the overall merge around the feature points looks great.

Not every image pair is so lucky, though - for pair 1 we see significant blending issues:

![failing stitching](./Stitched1.png "Stitched images (pair 1)")

Possibly this is fixable by more sophisticated blending, however, the author of this lab opted for leaving it at that due to time constraints. (Maybe at some points later though!)

Part 6: ORB & RANSAC
---

`uv run task6_ORB.py`

For this part, we will use the standard OpenCV instruments for everything except image stitching, as indicated. We are using KNN matches during the match search, as the standard matches has significant issues with monotonous patterns, such as with the books:

![bad keypoints](./keypoints_match.png "Example of the standard match algorithm struggling with the search")

To improve the matches, we will set the images to grayscale during matching. The code also allows to pass greyscale images from the start, if we want to see the corresponding result.

The second pair of images is the most challenging due to low overlap between the pair. To counter this, we increased the threshold in the Lowe's ratio test and increase the number of features.

The stitching results:

![stitching pair 1](./stitched1.png "First pair")

![stitching pair 2](./stitched2.png "Second pair")

![stitching pair 3](./stitched3.png "Third pair")

The result is comparable to the result of the manual algorithm from the previous part, both in the more and less successful parts. This may indicate, again, need for better blending, but also limitations of the images quality itself. 