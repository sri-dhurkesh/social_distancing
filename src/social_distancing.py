import time
import math
import glob
import yaml
import cv2
import os

vs = cv2.VideoCapture(video_path)
output_video_1, output_video_2 = None, None
# Loop until the end of the video stream
while True:
    # Load the image of the ground and resize it to the correct size
    img = cv2.imread("../img/chemin_1.png")
    bird_view_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Load the frame
    (frame_exists, frame) = vs.read()
    # Test if it has reached the end of the video
    if not frame_exists:
        break
    else:
        # Resize the image to the correct size
        frame = imutils.resize(frame, width=int(size_frame))

        # Make the predictions for this frame
        (boxes, scores, classes) = model.predict(frame)

        # Get the human detected in the frame and return the 2 points to build the bounding box
        array_boxes_detected = get_human_box_detection(boxes, scores[0].tolist(), classes[0].tolist(), frame.shape[0],
                                                       frame.shape[1])

        # Both of our lists that will contain the centroïds coordonates and the ground points
        array_centroids, array_groundpoints = get_centroids_and_groundpoints(array_boxes_detected)

        # Use the transform matrix to get the transformed coordonates
        transformed_downoids = compute_point_perspective_transformation(matrix, array_groundpoints)

        # Show every point on the top view image
        for point in transformed_downoids:
            x, y = point
            cv2.circle(bird_view_img, (x, y), BIG_CIRCLE, COLOR_GREEN, 2)
            cv2.circle(bird_view_img, (x, y), SMALL_CIRCLE, COLOR_GREEN, -1)

        # Check if 2 or more people have been detected (otherwise no need to detect)
        if len(transformed_downoids) >= 2:
            for index, downoid in enumerate(transformed_downoids):
                if not (downoid[0] > width or downoid[0] < 0 or downoid[1] > height + 200 or downoid[1] < 0):
                    cv2.rectangle(frame, (array_boxes_detected[index][1], array_boxes_detected[index][0]),
                                  (array_boxes_detected[index][3], array_boxes_detected[index][2]), COLOR_GREEN, 2)

            # Iterate over every possible 2 by 2 between the points combinations
            list_indexes = list(itertools.combinations(range(len(transformed_downoids)), 2))
            for i, pair in enumerate(itertools.combinations(transformed_downoids, r=2)):
                # Check if the distance between each combination of points is less than the minimum distance chosen
                if math.sqrt((pair[0][0] - pair[1][0]) ** 2 + (pair[0][1] - pair[1][1]) ** 2) < int(distance_minimum):
                    # Change the colors of the points that are too close from each other to red
                    if not (pair[0][0] > width or pair[0][0] < 0 or pair[0][1] > height + 200 or pair[0][1] < 0 or
                            pair[1][0] > width or pair[1][0] < 0 or pair[1][1] > height + 200 or pair[1][1] < 0):
                        change_color_on_topview(pair)
                        # Get the equivalent indexes of these points in the original frame and change the color to red
                        index_pt1 = list_indexes[i][0]
                        index_pt2 = list_indexes[i][1]
                        cv2.rectangle(frame, (array_boxes_detected[index_pt1][1], array_boxes_detected[index_pt1][0]),
                                      (array_boxes_detected[index_pt1][3], array_boxes_detected[index_pt1][2]),
                                      COLOR_RED, 2)
                        cv2.rectangle(frame, (array_boxes_detected[index_pt2][1], array_boxes_detected[index_pt2][0]),
                                      (array_boxes_detected[index_pt2][3], array_boxes_detected[index_pt2][2]),
                                      COLOR_RED, 2)

    # Draw the green rectangle to delimitate the detection zone
    draw_rectangle(corner_points)
    # Show both images
    cv2.imshow("Bird view", bird_view_img)
    cv2.imshow("Original picture", frame)

    key = cv2.waitKey(1) & 0xFF

    # Write the both outputs video to a local folders
    if output_video_1 is None and output_video_2 is None:
        fourcc1 = cv2.VideoWriter_fourcc(*"MJPG")
        output_video_1 = cv2.VideoWriter("../output/video.avi", fourcc1, 25, (frame.shape[1], frame.shape[0]), True)
        fourcc2 = cv2.VideoWriter_fourcc(*"MJPG")
        output_video_2 = cv2.VideoWriter("../output/bird_view.avi", fourcc2, 25,
                                         (bird_view_img.shape[1], bird_view_img.shape[0]), True)
    elif output_video_1 is not None and output_video_2 is not None:
        output_video_1.write(frame)
        output_video_2.write(bird_view_img)

    # Break the loop
    if key == ord("q"):
        break