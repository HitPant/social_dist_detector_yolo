from scipy.spatial import distance as dist
from function_pg.detection import detect_people
import numpy as np
from sms import send_sms
import imutils
import cv2
import json
import os
from datetime import datetime, timedelta
import threading


# check if config file exist
check= os.path.exists('conf.json')
if check == True:
    conf = json.load(open("conf.json")) #loads the values from config file
else:
    conf = json.load(open("default.json")) #default values


Min_distance= conf["MIN_DISTANCE"]
x= True
gpu = False

# load the COCO class labels our YOLO model was trained on
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = "yolov3.weights"
configPath = "yolov3.cfg"

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if gpu == True:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture("ped.mp4")
writer = None

if (vs.isOpened() == False):  # check if camera is initialized correctly
	print("Error opening video stream")  # gives error if not initialized




def rec_vid_sd():
	while vs.isOpened():  # starts a wile loop if initialized correctly
		(grabbed, frame) = vs.read()

		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not grabbed:
			break

		# resize the frame and then detect people (and only people) in it
		frame = imutils.resize(frame, width=500)
		results = detect_people(frame, net, ln,
			personIdx=LABELS.index("person"))

		# initialize the set of indexes that violate the minimum social
		# distance
		violate = set()

		time_rec = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

		# ensure there are *at least* two people detections (required in
		# order to compute our pairwise distance maps)
		if len(results) >= 2:
			# extract all centroids from the results and compute the
			# Euclidean distances between all pairs of the centroids
			centroids = np.array([r[2] for r in results])
			D = dist.cdist(centroids, centroids, metric="euclidean")

			# loop over the upper triangular of the distance matrix
			for i in range(0, D.shape[0]):
				for j in range(i + 1, D.shape[1]):
					# check to see if the distance between any two
					# centroid pairs is less than the configured number
					# of pixels
					if D[i, j] < Min_distance:
						# update our violation set with the indexes of
						# the centroid pairs
						violate.add(i)
						violate.add(j)

		# loop over the results
		for (i, (prob, bbox, centroid)) in enumerate(results):
			# extract the bounding box and centroid coordinates, then
			# initialize the color of the annotation
			(startX, startY, endX, endY) = bbox
			(cX, cY) = centroid
			color = (0, 255, 0)

			# if the index pair exists within the violation set, then
			# update the color
			if i in violate:
				color = (0, 0, 255)

			# draw (1) a bounding box around the person and (2) the
			# centroid coordinates of the person,
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.circle(frame, (cX, cY), 5, color, 1)

		# draw the total number of social distancing violations on the
		# output frame
		text = "Social Distancing Violations: {}".format(len(violate))
		cv2.putText(frame, text, (10, frame.shape[0] - 25),
			cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
		cv2.putText(frame, time_rec, (10, frame.shape[0] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
		# image_path = ("saved_image/" + time_rec + ".jpg")
		# cv2.imwrite(image_path, frame)

		FMT = '%Y-%m-%d %H:%M:%S'
		prev_time = datetime.now() - timedelta(seconds=conf["time_interval"])
		prev_time = prev_time.strftime(FMT)

		curr_time = datetime.now().strftime(FMT)
		tdelta = datetime.strptime(curr_time, FMT) - datetime.strptime(prev_time, FMT)

		if len(violate) != 0 and tdelta.seconds >= conf["time_interval"]:
			t2 = threading.Thread(target=send_sms, args=[len(violate)])  # run seperate thread to send sms alert while monitoring
			t2.start()
			prev_time= curr_time

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

			# 	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			# 	writer = cv2.VideoWriter(args["output"], fourcc, 25,
			# 							 (frame.shape[1], frame.shape[0]), True)
			#
			# # if the video writer is not None, write the frame to the output
			# # video file
			# if writer is not None:
			# 	writer.write(frame)

t1= threading.Thread(target= rec_vid_sd)
t1.start()
