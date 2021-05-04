import cv2

cap = cv2.VideoCapture("vid/bdd_vid_2_25fps.avi")

while True:

	# Start time
	# start = time.time()
	
	ret, frame = cap.read()

	if not ret:
		print("Error: failed to capture image")
		break

	else:
		print("Read imag ok!!!")