from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import RequestContext, loader
from django.http.response import StreamingHttpResponse

import cv2
import numpy as np
import datetime
import time

def index(request):
	template = loader.get_template('index.html')
	return HttpResponse(template.render({}, request))

def stream():
 
	# cap = cv2.VideoCapture(0)

	cap = cv2.VideoCapture("webcam/vid/bdd_vid_2_25fps.avi")
	ret, frame = cap.read()
	h, w, _ = frame.shape
	scale = 0.75

	while True:
		
		ret, frame = cap.read()
		frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

		grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		agv = np.average(grayscale)

		if agv >= 50: light_cond = "daytime"
		elif 20 <= agv < 50: light_cond = "twilight"
		elif 10 <= agv < 20: light_cond = "nighttime with street light"
		else: light_cond = "nighttime without street light"

		if not ret:
			print("Error: failed to capture image")
			break

		else:
			print("Read imag ok!!!", agv)
			cv2.putText(frame,'AGV = ' + str("%.2f" % agv),(10, 30), TEXT_FONT, 0.7, COLOR_YELLOW, 2)
			cv2.putText(frame,'Light Condition: ' + light_cond,(10, 60), TEXT_FONT, 0.7, COLOR_YELLOW, 2)

		time.sleep(0.1)

		cv2.imwrite('demo.jpg', frame)
		yield (b'--frame\r\n'
			   b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')


def video_feed(request):
	return StreamingHttpResponse(stream(), content_type='multipart/x-mixed-replace; boundary=frame')
