from flask import Flask, render_template, request, redirect, url_for
from RESULT_CODE import actions, model, mp_holistic, prob_viz, mediapipe_detection, draw_styled_landmarks, extract_keypoints
from flask_socketio import SocketIO, emit
from io import StringIO, BytesIO
from PIL import Image
import base64
import imutils
import numpy as np
import cv2
# from flask.ext.cors import CORS

async_mode = None

app = Flask(__name__)
socket_ = SocketIO(app, engineio_logger=True, logger=True, async_mode=async_mode, cors_allowed_origins=['http://127.0.0.1:5000'])

signs = ['No', 'Yes', 'Thankyou', 'Goodbye', 'Hello', 'Iloveyou', 'Please', 'Sorry', 'Youarewelcome']
i = 0

sequence = []
sentence = []
sign = []
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
out = cv2.VideoWriter('outpu1t.avi', fourcc, 20.0, (640, 480))




@socket_.on('image')
def image(data_image):
	sbuf = StringIO()
	sbuf.write(data_image)

	# decode and convert into image
	b = BytesIO(base64.b64decode(data_image))
	pimg = Image.open(b)

	## converting RGB to BGR, as opencv standards
	frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

	# Process the image frame
	frame = imutils.resize(frame, width=700)
	frame = cv2.flip(frame, 1)
	imgencode = cv2.imencode('.jpg', frame)[1]

	# base64 encode
	stringData = base64.b64encode(imgencode).decode('utf-8')
	b64_src = 'data:image/jpg;base64,'
	stringData = b64_src + stringData

	# emit the frame back
	emit('response_back', stringData)
 
	global sequence
	global sentence
	global sign

	colors = [(245,117,16), (117,245,16), (16,117,245)]
	threshold = 0.8
	# # from RESULT_CODE import res
	# if i != 8:
	# 	i = 1 + i
	# else: 
	# 	i = 0
	with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
		if cap.isOpened():

			# Read feed
			ret, frame = cap.read()

			# Make detections
			image, results = mediapipe_detection(frame, holistic)
			# print(results)
			
			# Draw landmarks
			draw_styled_landmarks(image, results)
			
			# 2. Prediction logic
			keypoints = extract_keypoints(results)
	#         sequence.insert(0,keypoints)
	#         sequence = sequence[:30]
	
			sequence.append(keypoints)
			sequence = sequence[-30:]
			
			if len(sequence) == 30:
				res = model.predict(np.expand_dims(sequence, axis=0))[0]
				sign = actions[np.argmax(res)]
				print(actions[np.argmax(res)])
				
				
			#3. Viz logic
				if res[np.argmax(res)] > threshold: 
					if len(sentence) > 0: 
						if actions[np.argmax(res)] != sentence[-1]:
							sentence.append(actions[np.argmax(res)])
					else:
						sentence.append(actions[np.argmax(res)])

				if len(sentence) > 5: 
					sentence = sentence[-5:]

				# Viz probabilities
				image = prob_viz(res, actions, image, colors)
				
			cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
			cv2.putText(image, ' '.join(sentence), (3,30), 
						cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
			out.write(image) 
			# Show to screen
			# cv2.imshow('OpenCV Feed', image)

		# # Break gracefully
		# if cv2.waitKey(10) & 0xFF == ord('q'):
		#     break
	# return render_template('index.html', value = sign) # signs[3]
	emit('update_sign', sign) # signs[3]


# defining home page
@app.route('/')
def homepage():
	return render_template('index.html', sync_mode=socket_.async_mode)
	
# @app.route('/text', methods=['GET', 'POST'])
# def text(comments=[]):
#     if request.method == 'GET':
#         return render_template('index.html', comments=comments)    
#     comments.append(request.form['text_input'])  
#     return redirect(url_for('text'))

# @app.route('/handle_data', methods=['POST'])
# def handle_data():
#     projectpath = request.form['projectFilepath']
#     # your code
#     return (request.form['projectFilePath'])

# @app.route('/my-link/')
# def my_link():
#   print ('I got clicked!')
#   return 'Click.'



if __name__ == '__main__':
	# sequence = []
	# sentence = []
	# threshold = 0.8
	# running app

	socket_.run(app, debug = True)
	#app.run(host = "localhost", port="5000", debug = True)
	print('Done')

	cap.release()
	out.release()
	cv2.destroyAllWindows()