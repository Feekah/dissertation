#Import necessary libraries
from flask import Flask, render_template, make_response, Response
import mediapipe as mp
import cv2
import pandas as pd
import pickle
import numpy as np
import csv
import joblib
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from mainprogram import calculate_angle, export_landmark_to_csv, preprocess_data, predict_rep, identify_most_common_label

#Initialize the Flask app
app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7,min_tracking_confidence=0.5)



csv_doc = "your_dataset.csv"


camera = cv2.VideoCapture(0)      
def gen():
    squat_counter = 0
    squat_stage = 'start'
    feed = None
    buffered_data = []
    movement = 0
    dataHasbeenProcessedOnce = False
   
    """Video streaming generator function."""
    
    while True:
        ret, frame = camera.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Recolor image to RGB
        image.flags.writeable = False
        results = pose.process(image) # Make detection
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Recolor back to BGR
        image.flags.writeable = True
      # Extract landmarks

        try:
          lms = results.pose_landmarks.landmark
          hip = [lms[mp_pose.PoseLandmark.LEFT_HIP.value].x, lms[mp_pose.PoseLandmark.LEFT_HIP.value].y]
          knee = [lms[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lms[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
          ankle = [lms[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lms[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
          shoulder = [lms[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lms[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
          foot_index = [lms[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, lms[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

          # Calculate angle
          angle = calculate_angle(hip, knee, ankle)
          angle2 = calculate_angle(shoulder, hip, knee)
          angle3 = calculate_angle(knee, ankle, foot_index)

          # Visualize angle
          cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(hip, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

          if angle < 150:
            movement = 1
            squat_stage = 'Down'
            buffered_data.append((results, squat_stage, angle, angle2, angle3))
            dataHasbeenProcessedOnce = False
          elif angle > 150 and angle <= 172 and squat_stage != 'start':
              movement  = movement+1 if movement == 1 and squat_stage != 'Up' else movement
              squat_counter = squat_counter + 1 if movement == 2 and squat_stage != 'Up' else squat_counter
              squat_stage = 'Up'
              buffered_data.append((results, squat_stage, angle, angle2, angle3))

              # Clear the buffer to store coordinates for the new repetition
          elif angle > 172 and squat_stage == 'Up' and dataHasbeenProcessedOnce == False and movement == 2:
              export_landmark_to_csv(csv_doc, buffered_data)
              # Preprocess the CSV data and feed it into the ML model
              processed_data = preprocess_data(csv_doc)
              predictions = predict_rep(processed_data)
              feed = identify_most_common_label(predictions)
              print("Results for this repetition:", feed)
              buffered_data.clear()

              dataHasbeenProcessedOnce = True

        except:
          pass

        cv2.rectangle(image, (0, 0), (280, 73), (280, 117, 16), -1)

        # Rep data for count
        cv2.putText(image, 'COUNT', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(squat_counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2, cv2.LINE_AA)

        # Rep data for Form
        cv2.putText(image, 'FORM', (90, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(feed),
                    (80, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2, cv2.LINE_AA)

        # Rep data for Stage
        cv2.putText(image, 'STAGE', (150, 12),  # Adjusted X-coordinate
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(squat_stage),
                    (150, 60),  # Adjusted X-coordinate
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2, cv2.LINE_AA)








        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
       
        ret, buffer = cv2.imencode('.jpeg', image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


           

@app.route('/video_feed', methods=['GET', 'POST'])

def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """ <head>
    <title>Live Streaming Demonstration</title>
    </head>
    <body>
    <div class="container">
        <div>
            <div><div><h2> <button id="start" >Start</button></h2>
                 <h2> <button id="stop" >Stop</button></h2></div>
                <h3>Live Streaming</h3>
                  <img id="videoCamera" src="" alt="click start to activate webcam" height="70%">
            </div>
        </div>
    </div>
    <script>
     document.getElementById("start").addEventListener("click",function(){
        document.getElementById("videoCamera").src="http://localhost:5000/video_feed";
      });

       document.getElementById("stop").addEventListener("click",function(){
        document.getElementById("videoCamera").src="";
      });
   </script>
    </body>
    """

if __name__ == "__main__":
    app.run(debug=True)