from flask import Flask,render_template,Response
import cv2
import time
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

class_names=['Apple;Apple_scab', 'Apple;Black_rot', 'Apple;Cedar_apple_rust', 'Apple;healthy', 'Blueberry;healthy', 'Cherry;healthy', 'Cherry;Powdery_mildew', 'Corn;Cercospora_leaf_spot_Gray_leaf_spot', 'Corn;Common_rust', 'Corn;healthy', 'Corn;Northern_Leaf_Blight', 'Grape;Black_rot', 'Grape;healthy', 'Heart Leaf Philodendron;citrus_greening', 'Heart Leaf Philodendron;healthy', 'Orange;Citrus_greening', 'Peach;Bacterial_spot', 'Peach;healthy', 'Pepper bell;Bacterial_spot', 'Pepper bell;healthy', 'Polka Dot;healthy', 'Potato;healthy', 'Potato;Late_blight', 'Potato;Early_blight', 'Raspberry;healthy', 'Soybean;healthy', 'Squash;Powdery_mildew', 'Strawberry;healthy', 'Strawberry;Leaf_scorch' 'Tomato;Target_Spot', 'Tomato;Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__;Tomato_mosaic_virus', 'Tomato_;Bacterial_spot', 'Tomato;Early_blight', 'Tomato;healthy', 'Tomato;Late_blight', 'Tomato;Leaf_Mold', 'Tomato;Septoria_leaf_spot']


def prediction_handler(p_prediction):
    prediction = p_prediction.split(';')
    if ("Healthy".lower() in prediction[-1].lower()):
        plant_name = prediction[0]
        plant_status = "Healthy"
        plant_disease = "NIL"
        plant_solution = "NIL"

    else: 
        plant_name = prediction[0]
        if "Unknown" in plant_name:
            plant_status = "Unknown"
            plant_disease = "NIL"
            plant_solution = "Take Another Picture"
            plant_disease = prediction[-1]

        else:
            plant_status = "Unhealthy"
            plant_disease = p_prediction.split(';')[-1]
            plant_solution = "Increase pH level of water"

    return {
        "species": plant_name, "status": plant_status, "disease": plant_disease, "solution": plant_solution
    }

app=Flask(__name__)
camera=cv2.VideoCapture(0)
model = tf.keras.models.load_model('mnet_model_Leaves_tl.h5', custom_objects={'KerasLayer': hub.KerasLayer})

font = cv2.FONT_HERSHEY_SIMPLEX
  

# fontScale
fontScale = 0.35
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 1

def gen_frames():
    
    while True:
        
        time.sleep(1) # delay of 1 second
        success,frame=camera.read() ## read the camera frame
   
        if not success:
            break
        else:
            
            frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
            img_ = np.expand_dims(frame, axis=0)
            pred = model.predict(img_)
            prediction_result = class_names[np.argmax(pred[0])]
            
            if pred[0][np.argmax(pred[0])] < 0.073:
                prediction_result = "Unknown"
            
            handler = prediction_handler(prediction_result)
            
            frame = cv2.copyMakeBorder(
    frame, 90, 0, 0, 100, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            
            frame = cv2.putText(frame, str("species : ")+str(handler['species']), (0,15), font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            frame = cv2.putText(frame, str("status : ")+str(handler['status']), (0,30), font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            frame = cv2.putText(frame, str("disease : ")+str(handler['disease']), (0,45), font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            frame = cv2.putText(frame, str("solution : ")+str(handler['solution']), (0,60), font, 
                   fontScale, color, thickness, cv2.LINE_AA)
            
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

            yield(b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)