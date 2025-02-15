from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import json
import threading

app = Flask(__name__)

# Initialize YOLO model
model = YOLO('yolov8n.pt')

# Open webcam
cap = cv2.VideoCapture(0)  # 0 represents the default webcam
show_webcam = False  # Flag to control webcam display
lock = threading.Lock()  # Lock for thread synchronization
detected_objects = []  # List to store detected object sentences


def object_detection():
    global show_webcam, detected_objects

    while True:
        with lock:
            if show_webcam:
                ret, frame = cap.read()  # Read frame from webcam
                if not ret:
                    print("Error: Unable to read frame from webcam.")
                    break

                # Perform object detection on the frame
                try:
                    results = model(frame)
                    result = results[0]
                except Exception as e:
                    print("Error during object detection:", e)
                    break

                # Get frame dimensions
                frame_height, frame_width, _ = frame.shape

                # Clear detected objects list
                detected_objects.clear()

                # Iterate through detected objects
                for box in result.boxes:
                    class_id = result.names[box.cls[0].item()]
                    cords = box.xyxy[0].tolist()
                    cords = [round(x) for x in cords]
                    center_x = round((cords[0] + cords[2]) / 2)
                    center_y = round((cords[1] + cords[3]) / 2)
                    width = cords[2] - cords[0]
                    height = cords[3] - cords[1]
                    conf = round(box.conf[0].item(), 2)

                    # Calculate midpoint and approximate distance
                    mid_x = (cords[0] + cords[2]) / 2
                    mid_y = (cords[1] + cords[3]) / 2
                    apx_distance = round(((1 - (width / frame_width))**1), 2)

                    # Generate result sentences for each detected object
                    result_sentence = f"The {class_id} is present at {apx_distance} meters."
                    detected_objects.append(result_sentence)

                    # Draw bounding box and label on the frame
                    cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f'{class_id} {conf}', (cords[0], cords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0), 3)
                    cv2.putText(frame, f'Distance: {apx_distance}', (cords[0], cords[1] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 3)

                # Encode the frame as JPEG
                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret:
                    print("Error: Unable to encode frame as JPEG.")
                    break
                frame_bytes = jpeg.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # If webcam is not shown, yield None
                yield None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(object_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle_webcam')
def toggle_webcam():
    global show_webcam
    show_webcam = not show_webcam
    return jsonify({'status': 'success', 'show_webcam': show_webcam})


@app.route('/show_results')
def show_results():
    global detected_objects
    return jsonify(detected_objects)


if __name__ == '__main__':
    app.run(debug=True)