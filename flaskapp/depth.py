from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import imutils.video
import numpy as np

app = Flask(__name__)

# Download the MiDaS model
midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid')
midas.to('cpu')
midas.eval()

# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.dpt_transform


# Function to perform depth estimation
def estimate_depth(frame):
    img = transform(frame).to('cpu')
    with torch.no_grad():
        prediction = midas(img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(frame.shape[0] // 2, frame.shape[1] // 2),  # Reduce resolution
            mode='bicubic',
            align_corners=False
        ).squeeze()
    return prediction


# Function to generate video frames
def generate_frames():
    # Set up video capture from webcam
    vs = imutils.video.VideoStream(src=0).start()

    # Initialize counter for throttling
    frame_count = 0

    while True:
        frame = vs.read()
        if frame is None:
            break

        # Throttle processing to every 5th frame
        frame_count += 1
        if frame_count % 5 != 0:
            continue

        # Perform depth estimation
        depth_map = estimate_depth(frame)

        # Convert depth map to a NumPy array
        depth_map_numpy = depth_map.numpy()

        # Normalize the depth map
        depth_map_normalized = cv2.normalize(depth_map_numpy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Encode depth map to JPEG format
        ret, buffer = cv2.imencode('.jpg', depth_map_normalized)
        depth_map_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + depth_map_bytes + b'\r\n')

    # Release the video stream
    vs.stop()


@app.route('/')
def index():
    return render_template('dpt.html')


@app.route('/depth_map')
def depth_map():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
