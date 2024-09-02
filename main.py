from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64

app = Flask(__name__)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

# Path to the default necklace image
default_necklace_image_path = 'static/necklace_1.png'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Check if 'frame' is in the request
    if 'frame' not in request.files:
        return jsonify({"error": "Missing frame"}), 400

    # Read the frame from the request
    file = request.files['frame']
    npimg = np.fromfile(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Check if a custom necklace image is provided, else use the default image
    if 'necklace_image' in request.files:
        jewellery_necklace = request.files['necklace_image'].read()
        necklace_image = cv2.imdecode(np.frombuffer(jewellery_necklace, np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        necklace_image = cv2.imread(default_necklace_image_path, cv2.IMREAD_UNCHANGED)
        if necklace_image is None:
            return jsonify({"error": "Default necklace image not found"}), 500

    # Process the frame with the provided or default necklace image
    result = process_necklace_design(frame, necklace_image)

    # Encode the result as a JPEG image
    _, buffer = cv2.imencode('.jpg', result)
    response_data = {
        'image': base64.b64encode(buffer).decode('utf-8')
    }

    return jsonify(response_data)

def process_necklace_design(frame, necklace_image):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            xminC = int(bboxC.xmin * iw)
            yminC = int(bboxC.ymin * ih)
            widthC = int(bboxC.width * iw)
            heightC = int(bboxC.height * ih)

            # Resize the necklace image to fit the detected face
            resized_necklace = cv2.resize(necklace_image, (widthC, heightC))

            necklace_start_x = xminC
            necklace_start_y = yminC + heightC

            # Overlay the necklace image onto the frame
            if resized_necklace.shape[2] == 4:
                alpha_channel = resized_necklace[:, :, 3] / 255.0
                overlay_rgb = resized_necklace[:, :, :3]

                for c in range(3):
                    frame[necklace_start_y:necklace_start_y+heightC, necklace_start_x:necklace_start_x+widthC, c] = \
                        (alpha_channel * overlay_rgb[:, :, c] +
                         (1 - alpha_channel) * frame[necklace_start_y:necklace_start_y+heightC, necklace_start_x:necklace_start_x+widthC, c])
            else:
                frame[necklace_start_y:necklace_start_y+heightC, necklace_start_x:necklace_start_x+widthC] = resized_necklace

    return frame

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80, debug=True)
