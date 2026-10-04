from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

app = Flask(__name__)

# YOLO model
model = YOLO("yolo11n.pt")

# Traffic signal timer
signal_start_time = time.time()
SIGNAL_DURATION = 5


def get_traffic_signal():
    elapsed = int(time.time() - signal_start_time)
    signal_number = (elapsed // SIGNAL_DURATION) % 3

    if signal_number == 0:
        return "RED"
    elif signal_number == 1:
        return "YELLOW"
    else:
        return "GREEN"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():

    try:

        file = request.files.get("frame")

        if file is None:
            return jsonify({
                "error": "No camera frame received"
            }), 400

        image_bytes = file.read()

        np_array = np.frombuffer(
            image_bytes,
            np.uint8
        )

        frame = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return jsonify({
                "error": "Invalid image"
            }), 400

        # Resize for faster processing
        frame = cv2.resize(
            frame,
            (640, 480)
        )

        # YOLO detection
        results = model(
            frame,
            imgsz=416,
            conf=0.35,
            verbose=False
        )

        boxes = results[0].boxes

        detected_objects = []

        for box in boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            class_name = model.names[class_id]

            detected_objects.append({
                "name": class_name,
                "confidence": round(
                    confidence * 100,
                    1
                )
            })

        signal = get_traffic_signal()

        # Obstacle detected
        if len(detected_objects) > 0:

            status = "STOPPED"
            direction = "STOP"
            obstacle = "DETECTED"
            route = "OBSTACLE DETECTED - STOP"

        # Red signal
        elif signal == "RED":

            status = "STOPPED"
            direction = "STOP"
            obstacle = "NONE"
            route = "RED SIGNAL - STOP"

        # Yellow signal
        elif signal == "YELLOW":

            status = "WAITING"
            direction = "WAIT"
            obstacle = "NONE"
            route = "YELLOW SIGNAL - WAIT"

        # Green signal
        else:

            status = "RUNNING"
            direction = "GO STRAIGHT"
            obstacle = "NONE"
            route = "GREEN SIGNAL - GO"

        return jsonify({

            "status": status,

            "direction": direction,

            "obstacle": obstacle,

            "route": route,

            "signal": signal,

            "objects": detected_objects

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )