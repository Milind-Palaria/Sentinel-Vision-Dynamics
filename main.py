import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np

from win10toast import ToastNotifier
from twilio.rest import Client

# Replace these values with your Twilio account SID and auth token
# account_sid = TWILIO_SID
# auth_token = TWILIO_AUTH_TOKEN

# Initialize the Twilio client
client = Client(account_sid, auth_token)



ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args


def send_sms(body, to_phone_number):
    # from_phone_number = TWILIO_PHONE_NUMBER

    # Send the message
    message = client.messages.create(
        body=body,
        from_=from_phone_number,
        to=to_phone_number
    )
    print(f"Message sent: {message.sid}")


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    toaster = ToastNotifier()
    

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        # detections = detections[detections.class_id != 0]
        if (detections[detections.class_id == 42]):
            send_sms("Dangerous Object Detected! => Fork, A Harmful object has been detected in the surveillance. Please look in this matter", "+917905636613")   
            toaster.show_toast(
             "Dangerous Object Detected! => Fork",
             "A Harmful object has been detected in the surveillance. Please look in this matter",
            duration=10  # duration in seconds
            )

        elif (detections[detections.class_id == 43]):
            send_sms("Dangerous Object Detected! => Knife, A Harmful object has been detected in the surveillance. Please look in this matter", "+917905636613") 
            toaster.show_toast(
             "Dangerous Object Detected! => Knife",
             "A Harmful object has been detected in the surveillance. Please look in this matter",
            duration=10  # duration in seconds
            )

        elif (detections[detections.class_id == 76]):
            send_sms("Dangerous Object Detected! => Scissors, A Harmful object has been detected in the surveillance. Please look in this matter", "+917905636613") 
            toaster.show_toast(
             "Dangerous Object Detected! => Scissors",
             "A Harmful object has been detected in the surveillance. Please look in this matter",
            duration=10  # duration in seconds
            )


        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)      
        
        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()