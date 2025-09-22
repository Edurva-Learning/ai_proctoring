import cv2
from ultralytics import YOLO

# Load YOLOv8 pre-trained model (you can choose yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
model = YOLO("yolov8n.pt")  # n=Nano (fast), s=Small, m=Medium, l=Large, x=Extra Large

def detectObject(frame):
    """
    Detect objects in the given frame using YOLOv8.
    Returns a list of detected objects with labels and confidence.
    """

    results = model(frame, verbose=False)  # Run YOLOv8 on the frame
    labels_this_frame = []

    for result in results:
        for box in result.boxes:
            # Get bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Add to results
            labels_this_frame.append((label, conf))

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return labels_this_frame


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        objects = detectObject(frame)
        print("Detected:", objects)

        cv2.imshow("YOLOv8 Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
