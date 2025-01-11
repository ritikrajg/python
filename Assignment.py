import cv2
import torch
import json
from google.colab.patches import cv2_imshow


def load_model(model_path='yolov5s.pt'):

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    return model

def run_inference(model, frame):

    results = model(frame)
    return results.pandas().xyxy[0].to_dict(orient="records")

def create_hierarchical_json(detections):
    structured_results = []
    object_id = 1
    for det in detections:
        main_obj = {
            "object": det['name'],
            "id": object_id,
            "bbox": [det['xmin'], det['ymin'], det['xmax'], det['ymax']],
            "subobject": []
        }
        structured_results.append(main_obj)
        object_id += 1
    return structured_results

def crop_and_save_subobject(frame, bbox, filename):
    x1, y1, x2, y2 = map(int, bbox)
    cropped_img = frame[y1:y2, x1:x2]
    cv2.imwrite(filename, cropped_img)

def process_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = run_inference(model, frame)
        structured_results = create_hierarchical_json(detections)


        with open('output.json', 'w') as f:
            json.dump(structured_results, f, indent=4)


        for det in structured_results:
            bbox = det['bbox']
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(frame, det['object'], (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2_imshow(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = load_model()
    process_video('sample_video.mp4', model)