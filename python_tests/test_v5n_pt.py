import cv2
import numpy as np
import torch
import time

# python3 python_tests/test_v5n_pt.py

def preprocess_image(image_path, input_size=(640, 640)):
    """
    Reads, resizes, and preprocesses an image for YOLOv5-style input.
    Handles different aspect ratios correctly.

    Args:
        image_path (str): Path to the input image file.
        input_size (tuple): The target size of the input image (width, height).
                        YOLOv5 typically uses (640, 640).

    Returns:
        tuple: A tuple containing:
            - img_resized (numpy.ndarray): Resized image, padded if necessary.
            - img_original (numpy.ndarray): The original, unresized image.
            - ratio (float): The scaling ratio used for resizing.
    """
    img_original = cv2.imread(image_path)  # Read the image using OpenCV
    if img_original is None:
        raise FileNotFoundError(f"Error: Image not found at {image_path}")
    img = img_original.copy()  # Create a copy to avoid modifying the original
    height, width = img.shape[:2]  # Get original image height and width
    input_width, input_height = input_size

    # Calculate the ratio for resizing, maintaining aspect ratio
    ratio = min(input_width / width, input_height / height)
    resized_width, resized_height = int(width * ratio), int(height * ratio)

    # Resize the image
    img_resized = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    # Pad the resized image to the target size with black pixels
    padded_img = np.zeros((input_height, input_width, 3), dtype=np.uint8)
    padded_img[(input_height - resized_height) // 2:(input_height - resized_height) // 2 + resized_height,
               (input_width - resized_width) // 2:(input_width - resized_width) // 2 + resized_width] = img_resized

    # Convert the image to the format expected by PyTorch (CHW)
    img_normalized = padded_img.astype(np.float32) / 255.0
    img_transposed = img_normalized.transpose(2, 0, 1)  # HWC to CHW
    img_tensor = torch.from_numpy(img_transposed).float()
    img_batch = img_tensor.unsqueeze(0)  # Add batch dimension

    return img_batch, img_original, ratio


def postprocess_detections(prediction, img_original, ratio, conf_thres=0.25, iou_thres=0.45):
    """
    Postprocesses the raw output from the YOLOv5 PyTorch model.
    Performs filtering based on confidence and IoU, and adjusts bounding box
    coordinates to the original image size.

    Args:
        prediction (torch.Tensor): The raw output from the YOLOv5 PyTorch model.
        img_original (numpy.ndarray): The original, unresized image.
        ratio (float): The scaling ratio used during preprocessing.
        conf_thres (float): Confidence threshold for filtering detections.
        iou_thres (float): IoU threshold for non-maximum suppression.

    Returns:
        list: A list of detections. Each detection is a list containing
            [x1, y1, x2, y2, confidence, class_id].  The bounding box
            coordinates (x1, y1, x2, y2) are in the coordinate space of the
            original image.
    """
    # Move predictions to CPU and convert to numpy array if it's on the GPU
    prediction = prediction.cpu().numpy()

    # The output shape from the YOLOv5 model is typically (1, num_boxes, 85)
    # where 85 = (x, y, w, h, confidence, class1, class2, ..., class80)
    #  prediction [0,:,:5] will be x,y,w,h, confidence
    #  prediction [0,:,5:] will be class probs

    # Get the height and width of the original image.
    original_height, original_width = img_original.shape[:2]

    # Filter out detections with confidence below the threshold
    conf_mask = prediction[:, 4] > conf_thres  # This line was changed from  prediction[0, :, 4] > conf_thres
    detections = prediction[conf_mask]  # Apply the mask

    if not detections.shape[0]:
        return []

    # Extract bounding box coordinates (x, y, w, h)
    boxes = detections[:, :4]
    # Extract confidence scores
    scores = detections[:, 4:5]
    # Extract class probabilities
    classes = detections[:, 5:]

    # Convert (x, y, w, h) to (x1, y1, x2, y2) format, relative to padded image
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    # Adjust bounding box coordinates to the original image size
    x1 /= ratio
    y1 /= ratio
    x2 /= ratio
    y2 /= ratio

    # Clip the bounding box coordinates to the original image boundaries
    x1 = np.clip(x1, 0, original_width)
    y1 = np.clip(y1, 0, original_height)
    x2 = np.clip(x2, 0, original_width)
    y2 = np.clip(y2, 0, original_height)

    boxes = np.stack((x1, y1, x2, y2), axis=1)

    # Get the class with the highest probability for each detection.
    class_ids = np.argmax(classes, axis=1).reshape(-1, 1)
    confidences = scores * np.take_along_axis(classes, class_ids, axis=1)

    # Combine the bounding boxes, confidences, and class IDs.
    detections = np.concatenate((boxes, confidences, class_ids), axis=1)

    # Perform non-maximum suppression (NMS) to filter out redundant detections.
    keep_indices = nms(boxes, confidences, iou_thres)
    filtered_detections = detections[keep_indices]

    return filtered_detections


def nms(boxes, scores, iou_threshold):
    """
    Performs non-maximum suppression (NMS) on a set of bounding boxes.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes (x1, y1, x2, y2).
        scores (numpy.ndarray): Array of corresponding confidence scores.
        iou_threshold (float): IoU threshold for filtering.

    Returns:
        list: Indices of the boxes to keep after NMS.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.flatten().argsort()[::-1]  # Get indices sorted by scores

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size > 1: # Check if there are more than one element in order
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
        else:
            break

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def draw_detections(img, detections, class_names):
    """
    Draws the bounding boxes, labels, and confidence scores on the image.

    Args:
        img (numpy.ndarray): The image to draw detections on.
        detections (list): A list of detections, where each detection is a list
            containing [x1, y1, x2, y2, confidence, class_id].
        class_names (list): A list of class names.
    """
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers for drawing
        class_name = class_names[int(class_id)]
        label = f"{class_name} {confidence:.2f}"  # Format the label

        # Generate a color for the bounding box and label.
        color = (0, 255, 0)  # Green

        # Draw the bounding box.
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw the label and background.
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


if __name__ == "__main__":
    # Load the YOLOv5 model
    model_path = "models_py/yolov5n.pt"  # Path to your YOLOv5 PyTorch model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    model.eval()  # Set the model to evaluation mode

    # Load class names
    class_names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
        "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair dryer", "toothbrush"
    ]

    # Load and preprocess the image
    image_path = "media/test.jpg"  # Path to your test image
    img_batch, img_original, ratio = preprocess_image(image_path)

    # Perform inference
    start_time = time.time()
    with torch.no_grad():  # Disable gradient calculation for inference
        prediction = model(img_batch)  # Forward pass: Pass the image through the model
        # For YOLOv5, the direct output of the model is used.  No need to get extra items.
        prediction = prediction[0]  # get the first output
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.4f} seconds")

    # Postprocess the outputs to get meaningful detections
    detections = postprocess_detections(prediction, img_original, ratio)

    # Draw
    img_with_detections = draw_detections(img_original.copy(), detections, class_names)

    # Display
    cv2.imshow("Detections", img_with_detections)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save
    cv2.imwrite("detections_pt.jpg", img_with_detections)