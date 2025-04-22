#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main() {
    // Load the YOLOv5 Nano (tiny) ONNX model.
    Net net = readNetFromONNX("../models_onnx/yolov5n.onnx");  // Replace with your model filename if needed.
    if (net.empty()) {
        cerr << "Failed to load model." << endl;
        return -1;
    }
    cout << "YOLOv5 model loaded successfully." << endl;

    // Define class names (COCO dataset for YOLOv5)
    vector<string> classNames = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    // Open the video file. if you need to use a webcam, replace "road8.mp4" with 0.
    // For example, to use a webcam, uncomment the line below:  

    //VideoCapture cap(0);

    // For testing with a video file, use the line below:
    VideoCapture cap("../media/road8.mp4");
    
    // Replace "road8.mp4" with your video path.
    if (!cap.isOpened()) {
        cerr << "Error opening video file." << endl;
        return -1;
    }

    const int netWidth = 640;
    const int netHeight = 640;
    float confThreshold = 0.33f;
    float nmsThreshold = 0.45f;

    while (cap.isOpened()) {
        Mat frame;
        cap.read(frame);

        // If the video ends, break the loop.qq
        if (frame.empty()) {
            cout << "End of video." << endl;
            break;
        }

        // Resize the frame to 640x640 for YOLOv5 input.
        Mat resizedFrame;
        resize(frame, resizedFrame, Size(netWidth, netHeight));

        // Create a blob from the resized image.
        Mat blob;
        blobFromImage(resizedFrame, blob, 1.0/255.0, Size(netWidth, netHeight), Scalar(), true, false);
        net.setInput(blob);

        // Run inference.
        Mat output = net.forward();
        int dimensions = output.size[2];
        int rows = output.size[1];

        vector<Rect> boxes;
        vector<int> classIds;
        vector<float> confidences;

        float* data = (float*)output.data;
        for (int i = 0; i < rows; i++) {
            float obj_conf = data[4];
            if (obj_conf >= confThreshold) {
                Mat scores(1, dimensions - 5, CV_32FC1, data + 5);
                Point classIdPoint;
                double maxScore;
                minMaxLoc(scores, 0, &maxScore, 0, &classIdPoint);
                if (maxScore >= confThreshold) {
                    float cx = data[0];
                    float cy = data[1];
                    float w  = data[2];
                    float h  = data[3];

                    int left = static_cast<int>((cx - w/2) * frame.cols / netWidth);
                    int top  = static_cast<int>((cy - h/2) * frame.rows / netHeight);
                    int boxW = static_cast<int>(w * frame.cols / netWidth);
                    int boxH = static_cast<int>(h * frame.rows / netHeight);

                    boxes.push_back(Rect(left, top, boxW, boxH));
                    classIds.push_back(classIdPoint.x);
                    confidences.push_back(obj_conf);
                }
            }
            data += dimensions;
        }

        // Apply Non-Maximum Suppression (NMS).
        vector<int> indices;
        NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

        // Draw the boxes and labels on the original frame.
        for (int idx : indices) {
            Rect box = boxes[idx];
            rectangle(frame, box, Scalar(0, 255, 0), 2);
            // Get the class name
            string label = format("%s %.2f", classNames[classIds[idx]].c_str(), confidences[idx]);
            int baseline;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            int topLabel = max(box.y, labelSize.height);
            rectangle(frame, Point(box.x, topLabel - labelSize.height),
                      Point(box.x + labelSize.width, topLabel + baseline), Scalar(0, 255, 0), FILLED);
            putText(frame, label, Point(box.x, topLabel), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
        }

        // Display the processed frame.
        imshow("Detection", frame);

        // Press 'q' to exit the video loop.
        if (waitKey(1) == 'q') {
            break;
        }
    }

    // Release the video capture object.
    cap.release();
    destroyAllWindows();

    return 0;
}

