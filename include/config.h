#ifndef CONFIG_H
#define CONFIG_H

// #define USE_FP16
#define USE_INT8

#include <string>
#include <vector>

const static char *kInputTensorName = "images";
const static char *kOutputTensorName = "output";
const static int kNumClass = 80;
const static int kBatchSize = 8;
const static int kGpuId = 0;
const static int kInputH = 640;
const static int kInputW = 640;
const static float kNmsThresh = 0.55f;
const static float kConfThresh = 0.45f;
const static int kMaxInputImageSize = 3000 * 3000;
const static int kMaxNumOutputBbox = 1000;

const std::string trtFile = "../weights/yolov8s.engine";

const std::string cacheFile = "./int8calib.table";
const std::string calibrationDataPath = "../images/";  // 存放用于 int8 量化校准的图像

const std::vector<std::string> vClassNames {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear",
    "hair drier", "toothbrush"
};

#endif  // CONFIG_H