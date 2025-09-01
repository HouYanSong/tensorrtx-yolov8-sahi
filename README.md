## C++ TensorRT YOLOv8-SAHI

### Introduction

This project introduces how to implement the high-performance deployment of `YOLOv8-SAHI` with `Int8 Engine` on embedded devices such as `Jetson`. The time consumption for testing the `image slice` and` batch inference` on `Jetson Orin Nano (8GB)` is only `0.04` seconds, more than `20 fps`.

![](https://modelbox-course.obs.cn-north-4.myhuaweicloud.com/tensorrtx-yolov8-sahi/sample.png)

### YOLOv8 Int8 Quantization
1. generate `yolov8s.wts` from pytorch with `yolov8s.pt`

```bash8
pip install ultralytics
python gen_wts.py
```

2. Export `yolov8s.engine` from `yolov8s.wts` with `kBatchSize = 8`
```bash
cmake -S . -B build
cmake --build build
cd build
./yolov8_sahi -s ../weights/yolov8s.wts ../weights/yolov8s.engine s
```
3. See more configs in `include/config.h`
```cpp
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
```
### YOLOv8-SAHI with Int8 Engine 
```bash
cd build
./yolov8_sahi -d ../weights/yolov8s.engine ../images/
```
The performance of YOLOv8-SAHI with Int8 Engine on `Jetson Orin Nano (8GB)` is as follows:
```bash
sample0102.png YOLOv8-SAHI: 1775ms
sample0206.png YOLOv8-SAHI: 46ms
sample0121.png YOLOv8-SAHI: 44ms
sample0058.png YOLOv8-SAHI: 44ms
sample0070.png YOLOv8-SAHI: 44ms
sample0324.png YOLOv8-SAHI: 43ms
sample0122.png YOLOv8-SAHI: 44ms
sample0086.png YOLOv8-SAHI: 45ms
sample0124.png YOLOv8-SAHI: 45ms
sample0230.png YOLOv8-SAHI: 45ms
sample0083.png YOLOv8-SAHI: 45ms
sample0135.png YOLOv8-SAHI: 44ms
sample0288.png YOLOv8-SAHI: 44ms
sample0101.png YOLOv8-SAHI: 44ms
sample0021.png YOLOv8-SAHI: 45ms
sample0252.png YOLOv8-SAHI: 43ms
sample0088.png YOLOv8-SAHI: 45ms
sample0216.png YOLOv8-SAHI: 44ms
sample0313.png YOLOv8-SAHI: 45ms
sample0321.png YOLOv8-SAHI: 43ms
sample0025.png YOLOv8-SAHI: 44ms
sample0196.png YOLOv8-SAHI: 45ms
sample0286.png YOLOv8-SAHI: 45ms
sample0007.png YOLOv8-SAHI: 45ms
sample0292.png YOLOv8-SAHI: 44ms
sample0109.png YOLOv8-SAHI: 45ms
sample0225.png YOLOv8-SAHI: 44ms
sample0200.png YOLOv8-SAHI: 44ms
sample0031.png YOLOv8-SAHI: 45ms
sample0095.png YOLOv8-SAHI: 45ms
sample0212.png YOLOv8-SAHI: 44ms
sample0214.png YOLOv8-SAHI: 45ms
sample0111.png YOLOv8-SAHI: 45ms
sample0302.png YOLOv8-SAHI: 44ms
sample0222.png YOLOv8-SAHI: 44ms
sample0256.png YOLOv8-SAHI: 44ms
sample0132.png YOLOv8-SAHI: 44ms
sample0281.png YOLOv8-SAHI: 44ms
sample0127.png YOLOv8-SAHI: 44ms
sample0307.png YOLOv8-SAHI: 43ms
sample0078.png YOLOv8-SAHI: 46ms
sample0119.png YOLOv8-SAHI: 44ms
sample0152.png YOLOv8-SAHI: 45ms
sample0029.png YOLOv8-SAHI: 44ms
sample0057.png YOLOv8-SAHI: 45ms
sample0243.png YOLOv8-SAHI: 44ms
sample0327.png YOLOv8-SAHI: 44ms
sample0203.png YOLOv8-SAHI: 44ms
sample0312.png YOLOv8-SAHI: 44ms
sample0319.png YOLOv8-SAHI: 45ms
sample0027.png YOLOv8-SAHI: 45ms
sample0238.png YOLOv8-SAHI: 45ms
sample0329.png YOLOv8-SAHI: 43ms
sample0142.png YOLOv8-SAHI: 44ms
sample0263.png YOLOv8-SAHI: 44ms
sample0240.png YOLOv8-SAHI: 44ms
sample0139.png YOLOv8-SAHI: 44ms
sample0227.png YOLOv8-SAHI: 45ms
sample0325.png YOLOv8-SAHI: 45ms
sample0108.png YOLOv8-SAHI: 44ms
sample0094.png YOLOv8-SAHI: 45ms
sample0023.png YOLOv8-SAHI: 45ms
sample0259.png YOLOv8-SAHI: 44ms
sample0160.png YOLOv8-SAHI: 45ms
sample0279.png YOLOv8-SAHI: 44ms
sample0104.png YOLOv8-SAHI: 45ms
sample0037.png YOLOv8-SAHI: 45ms
sample0042.png YOLOv8-SAHI: 44ms
sample0251.png YOLOv8-SAHI: 44ms
sample0105.png YOLOv8-SAHI: 45ms
sample0210.png YOLOv8-SAHI: 44ms
sample0059.png YOLOv8-SAHI: 45ms
sample0285.png YOLOv8-SAHI: 45ms
sample0258.png YOLOv8-SAHI: 45ms
sample0268.png YOLOv8-SAHI: 44ms
sample0156.png YOLOv8-SAHI: 45ms
sample0195.png YOLOv8-SAHI: 46ms
sample0235.png YOLOv8-SAHI: 44ms
sample0220.png YOLOv8-SAHI: 45ms
sample0019.png YOLOv8-SAHI: 45ms
sample0054.png YOLOv8-SAHI: 45ms
sample0005.png YOLOv8-SAHI: 45ms
sample0118.png YOLOv8-SAHI: 45ms
sample0282.png YOLOv8-SAHI: 45ms
sample0006.png YOLOv8-SAHI: 45ms
sample0090.png YOLOv8-SAHI: 45ms
sample0082.png YOLOv8-SAHI: 45ms
sample0040.png YOLOv8-SAHI: 45ms
sample0131.png YOLOv8-SAHI: 45ms
sample0049.png YOLOv8-SAHI: 45ms
sample0097.png YOLOv8-SAHI: 45ms
sample0198.png YOLOv8-SAHI: 44ms
sample0011.png YOLOv8-SAHI: 44ms
sample0219.png YOLOv8-SAHI: 44ms
sample0045.png YOLOv8-SAHI: 46ms
sample0223.png YOLOv8-SAHI: 44ms
sample0289.png YOLOv8-SAHI: 45ms
sample0017.png YOLOv8-SAHI: 46ms
sample0194.png YOLOv8-SAHI: 45ms
sample0245.png YOLOv8-SAHI: 44ms
sample0162.png YOLOv8-SAHI: 45ms
sample0207.png YOLOv8-SAHI: 44ms
sample0233.png YOLOv8-SAHI: 45ms
sample0123.png YOLOv8-SAHI: 44ms
sample0278.png YOLOv8-SAHI: 44ms
sample0248.png YOLOv8-SAHI: 44ms
sample0261.png YOLOv8-SAHI: 44ms
sample0298.png YOLOv8-SAHI: 44ms
sample0117.png YOLOv8-SAHI: 44ms
sample0115.png YOLOv8-SAHI: 45ms
sample0073.png YOLOv8-SAHI: 45ms
sample0161.png YOLOv8-SAHI: 46ms
sample0176.png YOLOv8-SAHI: 44ms
sample0035.png YOLOv8-SAHI: 45ms
sample0228.png YOLOv8-SAHI: 44ms
sample0169.png YOLOv8-SAHI: 45ms
sample0183.png YOLOv8-SAHI: 44ms
sample0337.png YOLOv8-SAHI: 44ms
sample0301.png YOLOv8-SAHI: 44ms
sample0271.png YOLOv8-SAHI: 44ms
sample0018.png YOLOv8-SAHI: 45ms
sample0209.png YOLOv8-SAHI: 45ms
sample0221.png YOLOv8-SAHI: 44ms
sample0202.png YOLOv8-SAHI: 46ms
sample0141.png YOLOv8-SAHI: 45ms
sample0153.png YOLOv8-SAHI: 44ms
sample0205.png YOLOv8-SAHI: 45ms
sample0224.png YOLOv8-SAHI: 44ms
sample0151.png YOLOv8-SAHI: 45ms
sample0215.png YOLOv8-SAHI: 45ms
sample0164.png YOLOv8-SAHI: 45ms
sample0253.png YOLOv8-SAHI: 44ms
sample0172.png YOLOv8-SAHI: 44ms
sample0262.png YOLOv8-SAHI: 45ms
sample0182.png YOLOv8-SAHI: 44ms
sample0053.png YOLOv8-SAHI: 46ms
sample0113.png YOLOv8-SAHI: 45ms
sample0333.png YOLOv8-SAHI: 45ms
sample0314.png YOLOv8-SAHI: 44ms
sample0020.png YOLOv8-SAHI: 46ms
sample0340.png YOLOv8-SAHI: 45ms
sample0077.png YOLOv8-SAHI: 45ms
sample0274.png YOLOv8-SAHI: 44ms
sample0072.png YOLOv8-SAHI: 45ms
sample0267.png YOLOv8-SAHI: 44ms
sample0303.png YOLOv8-SAHI: 45ms
sample0149.png YOLOv8-SAHI: 45ms
sample0075.png YOLOv8-SAHI: 44ms
sample0044.png YOLOv8-SAHI: 46ms
sample0334.png YOLOv8-SAHI: 44ms
sample0170.png YOLOv8-SAHI: 45ms
sample0048.png YOLOv8-SAHI: 45ms
sample0050.png YOLOv8-SAHI: 45ms
sample0093.png YOLOv8-SAHI: 46ms
sample0184.png YOLOv8-SAHI: 45ms
sample0145.png YOLOv8-SAHI: 45ms
sample0129.png YOLOv8-SAHI: 45ms
sample0269.png YOLOv8-SAHI: 44ms
sample0107.png YOLOv8-SAHI: 44ms
sample0328.png YOLOv8-SAHI: 44ms
sample0309.png YOLOv8-SAHI: 44ms
sample0091.png YOLOv8-SAHI: 45ms
sample0030.png YOLOv8-SAHI: 44ms
sample0234.png YOLOv8-SAHI: 44ms
sample0341.png YOLOv8-SAHI: 44ms
sample0036.png YOLOv8-SAHI: 45ms
sample0244.png YOLOv8-SAHI: 44ms
sample0140.png YOLOv8-SAHI: 43ms
sample0032.png YOLOv8-SAHI: 44ms
sample0166.png YOLOv8-SAHI: 45ms
sample0338.png YOLOv8-SAHI: 44ms
sample0114.png YOLOv8-SAHI: 44ms
sample0191.png YOLOv8-SAHI: 45ms
sample0322.png YOLOv8-SAHI: 44ms
sample0186.png YOLOv8-SAHI: 45ms
sample0249.png YOLOv8-SAHI: 44ms
sample0318.png YOLOv8-SAHI: 45ms
sample0080.png YOLOv8-SAHI: 46ms
sample0014.png YOLOv8-SAHI: 45ms
sample0038.png YOLOv8-SAHI: 45ms
sample0174.png YOLOv8-SAHI: 45ms
sample0034.png YOLOv8-SAHI: 45ms
sample0067.png YOLOv8-SAHI: 44ms
sample0335.png YOLOv8-SAHI: 43ms
sample0066.png YOLOv8-SAHI: 45ms
sample0043.png YOLOv8-SAHI: 45ms
sample0098.png YOLOv8-SAHI: 44ms
sample0181.png YOLOv8-SAHI: 44ms
sample0039.png YOLOv8-SAHI: 46ms
sample0231.png YOLOv8-SAHI: 45ms
sample0028.png YOLOv8-SAHI: 46ms
sample0084.png YOLOv8-SAHI: 45ms
sample0138.png YOLOv8-SAHI: 44ms
sample0008.png YOLOv8-SAHI: 45ms
sample0146.png YOLOv8-SAHI: 45ms
sample0266.png YOLOv8-SAHI: 45ms
sample0171.png YOLOv8-SAHI: 45ms
sample0120.png YOLOv8-SAHI: 45ms
sample0125.png YOLOv8-SAHI: 45ms
sample0237.png YOLOv8-SAHI: 45ms
sample0056.png YOLOv8-SAHI: 45ms
sample0242.png YOLOv8-SAHI: 44ms
sample0290.png YOLOv8-SAHI: 44ms
sample0287.png YOLOv8-SAHI: 44ms
sample0012.png YOLOv8-SAHI: 45ms
sample0199.png YOLOv8-SAHI: 45ms
sample0185.png YOLOv8-SAHI: 45ms
sample0010.png YOLOv8-SAHI: 45ms
sample0315.png YOLOv8-SAHI: 44ms
sample0015.png YOLOv8-SAHI: 45ms
sample0241.png YOLOv8-SAHI: 45ms
sample0130.png YOLOv8-SAHI: 45ms
sample0272.png YOLOv8-SAHI: 43ms
sample0187.png YOLOv8-SAHI: 45ms
sample0232.png YOLOv8-SAHI: 45ms
sample0330.png YOLOv8-SAHI: 44ms
sample0167.png YOLOv8-SAHI: 45ms
sample0273.png YOLOv8-SAHI: 45ms
sample0099.png YOLOv8-SAHI: 45ms
sample0163.png YOLOv8-SAHI: 44ms
sample0074.png YOLOv8-SAHI: 45ms
sample0299.png YOLOv8-SAHI: 45ms
sample0239.png YOLOv8-SAHI: 44ms
sample0189.png YOLOv8-SAHI: 46ms
sample0188.png YOLOv8-SAHI: 44ms
sample0047.png YOLOv8-SAHI: 46ms
sample0332.png YOLOv8-SAHI: 45ms
sample0046.png YOLOv8-SAHI: 44ms
sample0304.png YOLOv8-SAHI: 45ms
sample0087.png YOLOv8-SAHI: 45ms
sample0236.png YOLOv8-SAHI: 45ms
sample0226.png YOLOv8-SAHI: 45ms
sample0293.png YOLOv8-SAHI: 45ms
sample0068.png YOLOv8-SAHI: 45ms
sample0213.png YOLOv8-SAHI: 44ms
sample0336.png YOLOv8-SAHI: 44ms
sample0079.png YOLOv8-SAHI: 45ms
sample0004.png YOLOv8-SAHI: 44ms
sample0276.png YOLOv8-SAHI: 45ms
sample0158.png YOLOv8-SAHI: 45ms
sample0001.png YOLOv8-SAHI: 45ms
sample0192.png YOLOv8-SAHI: 45ms
sample0100.png YOLOv8-SAHI: 45ms
sample0143.png YOLOv8-SAHI: 44ms
sample0024.png YOLOv8-SAHI: 45ms
sample0081.png YOLOv8-SAHI: 45ms
sample0136.png YOLOv8-SAHI: 44ms
sample0265.png YOLOv8-SAHI: 44ms
sample0264.png YOLOv8-SAHI: 44ms
sample0178.png YOLOv8-SAHI: 44ms
sample0013.png YOLOv8-SAHI: 45ms
sample0275.png YOLOv8-SAHI: 44ms
sample0339.png YOLOv8-SAHI: 43ms
sample0033.png YOLOv8-SAHI: 45ms
sample0065.png YOLOv8-SAHI: 46ms
sample0071.png YOLOv8-SAHI: 44ms
sample0284.png YOLOv8-SAHI: 45ms
sample0106.png YOLOv8-SAHI: 44ms
sample0126.png YOLOv8-SAHI: 45ms
sample0055.png YOLOv8-SAHI: 45ms
sample0320.png YOLOv8-SAHI: 44ms
sample0305.png YOLOv8-SAHI: 44ms
sample0323.png YOLOv8-SAHI: 44ms
sample0089.png YOLOv8-SAHI: 45ms
sample0177.png YOLOv8-SAHI: 45ms
sample0175.png YOLOv8-SAHI: 45ms
sample0190.png YOLOv8-SAHI: 44ms
sample0063.png YOLOv8-SAHI: 45ms
sample0291.png YOLOv8-SAHI: 45ms
sample0092.png YOLOv8-SAHI: 45ms
sample0154.png YOLOv8-SAHI: 45ms
sample0294.png YOLOv8-SAHI: 45ms
sample0246.png YOLOv8-SAHI: 44ms
sample0201.png YOLOv8-SAHI: 46ms
sample0211.png YOLOv8-SAHI: 45ms
sample0150.png YOLOv8-SAHI: 44ms
sample0085.png YOLOv8-SAHI: 45ms
sample0308.png YOLOv8-SAHI: 44ms
sample0300.png YOLOv8-SAHI: 44ms
sample0277.png YOLOv8-SAHI: 44ms
sample0297.png YOLOv8-SAHI: 43ms
sample0317.png YOLOv8-SAHI: 44ms
sample0155.png YOLOv8-SAHI: 45ms
sample0041.png YOLOv8-SAHI: 45ms
sample0310.png YOLOv8-SAHI: 45ms
sample0148.png YOLOv8-SAHI: 45ms
sample0270.png YOLOv8-SAHI: 44ms
sample0197.png YOLOv8-SAHI: 46ms
sample0003.png YOLOv8-SAHI: 45ms
sample0110.png YOLOv8-SAHI: 45ms
sample0137.png YOLOv8-SAHI: 45ms
sample0193.png YOLOv8-SAHI: 44ms
sample0180.png YOLOv8-SAHI: 44ms
sample0179.png YOLOv8-SAHI: 45ms
sample0168.png YOLOv8-SAHI: 44ms
sample0306.png YOLOv8-SAHI: 44ms
sample0061.png YOLOv8-SAHI: 45ms
sample0260.png YOLOv8-SAHI: 44ms
sample0217.png YOLOv8-SAHI: 44ms
sample0076.png YOLOv8-SAHI: 44ms
sample0316.png YOLOv8-SAHI: 43ms
sample0229.png YOLOv8-SAHI: 44ms
sample0218.png YOLOv8-SAHI: 45ms
sample0022.png YOLOv8-SAHI: 45ms
sample0133.png YOLOv8-SAHI: 46ms
sample0326.png YOLOv8-SAHI: 45ms
sample0296.png YOLOv8-SAHI: 44ms
sample0062.png YOLOv8-SAHI: 45ms
sample0255.png YOLOv8-SAHI: 44ms
sample0144.png YOLOv8-SAHI: 45ms
sample0331.png YOLOv8-SAHI: 44ms
sample0208.png YOLOv8-SAHI: 46ms
sample0096.png YOLOv8-SAHI: 46ms
sample0295.png YOLOv8-SAHI: 44ms
sample0103.png YOLOv8-SAHI: 44ms
sample0112.png YOLOv8-SAHI: 44ms
sample0250.png YOLOv8-SAHI: 44ms
sample0009.png YOLOv8-SAHI: 45ms
sample0157.png YOLOv8-SAHI: 45ms
sample0254.png YOLOv8-SAHI: 43ms
sample0016.png YOLOv8-SAHI: 46ms
sample0052.png YOLOv8-SAHI: 44ms
sample0311.png YOLOv8-SAHI: 45ms
sample0060.png YOLOv8-SAHI: 44ms
sample0026.png YOLOv8-SAHI: 45ms
sample0002.png YOLOv8-SAHI: 45ms
sample0069.png YOLOv8-SAHI: 44ms
sample0128.png YOLOv8-SAHI: 44ms
sample0204.png YOLOv8-SAHI: 44ms
sample0147.png YOLOv8-SAHI: 45ms
sample0116.png YOLOv8-SAHI: 44ms
sample0051.png YOLOv8-SAHI: 44ms
sample0173.png YOLOv8-SAHI: 45ms
sample0283.png YOLOv8-SAHI: 44ms
sample0134.png YOLOv8-SAHI: 44ms
sample0064.png YOLOv8-SAHI: 45ms
sample0165.png YOLOv8-SAHI: 44ms
sample0159.png YOLOv8-SAHI: 45ms
sample0280.png YOLOv8-SAHI: 45ms
sample0257.png YOLOv8-SAHI: 44ms
sample0247.png YOLOv8-SAHI: 44ms
```

### References

[TensorRT-YOLOv8-ByteTrack](https://github.com/emptysoal/TensorRT-YOLOv8-ByteTrack/tree/main)