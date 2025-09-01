#include "postprocess.h"

void scale_bbox(cv::Mat& img, float (&bbox)[4])
{
    float r_w = kInputW / (img.cols * 1.0);
    float r_h = kInputH / (img.rows * 1.0);
    
    if (r_h > r_w) {
        // 高度方向有填充
        bbox[0] = bbox[0] / r_w;                                     // left
        bbox[2] = bbox[2] / r_w;                                     // right
        bbox[1] = (bbox[1] - (kInputH - r_w * img.rows) / 2) / r_w;  // top
        bbox[3] = (bbox[3] - (kInputH - r_w * img.rows) / 2) / r_w;  // bottom
    } else {
        // 宽度方向有填充
        bbox[0] = (bbox[0] - (kInputW - r_h * img.cols) / 2) / r_h;  // left
        bbox[2] = (bbox[2] - (kInputW - r_h * img.cols) / 2) / r_h;  // right
        bbox[1] = bbox[1] / r_h;                                     // top
        bbox[3] = bbox[3] / r_h;                                     // bottom
    }
}

static float iou(float lbox[4], float rbox[4]) 
{
    float interBox[] = {
            (std::max)(lbox[0], rbox[0]), //left
            (std::min)(lbox[2], rbox[2]), //right
            (std::max)(lbox[1], rbox[1]), //top
            (std::min)(lbox[3], rbox[3]), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    float unionBoxS = (lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) + (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) - interBoxS;
    return interBoxS / unionBoxS;
}

static bool cmp(const Detection &a, const Detection &b) 
{
    return a.conf > b.conf;
}

void batch_nms_process(std::vector<Detection> &res, float *output, std::vector<cv::Mat> &img_batch, int output_size, float conf_thresh, float nms_thresh, std::vector <std::vector<float>> slice_box_lt)
{
    int batch_size = img_batch.size();
    int det_size = sizeof(Detection) / sizeof(float);
    std::map<float, std::vector<Detection>> m;

    for (int i = 0; i < batch_size; i++)
    {
        float *output_ptr = &output[i * output_size];
        for (int j = 0; j < output_ptr[0]; j++)
        {
            if (output_ptr[1 + det_size * j + 4] <= conf_thresh) continue;
            Detection det;
            memcpy(&det, &output_ptr[1 + det_size * j], det_size * sizeof(float));
            if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
            scale_bbox(img_batch[i], det.bbox);
            det.bbox[0] += slice_box_lt[i][0];
            det.bbox[1] += slice_box_lt[i][1];
            det.bbox[2] += slice_box_lt[i][0];
            det.bbox[3] += slice_box_lt[i][1];
            m[det.class_id].push_back(det);
        }
    }
    
    for (auto it = m.begin(); it != m.end(); it++) {
        auto &dets = it->second;
        std::vector<bool> keep(dets.size(), true);
        
        for (size_t i = 0; i < dets.size(); ++i) {
            if (!keep[i]) continue;
            
            // 预先计算当前检测框的面积
            float box1_area = (dets[i].bbox[2] - dets[i].bbox[0]) * (dets[i].bbox[3] - dets[i].bbox[1]);
            
            for (size_t j = i + 1; j < dets.size(); ++j) {
                if (!keep[j]) continue;
                
                // 预先计算比较检测框的面积
                float box2_area = (dets[j].bbox[2] - dets[j].bbox[0]) * (dets[j].bbox[3] - dets[j].bbox[1]);
                
                // 计算IoU
                float iou_value = iou(dets[i].bbox, dets[j].bbox);
                
                bool should_suppress = false;
                
                // 标准NMS条件: IoU超过阈值
                if (iou_value > nms_thresh) {
                    should_suppress = true;
                } else {
                    // 计算交集区域
                    float interBox[] = {
                        (std::max)(dets[i].bbox[0], dets[j].bbox[0]),  // left
                        (std::min)(dets[i].bbox[2], dets[j].bbox[2]),  // right
                        (std::max)(dets[i].bbox[1], dets[j].bbox[1]),  // top
                        (std::min)(dets[i].bbox[3], dets[j].bbox[3]),  // bottom
                    };

                    // 检查是否有有效交集
                    if (interBox[2] < interBox[3] && interBox[0] < interBox[1]) {
                        float interArea = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
                        
                        // 检查是否一个框包含另一个框
                        if (interArea / box1_area > nms_thresh || interArea / box2_area > nms_thresh) {
                            should_suppress = true;
                        }
                    }
                }
                
                // 如果需要抑制，则保留面积大的框，抑制面积小的框
                if (should_suppress) {
                    if (box1_area > box2_area) {
                        keep[j] = false;  // 抑制较小的框j
                    } else {
                        keep[i] = false;  // 抑制较小的框i
                        break;            // 重新检查下一个未被抑制的框
                    }
                }
            }
        }
        
        // 将保留的检测框添加到结果中
        for (size_t i = 0; i < dets.size(); ++i) {
            if (keep[i]) {
                res.push_back(dets[i]);
            }
        }
    }
}

void draw_bbox_batch(cv::Mat &image_resize, std::vector<Detection>& res) 
{
    for (size_t i = 0; i < res.size(); i++)
    {
        float* bbox = res[i].bbox;
        float conf = res[i].conf;
        int class_id = res[i].class_id;
        cv::Rect_<float> rect(bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1]));
        cv::rectangle(image_resize, rect, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(image_resize, vClassNames[class_id].c_str(), cv::Point(rect.x, rect.y - 1), cv::FONT_HERSHEY_PLAIN,
                    1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }
}
