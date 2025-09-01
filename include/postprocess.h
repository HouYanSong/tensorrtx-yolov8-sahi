#pragma once

#include "types.h"
#include "NvInfer.h"
#include <opencv2/opencv.hpp>

void batch_nms_process(std::vector<Detection> &res, float *output, std::vector<cv::Mat> &img_batch, int output_size, float conf_thresh, float nms_thresh, std::vector <std::vector<float>> slice_box_lt);

void draw_bbox_batch(cv::Mat &image_resize, std::vector<Detection>& res); 



