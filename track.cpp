#include "model.h"
#include "utils.h"
#include <fstream>
#include <iostream>
#include "logging.h"
#include "cuda_utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include "BYTETracker.h"
#include <opencv2/opencv.hpp>

Logger gLogger;
using namespace nvinfer1;
const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

void deserialize_engine(std::string &engine_name, IRuntime **runtime, ICudaEngine **engine, IExecutionContext **context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char *serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

void prepare_buffer(ICudaEngine *engine, float **input_buffer_device, float **output_buffer_device,
                    float **output_buffer_host, float **decode_ptr_host, float **decode_ptr_device) {
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void **) input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));

    *output_buffer_host = new float[kBatchSize * kOutputSize];
}

void infer(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *output, int batchsize, float* decode_ptr_host, float* decode_ptr_device, int model_bboxes) {
    // infer on the batch asynchronously, and DMA output back to host
    context.enqueue(batchsize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

int run(char *videoPath)
{
    cudaSetDevice(kGpuId);
    std::string engine_path = trtFile;
    int model_bboxes;

    // Deserialize the engine from file
    IRuntime *runtime = nullptr;
    ICudaEngine *engine = nullptr;
    IExecutionContext *context = nullptr;
    deserialize_engine(engine_path, &runtime, &engine, &context);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cuda_preprocess_init(kMaxInputImageSize);
    auto out_dims = engine->getBindingDimensions(1);
    model_bboxes = out_dims.d[0];
    // Prepare cpu and gpu buffers
    float *device_buffers[2];
    float *output_buffer_host = nullptr;
    float *decode_ptr_host=nullptr;
    float *decode_ptr_device=nullptr;

    // Prepare input and output buffers
    prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &output_buffer_host, &decode_ptr_host, &decode_ptr_device);

    // 初始化cv::Rect对象：(x, y, width, height)
    std::vector<std::vector<cv::Rect>> slice_boxes
    {
        {cv::Rect(0, 0, 640, 640)},
        {cv::Rect(400, 0, 640, 640)},
        {cv::Rect(800, 0, 640, 640)},
        {cv::Rect(0, 440, 640, 640)},
        {cv::Rect(400, 440, 640, 640)},
        {cv::Rect(800, 440, 640, 640)},
        {cv::Rect(400, 220, 640, 640)},
        {cv::Rect(0, 0, 1440, 1080)},
    };

    std::vector <std::vector<float>> slice_box_lt
    {
        {0, 0},
        {400, 0},
        {800, 0},
        {0, 440},
        {400, 440},
        {800, 440},
        {400, 220},
        {0, 0},
    };

    // 提前创建大图内存空间
    cv::Mat large_image_buffer(1080, 1440, CV_8UC3, cv::Scalar(0, 0, 0));

    // 提前创建切片视图（避免重复创建）
    std::vector<cv::Mat> slice_views;
    for (size_t i = 0; i < slice_boxes.size(); ++i) {
        for (size_t j = 0; j < slice_boxes[i].size(); ++j) {
            cv::Rect rect = slice_boxes[i][j];
            slice_views.push_back(large_image_buffer(rect));
        }
    }

    // read video
    std::string input_video_path = std::string(videoPath);
    cv::VideoCapture cap(input_video_path);
    if ( !cap.isOpened() ) return 0;

    int width = cap.get(CAP_PROP_FRAME_WIDTH);
	int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));
    cout << "Total frames: " << nFrame << endl;

    cv::VideoWriter writer("result.mp4", VideoWriter::fourcc('a', 'v', 'c', '1'), fps, Size(width, height));

    // ByteTrack tracker
    BYTETracker tracker(fps, 30);

    cv::Mat frame;
    int total_ms = 0;
    int num_frames = 0;
    while (true)
    {
        if(!cap.read(frame)) break;
        num_frames ++;
        if (num_frames % 20 == 0)
        {
            cout << "Processing frame " << num_frames << " (" << num_frames * 1000000 / total_ms << " fps)" << endl;
        }
        if (frame.empty()) break;

        auto start = std::chrono::system_clock::now();

        // Image Slice
        cv::resize(frame, large_image_buffer, cv::Size(1440, 1080));
        std::vector<cv::Mat> img_batch;
        for (const auto& view : slice_views) {
            img_batch.push_back(view.clone());
        }
        // Batch Preprocess
        cuda_batch_preprocess(img_batch, device_buffers[0], kInputW, kInputH, stream);
        // Batch Inference
        infer(*context, stream, (void **)device_buffers, output_buffer_host, kBatchSize, decode_ptr_host, decode_ptr_device, model_bboxes);
        // Batch Postprocess
        std::vector<Detection> res;
        batch_nms_process(res, output_buffer_host, img_batch, kOutputSize, kConfThresh, kNmsThresh, slice_box_lt);

        // yolo output format to bytetrack input format, and filter bbox by class id
        std::vector<Object> objects;
        for (size_t j = 0; j < res.size(); j++){
            float* bbox = res[j].bbox;
            float conf = 0.8;  // res[j].conf;
            int classId = res[j].class_id;

            cv::Rect_<float> rect(bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1]));
            Object obj {rect, classId, conf};
            objects.push_back(obj);
        }

        // track
        std::vector<STrack> output_stracks = tracker.update(objects);

        // draw
        for (int i = 0; i < output_stracks.size(); i++)
        {
            int class_id = output_stracks[i].label;
            cv::Scalar s = tracker.get_color(class_id);
            std::vector<float> tlwh = output_stracks[i].tlwh;
            cv::putText(large_image_buffer, cv::format("%s %d", vClassNames[class_id].c_str(), output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 5), 
                        0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::rectangle(large_image_buffer, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
        }

        auto end = std::chrono::system_clock::now();
        total_ms = total_ms + std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        cv::resize(large_image_buffer, frame, frame.size());
        writer.write(frame);

        int c = cv::waitKey(1);
	    if (c == 27) break;  // ESC to exit
    }

    std::cout << "FPS: " << num_frames * 1000000 / total_ms << std::endl;
    writer.release();
    cap.release();

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(device_buffers[0]));
    CUDA_CHECK(cudaFree(device_buffers[1]));
    CUDA_CHECK(cudaFree(decode_ptr_device));
    delete[] decode_ptr_host;
    delete[] output_buffer_host;
    cuda_preprocess_destroy();
    // Destroy the engine
    delete context;
    delete engine;
    delete runtime;

    return 0;
}

int main(int argc, char *argv[])
{
    if (argc != 2 )
    {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "Usage: ./yolov8_sahi_track [video path]" << std::endl;
        std::cerr << "Example: ./yolov8_sahi_track ../midia/c3_1080.mp4" << std::endl;
        return -1;
    }

    return run(argv[1]);
}