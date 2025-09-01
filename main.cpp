#include "model.h"
#include "utils.h"
#include <fstream>
#include <iostream>
#include "logging.h"
#include "cuda_utils.h"
#include "preprocess.h"
#include "postprocess.h"
#include <opencv2/opencv.hpp>

Logger gLogger;
using namespace nvinfer1;
const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

void serialize_engine(std::string &wts_name, std::string &engine_name, std::string &sub_type) {
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();
    IHostMemory *serialized_engine = nullptr;

    if (sub_type == "n") {
        serialized_engine = buildEngineYolov8n(builder, config, DataType::kFLOAT, wts_name);
    } else if (sub_type == "s") {
        serialized_engine = buildEngineYolov8s(builder, config, DataType::kFLOAT, wts_name);
    } else if (sub_type == "m") {
        serialized_engine = buildEngineYolov8m(builder, config, DataType::kFLOAT, wts_name);
    } else if (sub_type == "l") {
        serialized_engine = buildEngineYolov8l(builder, config, DataType::kFLOAT, wts_name);
    } else if (sub_type == "x") {
        serialized_engine = buildEngineYolov8x(builder, config, DataType::kFLOAT, wts_name);
    }

    assert(serialized_engine);
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cout << "could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char *>(serialized_engine->data()), serialized_engine->size());

    delete builder;
    delete config;
    delete serialized_engine;
}

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

bool parse_args(int argc, char **argv, std::string &wts, std::string &engine, std::string &img_dir, std::string &sub_type) {
    if (std::string(argv[1]) == "-s" && argc == 5) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        sub_type = std::string(argv[4]);
    } else if (std::string(argv[1]) == "-d" && argc == 4) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
    } else {
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    cudaSetDevice(kGpuId);
    std::string wts_name = "";
    std::string engine_name = "";
    std::string img_dir;
    std::string sub_type = "";
    int model_bboxes;

    if (!parse_args(argc, argv, wts_name, engine_name, img_dir, sub_type)) {
        std::cerr << "Arguments not right!" << std::endl;
        std::cerr << "./yolov8 -s [.wts] [.engine] [n/s/m/l/x]  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov8 -d [.engine] ../images // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // Create a model using the API directly and serialize it to a file
    if (!wts_name.empty()) {
        serialize_engine(wts_name, engine_name, sub_type);
        return 0;
    }

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

    // Deserialize the engine from file
    IRuntime *runtime = nullptr;
    ICudaEngine *engine = nullptr;
    IExecutionContext *context = nullptr;
    deserialize_engine(engine_name, &runtime, &engine, &context);
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

    // Read images from directory
    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    prepare_buffer(engine, &device_buffers[0], &device_buffers[1], &output_buffer_host, &decode_ptr_host, &decode_ptr_device);

    for (size_t i = 0; i < file_names.size(); ++i) {
        cv::Mat image_resize;
        std::vector<cv::Mat> img_batch;

        cv::Mat image = cv::imread(img_dir + "/" + file_names[i]);
        cv::resize(image, image_resize, cv::Size(1440, 1080));

        // Image Slice
        for (size_t i = 0; i < slice_boxes.size(); ++i) {
            for (size_t j = 0; j < slice_boxes[i].size(); ++j) {
                cv::Rect rect = slice_boxes[i][j];
                cv::Mat img = image_resize(rect).clone();
                img_batch.push_back(img);
            }
        }

        // Start inference
        auto start = std::chrono::system_clock::now();

        // Batch Preprocess
        cuda_batch_preprocess(img_batch, device_buffers[0], kInputW, kInputH, stream);
        // Batch Inference
        infer(*context, stream, (void **)device_buffers, output_buffer_host, kBatchSize, decode_ptr_host, decode_ptr_device, model_bboxes);
        // Batch Postprocess
        std::vector<Detection> res;
        batch_nms_process(res, output_buffer_host, img_batch, kOutputSize, kConfThresh, kNmsThresh, slice_box_lt);
        
        // Inference Finished
        auto end = std::chrono::system_clock::now();
        std::cout << file_names[i] << " YOLOv8-SAHI: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        // Draw Boxes
        draw_bbox_batch(image_resize, res);
        cv::imwrite("_" + file_names[i], image_resize);

    }

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

