

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

// #include <json_ffi/conv_template.h>
#include <json_ffi/openai_api_protocol.h>

#include <serve/segment_runner/segment_runner.h>

using namespace tvm;
using namespace ffi;

using ChatCompletionRequest = mlc::llm::json_ffi::ChatCompletionRequest;
using ChatCompletionResponse = mlc::llm::json_ffi::ChatCompletionResponse;

float calculate_duration_avg(std::vector<std::chrono::duration<float>> list){
    if (list.empty()) return 0.0f;

    float sum = 0.0f;
    for (const auto& dur : list) {
        sum += dur.count();
    }
    return sum / list.size();
}

float calculate_duration_max(std::vector<std::chrono::duration<float>> list){
    if (list.empty()) return 0.0f;

    auto max_itr = std::max_element(list.begin(), list.end(),
        [](const auto& a, const auto& b) { return a.count() < b.count(); });
    return max_itr->count();
}

std::string readFileToString(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("파일을 열 수 없습니다: " + filePath);
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();  // 전체 파일 내용을 스트림으로 읽기
    return buffer.str();     // 문자열로 반환
}

int main(int argc, char* argv[]){
  std::string model_dir = "/home/rubis/workspace/llama/mlc-llm-models/llama-3.2-1b/workspace";
  std::string model_lib_path = model_dir + "/llama-3.2-1b-cuda.so";
  tvm::Device dev{kDLCUDA, 0};
  std::string mode = "local";

  int max_tokens_value = -1;
  int prefill_chunk_size = 8;

  if(argc > 1)
    max_tokens_value = atoi(argv[1]);

  if(argc > 2)
    prefill_chunk_size = atoi(argv[2]);


  SegmentRunner segment_runner;
  segment_runner.Init(model_dir, dev, model_lib_path, mode, prefill_chunk_size);
  segment_runner.SetSeed(4542); // For same experiment

  // std::string prompt("Why USA is the one of the strongest country?");
  std::string prompt = readFileToString("input.txt");
  
  std::vector<std::chrono::duration<float>> total_time_list;
  std::vector<std::chrono::duration<float>> request_time_list;
  std::vector<std::chrono::duration<float>> prefill_time_list;
  std::vector<std::chrono::duration<float>> inference_time_list;

  int n = 3;
  int warmup = 0;
  int max_tokens = 256;

  if(max_tokens_value > 0) max_tokens = max_tokens_value;

  for(int i = 0; i < n + warmup; i++){
    printf("instance %d\n", i);   
    auto total_start = std::chrono::high_resolution_clock::now();

    auto request_start = std::chrono::high_resolution_clock::now();
    // - Request
    segment_runner.Request(prompt, max_tokens);
    std::cout<<"Finish request"<<std::endl;
    auto request_end = std::chrono::high_resolution_clock::now();
    request_time_list.push_back(request_end - request_start);

    auto prefill_start = std::chrono::high_resolution_clock::now();
    // - Prefill
    while(!segment_runner.IsPrefillEnd()){
      auto s = std::chrono::high_resolution_clock::now();
      segment_runner.Prefill(1);
      auto e = std::chrono::high_resolution_clock::now();
      std::cout<<"---- prefill: "<<std::chrono::duration_cast<std::chrono::milliseconds>(e-s).count() <<"ms"<<std::endl;
    }
    auto prefill_end = std::chrono::high_resolution_clock::now();
    prefill_time_list.push_back(prefill_end - prefill_start);

    auto inference_start = std::chrono::high_resolution_clock::now();
    // - Inference
    bool is_end = false;
    std::string output;
    
    while(!segment_runner.IsEnd()){
      std::string delta = segment_runner.Execute(5);
      output += delta;
    }    

    auto inference_end = std::chrono::high_resolution_clock::now();
    inference_time_list.push_back(inference_end - inference_start);

    auto total_end = std::chrono::high_resolution_clock::now();
    total_time_list.push_back(total_end - total_start);


    std::cout<<"==============================="<<std::endl;
    std::cout<<output<<std::endl;
  }

  request_time_list.erase(request_time_list.begin(), request_time_list.begin()+warmup);    
  inference_time_list.erase(inference_time_list.begin(), inference_time_list.begin()+warmup);
  total_time_list.erase(total_time_list.begin(), total_time_list.begin()+warmup);
                
  float request_time_avg = calculate_duration_avg(request_time_list) * 1000;
  float request_time_max = calculate_duration_max(request_time_list) * 1000;
  float prefill_time_avg = calculate_duration_avg(prefill_time_list) * 1000;
  float prefill_time_max = calculate_duration_max(prefill_time_list) * 1000;
  float inference_time_avg = calculate_duration_avg(inference_time_list) * 1000;
  float inference_time_max = calculate_duration_max(inference_time_list) * 1000;
  float total_time_avg = calculate_duration_avg(total_time_list) * 1000;
  float total_time_max = calculate_duration_max(total_time_list) * 1000;

  std::cout << "===========================" << std::endl;
  std::cout << "# request time" << std::endl;
  std::cout << "Average response time: " << request_time_avg << "ms" << std::endl;
  std::cout << "Worst response time: " << request_time_max << "ms" << std::endl;
  std::cout << "===========================" << std::endl;
  std::cout << "# prefill time" << std::endl;
  std::cout << "Average response time: " << prefill_time_avg << "ms" << std::endl;
  std::cout << "Worst response time: " << prefill_time_max << "ms" << std::endl;
  std::cout << "===========================" << std::endl;
  std::cout << "# inference time" << std::endl;
  std::cout << "Average response time: " << inference_time_avg << "ms" << std::endl;
  std::cout << "Worst response time: " << inference_time_max << "ms" << std::endl;
  std::cout << "===========================" << std::endl;
  std::cout << "# total time" << std::endl;
  std::cout << "Average response time: " << total_time_avg << "ms" << std::endl;
  std::cout << "Worst response time: " << total_time_max << "ms" << std::endl;

  return 0;
}