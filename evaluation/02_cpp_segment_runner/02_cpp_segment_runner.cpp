

#include <iostream>
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

int main(int argc, char* argv[]){
  std::string model_dir = "/home/rubis/workspace/llama/mlc-llm-models/llama-3.2-1b/workspace";
  std::string model_lib_path = model_dir + "/llama-3.2-1b-cuda.so";
  tvm::Device dev{kDLCUDA, 0};
  std::string mode = "local";

  SegmentRunner segment_runner;
  segment_runner.Init(model_dir, dev, model_lib_path, mode);  
  segment_runner.SetSeed(4542); // For same experiment

  std::string prompt("Can you introduce yourself?");
  
  std::vector<std::chrono::duration<float>> total_time_list;
  std::vector<std::chrono::duration<float>> request_time_list;
  std::vector<std::chrono::duration<float>> inference_time_list;

  int max_tokens = 16;
  for(int i = 0; i < 120; i++){    
    auto total_start = std::chrono::high_resolution_clock::now();

    auto request_start = std::chrono::high_resolution_clock::now();
    // - Request
    segment_runner.Request(prompt, max_tokens);
    auto request_end = std::chrono::high_resolution_clock::now();
    request_time_list.push_back(request_end - request_start);


    auto inference_start = std::chrono::high_resolution_clock::now();
    // - Inference
    bool is_end = false;
    std::string output = prompt;
    
    while(!segment_runner.IsEnd()){
      std::string delta = segment_runner.Execute();

      output += delta;
    }    

    auto inference_end = std::chrono::high_resolution_clock::now();
    inference_time_list.push_back(inference_end - inference_start);

    auto total_end = std::chrono::high_resolution_clock::now();
    total_time_list.push_back(total_end - total_start);


    std::cout<<"==============================="<<std::endl;
    std::cout<<output<<std::endl;
  }

  request_time_list.erase(request_time_list.begin(), request_time_list.begin()+20);    
  inference_time_list.erase(inference_time_list.begin(), inference_time_list.begin()+20);
  total_time_list.erase(total_time_list.begin(), total_time_list.begin()+20);
                
  float request_time_avg = calculate_duration_avg(request_time_list) * 1000;
  float request_time_max = calculate_duration_max(request_time_list) * 1000;
  float inference_time_avg = calculate_duration_avg(inference_time_list) * 1000;
  float inference_time_max = calculate_duration_max(inference_time_list) * 1000;
  float total_time_avg = calculate_duration_avg(total_time_list) * 1000;
  float total_time_max = calculate_duration_max(total_time_list) * 1000;

  std::cout << "===========================" << std::endl;
  std::cout << "# request time" << std::endl;
  std::cout << "Average response time: " << request_time_avg << "ms" << std::endl;
  std::cout << "Worst response time: " << request_time_max << "ms" << std::endl;
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