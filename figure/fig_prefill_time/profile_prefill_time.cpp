

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
  std::string model_dir = "/home/rubis/workspace/llama/mlc-llm-models/llama-3.2-1b-instruct";
  std::string model_lib_path = model_dir + "/llama-3.2-1b-cuda.so";
  tvm::Device dev{kDLCUDA, 0};
  std::string mode = "local";
  std::string input_data = "input.txt";

  int max_tokens_value = -1;
  int prefill_chunk_size = 8;

  if(argc > 1)
    max_tokens_value = atoi(argv[1]);

  if(argc > 2)
    prefill_chunk_size = atoi(argv[2]);

  if(argc > 3)
    input_data = std::string(argv[3]);


  SegmentRunner segment_runner;  
  segment_runner.Init(model_dir, dev, model_lib_path, mode, prefill_chunk_size);
  segment_runner.SetSeed(4542); // For same experiment

  // std::string prompt("Why USA is the one of the strongest country?");
  std::string prompt = readFileToString(input_data);
  
  std::vector<std::chrono::duration<float>> total_time_list;
  std::vector<std::chrono::duration<float>> request_time_list;
  std::vector<std::chrono::duration<float>> prefill_time_list;
  std::vector<std::chrono::duration<float>> inference_time_list;

  int n = 1;
  int warmup = 1;
  int max_tokens = 256;

  if(max_tokens_value > 0) max_tokens = max_tokens_value;

  for(int i = 0; i < n + warmup; i++){
    // - Request    
    segment_runner.Request(prompt, max_tokens);    

    // - Prefill    
    while(!segment_runner.IsPrefillEnd()){
      auto s = std::chrono::high_resolution_clock::now();
      segment_runner.Prefill(1);
      auto e = std::chrono::high_resolution_clock::now();
      if(i>=warmup) std::cout<<"prefill: "<<std::chrono::duration_cast<std::chrono::milliseconds>(e-s).count() <<"ms"<<std::endl;
    }    
    
    // - Inference
    bool is_end = false;
    std::string output;
    
    while(!segment_runner.IsEnd()){
      std::string delta = segment_runner.Execute(5);
      output += delta;
    }    
  }

  return 0;
}