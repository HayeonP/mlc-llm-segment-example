

#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>
#include <sstream>
#include <variant>
#include <thread>
#include <any>
#include <algorithm>
#include <stop_token>



// #include <picojson.h>
// #include <serve/config.h>
// #include <serve/threaded_engine.h>
// #include <serve/data.h>
// #include <serve/config.h>
// #include <serve/request.h>
// #include <tokenizers/tokenizers.h>
// #include <tokenizers/streamer.h>
// #include <frontend/engine_base.h>
// #include <json_ffi/conv_template.h>
#include <json_ffi/openai_api_protocol.h>
// #include <frontend/mlc_chat_config.h>

// #include <tvm/runtime/device_api.h>
// #include <tvm/ffi/function.h>
// #include <tvm/ffi/container/array.h>
// #include <tvm/ffi/error.h>
// #include <tvm/ffi/string.h>
// #include <tvm/ffi/object.h>
// #include <tvm/ffi/optional.h>
// #include <tvm/ffi/any.h>
// #include <tvm/runtime/int_tuple.h>

#include <stdexcept>
#include <csignal>
#include <atomic>

#include <serve/segment_runner/utils.h>
#include <serve/segment_runner/blocking_queue.h>
#include <serve/segment_runner/scope_fail.h>
#include <serve/segment_runner/generator.h>
#include <serve/segment_runner/cpp_interface.h>

using namespace tvm;
using namespace ffi;

using ChatCompletionRequest = mlc::llm::json_ffi::ChatCompletionRequest;
using ChatCompletionResponse = mlc::llm::json_ffi::ChatCompletionResponse;

int main(int argc, char* argv[]){
  int max_tokens_value = -1;
  if(argc > 1)
    max_tokens_value = atoi(argv[1]);

  std::string model_dir = "/home/rubis/workspace/llama/mlc-llm-models/llama-3.2-1b/workspace";
  std::string model_lib_path = model_dir + "/llama-3.2-1b-cuda.so";
  tvm::Device dev{kDLCUDA, 0};
  std::string mode = "local";

  CppInterface cpp_interface;
  cpp_interface.init(model_dir, dev, model_lib_path, mode);
  cpp_interface.SetSeed(4542);

  std::optional<std::string> request_id = std::nullopt; // no request_id

  // std::string prompt("Answer the following question in one sentence. What is the capital of South Korea?");

  std::string prompt("Why USA is the one of the strongest country?");
  
  std::vector<std::chrono::duration<float>> total_time_list;

  int n = 1;
  int warmup = 2;
  int max_tokens = 256;
  if(max_tokens_value > 0) max_tokens = max_tokens_value;
  
  bool stream = false;
  ChatCompletionRequest request = cpp_interface.create_chat_completion_request(model_dir, prompt, max_tokens, stream);


  for(int i = 0; i < n + warmup; i++){
    printf("instance %d\n", i);   
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    ChatCompletionResponse response = cpp_interface.create(request_id, request);
    auto total_end = std::chrono::high_resolution_clock::now();
    
    total_time_list.push_back(total_end - total_start);

    std::cout<<"==============================="<<std::endl;    
    std::cout<<""<<cpp_interface.response_to_str(response)<<std::endl;
  }

  total_time_list.erase(total_time_list.begin(), total_time_list.begin()+warmup);
  float total_time_avg = mlc::llm::utils::calculate_duration_avg(total_time_list) * 1000;
  float total_time_max = mlc::llm::utils::calculate_duration_max(total_time_list) * 1000;

  std::cout << "===========================" << std::endl;
  std::cout << "# total time" << std::endl;
  std::cout << "Average response time: " << total_time_avg << "ms" << std::endl;
  std::cout << "Worst response time: " << total_time_max << "ms" << std::endl;

  return 0;
}