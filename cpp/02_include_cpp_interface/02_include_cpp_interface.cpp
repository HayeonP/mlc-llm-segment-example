

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
  std::string model_dir = "/home/rubis/workspace/llama/mlc-llm-models/llama-3.2-1b/workspace";
  std::string model_lib_path = model_dir + "/llama-3.2-1b-cuda.so";
  tvm::Device dev{kDLCUDA, 0};
  std::string mode = "local";

  CppInterface cpp_interface;
  cpp_interface.init(model_dir, dev, model_lib_path, mode);

  std::optional<std::string> request_id = std::nullopt; // no request_id

  // std::string prompt("Answer the following question in one sentence. What is the capital of South Korea?");

  std::string prompt("Samsung V.S. SK Hynix?");
  int max_tokens = 128;
  bool stream = false;
  ChatCompletionRequest request = cpp_interface.create_chat_completion_request(model_dir, prompt, max_tokens, stream);

  ChatCompletionResponse response = cpp_interface.create(request_id, request);

  std::cout<<"MLC-LLM Output: "<<cpp_interface.response_to_str(response)<<std::endl;

  return 0;
}