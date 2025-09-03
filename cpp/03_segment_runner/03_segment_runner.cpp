

#include <iostream>
#include <vector>

// #include <json_ffi/conv_template.h>
#include <json_ffi/openai_api_protocol.h>

#include <serve/segment_runner/segment_runner.h>

using namespace tvm;
using namespace ffi;

using ChatCompletionRequest = mlc::llm::json_ffi::ChatCompletionRequest;
using ChatCompletionResponse = mlc::llm::json_ffi::ChatCompletionResponse;

int main(int argc, char* argv[]){
  std::string model_dir = "/home/rubis/workspace/llama/mlc-llm-models/llama-3.2-1b/workspace";
  std::string model_lib_path = model_dir + "/llama-3.2-1b-cuda.so";
  tvm::Device dev{kDLCUDA, 0};
  std::string mode = "local";

  SegmentRunner segment_runner;
  segment_runner.Init(model_dir, dev, model_lib_path, mode);
  std::cout<<"[debug] After Init"<<std::endl;

  std::optional<std::string> request_id = std::nullopt; // no request_id

  // std::string prompt("Answer the following question in one sentence. What is the capital of South Korea?");

  std::string prompt("Can you introduce yourself?");
  
  // ChatCompletionRequest request = segment_runner.create_chat_completion_request(model_dir, prompt, max_tokens, stream);

  std::cout<< "# FIRST REQUEST" << std::endl;
  int max_tokens = 128;
  segment_runner.Request(prompt, max_tokens);
  std::cout<<"[debug] After request"<<std::endl;

  bool is_end = false;
  std::string output = prompt;

  while(!segment_runner.IsEnd()){
    std::cout<<"PRE_SEGMENT"<<std::endl;
    std::string delta = segment_runner.Execute();
    std::cout<<"POST_SEGMENT"<<std::endl;
    output += delta;
  }

  std::cout<<"MLC-LLM Output: "<<output<<std::endl;

  std::cout<< "# SECOND REQUEST" << std::endl;
  prompt = "How to graduate Ph.D course? It's too hard";
  segment_runner.Request(prompt, max_tokens);

  is_end = false;
  output = prompt;
  while(!segment_runner.IsEnd()){
    std::cout<<"PRE SEGMENT"<<std::endl;
    std::string delta = segment_runner.Execute();
    std::cout<<"POST SEGMENT"<<std::endl;
    output += delta;
  }

  std::cout<<"MLC-LLM Output: "<<output<<std::endl;

  return 0;
}