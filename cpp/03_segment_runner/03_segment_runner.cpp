

#include <iostream>
#include <vector>

// #include <json_ffi/conv_template.h>
#include <json_ffi/openai_api_protocol.h>

#include <serve/segment_runner/segment_runner.h>

using namespace tvm;
using namespace ffi;

using ChatCompletionRequest = mlc::llm::json_ffi::ChatCompletionRequest;
using ChatCompletionResponse = mlc::llm::json_ffi::ChatCompletionResponse;

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
  std::string model_dir = "/home/rubis/workspace/llama/mlc-llm-models/llama-3.2-1b";
  std::string model_lib_path = model_dir + "/llama-3.2-1b-cuda.so";
  tvm::Device dev{kDLCUDA, 0};
  std::string mode = "local";

  SegmentRunner segment_runner;
  int prefill_chunk_size = 256;
  segment_runner.Init(model_dir, dev, model_lib_path, mode, prefill_chunk_size);
  segment_runner.SetSeed(4542); // For same experiment
  std::cout<<"[debug] After Init"<<std::endl;

  std::optional<std::string> request_id = std::nullopt; // no request_id

  // std::string prompt("Answer the following question in one sentence. What is the capital of South Korea?");

  // std::string prompt("Can you introduce yourself?");
  std::string prompt = readFileToString("input.txt");
  
  // ChatCompletionRequest request = segment_runner.create_chat_completion_request(model_dir, prompt, max_tokens, stream);

  std::cout<< "# FIRST REQUEST" << std::endl;
  int max_tokens = 4096;
  segment_runner.Request(prompt, max_tokens);
  std::cout<<"[debug] After request"<<std::endl;
  
  while(!segment_runner.IsPrefillEnd()){
    segment_runner.Prefill(1);
    std::cout<<"PEFILL"<<std::endl;
  }

  bool is_end = false;
  std::string output = prompt;

  while(!segment_runner.IsEnd()){
    std::cout<<"PRE_SEGMENT"<<std::endl;
    std::string delta = segment_runner.Execute(1);
    std::cout<<"POST_SEGMENT"<<std::endl;
    output += delta;
  }

  std::cout<<"MLC-LLM Output: "<<output<<std::endl;

  return 0;
}