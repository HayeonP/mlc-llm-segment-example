

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

#include <picojson.h>
#include <serve/config.h>
#include <serve/threaded_engine.h>
#include <serve/data.h>
#include <serve/config.h>
#include <serve/request.h>
#include <tokenizers/tokenizers.h>
#include <tokenizers/streamer.h>
#include <frontend/engine_base.h>
#include <json_ffi/conv_template.h>
#include <json_ffi/openai_api_protocol.h>
#include <frontend/mlc_chat_config.h>

#include <tvm/runtime/device_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/optional.h>
#include <tvm/ffi/any.h>
#include <tvm/runtime/int_tuple.h>

#include <stdexcept>
#include <csignal>
#include <atomic>

#include "./utils.h"
#include "./thread_safe_queue.h"
#include "./scope_fail.h"
#include "./generator.h"

using namespace tvm;
using namespace ffi;

using TokenIds = IntTuple; // tvm::ffi::Shape
using String = tvm::ffi::String;

using ModelArg = std::unordered_map<std::string, std::string>;
using Conversation = mlc::llm::json_ffi::Conversation;
using ChatCompletionRequest = mlc::llm::json_ffi::ChatCompletionRequest;
using ChatCompletionMessage = mlc::llm::json_ffi::ChatCompletionMessage;
using ChatCompletionMessageContent = mlc::llm::json_ffi::ChatCompletionMessageContent;
using ChatCompletionStreamResponse = mlc::llm::json_ffi::ChatCompletionStreamResponse;
using ChatCompletionStreamResponseChoice = mlc::llm::json_ffi::ChatCompletionStreamResponseChoice;
using ChatCompletionResponse = mlc::llm::json_ffi::ChatCompletionResponse;
using ChatCompletionResponseChoice = mlc::llm::json_ffi::ChatCompletionResponseChoice;

std::atomic<bool> g_interrupted(false);



void signal_handler(int signal){
  if(signal == SIGINT){
    g_interrupted = true;
  }
}

void check_interrupt(){
  if (g_interrupted){
    throw std::runtime_error("SIGINT");
  }
}

struct CallbackStreamOutput {  
  String delta_text;
  Optional<Array<String>> delta_logprob_json_strs;
  Optional<String> finish_reason;
  Optional<String> request_final_usage_json_str;
};

struct SingleRequestStreamOutput {
  IntTuple delta_token_ids;
  Optional<Array<String>> delta_logprob_json_strs;
  Optional<String> finish_reason;
  Optional<String> request_final_usage_json_str;
  String extra_prefix_string;
};

struct TopLogProbs{
  std::string token;
  float logprob;
  std::optional<std::vector<int>> bytes;
};

using TopLogProbs = struct TopLogProbs;

struct LogProbsContent{
  std::string str;
  float logrpob;
  std::optional<std::vector<int>> bytes;
  std::vector<TopLogProbs> top_logprobs;
};

using LogProbsContent = struct LogProbsContent;

struct LogProbs{
  std::vector<LogProbsContent> content;
};

using LogProbs = struct LogProbs;

using CallbackStreamOutput = struct CallbackStreamOutput;
using SingleRequestStreamOutput = struct SingleRequestStreamOutput;

class CppEngine {
public:
  CppEngine() { std::signal(SIGINT, signal_handler); }
  ~CppEngine(){
    tvm::ffi::Function exit_background_loop_func = _engine_module->GetFunction("exit_background_loop");
    exit_background_loop_func();
  
    _background_loop_thread.join();
    _background_stream_back_loop_thread.join();
  }
  void init(std::string model, tvm::Device& device, std::string model_lib, std::string mode);
  ChatCompletionRequest create_chat_completion_request(std::string& model, std::string prompt, int max_tokens, bool stream);
  ChatCompletionResponse create(std::optional<std::string>& request_id, ChatCompletionRequest request); // class ChatCompletion -> create()
  std::string response_to_str(ChatCompletionResponse& response);

private:
  void _check_engine_config(std::string model, std::string model_lib, EngineMode mode, mlc::llm::serve::EngineConfig engine_config);
  std::vector<ModelInfo> _parse_members(std::string model, std::string model_lib);
  void _convert_model_info(ModelInfo model, Conversation& conversation, std::vector<std::string>& config_file_paths, std::string& output_model_path, std::string& output_model_lib);
  std::vector<ModelInfo> _parse_models(std::string model, std::string model_lib);
  void _process_model_args(std::vector<ModelInfo>& models, tvm::Device& device, mlc::llm::serve::EngineConfig& engine_config, std::vector<ModelArg>& output_model_args, std::vector<std::string>& output_config_file_paths, Conversation& output_conv_template);
  void _sync_request_stream_callback(tvm::ffi::Array<mlc::llm::serve::RequestStreamOutput> delta_outputs);
  std::variant<Generator<ChatCompletionStreamResponse>, ChatCompletionResponse> _chat_completion(Optional<String>& request_id, ChatCompletionRequest request);
  Generator<ChatCompletionStreamResponse> _handle_chat_completion(Optional<String>& request_id, ChatCompletionRequest request);
  Generator<std::vector<CallbackStreamOutput>> _generate(std::vector<TokenIds> prompts, mlc::llm::serve::GenerationConfig generation_config, Optional<String>& request_id);
  void _request_stream_callback_impl(std::vector<mlc::llm::serve::RequestStreamOutput> delta_outputs, std::vector<std::vector<CallbackStreamOutput>>& output_request_outputs, Optional<String>& output_request_final_usage_json_str);
  
  // Functions in engine_base.py
  std::optional<ChatCompletionStreamResponse> process_chat_completion_stream_output(std::vector<CallbackStreamOutput>& delta_outputs, mlc::llm::json_ffi::ChatCompletionRequest& request, Optional<String> request_id, bool use_function_calling, Array<Optional<String>> finish_reasons);
  ChatCompletionResponse wrap_chat_completion_response(std::string& request_id, std::string& model, std::vector<std::string>& output_texts, std::vector<std::string>& finish_reasons);  
private:
  Conversation _conv_template;
  std::vector<mlc::llm::json_ffi::ModelConfig> _model_config_list;
  mlc::llm::Tokenizer _tokenizer;
  tvm::runtime::Module _engine_module;
  std::thread _background_loop_thread;
  std::thread _background_stream_back_loop_thread;
  bool _terminated;
  mlc::llm::serve::EngineConfig _engine_config;
  int _max_input_sequence_length;
  std::optional<mlc::llm::serve::EventTraceRecorder> _trace_recorder;
  BlockingQueue<tvm::ffi::Array<mlc::llm::serve::RequestStreamOutput>> _sync_output_queue;
  std::vector<mlc::llm::TextStreamer> _sync_text_streamers;
};


ChatCompletionResponse CppEngine::create(std::optional<std::string>& request_id, ChatCompletionRequest request){
  Optional<String> request_id_;
  if(request_id.has_value()){
    request_id_ = String(request_id.value());
  }
  else{
    request_id_ = String(std::string("chatcmpl-") + mlc::llm::utils::Uuid4Hex());
  }

  try{
    auto response_ = _chat_completion(request_id_, request);
    ChatCompletionResponse* response = std::get_if<ChatCompletionResponse>(&response_);
    return *response;
  }
  catch (const std::bad_variant_access& ex){
    std::cout <<"[ERROR] Output of _chat_completion is invalid" << std::endl;
    exit(0);
  }

}

std::variant<Generator<ChatCompletionStreamResponse>, ChatCompletionResponse> CppEngine::_chat_completion(Optional<String>& request_id, ChatCompletionRequest request){
  Generator<ChatCompletionStreamResponse> cmpl_generator = _handle_chat_completion(request_id, request);
  if(request.stream){
    // # Stream response
    return std::move(cmpl_generator);
  }

  // # Normal response
  std::optional<picojson::value> result_final_usage;
  std::vector<std::string> output_texts(request.n);
  std::vector<std::string> finish_reasons(request.n);
  std::optional<std::vector<std::vector<LogProbsContent>>> logprob_results; // Doesn't support logprobs now (tvm v0.21.0 doesn't support it in CPP)
  if(request.logprobs){
    logprob_results = std::vector<std::vector<LogProbsContent>>(request.n);
  }

  while(cmpl_generator.move_next()){
    ChatCompletionStreamResponse response = cmpl_generator.current_value();

    //TODO: Doesn't care usage for now.

    for(auto choice : response.choices){
      if(choice.delta.content.IsNull()) continue;

      output_texts[choice.index] += choice.delta.content.Text();
      
      if(choice.finish_reason.has_value() && !finish_reasons[choice.index].empty()){
        finish_reasons[choice.index] = mlc::llm::utils::FinishReasonToStr(choice.finish_reason.value());
      }
      // TODO: Doesn't support logprobs now (tvm v0.21.0 doesn't support it in CPP)      
    }
  }  

  // TODO: Doesn't support function call for now

  std::string request_id_str(request_id.value());
  return wrap_chat_completion_response(request_id_str, request.model.value(), output_texts, finish_reasons); // TODO: Doesn't support logprob, funciton call, tool calls for now  
}

// TODO: Doesn't support logprob, funciton call, tool calls for now
ChatCompletionResponse CppEngine::wrap_chat_completion_response(std::string& request_id, std::string& model, std::vector<std::string>& output_texts, std::vector<std::string>& finish_reasons){
  ChatCompletionResponse response;

  response.id = request_id;
  for(int i = 0; i < output_texts.size(); ++i){
    ChatCompletionResponseChoice choice;
    std::string output_text = output_texts[i];
    std::string finish_reason = finish_reasons[i];
    // TODO: No tool calls for now
    
    choice.index = i;
    choice.finish_reason = mlc::llm::utils::StrToFinishReason(finish_reason);
    choice.message.role = "assistant";
    choice.message.content = ChatCompletionMessageContent(output_text); // TODO: Doesn't support tool_calls for now

    // TODO: Doesn't support logprobs for now

    response.choices.push_back(choice);
  }
  
  response.model = model;
  response.system_fingerprint = "";

  return response;
}


Generator<ChatCompletionStreamResponse> CppEngine::_handle_chat_completion(Optional<String>& request_id, ChatCompletionRequest request){
  // ***** engine_base.process_chat_completion_request ***** START
  if(!_trace_recorder.has_value()){
    std::cout<< "[ERROR] Trace recorder is not initialized" << std::endl;
    exit(0);
  }
  _trace_recorder.value()->AddEvent(request_id.value(), std::string("receive request"));
  
  std::string role;
  ChatCompletionMessageContent content;

  for(ChatCompletionMessage message : request.messages){
    role = message.role;
    content = message.content;
    if(role == "system"){
      if(!content.IsNull()){
        _conv_template.system_message = content.Text();
        continue;
      }
      _conv_template.system_message = "";
    }
    _conv_template.messages.push_back(message);
  }

  ChatCompletionMessage empty_assistant_message;
  empty_assistant_message.role = "assistant";
  _conv_template.messages.push_back(empty_assistant_message);

  // - Get the prompt from template, and encode to token ids.
  // - Check prompt length
  _trace_recorder.value()->AddEvent(request_id.value(), std::string("start tokenization"));
  
  std::vector<TokenIds> prompts;
  // ***** engine_utils.process_prompts ***** START // TODO: Support more types
  std::vector<std::string> input_prompts = mlc::llm::utils::ConvertConversationToPrompt(_conv_template); 

  auto tokenizer_encode_func_ = tvm::ffi::Function::GetGlobal("mlc.tokenizers.TokenizerEncode");
  if(!tokenizer_encode_func_.has_value()){
    std::cout<<"[ERROR] Cannot create threaded engine"<<std::endl;
    exit(0);
  }
  tvm::ffi::Function tokenizer_encode_func = tokenizer_encode_func_.value();

  // TODO: Case 1 and 2 are skipped
  // Case 1. The prompt is single string.
  // Case 2. The pormpt is a list of token ids. 

  // Case 3. A list of prompts
  for(auto& input_prompt : input_prompts){
    auto encoded_prompt = tokenizer_encode_func(_tokenizer, tvm::ffi::String(input_prompt)).cast<TokenIds>();
    prompts.push_back(encoded_prompt);
  }
  // return output_prompts;
  // ***** engine_utils.process_prompts ***** END

  _trace_recorder.value()->AddEvent(request_id.value(), std::string("finish tokenization"));

  if(_conv_template.system_prefix_token_ids.has_value()){
    // TODO: SKIP
  }

  // ***** check_and_get_prompts_length ***** START
  int prompt_length = 0;
  for(auto p : prompts) prompt_length += p.size();

  if(prompt_length > _max_input_sequence_length){
    std::cout << "[ERROR] Request prompt has " << prompt_length << "tokens in total,";
    std::cout <<" larger than the model input length limit " << _max_input_sequence_length << "." << std::endl;
    exit(0);
  }
  // ***** check_and_get_prompts_length ***** END
  
  // ***** engine_utils.get_generation_config ***** START
  ObjectPtr<mlc::llm::serve::GenerationConfigNode> generation_config_node = tvm::ffi::make_object<mlc::llm::serve::GenerationConfigNode>();
  auto extra_stop_token_ids = _conv_template.stop_token_ids;
  auto extra_stop_str = _conv_template.stop_str;

  // kwargs[arg_name] = getattr(request, arg_nbame)
  generation_config_node->n = request.n;
  if(request.temperature.has_value()) generation_config_node->temperature = request.temperature.value();
  if(request.top_p.has_value()) generation_config_node->top_p = request.top_p.value();
  if(request.max_tokens.has_value()) generation_config_node->max_tokens = request.max_tokens.value();
  if(request.frequency_penalty.has_value()) generation_config_node->frequency_penalty = request.frequency_penalty.value();
  if(request.presence_penalty.has_value()) generation_config_node->presence_penalty = request.presence_penalty.value();
  if(request.logit_bias.has_value()) generation_config_node->logit_bias = request.logit_bias.value();
  if(request.seed.has_value()) generation_config_node->seed = request.seed.value();
  if(request.response_format.has_value()) generation_config_node->response_format = request.response_format.value();
  if(request.debug_config.has_value()) generation_config_node->debug_config = request.debug_config.value();
  if(!request.max_tokens.has_value()) generation_config_node->max_tokens = -1;  // Setting to -1 means the generation will not stop until
                                                                // exceeding model capability or hit any stop criteria.
                                                                  
                                                                  
  if(request.stop.has_value()){
    // TODO: Skip
  }
  // TODO: We consider ChatCOmpletionRequest only
  generation_config_node->logprobs = request.logprobs;
  generation_config_node->top_logprobs = request.top_logprobs;
  // ***** engine_utils.get_generation_config ***** END

  if(extra_stop_token_ids.size() > 0){
    generation_config_node->stop_token_ids.insert(generation_config_node->stop_token_ids.end(), extra_stop_token_ids.begin(), extra_stop_token_ids.end());
  }

  if(extra_stop_str.size() > 0){
    generation_config_node->stop_strs.reserve(generation_config_node->stop_strs.size() + extra_stop_str.size());
    for (const auto& s : extra_stop_str) generation_config_node->stop_strs.push_back(tvm::ffi::String(s));
  }

  mlc::llm::serve::GenerationConfig generation_config(generation_config_node);

  // return prompts, generation_cfg, conv_template.use_function_calling, prompt_length
  // ***** engine_base.process_chat_completion_request ***** END

  // TODO: use_function_calling is always false (cpp struct Conversation doesn't have it)
  
  Array<Optional<String>> finish_reasons(generation_config->n, Optional<String>()); // TODO: push_bakc으로 해야하나?  
    
  _trace_recorder.value()->AddEvent(request_id.value(), std::string("invoke generate"));
  auto generate_output = _generate(prompts, generation_config, request_id);

  while(generate_output.move_next()){
    std::vector<CallbackStreamOutput> delta_outputs = generate_output.current_value();

    bool use_function_calling = false; // TODO: use_function_calling is always "false" for now.
    std::optional<ChatCompletionStreamResponse> response = process_chat_completion_stream_output(delta_outputs, request, request_id, false, finish_reasons);                                    

    // TODO: ### START HERE
    if(response.has_value()){
      auto v = response.value();

      ChatCompletionStreamResponse output_response;
      output_response = response.value();
      co_yield output_response;
    }
  }

  _trace_recorder.value()->AddEvent(request_id.value(), std::string("finish"));
}

std::optional<ChatCompletionStreamResponse> CppEngine::process_chat_completion_stream_output(std::vector<CallbackStreamOutput>& delta_outputs, mlc::llm::json_ffi::ChatCompletionRequest& request, Optional<String> request_id, bool use_function_calling, Array<Optional<String>> finish_reasons){
  std::optional<ChatCompletionStreamResponse> response;

  // # we always stream back the final chunk with usage
  Optional<String> is_final_chunk;
  if(delta_outputs[0].request_final_usage_json_str.has_value()){
    is_final_chunk = delta_outputs[0].request_final_usage_json_str;
  }
  if(is_final_chunk.has_value()){
    if(delta_outputs.size() != 1){
      std::cout<< "[ERROR] Final delta output size sholud not be bigger than 1" << std::endl;
      exit(0);
    }

    _trace_recorder.value()->AddEvent(request_id.value(), std::string("yield final usage"));

    ChatCompletionStreamResponse response_value;
    response_value.id = static_cast<std::string>(request_id.value());
    response_value.choices.clear();
    response_value.model = request.model.value();
    response_value.system_fingerprint = "";
    // ??? response.usage = model_validate_json

    // # non streaming mode always comes with usage
    if(!request.stream) return std::nullopt;

    // TODO: No stream options for now
    
    response = response_value;
    return response;
  }

  // # normal chunk
  if(!delta_outputs.size() == request.n){
    std::cout<<"[ERROR] delta_outputs.size() != request.n"<<std::endl;
    exit(0);
  }
  std::vector<ChatCompletionStreamResponseChoice> choices;
  
  for(int i = 0; i < delta_outputs.size(); i++){
    auto delta_output = delta_outputs[i];
    bool finish_reason_updated = false;
    if(delta_output.finish_reason.has_value() && finish_reasons[i].has_value()){
      // TODO: Skip use_function_calling condition for now.
      finish_reasons.Set(i, delta_output.finish_reason.value());
      finish_reason_updated = true;
    }
    if(!finish_reason_updated && delta_output.delta_text.empty()){
      // # Ignore empty delta text when finish reason is not updated.
      _trace_recorder.value()->AddEvent(request_id.value(), std::string("skip empty delta text"));
    }

    ChatCompletionStreamResponseChoice choice;
    choice.index = i;
    if(finish_reasons[i].has_value()){
      std::string finish_reason_str = finish_reasons[i].value();
      choice.finish_reason = mlc::llm::utils::StrToFinishReason(finish_reason_str);
    }
    else{
      choice.finish_reason = std::nullopt;
    }
    
    ChatCompletionMessage delta;
    delta.role = "assistant";
    delta.content = ChatCompletionMessageContent(static_cast<std::string>(delta_output.delta_text));
    
    choice.delta = delta;

    // TODO: Skip logprob
    choices.push_back(choice);
  }

  if(choices.size() == 0){
    // # Skip return when there is no delta output and no number of completion tokens.
    return std::nullopt;
  }

  ChatCompletionStreamResponse response_value;
  response_value.id = static_cast<std::string>(request_id.value());
  
  response_value.choices = std::move(choices);
  response_value.model = request.model.value();
  response_value.system_fingerprint = "";
  _trace_recorder.value()->AddEvent(request_id.value(), std::string("yield delta output"));

  response = response_value;
  
  return response;
}

// Return Iterator
Generator<std::vector<CallbackStreamOutput>> CppEngine::_generate(std::vector<TokenIds> prompts, mlc::llm::serve::GenerationConfig generation_config, Optional<String>& request_id){
  // TODO: We only cares List[List[int]] prompts for now 
  // **** convert_prompts_to_data ***** START 
  
  Array<mlc::llm::serve::TokenData> input_data;

  auto init_token_data_func_ = tvm::ffi::Function::GetGlobal("mlc.serve.TokenData");
  if(!init_token_data_func_.has_value()){
    std::cout<<"[ERROR] Cannot create token data"<<std::endl;
    exit(0);
  }
  tvm::ffi::Function init_token_data_func = init_token_data_func_.value();
  for(IntTuple& prompt : prompts){
    std::vector<tvm::ffi::AnyView> prompt_vec;
    for(auto& v : prompt){
      prompt_vec.push_back(static_cast<int32_t>(v));
    }
    
    // tvm::ffi::Any init_token_data_rv = init_token_data_func(tvm::ffi::PackedArgs(prompt_vec.data(), prompt_vec.size()));
    tvm::ffi::Any init_token_data_rv;
    init_token_data_func.CallPacked(tvm::ffi::PackedArgs(prompt_vec.data(), prompt_vec.size()), &init_token_data_rv);
    mlc::llm::serve::TokenData token_data = Downcast<mlc::llm::serve::TokenData>(init_token_data_rv);
    input_data.push_back(token_data);
  }
  
  // _ffi["create_request"]
  tvm::ffi::Function create_request_func = _engine_module->GetFunction("create_request");
  picojson::object obj = generation_config->AsJSON();
  picojson::value val(obj);
  std::string generation_config_str = val.serialize();
  
  tvm::ffi::Any create_request_rv = create_request_func(request_id, input_data, generation_config_str);
  
  
  mlc::llm::serve::Request request = Downcast<mlc::llm::serve::Request>(create_request_rv);
  // Record the stream in the tracker
  _sync_output_queue.clear();
  _sync_text_streamers.clear();
  for(int i = 0; i < generation_config->n; i++){
    _sync_text_streamers.push_back(mlc::llm::TextStreamer(_tokenizer));
  }

  // _ffi["add_request"]
  tvm::ffi::Function add_request_func = _engine_module->GetFunction("add_request");
  add_request_func(request);

  tvm::ffi::Function abort_request_func = _engine_module->GetFunction("abort_request");
  
  // abort_func is executed when this function returns
  ScopeFail guard([&abort_request_func] { abort_request_func(); });

  while(true){
    tvm::ffi::Array<mlc::llm::serve::RequestStreamOutput> delta_outputs_ = _sync_output_queue.get();
    std::vector<mlc::llm::serve::RequestStreamOutput> delta_outputs(delta_outputs_.begin(), delta_outputs_.end());
    std::vector<std::vector<CallbackStreamOutput>> request_outputs;
    Optional<String> request_final_usage_json_str;
    
    _request_stream_callback_impl(delta_outputs, request_outputs, request_final_usage_json_str);

    for(std::vector<CallbackStreamOutput>& request_output : request_outputs){
      co_yield request_output;
    }

    if(request_final_usage_json_str.has_value()){
      std::vector<CallbackStreamOutput> output;
      CallbackStreamOutput output_value;
      output_value.delta_text = "";
      output_value.delta_logprob_json_strs = std::nullopt;
      output_value.finish_reason = std::nullopt;
      output_value.request_final_usage_json_str = request_final_usage_json_str;
      output.push_back(output_value);
      co_yield output;
      break;
    }
  }
}

void CppEngine::_request_stream_callback_impl(std::vector<mlc::llm::serve::RequestStreamOutput> delta_outputs, std::vector<std::vector<CallbackStreamOutput>>& output_request_outputs, Optional<String>& output_request_final_usage_json_str){  
  std::vector<std::vector<CallbackStreamOutput>>& batch_outputs = output_request_outputs;

  for(auto v : batch_outputs) v.clear();
  batch_outputs.clear();

  auto request_stream_output_unpack_func_ = tvm::ffi::Function::GetGlobal("mlc.serve.RequestStreamOutputUnpack");
  if(!request_stream_output_unpack_func_.has_value()){
    std::cout<<"[ERROR] Cannot unpack request stream output"<<std::endl;
    exit(0);
  }
  tvm::ffi::Function request_stream_output_unpack_func = request_stream_output_unpack_func_.value();

  for(mlc::llm::serve::RequestStreamOutput delta_output : delta_outputs){
    String request_id;
    std::vector<SingleRequestStreamOutput> stream_outputs; // field[0]

    // ***** unpck() ***** START //
    Array<IntTuple> group_delta_token_ids; // field[1]
    Optional<Array<Array<String>>> group_delta_logprob_json_strs; // field[2]
    Array<Optional<String>> group_finish_reason; // field[3]
    Optional<String> request_final_usage_json_str; // field[4]
    Array<String> group_extra_prefix_string; // field[5]
    tvm::ffi::Any fields_ = request_stream_output_unpack_func(delta_output);
    Array<tvm::ffi::ObjectRef> fields = Downcast<Array<tvm::ffi::ObjectRef>>(fields_);
    
    request_id = Downcast<String>(fields[0]);    
    group_delta_token_ids = Downcast<Array<IntTuple>>(fields[1]);
    group_delta_logprob_json_strs = Downcast<Array<Array<String>> >(fields[2]);
    group_finish_reason = Downcast<Array<Optional<String>>>(fields[3]);
    request_final_usage_json_str = Downcast<Optional<String>>(fields[4]);
    group_extra_prefix_string = Downcast<Array<String>>(fields[5]);
    
    if(request_final_usage_json_str.has_value()){
      SingleRequestStreamOutput stream_output_value;
      stream_output_value.request_final_usage_json_str = request_final_usage_json_str.value();
      stream_outputs.push_back(stream_output_value);
    }
    else{
      for(int i = 0; i < group_delta_token_ids.size(); ++i){
        SingleRequestStreamOutput stream_output_value;
        
        IntTuple delta_token_ids = group_delta_token_ids[i];
        Optional<String> finish_reason = group_finish_reason[i];
        String extra_prefix_string = group_extra_prefix_string[i];
        Array<String> delta_logprob_json_strs;
        if(group_delta_logprob_json_strs.has_value()){
          delta_logprob_json_strs = group_delta_logprob_json_strs.value()[i];
        }
                
        stream_output_value.delta_token_ids = std::move(delta_token_ids);
                
        stream_output_value.delta_logprob_json_strs = std::move(delta_logprob_json_strs);
        if(finish_reason.has_value()){
          stream_output_value.finish_reason = finish_reason.value();
        }
        stream_output_value.request_final_usage_json_str = std::nullopt;
        stream_output_value.extra_prefix_string = extra_prefix_string;

        stream_outputs.push_back(stream_output_value);
      }
    }
    // return reuqest_id, stream_output_value
    // ***** unpck() ***** END //

    /////////////////////////////////////////////////////////////////////
    _trace_recorder.value()->AddEvent(request_id, std::string("start callback"));

    // final chunk is now always indicated by a chunk
    // where usage json is present
    // the backend engine always streams back this chunk
    // regardless of include_usage option    
    bool is_final_chunk = false;
    if(stream_outputs[0].request_final_usage_json_str.has_value()) is_final_chunk = true;
    if(is_final_chunk){
      output_request_final_usage_json_str = stream_outputs[0].request_final_usage_json_str;
      return;
    }

    std::vector<CallbackStreamOutput> outputs;
    for(int i = 0; i < stream_outputs.size(); ++i){
      SingleRequestStreamOutput stream_output = stream_outputs[i];      
      mlc::llm::TextStreamer text_streamer = _sync_text_streamers[i];
      
      _trace_recorder.value()->AddEvent(request_id, std::string("start detokenization"));
      
      String delta_text("");
      delta_text = delta_text + stream_output.extra_prefix_string;
      if(stream_output.delta_token_ids.size() > 0){
        delta_text = delta_text + text_streamer->Put({group_delta_token_ids[i]->data, group_delta_token_ids[i]->data + group_delta_token_ids[i]->size});
      }
      
      if(stream_output.finish_reason.has_value()){
        delta_text = delta_text + text_streamer->Finish();
      }
      
      _trace_recorder.value()->AddEvent(request_id, std::string("finish detokenization"));

      CallbackStreamOutput callback_stream_output_value;
      callback_stream_output_value.delta_text = delta_text;
      callback_stream_output_value.delta_logprob_json_strs = stream_output.delta_logprob_json_strs;
      callback_stream_output_value.finish_reason = stream_output.finish_reason;
      callback_stream_output_value.request_final_usage_json_str = stream_output.request_final_usage_json_str;
      outputs.push_back(callback_stream_output_value);
    }
    batch_outputs.push_back(outputs);
    _trace_recorder.value()->AddEvent(request_id, std::string("finish callback"));
  }

  output_request_final_usage_json_str = std::nullopt;
}

void CppEngine::_check_engine_config(std::string model, std::string model_lib, EngineMode mode, mlc::llm::serve::EngineConfig engine_config){
  if(engine_config->model != "" && engine_config->model != model){
    std::cout << "[ERROR] The argument \"model\" of engine constructor is \""<< model <<"\", while the \"model\" field in argument \"engine_config\" is \"" << engine_config->model <<"\". Please set the \"engine_config->model\" to \"\" or set it to the same as the argument \"model\"." << std::endl;
    exit(0);
  }

  if(engine_config->model_lib != "" && model_lib != "" && engine_config->model_lib != model_lib){
    std::cout << "[ERROR] The argument \"model_lib\" of engine constructor is \""<< model_lib <<"\", while the \"model_lib\" field in argument \"engine_config\" is \"" << engine_config->model_lib <<"\". Please set the \"engine_config->model_lib\" to \"\" or set it to the same as the argument \"model_lib\"." << std::endl;
    exit(0);
  }

  if(engine_config->kv_cache_page_size != 16){
    std::cout << "[ERROR] KV cache only supports page size 16. while \"kv_cache_page_size\" field in argument \"engine_config\" is \"" << engine_config->kv_cache_page_size << "\". Please set \"engine_config->kv_cache_page_size\" to 16." << std::endl;
    exit(0);
  }

  return;
}

// Not support additional models
std::vector<ModelInfo> CppEngine::_parse_members(std::string model, std::string model_lib){
  std::vector<ModelInfo> models;
  ModelInfo model_info;
  model_info.model = model;
  model_info.model_lib = model_lib;
  models.emplace_back(model_info);

  return models;
}

void CppEngine::_convert_model_info(ModelInfo model, Conversation& conversation, std::vector<std::string>& config_file_paths, std::string& output_model_path, std::string& output_model_lib){
  std::string model_path = model.model;
  std::string mlc_config_path = model_path + "/mlc-chat-config.json";
  config_file_paths.emplace_back(mlc_config_path);

  std::string mlc_config = mlc::llm::utils::ReadJSONAsString(mlc_config_path);
  MLCChatConfig mlc_chat_config;
  mlc_chat_config.FromJsonString(mlc_config);

  // TODO: Add None condition
  conversation = mlc_chat_config.conv_template;

  // TODO: Right now, it just set model.model_lib to output_model_lib
  
  output_model_path = model_path;
  output_model_lib = model.model_lib;

  return;
}

std::vector<ModelInfo> CppEngine::_parse_models(std::string model, std::string model_lib){ // Doesn't support addtional models
  std::vector<ModelInfo> models;
  ModelInfo model_info;
  model_info.model = model;
  model_info.model_lib = model_lib;
  models.emplace_back(model_info);
  return models;
}

void CppEngine::_process_model_args(std::vector<ModelInfo>& models, tvm::Device& device, mlc::llm::serve::EngineConfig& engine_config, std::vector<ModelArg>& output_model_args, std::vector<std::string>& output_config_file_paths, Conversation& output_conv_template){
  
  Conversation conversation;
  std::vector<std::string> config_file_paths;
  std::vector<ModelArg> model_args;

  for(auto model : models){
    std::string model_path;
    std::string model_lib_path;
    _convert_model_info(model, conversation, config_file_paths, model_path, model_lib_path);
    
    ModelArg model_arg = {
      {"model", model_path},
      {"model_lib", model_lib_path}
    };
    model_args.push_back(model_arg);
  }

  output_model_args = model_args;
  output_config_file_paths = config_file_paths;
  output_conv_template = conversation;

  return;
}


// TODO: Add engine config to args
void CppEngine::init(std::string model, tvm::Device& device, std::string model_lib, std::string mode){
  // - Check the fields fields of `engine_config`.
  mlc::llm::serve::EngineConfig engine_config(make_object<mlc::llm::serve::EngineConfigNode>());
  // _check_engine_config(model, model_lib, engine_config); // Not necessary

  // - Initialize model loading info.
  std::vector<ModelInfo> models = _parse_models(model, model_lib);
  std::vector<ModelArg> model_args;
  std::vector<std::string> model_config_paths;
  _process_model_args(models, device, engine_config, model_args, model_config_paths, _conv_template);
  
  // - Load the raw model config
  for(int i = 0; i < models.size(); i++){
    models[i].model_lib = model_args[i]["model_lib"];
    std::string model_config_json_string = mlc::llm::utils::ReadJSONAsString(model_config_paths[i]);
    picojson::value v;
    std::string err = picojson::parse(v, model_config_json_string);
    const picojson::object& obj = v.get<picojson::object>();
    _model_config_list.emplace_back(mlc::llm::json_ffi::ModelConfig::FromJSON(obj));
  }

  // - Pring logging for regarding the model selection
  // TODO: SKIP

  // - Initialize engine state and engine --> // 에매한게, python과 cpp의 EngineState class가 형태가 다르다
  // Skip creating engine state  
  
  // tvm.get_global_func["mlc.serve.create_threaded_engine"]
  auto create_threaded_engine_func_ = tvm::ffi::Function::GetGlobal("mlc.serve.create_threaded_engine");
  if(!create_threaded_engine_func_.has_value()){
    std::cout<<"[ERROR] Cannot create threaded engine"<<std::endl;
    exit(0);
  }
  tvm::ffi::Function create_threaded_engine_func = create_threaded_engine_func_.value();
  _engine_module = create_threaded_engine_func().cast<tvm::runtime::Module>();
  
  _tokenizer = mlc::llm::Tokenizer::FromPath(model_args[0]["model"]);

  // _ffi["init_threaded_engine"]
  tvm::ffi::Function init_threaded_engine_func = _engine_module->GetFunction("init_threaded_engine");

  tvm::ffi::Function get_request_stream_callback = tvm::ffi::Function::FromPacked([this](tvm::ffi::PackedArgs args, tvm::ffi::Any* rv) {
    auto delta_outputs = Downcast<tvm::ffi::Array<mlc::llm::serve::RequestStreamOutput>>(args[0]);
    this->_sync_request_stream_callback(delta_outputs);
  });
  tvm::ffi::Optional<tvm::ffi::Function> opt_callback(get_request_stream_callback);

  auto create_event_trace_recorder_func_ = tvm::ffi::Function::GetGlobal("mlc.serve.EventTraceRecorder");
  if(!create_event_trace_recorder_func_.has_value()){
    std::cout<<"[ERROR] Cannot create event trace recorder"<<std::endl;
    exit(0);
  }
  tvm::ffi::Function create_event_trace_recorder_func = create_event_trace_recorder_func_.value();
  _trace_recorder = create_event_trace_recorder_func().cast<mlc::llm::serve::EventTraceRecorder>();
  tvm::ffi::Optional<mlc::llm::serve::EventTraceRecorder> opt_recorder(_trace_recorder);

  init_threaded_engine_func(device, opt_callback, opt_recorder); 

  // - Create the background engine-driving thread and start the loop
  // _ffi["run_background_loop"]
  tvm::ffi::Function run_background_loop_func = _engine_module->GetFunction("run_background_loop");
  _background_loop_thread = std::thread([func = std::move(run_background_loop_func)](){
      func();
  });

  // _ffi["run_background_stream_back_loop"]
  tvm::ffi::Function run_background_stream_back_loop_func = _engine_module->GetFunction("run_background_stream_back_loop");
  _background_stream_back_loop_thread = std::thread([func = std::move(run_background_stream_back_loop_func)](){
      func();
  });

  _terminated = false;

  // Set to same as python value
  engine_config->model = tvm::ffi::String(model_args[0]["model"]);
  engine_config->model_lib = model_args[0]["model_lib"];
  // TODO: Support additional model
  engine_config->max_total_sequence_length = 8192;
  engine_config->max_single_sequence_length = 131072;
  engine_config->prefill_chunk_size = 8192;
  engine_config->prefix_cache_max_num_recycling_seqs = 4;
  engine_config->mode = mlc::llm::serve::EngineMode::kLocal; // TODO: Support async
  engine_config->verbose = true;

  // _ffi["reload"]
  tvm::ffi::Function reload_func = _engine_module->GetFunction("reload");
  reload_func(engine_config->AsJSONString());
  
  tvm::ffi::Function get_complete_engine_config_func = _engine_module->GetFunction("get_complete_engine_config");
  std::string complete_engine_config_json_str = get_complete_engine_config_func().cast<std::string>();
  _engine_config = std::move(mlc::llm::utils::ParseEngineConfigFromJSONString(complete_engine_config_json_str));
  
  _max_input_sequence_length = std::min(_engine_config->max_single_sequence_length, _engine_config->max_total_sequence_length);
  
  return;
}


void CppEngine::_sync_request_stream_callback(tvm::ffi::Array<mlc::llm::serve::RequestStreamOutput> delta_outputs){
  _sync_output_queue.put_nowait(delta_outputs);
}

ChatCompletionRequest CppEngine::create_chat_completion_request(std::string& model, std::string prompt, int max_tokens, bool stream){
  ChatCompletionRequest request;
  request.model = model;

  ChatCompletionMessage message;
  message.role = "user";
  message.content = ChatCompletionMessageContent(prompt);
  request.messages.push_back(message);
  request.max_tokens = max_tokens;

  return request;
}

std::string CppEngine::response_to_str(ChatCompletionResponse& response){
  std::string response_str;
  for(auto& choice : response.choices){
    response_str += choice.message.content.Text();
  }

  return response_str;
}

int main(int argc, char* argv[]){
  std::string model_dir = "/home/rubis/workspace/llama/mlc-llm-models/llama-3.2-1b/workspace";
  std::string model_lib_path = model_dir + "/llama-3.2-1b-cuda.so";
  tvm::Device dev{kDLCUDA, 0};
  std::string mode = "local";

  CppEngine engine;
  engine.init(model_dir, dev, model_lib_path, mode);

  std::optional<std::string> request_id = std::nullopt; // no request_id

  // std::string prompt("Answer the following question in one sentence. What is the capital of South Korea?");

  std::string prompt("Can you introduce yourself?");
  int max_tokens = 128;
  bool stream = false;
  ChatCompletionRequest request = engine.create_chat_completion_request(model_dir, prompt, max_tokens, stream);

  ChatCompletionResponse response = engine.create(request_id, request);

  std::cout<<"MLC-LLM Output: "<<engine.response_to_str(response)<<std::endl;

  return 0;
}