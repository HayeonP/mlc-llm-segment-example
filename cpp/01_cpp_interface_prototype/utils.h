#pragma once

#include <string>
#include <fstream>
#include <picojson.h>

#include <json_ffi/conv_template.h>
#include <json_ffi/openai_api_protocol.h>

#include <tvm/runtime/int_tuple.h>

using Conversation = mlc::llm::json_ffi::Conversation;
using ChatCompletionMessageContent = mlc::llm::json_ffi::ChatCompletionMessageContent;
using ChatCompletionMessage = mlc::llm::json_ffi::ChatCompletionMessage;
using ChatCompletionRequest = mlc::llm::json_ffi::ChatCompletionRequest;
using TokenIds = IntTuple;

namespace mlc{
namespace llm{
namespace utils{

std::string ReadJSONAsString(const std::string& json_path) {
  std::ifstream file(json_path);
  if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + json_path);
  }
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

inline bool HasKey(const picojson::object& o, const char* k) {
  return o.find(k) != o.end();
}
inline const picojson::value* GetPtr(const picojson::object& o, const char* k) {
  auto it = o.find(k);
  return it == o.end() ? nullptr : &it->second;
}
inline void ExpectType(bool cond, const char* msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

mlc::llm::serve::EngineConfig ParseEngineConfigFromJSONString(const std::string& json_str) {
  mlc::llm::serve::EngineConfig engine_config(make_object<mlc::llm::serve::EngineConfigNode>());
  picojson::value v;
  std::string err = picojson::parse(v, json_str);
  if (!err.empty()) {
    throw std::runtime_error(std::string("JSON parse error: ") + err);
  }
  ExpectType(v.is<picojson::object>(), "Top-level JSON must be an object.");
  const auto& obj = v.get<picojson::object>();

  // model
  if (auto pv = GetPtr(obj, "model")) {
    ExpectType(pv->is<std::string>(), "\"model\" must be a string.");
    engine_config->model = pv->get<std::string>();
  }
  // model_lib
  if (auto pv = GetPtr(obj, "model_lib")) {
    ExpectType(pv->is<std::string>(), "\"model_lib\" must be a string.");
    engine_config->model_lib = pv->get<std::string>();
  }
  // additional_models
  if (auto pv = GetPtr(obj, "additional_models")) {
    ExpectType(pv->is<picojson::array>(), "\"additional_models\" must be an array.");
    tvm::Array<tvm::String> arr;
    for (const auto& elem : pv->get<picojson::array>()) {
      ExpectType(elem.is<std::string>(), "\"additional_models\" elements must be strings.");
      arr.push_back(elem.get<std::string>());
    }
    engine_config->additional_models = std::move(arr);
  }
  // additional_model_libs (옵션: JSON에 없을 수 있음)
  if (auto pv = GetPtr(obj, "additional_model_libs")) {
    ExpectType(pv->is<picojson::array>(), "\"additional_model_libs\" must be an array.");
    tvm::Array<tvm::String> arr;
    for (const auto& elem : pv->get<picojson::array>()) {
      ExpectType(elem.is<std::string>(), "\"additional_model_libs\" elements must be strings.");
      arr.push_back(elem.get<std::string>());
    }
    engine_config->additional_model_libs = std::move(arr);
  }

  // mode -> enum
  if (auto pv = GetPtr(obj, "mode")) {
    ExpectType(pv->is<std::string>(), "\"mode\" must be a string.");
    engine_config->mode = EngineModeFromString(pv->get<std::string>());
  }

  // gpu_memory_utilization
  if (auto pv = GetPtr(obj, "gpu_memory_utilization")) {
    ExpectType(pv->is<double>(), "\"gpu_memory_utilization\" must be a number.");
    engine_config->gpu_memory_utilization = static_cast<float>(pv->get<double>());
  }

  // kv_cache_page_size
  if (auto pv = GetPtr(obj, "kv_cache_page_size")) {
    ExpectType(pv->is<double>(), "\"kv_cache_page_size\" must be a number.");
    engine_config->kv_cache_page_size = static_cast<int>(pv->get<double>());
  }

  // max_num_sequence
  if (auto pv = GetPtr(obj, "max_num_sequence")) {
    ExpectType(pv->is<double>(), "\"max_num_sequence\" must be a number.");
    engine_config->max_num_sequence = static_cast<int>(pv->get<double>());
  }

  // max_total_sequence_length
  if (auto pv = GetPtr(obj, "max_total_sequence_length")) {
    ExpectType(pv->is<double>(), "\"max_total_sequence_length\" must be a number.");
    engine_config->max_total_sequence_length = static_cast<int64_t>(pv->get<double>());
  }

  // max_single_sequence_length
  if (auto pv = GetPtr(obj, "max_single_sequence_length")) {
    ExpectType(pv->is<double>(), "\"max_single_sequence_length\" must be a number.");
    engine_config->max_single_sequence_length = static_cast<int64_t>(pv->get<double>());
  }

  // prefill_chunk_size
  if (auto pv = GetPtr(obj, "prefill_chunk_size")) {
    ExpectType(pv->is<double>(), "\"prefill_chunk_size\" must be a number.");
    engine_config->prefill_chunk_size = static_cast<int64_t>(pv->get<double>());
  }

  // max_history_size
  if (auto pv = GetPtr(obj, "max_history_size")) {
    ExpectType(pv->is<double>(), "\"max_history_size\" must be a number.");
    engine_config->max_history_size = static_cast<int>(pv->get<double>());
  }

  // prefix_cache_mode -> enum
  if (auto pv = GetPtr(obj, "prefix_cache_mode")) {
    ExpectType(pv->is<std::string>(), "\"prefix_cache_mode\" must be a string.");
    engine_config->prefix_cache_mode = PrefixCacheModeFromString(pv->get<std::string>());
  }

  // prefix_cache_max_num_recycling_seqs
  if (auto pv = GetPtr(obj, "prefix_cache_max_num_recycling_seqs")) {
    ExpectType(pv->is<double>(), "\"prefix_cache_max_num_recycling_seqs\" must be a number.");
    engine_config->prefix_cache_max_num_recycling_seqs = static_cast<int>(pv->get<double>());
  }

  // speculative_mode -> enum
  if (auto pv = GetPtr(obj, "speculative_mode")) {
    ExpectType(pv->is<std::string>(), "\"speculative_mode\" must be a string.");
    engine_config->speculative_mode = SpeculativeModeFromString(pv->get<std::string>());
  }

  // spec_draft_length
  if (auto pv = GetPtr(obj, "spec_draft_length")) {
    ExpectType(pv->is<double>(), "\"spec_draft_length\" must be a number.");
    engine_config->spec_draft_length = static_cast<int>(pv->get<double>());
  }

  // spec_tree_width (옵션)
  if (auto pv = GetPtr(obj, "spec_tree_width")) {
    ExpectType(pv->is<double>(), "\"spec_tree_width\" must be a number.");
    engine_config->spec_tree_width = static_cast<int>(pv->get<double>());
  }

  // prefill_mode -> enum
  if (auto pv = GetPtr(obj, "prefill_mode")) {
    ExpectType(pv->is<std::string>(), "\"prefill_mode\" must be a string.");
    engine_config->prefill_mode = PrefillModeFromString(pv->get<std::string>());
  }

  // verbose
  if (auto pv = GetPtr(obj, "verbose")) {
    ExpectType(pv->is<bool>(), "\"verbose\" must be a boolean.");
    engine_config->verbose = pv->get<bool>();
  }

  return engine_config;
}

inline std::string Uuid4Hex() {
  // 16바이트(128비트) 버퍼
  std::array<uint8_t, 16> bytes{};

  // 난수 생성기: 하드웨어 시드로 PRNG 시드 구성
  std::random_device rd;
  std::mt19937_64 gen((static_cast<uint64_t>(rd()) << 32) ^ rd());
  std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFFu);

  // 16바이트 채우기
  for (size_t i = 0; i < 16; i += 4) {
    uint32_t r = dist(gen);
    bytes[i + 0] = static_cast<uint8_t>((r >> 24) & 0xFF);
    bytes[i + 1] = static_cast<uint8_t>((r >> 16) & 0xFF);
    bytes[i + 2] = static_cast<uint8_t>((r >> 8) & 0xFF);
    bytes[i + 3] = static_cast<uint8_t>((r >> 0) & 0xFF);
  }

  // RFC 4122 규정에 맞게 version(4)과 variant(RFC) 비트 설정
  // time_hi_and_version의 상위 4비트를 0100(version 4)으로
  bytes[6] = static_cast<uint8_t>((bytes[6] & 0x0F) | 0x40);
  // clock_seq_hi_and_reserved의 상위 2비트를 10로
  bytes[8] = static_cast<uint8_t>((bytes[8] & 0x3F) | 0x80);

  // 하이픈 없이 32자리 소문자 hex 직렬화
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (auto b : bytes) {
    oss << std::nouppercase << std::setw(2) << static_cast<unsigned>(b);
  }
  return oss.str();
}

void PrintRequest(mlc::llm::json_ffi::ChatCompletionRequest rq){
  std::cout<<"[debug] <request>"<<std::endl;
  
  std::cout<<"[debug]    <message>"<<std::endl;
  
  if(!rq.messages[0].content.IsNull()) std::cout <<"    Content: "<< rq.messages[0].content.Text() << std::endl;
  else std::cout <<"    Content: None" << std::endl;
  
  std::cout <<"    role: "<< rq.messages[0].role << std::endl;

  if(rq.messages[0].name.has_value()) std::cout <<"    name: "<< rq.messages[0].name.value() << std::endl;
  else std::cout <<"    Name: None" << std::endl;

  if(rq.model.has_value()) std::cout<< "model: "<< rq.model.value()<<std::endl;
  else std::cout<< "model: None" <<std::endl;
  if(rq.frequency_penalty.has_value()) std::cout<< "frequency_penalty: "<< rq.frequency_penalty.value()<<std::endl;
  else std::cout<< "frequency_penalty: None" <<std::endl;
  if(rq.presence_penalty.has_value()) std::cout<< "presence_penalty: "<< rq.presence_penalty.value()<<std::endl;
  else std::cout<< "presence_penalty: None" <<std::endl;
  if(rq.seed.has_value()) std::cout<< "seed: "<< rq.seed.value()<<std::endl;
  else std::cout<< "seed: None" <<std::endl;
  if(rq.temperature.has_value()) std::cout<< "temperature: "<< rq.temperature.value()<<std::endl;
  else std::cout<< "temperature: None" <<std::endl;
  if(rq.top_p.has_value()) std::cout<< "top_p: "<< rq.max_tokens.value()<<std::endl;
  else std::cout<< "top_p: None" <<std::endl;
}

void PrintConversation(mlc::llm::json_ffi::Conversation conv){
    std::cout<<"name: "<<conv.name.value()<<std::endl;
    std::cout<<"system_template: " << conv.system_template<<std::endl;
    std::cout<<"system_message: " << conv.system_message<<std::endl;
    std::cout<<"system_prefix_token_ids: "<<std::endl;
    if(conv.system_prefix_token_ids.has_value()){
        for(auto v : conv.system_prefix_token_ids.value()){
            std::cout<<"    "<<v<<std::endl;
        }
    }
    std::cout<<"add_role_after_system_message: " << conv.add_role_after_system_message<<std::endl;
    std::cout<<"roles: "<<std::endl;
    for(const auto& pair : conv.roles) {
        std::cout<<"    ["<<pair.first<<"] "<<pair.second<<std::endl;
    }
    std::cout<<"role_templates: "<<std::endl;
    for(const auto& pair : conv.role_templates) {
        std::cout<<"    ["<<pair.first<<"] "<<pair.second<<std::endl;
    }
    std::cout<<"messages: " << std::endl;
    for(auto v : conv.messages){
        std::cout<<"    content: "<<v.content.Text()<<std::endl;
        std::cout<<"    role: "<<v.role<<std::endl;
        std::cout<<"    name: ";
        if(v.name.has_value()){ std::cout<<v.name.value(); };
        std::cout << std::endl;
        std::cout<<"    tool_call_id: ";
        if(v.tool_call_id.has_value()){ std::cout << v.tool_call_id.value(); }
        std::cout<<std::endl;
    }
    std::cout<<"seps: " << std::endl;
    for(auto v : conv.seps){
        std::cout<<"    "<<v<<std::endl;
    }
    std::cout<<"role_content_sep: " << conv.role_content_sep<<std::endl;
    std::cout<<"role_empty_sep: " << conv.role_empty_sep<<std::endl;
    std::cout<<"stop_str: " <<std::endl;
    for(auto v : conv.stop_str){
        std::cout<<"    "<<v<<std::endl;
    }
    std::cout<<"stop_token_ids: " <<std::endl;
    for(auto v : conv.stop_token_ids){
        std::cout<<"    "<<v<<std::endl;
    }
}


std::string ReplaceString(std::string s, const std::string& from, const std::string& to) {
  if (from.empty()) return s;  // 빈 패턴은 무한루프 방지
  size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.size(), to);
    pos += to.size();  // 겹침 방지
  }
  return s;
}

std::string GetRolePlaceholder(const std::string& role) {
  if (role == "SYSTEM") return "{system_message}";
  if (role == "USER") return "{user_message}";
  if (role == "ASSISTANT") return "{assistant_message}";
  if (role == "TOOL") return "{tool_message}";
  if (role == "FUNCTION") return "{function_string}";
  return "";
}

inline std::string ToUpper(const std::string& s) {
  std::string r = s;
  std::transform(r.begin(), r.end(), r.begin(), [](unsigned char c){ return std::toupper(c); });
  return r;
}

// TODO: This function assumes messages are string only
std::vector<std::string> _combine_consecutive_messages(std::vector<std::string> messages){
  if(messages.empty()) return std::vector<std::string>();

  std::vector<std::string> combined_messages;
  combined_messages.push_back(messages[0]);
  for(int i = 1; i < messages.size(); i++){ 
    combined_messages.back() = combined_messages.back() + messages[i];
  }

  return combined_messages;
}

std::vector<std::string> ConvertConversationToPrompt(Conversation& conv){
  // - Get the system message.
  std::string system_message_placeholder = "{system_message}";
  std::string system_msg = ReplaceString(conv.system_template, system_message_placeholder, conv.system_message);

  // - Get the message strings.
  std::vector<std::string> message_list;
  std::vector<std::string> separators = conv.seps;

  if(separators.size() == 1){
    separators.push_back(separators[0]);
  }

  if(!system_msg.empty()){
    message_list.push_back(system_msg);
  }

  for(int i = 0; i < conv.messages.size(); i++){
    std::string role = conv.messages[i].role;
    ChatCompletionMessageContent content = conv.messages[i].content;

    bool is_role_valid = false;
    if(!conv.roles.count(role)){
      std::cout << "[ERROR] Role \"" << role << "\" is not a supported role in conversation's roles." << std::endl;
      exit(0);
    }

    std::string separator;
    if(role == "assistant") separator = separators[1];
    else separator = separators[0];

    if(content.IsNull()){
      message_list.push_back(conv.roles[role] + conv.role_empty_sep);
      continue;
    }

    std::string role_prefix;
    if(!conv.add_role_after_system_message && system_msg != "" && i == 0){
      role_prefix.clear();
    }
    else{
      role_prefix = conv.roles[role] + conv.role_content_sep;
    }

    if(content.IsText()){
      std::string msg_ = role_prefix;
      std::string role_placeholder = GetRolePlaceholder(ToUpper(role));
      msg_ += ReplaceString(conv.role_templates[role], role_placeholder, content.Text());
      msg_ += separator;
      message_list.push_back(msg_);

      continue;
    }

    message_list.push_back(role_prefix);
    
    for(auto item : content.Parts()){
      if(item.find("type") == item.end()){
        std::cout << "[ERROR] Content item should have a type field" << std::endl;
        exit(0);
      }

      if(item["type"] == "text"){
        std::string msg_;
        std::string role_placeholder = GetRolePlaceholder(ToUpper(role));
        msg_ = ReplaceString(conv.role_templates[role], role_placeholder, item["text"]);
        message_list.push_back(msg_);
      }
      else if(item["type"] == "image_url"){ // TODO: Support image_url
        std::cout << "[ERROR] image_url is not supported yet." << std::endl;
        exit(0);
      }
      else{
        std::cout << "[ERROR] Unsupported content type: " << item["type"] << std::endl;
        exit(0);
      }
    }

    message_list.push_back(separator);
  }

  std::vector<std::string> prompts = _combine_consecutive_messages(message_list);

  // TODO: Skip processing for Data item
  return prompts;
}

static inline std::vector<int> TokenIds2IntVec(const TokenIds& t) {
  std::vector<int> v;
  v.reserve(t.size());
  for (int i = 0; i < t.size(); ++i) v.push_back(t[i]);
  return v;
}

static inline TokenIds IntVec2TokenIds(std::vector<int> v) {
  return TokenIds(v.begin(), v.end());
}

IntTuple AppendIntTuple(const IntTuple& a, const IntTuple& b) {
  std::vector<int64_t> out;
  out.reserve(a.size() + b.size());
  out.insert(out.end(), a.begin(), a.end());
  out.insert(out.end(), b.begin(), b.end());
  return IntTuple(std::move(out));  // std::vector<int64_t> 생성자를 사용
}

void PrintIntTuple(const IntTuple& t){
  std::cout<<t<<std::endl;
}

void IntTupleToInt64Vector(const IntTuple& int_tuple, std::vector<int64_t>& int_vector){
  int_vector = std::vector<int64_t>(int_tuple.begin(), int_tuple.end());
}

void IntTupleToInt32Vector(const IntTuple& int_tuple, std::vector<int32_t>& int_vector){
  int_vector = std::vector<int32_t>(int_tuple.begin(), int_tuple.end());
}

void StringArrayToStdArryVector(const tvm::ffi::Array<tvm::ffi::String>& string_array, std::vector<std::string>& std_string_vector){
  for(auto str : string_array){
    std_string_vector.push_back(static_cast<std::string>(str));
  }
}


std::string FinishReasonToStr(mlc::llm::json_ffi::FinishReason& finish_reason){
  switch(finish_reason){
    case mlc::llm::json_ffi::FinishReason::stop:
      return "stop";
    case mlc::llm::json_ffi::FinishReason::length:
      return "length";
    case mlc::llm::json_ffi::FinishReason::tool_calls:
      return "tool_calls";
    default:
      return "error";
  }
}

mlc::llm::json_ffi::FinishReason StrToFinishReason(std::string& finish_reason){
  if(finish_reason == "stop")
    return mlc::llm::json_ffi::FinishReason::stop;    
  else if(finish_reason == "length")
    return mlc::llm::json_ffi::FinishReason::length;
  else if(finish_reason == "tool_calls")
    return mlc::llm::json_ffi::FinishReason::tool_calls;
  else
    return mlc::llm::json_ffi::FinishReason::error;
}


} // using namespace mlc
} // using namespace llm
} // using namespace utils