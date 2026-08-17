// warpsmith - a dependency-free JSON writer.
//
// Benchmark results are the product of this repository, so they are emitted as
// machine-readable JSON that the Python tooling turns into tables and charts.
// A 100-line writer avoids adding a third-party dependency to a CUDA build.
#pragma once

#include <cmath>
#include <cstdio>
#include <ostream>
#include <string>
#include <vector>

namespace ws {

class JsonWriter {
 public:
  explicit JsonWriter(std::ostream& os) : os_(os) {}

  void begin_object() {
    comma();
    os_ << "{";
    stack_.push_back(false);
    ++depth_;
  }
  void end_object() {
    --depth_;
    stack_.pop_back();
    newline();
    os_ << "}";
    mark();
  }
  void begin_array() {
    comma();
    os_ << "[";
    stack_.push_back(false);
    ++depth_;
  }
  void end_array() {
    --depth_;
    stack_.pop_back();
    newline();
    os_ << "]";
    mark();
  }

  void key(const std::string& k) {
    comma();
    os_ << "\"" << escape(k) << "\": ";
    suppress_comma_ = true;
  }

  void value(const std::string& v) {
    comma();
    os_ << "\"" << escape(v) << "\"";
    mark();
  }
  void value(const char* v) { value(std::string(v)); }
  void value(bool v) {
    comma();
    os_ << (v ? "true" : "false");
    mark();
  }
  void value(long long v) {
    comma();
    os_ << v;
    mark();
  }
  void value(int v) { value(static_cast<long long>(v)); }
  void value(std::size_t v) { value(static_cast<long long>(v)); }
  void value(double v) {
    comma();
    if (std::isnan(v) || std::isinf(v)) {
      os_ << "null";
    } else {
      char buf[64];
      std::snprintf(buf, sizeof(buf), "%.6g", v);
      os_ << buf;
    }
    mark();
  }

  template <typename T>
  void field(const std::string& k, const T& v) {
    key(k);
    value(v);
  }

 private:
  void comma() {
    if (suppress_comma_) {
      suppress_comma_ = false;
      return;
    }
    if (!stack_.empty() && stack_.back()) os_ << ",";
    newline();
  }
  void newline() {
    if (depth_ > 0 || !stack_.empty()) {
      os_ << "\n";
      for (int i = 0; i < depth_; ++i) os_ << "  ";
    }
  }
  void mark() {
    if (!stack_.empty()) stack_.back() = true;
  }
  static std::string escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
      switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\t': out += "\\t"; break;
        case '\r': out += "\\r"; break;
        default: out += c;
      }
    }
    return out;
  }

  std::ostream& os_;
  std::vector<bool> stack_;
  int depth_ = 0;
  bool suppress_comma_ = false;
};

}  // namespace ws
