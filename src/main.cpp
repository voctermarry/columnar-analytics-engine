#include <iostream>
#include <string_view>

#include "columnar_analytics/version.hpp"

namespace {

void print_usage(std::ostream& out, std::string_view program) {
  out << "用法: " << program << " <命令>\n\n"
      << "命令:\n"
      << "  version    打印当前版本号\n"
      << "  help       打印本用法说明\n";
}

}  // namespace

int main(int argc, char** argv) {
  const std::string_view program = argc > 0 ? argv[0] : "columnar-analytics-engine";
  if (argc < 2) {
    print_usage(std::cout, program);
    return 0;
  }
  const std::string_view command = argv[1];
  if (command == "version") {
    std::cout << columnar_analytics::version() << '\n';
    return 0;
  }
  if (command == "help" || command == "--help" || command == "-h") {
    print_usage(std::cout, program);
    return 0;
  }
  std::cerr << "未知命令: " << command << '\n';
  print_usage(std::cerr, program);
  return 2;
}
