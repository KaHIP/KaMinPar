#include "converter/graph_converter.h"

#include "converter/basic_processors.h"
#include "converter/binary.h"
#include "converter/dac2012.h"
#include "converter/hmetis.h"
#include "converter/kagen.h"
#include "converter/matrix_market.h"
#include "converter/metis.h"
#include "converter/snap.h"
#include "definitions.h"
#include "kaminpar/application/arguments_parser.h"

using namespace kaminpar;
using namespace kaminpar::tool::converter;

GraphConverter create_graph_converter() {
  GraphConverter converter;

  converter.register_reader<MetisReader>("metis");
  converter.register_reader<MatrixMarketReader>("matrixmarket");
  converter.register_reader<KaGenReader>("kagen");
  converter.register_reader<SNAPReader>("snap");
  converter.register_reader<SNAPReader>("parmat");
  converter.register_reader<Dac2012Reader>("dac2012");
  converter.register_reader<BinaryReader>("binary");

  converter.register_writer<HMetisWriter>("hmetis");
  converter.register_writer<MetisWriter>("metis");
  converter.register_writer<BinaryWriter>("binary");

  converter.register_processor<StripNodeWeightsProcessor>("strip-node-weights");
  converter.register_processor<StripEdgeWeightsProcessor>("strip-edge-weights");
  converter.register_processor<StripIsolatedNodesProcessor>("strip-isolated-nodes");
  converter.register_processor<ExtractLargestComponent<ExtractMetric::N>>("extract-largest-cc-by-n");
  converter.register_processor<ExtractLargestComponent<ExtractMetric::M>>("extract-largest-cc-by-m");

  return converter;
}

std::string generate_output_filename(const std::string &input_filename, const std::string &output_format) {
  std::stringstream ss;

  const auto dash = input_filename.find_last_of('/');
  const std::string path = (dash != std::string::npos) ? input_filename.substr(0, dash + 1) : "";
  ss << path;

  const std::string simple_name = (dash != std::string::npos) ? input_filename.substr(dash + 1) : input_filename;
  const auto dot = simple_name.find_last_of('.');
  if (dot != std::string::npos) {
    const std::string ext = simple_name.substr(dot + 1);
    if (ext == output_format) {
      ss << simple_name << ".out";
    } else {
      ss << simple_name.substr(0, dot) << "." << output_format;
    }
  } else {
    ss << simple_name << "." << output_format;
  }

  return ss.str();
}

std::string get_filename_ext(const std::string &filename) {
  const auto dot = filename.find_last_of('.');
  return dot == std::string::npos ? "" : filename.substr(dot + 1);
}

int main(int argc, char *argv[]) {
  GraphConverter converter = create_graph_converter();

  std::string output_filename;
  std::string input_filename;
  std::string input_format;
  std::string output_format;
  std::string processors;

  Arguments args;
  args.positional()
      .argument("Input filename", "Input filename", &input_filename)
      .opt_argument("Output filename", "Output filename", &output_filename);

  auto &graph_format_ops = args.group("Graph format");
  graph_format_ops.argument("i-format", "Input format", &input_format);
  for (const auto &[name, reader] : converter.get_readers()) {
    graph_format_ops.line(std::string("-> ") + name + ": " + reader->description());
  }
  graph_format_ops.argument("o-format", "Output format", &output_format);
  for (const auto &[name, writer] : converter.get_writers()) {
    graph_format_ops.line(std::string("-> ") + name + ": " + writer->description());
  }

  auto &processor_opts = args.group("Graph processor");
  processor_opts.argument("processors", "Comma separated list of graph processors", &processors);
  for (const auto &[name, processor] : converter.get_processors()) {
    processor_opts.line(std::string("-> ") + name + ": " + processor->description());
  }

  args.parse(argc, argv);

  if (input_format.empty()) { input_format = get_filename_ext(input_filename); }
  if (!converter.reader_exists(input_format)) { FATAL_ERROR << "Invalid input format: " << input_format; }
  if (output_format.empty()) { output_format = input_format; }
  if (!converter.writer_exists(output_format)) { FATAL_ERROR << "Invalid output format: " << output_format; }
  if (output_filename.empty()) { output_filename = generate_output_filename(input_filename, output_format); }

  converter.convert(input_format, input_filename, output_format, output_filename, processors);
  return 0;
}
