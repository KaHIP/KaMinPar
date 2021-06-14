#include "converter/graph_converter.h"

#include "converter/basic_processors.h"
#include "converter/dac2012.h"
#include "converter/hmetis.h"
#include "converter/kagen.h"
#include "converter/matrix_market.h"
#include "converter/metis.h"
#include "converter/snap.h"
#include "definitions.h"

#include <cstdlib>
#include <getopt.h>

using namespace kaminpar::tool::converter;

GraphConverter create_graph_converter() {
  GraphConverter converter;
  converter.register_reader<MetisReader>("metis");
  converter.register_writer<MetisWriter>("metis");
  converter.register_reader<MatrixMarketReader>("matrixmarket");
  converter.register_reader<KaGenReader>("kagen");
  converter.register_writer<HMetisWriter>("hmetis");
  converter.register_reader<SNAPReader>("snap");
  converter.register_reader<SNAPReader>("parmat");
  converter.register_reader<Dac2012Reader>("dac2012");
  converter.register_processor<StripNodeWeightsProcessor>("strip-node-weights");
  converter.register_processor<StripEdgeWeightsProcessor>("strip-edge-weights");
  converter.register_processor<StripIsolatedNodesProcessor>("strip-isolated-nodes");
  converter.register_processor<ExtractLargestCC<true>>("extract-largest-cc-by-n");
  converter.register_processor<ExtractLargestCC<false>>("extract-largest-cc-by-m");
  return converter;
}

void print_help(int, char *argv[]) {
  const GraphConverter converter = create_graph_converter();
  LOG << "Usage: " << argv[0] << " options...";
  LOG;
  LOG << "Mandatory options are:";
  LOG << "\t--if=<filename>          \t\tInput filename.";
  LOG;
  LOG << "Optional options are:";
  LOG << "\t--of=<filename>   \t\tOutput filename. If unspecified, generate output filename from input filename.";
  LOG << "\t--import=<format> \t\tFile format of the input file. Default: metis graph format";
  LOG << "\t--export=<format> \t\tFile format of the output file. Default: metis graph format.";
  LOG << "\t--processors      \t\tIf present, a comma separated list of graph processors.";
  LOG << "\t--no-comment      \t\tDon't add comments to the output file.";
  LOG;
  LOG << "Possible options for --import are:";
  for (const auto &[name, reader] : converter.get_readers()) { LOG << "\t" << name << ": " << reader->description(); }
  LOG;
  LOG << "Possible options for --export are:";
  for (const auto &[name, writer] : converter.get_writers()) { LOG << "\t" << name << ": " << writer->description(); }
  LOG;
  LOG << "Possible options for --processors are:";
  for (const auto &[name, processor] : converter.get_processors()) {
    LOG << "\t" << name << ": " << processor->description();
  }
  LOG;
  LOG << "Examples:";
  LOG << "\tConvert a graph from matrix market format to METIS format:";
  LOG << "\t\t" << argv[0] << " --if=matrix.mtx --of=matrix.graph --import=matrixmarket";
  LOG << "\tRemove node and edge weights from a graph in METIS format:";
  LOG << "\t\t" << argv[0] << " --if=in.graph --of=out.graph --processors=strip-node-weights,strip-edge-weights";
  LOG;
}

static struct option getopt_options[] = {
    {.name = "help", .has_arg = no_argument, .flag = nullptr, .val = 'h'},             //
    {.name = "import", .has_arg = required_argument, .flag = nullptr, .val = 'I'},     //
    {.name = "export", .has_arg = required_argument, .flag = nullptr, .val = 'E'},     //
    {.name = "if", .has_arg = required_argument, .flag = nullptr, .val = 'i'},         //
    {.name = "of", .has_arg = required_argument, .flag = nullptr, .val = 'o'},         //
    {.name = "processors", .has_arg = required_argument, .flag = nullptr, .val = 'P'}, //
    {.name = "no-comment", .has_arg = no_argument, .flag = nullptr, .val = 'C'},       //
    {.name = nullptr, .has_arg = 0, .flag = nullptr, .val = 0}                         //
};

struct Options {
  std::string import_format{};
  std::string export_format{"metis"};
  std::string input_filename{};
  std::string output_filename{};
  std::string processors{};
  bool no_comment{false};
};

Options parse_options(int argc, char *argv[]) {
  Options opts;
  int index = 0;
  int c = 0;
  while ((c = getopt_long(argc, argv, "I:E:i:o:C", getopt_options, &index)) != -1) {
    switch (c) {
      case 'h': print_help(argc, argv); std::exit(0);
      case 'I': opts.import_format = optarg; break;
      case 'E': opts.export_format = optarg; break;
      case 'i': opts.input_filename = optarg; break;
      case 'o': opts.output_filename = optarg; break;
      case 'e': opts.export_format = optarg; break;
      case 'P': opts.processors = optarg; break;
      case 'C': opts.no_comment = true; break;
      case '?': break;
      default: FATAL_ERROR << "Undefined flag: " << static_cast<char>(c); break;
    }
  }

  const GraphConverter converter = create_graph_converter();
  if (opts.import_format.empty()) { opts.import_format = "metis"; }
  if (opts.export_format.empty()) { opts.export_format = "metis"; }
  if (!converter.reader_exists(opts.import_format)) { FATAL_ERROR << "Invalid import format: " << opts.import_format; }
  if (!converter.writer_exists(opts.export_format)) { FATAL_ERROR << "Invalid export format: " << opts.export_format; }
  if (opts.input_filename.empty()) { FATAL_ERROR << "--if must be set and cannot be empty."; }
  if (opts.output_filename.empty()) {
    const auto dash = opts.output_filename.find_last_of('/');
    const std::string simple_name = (dash != std::string::npos) ? opts.input_filename.substr(dash + 1)
                                                                : opts.input_filename;
    const std::string path = (dash != std::string::npos) ? opts.input_filename.substr(0, dash + 1) : "";
    const auto dot = simple_name.find_last_of('.');
    if (dot != std::string::npos) {
      const std::string ext = simple_name.substr(dot + 1);
      if (ext == "graph") {
        opts.output_filename = opts.import_format + ".out";
      } else {
        opts.output_filename = path + simple_name.substr(0, dot) + ".graph";
      }
    } else {
      opts.output_filename = opts.input_filename + ".graph";
    }

    LOG << "Output filename deduced from input filename: " << opts.output_filename;
  }

  return opts;
}

int main(int argc, char *argv[]) {
  std::stringstream comment_ss{};
  comment_ss << "Command='";
  for (int i = 0; i < argc; ++i) { comment_ss << argv[i] << " "; }
  comment_ss << "' Version=" << GIT_COMMIT_HASH;

  const Options opts = parse_options(argc, argv);
  GraphConverter converter = create_graph_converter();
  if (!opts.no_comment) { converter.set_comment(comment_ss.str()); }
  converter.convert(opts.import_format, opts.input_filename, opts.export_format, opts.output_filename, opts.processors);
  return 0;
}
