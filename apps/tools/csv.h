#pragma once

#include "definitions.h"

#include <fstream>
#include <numeric>
#include <ranges>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace kaminpar::tool {
using namespace std::string_view_literals;

struct Csv {
  std::vector<std::string> columns{};
  std::vector<std::vector<std::string>> rows{};
};

/**
 * Finds the indices of named CSV columns and returns them. Column names are case-insensitive. Crashes the program with
 * an error message if some column doesn't exist.
 * @param csv Some CSV file with header.
 * @param columns Some column names.
 * @return The indices of the columns, in the same order as the column names were given.
 */
std::vector<std::size_t> find_column_indices(const Csv &csv, const std::vector<std::string> &columns);

/**
 * Concatenates the given CSV objects keeping only the specified columns. If some specified column does not exist in
 * some given CSV object, the program crashes with a descriptive error message. Columns in the given CSV objects do not
 * have to be in the same order.
 * @param common_columns List of columns that should be kept.
 * @param csvs List of CSV objects that should be concatenated.
 * @return Concatenated CSV object.
 */
Csv merge_csvs(const std::vector<std::string> &common_columns, const std::vector<Csv> &csvs);

/**
 * Performs the specified aggregate function the given dataset. Available aggregate functions are:
 * count, avg, max, min, sum
 * Crashes the program when given an unknown aggregate function.
 *
 * @param function Aggregate function name.
 * @param values Dataset set to be aggregated.
 * @return Aggregated value.
 */
double aggregate(const std::string &function, const std::vector<double> &values);

/**
 * Performs a GROUP BY with aggregate operation on a CSV object. First, all rows are grouped by their content in the
 * given group_by columns. Then, the values of remaining values are aggregated using the selected aggregate functions.
 * **The values of the remaining columns must be numeric!**
 * @param csv CSV object to operate on.
 * @param group_by List of columns used as group key.
 * @param aggregates Aggregate functions for the remaining columns: the first element of the pair is a column name and
 * the second element is an aggregate function name.
 * @param expected_rows_per_aggregate If greater than 0, the expected number of elements per group. Prints a warning
 * for each group with more or less elements.
 * @return The resulting grouped and aggregate CSV object.
 */
Csv aggregate(const Csv &csv, const std::vector<std::string> &group_by,
              const std::vector<std::pair<std::string, std::string>> &aggregates,
              std::size_t expected_rows_per_aggregate = 0);

/**
 * Parses a CSV object from a CSV file. Crashes the program if something is wrong with the CSV file.
 * @param filename Filename of the CSV file.
 * @param del Delimited between columns.
 * @return Loaded CSV object.
 */
Csv load_csv(const std::string &filename, bool filter_failed = false, char del = ',', const bool quiet = false);

void print_row(const std::vector<std::string> &row, bool colorize = true, char del = ',');
void print_csv(const Csv &csv, bool colorize = true, char del = ',');
} // namespace kaminpar::tool