// Macro that evaluates to true or false depending on whether another macro is
// defined or undefined use DETECT_EXIST(SOME_OTHER_MACRO) to detect whether
// SOME_OTHER_MACRO is defined or undefined
//
// Copied from
// https://stackoverflow.com/questions/41265750/how-to-get-a-boolean-indicating-if-a-macro-is-defined-or-not
#define SECOND_ARG(A, B, ...) B
#define CONCAT2(A, B) A##B
#define DETECT_EXIST_TRUE ~, 1
#define DETECT_EXIST_IMPL(...) SECOND_ARG(__VA_ARGS__)
#define DETECT_EXIST(X) DETECT_EXIST_IMPL(CONCAT2(DETECT_EXIST_TRUE, X), 0, ~)
