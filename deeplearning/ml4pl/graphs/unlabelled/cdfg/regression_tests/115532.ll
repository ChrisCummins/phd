; ModuleID = '-'
source_filename = "-"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%struct.mpc_err_t = type { %struct.mpc_state_t, i32, i8*, i8*, i8**, i8 }
%struct.mpc_state_t = type { i64, i64, i64 }
%struct.__va_list_tag = type { i32, i32, i8*, i8* }
%struct.mpc_input_t = type { i32, i8*, %struct.mpc_state_t, i8*, i8*, %struct._IO_FILE*, i32, i32, i32, i32, %struct.mpc_state_t*, i8*, i8, i64, [512 x i8], [512 x %struct.mpc_mem_t] }
%struct.mpc_mem_t = type { [64 x i8] }
%struct.mpc_parser_t = type { i8, i8*, i8, %union.mpc_pdata_t }
%union.mpc_pdata_t = type { %struct.mpc_pdata_repeat_t }
%struct.mpc_pdata_repeat_t = type { i32, i8* (i32, i8**)*, %struct.mpc_parser_t*, void (i8*)* }
%union.mpc_result_t = type { %struct.mpc_err_t* }
%struct.mpc_pdata_single_t = type { i8 }
%struct.mpc_pdata_range_t = type { i8, i8 }
%struct.mpc_pdata_string_t = type { i8* }
%struct.mpc_pdata_satisfy_t = type { i32 (i8)* }
%struct.mpc_pdata_anchor_t = type { i32 (i8, i8)* }
%struct.mpc_pdata_fail_t = type { i8* }
%struct.mpc_pdata_lift_t = type { i8* ()*, i8* }
%struct.mpc_pdata_apply_t = type { %struct.mpc_parser_t*, i8* (i8*)* }
%struct.mpc_pdata_apply_to_t = type { %struct.mpc_parser_t*, i8* (i8*, i8*)*, i8* }
%struct.mpc_pdata_expect_t = type { %struct.mpc_parser_t*, i8* }
%struct.mpc_pdata_predict_t = type { %struct.mpc_parser_t* }
%struct.mpc_pdata_not_t = type { %struct.mpc_parser_t*, void (i8*)*, i8* ()* }
%struct.mpc_pdata_or_t = type { i32, %struct.mpc_parser_t** }
%struct.mpc_pdata_and_t = type { i32, i8* (i32, i8**)*, %struct.mpc_parser_t**, void (i8*)** }
%struct.mpc_ast_t = type { i8*, i8*, %struct.mpc_state_t, i32, %struct.mpc_ast_t** }
%struct.mpca_grammar_st_t = type { [1 x %struct.__va_list_tag]*, i32, %struct.mpc_parser_t**, i32 }
%struct.mpca_stmt_t = type { i8*, i8*, %struct.mpc_parser_t* }

@stdout = external global %struct._IO_FILE*, align 8
@.str = private unnamed_addr constant [3 x i8] c"%s\00", align 1
@.str.1 = private unnamed_addr constant [15 x i8] c"%s: error: %s\0A\00", align 1
@.str.2 = private unnamed_addr constant [27 x i8] c"%s:%i:%i: error: expected \00", align 1
@.str.3 = private unnamed_addr constant [24 x i8] c"ERROR: NOTHING EXPECTED\00", align 1
@.str.4 = private unnamed_addr constant [5 x i8] c"%s, \00", align 1
@.str.5 = private unnamed_addr constant [9 x i8] c"%s or %s\00", align 1
@.str.6 = private unnamed_addr constant [5 x i8] c" at \00", align 1
@.str.7 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@.str.8 = private unnamed_addr constant [14 x i8] c"Unknown Error\00", align 1
@.str.9 = private unnamed_addr constant [3 x i8] c"rb\00", align 1
@.str.10 = private unnamed_addr constant [21 x i8] c"Unable to open file!\00", align 1
@.str.11 = private unnamed_addr constant [40 x i8] c"Attempt to assign to Unretained Parser!\00", align 1
@.str.12 = private unnamed_addr constant [7 x i8] c"anchor\00", align 1
@.str.13 = private unnamed_addr constant [14 x i8] c"any character\00", align 1
@.str.14 = private unnamed_addr constant [5 x i8] c"'%c'\00", align 1
@.str.15 = private unnamed_addr constant [32 x i8] c"character between '%c' and '%c'\00", align 1
@.str.16 = private unnamed_addr constant [12 x i8] c"one of '%s'\00", align 1
@.str.17 = private unnamed_addr constant [13 x i8] c"none of '%s'\00", align 1
@.str.18 = private unnamed_addr constant [33 x i8] c"character satisfying function %p\00", align 1
@.str.19 = private unnamed_addr constant [5 x i8] c"\22%s\22\00", align 1
@.str.20 = private unnamed_addr constant [15 x i8] c"start of input\00", align 1
@.str.21 = private unnamed_addr constant [13 x i8] c"end of input\00", align 1
@.str.22 = private unnamed_addr constant [9 x i8] c"boundary\00", align 1
@.str.23 = private unnamed_addr constant [7 x i8] c" \0C\0A\0D\09\0B\00", align 1
@.str.24 = private unnamed_addr constant [11 x i8] c"whitespace\00", align 1
@.str.25 = private unnamed_addr constant [7 x i8] c"spaces\00", align 1
@.str.26 = private unnamed_addr constant [8 x i8] c"newline\00", align 1
@.str.27 = private unnamed_addr constant [4 x i8] c"tab\00", align 1
@.str.28 = private unnamed_addr constant [11 x i8] c"0123456789\00", align 1
@.str.29 = private unnamed_addr constant [6 x i8] c"digit\00", align 1
@.str.30 = private unnamed_addr constant [23 x i8] c"0123456789ABCDEFabcdef\00", align 1
@.str.31 = private unnamed_addr constant [10 x i8] c"hex digit\00", align 1
@.str.32 = private unnamed_addr constant [9 x i8] c"01234567\00", align 1
@.str.33 = private unnamed_addr constant [10 x i8] c"oct digit\00", align 1
@.str.34 = private unnamed_addr constant [7 x i8] c"digits\00", align 1
@.str.35 = private unnamed_addr constant [11 x i8] c"hex digits\00", align 1
@.str.36 = private unnamed_addr constant [11 x i8] c"oct digits\00", align 1
@.str.37 = private unnamed_addr constant [27 x i8] c"abcdefghijklmnopqrstuvwxyz\00", align 1
@.str.38 = private unnamed_addr constant [17 x i8] c"lowercase letter\00", align 1
@.str.39 = private unnamed_addr constant [27 x i8] c"ABCDEFGHIJKLMNOPQRSTUVWXYZ\00", align 1
@.str.40 = private unnamed_addr constant [17 x i8] c"uppercase letter\00", align 1
@.str.41 = private unnamed_addr constant [53 x i8] c"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\00", align 1
@.str.42 = private unnamed_addr constant [7 x i8] c"letter\00", align 1
@.str.43 = private unnamed_addr constant [11 x i8] c"underscore\00", align 1
@.str.44 = private unnamed_addr constant [13 x i8] c"alphanumeric\00", align 1
@.str.45 = private unnamed_addr constant [8 x i8] c"integer\00", align 1
@.str.46 = private unnamed_addr constant [12 x i8] c"hexadecimal\00", align 1
@.str.47 = private unnamed_addr constant [12 x i8] c"octadecimal\00", align 1
@.str.48 = private unnamed_addr constant [7 x i8] c"number\00", align 1
@.str.49 = private unnamed_addr constant [3 x i8] c"+-\00", align 1
@.str.50 = private unnamed_addr constant [3 x i8] c"eE\00", align 1
@.str.51 = private unnamed_addr constant [5 x i8] c"real\00", align 1
@.str.52 = private unnamed_addr constant [6 x i8] c"float\00", align 1
@.str.53 = private unnamed_addr constant [2 x i8] c"'\00", align 1
@.str.54 = private unnamed_addr constant [5 x i8] c"char\00", align 1
@.str.55 = private unnamed_addr constant [2 x i8] c"\22\00", align 1
@.str.56 = private unnamed_addr constant [7 x i8] c"string\00", align 1
@.str.57 = private unnamed_addr constant [2 x i8] c"/\00", align 1
@.str.58 = private unnamed_addr constant [6 x i8] c"regex\00", align 1
@.str.59 = private unnamed_addr constant [2 x i8] c"(\00", align 1
@.str.60 = private unnamed_addr constant [2 x i8] c")\00", align 1
@.str.61 = private unnamed_addr constant [2 x i8] c"<\00", align 1
@.str.62 = private unnamed_addr constant [2 x i8] c">\00", align 1
@.str.63 = private unnamed_addr constant [2 x i8] c"{\00", align 1
@.str.64 = private unnamed_addr constant [2 x i8] c"}\00", align 1
@.str.65 = private unnamed_addr constant [2 x i8] c"[\00", align 1
@.str.66 = private unnamed_addr constant [2 x i8] c"]\00", align 1
@.str.67 = private unnamed_addr constant [5 x i8] c"term\00", align 1
@.str.68 = private unnamed_addr constant [7 x i8] c"factor\00", align 1
@.str.69 = private unnamed_addr constant [5 x i8] c"base\00", align 1
@.str.70 = private unnamed_addr constant [6 x i8] c"range\00", align 1
@.str.71 = private unnamed_addr constant [3 x i8] c")|\00", align 1
@.str.72 = private unnamed_addr constant [18 x i8] c"<mpc_re_compiler>\00", align 1
@.str.73 = private unnamed_addr constant [18 x i8] c"Invalid Regex: %s\00", align 1
@mpc_escape_input_c = internal constant [11 x i8] c"\07\08\0C\0A\0D\09\0B\5C'\22\00", align 1
@mpc_escape_output_c = internal global [12 x i8*] [i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.116, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.117, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.118, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.119, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.120, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.121, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.122, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.123, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.124, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.125, i32 0, i32 0), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.126, i32 0, i32 0), i8* null], align 16
@mpc_escape_input_raw_re = internal constant [1 x i8] c"/", align 1
@mpc_escape_output_raw_re = internal global [2 x i8*] [i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.127, i32 0, i32 0), i8* null], align 16
@mpc_escape_input_raw_cstr = internal constant [1 x i8] c"\22", align 1
@mpc_escape_output_raw_cstr = internal global [2 x i8*] [i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.125, i32 0, i32 0), i8* null], align 16
@mpc_escape_input_raw_cchar = internal constant [1 x i8] c"'", align 1
@mpc_escape_output_raw_cchar = internal global [2 x i8*] [i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.124, i32 0, i32 0), i8* null], align 16
@.str.74 = private unnamed_addr constant [2 x i8] c"*\00", align 1
@.str.75 = private unnamed_addr constant [2 x i8] c"%\00", align 1
@.str.76 = private unnamed_addr constant [2 x i8] c"+\00", align 1
@.str.77 = private unnamed_addr constant [2 x i8] c"-\00", align 1
@.str.78 = private unnamed_addr constant [7 x i8] c"<test>\00", align 1
@.str.79 = private unnamed_addr constant [5 x i8] c"Got \00", align 1
@.str.80 = private unnamed_addr constant [10 x i8] c"Expected \00", align 1
@.str.81 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@.str.82 = private unnamed_addr constant [2 x i8] c"|\00", align 1
@.str.83 = private unnamed_addr constant [14 x i8] c"grammar_total\00", align 1
@.str.84 = private unnamed_addr constant [8 x i8] c"grammar\00", align 1
@.str.85 = private unnamed_addr constant [2 x i8] c"?\00", align 1
@.str.86 = private unnamed_addr constant [2 x i8] c"!\00", align 1
@.str.87 = private unnamed_addr constant [23 x i8] c"<mpc_grammar_compiler>\00", align 1
@.str.88 = private unnamed_addr constant [20 x i8] c"Invalid Grammar: %s\00", align 1
@.str.89 = private unnamed_addr constant [17 x i8] c"<mpca_lang_file>\00", align 1
@.str.90 = private unnamed_addr constant [17 x i8] c"<mpca_lang_pipe>\00", align 1
@.str.91 = private unnamed_addr constant [12 x i8] c"<mpca_lang>\00", align 1
@.str.92 = private unnamed_addr constant [7 x i8] c"Stats\0A\00", align 1
@.str.93 = private unnamed_addr constant [7 x i8] c"=====\0A\00", align 1
@.str.94 = private unnamed_addr constant [16 x i8] c"Node Count: %i\0A\00", align 1
@char_unescape_buffer = internal global [4 x i8] zeroinitializer, align 1
@.str.95 = private unnamed_addr constant [5 x i8] c"bell\00", align 1
@.str.96 = private unnamed_addr constant [10 x i8] c"backspace\00", align 1
@.str.97 = private unnamed_addr constant [9 x i8] c"formfeed\00", align 1
@.str.98 = private unnamed_addr constant [16 x i8] c"carriage return\00", align 1
@.str.99 = private unnamed_addr constant [13 x i8] c"vertical tab\00", align 1
@.str.100 = private unnamed_addr constant [6 x i8] c"space\00", align 1
@.str.101 = private unnamed_addr constant [18 x i8] c"Parser Undefined!\00", align 1
@.str.102 = private unnamed_addr constant [9 x i8] c"opposite\00", align 1
@.str.103 = private unnamed_addr constant [24 x i8] c"Unknown Parser Type Id!\00", align 1
@.str.104 = private unnamed_addr constant [16 x i8] c"one or more of \00", align 1
@.str.105 = private unnamed_addr constant [3 x i8] c", \00", align 1
@.str.106 = private unnamed_addr constant [5 x i8] c" or \00", align 1
@.str.107 = private unnamed_addr constant [7 x i8] c"%i of \00", align 1
@.str.108 = private unnamed_addr constant [64 x i8] c"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_\00", align 1
@.str.109 = private unnamed_addr constant [31 x i8] c"Invalid Regex Range Expression\00", align 1
@.str.110 = private unnamed_addr constant [2 x i8] c"\07\00", align 1
@.str.111 = private unnamed_addr constant [2 x i8] c"\0C\00", align 1
@.str.112 = private unnamed_addr constant [2 x i8] c"\0D\00", align 1
@.str.113 = private unnamed_addr constant [2 x i8] c"\09\00", align 1
@.str.114 = private unnamed_addr constant [2 x i8] c"\0B\00", align 1
@.str.115 = private unnamed_addr constant [2 x i8] c"\08\00", align 1
@.str.116 = private unnamed_addr constant [3 x i8] c"\5Ca\00", align 1
@.str.117 = private unnamed_addr constant [3 x i8] c"\5Cb\00", align 1
@.str.118 = private unnamed_addr constant [3 x i8] c"\5Cf\00", align 1
@.str.119 = private unnamed_addr constant [3 x i8] c"\5Cn\00", align 1
@.str.120 = private unnamed_addr constant [3 x i8] c"\5Cr\00", align 1
@.str.121 = private unnamed_addr constant [3 x i8] c"\5Ct\00", align 1
@.str.122 = private unnamed_addr constant [3 x i8] c"\5Cv\00", align 1
@.str.123 = private unnamed_addr constant [3 x i8] c"\5C\5C\00", align 1
@.str.124 = private unnamed_addr constant [3 x i8] c"\5C'\00", align 1
@.str.125 = private unnamed_addr constant [3 x i8] c"\5C\22\00", align 1
@.str.126 = private unnamed_addr constant [3 x i8] c"\5C0\00", align 1
@.str.127 = private unnamed_addr constant [3 x i8] c"\5C/\00", align 1
@.str.128 = private unnamed_addr constant [5 x i8] c"<%s>\00", align 1
@.str.129 = private unnamed_addr constant [7 x i8] c"<anon>\00", align 1
@.str.130 = private unnamed_addr constant [4 x i8] c"<?>\00", align 1
@.str.131 = private unnamed_addr constant [4 x i8] c"<:>\00", align 1
@.str.132 = private unnamed_addr constant [4 x i8] c"<!>\00", align 1
@.str.133 = private unnamed_addr constant [4 x i8] c"<#>\00", align 1
@.str.134 = private unnamed_addr constant [4 x i8] c"<S>\00", align 1
@.str.135 = private unnamed_addr constant [4 x i8] c"<@>\00", align 1
@.str.136 = private unnamed_addr constant [4 x i8] c"<.>\00", align 1
@.str.137 = private unnamed_addr constant [4 x i8] c"<f>\00", align 1
@.str.138 = private unnamed_addr constant [5 x i8] c"'%s'\00", align 1
@.str.139 = private unnamed_addr constant [8 x i8] c"[%s-%s]\00", align 1
@.str.140 = private unnamed_addr constant [5 x i8] c"[%s]\00", align 1
@.str.141 = private unnamed_addr constant [6 x i8] c"[^%s]\00", align 1
@.str.142 = private unnamed_addr constant [5 x i8] c"{%i}\00", align 1
@.str.143 = private unnamed_addr constant [4 x i8] c" | \00", align 1
@.str.144 = private unnamed_addr constant [2 x i8] c" \00", align 1
@.str.145 = private unnamed_addr constant [6 x i8] c"NULL\0A\00", align 1
@.str.146 = private unnamed_addr constant [3 x i8] c"  \00", align 1
@.str.147 = private unnamed_addr constant [17 x i8] c"%s:%lu:%lu '%s'\0A\00", align 1
@.str.148 = private unnamed_addr constant [5 x i8] c"%s \0A\00", align 1
@.str.149 = private unnamed_addr constant [52 x i8] c"No Parser in position %i! Only supplied %i Parsers!\00", align 1
@.str.150 = private unnamed_addr constant [21 x i8] c"Unknown Parser '%s'!\00", align 1
@.str.151 = private unnamed_addr constant [5 x i8] c"lang\00", align 1
@.str.152 = private unnamed_addr constant [5 x i8] c"stmt\00", align 1
@.str.153 = private unnamed_addr constant [2 x i8] c":\00", align 1
@.str.154 = private unnamed_addr constant [2 x i8] c";\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define void @mpc_err_delete(%struct.mpc_err_t*) #0 {
  %2 = alloca %struct.mpc_err_t*, align 8
  %3 = alloca i32, align 4
  store %struct.mpc_err_t* %0, %struct.mpc_err_t** %2, align 8
  store i32 0, i32* %3, align 4
  br label %4

; <label>:4:                                      ; preds = %18, %1
  %5 = load i32, i32* %3, align 4
  %6 = load %struct.mpc_err_t*, %struct.mpc_err_t** %2, align 8
  %7 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %6, i32 0, i32 1
  %8 = load i32, i32* %7, align 8
  %9 = icmp slt i32 %5, %8
  br i1 %9, label %10, label %21

; <label>:10:                                     ; preds = %4
  %11 = load %struct.mpc_err_t*, %struct.mpc_err_t** %2, align 8
  %12 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %11, i32 0, i32 4
  %13 = load i8**, i8*** %12, align 8
  %14 = load i32, i32* %3, align 4
  %15 = sext i32 %14 to i64
  %16 = getelementptr inbounds i8*, i8** %13, i64 %15
  %17 = load i8*, i8** %16, align 8
  call void @free(i8* %17) #5
  br label %18

; <label>:18:                                     ; preds = %10
  %19 = load i32, i32* %3, align 4
  %20 = add nsw i32 %19, 1
  store i32 %20, i32* %3, align 4
  br label %4

; <label>:21:                                     ; preds = %4
  %22 = load %struct.mpc_err_t*, %struct.mpc_err_t** %2, align 8
  %23 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %22, i32 0, i32 4
  %24 = load i8**, i8*** %23, align 8
  %25 = bitcast i8** %24 to i8*
  call void @free(i8* %25) #5
  %26 = load %struct.mpc_err_t*, %struct.mpc_err_t** %2, align 8
  %27 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %26, i32 0, i32 2
  %28 = load i8*, i8** %27, align 8
  call void @free(i8* %28) #5
  %29 = load %struct.mpc_err_t*, %struct.mpc_err_t** %2, align 8
  %30 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %29, i32 0, i32 3
  %31 = load i8*, i8** %30, align 8
  call void @free(i8* %31) #5
  %32 = load %struct.mpc_err_t*, %struct.mpc_err_t** %2, align 8
  %33 = bitcast %struct.mpc_err_t* %32 to i8*
  call void @free(i8* %33) #5
  ret void
}

; Function Attrs: nounwind
declare void @free(i8*) #1

; Function Attrs: noinline nounwind optnone uwtable
define void @mpc_err_print(%struct.mpc_err_t*) #0 {
  %2 = alloca %struct.mpc_err_t*, align 8
  store %struct.mpc_err_t* %0, %struct.mpc_err_t** %2, align 8
  %3 = load %struct.mpc_err_t*, %struct.mpc_err_t** %2, align 8
  %4 = load %struct._IO_FILE*, %struct._IO_FILE** @stdout, align 8
  call void @mpc_err_print_to(%struct.mpc_err_t* %3, %struct._IO_FILE* %4)
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define void @mpc_err_print_to(%struct.mpc_err_t*, %struct._IO_FILE*) #0 {
  %3 = alloca %struct.mpc_err_t*, align 8
  %4 = alloca %struct._IO_FILE*, align 8
  %5 = alloca i8*, align 8
  store %struct.mpc_err_t* %0, %struct.mpc_err_t** %3, align 8
  store %struct._IO_FILE* %1, %struct._IO_FILE** %4, align 8
  %6 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  %7 = call i8* @mpc_err_string(%struct.mpc_err_t* %6)
  store i8* %7, i8** %5, align 8
  %8 = load %struct._IO_FILE*, %struct._IO_FILE** %4, align 8
  %9 = load i8*, i8** %5, align 8
  %10 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %8, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i8* %9)
  %11 = load i8*, i8** %5, align 8
  call void @free(i8* %11) #5
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpc_err_string(%struct.mpc_err_t*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca %struct.mpc_err_t*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i8*, align 8
  store %struct.mpc_err_t* %0, %struct.mpc_err_t** %3, align 8
  store i32 0, i32* %5, align 4
  store i32 1023, i32* %6, align 4
  %8 = call noalias i8* @calloc(i64 1, i64 1024) #5
  store i8* %8, i8** %7, align 8
  %9 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  %10 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %9, i32 0, i32 3
  %11 = load i8*, i8** %10, align 8
  %12 = icmp ne i8* %11, null
  br i1 %12, label %13, label %22

; <label>:13:                                     ; preds = %1
  %14 = load i8*, i8** %7, align 8
  %15 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  %16 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %15, i32 0, i32 2
  %17 = load i8*, i8** %16, align 8
  %18 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  %19 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %18, i32 0, i32 3
  %20 = load i8*, i8** %19, align 8
  call void (i8*, i32*, i32*, i8*, ...) @mpc_err_string_cat(i8* %14, i32* %5, i32* %6, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.1, i32 0, i32 0), i8* %17, i8* %20)
  %21 = load i8*, i8** %7, align 8
  store i8* %21, i8** %2, align 8
  br label %115

; <label>:22:                                     ; preds = %1
  %23 = load i8*, i8** %7, align 8
  %24 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  %25 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %24, i32 0, i32 2
  %26 = load i8*, i8** %25, align 8
  %27 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  %28 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %27, i32 0, i32 0
  %29 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %28, i32 0, i32 1
  %30 = load i64, i64* %29, align 8
  %31 = add nsw i64 %30, 1
  %32 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  %33 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %32, i32 0, i32 0
  %34 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %33, i32 0, i32 2
  %35 = load i64, i64* %34, align 8
  %36 = add nsw i64 %35, 1
  call void (i8*, i32*, i32*, i8*, ...) @mpc_err_string_cat(i8* %23, i32* %5, i32* %6, i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.2, i32 0, i32 0), i8* %26, i64 %31, i64 %36)
  %37 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  %38 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %37, i32 0, i32 1
  %39 = load i32, i32* %38, align 8
  %40 = icmp eq i32 %39, 0
  br i1 %40, label %41, label %43

; <label>:41:                                     ; preds = %22
  %42 = load i8*, i8** %7, align 8
  call void (i8*, i32*, i32*, i8*, ...) @mpc_err_string_cat(i8* %42, i32* %5, i32* %6, i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str.3, i32 0, i32 0))
  br label %43

; <label>:43:                                     ; preds = %41, %22
  %44 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  %45 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %44, i32 0, i32 1
  %46 = load i32, i32* %45, align 8
  %47 = icmp eq i32 %46, 1
  br i1 %47, label %48, label %55

; <label>:48:                                     ; preds = %43
  %49 = load i8*, i8** %7, align 8
  %50 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  %51 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %50, i32 0, i32 4
  %52 = load i8**, i8*** %51, align 8
  %53 = getelementptr inbounds i8*, i8** %52, i64 0
  %54 = load i8*, i8** %53, align 8
  call void (i8*, i32*, i32*, i8*, ...) @mpc_err_string_cat(i8* %49, i32* %5, i32* %6, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i8* %54)
  br label %55

; <label>:55:                                     ; preds = %48, %43
  %56 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  %57 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %56, i32 0, i32 1
  %58 = load i32, i32* %57, align 8
  %59 = icmp sge i32 %58, 2
  br i1 %59, label %60, label %102

; <label>:60:                                     ; preds = %55
  store i32 0, i32* %4, align 4
  br label %61

; <label>:61:                                     ; preds = %77, %60
  %62 = load i32, i32* %4, align 4
  %63 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  %64 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %63, i32 0, i32 1
  %65 = load i32, i32* %64, align 8
  %66 = sub nsw i32 %65, 2
  %67 = icmp slt i32 %62, %66
  br i1 %67, label %68, label %80

; <label>:68:                                     ; preds = %61
  %69 = load i8*, i8** %7, align 8
  %70 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  %71 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %70, i32 0, i32 4
  %72 = load i8**, i8*** %71, align 8
  %73 = load i32, i32* %4, align 4
  %74 = sext i32 %73 to i64
  %75 = getelementptr inbounds i8*, i8** %72, i64 %74
  %76 = load i8*, i8** %75, align 8
  call void (i8*, i32*, i32*, i8*, ...) @mpc_err_string_cat(i8* %69, i32* %5, i32* %6, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.4, i32 0, i32 0), i8* %76)
  br label %77

; <label>:77:                                     ; preds = %68
  %78 = load i32, i32* %4, align 4
  %79 = add nsw i32 %78, 1
  store i32 %79, i32* %4, align 4
  br label %61

; <label>:80:                                     ; preds = %61
  %81 = load i8*, i8** %7, align 8
  %82 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  %83 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %82, i32 0, i32 4
  %84 = load i8**, i8*** %83, align 8
  %85 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  %86 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %85, i32 0, i32 1
  %87 = load i32, i32* %86, align 8
  %88 = sub nsw i32 %87, 2
  %89 = sext i32 %88 to i64
  %90 = getelementptr inbounds i8*, i8** %84, i64 %89
  %91 = load i8*, i8** %90, align 8
  %92 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  %93 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %92, i32 0, i32 4
  %94 = load i8**, i8*** %93, align 8
  %95 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  %96 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %95, i32 0, i32 1
  %97 = load i32, i32* %96, align 8
  %98 = sub nsw i32 %97, 1
  %99 = sext i32 %98 to i64
  %100 = getelementptr inbounds i8*, i8** %94, i64 %99
  %101 = load i8*, i8** %100, align 8
  call void (i8*, i32*, i32*, i8*, ...) @mpc_err_string_cat(i8* %81, i32* %5, i32* %6, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.5, i32 0, i32 0), i8* %91, i8* %101)
  br label %102

; <label>:102:                                    ; preds = %80, %55
  %103 = load i8*, i8** %7, align 8
  call void (i8*, i32*, i32*, i8*, ...) @mpc_err_string_cat(i8* %103, i32* %5, i32* %6, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.6, i32 0, i32 0))
  %104 = load i8*, i8** %7, align 8
  %105 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  %106 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %105, i32 0, i32 5
  %107 = load i8, i8* %106, align 8
  %108 = call i8* @mpc_err_char_unescape(i8 signext %107)
  call void (i8*, i32*, i32*, i8*, ...) @mpc_err_string_cat(i8* %104, i32* %5, i32* %6, i8* %108)
  %109 = load i8*, i8** %7, align 8
  call void (i8*, i32*, i32*, i8*, ...) @mpc_err_string_cat(i8* %109, i32* %5, i32* %6, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.7, i32 0, i32 0))
  %110 = load i8*, i8** %7, align 8
  %111 = load i8*, i8** %7, align 8
  %112 = call i64 @strlen(i8* %111) #7
  %113 = add i64 %112, 1
  %114 = call i8* @realloc(i8* %110, i64 %113) #5
  store i8* %114, i8** %2, align 8
  br label %115

; <label>:115:                                    ; preds = %102, %13
  %116 = load i8*, i8** %2, align 8
  ret i8* %116
}

declare i32 @fprintf(%struct._IO_FILE*, i8*, ...) #2

; Function Attrs: nounwind
declare noalias i8* @calloc(i64, i64) #1

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_err_string_cat(i8*, i32*, i32*, i8*, ...) #0 {
  %5 = alloca i8*, align 8
  %6 = alloca i32*, align 8
  %7 = alloca i32*, align 8
  %8 = alloca i8*, align 8
  %9 = alloca i32, align 4
  %10 = alloca [1 x %struct.__va_list_tag], align 16
  store i8* %0, i8** %5, align 8
  store i32* %1, i32** %6, align 8
  store i32* %2, i32** %7, align 8
  store i8* %3, i8** %8, align 8
  %11 = load i32*, i32** %7, align 8
  %12 = load i32, i32* %11, align 4
  %13 = load i32*, i32** %6, align 8
  %14 = load i32, i32* %13, align 4
  %15 = sub nsw i32 %12, %14
  store i32 %15, i32* %9, align 4
  %16 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %10, i32 0, i32 0
  %17 = bitcast %struct.__va_list_tag* %16 to i8*
  call void @llvm.va_start(i8* %17)
  %18 = load i32, i32* %9, align 4
  %19 = icmp slt i32 %18, 0
  br i1 %19, label %20, label %21

; <label>:20:                                     ; preds = %4
  store i32 0, i32* %9, align 4
  br label %21

; <label>:21:                                     ; preds = %20, %4
  %22 = load i8*, i8** %5, align 8
  %23 = load i32*, i32** %6, align 8
  %24 = load i32, i32* %23, align 4
  %25 = sext i32 %24 to i64
  %26 = getelementptr inbounds i8, i8* %22, i64 %25
  %27 = load i8*, i8** %8, align 8
  %28 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %10, i32 0, i32 0
  %29 = call i32 @vsprintf(i8* %26, i8* %27, %struct.__va_list_tag* %28) #5
  %30 = load i32*, i32** %6, align 8
  %31 = load i32, i32* %30, align 4
  %32 = add nsw i32 %31, %29
  store i32 %32, i32* %30, align 4
  %33 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %10, i32 0, i32 0
  %34 = bitcast %struct.__va_list_tag* %33 to i8*
  call void @llvm.va_end(i8* %34)
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpc_err_char_unescape(i8 signext) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca i8, align 1
  store i8 %0, i8* %3, align 1
  store i8 39, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @char_unescape_buffer, i64 0, i64 0), align 1
  store i8 32, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @char_unescape_buffer, i64 0, i64 1), align 1
  store i8 39, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @char_unescape_buffer, i64 0, i64 2), align 1
  store i8 0, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @char_unescape_buffer, i64 0, i64 3), align 1
  %4 = load i8, i8* %3, align 1
  %5 = sext i8 %4 to i32
  switch i32 %5, label %15 [
    i32 7, label %6
    i32 8, label %7
    i32 12, label %8
    i32 13, label %9
    i32 11, label %10
    i32 0, label %11
    i32 10, label %12
    i32 9, label %13
    i32 32, label %14
  ]

; <label>:6:                                      ; preds = %1
  store i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.95, i32 0, i32 0), i8** %2, align 8
  br label %17

; <label>:7:                                      ; preds = %1
  store i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.96, i32 0, i32 0), i8** %2, align 8
  br label %17

; <label>:8:                                      ; preds = %1
  store i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.97, i32 0, i32 0), i8** %2, align 8
  br label %17

; <label>:9:                                      ; preds = %1
  store i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.98, i32 0, i32 0), i8** %2, align 8
  br label %17

; <label>:10:                                     ; preds = %1
  store i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.99, i32 0, i32 0), i8** %2, align 8
  br label %17

; <label>:11:                                     ; preds = %1
  store i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.21, i32 0, i32 0), i8** %2, align 8
  br label %17

; <label>:12:                                     ; preds = %1
  store i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.26, i32 0, i32 0), i8** %2, align 8
  br label %17

; <label>:13:                                     ; preds = %1
  store i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.27, i32 0, i32 0), i8** %2, align 8
  br label %17

; <label>:14:                                     ; preds = %1
  store i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.100, i32 0, i32 0), i8** %2, align 8
  br label %17

; <label>:15:                                     ; preds = %1
  %16 = load i8, i8* %3, align 1
  store i8 %16, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @char_unescape_buffer, i64 0, i64 1), align 1
  store i8* getelementptr inbounds ([4 x i8], [4 x i8]* @char_unescape_buffer, i32 0, i32 0), i8** %2, align 8
  br label %17

; <label>:17:                                     ; preds = %15, %14, %13, %12, %11, %10, %9, %8, %7, %6
  %18 = load i8*, i8** %2, align 8
  ret i8* %18
}

; Function Attrs: nounwind
declare i8* @realloc(i8*, i64) #1

; Function Attrs: nounwind readonly
declare i64 @strlen(i8*) #3

; Function Attrs: noinline nounwind optnone uwtable
define i32 @mpc_parse_input(%struct.mpc_input_t*, %struct.mpc_parser_t*, %union.mpc_result_t*) #0 {
  %4 = alloca %struct.mpc_input_t*, align 8
  %5 = alloca %struct.mpc_parser_t*, align 8
  %6 = alloca %union.mpc_result_t*, align 8
  %7 = alloca i32, align 4
  %8 = alloca %struct.mpc_err_t*, align 8
  %9 = alloca %struct.mpc_state_t, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %4, align 8
  store %struct.mpc_parser_t* %1, %struct.mpc_parser_t** %5, align 8
  store %union.mpc_result_t* %2, %union.mpc_result_t** %6, align 8
  %10 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %11 = call %struct.mpc_err_t* @mpc_err_fail(%struct.mpc_input_t* %10, i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str.8, i32 0, i32 0))
  store %struct.mpc_err_t* %11, %struct.mpc_err_t** %8, align 8
  %12 = load %struct.mpc_err_t*, %struct.mpc_err_t** %8, align 8
  %13 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %12, i32 0, i32 0
  call void @mpc_state_invalid(%struct.mpc_state_t* sret %9)
  %14 = bitcast %struct.mpc_state_t* %13 to i8*
  %15 = bitcast %struct.mpc_state_t* %9 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %14, i8* %15, i64 24, i32 8, i1 false)
  %16 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %17 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %18 = load %union.mpc_result_t*, %union.mpc_result_t** %6, align 8
  %19 = call i32 @mpc_parse_run(%struct.mpc_input_t* %16, %struct.mpc_parser_t* %17, %union.mpc_result_t* %18, %struct.mpc_err_t** %8)
  store i32 %19, i32* %7, align 4
  %20 = load i32, i32* %7, align 4
  %21 = icmp ne i32 %20, 0
  br i1 %21, label %22, label %32

; <label>:22:                                     ; preds = %3
  %23 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %24 = load %struct.mpc_err_t*, %struct.mpc_err_t** %8, align 8
  call void @mpc_err_delete_internal(%struct.mpc_input_t* %23, %struct.mpc_err_t* %24)
  %25 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %26 = load %union.mpc_result_t*, %union.mpc_result_t** %6, align 8
  %27 = bitcast %union.mpc_result_t* %26 to i8**
  %28 = load i8*, i8** %27, align 8
  %29 = call i8* @mpc_export(%struct.mpc_input_t* %25, i8* %28)
  %30 = load %union.mpc_result_t*, %union.mpc_result_t** %6, align 8
  %31 = bitcast %union.mpc_result_t* %30 to i8**
  store i8* %29, i8** %31, align 8
  br label %43

; <label>:32:                                     ; preds = %3
  %33 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %34 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %35 = load %struct.mpc_err_t*, %struct.mpc_err_t** %8, align 8
  %36 = load %union.mpc_result_t*, %union.mpc_result_t** %6, align 8
  %37 = bitcast %union.mpc_result_t* %36 to %struct.mpc_err_t**
  %38 = load %struct.mpc_err_t*, %struct.mpc_err_t** %37, align 8
  %39 = call %struct.mpc_err_t* @mpc_err_merge(%struct.mpc_input_t* %34, %struct.mpc_err_t* %35, %struct.mpc_err_t* %38)
  %40 = call %struct.mpc_err_t* @mpc_err_export(%struct.mpc_input_t* %33, %struct.mpc_err_t* %39)
  %41 = load %union.mpc_result_t*, %union.mpc_result_t** %6, align 8
  %42 = bitcast %union.mpc_result_t* %41 to %struct.mpc_err_t**
  store %struct.mpc_err_t* %40, %struct.mpc_err_t** %42, align 8
  br label %43

; <label>:43:                                     ; preds = %32, %22
  %44 = load i32, i32* %7, align 4
  ret i32 %44
}

; Function Attrs: noinline nounwind optnone uwtable
define internal %struct.mpc_err_t* @mpc_err_fail(%struct.mpc_input_t*, i8*) #0 {
  %3 = alloca %struct.mpc_err_t*, align 8
  %4 = alloca %struct.mpc_input_t*, align 8
  %5 = alloca i8*, align 8
  %6 = alloca %struct.mpc_err_t*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %4, align 8
  store i8* %1, i8** %5, align 8
  %7 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %8 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %7, i32 0, i32 6
  %9 = load i32, i32* %8, align 8
  %10 = icmp ne i32 %9, 0
  br i1 %10, label %11, label %12

; <label>:11:                                     ; preds = %2
  store %struct.mpc_err_t* null, %struct.mpc_err_t** %3, align 8
  br label %57

; <label>:12:                                     ; preds = %2
  %13 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %14 = call i8* @mpc_malloc(%struct.mpc_input_t* %13, i64 64)
  %15 = bitcast i8* %14 to %struct.mpc_err_t*
  store %struct.mpc_err_t* %15, %struct.mpc_err_t** %6, align 8
  %16 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %17 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %18 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %17, i32 0, i32 1
  %19 = load i8*, i8** %18, align 8
  %20 = call i64 @strlen(i8* %19) #7
  %21 = add i64 %20, 1
  %22 = call i8* @mpc_malloc(%struct.mpc_input_t* %16, i64 %21)
  %23 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %24 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %23, i32 0, i32 2
  store i8* %22, i8** %24, align 8
  %25 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %26 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %25, i32 0, i32 2
  %27 = load i8*, i8** %26, align 8
  %28 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %29 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %28, i32 0, i32 1
  %30 = load i8*, i8** %29, align 8
  %31 = call i8* @strcpy(i8* %27, i8* %30) #5
  %32 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %33 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %32, i32 0, i32 0
  %34 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %35 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %34, i32 0, i32 2
  %36 = bitcast %struct.mpc_state_t* %33 to i8*
  %37 = bitcast %struct.mpc_state_t* %35 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %36, i8* %37, i64 24, i32 8, i1 false)
  %38 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %39 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %38, i32 0, i32 1
  store i32 0, i32* %39, align 8
  %40 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %41 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %40, i32 0, i32 4
  store i8** null, i8*** %41, align 8
  %42 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %43 = load i8*, i8** %5, align 8
  %44 = call i64 @strlen(i8* %43) #7
  %45 = add i64 %44, 1
  %46 = call i8* @mpc_malloc(%struct.mpc_input_t* %42, i64 %45)
  %47 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %48 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %47, i32 0, i32 3
  store i8* %46, i8** %48, align 8
  %49 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %50 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %49, i32 0, i32 3
  %51 = load i8*, i8** %50, align 8
  %52 = load i8*, i8** %5, align 8
  %53 = call i8* @strcpy(i8* %51, i8* %52) #5
  %54 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %55 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %54, i32 0, i32 5
  store i8 32, i8* %55, align 8
  %56 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  store %struct.mpc_err_t* %56, %struct.mpc_err_t** %3, align 8
  br label %57

; <label>:57:                                     ; preds = %12, %11
  %58 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  ret %struct.mpc_err_t* %58
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_state_invalid(%struct.mpc_state_t* noalias sret) #0 {
  %2 = alloca %struct.mpc_state_t, align 8
  %3 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %2, i32 0, i32 0
  store i64 -1, i64* %3, align 8
  %4 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %2, i32 0, i32 1
  store i64 -1, i64* %4, align 8
  %5 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %2, i32 0, i32 2
  store i64 -1, i64* %5, align 8
  %6 = bitcast %struct.mpc_state_t* %0 to i8*
  %7 = bitcast %struct.mpc_state_t* %2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %6, i8* %7, i64 24, i32 8, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #4

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @mpc_parse_run(%struct.mpc_input_t*, %struct.mpc_parser_t*, %union.mpc_result_t*, %struct.mpc_err_t**) #0 {
  %5 = alloca i32, align 4
  %6 = alloca %struct.mpc_input_t*, align 8
  %7 = alloca %struct.mpc_parser_t*, align 8
  %8 = alloca %union.mpc_result_t*, align 8
  %9 = alloca %struct.mpc_err_t**, align 8
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca [4 x %union.mpc_result_t], align 16
  %13 = alloca %union.mpc_result_t*, align 8
  %14 = alloca i32, align 4
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %6, align 8
  store %struct.mpc_parser_t* %1, %struct.mpc_parser_t** %7, align 8
  store %union.mpc_result_t* %2, %union.mpc_result_t** %8, align 8
  store %struct.mpc_err_t** %3, %struct.mpc_err_t*** %9, align 8
  store i32 0, i32* %10, align 4
  store i32 0, i32* %11, align 4
  store i32 4, i32* %14, align 4
  %15 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %16 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %15, i32 0, i32 2
  %17 = load i8, i8* %16, align 8
  %18 = sext i8 %17 to i32
  switch i32 %18, label %982 [
    i32 8, label %19
    i32 9, label %34
    i32 12, label %54
    i32 10, label %79
    i32 11, label %99
    i32 13, label %119
    i32 14, label %139
    i32 6, label %159
    i32 0, label %179
    i32 1, label %184
    i32 2, label %187
    i32 3, label %197
    i32 4, label %206
    i32 7, label %214
    i32 15, label %220
    i32 16, label %251
    i32 5, label %286
    i32 17, label %316
    i32 18, label %342
    i32 19, label %382
    i32 20, label %416
    i32 21, label %502
    i32 22, label %609
    i32 23, label %738
    i32 24, label %847
  ]

; <label>:19:                                     ; preds = %4
  %20 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %21 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %22 = bitcast %union.mpc_result_t* %21 to i8**
  %23 = call i32 @mpc_input_any(%struct.mpc_input_t* %20, i8** %22)
  %24 = icmp ne i32 %23, 0
  br i1 %24, label %25, label %31

; <label>:25:                                     ; preds = %19
  %26 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %27 = bitcast %union.mpc_result_t* %26 to i8**
  %28 = load i8*, i8** %27, align 8
  %29 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %30 = bitcast %union.mpc_result_t* %29 to i8**
  store i8* %28, i8** %30, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:31:                                     ; preds = %19
  %32 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %33 = bitcast %union.mpc_result_t* %32 to %struct.mpc_err_t**
  store %struct.mpc_err_t* null, %struct.mpc_err_t** %33, align 8
  store i32 0, i32* %5, align 4
  br label %987

; <label>:34:                                     ; preds = %4
  %35 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %36 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %37 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %36, i32 0, i32 3
  %38 = bitcast %union.mpc_pdata_t* %37 to %struct.mpc_pdata_single_t*
  %39 = getelementptr inbounds %struct.mpc_pdata_single_t, %struct.mpc_pdata_single_t* %38, i32 0, i32 0
  %40 = load i8, i8* %39, align 8
  %41 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %42 = bitcast %union.mpc_result_t* %41 to i8**
  %43 = call i32 @mpc_input_char(%struct.mpc_input_t* %35, i8 signext %40, i8** %42)
  %44 = icmp ne i32 %43, 0
  br i1 %44, label %45, label %51

; <label>:45:                                     ; preds = %34
  %46 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %47 = bitcast %union.mpc_result_t* %46 to i8**
  %48 = load i8*, i8** %47, align 8
  %49 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %50 = bitcast %union.mpc_result_t* %49 to i8**
  store i8* %48, i8** %50, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:51:                                     ; preds = %34
  %52 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %53 = bitcast %union.mpc_result_t* %52 to %struct.mpc_err_t**
  store %struct.mpc_err_t* null, %struct.mpc_err_t** %53, align 8
  store i32 0, i32* %5, align 4
  br label %987

; <label>:54:                                     ; preds = %4
  %55 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %56 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %57 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %56, i32 0, i32 3
  %58 = bitcast %union.mpc_pdata_t* %57 to %struct.mpc_pdata_range_t*
  %59 = getelementptr inbounds %struct.mpc_pdata_range_t, %struct.mpc_pdata_range_t* %58, i32 0, i32 0
  %60 = load i8, i8* %59, align 8
  %61 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %62 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %61, i32 0, i32 3
  %63 = bitcast %union.mpc_pdata_t* %62 to %struct.mpc_pdata_range_t*
  %64 = getelementptr inbounds %struct.mpc_pdata_range_t, %struct.mpc_pdata_range_t* %63, i32 0, i32 1
  %65 = load i8, i8* %64, align 1
  %66 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %67 = bitcast %union.mpc_result_t* %66 to i8**
  %68 = call i32 @mpc_input_range(%struct.mpc_input_t* %55, i8 signext %60, i8 signext %65, i8** %67)
  %69 = icmp ne i32 %68, 0
  br i1 %69, label %70, label %76

; <label>:70:                                     ; preds = %54
  %71 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %72 = bitcast %union.mpc_result_t* %71 to i8**
  %73 = load i8*, i8** %72, align 8
  %74 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %75 = bitcast %union.mpc_result_t* %74 to i8**
  store i8* %73, i8** %75, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:76:                                     ; preds = %54
  %77 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %78 = bitcast %union.mpc_result_t* %77 to %struct.mpc_err_t**
  store %struct.mpc_err_t* null, %struct.mpc_err_t** %78, align 8
  store i32 0, i32* %5, align 4
  br label %987

; <label>:79:                                     ; preds = %4
  %80 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %81 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %82 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %81, i32 0, i32 3
  %83 = bitcast %union.mpc_pdata_t* %82 to %struct.mpc_pdata_string_t*
  %84 = getelementptr inbounds %struct.mpc_pdata_string_t, %struct.mpc_pdata_string_t* %83, i32 0, i32 0
  %85 = load i8*, i8** %84, align 8
  %86 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %87 = bitcast %union.mpc_result_t* %86 to i8**
  %88 = call i32 @mpc_input_oneof(%struct.mpc_input_t* %80, i8* %85, i8** %87)
  %89 = icmp ne i32 %88, 0
  br i1 %89, label %90, label %96

; <label>:90:                                     ; preds = %79
  %91 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %92 = bitcast %union.mpc_result_t* %91 to i8**
  %93 = load i8*, i8** %92, align 8
  %94 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %95 = bitcast %union.mpc_result_t* %94 to i8**
  store i8* %93, i8** %95, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:96:                                     ; preds = %79
  %97 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %98 = bitcast %union.mpc_result_t* %97 to %struct.mpc_err_t**
  store %struct.mpc_err_t* null, %struct.mpc_err_t** %98, align 8
  store i32 0, i32* %5, align 4
  br label %987

; <label>:99:                                     ; preds = %4
  %100 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %101 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %102 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %101, i32 0, i32 3
  %103 = bitcast %union.mpc_pdata_t* %102 to %struct.mpc_pdata_string_t*
  %104 = getelementptr inbounds %struct.mpc_pdata_string_t, %struct.mpc_pdata_string_t* %103, i32 0, i32 0
  %105 = load i8*, i8** %104, align 8
  %106 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %107 = bitcast %union.mpc_result_t* %106 to i8**
  %108 = call i32 @mpc_input_noneof(%struct.mpc_input_t* %100, i8* %105, i8** %107)
  %109 = icmp ne i32 %108, 0
  br i1 %109, label %110, label %116

; <label>:110:                                    ; preds = %99
  %111 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %112 = bitcast %union.mpc_result_t* %111 to i8**
  %113 = load i8*, i8** %112, align 8
  %114 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %115 = bitcast %union.mpc_result_t* %114 to i8**
  store i8* %113, i8** %115, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:116:                                    ; preds = %99
  %117 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %118 = bitcast %union.mpc_result_t* %117 to %struct.mpc_err_t**
  store %struct.mpc_err_t* null, %struct.mpc_err_t** %118, align 8
  store i32 0, i32* %5, align 4
  br label %987

; <label>:119:                                    ; preds = %4
  %120 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %121 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %122 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %121, i32 0, i32 3
  %123 = bitcast %union.mpc_pdata_t* %122 to %struct.mpc_pdata_satisfy_t*
  %124 = getelementptr inbounds %struct.mpc_pdata_satisfy_t, %struct.mpc_pdata_satisfy_t* %123, i32 0, i32 0
  %125 = load i32 (i8)*, i32 (i8)** %124, align 8
  %126 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %127 = bitcast %union.mpc_result_t* %126 to i8**
  %128 = call i32 @mpc_input_satisfy(%struct.mpc_input_t* %120, i32 (i8)* %125, i8** %127)
  %129 = icmp ne i32 %128, 0
  br i1 %129, label %130, label %136

; <label>:130:                                    ; preds = %119
  %131 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %132 = bitcast %union.mpc_result_t* %131 to i8**
  %133 = load i8*, i8** %132, align 8
  %134 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %135 = bitcast %union.mpc_result_t* %134 to i8**
  store i8* %133, i8** %135, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:136:                                    ; preds = %119
  %137 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %138 = bitcast %union.mpc_result_t* %137 to %struct.mpc_err_t**
  store %struct.mpc_err_t* null, %struct.mpc_err_t** %138, align 8
  store i32 0, i32* %5, align 4
  br label %987

; <label>:139:                                    ; preds = %4
  %140 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %141 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %142 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %141, i32 0, i32 3
  %143 = bitcast %union.mpc_pdata_t* %142 to %struct.mpc_pdata_string_t*
  %144 = getelementptr inbounds %struct.mpc_pdata_string_t, %struct.mpc_pdata_string_t* %143, i32 0, i32 0
  %145 = load i8*, i8** %144, align 8
  %146 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %147 = bitcast %union.mpc_result_t* %146 to i8**
  %148 = call i32 @mpc_input_string(%struct.mpc_input_t* %140, i8* %145, i8** %147)
  %149 = icmp ne i32 %148, 0
  br i1 %149, label %150, label %156

; <label>:150:                                    ; preds = %139
  %151 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %152 = bitcast %union.mpc_result_t* %151 to i8**
  %153 = load i8*, i8** %152, align 8
  %154 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %155 = bitcast %union.mpc_result_t* %154 to i8**
  store i8* %153, i8** %155, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:156:                                    ; preds = %139
  %157 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %158 = bitcast %union.mpc_result_t* %157 to %struct.mpc_err_t**
  store %struct.mpc_err_t* null, %struct.mpc_err_t** %158, align 8
  store i32 0, i32* %5, align 4
  br label %987

; <label>:159:                                    ; preds = %4
  %160 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %161 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %162 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %161, i32 0, i32 3
  %163 = bitcast %union.mpc_pdata_t* %162 to %struct.mpc_pdata_anchor_t*
  %164 = getelementptr inbounds %struct.mpc_pdata_anchor_t, %struct.mpc_pdata_anchor_t* %163, i32 0, i32 0
  %165 = load i32 (i8, i8)*, i32 (i8, i8)** %164, align 8
  %166 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %167 = bitcast %union.mpc_result_t* %166 to i8**
  %168 = call i32 @mpc_input_anchor(%struct.mpc_input_t* %160, i32 (i8, i8)* %165, i8** %167)
  %169 = icmp ne i32 %168, 0
  br i1 %169, label %170, label %176

; <label>:170:                                    ; preds = %159
  %171 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %172 = bitcast %union.mpc_result_t* %171 to i8**
  %173 = load i8*, i8** %172, align 8
  %174 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %175 = bitcast %union.mpc_result_t* %174 to i8**
  store i8* %173, i8** %175, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:176:                                    ; preds = %159
  %177 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %178 = bitcast %union.mpc_result_t* %177 to %struct.mpc_err_t**
  store %struct.mpc_err_t* null, %struct.mpc_err_t** %178, align 8
  store i32 0, i32* %5, align 4
  br label %987

; <label>:179:                                    ; preds = %4
  %180 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %181 = call %struct.mpc_err_t* @mpc_err_fail(%struct.mpc_input_t* %180, i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.101, i32 0, i32 0))
  %182 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %183 = bitcast %union.mpc_result_t* %182 to %struct.mpc_err_t**
  store %struct.mpc_err_t* %181, %struct.mpc_err_t** %183, align 8
  store i32 0, i32* %5, align 4
  br label %987

; <label>:184:                                    ; preds = %4
  %185 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %186 = bitcast %union.mpc_result_t* %185 to i8**
  store i8* null, i8** %186, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:187:                                    ; preds = %4
  %188 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %189 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %190 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %189, i32 0, i32 3
  %191 = bitcast %union.mpc_pdata_t* %190 to %struct.mpc_pdata_fail_t*
  %192 = getelementptr inbounds %struct.mpc_pdata_fail_t, %struct.mpc_pdata_fail_t* %191, i32 0, i32 0
  %193 = load i8*, i8** %192, align 8
  %194 = call %struct.mpc_err_t* @mpc_err_fail(%struct.mpc_input_t* %188, i8* %193)
  %195 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %196 = bitcast %union.mpc_result_t* %195 to %struct.mpc_err_t**
  store %struct.mpc_err_t* %194, %struct.mpc_err_t** %196, align 8
  store i32 0, i32* %5, align 4
  br label %987

; <label>:197:                                    ; preds = %4
  %198 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %199 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %198, i32 0, i32 3
  %200 = bitcast %union.mpc_pdata_t* %199 to %struct.mpc_pdata_lift_t*
  %201 = getelementptr inbounds %struct.mpc_pdata_lift_t, %struct.mpc_pdata_lift_t* %200, i32 0, i32 0
  %202 = load i8* ()*, i8* ()** %201, align 8
  %203 = call i8* %202()
  %204 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %205 = bitcast %union.mpc_result_t* %204 to i8**
  store i8* %203, i8** %205, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:206:                                    ; preds = %4
  %207 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %208 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %207, i32 0, i32 3
  %209 = bitcast %union.mpc_pdata_t* %208 to %struct.mpc_pdata_lift_t*
  %210 = getelementptr inbounds %struct.mpc_pdata_lift_t, %struct.mpc_pdata_lift_t* %209, i32 0, i32 1
  %211 = load i8*, i8** %210, align 8
  %212 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %213 = bitcast %union.mpc_result_t* %212 to i8**
  store i8* %211, i8** %213, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:214:                                    ; preds = %4
  %215 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %216 = call %struct.mpc_state_t* @mpc_input_state_copy(%struct.mpc_input_t* %215)
  %217 = bitcast %struct.mpc_state_t* %216 to i8*
  %218 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %219 = bitcast %union.mpc_result_t* %218 to i8**
  store i8* %217, i8** %219, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:220:                                    ; preds = %4
  %221 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %222 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %223 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %222, i32 0, i32 3
  %224 = bitcast %union.mpc_pdata_t* %223 to %struct.mpc_pdata_apply_t*
  %225 = getelementptr inbounds %struct.mpc_pdata_apply_t, %struct.mpc_pdata_apply_t* %224, i32 0, i32 0
  %226 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %225, align 8
  %227 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %228 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %9, align 8
  %229 = call i32 @mpc_parse_run(%struct.mpc_input_t* %221, %struct.mpc_parser_t* %226, %union.mpc_result_t* %227, %struct.mpc_err_t** %228)
  %230 = icmp ne i32 %229, 0
  br i1 %230, label %231, label %244

; <label>:231:                                    ; preds = %220
  %232 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %233 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %234 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %233, i32 0, i32 3
  %235 = bitcast %union.mpc_pdata_t* %234 to %struct.mpc_pdata_apply_t*
  %236 = getelementptr inbounds %struct.mpc_pdata_apply_t, %struct.mpc_pdata_apply_t* %235, i32 0, i32 1
  %237 = load i8* (i8*)*, i8* (i8*)** %236, align 8
  %238 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %239 = bitcast %union.mpc_result_t* %238 to i8**
  %240 = load i8*, i8** %239, align 8
  %241 = call i8* @mpc_parse_apply(%struct.mpc_input_t* %232, i8* (i8*)* %237, i8* %240)
  %242 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %243 = bitcast %union.mpc_result_t* %242 to i8**
  store i8* %241, i8** %243, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:244:                                    ; preds = %220
  %245 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %246 = bitcast %union.mpc_result_t* %245 to i8**
  %247 = load i8*, i8** %246, align 8
  %248 = bitcast i8* %247 to %struct.mpc_err_t*
  %249 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %250 = bitcast %union.mpc_result_t* %249 to %struct.mpc_err_t**
  store %struct.mpc_err_t* %248, %struct.mpc_err_t** %250, align 8
  store i32 0, i32* %5, align 4
  br label %987

; <label>:251:                                    ; preds = %4
  %252 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %253 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %254 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %253, i32 0, i32 3
  %255 = bitcast %union.mpc_pdata_t* %254 to %struct.mpc_pdata_apply_to_t*
  %256 = getelementptr inbounds %struct.mpc_pdata_apply_to_t, %struct.mpc_pdata_apply_to_t* %255, i32 0, i32 0
  %257 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %256, align 8
  %258 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %259 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %9, align 8
  %260 = call i32 @mpc_parse_run(%struct.mpc_input_t* %252, %struct.mpc_parser_t* %257, %union.mpc_result_t* %258, %struct.mpc_err_t** %259)
  %261 = icmp ne i32 %260, 0
  br i1 %261, label %262, label %280

; <label>:262:                                    ; preds = %251
  %263 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %264 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %265 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %264, i32 0, i32 3
  %266 = bitcast %union.mpc_pdata_t* %265 to %struct.mpc_pdata_apply_to_t*
  %267 = getelementptr inbounds %struct.mpc_pdata_apply_to_t, %struct.mpc_pdata_apply_to_t* %266, i32 0, i32 1
  %268 = load i8* (i8*, i8*)*, i8* (i8*, i8*)** %267, align 8
  %269 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %270 = bitcast %union.mpc_result_t* %269 to i8**
  %271 = load i8*, i8** %270, align 8
  %272 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %273 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %272, i32 0, i32 3
  %274 = bitcast %union.mpc_pdata_t* %273 to %struct.mpc_pdata_apply_to_t*
  %275 = getelementptr inbounds %struct.mpc_pdata_apply_to_t, %struct.mpc_pdata_apply_to_t* %274, i32 0, i32 2
  %276 = load i8*, i8** %275, align 8
  %277 = call i8* @mpc_parse_apply_to(%struct.mpc_input_t* %263, i8* (i8*, i8*)* %268, i8* %271, i8* %276)
  %278 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %279 = bitcast %union.mpc_result_t* %278 to i8**
  store i8* %277, i8** %279, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:280:                                    ; preds = %251
  %281 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %282 = bitcast %union.mpc_result_t* %281 to %struct.mpc_err_t**
  %283 = load %struct.mpc_err_t*, %struct.mpc_err_t** %282, align 8
  %284 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %285 = bitcast %union.mpc_result_t* %284 to %struct.mpc_err_t**
  store %struct.mpc_err_t* %283, %struct.mpc_err_t** %285, align 8
  store i32 0, i32* %5, align 4
  br label %987

; <label>:286:                                    ; preds = %4
  %287 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  call void @mpc_input_suppress_enable(%struct.mpc_input_t* %287)
  %288 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %289 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %290 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %289, i32 0, i32 3
  %291 = bitcast %union.mpc_pdata_t* %290 to %struct.mpc_pdata_expect_t*
  %292 = getelementptr inbounds %struct.mpc_pdata_expect_t, %struct.mpc_pdata_expect_t* %291, i32 0, i32 0
  %293 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %292, align 8
  %294 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %295 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %9, align 8
  %296 = call i32 @mpc_parse_run(%struct.mpc_input_t* %288, %struct.mpc_parser_t* %293, %union.mpc_result_t* %294, %struct.mpc_err_t** %295)
  %297 = icmp ne i32 %296, 0
  br i1 %297, label %298, label %305

; <label>:298:                                    ; preds = %286
  %299 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  call void @mpc_input_suppress_disable(%struct.mpc_input_t* %299)
  %300 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %301 = bitcast %union.mpc_result_t* %300 to i8**
  %302 = load i8*, i8** %301, align 8
  %303 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %304 = bitcast %union.mpc_result_t* %303 to i8**
  store i8* %302, i8** %304, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:305:                                    ; preds = %286
  %306 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  call void @mpc_input_suppress_disable(%struct.mpc_input_t* %306)
  %307 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %308 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %309 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %308, i32 0, i32 3
  %310 = bitcast %union.mpc_pdata_t* %309 to %struct.mpc_pdata_expect_t*
  %311 = getelementptr inbounds %struct.mpc_pdata_expect_t, %struct.mpc_pdata_expect_t* %310, i32 0, i32 1
  %312 = load i8*, i8** %311, align 8
  %313 = call %struct.mpc_err_t* @mpc_err_new(%struct.mpc_input_t* %307, i8* %312)
  %314 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %315 = bitcast %union.mpc_result_t* %314 to %struct.mpc_err_t**
  store %struct.mpc_err_t* %313, %struct.mpc_err_t** %315, align 8
  store i32 0, i32* %5, align 4
  br label %987

; <label>:316:                                    ; preds = %4
  %317 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  call void @mpc_input_backtrack_disable(%struct.mpc_input_t* %317)
  %318 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %319 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %320 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %319, i32 0, i32 3
  %321 = bitcast %union.mpc_pdata_t* %320 to %struct.mpc_pdata_predict_t*
  %322 = getelementptr inbounds %struct.mpc_pdata_predict_t, %struct.mpc_pdata_predict_t* %321, i32 0, i32 0
  %323 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %322, align 8
  %324 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %325 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %9, align 8
  %326 = call i32 @mpc_parse_run(%struct.mpc_input_t* %318, %struct.mpc_parser_t* %323, %union.mpc_result_t* %324, %struct.mpc_err_t** %325)
  %327 = icmp ne i32 %326, 0
  br i1 %327, label %328, label %335

; <label>:328:                                    ; preds = %316
  %329 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  call void @mpc_input_backtrack_enable(%struct.mpc_input_t* %329)
  %330 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %331 = bitcast %union.mpc_result_t* %330 to i8**
  %332 = load i8*, i8** %331, align 8
  %333 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %334 = bitcast %union.mpc_result_t* %333 to i8**
  store i8* %332, i8** %334, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:335:                                    ; preds = %316
  %336 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  call void @mpc_input_backtrack_enable(%struct.mpc_input_t* %336)
  %337 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %338 = bitcast %union.mpc_result_t* %337 to %struct.mpc_err_t**
  %339 = load %struct.mpc_err_t*, %struct.mpc_err_t** %338, align 8
  %340 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %341 = bitcast %union.mpc_result_t* %340 to %struct.mpc_err_t**
  store %struct.mpc_err_t* %339, %struct.mpc_err_t** %341, align 8
  store i32 0, i32* %5, align 4
  br label %987

; <label>:342:                                    ; preds = %4
  %343 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  call void @mpc_input_mark(%struct.mpc_input_t* %343)
  %344 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  call void @mpc_input_suppress_enable(%struct.mpc_input_t* %344)
  %345 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %346 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %347 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %346, i32 0, i32 3
  %348 = bitcast %union.mpc_pdata_t* %347 to %struct.mpc_pdata_not_t*
  %349 = getelementptr inbounds %struct.mpc_pdata_not_t, %struct.mpc_pdata_not_t* %348, i32 0, i32 0
  %350 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %349, align 8
  %351 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %352 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %9, align 8
  %353 = call i32 @mpc_parse_run(%struct.mpc_input_t* %345, %struct.mpc_parser_t* %350, %union.mpc_result_t* %351, %struct.mpc_err_t** %352)
  %354 = icmp ne i32 %353, 0
  br i1 %354, label %355, label %371

; <label>:355:                                    ; preds = %342
  %356 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  call void @mpc_input_rewind(%struct.mpc_input_t* %356)
  %357 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  call void @mpc_input_suppress_disable(%struct.mpc_input_t* %357)
  %358 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %359 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %360 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %359, i32 0, i32 3
  %361 = bitcast %union.mpc_pdata_t* %360 to %struct.mpc_pdata_not_t*
  %362 = getelementptr inbounds %struct.mpc_pdata_not_t, %struct.mpc_pdata_not_t* %361, i32 0, i32 1
  %363 = load void (i8*)*, void (i8*)** %362, align 8
  %364 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %365 = bitcast %union.mpc_result_t* %364 to i8**
  %366 = load i8*, i8** %365, align 8
  call void @mpc_parse_dtor(%struct.mpc_input_t* %358, void (i8*)* %363, i8* %366)
  %367 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %368 = call %struct.mpc_err_t* @mpc_err_new(%struct.mpc_input_t* %367, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.102, i32 0, i32 0))
  %369 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %370 = bitcast %union.mpc_result_t* %369 to %struct.mpc_err_t**
  store %struct.mpc_err_t* %368, %struct.mpc_err_t** %370, align 8
  store i32 0, i32* %5, align 4
  br label %987

; <label>:371:                                    ; preds = %342
  %372 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  call void @mpc_input_unmark(%struct.mpc_input_t* %372)
  %373 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  call void @mpc_input_suppress_disable(%struct.mpc_input_t* %373)
  %374 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %375 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %374, i32 0, i32 3
  %376 = bitcast %union.mpc_pdata_t* %375 to %struct.mpc_pdata_not_t*
  %377 = getelementptr inbounds %struct.mpc_pdata_not_t, %struct.mpc_pdata_not_t* %376, i32 0, i32 2
  %378 = load i8* ()*, i8* ()** %377, align 8
  %379 = call i8* %378()
  %380 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %381 = bitcast %union.mpc_result_t* %380 to i8**
  store i8* %379, i8** %381, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:382:                                    ; preds = %4
  %383 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %384 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %385 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %384, i32 0, i32 3
  %386 = bitcast %union.mpc_pdata_t* %385 to %struct.mpc_pdata_not_t*
  %387 = getelementptr inbounds %struct.mpc_pdata_not_t, %struct.mpc_pdata_not_t* %386, i32 0, i32 0
  %388 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %387, align 8
  %389 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %390 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %9, align 8
  %391 = call i32 @mpc_parse_run(%struct.mpc_input_t* %383, %struct.mpc_parser_t* %388, %union.mpc_result_t* %389, %struct.mpc_err_t** %390)
  %392 = icmp ne i32 %391, 0
  br i1 %392, label %393, label %399

; <label>:393:                                    ; preds = %382
  %394 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %395 = bitcast %union.mpc_result_t* %394 to i8**
  %396 = load i8*, i8** %395, align 8
  %397 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %398 = bitcast %union.mpc_result_t* %397 to i8**
  store i8* %396, i8** %398, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:399:                                    ; preds = %382
  %400 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %401 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %9, align 8
  %402 = load %struct.mpc_err_t*, %struct.mpc_err_t** %401, align 8
  %403 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %404 = bitcast %union.mpc_result_t* %403 to %struct.mpc_err_t**
  %405 = load %struct.mpc_err_t*, %struct.mpc_err_t** %404, align 8
  %406 = call %struct.mpc_err_t* @mpc_err_merge(%struct.mpc_input_t* %400, %struct.mpc_err_t* %402, %struct.mpc_err_t* %405)
  %407 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %9, align 8
  store %struct.mpc_err_t* %406, %struct.mpc_err_t** %407, align 8
  %408 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %409 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %408, i32 0, i32 3
  %410 = bitcast %union.mpc_pdata_t* %409 to %struct.mpc_pdata_not_t*
  %411 = getelementptr inbounds %struct.mpc_pdata_not_t, %struct.mpc_pdata_not_t* %410, i32 0, i32 2
  %412 = load i8* ()*, i8* ()** %411, align 8
  %413 = call i8* %412()
  %414 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %415 = bitcast %union.mpc_result_t* %414 to i8**
  store i8* %413, i8** %415, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:416:                                    ; preds = %4
  %417 = getelementptr inbounds [4 x %union.mpc_result_t], [4 x %union.mpc_result_t]* %12, i32 0, i32 0
  store %union.mpc_result_t* %417, %union.mpc_result_t** %13, align 8
  br label %418

; <label>:418:                                    ; preds = %470, %416
  %419 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %420 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %421 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %420, i32 0, i32 3
  %422 = bitcast %union.mpc_pdata_t* %421 to %struct.mpc_pdata_repeat_t*
  %423 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %422, i32 0, i32 2
  %424 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %423, align 8
  %425 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %426 = load i32, i32* %10, align 4
  %427 = sext i32 %426 to i64
  %428 = getelementptr inbounds %union.mpc_result_t, %union.mpc_result_t* %425, i64 %427
  %429 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %9, align 8
  %430 = call i32 @mpc_parse_run(%struct.mpc_input_t* %419, %struct.mpc_parser_t* %424, %union.mpc_result_t* %428, %struct.mpc_err_t** %429)
  %431 = icmp ne i32 %430, 0
  br i1 %431, label %432, label %471

; <label>:432:                                    ; preds = %418
  %433 = load i32, i32* %10, align 4
  %434 = add nsw i32 %433, 1
  store i32 %434, i32* %10, align 4
  %435 = load i32, i32* %10, align 4
  %436 = icmp eq i32 %435, 4
  br i1 %436, label %437, label %452

; <label>:437:                                    ; preds = %432
  %438 = load i32, i32* %10, align 4
  %439 = load i32, i32* %10, align 4
  %440 = sdiv i32 %439, 2
  %441 = add nsw i32 %438, %440
  store i32 %441, i32* %14, align 4
  %442 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %443 = load i32, i32* %14, align 4
  %444 = sext i32 %443 to i64
  %445 = mul i64 8, %444
  %446 = call i8* @mpc_malloc(%struct.mpc_input_t* %442, i64 %445)
  %447 = bitcast i8* %446 to %union.mpc_result_t*
  store %union.mpc_result_t* %447, %union.mpc_result_t** %13, align 8
  %448 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %449 = bitcast %union.mpc_result_t* %448 to i8*
  %450 = getelementptr inbounds [4 x %union.mpc_result_t], [4 x %union.mpc_result_t]* %12, i32 0, i32 0
  %451 = bitcast %union.mpc_result_t* %450 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %449, i8* %451, i64 32, i32 8, i1 false)
  br label %470

; <label>:452:                                    ; preds = %432
  %453 = load i32, i32* %10, align 4
  %454 = load i32, i32* %14, align 4
  %455 = icmp sge i32 %453, %454
  br i1 %455, label %456, label %469

; <label>:456:                                    ; preds = %452
  %457 = load i32, i32* %10, align 4
  %458 = load i32, i32* %10, align 4
  %459 = sdiv i32 %458, 2
  %460 = add nsw i32 %457, %459
  store i32 %460, i32* %14, align 4
  %461 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %462 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %463 = bitcast %union.mpc_result_t* %462 to i8*
  %464 = load i32, i32* %14, align 4
  %465 = sext i32 %464 to i64
  %466 = mul i64 8, %465
  %467 = call i8* @mpc_realloc(%struct.mpc_input_t* %461, i8* %463, i64 %466)
  %468 = bitcast i8* %467 to %union.mpc_result_t*
  store %union.mpc_result_t* %468, %union.mpc_result_t** %13, align 8
  br label %469

; <label>:469:                                    ; preds = %456, %452
  br label %470

; <label>:470:                                    ; preds = %469, %437
  br label %418

; <label>:471:                                    ; preds = %418
  %472 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %473 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %9, align 8
  %474 = load %struct.mpc_err_t*, %struct.mpc_err_t** %473, align 8
  %475 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %476 = load i32, i32* %10, align 4
  %477 = sext i32 %476 to i64
  %478 = getelementptr inbounds %union.mpc_result_t, %union.mpc_result_t* %475, i64 %477
  %479 = bitcast %union.mpc_result_t* %478 to %struct.mpc_err_t**
  %480 = load %struct.mpc_err_t*, %struct.mpc_err_t** %479, align 8
  %481 = call %struct.mpc_err_t* @mpc_err_merge(%struct.mpc_input_t* %472, %struct.mpc_err_t* %474, %struct.mpc_err_t* %480)
  %482 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %9, align 8
  store %struct.mpc_err_t* %481, %struct.mpc_err_t** %482, align 8
  %483 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %484 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %485 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %484, i32 0, i32 3
  %486 = bitcast %union.mpc_pdata_t* %485 to %struct.mpc_pdata_repeat_t*
  %487 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %486, i32 0, i32 1
  %488 = load i8* (i32, i8**)*, i8* (i32, i8**)** %487, align 8
  %489 = load i32, i32* %10, align 4
  %490 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %491 = bitcast %union.mpc_result_t* %490 to i8**
  %492 = call i8* @mpc_parse_fold(%struct.mpc_input_t* %483, i8* (i32, i8**)* %488, i32 %489, i8** %491)
  %493 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %494 = bitcast %union.mpc_result_t* %493 to i8**
  store i8* %492, i8** %494, align 8
  %495 = load i32, i32* %10, align 4
  %496 = icmp sge i32 %495, 4
  br i1 %496, label %497, label %501

; <label>:497:                                    ; preds = %471
  %498 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %499 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %500 = bitcast %union.mpc_result_t* %499 to i8*
  call void @mpc_free(%struct.mpc_input_t* %498, i8* %500)
  br label %501

; <label>:501:                                    ; preds = %497, %471
  store i32 1, i32* %5, align 4
  br label %987

; <label>:502:                                    ; preds = %4
  %503 = getelementptr inbounds [4 x %union.mpc_result_t], [4 x %union.mpc_result_t]* %12, i32 0, i32 0
  store %union.mpc_result_t* %503, %union.mpc_result_t** %13, align 8
  br label %504

; <label>:504:                                    ; preds = %556, %502
  %505 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %506 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %507 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %506, i32 0, i32 3
  %508 = bitcast %union.mpc_pdata_t* %507 to %struct.mpc_pdata_repeat_t*
  %509 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %508, i32 0, i32 2
  %510 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %509, align 8
  %511 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %512 = load i32, i32* %10, align 4
  %513 = sext i32 %512 to i64
  %514 = getelementptr inbounds %union.mpc_result_t, %union.mpc_result_t* %511, i64 %513
  %515 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %9, align 8
  %516 = call i32 @mpc_parse_run(%struct.mpc_input_t* %505, %struct.mpc_parser_t* %510, %union.mpc_result_t* %514, %struct.mpc_err_t** %515)
  %517 = icmp ne i32 %516, 0
  br i1 %517, label %518, label %557

; <label>:518:                                    ; preds = %504
  %519 = load i32, i32* %10, align 4
  %520 = add nsw i32 %519, 1
  store i32 %520, i32* %10, align 4
  %521 = load i32, i32* %10, align 4
  %522 = icmp eq i32 %521, 4
  br i1 %522, label %523, label %538

; <label>:523:                                    ; preds = %518
  %524 = load i32, i32* %10, align 4
  %525 = load i32, i32* %10, align 4
  %526 = sdiv i32 %525, 2
  %527 = add nsw i32 %524, %526
  store i32 %527, i32* %14, align 4
  %528 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %529 = load i32, i32* %14, align 4
  %530 = sext i32 %529 to i64
  %531 = mul i64 8, %530
  %532 = call i8* @mpc_malloc(%struct.mpc_input_t* %528, i64 %531)
  %533 = bitcast i8* %532 to %union.mpc_result_t*
  store %union.mpc_result_t* %533, %union.mpc_result_t** %13, align 8
  %534 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %535 = bitcast %union.mpc_result_t* %534 to i8*
  %536 = getelementptr inbounds [4 x %union.mpc_result_t], [4 x %union.mpc_result_t]* %12, i32 0, i32 0
  %537 = bitcast %union.mpc_result_t* %536 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %535, i8* %537, i64 32, i32 8, i1 false)
  br label %556

; <label>:538:                                    ; preds = %518
  %539 = load i32, i32* %10, align 4
  %540 = load i32, i32* %14, align 4
  %541 = icmp sge i32 %539, %540
  br i1 %541, label %542, label %555

; <label>:542:                                    ; preds = %538
  %543 = load i32, i32* %10, align 4
  %544 = load i32, i32* %10, align 4
  %545 = sdiv i32 %544, 2
  %546 = add nsw i32 %543, %545
  store i32 %546, i32* %14, align 4
  %547 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %548 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %549 = bitcast %union.mpc_result_t* %548 to i8*
  %550 = load i32, i32* %14, align 4
  %551 = sext i32 %550 to i64
  %552 = mul i64 8, %551
  %553 = call i8* @mpc_realloc(%struct.mpc_input_t* %547, i8* %549, i64 %552)
  %554 = bitcast i8* %553 to %union.mpc_result_t*
  store %union.mpc_result_t* %554, %union.mpc_result_t** %13, align 8
  br label %555

; <label>:555:                                    ; preds = %542, %538
  br label %556

; <label>:556:                                    ; preds = %555, %523
  br label %504

; <label>:557:                                    ; preds = %504
  %558 = load i32, i32* %10, align 4
  %559 = icmp eq i32 %558, 0
  br i1 %559, label %560, label %578

; <label>:560:                                    ; preds = %557
  %561 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %562 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %563 = load i32, i32* %10, align 4
  %564 = sext i32 %563 to i64
  %565 = getelementptr inbounds %union.mpc_result_t, %union.mpc_result_t* %562, i64 %564
  %566 = bitcast %union.mpc_result_t* %565 to %struct.mpc_err_t**
  %567 = load %struct.mpc_err_t*, %struct.mpc_err_t** %566, align 8
  %568 = call %struct.mpc_err_t* @mpc_err_many1(%struct.mpc_input_t* %561, %struct.mpc_err_t* %567)
  %569 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %570 = bitcast %union.mpc_result_t* %569 to %struct.mpc_err_t**
  store %struct.mpc_err_t* %568, %struct.mpc_err_t** %570, align 8
  %571 = load i32, i32* %10, align 4
  %572 = icmp sge i32 %571, 4
  br i1 %572, label %573, label %577

; <label>:573:                                    ; preds = %560
  %574 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %575 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %576 = bitcast %union.mpc_result_t* %575 to i8*
  call void @mpc_free(%struct.mpc_input_t* %574, i8* %576)
  br label %577

; <label>:577:                                    ; preds = %573, %560
  store i32 0, i32* %5, align 4
  br label %987

; <label>:578:                                    ; preds = %557
  %579 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %580 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %9, align 8
  %581 = load %struct.mpc_err_t*, %struct.mpc_err_t** %580, align 8
  %582 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %583 = load i32, i32* %10, align 4
  %584 = sext i32 %583 to i64
  %585 = getelementptr inbounds %union.mpc_result_t, %union.mpc_result_t* %582, i64 %584
  %586 = bitcast %union.mpc_result_t* %585 to %struct.mpc_err_t**
  %587 = load %struct.mpc_err_t*, %struct.mpc_err_t** %586, align 8
  %588 = call %struct.mpc_err_t* @mpc_err_merge(%struct.mpc_input_t* %579, %struct.mpc_err_t* %581, %struct.mpc_err_t* %587)
  %589 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %9, align 8
  store %struct.mpc_err_t* %588, %struct.mpc_err_t** %589, align 8
  %590 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %591 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %592 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %591, i32 0, i32 3
  %593 = bitcast %union.mpc_pdata_t* %592 to %struct.mpc_pdata_repeat_t*
  %594 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %593, i32 0, i32 1
  %595 = load i8* (i32, i8**)*, i8* (i32, i8**)** %594, align 8
  %596 = load i32, i32* %10, align 4
  %597 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %598 = bitcast %union.mpc_result_t* %597 to i8**
  %599 = call i8* @mpc_parse_fold(%struct.mpc_input_t* %590, i8* (i32, i8**)* %595, i32 %596, i8** %598)
  %600 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %601 = bitcast %union.mpc_result_t* %600 to i8**
  store i8* %599, i8** %601, align 8
  %602 = load i32, i32* %10, align 4
  %603 = icmp sge i32 %602, 4
  br i1 %603, label %604, label %608

; <label>:604:                                    ; preds = %578
  %605 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %606 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %607 = bitcast %union.mpc_result_t* %606 to i8*
  call void @mpc_free(%struct.mpc_input_t* %605, i8* %607)
  br label %608

; <label>:608:                                    ; preds = %604, %578
  store i32 1, i32* %5, align 4
  br label %987

; <label>:609:                                    ; preds = %4
  %610 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %611 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %610, i32 0, i32 3
  %612 = bitcast %union.mpc_pdata_t* %611 to %struct.mpc_pdata_repeat_t*
  %613 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %612, i32 0, i32 0
  %614 = load i32, i32* %613, align 8
  %615 = icmp sgt i32 %614, 4
  br i1 %615, label %616, label %626

; <label>:616:                                    ; preds = %609
  %617 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %618 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %619 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %618, i32 0, i32 3
  %620 = bitcast %union.mpc_pdata_t* %619 to %struct.mpc_pdata_repeat_t*
  %621 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %620, i32 0, i32 0
  %622 = load i32, i32* %621, align 8
  %623 = sext i32 %622 to i64
  %624 = mul i64 8, %623
  %625 = call i8* @mpc_malloc(%struct.mpc_input_t* %617, i64 %624)
  br label %629

; <label>:626:                                    ; preds = %609
  %627 = getelementptr inbounds [4 x %union.mpc_result_t], [4 x %union.mpc_result_t]* %12, i32 0, i32 0
  %628 = bitcast %union.mpc_result_t* %627 to i8*
  br label %629

; <label>:629:                                    ; preds = %626, %616
  %630 = phi i8* [ %625, %616 ], [ %628, %626 ]
  %631 = bitcast i8* %630 to %union.mpc_result_t*
  store %union.mpc_result_t* %631, %union.mpc_result_t** %13, align 8
  br label %632

; <label>:632:                                    ; preds = %657, %629
  %633 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %634 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %635 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %634, i32 0, i32 3
  %636 = bitcast %union.mpc_pdata_t* %635 to %struct.mpc_pdata_repeat_t*
  %637 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %636, i32 0, i32 2
  %638 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %637, align 8
  %639 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %640 = load i32, i32* %10, align 4
  %641 = sext i32 %640 to i64
  %642 = getelementptr inbounds %union.mpc_result_t, %union.mpc_result_t* %639, i64 %641
  %643 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %9, align 8
  %644 = call i32 @mpc_parse_run(%struct.mpc_input_t* %633, %struct.mpc_parser_t* %638, %union.mpc_result_t* %642, %struct.mpc_err_t** %643)
  %645 = icmp ne i32 %644, 0
  br i1 %645, label %646, label %658

; <label>:646:                                    ; preds = %632
  %647 = load i32, i32* %10, align 4
  %648 = add nsw i32 %647, 1
  store i32 %648, i32* %10, align 4
  %649 = load i32, i32* %10, align 4
  %650 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %651 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %650, i32 0, i32 3
  %652 = bitcast %union.mpc_pdata_t* %651 to %struct.mpc_pdata_repeat_t*
  %653 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %652, i32 0, i32 0
  %654 = load i32, i32* %653, align 8
  %655 = icmp eq i32 %649, %654
  br i1 %655, label %656, label %657

; <label>:656:                                    ; preds = %646
  br label %658

; <label>:657:                                    ; preds = %646
  br label %632

; <label>:658:                                    ; preds = %656, %632
  %659 = load i32, i32* %10, align 4
  %660 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %661 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %660, i32 0, i32 3
  %662 = bitcast %union.mpc_pdata_t* %661 to %struct.mpc_pdata_repeat_t*
  %663 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %662, i32 0, i32 0
  %664 = load i32, i32* %663, align 8
  %665 = icmp eq i32 %659, %664
  br i1 %665, label %666, label %690

; <label>:666:                                    ; preds = %658
  %667 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %668 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %669 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %668, i32 0, i32 3
  %670 = bitcast %union.mpc_pdata_t* %669 to %struct.mpc_pdata_repeat_t*
  %671 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %670, i32 0, i32 1
  %672 = load i8* (i32, i8**)*, i8* (i32, i8**)** %671, align 8
  %673 = load i32, i32* %10, align 4
  %674 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %675 = bitcast %union.mpc_result_t* %674 to i8**
  %676 = call i8* @mpc_parse_fold(%struct.mpc_input_t* %667, i8* (i32, i8**)* %672, i32 %673, i8** %675)
  %677 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %678 = bitcast %union.mpc_result_t* %677 to i8**
  store i8* %676, i8** %678, align 8
  %679 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %680 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %679, i32 0, i32 3
  %681 = bitcast %union.mpc_pdata_t* %680 to %struct.mpc_pdata_repeat_t*
  %682 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %681, i32 0, i32 0
  %683 = load i32, i32* %682, align 8
  %684 = icmp sgt i32 %683, 4
  br i1 %684, label %685, label %689

; <label>:685:                                    ; preds = %666
  %686 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %687 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %688 = bitcast %union.mpc_result_t* %687 to i8*
  call void @mpc_free(%struct.mpc_input_t* %686, i8* %688)
  br label %689

; <label>:689:                                    ; preds = %685, %666
  store i32 1, i32* %5, align 4
  br label %987

; <label>:690:                                    ; preds = %658
  store i32 0, i32* %11, align 4
  br label %691

; <label>:691:                                    ; preds = %708, %690
  %692 = load i32, i32* %11, align 4
  %693 = load i32, i32* %10, align 4
  %694 = icmp slt i32 %692, %693
  br i1 %694, label %695, label %711

; <label>:695:                                    ; preds = %691
  %696 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %697 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %698 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %697, i32 0, i32 3
  %699 = bitcast %union.mpc_pdata_t* %698 to %struct.mpc_pdata_repeat_t*
  %700 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %699, i32 0, i32 3
  %701 = load void (i8*)*, void (i8*)** %700, align 8
  %702 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %703 = load i32, i32* %11, align 4
  %704 = sext i32 %703 to i64
  %705 = getelementptr inbounds %union.mpc_result_t, %union.mpc_result_t* %702, i64 %704
  %706 = bitcast %union.mpc_result_t* %705 to i8**
  %707 = load i8*, i8** %706, align 8
  call void @mpc_parse_dtor(%struct.mpc_input_t* %696, void (i8*)* %701, i8* %707)
  br label %708

; <label>:708:                                    ; preds = %695
  %709 = load i32, i32* %11, align 4
  %710 = add nsw i32 %709, 1
  store i32 %710, i32* %11, align 4
  br label %691

; <label>:711:                                    ; preds = %691
  %712 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %713 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %714 = load i32, i32* %10, align 4
  %715 = sext i32 %714 to i64
  %716 = getelementptr inbounds %union.mpc_result_t, %union.mpc_result_t* %713, i64 %715
  %717 = bitcast %union.mpc_result_t* %716 to %struct.mpc_err_t**
  %718 = load %struct.mpc_err_t*, %struct.mpc_err_t** %717, align 8
  %719 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %720 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %719, i32 0, i32 3
  %721 = bitcast %union.mpc_pdata_t* %720 to %struct.mpc_pdata_repeat_t*
  %722 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %721, i32 0, i32 0
  %723 = load i32, i32* %722, align 8
  %724 = call %struct.mpc_err_t* @mpc_err_count(%struct.mpc_input_t* %712, %struct.mpc_err_t* %718, i32 %723)
  %725 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %726 = bitcast %union.mpc_result_t* %725 to %struct.mpc_err_t**
  store %struct.mpc_err_t* %724, %struct.mpc_err_t** %726, align 8
  %727 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %728 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %727, i32 0, i32 3
  %729 = bitcast %union.mpc_pdata_t* %728 to %struct.mpc_pdata_repeat_t*
  %730 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %729, i32 0, i32 0
  %731 = load i32, i32* %730, align 8
  %732 = icmp sgt i32 %731, 4
  br i1 %732, label %733, label %737

; <label>:733:                                    ; preds = %711
  %734 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %735 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %736 = bitcast %union.mpc_result_t* %735 to i8*
  call void @mpc_free(%struct.mpc_input_t* %734, i8* %736)
  br label %737

; <label>:737:                                    ; preds = %733, %711
  store i32 0, i32* %5, align 4
  br label %987

; <label>:738:                                    ; preds = %4
  %739 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %740 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %739, i32 0, i32 3
  %741 = bitcast %union.mpc_pdata_t* %740 to %struct.mpc_pdata_or_t*
  %742 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %741, i32 0, i32 0
  %743 = load i32, i32* %742, align 8
  %744 = icmp eq i32 %743, 0
  br i1 %744, label %745, label %748

; <label>:745:                                    ; preds = %738
  %746 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %747 = bitcast %union.mpc_result_t* %746 to i8**
  store i8* null, i8** %747, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:748:                                    ; preds = %738
  %749 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %750 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %749, i32 0, i32 3
  %751 = bitcast %union.mpc_pdata_t* %750 to %struct.mpc_pdata_or_t*
  %752 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %751, i32 0, i32 0
  %753 = load i32, i32* %752, align 8
  %754 = icmp sgt i32 %753, 4
  br i1 %754, label %755, label %765

; <label>:755:                                    ; preds = %748
  %756 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %757 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %758 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %757, i32 0, i32 3
  %759 = bitcast %union.mpc_pdata_t* %758 to %struct.mpc_pdata_or_t*
  %760 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %759, i32 0, i32 0
  %761 = load i32, i32* %760, align 8
  %762 = sext i32 %761 to i64
  %763 = mul i64 8, %762
  %764 = call i8* @mpc_malloc(%struct.mpc_input_t* %756, i64 %763)
  br label %768

; <label>:765:                                    ; preds = %748
  %766 = getelementptr inbounds [4 x %union.mpc_result_t], [4 x %union.mpc_result_t]* %12, i32 0, i32 0
  %767 = bitcast %union.mpc_result_t* %766 to i8*
  br label %768

; <label>:768:                                    ; preds = %765, %755
  %769 = phi i8* [ %764, %755 ], [ %767, %765 ]
  %770 = bitcast i8* %769 to %union.mpc_result_t*
  store %union.mpc_result_t* %770, %union.mpc_result_t** %13, align 8
  store i32 0, i32* %10, align 4
  br label %771

; <label>:771:                                    ; preds = %830, %768
  %772 = load i32, i32* %10, align 4
  %773 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %774 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %773, i32 0, i32 3
  %775 = bitcast %union.mpc_pdata_t* %774 to %struct.mpc_pdata_or_t*
  %776 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %775, i32 0, i32 0
  %777 = load i32, i32* %776, align 8
  %778 = icmp slt i32 %772, %777
  br i1 %778, label %779, label %833

; <label>:779:                                    ; preds = %771
  %780 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %781 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %782 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %781, i32 0, i32 3
  %783 = bitcast %union.mpc_pdata_t* %782 to %struct.mpc_pdata_or_t*
  %784 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %783, i32 0, i32 1
  %785 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %784, align 8
  %786 = load i32, i32* %10, align 4
  %787 = sext i32 %786 to i64
  %788 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %785, i64 %787
  %789 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %788, align 8
  %790 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %791 = load i32, i32* %10, align 4
  %792 = sext i32 %791 to i64
  %793 = getelementptr inbounds %union.mpc_result_t, %union.mpc_result_t* %790, i64 %792
  %794 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %9, align 8
  %795 = call i32 @mpc_parse_run(%struct.mpc_input_t* %780, %struct.mpc_parser_t* %789, %union.mpc_result_t* %793, %struct.mpc_err_t** %794)
  %796 = icmp ne i32 %795, 0
  br i1 %796, label %797, label %817

; <label>:797:                                    ; preds = %779
  %798 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %799 = load i32, i32* %10, align 4
  %800 = sext i32 %799 to i64
  %801 = getelementptr inbounds %union.mpc_result_t, %union.mpc_result_t* %798, i64 %800
  %802 = bitcast %union.mpc_result_t* %801 to i8**
  %803 = load i8*, i8** %802, align 8
  %804 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %805 = bitcast %union.mpc_result_t* %804 to i8**
  store i8* %803, i8** %805, align 8
  %806 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %807 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %806, i32 0, i32 3
  %808 = bitcast %union.mpc_pdata_t* %807 to %struct.mpc_pdata_or_t*
  %809 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %808, i32 0, i32 0
  %810 = load i32, i32* %809, align 8
  %811 = icmp sgt i32 %810, 4
  br i1 %811, label %812, label %816

; <label>:812:                                    ; preds = %797
  %813 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %814 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %815 = bitcast %union.mpc_result_t* %814 to i8*
  call void @mpc_free(%struct.mpc_input_t* %813, i8* %815)
  br label %816

; <label>:816:                                    ; preds = %812, %797
  store i32 1, i32* %5, align 4
  br label %987

; <label>:817:                                    ; preds = %779
  %818 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %819 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %9, align 8
  %820 = load %struct.mpc_err_t*, %struct.mpc_err_t** %819, align 8
  %821 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %822 = load i32, i32* %10, align 4
  %823 = sext i32 %822 to i64
  %824 = getelementptr inbounds %union.mpc_result_t, %union.mpc_result_t* %821, i64 %823
  %825 = bitcast %union.mpc_result_t* %824 to %struct.mpc_err_t**
  %826 = load %struct.mpc_err_t*, %struct.mpc_err_t** %825, align 8
  %827 = call %struct.mpc_err_t* @mpc_err_merge(%struct.mpc_input_t* %818, %struct.mpc_err_t* %820, %struct.mpc_err_t* %826)
  %828 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %9, align 8
  store %struct.mpc_err_t* %827, %struct.mpc_err_t** %828, align 8
  br label %829

; <label>:829:                                    ; preds = %817
  br label %830

; <label>:830:                                    ; preds = %829
  %831 = load i32, i32* %10, align 4
  %832 = add nsw i32 %831, 1
  store i32 %832, i32* %10, align 4
  br label %771

; <label>:833:                                    ; preds = %771
  %834 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %835 = bitcast %union.mpc_result_t* %834 to %struct.mpc_err_t**
  store %struct.mpc_err_t* null, %struct.mpc_err_t** %835, align 8
  %836 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %837 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %836, i32 0, i32 3
  %838 = bitcast %union.mpc_pdata_t* %837 to %struct.mpc_pdata_or_t*
  %839 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %838, i32 0, i32 0
  %840 = load i32, i32* %839, align 8
  %841 = icmp sgt i32 %840, 4
  br i1 %841, label %842, label %846

; <label>:842:                                    ; preds = %833
  %843 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %844 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %845 = bitcast %union.mpc_result_t* %844 to i8*
  call void @mpc_free(%struct.mpc_input_t* %843, i8* %845)
  br label %846

; <label>:846:                                    ; preds = %842, %833
  store i32 0, i32* %5, align 4
  br label %987

; <label>:847:                                    ; preds = %4
  %848 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %849 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %848, i32 0, i32 3
  %850 = bitcast %union.mpc_pdata_t* %849 to %struct.mpc_pdata_and_t*
  %851 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %850, i32 0, i32 0
  %852 = load i32, i32* %851, align 8
  %853 = icmp eq i32 %852, 0
  br i1 %853, label %854, label %857

; <label>:854:                                    ; preds = %847
  %855 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %856 = bitcast %union.mpc_result_t* %855 to i8**
  store i8* null, i8** %856, align 8
  store i32 1, i32* %5, align 4
  br label %987

; <label>:857:                                    ; preds = %847
  %858 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %859 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %858, i32 0, i32 3
  %860 = bitcast %union.mpc_pdata_t* %859 to %struct.mpc_pdata_or_t*
  %861 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %860, i32 0, i32 0
  %862 = load i32, i32* %861, align 8
  %863 = icmp sgt i32 %862, 4
  br i1 %863, label %864, label %874

; <label>:864:                                    ; preds = %857
  %865 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %866 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %867 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %866, i32 0, i32 3
  %868 = bitcast %union.mpc_pdata_t* %867 to %struct.mpc_pdata_or_t*
  %869 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %868, i32 0, i32 0
  %870 = load i32, i32* %869, align 8
  %871 = sext i32 %870 to i64
  %872 = mul i64 8, %871
  %873 = call i8* @mpc_malloc(%struct.mpc_input_t* %865, i64 %872)
  br label %877

; <label>:874:                                    ; preds = %857
  %875 = getelementptr inbounds [4 x %union.mpc_result_t], [4 x %union.mpc_result_t]* %12, i32 0, i32 0
  %876 = bitcast %union.mpc_result_t* %875 to i8*
  br label %877

; <label>:877:                                    ; preds = %874, %864
  %878 = phi i8* [ %873, %864 ], [ %876, %874 ]
  %879 = bitcast i8* %878 to %union.mpc_result_t*
  store %union.mpc_result_t* %879, %union.mpc_result_t** %13, align 8
  %880 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  call void @mpc_input_mark(%struct.mpc_input_t* %880)
  store i32 0, i32* %10, align 4
  br label %881

; <label>:881:                                    ; preds = %954, %877
  %882 = load i32, i32* %10, align 4
  %883 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %884 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %883, i32 0, i32 3
  %885 = bitcast %union.mpc_pdata_t* %884 to %struct.mpc_pdata_and_t*
  %886 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %885, i32 0, i32 0
  %887 = load i32, i32* %886, align 8
  %888 = icmp slt i32 %882, %887
  br i1 %888, label %889, label %957

; <label>:889:                                    ; preds = %881
  %890 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %891 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %892 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %891, i32 0, i32 3
  %893 = bitcast %union.mpc_pdata_t* %892 to %struct.mpc_pdata_and_t*
  %894 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %893, i32 0, i32 2
  %895 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %894, align 8
  %896 = load i32, i32* %10, align 4
  %897 = sext i32 %896 to i64
  %898 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %895, i64 %897
  %899 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %898, align 8
  %900 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %901 = load i32, i32* %10, align 4
  %902 = sext i32 %901 to i64
  %903 = getelementptr inbounds %union.mpc_result_t, %union.mpc_result_t* %900, i64 %902
  %904 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %9, align 8
  %905 = call i32 @mpc_parse_run(%struct.mpc_input_t* %890, %struct.mpc_parser_t* %899, %union.mpc_result_t* %903, %struct.mpc_err_t** %904)
  %906 = icmp ne i32 %905, 0
  br i1 %906, label %953, label %907

; <label>:907:                                    ; preds = %889
  %908 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  call void @mpc_input_rewind(%struct.mpc_input_t* %908)
  store i32 0, i32* %11, align 4
  br label %909

; <label>:909:                                    ; preds = %930, %907
  %910 = load i32, i32* %11, align 4
  %911 = load i32, i32* %10, align 4
  %912 = icmp slt i32 %910, %911
  br i1 %912, label %913, label %933

; <label>:913:                                    ; preds = %909
  %914 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %915 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %916 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %915, i32 0, i32 3
  %917 = bitcast %union.mpc_pdata_t* %916 to %struct.mpc_pdata_and_t*
  %918 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %917, i32 0, i32 3
  %919 = load void (i8*)**, void (i8*)*** %918, align 8
  %920 = load i32, i32* %11, align 4
  %921 = sext i32 %920 to i64
  %922 = getelementptr inbounds void (i8*)*, void (i8*)** %919, i64 %921
  %923 = load void (i8*)*, void (i8*)** %922, align 8
  %924 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %925 = load i32, i32* %11, align 4
  %926 = sext i32 %925 to i64
  %927 = getelementptr inbounds %union.mpc_result_t, %union.mpc_result_t* %924, i64 %926
  %928 = bitcast %union.mpc_result_t* %927 to i8**
  %929 = load i8*, i8** %928, align 8
  call void @mpc_parse_dtor(%struct.mpc_input_t* %914, void (i8*)* %923, i8* %929)
  br label %930

; <label>:930:                                    ; preds = %913
  %931 = load i32, i32* %11, align 4
  %932 = add nsw i32 %931, 1
  store i32 %932, i32* %11, align 4
  br label %909

; <label>:933:                                    ; preds = %909
  %934 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %935 = load i32, i32* %10, align 4
  %936 = sext i32 %935 to i64
  %937 = getelementptr inbounds %union.mpc_result_t, %union.mpc_result_t* %934, i64 %936
  %938 = bitcast %union.mpc_result_t* %937 to %struct.mpc_err_t**
  %939 = load %struct.mpc_err_t*, %struct.mpc_err_t** %938, align 8
  %940 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %941 = bitcast %union.mpc_result_t* %940 to %struct.mpc_err_t**
  store %struct.mpc_err_t* %939, %struct.mpc_err_t** %941, align 8
  %942 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %943 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %942, i32 0, i32 3
  %944 = bitcast %union.mpc_pdata_t* %943 to %struct.mpc_pdata_or_t*
  %945 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %944, i32 0, i32 0
  %946 = load i32, i32* %945, align 8
  %947 = icmp sgt i32 %946, 4
  br i1 %947, label %948, label %952

; <label>:948:                                    ; preds = %933
  %949 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %950 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %951 = bitcast %union.mpc_result_t* %950 to i8*
  call void @mpc_free(%struct.mpc_input_t* %949, i8* %951)
  br label %952

; <label>:952:                                    ; preds = %948, %933
  store i32 0, i32* %5, align 4
  br label %987

; <label>:953:                                    ; preds = %889
  br label %954

; <label>:954:                                    ; preds = %953
  %955 = load i32, i32* %10, align 4
  %956 = add nsw i32 %955, 1
  store i32 %956, i32* %10, align 4
  br label %881

; <label>:957:                                    ; preds = %881
  %958 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  call void @mpc_input_unmark(%struct.mpc_input_t* %958)
  %959 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %960 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %961 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %960, i32 0, i32 3
  %962 = bitcast %union.mpc_pdata_t* %961 to %struct.mpc_pdata_and_t*
  %963 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %962, i32 0, i32 1
  %964 = load i8* (i32, i8**)*, i8* (i32, i8**)** %963, align 8
  %965 = load i32, i32* %10, align 4
  %966 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %967 = bitcast %union.mpc_result_t* %966 to i8**
  %968 = call i8* @mpc_parse_fold(%struct.mpc_input_t* %959, i8* (i32, i8**)* %964, i32 %965, i8** %967)
  %969 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %970 = bitcast %union.mpc_result_t* %969 to i8**
  store i8* %968, i8** %970, align 8
  %971 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %972 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %971, i32 0, i32 3
  %973 = bitcast %union.mpc_pdata_t* %972 to %struct.mpc_pdata_or_t*
  %974 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %973, i32 0, i32 0
  %975 = load i32, i32* %974, align 8
  %976 = icmp sgt i32 %975, 4
  br i1 %976, label %977, label %981

; <label>:977:                                    ; preds = %957
  %978 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %979 = load %union.mpc_result_t*, %union.mpc_result_t** %13, align 8
  %980 = bitcast %union.mpc_result_t* %979 to i8*
  call void @mpc_free(%struct.mpc_input_t* %978, i8* %980)
  br label %981

; <label>:981:                                    ; preds = %977, %957
  store i32 1, i32* %5, align 4
  br label %987

; <label>:982:                                    ; preds = %4
  %983 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %984 = call %struct.mpc_err_t* @mpc_err_fail(%struct.mpc_input_t* %983, i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str.103, i32 0, i32 0))
  %985 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %986 = bitcast %union.mpc_result_t* %985 to %struct.mpc_err_t**
  store %struct.mpc_err_t* %984, %struct.mpc_err_t** %986, align 8
  store i32 0, i32* %5, align 4
  br label %987

; <label>:987:                                    ; preds = %982, %981, %952, %854, %846, %816, %745, %737, %689, %608, %577, %501, %399, %393, %371, %355, %335, %328, %305, %298, %280, %262, %244, %231, %214, %206, %197, %187, %184, %179, %176, %170, %156, %150, %136, %130, %116, %110, %96, %90, %76, %70, %51, %45, %31, %25
  %988 = load i32, i32* %5, align 4
  ret i32 %988
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_err_delete_internal(%struct.mpc_input_t*, %struct.mpc_err_t*) #0 {
  %3 = alloca %struct.mpc_input_t*, align 8
  %4 = alloca %struct.mpc_err_t*, align 8
  %5 = alloca i32, align 4
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %3, align 8
  store %struct.mpc_err_t* %1, %struct.mpc_err_t** %4, align 8
  %6 = load %struct.mpc_err_t*, %struct.mpc_err_t** %4, align 8
  %7 = icmp ne %struct.mpc_err_t* %6, null
  br i1 %7, label %9, label %8

; <label>:8:                                      ; preds = %2
  br label %45

; <label>:9:                                      ; preds = %2
  store i32 0, i32* %5, align 4
  br label %10

; <label>:10:                                     ; preds = %25, %9
  %11 = load i32, i32* %5, align 4
  %12 = load %struct.mpc_err_t*, %struct.mpc_err_t** %4, align 8
  %13 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %12, i32 0, i32 1
  %14 = load i32, i32* %13, align 8
  %15 = icmp slt i32 %11, %14
  br i1 %15, label %16, label %28

; <label>:16:                                     ; preds = %10
  %17 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %18 = load %struct.mpc_err_t*, %struct.mpc_err_t** %4, align 8
  %19 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %18, i32 0, i32 4
  %20 = load i8**, i8*** %19, align 8
  %21 = load i32, i32* %5, align 4
  %22 = sext i32 %21 to i64
  %23 = getelementptr inbounds i8*, i8** %20, i64 %22
  %24 = load i8*, i8** %23, align 8
  call void @mpc_free(%struct.mpc_input_t* %17, i8* %24)
  br label %25

; <label>:25:                                     ; preds = %16
  %26 = load i32, i32* %5, align 4
  %27 = add nsw i32 %26, 1
  store i32 %27, i32* %5, align 4
  br label %10

; <label>:28:                                     ; preds = %10
  %29 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %30 = load %struct.mpc_err_t*, %struct.mpc_err_t** %4, align 8
  %31 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %30, i32 0, i32 4
  %32 = load i8**, i8*** %31, align 8
  %33 = bitcast i8** %32 to i8*
  call void @mpc_free(%struct.mpc_input_t* %29, i8* %33)
  %34 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %35 = load %struct.mpc_err_t*, %struct.mpc_err_t** %4, align 8
  %36 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %35, i32 0, i32 2
  %37 = load i8*, i8** %36, align 8
  call void @mpc_free(%struct.mpc_input_t* %34, i8* %37)
  %38 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %39 = load %struct.mpc_err_t*, %struct.mpc_err_t** %4, align 8
  %40 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %39, i32 0, i32 3
  %41 = load i8*, i8** %40, align 8
  call void @mpc_free(%struct.mpc_input_t* %38, i8* %41)
  %42 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %43 = load %struct.mpc_err_t*, %struct.mpc_err_t** %4, align 8
  %44 = bitcast %struct.mpc_err_t* %43 to i8*
  call void @mpc_free(%struct.mpc_input_t* %42, i8* %44)
  br label %45

; <label>:45:                                     ; preds = %28, %8
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpc_export(%struct.mpc_input_t*, i8*) #0 {
  %3 = alloca i8*, align 8
  %4 = alloca %struct.mpc_input_t*, align 8
  %5 = alloca i8*, align 8
  %6 = alloca i8*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %4, align 8
  store i8* %1, i8** %5, align 8
  store i8* null, i8** %6, align 8
  %7 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %8 = load i8*, i8** %5, align 8
  %9 = call i32 @mpc_mem_ptr(%struct.mpc_input_t* %7, i8* %8)
  %10 = icmp ne i32 %9, 0
  br i1 %10, label %13, label %11

; <label>:11:                                     ; preds = %2
  %12 = load i8*, i8** %5, align 8
  store i8* %12, i8** %3, align 8
  br label %20

; <label>:13:                                     ; preds = %2
  %14 = call noalias i8* @malloc(i64 64) #5
  store i8* %14, i8** %6, align 8
  %15 = load i8*, i8** %6, align 8
  %16 = load i8*, i8** %5, align 8
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %15, i8* %16, i64 64, i32 1, i1 false)
  %17 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %18 = load i8*, i8** %5, align 8
  call void @mpc_free(%struct.mpc_input_t* %17, i8* %18)
  %19 = load i8*, i8** %6, align 8
  store i8* %19, i8** %3, align 8
  br label %20

; <label>:20:                                     ; preds = %13, %11
  %21 = load i8*, i8** %3, align 8
  ret i8* %21
}

; Function Attrs: noinline nounwind optnone uwtable
define internal %struct.mpc_err_t* @mpc_err_export(%struct.mpc_input_t*, %struct.mpc_err_t*) #0 {
  %3 = alloca %struct.mpc_input_t*, align 8
  %4 = alloca %struct.mpc_err_t*, align 8
  %5 = alloca i32, align 4
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %3, align 8
  store %struct.mpc_err_t* %1, %struct.mpc_err_t** %4, align 8
  store i32 0, i32* %5, align 4
  br label %6

; <label>:6:                                      ; preds = %28, %2
  %7 = load i32, i32* %5, align 4
  %8 = load %struct.mpc_err_t*, %struct.mpc_err_t** %4, align 8
  %9 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %8, i32 0, i32 1
  %10 = load i32, i32* %9, align 8
  %11 = icmp slt i32 %7, %10
  br i1 %11, label %12, label %31

; <label>:12:                                     ; preds = %6
  %13 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %14 = load %struct.mpc_err_t*, %struct.mpc_err_t** %4, align 8
  %15 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %14, i32 0, i32 4
  %16 = load i8**, i8*** %15, align 8
  %17 = load i32, i32* %5, align 4
  %18 = sext i32 %17 to i64
  %19 = getelementptr inbounds i8*, i8** %16, i64 %18
  %20 = load i8*, i8** %19, align 8
  %21 = call i8* @mpc_export(%struct.mpc_input_t* %13, i8* %20)
  %22 = load %struct.mpc_err_t*, %struct.mpc_err_t** %4, align 8
  %23 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %22, i32 0, i32 4
  %24 = load i8**, i8*** %23, align 8
  %25 = load i32, i32* %5, align 4
  %26 = sext i32 %25 to i64
  %27 = getelementptr inbounds i8*, i8** %24, i64 %26
  store i8* %21, i8** %27, align 8
  br label %28

; <label>:28:                                     ; preds = %12
  %29 = load i32, i32* %5, align 4
  %30 = add nsw i32 %29, 1
  store i32 %30, i32* %5, align 4
  br label %6

; <label>:31:                                     ; preds = %6
  %32 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %33 = load %struct.mpc_err_t*, %struct.mpc_err_t** %4, align 8
  %34 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %33, i32 0, i32 4
  %35 = load i8**, i8*** %34, align 8
  %36 = bitcast i8** %35 to i8*
  %37 = call i8* @mpc_export(%struct.mpc_input_t* %32, i8* %36)
  %38 = bitcast i8* %37 to i8**
  %39 = load %struct.mpc_err_t*, %struct.mpc_err_t** %4, align 8
  %40 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %39, i32 0, i32 4
  store i8** %38, i8*** %40, align 8
  %41 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %42 = load %struct.mpc_err_t*, %struct.mpc_err_t** %4, align 8
  %43 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %42, i32 0, i32 2
  %44 = load i8*, i8** %43, align 8
  %45 = call i8* @mpc_export(%struct.mpc_input_t* %41, i8* %44)
  %46 = load %struct.mpc_err_t*, %struct.mpc_err_t** %4, align 8
  %47 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %46, i32 0, i32 2
  store i8* %45, i8** %47, align 8
  %48 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %49 = load %struct.mpc_err_t*, %struct.mpc_err_t** %4, align 8
  %50 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %49, i32 0, i32 3
  %51 = load i8*, i8** %50, align 8
  %52 = call i8* @mpc_export(%struct.mpc_input_t* %48, i8* %51)
  %53 = load %struct.mpc_err_t*, %struct.mpc_err_t** %4, align 8
  %54 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %53, i32 0, i32 3
  store i8* %52, i8** %54, align 8
  %55 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %56 = load %struct.mpc_err_t*, %struct.mpc_err_t** %4, align 8
  %57 = bitcast %struct.mpc_err_t* %56 to i8*
  %58 = call i8* @mpc_export(%struct.mpc_input_t* %55, i8* %57)
  %59 = bitcast i8* %58 to %struct.mpc_err_t*
  ret %struct.mpc_err_t* %59
}

; Function Attrs: noinline nounwind optnone uwtable
define internal %struct.mpc_err_t* @mpc_err_merge(%struct.mpc_input_t*, %struct.mpc_err_t*, %struct.mpc_err_t*) #0 {
  %4 = alloca %struct.mpc_input_t*, align 8
  %5 = alloca %struct.mpc_err_t*, align 8
  %6 = alloca %struct.mpc_err_t*, align 8
  %7 = alloca [2 x %struct.mpc_err_t*], align 16
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %4, align 8
  store %struct.mpc_err_t* %1, %struct.mpc_err_t** %5, align 8
  store %struct.mpc_err_t* %2, %struct.mpc_err_t** %6, align 8
  %8 = load %struct.mpc_err_t*, %struct.mpc_err_t** %5, align 8
  %9 = getelementptr inbounds [2 x %struct.mpc_err_t*], [2 x %struct.mpc_err_t*]* %7, i64 0, i64 0
  store %struct.mpc_err_t* %8, %struct.mpc_err_t** %9, align 16
  %10 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %11 = getelementptr inbounds [2 x %struct.mpc_err_t*], [2 x %struct.mpc_err_t*]* %7, i64 0, i64 1
  store %struct.mpc_err_t* %10, %struct.mpc_err_t** %11, align 8
  %12 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %13 = getelementptr inbounds [2 x %struct.mpc_err_t*], [2 x %struct.mpc_err_t*]* %7, i32 0, i32 0
  %14 = call %struct.mpc_err_t* @mpc_err_or(%struct.mpc_input_t* %12, %struct.mpc_err_t** %13, i32 2)
  ret %struct.mpc_err_t* %14
}

; Function Attrs: noinline nounwind optnone uwtable
define i32 @mpc_parse(i8*, i8*, %struct.mpc_parser_t*, %union.mpc_result_t*) #0 {
  %5 = alloca i8*, align 8
  %6 = alloca i8*, align 8
  %7 = alloca %struct.mpc_parser_t*, align 8
  %8 = alloca %union.mpc_result_t*, align 8
  %9 = alloca i32, align 4
  %10 = alloca %struct.mpc_input_t*, align 8
  store i8* %0, i8** %5, align 8
  store i8* %1, i8** %6, align 8
  store %struct.mpc_parser_t* %2, %struct.mpc_parser_t** %7, align 8
  store %union.mpc_result_t* %3, %union.mpc_result_t** %8, align 8
  %11 = load i8*, i8** %5, align 8
  %12 = load i8*, i8** %6, align 8
  %13 = call %struct.mpc_input_t* @mpc_input_new_string(i8* %11, i8* %12)
  store %struct.mpc_input_t* %13, %struct.mpc_input_t** %10, align 8
  %14 = load %struct.mpc_input_t*, %struct.mpc_input_t** %10, align 8
  %15 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %16 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %17 = call i32 @mpc_parse_input(%struct.mpc_input_t* %14, %struct.mpc_parser_t* %15, %union.mpc_result_t* %16)
  store i32 %17, i32* %9, align 4
  %18 = load %struct.mpc_input_t*, %struct.mpc_input_t** %10, align 8
  call void @mpc_input_delete(%struct.mpc_input_t* %18)
  %19 = load i32, i32* %9, align 4
  ret i32 %19
}

; Function Attrs: noinline nounwind optnone uwtable
define internal %struct.mpc_input_t* @mpc_input_new_string(i8*, i8*) #0 {
  %3 = alloca i8*, align 8
  %4 = alloca i8*, align 8
  %5 = alloca %struct.mpc_input_t*, align 8
  %6 = alloca %struct.mpc_state_t, align 8
  store i8* %0, i8** %3, align 8
  store i8* %1, i8** %4, align 8
  %7 = call noalias i8* @malloc(i64 33392) #5
  %8 = bitcast i8* %7 to %struct.mpc_input_t*
  store %struct.mpc_input_t* %8, %struct.mpc_input_t** %5, align 8
  %9 = load i8*, i8** %3, align 8
  %10 = call i64 @strlen(i8* %9) #7
  %11 = add i64 %10, 1
  %12 = call noalias i8* @malloc(i64 %11) #5
  %13 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %14 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %13, i32 0, i32 1
  store i8* %12, i8** %14, align 8
  %15 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %16 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %15, i32 0, i32 1
  %17 = load i8*, i8** %16, align 8
  %18 = load i8*, i8** %3, align 8
  %19 = call i8* @strcpy(i8* %17, i8* %18) #5
  %20 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %21 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %20, i32 0, i32 0
  store i32 0, i32* %21, align 8
  %22 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %23 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %22, i32 0, i32 2
  call void @mpc_state_new(%struct.mpc_state_t* sret %6)
  %24 = bitcast %struct.mpc_state_t* %23 to i8*
  %25 = bitcast %struct.mpc_state_t* %6 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %24, i8* %25, i64 24, i32 8, i1 false)
  %26 = load i8*, i8** %4, align 8
  %27 = call i64 @strlen(i8* %26) #7
  %28 = add i64 %27, 1
  %29 = call noalias i8* @malloc(i64 %28) #5
  %30 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %31 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %30, i32 0, i32 3
  store i8* %29, i8** %31, align 8
  %32 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %33 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %32, i32 0, i32 3
  %34 = load i8*, i8** %33, align 8
  %35 = load i8*, i8** %4, align 8
  %36 = call i8* @strcpy(i8* %34, i8* %35) #5
  %37 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %38 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %37, i32 0, i32 4
  store i8* null, i8** %38, align 8
  %39 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %40 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %39, i32 0, i32 5
  store %struct._IO_FILE* null, %struct._IO_FILE** %40, align 8
  %41 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %42 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %41, i32 0, i32 6
  store i32 0, i32* %42, align 8
  %43 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %44 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %43, i32 0, i32 7
  store i32 1, i32* %44, align 4
  %45 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %46 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %45, i32 0, i32 9
  store i32 0, i32* %46, align 4
  %47 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %48 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %47, i32 0, i32 8
  store i32 32, i32* %48, align 8
  %49 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %50 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %49, i32 0, i32 8
  %51 = load i32, i32* %50, align 8
  %52 = sext i32 %51 to i64
  %53 = mul i64 24, %52
  %54 = call noalias i8* @malloc(i64 %53) #5
  %55 = bitcast i8* %54 to %struct.mpc_state_t*
  %56 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %57 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %56, i32 0, i32 10
  store %struct.mpc_state_t* %55, %struct.mpc_state_t** %57, align 8
  %58 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %59 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %58, i32 0, i32 8
  %60 = load i32, i32* %59, align 8
  %61 = sext i32 %60 to i64
  %62 = mul i64 1, %61
  %63 = call noalias i8* @malloc(i64 %62) #5
  %64 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %65 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %64, i32 0, i32 11
  store i8* %63, i8** %65, align 8
  %66 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %67 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %66, i32 0, i32 12
  store i8 0, i8* %67, align 8
  %68 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %69 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %68, i32 0, i32 13
  store i64 0, i64* %69, align 8
  %70 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %71 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %70, i32 0, i32 14
  %72 = getelementptr inbounds [512 x i8], [512 x i8]* %71, i32 0, i32 0
  call void @llvm.memset.p0i8.i64(i8* %72, i8 0, i64 512, i32 8, i1 false)
  %73 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  ret %struct.mpc_input_t* %73
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_input_delete(%struct.mpc_input_t*) #0 {
  %2 = alloca %struct.mpc_input_t*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %2, align 8
  %3 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %4 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %3, i32 0, i32 1
  %5 = load i8*, i8** %4, align 8
  call void @free(i8* %5) #5
  %6 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %7 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %6, i32 0, i32 0
  %8 = load i32, i32* %7, align 8
  %9 = icmp eq i32 %8, 0
  br i1 %9, label %10, label %14

; <label>:10:                                     ; preds = %1
  %11 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %12 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %11, i32 0, i32 3
  %13 = load i8*, i8** %12, align 8
  call void @free(i8* %13) #5
  br label %14

; <label>:14:                                     ; preds = %10, %1
  %15 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %16 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %15, i32 0, i32 0
  %17 = load i32, i32* %16, align 8
  %18 = icmp eq i32 %17, 2
  br i1 %18, label %19, label %23

; <label>:19:                                     ; preds = %14
  %20 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %21 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %20, i32 0, i32 4
  %22 = load i8*, i8** %21, align 8
  call void @free(i8* %22) #5
  br label %23

; <label>:23:                                     ; preds = %19, %14
  %24 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %25 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %24, i32 0, i32 10
  %26 = load %struct.mpc_state_t*, %struct.mpc_state_t** %25, align 8
  %27 = bitcast %struct.mpc_state_t* %26 to i8*
  call void @free(i8* %27) #5
  %28 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %29 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %28, i32 0, i32 11
  %30 = load i8*, i8** %29, align 8
  call void @free(i8* %30) #5
  %31 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %32 = bitcast %struct.mpc_input_t* %31 to i8*
  call void @free(i8* %32) #5
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define i32 @mpc_parse_file(i8*, %struct._IO_FILE*, %struct.mpc_parser_t*, %union.mpc_result_t*) #0 {
  %5 = alloca i8*, align 8
  %6 = alloca %struct._IO_FILE*, align 8
  %7 = alloca %struct.mpc_parser_t*, align 8
  %8 = alloca %union.mpc_result_t*, align 8
  %9 = alloca i32, align 4
  %10 = alloca %struct.mpc_input_t*, align 8
  store i8* %0, i8** %5, align 8
  store %struct._IO_FILE* %1, %struct._IO_FILE** %6, align 8
  store %struct.mpc_parser_t* %2, %struct.mpc_parser_t** %7, align 8
  store %union.mpc_result_t* %3, %union.mpc_result_t** %8, align 8
  %11 = load i8*, i8** %5, align 8
  %12 = load %struct._IO_FILE*, %struct._IO_FILE** %6, align 8
  %13 = call %struct.mpc_input_t* @mpc_input_new_file(i8* %11, %struct._IO_FILE* %12)
  store %struct.mpc_input_t* %13, %struct.mpc_input_t** %10, align 8
  %14 = load %struct.mpc_input_t*, %struct.mpc_input_t** %10, align 8
  %15 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %16 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %17 = call i32 @mpc_parse_input(%struct.mpc_input_t* %14, %struct.mpc_parser_t* %15, %union.mpc_result_t* %16)
  store i32 %17, i32* %9, align 4
  %18 = load %struct.mpc_input_t*, %struct.mpc_input_t** %10, align 8
  call void @mpc_input_delete(%struct.mpc_input_t* %18)
  %19 = load i32, i32* %9, align 4
  ret i32 %19
}

; Function Attrs: noinline nounwind optnone uwtable
define internal %struct.mpc_input_t* @mpc_input_new_file(i8*, %struct._IO_FILE*) #0 {
  %3 = alloca i8*, align 8
  %4 = alloca %struct._IO_FILE*, align 8
  %5 = alloca %struct.mpc_input_t*, align 8
  %6 = alloca %struct.mpc_state_t, align 8
  store i8* %0, i8** %3, align 8
  store %struct._IO_FILE* %1, %struct._IO_FILE** %4, align 8
  %7 = call noalias i8* @malloc(i64 33392) #5
  %8 = bitcast i8* %7 to %struct.mpc_input_t*
  store %struct.mpc_input_t* %8, %struct.mpc_input_t** %5, align 8
  %9 = load i8*, i8** %3, align 8
  %10 = call i64 @strlen(i8* %9) #7
  %11 = add i64 %10, 1
  %12 = call noalias i8* @malloc(i64 %11) #5
  %13 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %14 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %13, i32 0, i32 1
  store i8* %12, i8** %14, align 8
  %15 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %16 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %15, i32 0, i32 1
  %17 = load i8*, i8** %16, align 8
  %18 = load i8*, i8** %3, align 8
  %19 = call i8* @strcpy(i8* %17, i8* %18) #5
  %20 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %21 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %20, i32 0, i32 0
  store i32 1, i32* %21, align 8
  %22 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %23 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %22, i32 0, i32 2
  call void @mpc_state_new(%struct.mpc_state_t* sret %6)
  %24 = bitcast %struct.mpc_state_t* %23 to i8*
  %25 = bitcast %struct.mpc_state_t* %6 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %24, i8* %25, i64 24, i32 8, i1 false)
  %26 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %27 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %26, i32 0, i32 3
  store i8* null, i8** %27, align 8
  %28 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %29 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %28, i32 0, i32 4
  store i8* null, i8** %29, align 8
  %30 = load %struct._IO_FILE*, %struct._IO_FILE** %4, align 8
  %31 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %32 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %31, i32 0, i32 5
  store %struct._IO_FILE* %30, %struct._IO_FILE** %32, align 8
  %33 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %34 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %33, i32 0, i32 6
  store i32 0, i32* %34, align 8
  %35 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %36 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %35, i32 0, i32 7
  store i32 1, i32* %36, align 4
  %37 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %38 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %37, i32 0, i32 9
  store i32 0, i32* %38, align 4
  %39 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %40 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %39, i32 0, i32 8
  store i32 32, i32* %40, align 8
  %41 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %42 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %41, i32 0, i32 8
  %43 = load i32, i32* %42, align 8
  %44 = sext i32 %43 to i64
  %45 = mul i64 24, %44
  %46 = call noalias i8* @malloc(i64 %45) #5
  %47 = bitcast i8* %46 to %struct.mpc_state_t*
  %48 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %49 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %48, i32 0, i32 10
  store %struct.mpc_state_t* %47, %struct.mpc_state_t** %49, align 8
  %50 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %51 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %50, i32 0, i32 8
  %52 = load i32, i32* %51, align 8
  %53 = sext i32 %52 to i64
  %54 = mul i64 1, %53
  %55 = call noalias i8* @malloc(i64 %54) #5
  %56 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %57 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %56, i32 0, i32 11
  store i8* %55, i8** %57, align 8
  %58 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %59 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %58, i32 0, i32 12
  store i8 0, i8* %59, align 8
  %60 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %61 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %60, i32 0, i32 13
  store i64 0, i64* %61, align 8
  %62 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %63 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %62, i32 0, i32 14
  %64 = getelementptr inbounds [512 x i8], [512 x i8]* %63, i32 0, i32 0
  call void @llvm.memset.p0i8.i64(i8* %64, i8 0, i64 512, i32 8, i1 false)
  %65 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  ret %struct.mpc_input_t* %65
}

; Function Attrs: noinline nounwind optnone uwtable
define i32 @mpc_parse_pipe(i8*, %struct._IO_FILE*, %struct.mpc_parser_t*, %union.mpc_result_t*) #0 {
  %5 = alloca i8*, align 8
  %6 = alloca %struct._IO_FILE*, align 8
  %7 = alloca %struct.mpc_parser_t*, align 8
  %8 = alloca %union.mpc_result_t*, align 8
  %9 = alloca i32, align 4
  %10 = alloca %struct.mpc_input_t*, align 8
  store i8* %0, i8** %5, align 8
  store %struct._IO_FILE* %1, %struct._IO_FILE** %6, align 8
  store %struct.mpc_parser_t* %2, %struct.mpc_parser_t** %7, align 8
  store %union.mpc_result_t* %3, %union.mpc_result_t** %8, align 8
  %11 = load i8*, i8** %5, align 8
  %12 = load %struct._IO_FILE*, %struct._IO_FILE** %6, align 8
  %13 = call %struct.mpc_input_t* @mpc_input_new_pipe(i8* %11, %struct._IO_FILE* %12)
  store %struct.mpc_input_t* %13, %struct.mpc_input_t** %10, align 8
  %14 = load %struct.mpc_input_t*, %struct.mpc_input_t** %10, align 8
  %15 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %16 = load %union.mpc_result_t*, %union.mpc_result_t** %8, align 8
  %17 = call i32 @mpc_parse_input(%struct.mpc_input_t* %14, %struct.mpc_parser_t* %15, %union.mpc_result_t* %16)
  store i32 %17, i32* %9, align 4
  %18 = load %struct.mpc_input_t*, %struct.mpc_input_t** %10, align 8
  call void @mpc_input_delete(%struct.mpc_input_t* %18)
  %19 = load i32, i32* %9, align 4
  ret i32 %19
}

; Function Attrs: noinline nounwind optnone uwtable
define internal %struct.mpc_input_t* @mpc_input_new_pipe(i8*, %struct._IO_FILE*) #0 {
  %3 = alloca i8*, align 8
  %4 = alloca %struct._IO_FILE*, align 8
  %5 = alloca %struct.mpc_input_t*, align 8
  %6 = alloca %struct.mpc_state_t, align 8
  store i8* %0, i8** %3, align 8
  store %struct._IO_FILE* %1, %struct._IO_FILE** %4, align 8
  %7 = call noalias i8* @malloc(i64 33392) #5
  %8 = bitcast i8* %7 to %struct.mpc_input_t*
  store %struct.mpc_input_t* %8, %struct.mpc_input_t** %5, align 8
  %9 = load i8*, i8** %3, align 8
  %10 = call i64 @strlen(i8* %9) #7
  %11 = add i64 %10, 1
  %12 = call noalias i8* @malloc(i64 %11) #5
  %13 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %14 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %13, i32 0, i32 1
  store i8* %12, i8** %14, align 8
  %15 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %16 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %15, i32 0, i32 1
  %17 = load i8*, i8** %16, align 8
  %18 = load i8*, i8** %3, align 8
  %19 = call i8* @strcpy(i8* %17, i8* %18) #5
  %20 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %21 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %20, i32 0, i32 0
  store i32 2, i32* %21, align 8
  %22 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %23 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %22, i32 0, i32 2
  call void @mpc_state_new(%struct.mpc_state_t* sret %6)
  %24 = bitcast %struct.mpc_state_t* %23 to i8*
  %25 = bitcast %struct.mpc_state_t* %6 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %24, i8* %25, i64 24, i32 8, i1 false)
  %26 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %27 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %26, i32 0, i32 3
  store i8* null, i8** %27, align 8
  %28 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %29 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %28, i32 0, i32 4
  store i8* null, i8** %29, align 8
  %30 = load %struct._IO_FILE*, %struct._IO_FILE** %4, align 8
  %31 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %32 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %31, i32 0, i32 5
  store %struct._IO_FILE* %30, %struct._IO_FILE** %32, align 8
  %33 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %34 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %33, i32 0, i32 6
  store i32 0, i32* %34, align 8
  %35 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %36 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %35, i32 0, i32 7
  store i32 1, i32* %36, align 4
  %37 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %38 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %37, i32 0, i32 9
  store i32 0, i32* %38, align 4
  %39 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %40 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %39, i32 0, i32 8
  store i32 32, i32* %40, align 8
  %41 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %42 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %41, i32 0, i32 8
  %43 = load i32, i32* %42, align 8
  %44 = sext i32 %43 to i64
  %45 = mul i64 24, %44
  %46 = call noalias i8* @malloc(i64 %45) #5
  %47 = bitcast i8* %46 to %struct.mpc_state_t*
  %48 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %49 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %48, i32 0, i32 10
  store %struct.mpc_state_t* %47, %struct.mpc_state_t** %49, align 8
  %50 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %51 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %50, i32 0, i32 8
  %52 = load i32, i32* %51, align 8
  %53 = sext i32 %52 to i64
  %54 = mul i64 1, %53
  %55 = call noalias i8* @malloc(i64 %54) #5
  %56 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %57 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %56, i32 0, i32 11
  store i8* %55, i8** %57, align 8
  %58 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %59 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %58, i32 0, i32 12
  store i8 0, i8* %59, align 8
  %60 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %61 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %60, i32 0, i32 13
  store i64 0, i64* %61, align 8
  %62 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %63 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %62, i32 0, i32 14
  %64 = getelementptr inbounds [512 x i8], [512 x i8]* %63, i32 0, i32 0
  call void @llvm.memset.p0i8.i64(i8* %64, i8 0, i64 512, i32 8, i1 false)
  %65 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  ret %struct.mpc_input_t* %65
}

; Function Attrs: noinline nounwind optnone uwtable
define i32 @mpc_parse_contents(i8*, %struct.mpc_parser_t*, %union.mpc_result_t*) #0 {
  %4 = alloca i32, align 4
  %5 = alloca i8*, align 8
  %6 = alloca %struct.mpc_parser_t*, align 8
  %7 = alloca %union.mpc_result_t*, align 8
  %8 = alloca %struct._IO_FILE*, align 8
  %9 = alloca i32, align 4
  store i8* %0, i8** %5, align 8
  store %struct.mpc_parser_t* %1, %struct.mpc_parser_t** %6, align 8
  store %union.mpc_result_t* %2, %union.mpc_result_t** %7, align 8
  %10 = load i8*, i8** %5, align 8
  %11 = call %struct._IO_FILE* @fopen(i8* %10, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.9, i32 0, i32 0))
  store %struct._IO_FILE* %11, %struct._IO_FILE** %8, align 8
  %12 = load %struct._IO_FILE*, %struct._IO_FILE** %8, align 8
  %13 = icmp ne %struct._IO_FILE* %12, null
  br i1 %13, label %21, label %14

; <label>:14:                                     ; preds = %3
  %15 = load %union.mpc_result_t*, %union.mpc_result_t** %7, align 8
  %16 = bitcast %union.mpc_result_t* %15 to i8**
  store i8* null, i8** %16, align 8
  %17 = load i8*, i8** %5, align 8
  %18 = call %struct.mpc_err_t* @mpc_err_file(i8* %17, i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.10, i32 0, i32 0))
  %19 = load %union.mpc_result_t*, %union.mpc_result_t** %7, align 8
  %20 = bitcast %union.mpc_result_t* %19 to %struct.mpc_err_t**
  store %struct.mpc_err_t* %18, %struct.mpc_err_t** %20, align 8
  store i32 0, i32* %4, align 4
  br label %30

; <label>:21:                                     ; preds = %3
  %22 = load i8*, i8** %5, align 8
  %23 = load %struct._IO_FILE*, %struct._IO_FILE** %8, align 8
  %24 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %6, align 8
  %25 = load %union.mpc_result_t*, %union.mpc_result_t** %7, align 8
  %26 = call i32 @mpc_parse_file(i8* %22, %struct._IO_FILE* %23, %struct.mpc_parser_t* %24, %union.mpc_result_t* %25)
  store i32 %26, i32* %9, align 4
  %27 = load %struct._IO_FILE*, %struct._IO_FILE** %8, align 8
  %28 = call i32 @fclose(%struct._IO_FILE* %27)
  %29 = load i32, i32* %9, align 4
  store i32 %29, i32* %4, align 4
  br label %30

; <label>:30:                                     ; preds = %21, %14
  %31 = load i32, i32* %4, align 4
  ret i32 %31
}

declare %struct._IO_FILE* @fopen(i8*, i8*) #2

; Function Attrs: noinline nounwind optnone uwtable
define internal %struct.mpc_err_t* @mpc_err_file(i8*, i8*) #0 {
  %3 = alloca i8*, align 8
  %4 = alloca i8*, align 8
  %5 = alloca %struct.mpc_err_t*, align 8
  %6 = alloca %struct.mpc_state_t, align 8
  store i8* %0, i8** %3, align 8
  store i8* %1, i8** %4, align 8
  %7 = call noalias i8* @malloc(i64 64) #5
  %8 = bitcast i8* %7 to %struct.mpc_err_t*
  store %struct.mpc_err_t* %8, %struct.mpc_err_t** %5, align 8
  %9 = load i8*, i8** %3, align 8
  %10 = call i64 @strlen(i8* %9) #7
  %11 = add i64 %10, 1
  %12 = call noalias i8* @malloc(i64 %11) #5
  %13 = load %struct.mpc_err_t*, %struct.mpc_err_t** %5, align 8
  %14 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %13, i32 0, i32 2
  store i8* %12, i8** %14, align 8
  %15 = load %struct.mpc_err_t*, %struct.mpc_err_t** %5, align 8
  %16 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %15, i32 0, i32 2
  %17 = load i8*, i8** %16, align 8
  %18 = load i8*, i8** %3, align 8
  %19 = call i8* @strcpy(i8* %17, i8* %18) #5
  %20 = load %struct.mpc_err_t*, %struct.mpc_err_t** %5, align 8
  %21 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %20, i32 0, i32 0
  call void @mpc_state_new(%struct.mpc_state_t* sret %6)
  %22 = bitcast %struct.mpc_state_t* %21 to i8*
  %23 = bitcast %struct.mpc_state_t* %6 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %22, i8* %23, i64 24, i32 8, i1 false)
  %24 = load %struct.mpc_err_t*, %struct.mpc_err_t** %5, align 8
  %25 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %24, i32 0, i32 1
  store i32 0, i32* %25, align 8
  %26 = load %struct.mpc_err_t*, %struct.mpc_err_t** %5, align 8
  %27 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %26, i32 0, i32 4
  store i8** null, i8*** %27, align 8
  %28 = load i8*, i8** %4, align 8
  %29 = call i64 @strlen(i8* %28) #7
  %30 = add i64 %29, 1
  %31 = call noalias i8* @malloc(i64 %30) #5
  %32 = load %struct.mpc_err_t*, %struct.mpc_err_t** %5, align 8
  %33 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %32, i32 0, i32 3
  store i8* %31, i8** %33, align 8
  %34 = load %struct.mpc_err_t*, %struct.mpc_err_t** %5, align 8
  %35 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %34, i32 0, i32 3
  %36 = load i8*, i8** %35, align 8
  %37 = load i8*, i8** %4, align 8
  %38 = call i8* @strcpy(i8* %36, i8* %37) #5
  %39 = load %struct.mpc_err_t*, %struct.mpc_err_t** %5, align 8
  %40 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %39, i32 0, i32 5
  store i8 32, i8* %40, align 8
  %41 = load %struct.mpc_err_t*, %struct.mpc_err_t** %5, align 8
  ret %struct.mpc_err_t* %41
}

declare i32 @fclose(%struct._IO_FILE*) #2

; Function Attrs: noinline nounwind optnone uwtable
define void @mpc_delete(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  %3 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %4 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %3, i32 0, i32 0
  %5 = load i8, i8* %4, align 8
  %6 = icmp ne i8 %5, 0
  br i1 %6, label %7, label %21

; <label>:7:                                      ; preds = %1
  %8 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %9 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %8, i32 0, i32 2
  %10 = load i8, i8* %9, align 8
  %11 = sext i8 %10 to i32
  %12 = icmp ne i32 %11, 0
  br i1 %12, label %13, label %15

; <label>:13:                                     ; preds = %7
  %14 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  call void @mpc_undefine_unretained(%struct.mpc_parser_t* %14, i32 0)
  br label %15

; <label>:15:                                     ; preds = %13, %7
  %16 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %17 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %16, i32 0, i32 1
  %18 = load i8*, i8** %17, align 8
  call void @free(i8* %18) #5
  %19 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %20 = bitcast %struct.mpc_parser_t* %19 to i8*
  call void @free(i8* %20) #5
  br label %23

; <label>:21:                                     ; preds = %1
  %22 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  call void @mpc_undefine_unretained(%struct.mpc_parser_t* %22, i32 0)
  br label %23

; <label>:23:                                     ; preds = %21, %15
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_undefine_unretained(%struct.mpc_parser_t*, i32) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca i32, align 4
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store i32 %1, i32* %4, align 4
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %5, i32 0, i32 0
  %7 = load i8, i8* %6, align 8
  %8 = sext i8 %7 to i32
  %9 = icmp ne i32 %8, 0
  br i1 %9, label %10, label %14

; <label>:10:                                     ; preds = %2
  %11 = load i32, i32* %4, align 4
  %12 = icmp ne i32 %11, 0
  br i1 %12, label %14, label %13

; <label>:13:                                     ; preds = %10
  br label %86

; <label>:14:                                     ; preds = %10, %2
  %15 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %16 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %15, i32 0, i32 2
  %17 = load i8, i8* %16, align 8
  %18 = sext i8 %17 to i32
  switch i32 %18, label %76 [
    i32 2, label %19
    i32 10, label %25
    i32 11, label %25
    i32 14, label %25
    i32 15, label %31
    i32 16, label %37
    i32 17, label %43
    i32 19, label %49
    i32 18, label %49
    i32 5, label %55
    i32 20, label %66
    i32 21, label %66
    i32 22, label %66
    i32 23, label %72
    i32 24, label %74
  ]

; <label>:19:                                     ; preds = %14
  %20 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %21 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %20, i32 0, i32 3
  %22 = bitcast %union.mpc_pdata_t* %21 to %struct.mpc_pdata_fail_t*
  %23 = getelementptr inbounds %struct.mpc_pdata_fail_t, %struct.mpc_pdata_fail_t* %22, i32 0, i32 0
  %24 = load i8*, i8** %23, align 8
  call void @free(i8* %24) #5
  br label %77

; <label>:25:                                     ; preds = %14, %14, %14
  %26 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %27 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %26, i32 0, i32 3
  %28 = bitcast %union.mpc_pdata_t* %27 to %struct.mpc_pdata_string_t*
  %29 = getelementptr inbounds %struct.mpc_pdata_string_t, %struct.mpc_pdata_string_t* %28, i32 0, i32 0
  %30 = load i8*, i8** %29, align 8
  call void @free(i8* %30) #5
  br label %77

; <label>:31:                                     ; preds = %14
  %32 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %33 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %32, i32 0, i32 3
  %34 = bitcast %union.mpc_pdata_t* %33 to %struct.mpc_pdata_apply_t*
  %35 = getelementptr inbounds %struct.mpc_pdata_apply_t, %struct.mpc_pdata_apply_t* %34, i32 0, i32 0
  %36 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %35, align 8
  call void @mpc_undefine_unretained(%struct.mpc_parser_t* %36, i32 0)
  br label %77

; <label>:37:                                     ; preds = %14
  %38 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %39 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %38, i32 0, i32 3
  %40 = bitcast %union.mpc_pdata_t* %39 to %struct.mpc_pdata_apply_to_t*
  %41 = getelementptr inbounds %struct.mpc_pdata_apply_to_t, %struct.mpc_pdata_apply_to_t* %40, i32 0, i32 0
  %42 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %41, align 8
  call void @mpc_undefine_unretained(%struct.mpc_parser_t* %42, i32 0)
  br label %77

; <label>:43:                                     ; preds = %14
  %44 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %45 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %44, i32 0, i32 3
  %46 = bitcast %union.mpc_pdata_t* %45 to %struct.mpc_pdata_predict_t*
  %47 = getelementptr inbounds %struct.mpc_pdata_predict_t, %struct.mpc_pdata_predict_t* %46, i32 0, i32 0
  %48 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %47, align 8
  call void @mpc_undefine_unretained(%struct.mpc_parser_t* %48, i32 0)
  br label %77

; <label>:49:                                     ; preds = %14, %14
  %50 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %51 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %50, i32 0, i32 3
  %52 = bitcast %union.mpc_pdata_t* %51 to %struct.mpc_pdata_not_t*
  %53 = getelementptr inbounds %struct.mpc_pdata_not_t, %struct.mpc_pdata_not_t* %52, i32 0, i32 0
  %54 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %53, align 8
  call void @mpc_undefine_unretained(%struct.mpc_parser_t* %54, i32 0)
  br label %77

; <label>:55:                                     ; preds = %14
  %56 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %57 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %56, i32 0, i32 3
  %58 = bitcast %union.mpc_pdata_t* %57 to %struct.mpc_pdata_expect_t*
  %59 = getelementptr inbounds %struct.mpc_pdata_expect_t, %struct.mpc_pdata_expect_t* %58, i32 0, i32 0
  %60 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %59, align 8
  call void @mpc_undefine_unretained(%struct.mpc_parser_t* %60, i32 0)
  %61 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %62 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %61, i32 0, i32 3
  %63 = bitcast %union.mpc_pdata_t* %62 to %struct.mpc_pdata_expect_t*
  %64 = getelementptr inbounds %struct.mpc_pdata_expect_t, %struct.mpc_pdata_expect_t* %63, i32 0, i32 1
  %65 = load i8*, i8** %64, align 8
  call void @free(i8* %65) #5
  br label %77

; <label>:66:                                     ; preds = %14, %14, %14
  %67 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %68 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %67, i32 0, i32 3
  %69 = bitcast %union.mpc_pdata_t* %68 to %struct.mpc_pdata_repeat_t*
  %70 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %69, i32 0, i32 2
  %71 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %70, align 8
  call void @mpc_undefine_unretained(%struct.mpc_parser_t* %71, i32 0)
  br label %77

; <label>:72:                                     ; preds = %14
  %73 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  call void @mpc_undefine_or(%struct.mpc_parser_t* %73)
  br label %77

; <label>:74:                                     ; preds = %14
  %75 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  call void @mpc_undefine_and(%struct.mpc_parser_t* %75)
  br label %77

; <label>:76:                                     ; preds = %14
  br label %77

; <label>:77:                                     ; preds = %76, %74, %72, %66, %55, %49, %43, %37, %31, %25, %19
  %78 = load i32, i32* %4, align 4
  %79 = icmp ne i32 %78, 0
  br i1 %79, label %86, label %80

; <label>:80:                                     ; preds = %77
  %81 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %82 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %81, i32 0, i32 1
  %83 = load i8*, i8** %82, align 8
  call void @free(i8* %83) #5
  %84 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %85 = bitcast %struct.mpc_parser_t* %84 to i8*
  call void @free(i8* %85) #5
  br label %86

; <label>:86:                                     ; preds = %13, %80, %77
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_new(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca %struct.mpc_parser_t*, align 8
  store i8* %0, i8** %2, align 8
  %4 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %4, %struct.mpc_parser_t** %3, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %5, i32 0, i32 0
  store i8 1, i8* %6, align 8
  %7 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %8 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %7, i32 0, i32 1
  %9 = load i8*, i8** %8, align 8
  %10 = load i8*, i8** %2, align 8
  %11 = call i64 @strlen(i8* %10) #7
  %12 = add i64 %11, 1
  %13 = call i8* @realloc(i8* %9, i64 %12) #5
  %14 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %15 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %14, i32 0, i32 1
  store i8* %13, i8** %15, align 8
  %16 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %17 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %16, i32 0, i32 1
  %18 = load i8*, i8** %17, align 8
  %19 = load i8*, i8** %2, align 8
  %20 = call i8* @strcpy(i8* %18, i8* %19) #5
  %21 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  ret %struct.mpc_parser_t* %21
}

; Function Attrs: noinline nounwind optnone uwtable
define internal %struct.mpc_parser_t* @mpc_undefined() #0 {
  %1 = alloca %struct.mpc_parser_t*, align 8
  %2 = call noalias i8* @calloc(i64 1, i64 56) #5
  %3 = bitcast i8* %2 to %struct.mpc_parser_t*
  store %struct.mpc_parser_t* %3, %struct.mpc_parser_t** %1, align 8
  %4 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1, align 8
  %5 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %4, i32 0, i32 0
  store i8 0, i8* %5, align 8
  %6 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1, align 8
  %7 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %6, i32 0, i32 2
  store i8 0, i8* %7, align 8
  %8 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1, align 8
  %9 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %8, i32 0, i32 1
  store i8* null, i8** %9, align 8
  %10 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1, align 8
  ret %struct.mpc_parser_t* %10
}

; Function Attrs: nounwind
declare i8* @strcpy(i8*, i8*) #1

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_copy(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca i32, align 4
  %5 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store i32 0, i32* %4, align 4
  %6 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %7 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %6, i32 0, i32 0
  %8 = load i8, i8* %7, align 8
  %9 = icmp ne i8 %8, 0
  br i1 %9, label %10, label %12

; <label>:10:                                     ; preds = %1
  %11 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  store %struct.mpc_parser_t* %11, %struct.mpc_parser_t** %2, align 8
  br label %330

; <label>:12:                                     ; preds = %1
  %13 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %13, %struct.mpc_parser_t** %5, align 8
  %14 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %15 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %14, i32 0, i32 0
  %16 = load i8, i8* %15, align 8
  %17 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %18 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %17, i32 0, i32 0
  store i8 %16, i8* %18, align 8
  %19 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %20 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %19, i32 0, i32 2
  %21 = load i8, i8* %20, align 8
  %22 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %23 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %22, i32 0, i32 2
  store i8 %21, i8* %23, align 8
  %24 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %25 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %24, i32 0, i32 3
  %26 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %27 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %26, i32 0, i32 3
  %28 = bitcast %union.mpc_pdata_t* %25 to i8*
  %29 = bitcast %union.mpc_pdata_t* %27 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %28, i8* %29, i64 32, i32 8, i1 false)
  %30 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %31 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %30, i32 0, i32 1
  %32 = load i8*, i8** %31, align 8
  %33 = icmp ne i8* %32, null
  br i1 %33, label %34, label %50

; <label>:34:                                     ; preds = %12
  %35 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %36 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %35, i32 0, i32 1
  %37 = load i8*, i8** %36, align 8
  %38 = call i64 @strlen(i8* %37) #7
  %39 = add i64 %38, 1
  %40 = call noalias i8* @malloc(i64 %39) #5
  %41 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %42 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %41, i32 0, i32 1
  store i8* %40, i8** %42, align 8
  %43 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %44 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %43, i32 0, i32 1
  %45 = load i8*, i8** %44, align 8
  %46 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %47 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %46, i32 0, i32 1
  %48 = load i8*, i8** %47, align 8
  %49 = call i8* @strcpy(i8* %45, i8* %48) #5
  br label %50

; <label>:50:                                     ; preds = %34, %12
  %51 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %52 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %51, i32 0, i32 2
  %53 = load i8, i8* %52, align 8
  %54 = sext i8 %53 to i32
  switch i32 %54, label %327 [
    i32 2, label %55
    i32 10, label %79
    i32 11, label %79
    i32 14, label %79
    i32 15, label %103
    i32 16, label %114
    i32 17, label %125
    i32 19, label %136
    i32 18, label %136
    i32 5, label %147
    i32 20, label %181
    i32 21, label %181
    i32 22, label %181
    i32 23, label %192
    i32 24, label %237
  ]

; <label>:55:                                     ; preds = %50
  %56 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %57 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %56, i32 0, i32 3
  %58 = bitcast %union.mpc_pdata_t* %57 to %struct.mpc_pdata_fail_t*
  %59 = getelementptr inbounds %struct.mpc_pdata_fail_t, %struct.mpc_pdata_fail_t* %58, i32 0, i32 0
  %60 = load i8*, i8** %59, align 8
  %61 = call i64 @strlen(i8* %60) #7
  %62 = add i64 %61, 1
  %63 = call noalias i8* @malloc(i64 %62) #5
  %64 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %65 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %64, i32 0, i32 3
  %66 = bitcast %union.mpc_pdata_t* %65 to %struct.mpc_pdata_fail_t*
  %67 = getelementptr inbounds %struct.mpc_pdata_fail_t, %struct.mpc_pdata_fail_t* %66, i32 0, i32 0
  store i8* %63, i8** %67, align 8
  %68 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %69 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %68, i32 0, i32 3
  %70 = bitcast %union.mpc_pdata_t* %69 to %struct.mpc_pdata_fail_t*
  %71 = getelementptr inbounds %struct.mpc_pdata_fail_t, %struct.mpc_pdata_fail_t* %70, i32 0, i32 0
  %72 = load i8*, i8** %71, align 8
  %73 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %74 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %73, i32 0, i32 3
  %75 = bitcast %union.mpc_pdata_t* %74 to %struct.mpc_pdata_fail_t*
  %76 = getelementptr inbounds %struct.mpc_pdata_fail_t, %struct.mpc_pdata_fail_t* %75, i32 0, i32 0
  %77 = load i8*, i8** %76, align 8
  %78 = call i8* @strcpy(i8* %72, i8* %77) #5
  br label %328

; <label>:79:                                     ; preds = %50, %50, %50
  %80 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %81 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %80, i32 0, i32 3
  %82 = bitcast %union.mpc_pdata_t* %81 to %struct.mpc_pdata_string_t*
  %83 = getelementptr inbounds %struct.mpc_pdata_string_t, %struct.mpc_pdata_string_t* %82, i32 0, i32 0
  %84 = load i8*, i8** %83, align 8
  %85 = call i64 @strlen(i8* %84) #7
  %86 = add i64 %85, 1
  %87 = call noalias i8* @malloc(i64 %86) #5
  %88 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %89 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %88, i32 0, i32 3
  %90 = bitcast %union.mpc_pdata_t* %89 to %struct.mpc_pdata_string_t*
  %91 = getelementptr inbounds %struct.mpc_pdata_string_t, %struct.mpc_pdata_string_t* %90, i32 0, i32 0
  store i8* %87, i8** %91, align 8
  %92 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %93 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %92, i32 0, i32 3
  %94 = bitcast %union.mpc_pdata_t* %93 to %struct.mpc_pdata_string_t*
  %95 = getelementptr inbounds %struct.mpc_pdata_string_t, %struct.mpc_pdata_string_t* %94, i32 0, i32 0
  %96 = load i8*, i8** %95, align 8
  %97 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %98 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %97, i32 0, i32 3
  %99 = bitcast %union.mpc_pdata_t* %98 to %struct.mpc_pdata_string_t*
  %100 = getelementptr inbounds %struct.mpc_pdata_string_t, %struct.mpc_pdata_string_t* %99, i32 0, i32 0
  %101 = load i8*, i8** %100, align 8
  %102 = call i8* @strcpy(i8* %96, i8* %101) #5
  br label %328

; <label>:103:                                    ; preds = %50
  %104 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %105 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %104, i32 0, i32 3
  %106 = bitcast %union.mpc_pdata_t* %105 to %struct.mpc_pdata_apply_t*
  %107 = getelementptr inbounds %struct.mpc_pdata_apply_t, %struct.mpc_pdata_apply_t* %106, i32 0, i32 0
  %108 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %107, align 8
  %109 = call %struct.mpc_parser_t* @mpc_copy(%struct.mpc_parser_t* %108)
  %110 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %111 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %110, i32 0, i32 3
  %112 = bitcast %union.mpc_pdata_t* %111 to %struct.mpc_pdata_apply_t*
  %113 = getelementptr inbounds %struct.mpc_pdata_apply_t, %struct.mpc_pdata_apply_t* %112, i32 0, i32 0
  store %struct.mpc_parser_t* %109, %struct.mpc_parser_t** %113, align 8
  br label %328

; <label>:114:                                    ; preds = %50
  %115 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %116 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %115, i32 0, i32 3
  %117 = bitcast %union.mpc_pdata_t* %116 to %struct.mpc_pdata_apply_to_t*
  %118 = getelementptr inbounds %struct.mpc_pdata_apply_to_t, %struct.mpc_pdata_apply_to_t* %117, i32 0, i32 0
  %119 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %118, align 8
  %120 = call %struct.mpc_parser_t* @mpc_copy(%struct.mpc_parser_t* %119)
  %121 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %122 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %121, i32 0, i32 3
  %123 = bitcast %union.mpc_pdata_t* %122 to %struct.mpc_pdata_apply_to_t*
  %124 = getelementptr inbounds %struct.mpc_pdata_apply_to_t, %struct.mpc_pdata_apply_to_t* %123, i32 0, i32 0
  store %struct.mpc_parser_t* %120, %struct.mpc_parser_t** %124, align 8
  br label %328

; <label>:125:                                    ; preds = %50
  %126 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %127 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %126, i32 0, i32 3
  %128 = bitcast %union.mpc_pdata_t* %127 to %struct.mpc_pdata_predict_t*
  %129 = getelementptr inbounds %struct.mpc_pdata_predict_t, %struct.mpc_pdata_predict_t* %128, i32 0, i32 0
  %130 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %129, align 8
  %131 = call %struct.mpc_parser_t* @mpc_copy(%struct.mpc_parser_t* %130)
  %132 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %133 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %132, i32 0, i32 3
  %134 = bitcast %union.mpc_pdata_t* %133 to %struct.mpc_pdata_predict_t*
  %135 = getelementptr inbounds %struct.mpc_pdata_predict_t, %struct.mpc_pdata_predict_t* %134, i32 0, i32 0
  store %struct.mpc_parser_t* %131, %struct.mpc_parser_t** %135, align 8
  br label %328

; <label>:136:                                    ; preds = %50, %50
  %137 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %138 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %137, i32 0, i32 3
  %139 = bitcast %union.mpc_pdata_t* %138 to %struct.mpc_pdata_not_t*
  %140 = getelementptr inbounds %struct.mpc_pdata_not_t, %struct.mpc_pdata_not_t* %139, i32 0, i32 0
  %141 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %140, align 8
  %142 = call %struct.mpc_parser_t* @mpc_copy(%struct.mpc_parser_t* %141)
  %143 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %144 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %143, i32 0, i32 3
  %145 = bitcast %union.mpc_pdata_t* %144 to %struct.mpc_pdata_not_t*
  %146 = getelementptr inbounds %struct.mpc_pdata_not_t, %struct.mpc_pdata_not_t* %145, i32 0, i32 0
  store %struct.mpc_parser_t* %142, %struct.mpc_parser_t** %146, align 8
  br label %328

; <label>:147:                                    ; preds = %50
  %148 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %149 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %148, i32 0, i32 3
  %150 = bitcast %union.mpc_pdata_t* %149 to %struct.mpc_pdata_expect_t*
  %151 = getelementptr inbounds %struct.mpc_pdata_expect_t, %struct.mpc_pdata_expect_t* %150, i32 0, i32 0
  %152 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %151, align 8
  %153 = call %struct.mpc_parser_t* @mpc_copy(%struct.mpc_parser_t* %152)
  %154 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %155 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %154, i32 0, i32 3
  %156 = bitcast %union.mpc_pdata_t* %155 to %struct.mpc_pdata_expect_t*
  %157 = getelementptr inbounds %struct.mpc_pdata_expect_t, %struct.mpc_pdata_expect_t* %156, i32 0, i32 0
  store %struct.mpc_parser_t* %153, %struct.mpc_parser_t** %157, align 8
  %158 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %159 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %158, i32 0, i32 3
  %160 = bitcast %union.mpc_pdata_t* %159 to %struct.mpc_pdata_expect_t*
  %161 = getelementptr inbounds %struct.mpc_pdata_expect_t, %struct.mpc_pdata_expect_t* %160, i32 0, i32 1
  %162 = load i8*, i8** %161, align 8
  %163 = call i64 @strlen(i8* %162) #7
  %164 = add i64 %163, 1
  %165 = call noalias i8* @malloc(i64 %164) #5
  %166 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %167 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %166, i32 0, i32 3
  %168 = bitcast %union.mpc_pdata_t* %167 to %struct.mpc_pdata_expect_t*
  %169 = getelementptr inbounds %struct.mpc_pdata_expect_t, %struct.mpc_pdata_expect_t* %168, i32 0, i32 1
  store i8* %165, i8** %169, align 8
  %170 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %171 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %170, i32 0, i32 3
  %172 = bitcast %union.mpc_pdata_t* %171 to %struct.mpc_pdata_expect_t*
  %173 = getelementptr inbounds %struct.mpc_pdata_expect_t, %struct.mpc_pdata_expect_t* %172, i32 0, i32 1
  %174 = load i8*, i8** %173, align 8
  %175 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %176 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %175, i32 0, i32 3
  %177 = bitcast %union.mpc_pdata_t* %176 to %struct.mpc_pdata_expect_t*
  %178 = getelementptr inbounds %struct.mpc_pdata_expect_t, %struct.mpc_pdata_expect_t* %177, i32 0, i32 1
  %179 = load i8*, i8** %178, align 8
  %180 = call i8* @strcpy(i8* %174, i8* %179) #5
  br label %328

; <label>:181:                                    ; preds = %50, %50, %50
  %182 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %183 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %182, i32 0, i32 3
  %184 = bitcast %union.mpc_pdata_t* %183 to %struct.mpc_pdata_repeat_t*
  %185 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %184, i32 0, i32 2
  %186 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %185, align 8
  %187 = call %struct.mpc_parser_t* @mpc_copy(%struct.mpc_parser_t* %186)
  %188 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %189 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %188, i32 0, i32 3
  %190 = bitcast %union.mpc_pdata_t* %189 to %struct.mpc_pdata_repeat_t*
  %191 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %190, i32 0, i32 2
  store %struct.mpc_parser_t* %187, %struct.mpc_parser_t** %191, align 8
  br label %328

; <label>:192:                                    ; preds = %50
  %193 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %194 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %193, i32 0, i32 3
  %195 = bitcast %union.mpc_pdata_t* %194 to %struct.mpc_pdata_or_t*
  %196 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %195, i32 0, i32 0
  %197 = load i32, i32* %196, align 8
  %198 = sext i32 %197 to i64
  %199 = mul i64 %198, 8
  %200 = call noalias i8* @malloc(i64 %199) #5
  %201 = bitcast i8* %200 to %struct.mpc_parser_t**
  %202 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %203 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %202, i32 0, i32 3
  %204 = bitcast %union.mpc_pdata_t* %203 to %struct.mpc_pdata_or_t*
  %205 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %204, i32 0, i32 1
  store %struct.mpc_parser_t** %201, %struct.mpc_parser_t*** %205, align 8
  store i32 0, i32* %4, align 4
  br label %206

; <label>:206:                                    ; preds = %233, %192
  %207 = load i32, i32* %4, align 4
  %208 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %209 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %208, i32 0, i32 3
  %210 = bitcast %union.mpc_pdata_t* %209 to %struct.mpc_pdata_or_t*
  %211 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %210, i32 0, i32 0
  %212 = load i32, i32* %211, align 8
  %213 = icmp slt i32 %207, %212
  br i1 %213, label %214, label %236

; <label>:214:                                    ; preds = %206
  %215 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %216 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %215, i32 0, i32 3
  %217 = bitcast %union.mpc_pdata_t* %216 to %struct.mpc_pdata_or_t*
  %218 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %217, i32 0, i32 1
  %219 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %218, align 8
  %220 = load i32, i32* %4, align 4
  %221 = sext i32 %220 to i64
  %222 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %219, i64 %221
  %223 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %222, align 8
  %224 = call %struct.mpc_parser_t* @mpc_copy(%struct.mpc_parser_t* %223)
  %225 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %226 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %225, i32 0, i32 3
  %227 = bitcast %union.mpc_pdata_t* %226 to %struct.mpc_pdata_or_t*
  %228 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %227, i32 0, i32 1
  %229 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %228, align 8
  %230 = load i32, i32* %4, align 4
  %231 = sext i32 %230 to i64
  %232 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %229, i64 %231
  store %struct.mpc_parser_t* %224, %struct.mpc_parser_t** %232, align 8
  br label %233

; <label>:233:                                    ; preds = %214
  %234 = load i32, i32* %4, align 4
  %235 = add nsw i32 %234, 1
  store i32 %235, i32* %4, align 4
  br label %206

; <label>:236:                                    ; preds = %206
  br label %328

; <label>:237:                                    ; preds = %50
  %238 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %239 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %238, i32 0, i32 3
  %240 = bitcast %union.mpc_pdata_t* %239 to %struct.mpc_pdata_and_t*
  %241 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %240, i32 0, i32 0
  %242 = load i32, i32* %241, align 8
  %243 = sext i32 %242 to i64
  %244 = mul i64 %243, 8
  %245 = call noalias i8* @malloc(i64 %244) #5
  %246 = bitcast i8* %245 to %struct.mpc_parser_t**
  %247 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %248 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %247, i32 0, i32 3
  %249 = bitcast %union.mpc_pdata_t* %248 to %struct.mpc_pdata_and_t*
  %250 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %249, i32 0, i32 2
  store %struct.mpc_parser_t** %246, %struct.mpc_parser_t*** %250, align 8
  store i32 0, i32* %4, align 4
  br label %251

; <label>:251:                                    ; preds = %278, %237
  %252 = load i32, i32* %4, align 4
  %253 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %254 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %253, i32 0, i32 3
  %255 = bitcast %union.mpc_pdata_t* %254 to %struct.mpc_pdata_and_t*
  %256 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %255, i32 0, i32 0
  %257 = load i32, i32* %256, align 8
  %258 = icmp slt i32 %252, %257
  br i1 %258, label %259, label %281

; <label>:259:                                    ; preds = %251
  %260 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %261 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %260, i32 0, i32 3
  %262 = bitcast %union.mpc_pdata_t* %261 to %struct.mpc_pdata_and_t*
  %263 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %262, i32 0, i32 2
  %264 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %263, align 8
  %265 = load i32, i32* %4, align 4
  %266 = sext i32 %265 to i64
  %267 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %264, i64 %266
  %268 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %267, align 8
  %269 = call %struct.mpc_parser_t* @mpc_copy(%struct.mpc_parser_t* %268)
  %270 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %271 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %270, i32 0, i32 3
  %272 = bitcast %union.mpc_pdata_t* %271 to %struct.mpc_pdata_and_t*
  %273 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %272, i32 0, i32 2
  %274 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %273, align 8
  %275 = load i32, i32* %4, align 4
  %276 = sext i32 %275 to i64
  %277 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %274, i64 %276
  store %struct.mpc_parser_t* %269, %struct.mpc_parser_t** %277, align 8
  br label %278

; <label>:278:                                    ; preds = %259
  %279 = load i32, i32* %4, align 4
  %280 = add nsw i32 %279, 1
  store i32 %280, i32* %4, align 4
  br label %251

; <label>:281:                                    ; preds = %251
  %282 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %283 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %282, i32 0, i32 3
  %284 = bitcast %union.mpc_pdata_t* %283 to %struct.mpc_pdata_and_t*
  %285 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %284, i32 0, i32 0
  %286 = load i32, i32* %285, align 8
  %287 = sub nsw i32 %286, 1
  %288 = sext i32 %287 to i64
  %289 = mul i64 %288, 8
  %290 = call noalias i8* @malloc(i64 %289) #5
  %291 = bitcast i8* %290 to void (i8*)**
  %292 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %293 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %292, i32 0, i32 3
  %294 = bitcast %union.mpc_pdata_t* %293 to %struct.mpc_pdata_and_t*
  %295 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %294, i32 0, i32 3
  store void (i8*)** %291, void (i8*)*** %295, align 8
  store i32 0, i32* %4, align 4
  br label %296

; <label>:296:                                    ; preds = %323, %281
  %297 = load i32, i32* %4, align 4
  %298 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %299 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %298, i32 0, i32 3
  %300 = bitcast %union.mpc_pdata_t* %299 to %struct.mpc_pdata_and_t*
  %301 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %300, i32 0, i32 0
  %302 = load i32, i32* %301, align 8
  %303 = sub nsw i32 %302, 1
  %304 = icmp slt i32 %297, %303
  br i1 %304, label %305, label %326

; <label>:305:                                    ; preds = %296
  %306 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %307 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %306, i32 0, i32 3
  %308 = bitcast %union.mpc_pdata_t* %307 to %struct.mpc_pdata_and_t*
  %309 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %308, i32 0, i32 3
  %310 = load void (i8*)**, void (i8*)*** %309, align 8
  %311 = load i32, i32* %4, align 4
  %312 = sext i32 %311 to i64
  %313 = getelementptr inbounds void (i8*)*, void (i8*)** %310, i64 %312
  %314 = load void (i8*)*, void (i8*)** %313, align 8
  %315 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %316 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %315, i32 0, i32 3
  %317 = bitcast %union.mpc_pdata_t* %316 to %struct.mpc_pdata_and_t*
  %318 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %317, i32 0, i32 3
  %319 = load void (i8*)**, void (i8*)*** %318, align 8
  %320 = load i32, i32* %4, align 4
  %321 = sext i32 %320 to i64
  %322 = getelementptr inbounds void (i8*)*, void (i8*)** %319, i64 %321
  store void (i8*)* %314, void (i8*)** %322, align 8
  br label %323

; <label>:323:                                    ; preds = %305
  %324 = load i32, i32* %4, align 4
  %325 = add nsw i32 %324, 1
  store i32 %325, i32* %4, align 4
  br label %296

; <label>:326:                                    ; preds = %296
  br label %328

; <label>:327:                                    ; preds = %50
  br label %328

; <label>:328:                                    ; preds = %327, %326, %236, %181, %147, %136, %125, %114, %103, %79, %55
  %329 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  store %struct.mpc_parser_t* %329, %struct.mpc_parser_t** %2, align 8
  br label %330

; <label>:330:                                    ; preds = %328, %10
  %331 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  ret %struct.mpc_parser_t* %331
}

; Function Attrs: nounwind
declare noalias i8* @malloc(i64) #1

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_undefine(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  %3 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  call void @mpc_undefine_unretained(%struct.mpc_parser_t* %3, i32 1)
  %4 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %5 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %4, i32 0, i32 2
  store i8 0, i8* %5, align 8
  %6 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  ret %struct.mpc_parser_t* %6
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_define(%struct.mpc_parser_t*, %struct.mpc_parser_t*) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca %struct.mpc_parser_t*, align 8
  %5 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store %struct.mpc_parser_t* %1, %struct.mpc_parser_t** %4, align 8
  %6 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %7 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %6, i32 0, i32 0
  %8 = load i8, i8* %7, align 8
  %9 = icmp ne i8 %8, 0
  br i1 %9, label %10, label %22

; <label>:10:                                     ; preds = %2
  %11 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %12 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %11, i32 0, i32 2
  %13 = load i8, i8* %12, align 8
  %14 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %15 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %14, i32 0, i32 2
  store i8 %13, i8* %15, align 8
  %16 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %17 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %16, i32 0, i32 3
  %18 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %19 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %18, i32 0, i32 3
  %20 = bitcast %union.mpc_pdata_t* %17 to i8*
  %21 = bitcast %union.mpc_pdata_t* %19 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %20, i8* %21, i64 32, i32 8, i1 false)
  br label %37

; <label>:22:                                     ; preds = %2
  %23 = call %struct.mpc_parser_t* (i8*, ...) @mpc_failf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.11, i32 0, i32 0))
  store %struct.mpc_parser_t* %23, %struct.mpc_parser_t** %5, align 8
  %24 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %25 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %24, i32 0, i32 2
  %26 = load i8, i8* %25, align 8
  %27 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %28 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %27, i32 0, i32 2
  store i8 %26, i8* %28, align 8
  %29 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %30 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %29, i32 0, i32 3
  %31 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %32 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %31, i32 0, i32 3
  %33 = bitcast %union.mpc_pdata_t* %30 to i8*
  %34 = bitcast %union.mpc_pdata_t* %32 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %33, i8* %34, i64 32, i32 8, i1 false)
  %35 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %36 = bitcast %struct.mpc_parser_t* %35 to i8*
  call void @free(i8* %36) #5
  br label %37

; <label>:37:                                     ; preds = %22, %10
  %38 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %39 = bitcast %struct.mpc_parser_t* %38 to i8*
  call void @free(i8* %39) #5
  %40 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  ret %struct.mpc_parser_t* %40
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_failf(i8*, ...) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca [1 x %struct.__va_list_tag], align 16
  %4 = alloca i8*, align 8
  %5 = alloca %struct.mpc_parser_t*, align 8
  store i8* %0, i8** %2, align 8
  %6 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %6, %struct.mpc_parser_t** %5, align 8
  %7 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %8 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %7, i32 0, i32 2
  store i8 2, i8* %8, align 8
  %9 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %3, i32 0, i32 0
  %10 = bitcast %struct.__va_list_tag* %9 to i8*
  call void @llvm.va_start(i8* %10)
  %11 = call noalias i8* @malloc(i64 2048) #5
  store i8* %11, i8** %4, align 8
  %12 = load i8*, i8** %4, align 8
  %13 = load i8*, i8** %2, align 8
  %14 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %3, i32 0, i32 0
  %15 = call i32 @vsprintf(i8* %12, i8* %13, %struct.__va_list_tag* %14) #5
  %16 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %3, i32 0, i32 0
  %17 = bitcast %struct.__va_list_tag* %16 to i8*
  call void @llvm.va_end(i8* %17)
  %18 = load i8*, i8** %4, align 8
  %19 = load i8*, i8** %4, align 8
  %20 = call i64 @strlen(i8* %19) #7
  %21 = add i64 %20, 1
  %22 = call i8* @realloc(i8* %18, i64 %21) #5
  store i8* %22, i8** %4, align 8
  %23 = load i8*, i8** %4, align 8
  %24 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %25 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %24, i32 0, i32 3
  %26 = bitcast %union.mpc_pdata_t* %25 to %struct.mpc_pdata_fail_t*
  %27 = getelementptr inbounds %struct.mpc_pdata_fail_t, %struct.mpc_pdata_fail_t* %26, i32 0, i32 0
  store i8* %23, i8** %27, align 8
  %28 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  ret %struct.mpc_parser_t* %28
}

; Function Attrs: noinline nounwind optnone uwtable
define void @mpc_cleanup(i32, ...) #0 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca %struct.mpc_parser_t**, align 8
  %5 = alloca [1 x %struct.__va_list_tag], align 16
  store i32 %0, i32* %2, align 4
  %6 = load i32, i32* %2, align 4
  %7 = sext i32 %6 to i64
  %8 = mul i64 8, %7
  %9 = call noalias i8* @malloc(i64 %8) #5
  %10 = bitcast i8* %9 to %struct.mpc_parser_t**
  store %struct.mpc_parser_t** %10, %struct.mpc_parser_t*** %4, align 8
  %11 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %5, i32 0, i32 0
  %12 = bitcast %struct.__va_list_tag* %11 to i8*
  call void @llvm.va_start(i8* %12)
  store i32 0, i32* %3, align 4
  br label %13

; <label>:13:                                     ; preds = %40, %1
  %14 = load i32, i32* %3, align 4
  %15 = load i32, i32* %2, align 4
  %16 = icmp slt i32 %14, %15
  br i1 %16, label %17, label %43

; <label>:17:                                     ; preds = %13
  %18 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %5, i32 0, i32 0
  %19 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %18, i32 0, i32 0
  %20 = load i32, i32* %19, align 16
  %21 = icmp ule i32 %20, 40
  br i1 %21, label %22, label %28

; <label>:22:                                     ; preds = %17
  %23 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %18, i32 0, i32 3
  %24 = load i8*, i8** %23, align 16
  %25 = getelementptr i8, i8* %24, i32 %20
  %26 = bitcast i8* %25 to %struct.mpc_parser_t**
  %27 = add i32 %20, 8
  store i32 %27, i32* %19, align 16
  br label %33

; <label>:28:                                     ; preds = %17
  %29 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %18, i32 0, i32 2
  %30 = load i8*, i8** %29, align 8
  %31 = bitcast i8* %30 to %struct.mpc_parser_t**
  %32 = getelementptr i8, i8* %30, i32 8
  store i8* %32, i8** %29, align 8
  br label %33

; <label>:33:                                     ; preds = %28, %22
  %34 = phi %struct.mpc_parser_t** [ %26, %22 ], [ %31, %28 ]
  %35 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %34, align 8
  %36 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %4, align 8
  %37 = load i32, i32* %3, align 4
  %38 = sext i32 %37 to i64
  %39 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %36, i64 %38
  store %struct.mpc_parser_t* %35, %struct.mpc_parser_t** %39, align 8
  br label %40

; <label>:40:                                     ; preds = %33
  %41 = load i32, i32* %3, align 4
  %42 = add nsw i32 %41, 1
  store i32 %42, i32* %3, align 4
  br label %13

; <label>:43:                                     ; preds = %13
  store i32 0, i32* %3, align 4
  br label %44

; <label>:44:                                     ; preds = %55, %43
  %45 = load i32, i32* %3, align 4
  %46 = load i32, i32* %2, align 4
  %47 = icmp slt i32 %45, %46
  br i1 %47, label %48, label %58

; <label>:48:                                     ; preds = %44
  %49 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %4, align 8
  %50 = load i32, i32* %3, align 4
  %51 = sext i32 %50 to i64
  %52 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %49, i64 %51
  %53 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %52, align 8
  %54 = call %struct.mpc_parser_t* @mpc_undefine(%struct.mpc_parser_t* %53)
  br label %55

; <label>:55:                                     ; preds = %48
  %56 = load i32, i32* %3, align 4
  %57 = add nsw i32 %56, 1
  store i32 %57, i32* %3, align 4
  br label %44

; <label>:58:                                     ; preds = %44
  store i32 0, i32* %3, align 4
  br label %59

; <label>:59:                                     ; preds = %69, %58
  %60 = load i32, i32* %3, align 4
  %61 = load i32, i32* %2, align 4
  %62 = icmp slt i32 %60, %61
  br i1 %62, label %63, label %72

; <label>:63:                                     ; preds = %59
  %64 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %4, align 8
  %65 = load i32, i32* %3, align 4
  %66 = sext i32 %65 to i64
  %67 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %64, i64 %66
  %68 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %67, align 8
  call void @mpc_delete(%struct.mpc_parser_t* %68)
  br label %69

; <label>:69:                                     ; preds = %63
  %70 = load i32, i32* %3, align 4
  %71 = add nsw i32 %70, 1
  store i32 %71, i32* %3, align 4
  br label %59

; <label>:72:                                     ; preds = %59
  %73 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %5, i32 0, i32 0
  %74 = bitcast %struct.__va_list_tag* %73 to i8*
  call void @llvm.va_end(i8* %74)
  %75 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %4, align 8
  %76 = bitcast %struct.mpc_parser_t** %75 to i8*
  call void @free(i8* %76) #5
  ret void
}

; Function Attrs: nounwind
declare void @llvm.va_start(i8*) #5

; Function Attrs: nounwind
declare void @llvm.va_end(i8*) #5

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_pass() #0 {
  %1 = alloca %struct.mpc_parser_t*, align 8
  %2 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %2, %struct.mpc_parser_t** %1, align 8
  %3 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1, align 8
  %4 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %3, i32 0, i32 2
  store i8 1, i8* %4, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1, align 8
  ret %struct.mpc_parser_t* %5
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_fail(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca %struct.mpc_parser_t*, align 8
  store i8* %0, i8** %2, align 8
  %4 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %4, %struct.mpc_parser_t** %3, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %5, i32 0, i32 2
  store i8 2, i8* %6, align 8
  %7 = load i8*, i8** %2, align 8
  %8 = call i64 @strlen(i8* %7) #7
  %9 = add i64 %8, 1
  %10 = call noalias i8* @malloc(i64 %9) #5
  %11 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %12 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %11, i32 0, i32 3
  %13 = bitcast %union.mpc_pdata_t* %12 to %struct.mpc_pdata_fail_t*
  %14 = getelementptr inbounds %struct.mpc_pdata_fail_t, %struct.mpc_pdata_fail_t* %13, i32 0, i32 0
  store i8* %10, i8** %14, align 8
  %15 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %16 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %15, i32 0, i32 3
  %17 = bitcast %union.mpc_pdata_t* %16 to %struct.mpc_pdata_fail_t*
  %18 = getelementptr inbounds %struct.mpc_pdata_fail_t, %struct.mpc_pdata_fail_t* %17, i32 0, i32 0
  %19 = load i8*, i8** %18, align 8
  %20 = load i8*, i8** %2, align 8
  %21 = call i8* @strcpy(i8* %19, i8* %20) #5
  %22 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  ret %struct.mpc_parser_t* %22
}

; Function Attrs: nounwind
declare i32 @vsprintf(i8*, i8*, %struct.__va_list_tag*) #1

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_lift_val(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca %struct.mpc_parser_t*, align 8
  store i8* %0, i8** %2, align 8
  %4 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %4, %struct.mpc_parser_t** %3, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %5, i32 0, i32 2
  store i8 4, i8* %6, align 8
  %7 = load i8*, i8** %2, align 8
  %8 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %9 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %8, i32 0, i32 3
  %10 = bitcast %union.mpc_pdata_t* %9 to %struct.mpc_pdata_lift_t*
  %11 = getelementptr inbounds %struct.mpc_pdata_lift_t, %struct.mpc_pdata_lift_t* %10, i32 0, i32 1
  store i8* %7, i8** %11, align 8
  %12 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  ret %struct.mpc_parser_t* %12
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_lift(i8* ()*) #0 {
  %2 = alloca i8* ()*, align 8
  %3 = alloca %struct.mpc_parser_t*, align 8
  store i8* ()* %0, i8* ()** %2, align 8
  %4 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %4, %struct.mpc_parser_t** %3, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %5, i32 0, i32 2
  store i8 3, i8* %6, align 8
  %7 = load i8* ()*, i8* ()** %2, align 8
  %8 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %9 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %8, i32 0, i32 3
  %10 = bitcast %union.mpc_pdata_t* %9 to %struct.mpc_pdata_lift_t*
  %11 = getelementptr inbounds %struct.mpc_pdata_lift_t, %struct.mpc_pdata_lift_t* %10, i32 0, i32 0
  store i8* ()* %7, i8* ()** %11, align 8
  %12 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  ret %struct.mpc_parser_t* %12
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_anchor(i32 (i8, i8)*) #0 {
  %2 = alloca i32 (i8, i8)*, align 8
  %3 = alloca %struct.mpc_parser_t*, align 8
  store i32 (i8, i8)* %0, i32 (i8, i8)** %2, align 8
  %4 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %4, %struct.mpc_parser_t** %3, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %5, i32 0, i32 2
  store i8 6, i8* %6, align 8
  %7 = load i32 (i8, i8)*, i32 (i8, i8)** %2, align 8
  %8 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %9 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %8, i32 0, i32 3
  %10 = bitcast %union.mpc_pdata_t* %9 to %struct.mpc_pdata_anchor_t*
  %11 = getelementptr inbounds %struct.mpc_pdata_anchor_t, %struct.mpc_pdata_anchor_t* %10, i32 0, i32 0
  store i32 (i8, i8)* %7, i32 (i8, i8)** %11, align 8
  %12 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %13 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %12, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.12, i32 0, i32 0))
  ret %struct.mpc_parser_t* %13
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t*, i8*) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca i8*, align 8
  %5 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store i8* %1, i8** %4, align 8
  %6 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %6, %struct.mpc_parser_t** %5, align 8
  %7 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %8 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %7, i32 0, i32 2
  store i8 5, i8* %8, align 8
  %9 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %10 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %11 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %10, i32 0, i32 3
  %12 = bitcast %union.mpc_pdata_t* %11 to %struct.mpc_pdata_expect_t*
  %13 = getelementptr inbounds %struct.mpc_pdata_expect_t, %struct.mpc_pdata_expect_t* %12, i32 0, i32 0
  store %struct.mpc_parser_t* %9, %struct.mpc_parser_t** %13, align 8
  %14 = load i8*, i8** %4, align 8
  %15 = call i64 @strlen(i8* %14) #7
  %16 = add i64 %15, 1
  %17 = call noalias i8* @malloc(i64 %16) #5
  %18 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %19 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %18, i32 0, i32 3
  %20 = bitcast %union.mpc_pdata_t* %19 to %struct.mpc_pdata_expect_t*
  %21 = getelementptr inbounds %struct.mpc_pdata_expect_t, %struct.mpc_pdata_expect_t* %20, i32 0, i32 1
  store i8* %17, i8** %21, align 8
  %22 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %23 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %22, i32 0, i32 3
  %24 = bitcast %union.mpc_pdata_t* %23 to %struct.mpc_pdata_expect_t*
  %25 = getelementptr inbounds %struct.mpc_pdata_expect_t, %struct.mpc_pdata_expect_t* %24, i32 0, i32 1
  %26 = load i8*, i8** %25, align 8
  %27 = load i8*, i8** %4, align 8
  %28 = call i8* @strcpy(i8* %26, i8* %27) #5
  %29 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  ret %struct.mpc_parser_t* %29
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_state() #0 {
  %1 = alloca %struct.mpc_parser_t*, align 8
  %2 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %2, %struct.mpc_parser_t** %1, align 8
  %3 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1, align 8
  %4 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %3, i32 0, i32 2
  store i8 7, i8* %4, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1, align 8
  ret %struct.mpc_parser_t* %5
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_expectf(%struct.mpc_parser_t*, i8*, ...) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca i8*, align 8
  %5 = alloca [1 x %struct.__va_list_tag], align 16
  %6 = alloca i8*, align 8
  %7 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store i8* %1, i8** %4, align 8
  %8 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %8, %struct.mpc_parser_t** %7, align 8
  %9 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %10 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %9, i32 0, i32 2
  store i8 5, i8* %10, align 8
  %11 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %5, i32 0, i32 0
  %12 = bitcast %struct.__va_list_tag* %11 to i8*
  call void @llvm.va_start(i8* %12)
  %13 = call noalias i8* @malloc(i64 2048) #5
  store i8* %13, i8** %6, align 8
  %14 = load i8*, i8** %6, align 8
  %15 = load i8*, i8** %4, align 8
  %16 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %5, i32 0, i32 0
  %17 = call i32 @vsprintf(i8* %14, i8* %15, %struct.__va_list_tag* %16) #5
  %18 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %5, i32 0, i32 0
  %19 = bitcast %struct.__va_list_tag* %18 to i8*
  call void @llvm.va_end(i8* %19)
  %20 = load i8*, i8** %6, align 8
  %21 = load i8*, i8** %6, align 8
  %22 = call i64 @strlen(i8* %21) #7
  %23 = add i64 %22, 1
  %24 = call i8* @realloc(i8* %20, i64 %23) #5
  store i8* %24, i8** %6, align 8
  %25 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %26 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %27 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %26, i32 0, i32 3
  %28 = bitcast %union.mpc_pdata_t* %27 to %struct.mpc_pdata_expect_t*
  %29 = getelementptr inbounds %struct.mpc_pdata_expect_t, %struct.mpc_pdata_expect_t* %28, i32 0, i32 0
  store %struct.mpc_parser_t* %25, %struct.mpc_parser_t** %29, align 8
  %30 = load i8*, i8** %6, align 8
  %31 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %32 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %31, i32 0, i32 3
  %33 = bitcast %union.mpc_pdata_t* %32 to %struct.mpc_pdata_expect_t*
  %34 = getelementptr inbounds %struct.mpc_pdata_expect_t, %struct.mpc_pdata_expect_t* %33, i32 0, i32 1
  store i8* %30, i8** %34, align 8
  %35 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  ret %struct.mpc_parser_t* %35
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_any() #0 {
  %1 = alloca %struct.mpc_parser_t*, align 8
  %2 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %2, %struct.mpc_parser_t** %1, align 8
  %3 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1, align 8
  %4 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %3, i32 0, i32 2
  store i8 8, i8* %4, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1, align 8
  %6 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %5, i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str.13, i32 0, i32 0))
  ret %struct.mpc_parser_t* %6
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_char(i8 signext) #0 {
  %2 = alloca i8, align 1
  %3 = alloca %struct.mpc_parser_t*, align 8
  store i8 %0, i8* %2, align 1
  %4 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %4, %struct.mpc_parser_t** %3, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %5, i32 0, i32 2
  store i8 9, i8* %6, align 8
  %7 = load i8, i8* %2, align 1
  %8 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %9 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %8, i32 0, i32 3
  %10 = bitcast %union.mpc_pdata_t* %9 to %struct.mpc_pdata_single_t*
  %11 = getelementptr inbounds %struct.mpc_pdata_single_t, %struct.mpc_pdata_single_t* %10, i32 0, i32 0
  store i8 %7, i8* %11, align 8
  %12 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %13 = load i8, i8* %2, align 1
  %14 = sext i8 %13 to i32
  %15 = call %struct.mpc_parser_t* (%struct.mpc_parser_t*, i8*, ...) @mpc_expectf(%struct.mpc_parser_t* %12, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.14, i32 0, i32 0), i32 %14)
  ret %struct.mpc_parser_t* %15
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_range(i8 signext, i8 signext) #0 {
  %3 = alloca i8, align 1
  %4 = alloca i8, align 1
  %5 = alloca %struct.mpc_parser_t*, align 8
  store i8 %0, i8* %3, align 1
  store i8 %1, i8* %4, align 1
  %6 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %6, %struct.mpc_parser_t** %5, align 8
  %7 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %8 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %7, i32 0, i32 2
  store i8 12, i8* %8, align 8
  %9 = load i8, i8* %3, align 1
  %10 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %11 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %10, i32 0, i32 3
  %12 = bitcast %union.mpc_pdata_t* %11 to %struct.mpc_pdata_range_t*
  %13 = getelementptr inbounds %struct.mpc_pdata_range_t, %struct.mpc_pdata_range_t* %12, i32 0, i32 0
  store i8 %9, i8* %13, align 8
  %14 = load i8, i8* %4, align 1
  %15 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %16 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %15, i32 0, i32 3
  %17 = bitcast %union.mpc_pdata_t* %16 to %struct.mpc_pdata_range_t*
  %18 = getelementptr inbounds %struct.mpc_pdata_range_t, %struct.mpc_pdata_range_t* %17, i32 0, i32 1
  store i8 %14, i8* %18, align 1
  %19 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %20 = load i8, i8* %3, align 1
  %21 = sext i8 %20 to i32
  %22 = load i8, i8* %4, align 1
  %23 = sext i8 %22 to i32
  %24 = call %struct.mpc_parser_t* (%struct.mpc_parser_t*, i8*, ...) @mpc_expectf(%struct.mpc_parser_t* %19, i8* getelementptr inbounds ([32 x i8], [32 x i8]* @.str.15, i32 0, i32 0), i32 %21, i32 %23)
  ret %struct.mpc_parser_t* %24
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_oneof(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca %struct.mpc_parser_t*, align 8
  store i8* %0, i8** %2, align 8
  %4 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %4, %struct.mpc_parser_t** %3, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %5, i32 0, i32 2
  store i8 10, i8* %6, align 8
  %7 = load i8*, i8** %2, align 8
  %8 = call i64 @strlen(i8* %7) #7
  %9 = add i64 %8, 1
  %10 = call noalias i8* @malloc(i64 %9) #5
  %11 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %12 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %11, i32 0, i32 3
  %13 = bitcast %union.mpc_pdata_t* %12 to %struct.mpc_pdata_string_t*
  %14 = getelementptr inbounds %struct.mpc_pdata_string_t, %struct.mpc_pdata_string_t* %13, i32 0, i32 0
  store i8* %10, i8** %14, align 8
  %15 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %16 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %15, i32 0, i32 3
  %17 = bitcast %union.mpc_pdata_t* %16 to %struct.mpc_pdata_string_t*
  %18 = getelementptr inbounds %struct.mpc_pdata_string_t, %struct.mpc_pdata_string_t* %17, i32 0, i32 0
  %19 = load i8*, i8** %18, align 8
  %20 = load i8*, i8** %2, align 8
  %21 = call i8* @strcpy(i8* %19, i8* %20) #5
  %22 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %23 = load i8*, i8** %2, align 8
  %24 = call %struct.mpc_parser_t* (%struct.mpc_parser_t*, i8*, ...) @mpc_expectf(%struct.mpc_parser_t* %22, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.16, i32 0, i32 0), i8* %23)
  ret %struct.mpc_parser_t* %24
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_noneof(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca %struct.mpc_parser_t*, align 8
  store i8* %0, i8** %2, align 8
  %4 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %4, %struct.mpc_parser_t** %3, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %5, i32 0, i32 2
  store i8 11, i8* %6, align 8
  %7 = load i8*, i8** %2, align 8
  %8 = call i64 @strlen(i8* %7) #7
  %9 = add i64 %8, 1
  %10 = call noalias i8* @malloc(i64 %9) #5
  %11 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %12 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %11, i32 0, i32 3
  %13 = bitcast %union.mpc_pdata_t* %12 to %struct.mpc_pdata_string_t*
  %14 = getelementptr inbounds %struct.mpc_pdata_string_t, %struct.mpc_pdata_string_t* %13, i32 0, i32 0
  store i8* %10, i8** %14, align 8
  %15 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %16 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %15, i32 0, i32 3
  %17 = bitcast %union.mpc_pdata_t* %16 to %struct.mpc_pdata_string_t*
  %18 = getelementptr inbounds %struct.mpc_pdata_string_t, %struct.mpc_pdata_string_t* %17, i32 0, i32 0
  %19 = load i8*, i8** %18, align 8
  %20 = load i8*, i8** %2, align 8
  %21 = call i8* @strcpy(i8* %19, i8* %20) #5
  %22 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %23 = load i8*, i8** %2, align 8
  %24 = call %struct.mpc_parser_t* (%struct.mpc_parser_t*, i8*, ...) @mpc_expectf(%struct.mpc_parser_t* %22, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.17, i32 0, i32 0), i8* %23)
  ret %struct.mpc_parser_t* %24
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_satisfy(i32 (i8)*) #0 {
  %2 = alloca i32 (i8)*, align 8
  %3 = alloca %struct.mpc_parser_t*, align 8
  store i32 (i8)* %0, i32 (i8)** %2, align 8
  %4 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %4, %struct.mpc_parser_t** %3, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %5, i32 0, i32 2
  store i8 13, i8* %6, align 8
  %7 = load i32 (i8)*, i32 (i8)** %2, align 8
  %8 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %9 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %8, i32 0, i32 3
  %10 = bitcast %union.mpc_pdata_t* %9 to %struct.mpc_pdata_satisfy_t*
  %11 = getelementptr inbounds %struct.mpc_pdata_satisfy_t, %struct.mpc_pdata_satisfy_t* %10, i32 0, i32 0
  store i32 (i8)* %7, i32 (i8)** %11, align 8
  %12 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %13 = load i32 (i8)*, i32 (i8)** %2, align 8
  %14 = call %struct.mpc_parser_t* (%struct.mpc_parser_t*, i8*, ...) @mpc_expectf(%struct.mpc_parser_t* %12, i8* getelementptr inbounds ([33 x i8], [33 x i8]* @.str.18, i32 0, i32 0), i32 (i8)* %13)
  ret %struct.mpc_parser_t* %14
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_string(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca %struct.mpc_parser_t*, align 8
  store i8* %0, i8** %2, align 8
  %4 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %4, %struct.mpc_parser_t** %3, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %5, i32 0, i32 2
  store i8 14, i8* %6, align 8
  %7 = load i8*, i8** %2, align 8
  %8 = call i64 @strlen(i8* %7) #7
  %9 = add i64 %8, 1
  %10 = call noalias i8* @malloc(i64 %9) #5
  %11 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %12 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %11, i32 0, i32 3
  %13 = bitcast %union.mpc_pdata_t* %12 to %struct.mpc_pdata_string_t*
  %14 = getelementptr inbounds %struct.mpc_pdata_string_t, %struct.mpc_pdata_string_t* %13, i32 0, i32 0
  store i8* %10, i8** %14, align 8
  %15 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %16 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %15, i32 0, i32 3
  %17 = bitcast %union.mpc_pdata_t* %16 to %struct.mpc_pdata_string_t*
  %18 = getelementptr inbounds %struct.mpc_pdata_string_t, %struct.mpc_pdata_string_t* %17, i32 0, i32 0
  %19 = load i8*, i8** %18, align 8
  %20 = load i8*, i8** %2, align 8
  %21 = call i8* @strcpy(i8* %19, i8* %20) #5
  %22 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %23 = load i8*, i8** %2, align 8
  %24 = call %struct.mpc_parser_t* (%struct.mpc_parser_t*, i8*, ...) @mpc_expectf(%struct.mpc_parser_t* %22, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.19, i32 0, i32 0), i8* %23)
  ret %struct.mpc_parser_t* %24
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_apply(%struct.mpc_parser_t*, i8* (i8*)*) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca i8* (i8*)*, align 8
  %5 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store i8* (i8*)* %1, i8* (i8*)** %4, align 8
  %6 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %6, %struct.mpc_parser_t** %5, align 8
  %7 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %8 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %7, i32 0, i32 2
  store i8 15, i8* %8, align 8
  %9 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %10 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %11 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %10, i32 0, i32 3
  %12 = bitcast %union.mpc_pdata_t* %11 to %struct.mpc_pdata_apply_t*
  %13 = getelementptr inbounds %struct.mpc_pdata_apply_t, %struct.mpc_pdata_apply_t* %12, i32 0, i32 0
  store %struct.mpc_parser_t* %9, %struct.mpc_parser_t** %13, align 8
  %14 = load i8* (i8*)*, i8* (i8*)** %4, align 8
  %15 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %16 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %15, i32 0, i32 3
  %17 = bitcast %union.mpc_pdata_t* %16 to %struct.mpc_pdata_apply_t*
  %18 = getelementptr inbounds %struct.mpc_pdata_apply_t, %struct.mpc_pdata_apply_t* %17, i32 0, i32 1
  store i8* (i8*)* %14, i8* (i8*)** %18, align 8
  %19 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  ret %struct.mpc_parser_t* %19
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_apply_to(%struct.mpc_parser_t*, i8* (i8*, i8*)*, i8*) #0 {
  %4 = alloca %struct.mpc_parser_t*, align 8
  %5 = alloca i8* (i8*, i8*)*, align 8
  %6 = alloca i8*, align 8
  %7 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %4, align 8
  store i8* (i8*, i8*)* %1, i8* (i8*, i8*)** %5, align 8
  store i8* %2, i8** %6, align 8
  %8 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %8, %struct.mpc_parser_t** %7, align 8
  %9 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %10 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %9, i32 0, i32 2
  store i8 16, i8* %10, align 8
  %11 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %12 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %13 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %12, i32 0, i32 3
  %14 = bitcast %union.mpc_pdata_t* %13 to %struct.mpc_pdata_apply_to_t*
  %15 = getelementptr inbounds %struct.mpc_pdata_apply_to_t, %struct.mpc_pdata_apply_to_t* %14, i32 0, i32 0
  store %struct.mpc_parser_t* %11, %struct.mpc_parser_t** %15, align 8
  %16 = load i8* (i8*, i8*)*, i8* (i8*, i8*)** %5, align 8
  %17 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %18 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %17, i32 0, i32 3
  %19 = bitcast %union.mpc_pdata_t* %18 to %struct.mpc_pdata_apply_to_t*
  %20 = getelementptr inbounds %struct.mpc_pdata_apply_to_t, %struct.mpc_pdata_apply_to_t* %19, i32 0, i32 1
  store i8* (i8*, i8*)* %16, i8* (i8*, i8*)** %20, align 8
  %21 = load i8*, i8** %6, align 8
  %22 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %23 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %22, i32 0, i32 3
  %24 = bitcast %union.mpc_pdata_t* %23 to %struct.mpc_pdata_apply_to_t*
  %25 = getelementptr inbounds %struct.mpc_pdata_apply_to_t, %struct.mpc_pdata_apply_to_t* %24, i32 0, i32 2
  store i8* %21, i8** %25, align 8
  %26 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  ret %struct.mpc_parser_t* %26
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_predictive(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  %3 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  %4 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %4, %struct.mpc_parser_t** %3, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %5, i32 0, i32 2
  store i8 17, i8* %6, align 8
  %7 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %8 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %9 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %8, i32 0, i32 3
  %10 = bitcast %union.mpc_pdata_t* %9 to %struct.mpc_pdata_predict_t*
  %11 = getelementptr inbounds %struct.mpc_pdata_predict_t, %struct.mpc_pdata_predict_t* %10, i32 0, i32 0
  store %struct.mpc_parser_t* %7, %struct.mpc_parser_t** %11, align 8
  %12 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  ret %struct.mpc_parser_t* %12
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_not_lift(%struct.mpc_parser_t*, void (i8*)*, i8* ()*) #0 {
  %4 = alloca %struct.mpc_parser_t*, align 8
  %5 = alloca void (i8*)*, align 8
  %6 = alloca i8* ()*, align 8
  %7 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %4, align 8
  store void (i8*)* %1, void (i8*)** %5, align 8
  store i8* ()* %2, i8* ()** %6, align 8
  %8 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %8, %struct.mpc_parser_t** %7, align 8
  %9 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %10 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %9, i32 0, i32 2
  store i8 18, i8* %10, align 8
  %11 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %12 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %13 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %12, i32 0, i32 3
  %14 = bitcast %union.mpc_pdata_t* %13 to %struct.mpc_pdata_not_t*
  %15 = getelementptr inbounds %struct.mpc_pdata_not_t, %struct.mpc_pdata_not_t* %14, i32 0, i32 0
  store %struct.mpc_parser_t* %11, %struct.mpc_parser_t** %15, align 8
  %16 = load void (i8*)*, void (i8*)** %5, align 8
  %17 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %18 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %17, i32 0, i32 3
  %19 = bitcast %union.mpc_pdata_t* %18 to %struct.mpc_pdata_not_t*
  %20 = getelementptr inbounds %struct.mpc_pdata_not_t, %struct.mpc_pdata_not_t* %19, i32 0, i32 1
  store void (i8*)* %16, void (i8*)** %20, align 8
  %21 = load i8* ()*, i8* ()** %6, align 8
  %22 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %23 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %22, i32 0, i32 3
  %24 = bitcast %union.mpc_pdata_t* %23 to %struct.mpc_pdata_not_t*
  %25 = getelementptr inbounds %struct.mpc_pdata_not_t, %struct.mpc_pdata_not_t* %24, i32 0, i32 2
  store i8* ()* %21, i8* ()** %25, align 8
  %26 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  ret %struct.mpc_parser_t* %26
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_not(%struct.mpc_parser_t*, void (i8*)*) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca void (i8*)*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store void (i8*)* %1, void (i8*)** %4, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = load void (i8*)*, void (i8*)** %4, align 8
  %7 = call %struct.mpc_parser_t* @mpc_not_lift(%struct.mpc_parser_t* %5, void (i8*)* %6, i8* ()* @mpcf_ctor_null)
  ret %struct.mpc_parser_t* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_ctor_null() #0 {
  ret i8* null
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_maybe_lift(%struct.mpc_parser_t*, i8* ()*) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca i8* ()*, align 8
  %5 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store i8* ()* %1, i8* ()** %4, align 8
  %6 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %6, %struct.mpc_parser_t** %5, align 8
  %7 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %8 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %7, i32 0, i32 2
  store i8 19, i8* %8, align 8
  %9 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %10 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %11 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %10, i32 0, i32 3
  %12 = bitcast %union.mpc_pdata_t* %11 to %struct.mpc_pdata_not_t*
  %13 = getelementptr inbounds %struct.mpc_pdata_not_t, %struct.mpc_pdata_not_t* %12, i32 0, i32 0
  store %struct.mpc_parser_t* %9, %struct.mpc_parser_t** %13, align 8
  %14 = load i8* ()*, i8* ()** %4, align 8
  %15 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %16 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %15, i32 0, i32 3
  %17 = bitcast %union.mpc_pdata_t* %16 to %struct.mpc_pdata_not_t*
  %18 = getelementptr inbounds %struct.mpc_pdata_not_t, %struct.mpc_pdata_not_t* %17, i32 0, i32 2
  store i8* ()* %14, i8* ()** %18, align 8
  %19 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  ret %struct.mpc_parser_t* %19
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_maybe(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  %3 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %4 = call %struct.mpc_parser_t* @mpc_maybe_lift(%struct.mpc_parser_t* %3, i8* ()* @mpcf_ctor_null)
  ret %struct.mpc_parser_t* %4
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_many(i8* (i32, i8**)*, %struct.mpc_parser_t*) #0 {
  %3 = alloca i8* (i32, i8**)*, align 8
  %4 = alloca %struct.mpc_parser_t*, align 8
  %5 = alloca %struct.mpc_parser_t*, align 8
  store i8* (i32, i8**)* %0, i8* (i32, i8**)** %3, align 8
  store %struct.mpc_parser_t* %1, %struct.mpc_parser_t** %4, align 8
  %6 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %6, %struct.mpc_parser_t** %5, align 8
  %7 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %8 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %7, i32 0, i32 2
  store i8 20, i8* %8, align 8
  %9 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %10 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %11 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %10, i32 0, i32 3
  %12 = bitcast %union.mpc_pdata_t* %11 to %struct.mpc_pdata_repeat_t*
  %13 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %12, i32 0, i32 2
  store %struct.mpc_parser_t* %9, %struct.mpc_parser_t** %13, align 8
  %14 = load i8* (i32, i8**)*, i8* (i32, i8**)** %3, align 8
  %15 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %16 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %15, i32 0, i32 3
  %17 = bitcast %union.mpc_pdata_t* %16 to %struct.mpc_pdata_repeat_t*
  %18 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %17, i32 0, i32 1
  store i8* (i32, i8**)* %14, i8* (i32, i8**)** %18, align 8
  %19 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  ret %struct.mpc_parser_t* %19
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_many1(i8* (i32, i8**)*, %struct.mpc_parser_t*) #0 {
  %3 = alloca i8* (i32, i8**)*, align 8
  %4 = alloca %struct.mpc_parser_t*, align 8
  %5 = alloca %struct.mpc_parser_t*, align 8
  store i8* (i32, i8**)* %0, i8* (i32, i8**)** %3, align 8
  store %struct.mpc_parser_t* %1, %struct.mpc_parser_t** %4, align 8
  %6 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %6, %struct.mpc_parser_t** %5, align 8
  %7 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %8 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %7, i32 0, i32 2
  store i8 21, i8* %8, align 8
  %9 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %10 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %11 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %10, i32 0, i32 3
  %12 = bitcast %union.mpc_pdata_t* %11 to %struct.mpc_pdata_repeat_t*
  %13 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %12, i32 0, i32 2
  store %struct.mpc_parser_t* %9, %struct.mpc_parser_t** %13, align 8
  %14 = load i8* (i32, i8**)*, i8* (i32, i8**)** %3, align 8
  %15 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %16 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %15, i32 0, i32 3
  %17 = bitcast %union.mpc_pdata_t* %16 to %struct.mpc_pdata_repeat_t*
  %18 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %17, i32 0, i32 1
  store i8* (i32, i8**)* %14, i8* (i32, i8**)** %18, align 8
  %19 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  ret %struct.mpc_parser_t* %19
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_count(i32, i8* (i32, i8**)*, %struct.mpc_parser_t*, void (i8*)*) #0 {
  %5 = alloca i32, align 4
  %6 = alloca i8* (i32, i8**)*, align 8
  %7 = alloca %struct.mpc_parser_t*, align 8
  %8 = alloca void (i8*)*, align 8
  %9 = alloca %struct.mpc_parser_t*, align 8
  store i32 %0, i32* %5, align 4
  store i8* (i32, i8**)* %1, i8* (i32, i8**)** %6, align 8
  store %struct.mpc_parser_t* %2, %struct.mpc_parser_t** %7, align 8
  store void (i8*)* %3, void (i8*)** %8, align 8
  %10 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %10, %struct.mpc_parser_t** %9, align 8
  %11 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  %12 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %11, i32 0, i32 2
  store i8 22, i8* %12, align 8
  %13 = load i32, i32* %5, align 4
  %14 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  %15 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %14, i32 0, i32 3
  %16 = bitcast %union.mpc_pdata_t* %15 to %struct.mpc_pdata_repeat_t*
  %17 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %16, i32 0, i32 0
  store i32 %13, i32* %17, align 8
  %18 = load i8* (i32, i8**)*, i8* (i32, i8**)** %6, align 8
  %19 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  %20 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %19, i32 0, i32 3
  %21 = bitcast %union.mpc_pdata_t* %20 to %struct.mpc_pdata_repeat_t*
  %22 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %21, i32 0, i32 1
  store i8* (i32, i8**)* %18, i8* (i32, i8**)** %22, align 8
  %23 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %24 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  %25 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %24, i32 0, i32 3
  %26 = bitcast %union.mpc_pdata_t* %25 to %struct.mpc_pdata_repeat_t*
  %27 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %26, i32 0, i32 2
  store %struct.mpc_parser_t* %23, %struct.mpc_parser_t** %27, align 8
  %28 = load void (i8*)*, void (i8*)** %8, align 8
  %29 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  %30 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %29, i32 0, i32 3
  %31 = bitcast %union.mpc_pdata_t* %30 to %struct.mpc_pdata_repeat_t*
  %32 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %31, i32 0, i32 3
  store void (i8*)* %28, void (i8*)** %32, align 8
  %33 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  ret %struct.mpc_parser_t* %33
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_or(i32, ...) #0 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca [1 x %struct.__va_list_tag], align 16
  %5 = alloca %struct.mpc_parser_t*, align 8
  store i32 %0, i32* %2, align 4
  %6 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %6, %struct.mpc_parser_t** %5, align 8
  %7 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %8 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %7, i32 0, i32 2
  store i8 23, i8* %8, align 8
  %9 = load i32, i32* %2, align 4
  %10 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %11 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %10, i32 0, i32 3
  %12 = bitcast %union.mpc_pdata_t* %11 to %struct.mpc_pdata_or_t*
  %13 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %12, i32 0, i32 0
  store i32 %9, i32* %13, align 8
  %14 = load i32, i32* %2, align 4
  %15 = sext i32 %14 to i64
  %16 = mul i64 8, %15
  %17 = call noalias i8* @malloc(i64 %16) #5
  %18 = bitcast i8* %17 to %struct.mpc_parser_t**
  %19 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %20 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %19, i32 0, i32 3
  %21 = bitcast %union.mpc_pdata_t* %20 to %struct.mpc_pdata_or_t*
  %22 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %21, i32 0, i32 1
  store %struct.mpc_parser_t** %18, %struct.mpc_parser_t*** %22, align 8
  %23 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %4, i32 0, i32 0
  %24 = bitcast %struct.__va_list_tag* %23 to i8*
  call void @llvm.va_start(i8* %24)
  store i32 0, i32* %3, align 4
  br label %25

; <label>:25:                                     ; preds = %56, %1
  %26 = load i32, i32* %3, align 4
  %27 = load i32, i32* %2, align 4
  %28 = icmp slt i32 %26, %27
  br i1 %28, label %29, label %59

; <label>:29:                                     ; preds = %25
  %30 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %4, i32 0, i32 0
  %31 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %30, i32 0, i32 0
  %32 = load i32, i32* %31, align 16
  %33 = icmp ule i32 %32, 40
  br i1 %33, label %34, label %40

; <label>:34:                                     ; preds = %29
  %35 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %30, i32 0, i32 3
  %36 = load i8*, i8** %35, align 16
  %37 = getelementptr i8, i8* %36, i32 %32
  %38 = bitcast i8* %37 to %struct.mpc_parser_t**
  %39 = add i32 %32, 8
  store i32 %39, i32* %31, align 16
  br label %45

; <label>:40:                                     ; preds = %29
  %41 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %30, i32 0, i32 2
  %42 = load i8*, i8** %41, align 8
  %43 = bitcast i8* %42 to %struct.mpc_parser_t**
  %44 = getelementptr i8, i8* %42, i32 8
  store i8* %44, i8** %41, align 8
  br label %45

; <label>:45:                                     ; preds = %40, %34
  %46 = phi %struct.mpc_parser_t** [ %38, %34 ], [ %43, %40 ]
  %47 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %46, align 8
  %48 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %49 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %48, i32 0, i32 3
  %50 = bitcast %union.mpc_pdata_t* %49 to %struct.mpc_pdata_or_t*
  %51 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %50, i32 0, i32 1
  %52 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %51, align 8
  %53 = load i32, i32* %3, align 4
  %54 = sext i32 %53 to i64
  %55 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %52, i64 %54
  store %struct.mpc_parser_t* %47, %struct.mpc_parser_t** %55, align 8
  br label %56

; <label>:56:                                     ; preds = %45
  %57 = load i32, i32* %3, align 4
  %58 = add nsw i32 %57, 1
  store i32 %58, i32* %3, align 4
  br label %25

; <label>:59:                                     ; preds = %25
  %60 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %4, i32 0, i32 0
  %61 = bitcast %struct.__va_list_tag* %60 to i8*
  call void @llvm.va_end(i8* %61)
  %62 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  ret %struct.mpc_parser_t* %62
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_and(i32, i8* (i32, i8**)*, ...) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i8* (i32, i8**)*, align 8
  %5 = alloca i32, align 4
  %6 = alloca [1 x %struct.__va_list_tag], align 16
  %7 = alloca %struct.mpc_parser_t*, align 8
  store i32 %0, i32* %3, align 4
  store i8* (i32, i8**)* %1, i8* (i32, i8**)** %4, align 8
  %8 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %8, %struct.mpc_parser_t** %7, align 8
  %9 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %10 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %9, i32 0, i32 2
  store i8 24, i8* %10, align 8
  %11 = load i32, i32* %3, align 4
  %12 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %13 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %12, i32 0, i32 3
  %14 = bitcast %union.mpc_pdata_t* %13 to %struct.mpc_pdata_and_t*
  %15 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %14, i32 0, i32 0
  store i32 %11, i32* %15, align 8
  %16 = load i8* (i32, i8**)*, i8* (i32, i8**)** %4, align 8
  %17 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %18 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %17, i32 0, i32 3
  %19 = bitcast %union.mpc_pdata_t* %18 to %struct.mpc_pdata_and_t*
  %20 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %19, i32 0, i32 1
  store i8* (i32, i8**)* %16, i8* (i32, i8**)** %20, align 8
  %21 = load i32, i32* %3, align 4
  %22 = sext i32 %21 to i64
  %23 = mul i64 8, %22
  %24 = call noalias i8* @malloc(i64 %23) #5
  %25 = bitcast i8* %24 to %struct.mpc_parser_t**
  %26 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %27 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %26, i32 0, i32 3
  %28 = bitcast %union.mpc_pdata_t* %27 to %struct.mpc_pdata_and_t*
  %29 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %28, i32 0, i32 2
  store %struct.mpc_parser_t** %25, %struct.mpc_parser_t*** %29, align 8
  %30 = load i32, i32* %3, align 4
  %31 = sub nsw i32 %30, 1
  %32 = sext i32 %31 to i64
  %33 = mul i64 8, %32
  %34 = call noalias i8* @malloc(i64 %33) #5
  %35 = bitcast i8* %34 to void (i8*)**
  %36 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %37 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %36, i32 0, i32 3
  %38 = bitcast %union.mpc_pdata_t* %37 to %struct.mpc_pdata_and_t*
  %39 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %38, i32 0, i32 3
  store void (i8*)** %35, void (i8*)*** %39, align 8
  %40 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %6, i32 0, i32 0
  %41 = bitcast %struct.__va_list_tag* %40 to i8*
  call void @llvm.va_start(i8* %41)
  store i32 0, i32* %5, align 4
  br label %42

; <label>:42:                                     ; preds = %73, %2
  %43 = load i32, i32* %5, align 4
  %44 = load i32, i32* %3, align 4
  %45 = icmp slt i32 %43, %44
  br i1 %45, label %46, label %76

; <label>:46:                                     ; preds = %42
  %47 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %6, i32 0, i32 0
  %48 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %47, i32 0, i32 0
  %49 = load i32, i32* %48, align 16
  %50 = icmp ule i32 %49, 40
  br i1 %50, label %51, label %57

; <label>:51:                                     ; preds = %46
  %52 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %47, i32 0, i32 3
  %53 = load i8*, i8** %52, align 16
  %54 = getelementptr i8, i8* %53, i32 %49
  %55 = bitcast i8* %54 to %struct.mpc_parser_t**
  %56 = add i32 %49, 8
  store i32 %56, i32* %48, align 16
  br label %62

; <label>:57:                                     ; preds = %46
  %58 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %47, i32 0, i32 2
  %59 = load i8*, i8** %58, align 8
  %60 = bitcast i8* %59 to %struct.mpc_parser_t**
  %61 = getelementptr i8, i8* %59, i32 8
  store i8* %61, i8** %58, align 8
  br label %62

; <label>:62:                                     ; preds = %57, %51
  %63 = phi %struct.mpc_parser_t** [ %55, %51 ], [ %60, %57 ]
  %64 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %63, align 8
  %65 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %66 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %65, i32 0, i32 3
  %67 = bitcast %union.mpc_pdata_t* %66 to %struct.mpc_pdata_and_t*
  %68 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %67, i32 0, i32 2
  %69 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %68, align 8
  %70 = load i32, i32* %5, align 4
  %71 = sext i32 %70 to i64
  %72 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %69, i64 %71
  store %struct.mpc_parser_t* %64, %struct.mpc_parser_t** %72, align 8
  br label %73

; <label>:73:                                     ; preds = %62
  %74 = load i32, i32* %5, align 4
  %75 = add nsw i32 %74, 1
  store i32 %75, i32* %5, align 4
  br label %42

; <label>:76:                                     ; preds = %42
  store i32 0, i32* %5, align 4
  br label %77

; <label>:77:                                     ; preds = %109, %76
  %78 = load i32, i32* %5, align 4
  %79 = load i32, i32* %3, align 4
  %80 = sub nsw i32 %79, 1
  %81 = icmp slt i32 %78, %80
  br i1 %81, label %82, label %112

; <label>:82:                                     ; preds = %77
  %83 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %6, i32 0, i32 0
  %84 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %83, i32 0, i32 0
  %85 = load i32, i32* %84, align 16
  %86 = icmp ule i32 %85, 40
  br i1 %86, label %87, label %93

; <label>:87:                                     ; preds = %82
  %88 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %83, i32 0, i32 3
  %89 = load i8*, i8** %88, align 16
  %90 = getelementptr i8, i8* %89, i32 %85
  %91 = bitcast i8* %90 to void (i8*)**
  %92 = add i32 %85, 8
  store i32 %92, i32* %84, align 16
  br label %98

; <label>:93:                                     ; preds = %82
  %94 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %83, i32 0, i32 2
  %95 = load i8*, i8** %94, align 8
  %96 = bitcast i8* %95 to void (i8*)**
  %97 = getelementptr i8, i8* %95, i32 8
  store i8* %97, i8** %94, align 8
  br label %98

; <label>:98:                                     ; preds = %93, %87
  %99 = phi void (i8*)** [ %91, %87 ], [ %96, %93 ]
  %100 = load void (i8*)*, void (i8*)** %99, align 8
  %101 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %102 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %101, i32 0, i32 3
  %103 = bitcast %union.mpc_pdata_t* %102 to %struct.mpc_pdata_and_t*
  %104 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %103, i32 0, i32 3
  %105 = load void (i8*)**, void (i8*)*** %104, align 8
  %106 = load i32, i32* %5, align 4
  %107 = sext i32 %106 to i64
  %108 = getelementptr inbounds void (i8*)*, void (i8*)** %105, i64 %107
  store void (i8*)* %100, void (i8*)** %108, align 8
  br label %109

; <label>:109:                                    ; preds = %98
  %110 = load i32, i32* %5, align 4
  %111 = add nsw i32 %110, 1
  store i32 %111, i32* %5, align 4
  br label %77

; <label>:112:                                    ; preds = %77
  %113 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %6, i32 0, i32 0
  %114 = bitcast %struct.__va_list_tag* %113 to i8*
  call void @llvm.va_end(i8* %114)
  %115 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  ret %struct.mpc_parser_t* %115
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_soi() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_anchor(i32 (i8, i8)* @mpc_soi_anchor)
  %2 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %1, i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.20, i32 0, i32 0))
  ret %struct.mpc_parser_t* %2
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @mpc_soi_anchor(i8 signext, i8 signext) #0 {
  %3 = alloca i8, align 1
  %4 = alloca i8, align 1
  store i8 %0, i8* %3, align 1
  store i8 %1, i8* %4, align 1
  %5 = load i8, i8* %4, align 1
  %6 = load i8, i8* %3, align 1
  %7 = sext i8 %6 to i32
  %8 = icmp eq i32 %7, 0
  %9 = zext i1 %8 to i32
  ret i32 %9
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_eoi() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_anchor(i32 (i8, i8)* @mpc_eoi_anchor)
  %2 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %1, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.21, i32 0, i32 0))
  ret %struct.mpc_parser_t* %2
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @mpc_eoi_anchor(i8 signext, i8 signext) #0 {
  %3 = alloca i8, align 1
  %4 = alloca i8, align 1
  store i8 %0, i8* %3, align 1
  store i8 %1, i8* %4, align 1
  %5 = load i8, i8* %3, align 1
  %6 = load i8, i8* %4, align 1
  %7 = sext i8 %6 to i32
  %8 = icmp eq i32 %7, 0
  %9 = zext i1 %8 to i32
  ret i32 %9
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_boundary() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_anchor(i32 (i8, i8)* @mpc_boundary_anchor)
  %2 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %1, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.22, i32 0, i32 0))
  ret %struct.mpc_parser_t* %2
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @mpc_boundary_anchor(i8 signext, i8 signext) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i8, align 1
  %5 = alloca i8, align 1
  %6 = alloca i8*, align 8
  store i8 %0, i8* %4, align 1
  store i8 %1, i8* %5, align 1
  store i8* getelementptr inbounds ([64 x i8], [64 x i8]* @.str.108, i32 0, i32 0), i8** %6, align 8
  %7 = load i8*, i8** %6, align 8
  %8 = load i8, i8* %5, align 1
  %9 = sext i8 %8 to i32
  %10 = call i8* @strchr(i8* %7, i32 %9) #7
  %11 = icmp ne i8* %10, null
  br i1 %11, label %12, label %17

; <label>:12:                                     ; preds = %2
  %13 = load i8, i8* %4, align 1
  %14 = sext i8 %13 to i32
  %15 = icmp eq i32 %14, 0
  br i1 %15, label %16, label %17

; <label>:16:                                     ; preds = %12
  store i32 1, i32* %3, align 4
  br label %55

; <label>:17:                                     ; preds = %12, %2
  %18 = load i8*, i8** %6, align 8
  %19 = load i8, i8* %4, align 1
  %20 = sext i8 %19 to i32
  %21 = call i8* @strchr(i8* %18, i32 %20) #7
  %22 = icmp ne i8* %21, null
  br i1 %22, label %23, label %28

; <label>:23:                                     ; preds = %17
  %24 = load i8, i8* %5, align 1
  %25 = sext i8 %24 to i32
  %26 = icmp eq i32 %25, 0
  br i1 %26, label %27, label %28

; <label>:27:                                     ; preds = %23
  store i32 1, i32* %3, align 4
  br label %55

; <label>:28:                                     ; preds = %23, %17
  %29 = load i8*, i8** %6, align 8
  %30 = load i8, i8* %5, align 1
  %31 = sext i8 %30 to i32
  %32 = call i8* @strchr(i8* %29, i32 %31) #7
  %33 = icmp ne i8* %32, null
  br i1 %33, label %34, label %41

; <label>:34:                                     ; preds = %28
  %35 = load i8*, i8** %6, align 8
  %36 = load i8, i8* %4, align 1
  %37 = sext i8 %36 to i32
  %38 = call i8* @strchr(i8* %35, i32 %37) #7
  %39 = icmp ne i8* %38, null
  br i1 %39, label %41, label %40

; <label>:40:                                     ; preds = %34
  store i32 1, i32* %3, align 4
  br label %55

; <label>:41:                                     ; preds = %34, %28
  %42 = load i8*, i8** %6, align 8
  %43 = load i8, i8* %5, align 1
  %44 = sext i8 %43 to i32
  %45 = call i8* @strchr(i8* %42, i32 %44) #7
  %46 = icmp ne i8* %45, null
  br i1 %46, label %54, label %47

; <label>:47:                                     ; preds = %41
  %48 = load i8*, i8** %6, align 8
  %49 = load i8, i8* %4, align 1
  %50 = sext i8 %49 to i32
  %51 = call i8* @strchr(i8* %48, i32 %50) #7
  %52 = icmp ne i8* %51, null
  br i1 %52, label %53, label %54

; <label>:53:                                     ; preds = %47
  store i32 1, i32* %3, align 4
  br label %55

; <label>:54:                                     ; preds = %47, %41
  store i32 0, i32* %3, align 4
  br label %55

; <label>:55:                                     ; preds = %54, %53, %40, %27, %16
  %56 = load i32, i32* %3, align 4
  ret i32 %56
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_whitespace() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_oneof(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.23, i32 0, i32 0))
  %2 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %1, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.24, i32 0, i32 0))
  ret %struct.mpc_parser_t* %2
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_whitespaces() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_whitespace()
  %2 = call %struct.mpc_parser_t* @mpc_many(i8* (i32, i8**)* @mpcf_strfold, %struct.mpc_parser_t* %1)
  %3 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %2, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.25, i32 0, i32 0))
  ret %struct.mpc_parser_t* %3
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_strfold(i32, i8**) #0 {
  %3 = alloca i8*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i8**, align 8
  %6 = alloca i32, align 4
  %7 = alloca i64, align 8
  store i32 %0, i32* %4, align 4
  store i8** %1, i8*** %5, align 8
  store i64 0, i64* %7, align 8
  %8 = load i32, i32* %4, align 4
  %9 = icmp eq i32 %8, 0
  br i1 %9, label %10, label %12

; <label>:10:                                     ; preds = %2
  %11 = call noalias i8* @calloc(i64 1, i64 1) #5
  store i8* %11, i8** %3, align 8
  br label %64

; <label>:12:                                     ; preds = %2
  store i32 0, i32* %6, align 4
  br label %13

; <label>:13:                                     ; preds = %26, %12
  %14 = load i32, i32* %6, align 4
  %15 = load i32, i32* %4, align 4
  %16 = icmp slt i32 %14, %15
  br i1 %16, label %17, label %29

; <label>:17:                                     ; preds = %13
  %18 = load i8**, i8*** %5, align 8
  %19 = load i32, i32* %6, align 4
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds i8*, i8** %18, i64 %20
  %22 = load i8*, i8** %21, align 8
  %23 = call i64 @strlen(i8* %22) #7
  %24 = load i64, i64* %7, align 8
  %25 = add i64 %24, %23
  store i64 %25, i64* %7, align 8
  br label %26

; <label>:26:                                     ; preds = %17
  %27 = load i32, i32* %6, align 4
  %28 = add nsw i32 %27, 1
  store i32 %28, i32* %6, align 4
  br label %13

; <label>:29:                                     ; preds = %13
  %30 = load i8**, i8*** %5, align 8
  %31 = getelementptr inbounds i8*, i8** %30, i64 0
  %32 = load i8*, i8** %31, align 8
  %33 = load i64, i64* %7, align 8
  %34 = add i64 %33, 1
  %35 = call i8* @realloc(i8* %32, i64 %34) #5
  %36 = load i8**, i8*** %5, align 8
  %37 = getelementptr inbounds i8*, i8** %36, i64 0
  store i8* %35, i8** %37, align 8
  store i32 1, i32* %6, align 4
  br label %38

; <label>:38:                                     ; preds = %57, %29
  %39 = load i32, i32* %6, align 4
  %40 = load i32, i32* %4, align 4
  %41 = icmp slt i32 %39, %40
  br i1 %41, label %42, label %60

; <label>:42:                                     ; preds = %38
  %43 = load i8**, i8*** %5, align 8
  %44 = getelementptr inbounds i8*, i8** %43, i64 0
  %45 = load i8*, i8** %44, align 8
  %46 = load i8**, i8*** %5, align 8
  %47 = load i32, i32* %6, align 4
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds i8*, i8** %46, i64 %48
  %50 = load i8*, i8** %49, align 8
  %51 = call i8* @strcat(i8* %45, i8* %50) #5
  %52 = load i8**, i8*** %5, align 8
  %53 = load i32, i32* %6, align 4
  %54 = sext i32 %53 to i64
  %55 = getelementptr inbounds i8*, i8** %52, i64 %54
  %56 = load i8*, i8** %55, align 8
  call void @free(i8* %56) #5
  br label %57

; <label>:57:                                     ; preds = %42
  %58 = load i32, i32* %6, align 4
  %59 = add nsw i32 %58, 1
  store i32 %59, i32* %6, align 4
  br label %38

; <label>:60:                                     ; preds = %38
  %61 = load i8**, i8*** %5, align 8
  %62 = getelementptr inbounds i8*, i8** %61, i64 0
  %63 = load i8*, i8** %62, align 8
  store i8* %63, i8** %3, align 8
  br label %64

; <label>:64:                                     ; preds = %60, %10
  %65 = load i8*, i8** %3, align 8
  ret i8* %65
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_blank() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_whitespaces()
  %2 = call %struct.mpc_parser_t* @mpc_apply(%struct.mpc_parser_t* %1, i8* (i8*)* @mpcf_free)
  %3 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %2, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.24, i32 0, i32 0))
  ret %struct.mpc_parser_t* %3
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_free(i8*) #0 {
  %2 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %3 = load i8*, i8** %2, align 8
  call void @free(i8* %3) #5
  ret i8* null
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_newline() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_char(i8 signext 10)
  %2 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %1, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.26, i32 0, i32 0))
  ret %struct.mpc_parser_t* %2
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_tab() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_char(i8 signext 9)
  %2 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %1, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.27, i32 0, i32 0))
  ret %struct.mpc_parser_t* %2
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_escape() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_char(i8 signext 92)
  %2 = call %struct.mpc_parser_t* @mpc_any()
  %3 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcf_strfold, %struct.mpc_parser_t* %1, %struct.mpc_parser_t* %2, void (i8*)* @free)
  ret %struct.mpc_parser_t* %3
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_digit() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_oneof(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.28, i32 0, i32 0))
  %2 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %1, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.29, i32 0, i32 0))
  ret %struct.mpc_parser_t* %2
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_hexdigit() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_oneof(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.30, i32 0, i32 0))
  %2 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %1, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.31, i32 0, i32 0))
  ret %struct.mpc_parser_t* %2
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_octdigit() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_oneof(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.32, i32 0, i32 0))
  %2 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %1, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.33, i32 0, i32 0))
  ret %struct.mpc_parser_t* %2
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_digits() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_digit()
  %2 = call %struct.mpc_parser_t* @mpc_many1(i8* (i32, i8**)* @mpcf_strfold, %struct.mpc_parser_t* %1)
  %3 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %2, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.34, i32 0, i32 0))
  ret %struct.mpc_parser_t* %3
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_hexdigits() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_hexdigit()
  %2 = call %struct.mpc_parser_t* @mpc_many1(i8* (i32, i8**)* @mpcf_strfold, %struct.mpc_parser_t* %1)
  %3 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %2, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.35, i32 0, i32 0))
  ret %struct.mpc_parser_t* %3
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_octdigits() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_octdigit()
  %2 = call %struct.mpc_parser_t* @mpc_many1(i8* (i32, i8**)* @mpcf_strfold, %struct.mpc_parser_t* %1)
  %3 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %2, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.36, i32 0, i32 0))
  ret %struct.mpc_parser_t* %3
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_lower() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_oneof(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.37, i32 0, i32 0))
  %2 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %1, i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.38, i32 0, i32 0))
  ret %struct.mpc_parser_t* %2
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_upper() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_oneof(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str.39, i32 0, i32 0))
  %2 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %1, i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.40, i32 0, i32 0))
  ret %struct.mpc_parser_t* %2
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_alpha() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_oneof(i8* getelementptr inbounds ([53 x i8], [53 x i8]* @.str.41, i32 0, i32 0))
  %2 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %1, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.42, i32 0, i32 0))
  ret %struct.mpc_parser_t* %2
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_underscore() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_char(i8 signext 95)
  %2 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %1, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.43, i32 0, i32 0))
  ret %struct.mpc_parser_t* %2
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_alphanum() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_alpha()
  %2 = call %struct.mpc_parser_t* @mpc_digit()
  %3 = call %struct.mpc_parser_t* @mpc_underscore()
  %4 = call %struct.mpc_parser_t* (i32, ...) @mpc_or(i32 3, %struct.mpc_parser_t* %1, %struct.mpc_parser_t* %2, %struct.mpc_parser_t* %3)
  %5 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %4, i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.44, i32 0, i32 0))
  ret %struct.mpc_parser_t* %5
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_int() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_digits()
  %2 = call %struct.mpc_parser_t* @mpc_apply(%struct.mpc_parser_t* %1, i8* (i8*)* @mpcf_int)
  %3 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %2, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.45, i32 0, i32 0))
  ret %struct.mpc_parser_t* %3
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_int(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca i32*, align 8
  store i8* %0, i8** %2, align 8
  %4 = call noalias i8* @malloc(i64 4) #5
  %5 = bitcast i8* %4 to i32*
  store i32* %5, i32** %3, align 8
  %6 = load i8*, i8** %2, align 8
  %7 = call i64 @strtol(i8* %6, i8** null, i32 10) #5
  %8 = trunc i64 %7 to i32
  %9 = load i32*, i32** %3, align 8
  store i32 %8, i32* %9, align 4
  %10 = load i8*, i8** %2, align 8
  call void @free(i8* %10) #5
  %11 = load i32*, i32** %3, align 8
  %12 = bitcast i32* %11 to i8*
  ret i8* %12
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_hex() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_hexdigits()
  %2 = call %struct.mpc_parser_t* @mpc_apply(%struct.mpc_parser_t* %1, i8* (i8*)* @mpcf_hex)
  %3 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %2, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.46, i32 0, i32 0))
  ret %struct.mpc_parser_t* %3
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_hex(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca i32*, align 8
  store i8* %0, i8** %2, align 8
  %4 = call noalias i8* @malloc(i64 4) #5
  %5 = bitcast i8* %4 to i32*
  store i32* %5, i32** %3, align 8
  %6 = load i8*, i8** %2, align 8
  %7 = call i64 @strtol(i8* %6, i8** null, i32 16) #5
  %8 = trunc i64 %7 to i32
  %9 = load i32*, i32** %3, align 8
  store i32 %8, i32* %9, align 4
  %10 = load i8*, i8** %2, align 8
  call void @free(i8* %10) #5
  %11 = load i32*, i32** %3, align 8
  %12 = bitcast i32* %11 to i8*
  ret i8* %12
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_oct() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_octdigits()
  %2 = call %struct.mpc_parser_t* @mpc_apply(%struct.mpc_parser_t* %1, i8* (i8*)* @mpcf_oct)
  %3 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %2, i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.47, i32 0, i32 0))
  ret %struct.mpc_parser_t* %3
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_oct(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca i32*, align 8
  store i8* %0, i8** %2, align 8
  %4 = call noalias i8* @malloc(i64 4) #5
  %5 = bitcast i8* %4 to i32*
  store i32* %5, i32** %3, align 8
  %6 = load i8*, i8** %2, align 8
  %7 = call i64 @strtol(i8* %6, i8** null, i32 8) #5
  %8 = trunc i64 %7 to i32
  %9 = load i32*, i32** %3, align 8
  store i32 %8, i32* %9, align 4
  %10 = load i8*, i8** %2, align 8
  call void @free(i8* %10) #5
  %11 = load i32*, i32** %3, align 8
  %12 = bitcast i32* %11 to i8*
  ret i8* %12
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_number() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_int()
  %2 = call %struct.mpc_parser_t* @mpc_hex()
  %3 = call %struct.mpc_parser_t* @mpc_oct()
  %4 = call %struct.mpc_parser_t* (i32, ...) @mpc_or(i32 3, %struct.mpc_parser_t* %1, %struct.mpc_parser_t* %2, %struct.mpc_parser_t* %3)
  %5 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %4, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.48, i32 0, i32 0))
  ret %struct.mpc_parser_t* %5
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_real() #0 {
  %1 = alloca %struct.mpc_parser_t*, align 8
  %2 = alloca %struct.mpc_parser_t*, align 8
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca %struct.mpc_parser_t*, align 8
  %5 = alloca %struct.mpc_parser_t*, align 8
  %6 = alloca %struct.mpc_parser_t*, align 8
  %7 = alloca %struct.mpc_parser_t*, align 8
  %8 = call %struct.mpc_parser_t* @mpc_oneof(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.49, i32 0, i32 0))
  %9 = call %struct.mpc_parser_t* @mpc_maybe_lift(%struct.mpc_parser_t* %8, i8* ()* @mpcf_ctor_str)
  store %struct.mpc_parser_t* %9, %struct.mpc_parser_t** %1, align 8
  %10 = call %struct.mpc_parser_t* @mpc_digits()
  store %struct.mpc_parser_t* %10, %struct.mpc_parser_t** %2, align 8
  %11 = call %struct.mpc_parser_t* @mpc_char(i8 signext 46)
  %12 = call %struct.mpc_parser_t* @mpc_digits()
  %13 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcf_strfold, %struct.mpc_parser_t* %11, %struct.mpc_parser_t* %12, void (i8*)* @free)
  %14 = call %struct.mpc_parser_t* @mpc_maybe_lift(%struct.mpc_parser_t* %13, i8* ()* @mpcf_ctor_str)
  store %struct.mpc_parser_t* %14, %struct.mpc_parser_t** %3, align 8
  %15 = call %struct.mpc_parser_t* @mpc_oneof(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.50, i32 0, i32 0))
  store %struct.mpc_parser_t* %15, %struct.mpc_parser_t** %4, align 8
  %16 = call %struct.mpc_parser_t* @mpc_oneof(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.49, i32 0, i32 0))
  %17 = call %struct.mpc_parser_t* @mpc_maybe_lift(%struct.mpc_parser_t* %16, i8* ()* @mpcf_ctor_str)
  store %struct.mpc_parser_t* %17, %struct.mpc_parser_t** %5, align 8
  %18 = call %struct.mpc_parser_t* @mpc_digits()
  store %struct.mpc_parser_t* %18, %struct.mpc_parser_t** %6, align 8
  %19 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %20 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %21 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %6, align 8
  %22 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 3, i8* (i32, i8**)* @mpcf_strfold, %struct.mpc_parser_t* %19, %struct.mpc_parser_t* %20, %struct.mpc_parser_t* %21, void (i8*)* @free, void (i8*)* @free)
  %23 = call %struct.mpc_parser_t* @mpc_maybe_lift(%struct.mpc_parser_t* %22, i8* ()* @mpcf_ctor_str)
  store %struct.mpc_parser_t* %23, %struct.mpc_parser_t** %7, align 8
  %24 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1, align 8
  %25 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %26 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %27 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %28 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 4, i8* (i32, i8**)* @mpcf_strfold, %struct.mpc_parser_t* %24, %struct.mpc_parser_t* %25, %struct.mpc_parser_t* %26, %struct.mpc_parser_t* %27, void (i8*)* @free, void (i8*)* @free, void (i8*)* @free)
  %29 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %28, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.51, i32 0, i32 0))
  ret %struct.mpc_parser_t* %29
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_ctor_str() #0 {
  %1 = call noalias i8* @calloc(i64 1, i64 1) #5
  ret i8* %1
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_float() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_real()
  %2 = call %struct.mpc_parser_t* @mpc_apply(%struct.mpc_parser_t* %1, i8* (i8*)* @mpcf_float)
  %3 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %2, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.52, i32 0, i32 0))
  ret %struct.mpc_parser_t* %3
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_float(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca float*, align 8
  store i8* %0, i8** %2, align 8
  %4 = call noalias i8* @malloc(i64 4) #5
  %5 = bitcast i8* %4 to float*
  store float* %5, float** %3, align 8
  %6 = load i8*, i8** %2, align 8
  %7 = call double @strtod(i8* %6, i8** null) #5
  %8 = fptrunc double %7 to float
  %9 = load float*, float** %3, align 8
  store float %8, float* %9, align 4
  %10 = load i8*, i8** %2, align 8
  call void @free(i8* %10) #5
  %11 = load float*, float** %3, align 8
  %12 = bitcast float* %11 to i8*
  ret i8* %12
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_char_lit() #0 {
  %1 = call %struct.mpc_parser_t* @mpc_escape()
  %2 = call %struct.mpc_parser_t* @mpc_any()
  %3 = call %struct.mpc_parser_t* (i32, ...) @mpc_or(i32 2, %struct.mpc_parser_t* %1, %struct.mpc_parser_t* %2)
  %4 = call %struct.mpc_parser_t* @mpc_between(%struct.mpc_parser_t* %3, void (i8*)* @free, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.53, i32 0, i32 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.53, i32 0, i32 0))
  %5 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %4, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.54, i32 0, i32 0))
  ret %struct.mpc_parser_t* %5
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_between(%struct.mpc_parser_t*, void (i8*)*, i8*, i8*) #0 {
  %5 = alloca %struct.mpc_parser_t*, align 8
  %6 = alloca void (i8*)*, align 8
  %7 = alloca i8*, align 8
  %8 = alloca i8*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %5, align 8
  store void (i8*)* %1, void (i8*)** %6, align 8
  store i8* %2, i8** %7, align 8
  store i8* %3, i8** %8, align 8
  %9 = load i8*, i8** %7, align 8
  %10 = call %struct.mpc_parser_t* @mpc_string(i8* %9)
  %11 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %12 = load i8*, i8** %8, align 8
  %13 = call %struct.mpc_parser_t* @mpc_string(i8* %12)
  %14 = load void (i8*)*, void (i8*)** %6, align 8
  %15 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 3, i8* (i32, i8**)* @mpcf_snd_free, %struct.mpc_parser_t* %10, %struct.mpc_parser_t* %11, %struct.mpc_parser_t* %13, void (i8*)* @free, void (i8*)* %14)
  ret %struct.mpc_parser_t* %15
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_string_lit() #0 {
  %1 = alloca %struct.mpc_parser_t*, align 8
  %2 = call %struct.mpc_parser_t* @mpc_escape()
  %3 = call %struct.mpc_parser_t* @mpc_noneof(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.55, i32 0, i32 0))
  %4 = call %struct.mpc_parser_t* (i32, ...) @mpc_or(i32 2, %struct.mpc_parser_t* %2, %struct.mpc_parser_t* %3)
  store %struct.mpc_parser_t* %4, %struct.mpc_parser_t** %1, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1, align 8
  %6 = call %struct.mpc_parser_t* @mpc_many(i8* (i32, i8**)* @mpcf_strfold, %struct.mpc_parser_t* %5)
  %7 = call %struct.mpc_parser_t* @mpc_between(%struct.mpc_parser_t* %6, void (i8*)* @free, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.55, i32 0, i32 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.55, i32 0, i32 0))
  %8 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %7, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.56, i32 0, i32 0))
  ret %struct.mpc_parser_t* %8
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_regex_lit() #0 {
  %1 = alloca %struct.mpc_parser_t*, align 8
  %2 = call %struct.mpc_parser_t* @mpc_escape()
  %3 = call %struct.mpc_parser_t* @mpc_noneof(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.57, i32 0, i32 0))
  %4 = call %struct.mpc_parser_t* (i32, ...) @mpc_or(i32 2, %struct.mpc_parser_t* %2, %struct.mpc_parser_t* %3)
  store %struct.mpc_parser_t* %4, %struct.mpc_parser_t** %1, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1, align 8
  %6 = call %struct.mpc_parser_t* @mpc_many(i8* (i32, i8**)* @mpcf_strfold, %struct.mpc_parser_t* %5)
  %7 = call %struct.mpc_parser_t* @mpc_between(%struct.mpc_parser_t* %6, void (i8*)* @free, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.57, i32 0, i32 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.57, i32 0, i32 0))
  %8 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %7, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.58, i32 0, i32 0))
  ret %struct.mpc_parser_t* %8
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_ident() #0 {
  %1 = alloca %struct.mpc_parser_t*, align 8
  %2 = alloca %struct.mpc_parser_t*, align 8
  %3 = call %struct.mpc_parser_t* @mpc_alpha()
  %4 = call %struct.mpc_parser_t* @mpc_underscore()
  %5 = call %struct.mpc_parser_t* (i32, ...) @mpc_or(i32 2, %struct.mpc_parser_t* %3, %struct.mpc_parser_t* %4)
  store %struct.mpc_parser_t* %5, %struct.mpc_parser_t** %1, align 8
  %6 = call %struct.mpc_parser_t* @mpc_alphanum()
  %7 = call %struct.mpc_parser_t* @mpc_many(i8* (i32, i8**)* @mpcf_strfold, %struct.mpc_parser_t* %6)
  store %struct.mpc_parser_t* %7, %struct.mpc_parser_t** %2, align 8
  %8 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1, align 8
  %9 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %10 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcf_strfold, %struct.mpc_parser_t* %8, %struct.mpc_parser_t* %9, void (i8*)* @free)
  ret %struct.mpc_parser_t* %10
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_startwith(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  %3 = call %struct.mpc_parser_t* @mpc_soi()
  %4 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %5 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcf_snd, %struct.mpc_parser_t* %3, %struct.mpc_parser_t* %4, void (i8*)* @mpcf_dtor_null)
  ret %struct.mpc_parser_t* %5
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_snd(i32, i8**) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i8**, align 8
  store i32 %0, i32* %3, align 4
  store i8** %1, i8*** %4, align 8
  %5 = load i32, i32* %3, align 4
  %6 = load i8**, i8*** %4, align 8
  %7 = getelementptr inbounds i8*, i8** %6, i64 1
  %8 = load i8*, i8** %7, align 8
  ret i8* %8
}

; Function Attrs: noinline nounwind optnone uwtable
define void @mpcf_dtor_null(i8*) #0 {
  %2 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %3 = load i8*, i8** %2, align 8
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_endwith(%struct.mpc_parser_t*, void (i8*)*) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca void (i8*)*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store void (i8*)* %1, void (i8*)** %4, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = call %struct.mpc_parser_t* @mpc_eoi()
  %7 = load void (i8*)*, void (i8*)** %4, align 8
  %8 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcf_fst, %struct.mpc_parser_t* %5, %struct.mpc_parser_t* %6, void (i8*)* %7)
  ret %struct.mpc_parser_t* %8
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_fst(i32, i8**) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i8**, align 8
  store i32 %0, i32* %3, align 4
  store i8** %1, i8*** %4, align 8
  %5 = load i32, i32* %3, align 4
  %6 = load i8**, i8*** %4, align 8
  %7 = getelementptr inbounds i8*, i8** %6, i64 0
  %8 = load i8*, i8** %7, align 8
  ret i8* %8
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_whole(%struct.mpc_parser_t*, void (i8*)*) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca void (i8*)*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store void (i8*)* %1, void (i8*)** %4, align 8
  %5 = call %struct.mpc_parser_t* @mpc_soi()
  %6 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %7 = call %struct.mpc_parser_t* @mpc_eoi()
  %8 = load void (i8*)*, void (i8*)** %4, align 8
  %9 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 3, i8* (i32, i8**)* @mpcf_snd, %struct.mpc_parser_t* %5, %struct.mpc_parser_t* %6, %struct.mpc_parser_t* %7, void (i8*)* @mpcf_dtor_null, void (i8*)* %8)
  ret %struct.mpc_parser_t* %9
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_stripl(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  %3 = call %struct.mpc_parser_t* @mpc_blank()
  %4 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %5 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcf_snd, %struct.mpc_parser_t* %3, %struct.mpc_parser_t* %4, void (i8*)* @mpcf_dtor_null)
  ret %struct.mpc_parser_t* %5
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_stripr(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  %3 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %4 = call %struct.mpc_parser_t* @mpc_blank()
  %5 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcf_fst, %struct.mpc_parser_t* %3, %struct.mpc_parser_t* %4, void (i8*)* @mpcf_dtor_null)
  ret %struct.mpc_parser_t* %5
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_strip(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  %3 = call %struct.mpc_parser_t* @mpc_blank()
  %4 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %5 = call %struct.mpc_parser_t* @mpc_blank()
  %6 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 3, i8* (i32, i8**)* @mpcf_snd, %struct.mpc_parser_t* %3, %struct.mpc_parser_t* %4, %struct.mpc_parser_t* %5, void (i8*)* @mpcf_dtor_null, void (i8*)* @mpcf_dtor_null)
  ret %struct.mpc_parser_t* %6
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_tok(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  %3 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %4 = call %struct.mpc_parser_t* @mpc_blank()
  %5 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcf_fst, %struct.mpc_parser_t* %3, %struct.mpc_parser_t* %4, void (i8*)* @mpcf_dtor_null)
  ret %struct.mpc_parser_t* %5
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_sym(i8*) #0 {
  %2 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %3 = load i8*, i8** %2, align 8
  %4 = call %struct.mpc_parser_t* @mpc_string(i8* %3)
  %5 = call %struct.mpc_parser_t* @mpc_tok(%struct.mpc_parser_t* %4)
  ret %struct.mpc_parser_t* %5
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_total(%struct.mpc_parser_t*, void (i8*)*) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca void (i8*)*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store void (i8*)* %1, void (i8*)** %4, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = call %struct.mpc_parser_t* @mpc_strip(%struct.mpc_parser_t* %5)
  %7 = load void (i8*)*, void (i8*)** %4, align 8
  %8 = call %struct.mpc_parser_t* @mpc_whole(%struct.mpc_parser_t* %6, void (i8*)* %7)
  ret %struct.mpc_parser_t* %8
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_snd_free(i32, i8**) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i8**, align 8
  store i32 %0, i32* %3, align 4
  store i8** %1, i8*** %4, align 8
  %5 = load i32, i32* %3, align 4
  %6 = load i8**, i8*** %4, align 8
  %7 = call i8* @mpcf_nth_free(i32 %5, i8** %6, i32 1)
  ret i8* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_parens(%struct.mpc_parser_t*, void (i8*)*) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca void (i8*)*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store void (i8*)* %1, void (i8*)** %4, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = load void (i8*)*, void (i8*)** %4, align 8
  %7 = call %struct.mpc_parser_t* @mpc_between(%struct.mpc_parser_t* %5, void (i8*)* %6, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.59, i32 0, i32 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.60, i32 0, i32 0))
  ret %struct.mpc_parser_t* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_braces(%struct.mpc_parser_t*, void (i8*)*) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca void (i8*)*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store void (i8*)* %1, void (i8*)** %4, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = load void (i8*)*, void (i8*)** %4, align 8
  %7 = call %struct.mpc_parser_t* @mpc_between(%struct.mpc_parser_t* %5, void (i8*)* %6, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.61, i32 0, i32 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.62, i32 0, i32 0))
  ret %struct.mpc_parser_t* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_brackets(%struct.mpc_parser_t*, void (i8*)*) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca void (i8*)*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store void (i8*)* %1, void (i8*)** %4, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = load void (i8*)*, void (i8*)** %4, align 8
  %7 = call %struct.mpc_parser_t* @mpc_between(%struct.mpc_parser_t* %5, void (i8*)* %6, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.63, i32 0, i32 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.64, i32 0, i32 0))
  ret %struct.mpc_parser_t* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_squares(%struct.mpc_parser_t*, void (i8*)*) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca void (i8*)*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store void (i8*)* %1, void (i8*)** %4, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = load void (i8*)*, void (i8*)** %4, align 8
  %7 = call %struct.mpc_parser_t* @mpc_between(%struct.mpc_parser_t* %5, void (i8*)* %6, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.65, i32 0, i32 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.66, i32 0, i32 0))
  ret %struct.mpc_parser_t* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_tok_between(%struct.mpc_parser_t*, void (i8*)*, i8*, i8*) #0 {
  %5 = alloca %struct.mpc_parser_t*, align 8
  %6 = alloca void (i8*)*, align 8
  %7 = alloca i8*, align 8
  %8 = alloca i8*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %5, align 8
  store void (i8*)* %1, void (i8*)** %6, align 8
  store i8* %2, i8** %7, align 8
  store i8* %3, i8** %8, align 8
  %9 = load i8*, i8** %7, align 8
  %10 = call %struct.mpc_parser_t* @mpc_sym(i8* %9)
  %11 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %12 = call %struct.mpc_parser_t* @mpc_tok(%struct.mpc_parser_t* %11)
  %13 = load i8*, i8** %8, align 8
  %14 = call %struct.mpc_parser_t* @mpc_sym(i8* %13)
  %15 = load void (i8*)*, void (i8*)** %6, align 8
  %16 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 3, i8* (i32, i8**)* @mpcf_snd_free, %struct.mpc_parser_t* %10, %struct.mpc_parser_t* %12, %struct.mpc_parser_t* %14, void (i8*)* @free, void (i8*)* %15)
  ret %struct.mpc_parser_t* %16
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_tok_parens(%struct.mpc_parser_t*, void (i8*)*) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca void (i8*)*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store void (i8*)* %1, void (i8*)** %4, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = load void (i8*)*, void (i8*)** %4, align 8
  %7 = call %struct.mpc_parser_t* @mpc_tok_between(%struct.mpc_parser_t* %5, void (i8*)* %6, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.59, i32 0, i32 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.60, i32 0, i32 0))
  ret %struct.mpc_parser_t* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_tok_braces(%struct.mpc_parser_t*, void (i8*)*) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca void (i8*)*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store void (i8*)* %1, void (i8*)** %4, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = load void (i8*)*, void (i8*)** %4, align 8
  %7 = call %struct.mpc_parser_t* @mpc_tok_between(%struct.mpc_parser_t* %5, void (i8*)* %6, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.61, i32 0, i32 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.62, i32 0, i32 0))
  ret %struct.mpc_parser_t* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_tok_brackets(%struct.mpc_parser_t*, void (i8*)*) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca void (i8*)*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store void (i8*)* %1, void (i8*)** %4, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = load void (i8*)*, void (i8*)** %4, align 8
  %7 = call %struct.mpc_parser_t* @mpc_tok_between(%struct.mpc_parser_t* %5, void (i8*)* %6, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.63, i32 0, i32 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.64, i32 0, i32 0))
  ret %struct.mpc_parser_t* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_tok_squares(%struct.mpc_parser_t*, void (i8*)*) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca void (i8*)*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store void (i8*)* %1, void (i8*)** %4, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = load void (i8*)*, void (i8*)** %4, align 8
  %7 = call %struct.mpc_parser_t* @mpc_tok_between(%struct.mpc_parser_t* %5, void (i8*)* %6, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.65, i32 0, i32 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.66, i32 0, i32 0))
  ret %struct.mpc_parser_t* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpc_re(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca i8*, align 8
  %4 = alloca %struct.mpc_parser_t*, align 8
  %5 = alloca %union.mpc_result_t, align 8
  %6 = alloca %struct.mpc_parser_t*, align 8
  %7 = alloca %struct.mpc_parser_t*, align 8
  %8 = alloca %struct.mpc_parser_t*, align 8
  %9 = alloca %struct.mpc_parser_t*, align 8
  %10 = alloca %struct.mpc_parser_t*, align 8
  %11 = alloca %struct.mpc_parser_t*, align 8
  store i8* %0, i8** %2, align 8
  %12 = call %struct.mpc_parser_t* @mpc_new(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.58, i32 0, i32 0))
  store %struct.mpc_parser_t* %12, %struct.mpc_parser_t** %6, align 8
  %13 = call %struct.mpc_parser_t* @mpc_new(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.67, i32 0, i32 0))
  store %struct.mpc_parser_t* %13, %struct.mpc_parser_t** %7, align 8
  %14 = call %struct.mpc_parser_t* @mpc_new(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.68, i32 0, i32 0))
  store %struct.mpc_parser_t* %14, %struct.mpc_parser_t** %8, align 8
  %15 = call %struct.mpc_parser_t* @mpc_new(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.69, i32 0, i32 0))
  store %struct.mpc_parser_t* %15, %struct.mpc_parser_t** %9, align 8
  %16 = call %struct.mpc_parser_t* @mpc_new(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.70, i32 0, i32 0))
  store %struct.mpc_parser_t* %16, %struct.mpc_parser_t** %10, align 8
  %17 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %6, align 8
  %18 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %19 = call %struct.mpc_parser_t* @mpc_char(i8 signext 124)
  %20 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %6, align 8
  %21 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcf_snd_free, %struct.mpc_parser_t* %19, %struct.mpc_parser_t* %20, void (i8*)* @free)
  %22 = call %struct.mpc_parser_t* @mpc_maybe(%struct.mpc_parser_t* %21)
  %23 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcf_re_or, %struct.mpc_parser_t* %18, %struct.mpc_parser_t* %22, void (i8*)* bitcast (void (%struct.mpc_parser_t*)* @mpc_delete to void (i8*)*))
  %24 = call %struct.mpc_parser_t* @mpc_define(%struct.mpc_parser_t* %17, %struct.mpc_parser_t* %23)
  %25 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %26 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %27 = call %struct.mpc_parser_t* @mpc_many(i8* (i32, i8**)* @mpcf_re_and, %struct.mpc_parser_t* %26)
  %28 = call %struct.mpc_parser_t* @mpc_define(%struct.mpc_parser_t* %25, %struct.mpc_parser_t* %27)
  %29 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %30 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  %31 = call %struct.mpc_parser_t* @mpc_char(i8 signext 42)
  %32 = call %struct.mpc_parser_t* @mpc_char(i8 signext 43)
  %33 = call %struct.mpc_parser_t* @mpc_char(i8 signext 63)
  %34 = call %struct.mpc_parser_t* @mpc_int()
  %35 = call %struct.mpc_parser_t* @mpc_brackets(%struct.mpc_parser_t* %34, void (i8*)* @free)
  %36 = call %struct.mpc_parser_t* @mpc_pass()
  %37 = call %struct.mpc_parser_t* (i32, ...) @mpc_or(i32 5, %struct.mpc_parser_t* %31, %struct.mpc_parser_t* %32, %struct.mpc_parser_t* %33, %struct.mpc_parser_t* %35, %struct.mpc_parser_t* %36)
  %38 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcf_re_repeat, %struct.mpc_parser_t* %30, %struct.mpc_parser_t* %37, void (i8*)* bitcast (void (%struct.mpc_parser_t*)* @mpc_delete to void (i8*)*))
  %39 = call %struct.mpc_parser_t* @mpc_define(%struct.mpc_parser_t* %29, %struct.mpc_parser_t* %38)
  %40 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  %41 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %6, align 8
  %42 = call %struct.mpc_parser_t* @mpc_parens(%struct.mpc_parser_t* %41, void (i8*)* bitcast (void (%struct.mpc_parser_t*)* @mpc_delete to void (i8*)*))
  %43 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %10, align 8
  %44 = call %struct.mpc_parser_t* @mpc_squares(%struct.mpc_parser_t* %43, void (i8*)* bitcast (void (%struct.mpc_parser_t*)* @mpc_delete to void (i8*)*))
  %45 = call %struct.mpc_parser_t* @mpc_escape()
  %46 = call %struct.mpc_parser_t* @mpc_apply(%struct.mpc_parser_t* %45, i8* (i8*)* @mpcf_re_escape)
  %47 = call %struct.mpc_parser_t* @mpc_noneof(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.71, i32 0, i32 0))
  %48 = call %struct.mpc_parser_t* @mpc_apply(%struct.mpc_parser_t* %47, i8* (i8*)* @mpcf_re_escape)
  %49 = call %struct.mpc_parser_t* (i32, ...) @mpc_or(i32 4, %struct.mpc_parser_t* %42, %struct.mpc_parser_t* %44, %struct.mpc_parser_t* %46, %struct.mpc_parser_t* %48)
  %50 = call %struct.mpc_parser_t* @mpc_define(%struct.mpc_parser_t* %40, %struct.mpc_parser_t* %49)
  %51 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %10, align 8
  %52 = call %struct.mpc_parser_t* @mpc_escape()
  %53 = call %struct.mpc_parser_t* @mpc_noneof(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.66, i32 0, i32 0))
  %54 = call %struct.mpc_parser_t* (i32, ...) @mpc_or(i32 2, %struct.mpc_parser_t* %52, %struct.mpc_parser_t* %53)
  %55 = call %struct.mpc_parser_t* @mpc_many(i8* (i32, i8**)* @mpcf_strfold, %struct.mpc_parser_t* %54)
  %56 = call %struct.mpc_parser_t* @mpc_apply(%struct.mpc_parser_t* %55, i8* (i8*)* @mpcf_re_range)
  %57 = call %struct.mpc_parser_t* @mpc_define(%struct.mpc_parser_t* %51, %struct.mpc_parser_t* %56)
  %58 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %6, align 8
  %59 = call %struct.mpc_parser_t* @mpc_predictive(%struct.mpc_parser_t* %58)
  %60 = call %struct.mpc_parser_t* @mpc_whole(%struct.mpc_parser_t* %59, void (i8*)* bitcast (void (%struct.mpc_parser_t*)* @mpc_delete to void (i8*)*))
  store %struct.mpc_parser_t* %60, %struct.mpc_parser_t** %11, align 8
  %61 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %11, align 8
  call void @mpc_optimise(%struct.mpc_parser_t* %61)
  %62 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %6, align 8
  call void @mpc_optimise(%struct.mpc_parser_t* %62)
  %63 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  call void @mpc_optimise(%struct.mpc_parser_t* %63)
  %64 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  call void @mpc_optimise(%struct.mpc_parser_t* %64)
  %65 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  call void @mpc_optimise(%struct.mpc_parser_t* %65)
  %66 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %10, align 8
  call void @mpc_optimise(%struct.mpc_parser_t* %66)
  %67 = load i8*, i8** %2, align 8
  %68 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %11, align 8
  %69 = call i32 @mpc_parse(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.72, i32 0, i32 0), i8* %67, %struct.mpc_parser_t* %68, %union.mpc_result_t* %5)
  %70 = icmp ne i32 %69, 0
  br i1 %70, label %83, label %71

; <label>:71:                                     ; preds = %1
  %72 = bitcast %union.mpc_result_t* %5 to %struct.mpc_err_t**
  %73 = load %struct.mpc_err_t*, %struct.mpc_err_t** %72, align 8
  %74 = call i8* @mpc_err_string(%struct.mpc_err_t* %73)
  store i8* %74, i8** %3, align 8
  %75 = load i8*, i8** %3, align 8
  %76 = call %struct.mpc_parser_t* (i8*, ...) @mpc_failf(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @.str.73, i32 0, i32 0), i8* %75)
  store %struct.mpc_parser_t* %76, %struct.mpc_parser_t** %4, align 8
  %77 = bitcast %union.mpc_result_t* %5 to %struct.mpc_err_t**
  %78 = load %struct.mpc_err_t*, %struct.mpc_err_t** %77, align 8
  call void @mpc_err_delete(%struct.mpc_err_t* %78)
  %79 = load i8*, i8** %3, align 8
  call void @free(i8* %79) #5
  %80 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %81 = bitcast %struct.mpc_parser_t* %80 to i8*
  %82 = bitcast %union.mpc_result_t* %5 to i8**
  store i8* %81, i8** %82, align 8
  br label %83

; <label>:83:                                     ; preds = %71, %1
  %84 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %11, align 8
  %85 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %6, align 8
  %86 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %87 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %88 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  %89 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %10, align 8
  call void (i32, ...) @mpc_cleanup(i32 6, %struct.mpc_parser_t* %84, %struct.mpc_parser_t* %85, %struct.mpc_parser_t* %86, %struct.mpc_parser_t* %87, %struct.mpc_parser_t* %88, %struct.mpc_parser_t* %89)
  %90 = bitcast %union.mpc_result_t* %5 to i8**
  %91 = load i8*, i8** %90, align 8
  %92 = bitcast i8* %91 to %struct.mpc_parser_t*
  call void @mpc_optimise(%struct.mpc_parser_t* %92)
  %93 = bitcast %union.mpc_result_t* %5 to i8**
  %94 = load i8*, i8** %93, align 8
  %95 = bitcast i8* %94 to %struct.mpc_parser_t*
  ret %struct.mpc_parser_t* %95
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcf_re_or(i32, i8**) #0 {
  %3 = alloca i8*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i8**, align 8
  store i32 %0, i32* %4, align 4
  store i8** %1, i8*** %5, align 8
  %6 = load i32, i32* %4, align 4
  %7 = load i8**, i8*** %5, align 8
  %8 = getelementptr inbounds i8*, i8** %7, i64 1
  %9 = load i8*, i8** %8, align 8
  %10 = icmp ne i8* %9, null
  br i1 %10, label %15, label %11

; <label>:11:                                     ; preds = %2
  %12 = load i8**, i8*** %5, align 8
  %13 = getelementptr inbounds i8*, i8** %12, i64 0
  %14 = load i8*, i8** %13, align 8
  store i8* %14, i8** %3, align 8
  br label %24

; <label>:15:                                     ; preds = %2
  %16 = load i8**, i8*** %5, align 8
  %17 = getelementptr inbounds i8*, i8** %16, i64 0
  %18 = load i8*, i8** %17, align 8
  %19 = load i8**, i8*** %5, align 8
  %20 = getelementptr inbounds i8*, i8** %19, i64 1
  %21 = load i8*, i8** %20, align 8
  %22 = call %struct.mpc_parser_t* (i32, ...) @mpc_or(i32 2, i8* %18, i8* %21)
  %23 = bitcast %struct.mpc_parser_t* %22 to i8*
  store i8* %23, i8** %3, align 8
  br label %24

; <label>:24:                                     ; preds = %15, %11
  %25 = load i8*, i8** %3, align 8
  ret i8* %25
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcf_re_and(i32, i8**) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i8**, align 8
  %5 = alloca i32, align 4
  %6 = alloca %struct.mpc_parser_t*, align 8
  store i32 %0, i32* %3, align 4
  store i8** %1, i8*** %4, align 8
  %7 = call %struct.mpc_parser_t* @mpc_lift(i8* ()* @mpcf_ctor_str)
  store %struct.mpc_parser_t* %7, %struct.mpc_parser_t** %6, align 8
  store i32 0, i32* %5, align 4
  br label %8

; <label>:8:                                      ; preds = %20, %2
  %9 = load i32, i32* %5, align 4
  %10 = load i32, i32* %3, align 4
  %11 = icmp slt i32 %9, %10
  br i1 %11, label %12, label %23

; <label>:12:                                     ; preds = %8
  %13 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %6, align 8
  %14 = load i8**, i8*** %4, align 8
  %15 = load i32, i32* %5, align 4
  %16 = sext i32 %15 to i64
  %17 = getelementptr inbounds i8*, i8** %14, i64 %16
  %18 = load i8*, i8** %17, align 8
  %19 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcf_strfold, %struct.mpc_parser_t* %13, i8* %18, void (i8*)* @free)
  store %struct.mpc_parser_t* %19, %struct.mpc_parser_t** %6, align 8
  br label %20

; <label>:20:                                     ; preds = %12
  %21 = load i32, i32* %5, align 4
  %22 = add nsw i32 %21, 1
  store i32 %22, i32* %5, align 4
  br label %8

; <label>:23:                                     ; preds = %8
  %24 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %6, align 8
  %25 = bitcast %struct.mpc_parser_t* %24 to i8*
  ret i8* %25
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcf_re_repeat(i32, i8**) #0 {
  %3 = alloca i8*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i8**, align 8
  %6 = alloca i32, align 4
  store i32 %0, i32* %4, align 4
  store i8** %1, i8*** %5, align 8
  %7 = load i32, i32* %4, align 4
  %8 = load i8**, i8*** %5, align 8
  %9 = getelementptr inbounds i8*, i8** %8, i64 1
  %10 = load i8*, i8** %9, align 8
  %11 = icmp ne i8* %10, null
  br i1 %11, label %16, label %12

; <label>:12:                                     ; preds = %2
  %13 = load i8**, i8*** %5, align 8
  %14 = getelementptr inbounds i8*, i8** %13, i64 0
  %15 = load i8*, i8** %14, align 8
  store i8* %15, i8** %3, align 8
  br label %80

; <label>:16:                                     ; preds = %2
  %17 = load i8**, i8*** %5, align 8
  %18 = getelementptr inbounds i8*, i8** %17, i64 1
  %19 = load i8*, i8** %18, align 8
  %20 = call i32 @strcmp(i8* %19, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.74, i32 0, i32 0)) #7
  %21 = icmp eq i32 %20, 0
  br i1 %21, label %22, label %32

; <label>:22:                                     ; preds = %16
  %23 = load i8**, i8*** %5, align 8
  %24 = getelementptr inbounds i8*, i8** %23, i64 1
  %25 = load i8*, i8** %24, align 8
  call void @free(i8* %25) #5
  %26 = load i8**, i8*** %5, align 8
  %27 = getelementptr inbounds i8*, i8** %26, i64 0
  %28 = load i8*, i8** %27, align 8
  %29 = bitcast i8* %28 to %struct.mpc_parser_t*
  %30 = call %struct.mpc_parser_t* @mpc_many(i8* (i32, i8**)* @mpcf_strfold, %struct.mpc_parser_t* %29)
  %31 = bitcast %struct.mpc_parser_t* %30 to i8*
  store i8* %31, i8** %3, align 8
  br label %80

; <label>:32:                                     ; preds = %16
  %33 = load i8**, i8*** %5, align 8
  %34 = getelementptr inbounds i8*, i8** %33, i64 1
  %35 = load i8*, i8** %34, align 8
  %36 = call i32 @strcmp(i8* %35, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.76, i32 0, i32 0)) #7
  %37 = icmp eq i32 %36, 0
  br i1 %37, label %38, label %48

; <label>:38:                                     ; preds = %32
  %39 = load i8**, i8*** %5, align 8
  %40 = getelementptr inbounds i8*, i8** %39, i64 1
  %41 = load i8*, i8** %40, align 8
  call void @free(i8* %41) #5
  %42 = load i8**, i8*** %5, align 8
  %43 = getelementptr inbounds i8*, i8** %42, i64 0
  %44 = load i8*, i8** %43, align 8
  %45 = bitcast i8* %44 to %struct.mpc_parser_t*
  %46 = call %struct.mpc_parser_t* @mpc_many1(i8* (i32, i8**)* @mpcf_strfold, %struct.mpc_parser_t* %45)
  %47 = bitcast %struct.mpc_parser_t* %46 to i8*
  store i8* %47, i8** %3, align 8
  br label %80

; <label>:48:                                     ; preds = %32
  %49 = load i8**, i8*** %5, align 8
  %50 = getelementptr inbounds i8*, i8** %49, i64 1
  %51 = load i8*, i8** %50, align 8
  %52 = call i32 @strcmp(i8* %51, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.85, i32 0, i32 0)) #7
  %53 = icmp eq i32 %52, 0
  br i1 %53, label %54, label %64

; <label>:54:                                     ; preds = %48
  %55 = load i8**, i8*** %5, align 8
  %56 = getelementptr inbounds i8*, i8** %55, i64 1
  %57 = load i8*, i8** %56, align 8
  call void @free(i8* %57) #5
  %58 = load i8**, i8*** %5, align 8
  %59 = getelementptr inbounds i8*, i8** %58, i64 0
  %60 = load i8*, i8** %59, align 8
  %61 = bitcast i8* %60 to %struct.mpc_parser_t*
  %62 = call %struct.mpc_parser_t* @mpc_maybe_lift(%struct.mpc_parser_t* %61, i8* ()* @mpcf_ctor_str)
  %63 = bitcast %struct.mpc_parser_t* %62 to i8*
  store i8* %63, i8** %3, align 8
  br label %80

; <label>:64:                                     ; preds = %48
  %65 = load i8**, i8*** %5, align 8
  %66 = getelementptr inbounds i8*, i8** %65, i64 1
  %67 = load i8*, i8** %66, align 8
  %68 = bitcast i8* %67 to i32*
  %69 = load i32, i32* %68, align 4
  store i32 %69, i32* %6, align 4
  %70 = load i8**, i8*** %5, align 8
  %71 = getelementptr inbounds i8*, i8** %70, i64 1
  %72 = load i8*, i8** %71, align 8
  call void @free(i8* %72) #5
  %73 = load i32, i32* %6, align 4
  %74 = load i8**, i8*** %5, align 8
  %75 = getelementptr inbounds i8*, i8** %74, i64 0
  %76 = load i8*, i8** %75, align 8
  %77 = bitcast i8* %76 to %struct.mpc_parser_t*
  %78 = call %struct.mpc_parser_t* @mpc_count(i32 %73, i8* (i32, i8**)* @mpcf_strfold, %struct.mpc_parser_t* %77, void (i8*)* @free)
  %79 = bitcast %struct.mpc_parser_t* %78 to i8*
  store i8* %79, i8** %3, align 8
  br label %80

; <label>:80:                                     ; preds = %64, %54, %38, %22, %12
  %81 = load i8*, i8** %3, align 8
  ret i8* %81
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcf_re_escape(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca i8*, align 8
  %4 = alloca i8*, align 8
  %5 = alloca %struct.mpc_parser_t*, align 8
  store i8* %0, i8** %3, align 8
  %6 = load i8*, i8** %3, align 8
  store i8* %6, i8** %4, align 8
  %7 = load i8*, i8** %4, align 8
  %8 = getelementptr inbounds i8, i8* %7, i64 0
  %9 = load i8, i8* %8, align 1
  %10 = sext i8 %9 to i32
  %11 = icmp eq i32 %10, 46
  br i1 %11, label %12, label %16

; <label>:12:                                     ; preds = %1
  %13 = load i8*, i8** %4, align 8
  call void @free(i8* %13) #5
  %14 = call %struct.mpc_parser_t* @mpc_any()
  %15 = bitcast %struct.mpc_parser_t* %14 to i8*
  store i8* %15, i8** %2, align 8
  br label %73

; <label>:16:                                     ; preds = %1
  %17 = load i8*, i8** %4, align 8
  %18 = getelementptr inbounds i8, i8* %17, i64 0
  %19 = load i8, i8* %18, align 1
  %20 = sext i8 %19 to i32
  %21 = icmp eq i32 %20, 94
  br i1 %21, label %22, label %28

; <label>:22:                                     ; preds = %16
  %23 = load i8*, i8** %4, align 8
  call void @free(i8* %23) #5
  %24 = call %struct.mpc_parser_t* @mpc_soi()
  %25 = call %struct.mpc_parser_t* @mpc_lift(i8* ()* @mpcf_ctor_str)
  %26 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcf_snd, %struct.mpc_parser_t* %24, %struct.mpc_parser_t* %25, void (i8*)* @free)
  %27 = bitcast %struct.mpc_parser_t* %26 to i8*
  store i8* %27, i8** %2, align 8
  br label %73

; <label>:28:                                     ; preds = %16
  %29 = load i8*, i8** %4, align 8
  %30 = getelementptr inbounds i8, i8* %29, i64 0
  %31 = load i8, i8* %30, align 1
  %32 = sext i8 %31 to i32
  %33 = icmp eq i32 %32, 36
  br i1 %33, label %34, label %40

; <label>:34:                                     ; preds = %28
  %35 = load i8*, i8** %4, align 8
  call void @free(i8* %35) #5
  %36 = call %struct.mpc_parser_t* @mpc_eoi()
  %37 = call %struct.mpc_parser_t* @mpc_lift(i8* ()* @mpcf_ctor_str)
  %38 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcf_snd, %struct.mpc_parser_t* %36, %struct.mpc_parser_t* %37, void (i8*)* @free)
  %39 = bitcast %struct.mpc_parser_t* %38 to i8*
  store i8* %39, i8** %2, align 8
  br label %73

; <label>:40:                                     ; preds = %28
  %41 = load i8*, i8** %4, align 8
  %42 = getelementptr inbounds i8, i8* %41, i64 0
  %43 = load i8, i8* %42, align 1
  %44 = sext i8 %43 to i32
  %45 = icmp eq i32 %44, 92
  br i1 %45, label %46, label %65

; <label>:46:                                     ; preds = %40
  %47 = load i8*, i8** %4, align 8
  %48 = getelementptr inbounds i8, i8* %47, i64 1
  %49 = load i8, i8* %48, align 1
  %50 = call %struct.mpc_parser_t* @mpc_re_escape_char(i8 signext %49)
  store %struct.mpc_parser_t* %50, %struct.mpc_parser_t** %5, align 8
  %51 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %52 = icmp ne %struct.mpc_parser_t* %51, null
  br i1 %52, label %58, label %53

; <label>:53:                                     ; preds = %46
  %54 = load i8*, i8** %4, align 8
  %55 = getelementptr inbounds i8, i8* %54, i64 1
  %56 = load i8, i8* %55, align 1
  %57 = call %struct.mpc_parser_t* @mpc_char(i8 signext %56)
  br label %60

; <label>:58:                                     ; preds = %46
  %59 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  br label %60

; <label>:60:                                     ; preds = %58, %53
  %61 = phi %struct.mpc_parser_t* [ %57, %53 ], [ %59, %58 ]
  store %struct.mpc_parser_t* %61, %struct.mpc_parser_t** %5, align 8
  %62 = load i8*, i8** %4, align 8
  call void @free(i8* %62) #5
  %63 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %64 = bitcast %struct.mpc_parser_t* %63 to i8*
  store i8* %64, i8** %2, align 8
  br label %73

; <label>:65:                                     ; preds = %40
  %66 = load i8*, i8** %4, align 8
  %67 = getelementptr inbounds i8, i8* %66, i64 0
  %68 = load i8, i8* %67, align 1
  %69 = call %struct.mpc_parser_t* @mpc_char(i8 signext %68)
  store %struct.mpc_parser_t* %69, %struct.mpc_parser_t** %5, align 8
  %70 = load i8*, i8** %4, align 8
  call void @free(i8* %70) #5
  %71 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %72 = bitcast %struct.mpc_parser_t* %71 to i8*
  store i8* %72, i8** %2, align 8
  br label %73

; <label>:73:                                     ; preds = %65, %60, %34, %22, %12
  %74 = load i8*, i8** %2, align 8
  ret i8* %74
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcf_re_range(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca i8*, align 8
  %4 = alloca %struct.mpc_parser_t*, align 8
  %5 = alloca i64, align 8
  %6 = alloca i64, align 8
  %7 = alloca i64, align 8
  %8 = alloca i64, align 8
  %9 = alloca i8*, align 8
  %10 = alloca i8*, align 8
  %11 = alloca i32, align 4
  %12 = alloca i8*, align 8
  store i8* %0, i8** %3, align 8
  store i8* null, i8** %9, align 8
  %13 = load i8*, i8** %3, align 8
  store i8* %13, i8** %10, align 8
  %14 = load i8*, i8** %10, align 8
  %15 = getelementptr inbounds i8, i8* %14, i64 0
  %16 = load i8, i8* %15, align 1
  %17 = sext i8 %16 to i32
  %18 = icmp eq i32 %17, 94
  %19 = zext i1 %18 to i64
  %20 = select i1 %18, i32 1, i32 0
  store i32 %20, i32* %11, align 4
  %21 = load i8*, i8** %10, align 8
  %22 = getelementptr inbounds i8, i8* %21, i64 0
  %23 = load i8, i8* %22, align 1
  %24 = sext i8 %23 to i32
  %25 = icmp eq i32 %24, 0
  br i1 %25, label %26, label %30

; <label>:26:                                     ; preds = %1
  %27 = load i8*, i8** %3, align 8
  call void @free(i8* %27) #5
  %28 = call %struct.mpc_parser_t* @mpc_fail(i8* getelementptr inbounds ([31 x i8], [31 x i8]* @.str.109, i32 0, i32 0))
  %29 = bitcast %struct.mpc_parser_t* %28 to i8*
  store i8* %29, i8** %2, align 8
  br label %223

; <label>:30:                                     ; preds = %1
  %31 = load i8*, i8** %10, align 8
  %32 = getelementptr inbounds i8, i8* %31, i64 0
  %33 = load i8, i8* %32, align 1
  %34 = sext i8 %33 to i32
  %35 = icmp eq i32 %34, 94
  br i1 %35, label %36, label %46

; <label>:36:                                     ; preds = %30
  %37 = load i8*, i8** %10, align 8
  %38 = getelementptr inbounds i8, i8* %37, i64 1
  %39 = load i8, i8* %38, align 1
  %40 = sext i8 %39 to i32
  %41 = icmp eq i32 %40, 0
  br i1 %41, label %42, label %46

; <label>:42:                                     ; preds = %36
  %43 = load i8*, i8** %3, align 8
  call void @free(i8* %43) #5
  %44 = call %struct.mpc_parser_t* @mpc_fail(i8* getelementptr inbounds ([31 x i8], [31 x i8]* @.str.109, i32 0, i32 0))
  %45 = bitcast %struct.mpc_parser_t* %44 to i8*
  store i8* %45, i8** %2, align 8
  br label %223

; <label>:46:                                     ; preds = %36, %30
  %47 = call noalias i8* @calloc(i64 1, i64 1) #5
  store i8* %47, i8** %12, align 8
  %48 = load i32, i32* %11, align 4
  %49 = sext i32 %48 to i64
  store i64 %49, i64* %5, align 8
  br label %50

; <label>:50:                                     ; preds = %205, %46
  %51 = load i64, i64* %5, align 8
  %52 = load i8*, i8** %10, align 8
  %53 = call i64 @strlen(i8* %52) #7
  %54 = icmp ult i64 %51, %53
  br i1 %54, label %55, label %208

; <label>:55:                                     ; preds = %50
  %56 = load i8*, i8** %10, align 8
  %57 = load i64, i64* %5, align 8
  %58 = getelementptr inbounds i8, i8* %56, i64 %57
  %59 = load i8, i8* %58, align 1
  %60 = sext i8 %59 to i32
  %61 = icmp eq i32 %60, 92
  br i1 %61, label %62, label %108

; <label>:62:                                     ; preds = %55
  %63 = load i8*, i8** %10, align 8
  %64 = load i64, i64* %5, align 8
  %65 = add i64 %64, 1
  %66 = getelementptr inbounds i8, i8* %63, i64 %65
  %67 = load i8, i8* %66, align 1
  %68 = call i8* @mpc_re_range_escape_char(i8 signext %67)
  store i8* %68, i8** %9, align 8
  %69 = load i8*, i8** %9, align 8
  %70 = icmp ne i8* %69, null
  br i1 %70, label %71, label %83

; <label>:71:                                     ; preds = %62
  %72 = load i8*, i8** %12, align 8
  %73 = load i8*, i8** %12, align 8
  %74 = call i64 @strlen(i8* %73) #7
  %75 = load i8*, i8** %9, align 8
  %76 = call i64 @strlen(i8* %75) #7
  %77 = add i64 %74, %76
  %78 = add i64 %77, 1
  %79 = call i8* @realloc(i8* %72, i64 %78) #5
  store i8* %79, i8** %12, align 8
  %80 = load i8*, i8** %12, align 8
  %81 = load i8*, i8** %9, align 8
  %82 = call i8* @strcat(i8* %80, i8* %81) #5
  br label %105

; <label>:83:                                     ; preds = %62
  %84 = load i8*, i8** %12, align 8
  %85 = load i8*, i8** %12, align 8
  %86 = call i64 @strlen(i8* %85) #7
  %87 = add i64 %86, 1
  %88 = add i64 %87, 1
  %89 = call i8* @realloc(i8* %84, i64 %88) #5
  store i8* %89, i8** %12, align 8
  %90 = load i8*, i8** %12, align 8
  %91 = load i8*, i8** %12, align 8
  %92 = call i64 @strlen(i8* %91) #7
  %93 = add i64 %92, 1
  %94 = getelementptr inbounds i8, i8* %90, i64 %93
  store i8 0, i8* %94, align 1
  %95 = load i8*, i8** %10, align 8
  %96 = load i64, i64* %5, align 8
  %97 = add i64 %96, 1
  %98 = getelementptr inbounds i8, i8* %95, i64 %97
  %99 = load i8, i8* %98, align 1
  %100 = load i8*, i8** %12, align 8
  %101 = load i8*, i8** %12, align 8
  %102 = call i64 @strlen(i8* %101) #7
  %103 = add i64 %102, 0
  %104 = getelementptr inbounds i8, i8* %100, i64 %103
  store i8 %99, i8* %104, align 1
  br label %105

; <label>:105:                                    ; preds = %83, %71
  %106 = load i64, i64* %5, align 8
  %107 = add i64 %106, 1
  store i64 %107, i64* %5, align 8
  br label %204

; <label>:108:                                    ; preds = %55
  %109 = load i8*, i8** %10, align 8
  %110 = load i64, i64* %5, align 8
  %111 = getelementptr inbounds i8, i8* %109, i64 %110
  %112 = load i8, i8* %111, align 1
  %113 = sext i8 %112 to i32
  %114 = icmp eq i32 %113, 45
  br i1 %114, label %115, label %182

; <label>:115:                                    ; preds = %108
  %116 = load i8*, i8** %10, align 8
  %117 = load i64, i64* %5, align 8
  %118 = add i64 %117, 1
  %119 = getelementptr inbounds i8, i8* %116, i64 %118
  %120 = load i8, i8* %119, align 1
  %121 = sext i8 %120 to i32
  %122 = icmp eq i32 %121, 0
  br i1 %122, label %126, label %123

; <label>:123:                                    ; preds = %115
  %124 = load i64, i64* %5, align 8
  %125 = icmp eq i64 %124, 0
  br i1 %125, label %126, label %135

; <label>:126:                                    ; preds = %123, %115
  %127 = load i8*, i8** %12, align 8
  %128 = load i8*, i8** %12, align 8
  %129 = call i64 @strlen(i8* %128) #7
  %130 = add i64 %129, 1
  %131 = add i64 %130, 1
  %132 = call i8* @realloc(i8* %127, i64 %131) #5
  store i8* %132, i8** %12, align 8
  %133 = load i8*, i8** %12, align 8
  %134 = call i8* @strcat(i8* %133, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.77, i32 0, i32 0)) #5
  br label %181

; <label>:135:                                    ; preds = %123
  %136 = load i8*, i8** %10, align 8
  %137 = load i64, i64* %5, align 8
  %138 = sub i64 %137, 1
  %139 = getelementptr inbounds i8, i8* %136, i64 %138
  %140 = load i8, i8* %139, align 1
  %141 = sext i8 %140 to i32
  %142 = add nsw i32 %141, 1
  %143 = sext i32 %142 to i64
  store i64 %143, i64* %7, align 8
  %144 = load i8*, i8** %10, align 8
  %145 = load i64, i64* %5, align 8
  %146 = add i64 %145, 1
  %147 = getelementptr inbounds i8, i8* %144, i64 %146
  %148 = load i8, i8* %147, align 1
  %149 = sext i8 %148 to i32
  %150 = sub nsw i32 %149, 1
  %151 = sext i32 %150 to i64
  store i64 %151, i64* %8, align 8
  %152 = load i64, i64* %7, align 8
  store i64 %152, i64* %6, align 8
  br label %153

; <label>:153:                                    ; preds = %177, %135
  %154 = load i64, i64* %6, align 8
  %155 = load i64, i64* %8, align 8
  %156 = icmp ule i64 %154, %155
  br i1 %156, label %157, label %180

; <label>:157:                                    ; preds = %153
  %158 = load i8*, i8** %12, align 8
  %159 = load i8*, i8** %12, align 8
  %160 = call i64 @strlen(i8* %159) #7
  %161 = add i64 %160, 1
  %162 = add i64 %161, 1
  %163 = add i64 %162, 1
  %164 = call i8* @realloc(i8* %158, i64 %163) #5
  store i8* %164, i8** %12, align 8
  %165 = load i8*, i8** %12, align 8
  %166 = load i8*, i8** %12, align 8
  %167 = call i64 @strlen(i8* %166) #7
  %168 = add i64 %167, 1
  %169 = getelementptr inbounds i8, i8* %165, i64 %168
  store i8 0, i8* %169, align 1
  %170 = load i64, i64* %6, align 8
  %171 = trunc i64 %170 to i8
  %172 = load i8*, i8** %12, align 8
  %173 = load i8*, i8** %12, align 8
  %174 = call i64 @strlen(i8* %173) #7
  %175 = add i64 %174, 0
  %176 = getelementptr inbounds i8, i8* %172, i64 %175
  store i8 %171, i8* %176, align 1
  br label %177

; <label>:177:                                    ; preds = %157
  %178 = load i64, i64* %6, align 8
  %179 = add i64 %178, 1
  store i64 %179, i64* %6, align 8
  br label %153

; <label>:180:                                    ; preds = %153
  br label %181

; <label>:181:                                    ; preds = %180, %126
  br label %203

; <label>:182:                                    ; preds = %108
  %183 = load i8*, i8** %12, align 8
  %184 = load i8*, i8** %12, align 8
  %185 = call i64 @strlen(i8* %184) #7
  %186 = add i64 %185, 1
  %187 = add i64 %186, 1
  %188 = call i8* @realloc(i8* %183, i64 %187) #5
  store i8* %188, i8** %12, align 8
  %189 = load i8*, i8** %12, align 8
  %190 = load i8*, i8** %12, align 8
  %191 = call i64 @strlen(i8* %190) #7
  %192 = add i64 %191, 1
  %193 = getelementptr inbounds i8, i8* %189, i64 %192
  store i8 0, i8* %193, align 1
  %194 = load i8*, i8** %10, align 8
  %195 = load i64, i64* %5, align 8
  %196 = getelementptr inbounds i8, i8* %194, i64 %195
  %197 = load i8, i8* %196, align 1
  %198 = load i8*, i8** %12, align 8
  %199 = load i8*, i8** %12, align 8
  %200 = call i64 @strlen(i8* %199) #7
  %201 = add i64 %200, 0
  %202 = getelementptr inbounds i8, i8* %198, i64 %201
  store i8 %197, i8* %202, align 1
  br label %203

; <label>:203:                                    ; preds = %182, %181
  br label %204

; <label>:204:                                    ; preds = %203, %105
  br label %205

; <label>:205:                                    ; preds = %204
  %206 = load i64, i64* %5, align 8
  %207 = add i64 %206, 1
  store i64 %207, i64* %5, align 8
  br label %50

; <label>:208:                                    ; preds = %50
  %209 = load i32, i32* %11, align 4
  %210 = icmp eq i32 %209, 1
  br i1 %210, label %211, label %214

; <label>:211:                                    ; preds = %208
  %212 = load i8*, i8** %12, align 8
  %213 = call %struct.mpc_parser_t* @mpc_noneof(i8* %212)
  br label %217

; <label>:214:                                    ; preds = %208
  %215 = load i8*, i8** %12, align 8
  %216 = call %struct.mpc_parser_t* @mpc_oneof(i8* %215)
  br label %217

; <label>:217:                                    ; preds = %214, %211
  %218 = phi %struct.mpc_parser_t* [ %213, %211 ], [ %216, %214 ]
  store %struct.mpc_parser_t* %218, %struct.mpc_parser_t** %4, align 8
  %219 = load i8*, i8** %3, align 8
  call void @free(i8* %219) #5
  %220 = load i8*, i8** %12, align 8
  call void @free(i8* %220) #5
  %221 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %222 = bitcast %struct.mpc_parser_t* %221 to i8*
  store i8* %222, i8** %2, align 8
  br label %223

; <label>:223:                                    ; preds = %217, %42, %26
  %224 = load i8*, i8** %2, align 8
  ret i8* %224
}

; Function Attrs: noinline nounwind optnone uwtable
define void @mpc_optimise(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  %3 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  call void @mpc_optimise_unretained(%struct.mpc_parser_t* %3, i32 1)
  ret void
}

; Function Attrs: nounwind
declare i64 @strtol(i8*, i8**, i32) #1

; Function Attrs: nounwind
declare double @strtod(i8*, i8**) #1

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_strtriml(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %4 = load i8*, i8** %2, align 8
  store i8* %4, i8** %3, align 8
  br label %5

; <label>:5:                                      ; preds = %17, %1
  %6 = call i16** @__ctype_b_loc() #8
  %7 = load i16*, i16** %6, align 8
  %8 = load i8*, i8** %3, align 8
  %9 = load i8, i8* %8, align 1
  %10 = sext i8 %9 to i32
  %11 = sext i32 %10 to i64
  %12 = getelementptr inbounds i16, i16* %7, i64 %11
  %13 = load i16, i16* %12, align 2
  %14 = zext i16 %13 to i32
  %15 = and i32 %14, 8192
  %16 = icmp ne i32 %15, 0
  br i1 %16, label %17, label %23

; <label>:17:                                     ; preds = %5
  %18 = load i8*, i8** %3, align 8
  %19 = load i8*, i8** %3, align 8
  %20 = getelementptr inbounds i8, i8* %19, i64 1
  %21 = load i8*, i8** %3, align 8
  %22 = call i64 @strlen(i8* %21) #7
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %18, i8* %20, i64 %22, i32 1, i1 false)
  br label %5

; <label>:23:                                     ; preds = %5
  %24 = load i8*, i8** %3, align 8
  ret i8* %24
}

; Function Attrs: nounwind readnone
declare i16** @__ctype_b_loc() #6

; Function Attrs: argmemonly nounwind
declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #4

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_strtrimr(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca i8*, align 8
  %4 = alloca i64, align 8
  store i8* %0, i8** %2, align 8
  %5 = load i8*, i8** %2, align 8
  store i8* %5, i8** %3, align 8
  %6 = load i8*, i8** %3, align 8
  %7 = call i64 @strlen(i8* %6) #7
  store i64 %7, i64* %4, align 8
  br label %8

; <label>:8:                                      ; preds = %23, %1
  %9 = call i16** @__ctype_b_loc() #8
  %10 = load i16*, i16** %9, align 8
  %11 = load i8*, i8** %3, align 8
  %12 = load i64, i64* %4, align 8
  %13 = sub i64 %12, 1
  %14 = getelementptr inbounds i8, i8* %11, i64 %13
  %15 = load i8, i8* %14, align 1
  %16 = sext i8 %15 to i32
  %17 = sext i32 %16 to i64
  %18 = getelementptr inbounds i16, i16* %10, i64 %17
  %19 = load i16, i16* %18, align 2
  %20 = zext i16 %19 to i32
  %21 = and i32 %20, 8192
  %22 = icmp ne i32 %21, 0
  br i1 %22, label %23, label %30

; <label>:23:                                     ; preds = %8
  %24 = load i8*, i8** %3, align 8
  %25 = load i64, i64* %4, align 8
  %26 = sub i64 %25, 1
  %27 = getelementptr inbounds i8, i8* %24, i64 %26
  store i8 0, i8* %27, align 1
  %28 = load i64, i64* %4, align 8
  %29 = add i64 %28, -1
  store i64 %29, i64* %4, align 8
  br label %8

; <label>:30:                                     ; preds = %8
  %31 = load i8*, i8** %3, align 8
  ret i8* %31
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_strtrim(i8*) #0 {
  %2 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %3 = load i8*, i8** %2, align 8
  %4 = call i8* @mpcf_strtrimr(i8* %3)
  %5 = call i8* @mpcf_strtriml(i8* %4)
  ret i8* %5
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_escape(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %4 = load i8*, i8** %2, align 8
  %5 = call i8* @mpcf_escape_new(i8* %4, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @mpc_escape_input_c, i32 0, i32 0), i8** getelementptr inbounds ([12 x i8*], [12 x i8*]* @mpc_escape_output_c, i32 0, i32 0))
  store i8* %5, i8** %3, align 8
  %6 = load i8*, i8** %2, align 8
  call void @free(i8* %6) #5
  %7 = load i8*, i8** %3, align 8
  ret i8* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcf_escape_new(i8*, i8*, i8**) #0 {
  %4 = alloca i8*, align 8
  %5 = alloca i8*, align 8
  %6 = alloca i8**, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca [2 x i8], align 1
  %10 = alloca i8*, align 8
  %11 = alloca i8*, align 8
  store i8* %0, i8** %4, align 8
  store i8* %1, i8** %5, align 8
  store i8** %2, i8*** %6, align 8
  %12 = load i8*, i8** %4, align 8
  store i8* %12, i8** %10, align 8
  %13 = call noalias i8* @calloc(i64 1, i64 1) #5
  store i8* %13, i8** %11, align 8
  br label %14

; <label>:14:                                     ; preds = %76, %3
  %15 = load i8*, i8** %10, align 8
  %16 = load i8, i8* %15, align 1
  %17 = icmp ne i8 %16, 0
  br i1 %17, label %18, label %79

; <label>:18:                                     ; preds = %14
  store i32 0, i32* %7, align 4
  store i32 0, i32* %8, align 4
  br label %19

; <label>:19:                                     ; preds = %57, %18
  %20 = load i8**, i8*** %6, align 8
  %21 = load i32, i32* %7, align 4
  %22 = sext i32 %21 to i64
  %23 = getelementptr inbounds i8*, i8** %20, i64 %22
  %24 = load i8*, i8** %23, align 8
  %25 = icmp ne i8* %24, null
  br i1 %25, label %26, label %60

; <label>:26:                                     ; preds = %19
  %27 = load i8*, i8** %10, align 8
  %28 = load i8, i8* %27, align 1
  %29 = sext i8 %28 to i32
  %30 = load i8*, i8** %5, align 8
  %31 = load i32, i32* %7, align 4
  %32 = sext i32 %31 to i64
  %33 = getelementptr inbounds i8, i8* %30, i64 %32
  %34 = load i8, i8* %33, align 1
  %35 = sext i8 %34 to i32
  %36 = icmp eq i32 %29, %35
  br i1 %36, label %37, label %57

; <label>:37:                                     ; preds = %26
  %38 = load i8*, i8** %11, align 8
  %39 = load i8*, i8** %11, align 8
  %40 = call i64 @strlen(i8* %39) #7
  %41 = load i8**, i8*** %6, align 8
  %42 = load i32, i32* %7, align 4
  %43 = sext i32 %42 to i64
  %44 = getelementptr inbounds i8*, i8** %41, i64 %43
  %45 = load i8*, i8** %44, align 8
  %46 = call i64 @strlen(i8* %45) #7
  %47 = add i64 %40, %46
  %48 = add i64 %47, 1
  %49 = call i8* @realloc(i8* %38, i64 %48) #5
  store i8* %49, i8** %11, align 8
  %50 = load i8*, i8** %11, align 8
  %51 = load i8**, i8*** %6, align 8
  %52 = load i32, i32* %7, align 4
  %53 = sext i32 %52 to i64
  %54 = getelementptr inbounds i8*, i8** %51, i64 %53
  %55 = load i8*, i8** %54, align 8
  %56 = call i8* @strcat(i8* %50, i8* %55) #5
  store i32 1, i32* %8, align 4
  br label %60

; <label>:57:                                     ; preds = %26
  %58 = load i32, i32* %7, align 4
  %59 = add nsw i32 %58, 1
  store i32 %59, i32* %7, align 4
  br label %19

; <label>:60:                                     ; preds = %37, %19
  %61 = load i32, i32* %8, align 4
  %62 = icmp ne i32 %61, 0
  br i1 %62, label %76, label %63

; <label>:63:                                     ; preds = %60
  %64 = load i8*, i8** %11, align 8
  %65 = load i8*, i8** %11, align 8
  %66 = call i64 @strlen(i8* %65) #7
  %67 = add i64 %66, 2
  %68 = call i8* @realloc(i8* %64, i64 %67) #5
  store i8* %68, i8** %11, align 8
  %69 = load i8*, i8** %10, align 8
  %70 = load i8, i8* %69, align 1
  %71 = getelementptr inbounds [2 x i8], [2 x i8]* %9, i64 0, i64 0
  store i8 %70, i8* %71, align 1
  %72 = getelementptr inbounds [2 x i8], [2 x i8]* %9, i64 0, i64 1
  store i8 0, i8* %72, align 1
  %73 = load i8*, i8** %11, align 8
  %74 = getelementptr inbounds [2 x i8], [2 x i8]* %9, i32 0, i32 0
  %75 = call i8* @strcat(i8* %73, i8* %74) #5
  br label %76

; <label>:76:                                     ; preds = %63, %60
  %77 = load i8*, i8** %10, align 8
  %78 = getelementptr inbounds i8, i8* %77, i32 1
  store i8* %78, i8** %10, align 8
  br label %14

; <label>:79:                                     ; preds = %14
  %80 = load i8*, i8** %11, align 8
  ret i8* %80
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_unescape(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %4 = load i8*, i8** %2, align 8
  %5 = call i8* @mpcf_unescape_new(i8* %4, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @mpc_escape_input_c, i32 0, i32 0), i8** getelementptr inbounds ([12 x i8*], [12 x i8*]* @mpc_escape_output_c, i32 0, i32 0))
  store i8* %5, i8** %3, align 8
  %6 = load i8*, i8** %2, align 8
  call void @free(i8* %6) #5
  %7 = load i8*, i8** %3, align 8
  ret i8* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcf_unescape_new(i8*, i8*, i8**) #0 {
  %4 = alloca i8*, align 8
  %5 = alloca i8*, align 8
  %6 = alloca i8**, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca [2 x i8], align 1
  %10 = alloca i8*, align 8
  %11 = alloca i8*, align 8
  store i8* %0, i8** %4, align 8
  store i8* %1, i8** %5, align 8
  store i8** %2, i8*** %6, align 8
  store i32 0, i32* %8, align 4
  %12 = load i8*, i8** %4, align 8
  store i8* %12, i8** %10, align 8
  %13 = call noalias i8* @calloc(i64 1, i64 1) #5
  store i8* %13, i8** %11, align 8
  br label %14

; <label>:14:                                     ; preds = %102, %3
  %15 = load i8*, i8** %10, align 8
  %16 = load i8, i8* %15, align 1
  %17 = icmp ne i8 %16, 0
  br i1 %17, label %18, label %103

; <label>:18:                                     ; preds = %14
  store i32 0, i32* %7, align 4
  store i32 0, i32* %8, align 4
  br label %19

; <label>:19:                                     ; preds = %73, %18
  %20 = load i8**, i8*** %6, align 8
  %21 = load i32, i32* %7, align 4
  %22 = sext i32 %21 to i64
  %23 = getelementptr inbounds i8*, i8** %20, i64 %22
  %24 = load i8*, i8** %23, align 8
  %25 = icmp ne i8* %24, null
  br i1 %25, label %26, label %76

; <label>:26:                                     ; preds = %19
  %27 = load i8*, i8** %10, align 8
  %28 = getelementptr inbounds i8, i8* %27, i64 0
  %29 = load i8, i8* %28, align 1
  %30 = sext i8 %29 to i32
  %31 = load i8**, i8*** %6, align 8
  %32 = load i32, i32* %7, align 4
  %33 = sext i32 %32 to i64
  %34 = getelementptr inbounds i8*, i8** %31, i64 %33
  %35 = load i8*, i8** %34, align 8
  %36 = getelementptr inbounds i8, i8* %35, i64 0
  %37 = load i8, i8* %36, align 1
  %38 = sext i8 %37 to i32
  %39 = icmp eq i32 %30, %38
  br i1 %39, label %40, label %73

; <label>:40:                                     ; preds = %26
  %41 = load i8*, i8** %10, align 8
  %42 = getelementptr inbounds i8, i8* %41, i64 1
  %43 = load i8, i8* %42, align 1
  %44 = sext i8 %43 to i32
  %45 = load i8**, i8*** %6, align 8
  %46 = load i32, i32* %7, align 4
  %47 = sext i32 %46 to i64
  %48 = getelementptr inbounds i8*, i8** %45, i64 %47
  %49 = load i8*, i8** %48, align 8
  %50 = getelementptr inbounds i8, i8* %49, i64 1
  %51 = load i8, i8* %50, align 1
  %52 = sext i8 %51 to i32
  %53 = icmp eq i32 %44, %52
  br i1 %53, label %54, label %73

; <label>:54:                                     ; preds = %40
  %55 = load i8*, i8** %11, align 8
  %56 = load i8*, i8** %11, align 8
  %57 = call i64 @strlen(i8* %56) #7
  %58 = add i64 %57, 1
  %59 = add i64 %58, 1
  %60 = call i8* @realloc(i8* %55, i64 %59) #5
  store i8* %60, i8** %11, align 8
  %61 = load i8*, i8** %5, align 8
  %62 = load i32, i32* %7, align 4
  %63 = sext i32 %62 to i64
  %64 = getelementptr inbounds i8, i8* %61, i64 %63
  %65 = load i8, i8* %64, align 1
  %66 = getelementptr inbounds [2 x i8], [2 x i8]* %9, i64 0, i64 0
  store i8 %65, i8* %66, align 1
  %67 = getelementptr inbounds [2 x i8], [2 x i8]* %9, i64 0, i64 1
  store i8 0, i8* %67, align 1
  %68 = load i8*, i8** %11, align 8
  %69 = getelementptr inbounds [2 x i8], [2 x i8]* %9, i32 0, i32 0
  %70 = call i8* @strcat(i8* %68, i8* %69) #5
  store i32 1, i32* %8, align 4
  %71 = load i8*, i8** %10, align 8
  %72 = getelementptr inbounds i8, i8* %71, i32 1
  store i8* %72, i8** %10, align 8
  br label %76

; <label>:73:                                     ; preds = %40, %26
  %74 = load i32, i32* %7, align 4
  %75 = add nsw i32 %74, 1
  store i32 %75, i32* %7, align 4
  br label %19

; <label>:76:                                     ; preds = %54, %19
  %77 = load i32, i32* %8, align 4
  %78 = icmp ne i32 %77, 0
  br i1 %78, label %93, label %79

; <label>:79:                                     ; preds = %76
  %80 = load i8*, i8** %11, align 8
  %81 = load i8*, i8** %11, align 8
  %82 = call i64 @strlen(i8* %81) #7
  %83 = add i64 %82, 1
  %84 = add i64 %83, 1
  %85 = call i8* @realloc(i8* %80, i64 %84) #5
  store i8* %85, i8** %11, align 8
  %86 = load i8*, i8** %10, align 8
  %87 = load i8, i8* %86, align 1
  %88 = getelementptr inbounds [2 x i8], [2 x i8]* %9, i64 0, i64 0
  store i8 %87, i8* %88, align 1
  %89 = getelementptr inbounds [2 x i8], [2 x i8]* %9, i64 0, i64 1
  store i8 0, i8* %89, align 1
  %90 = load i8*, i8** %11, align 8
  %91 = getelementptr inbounds [2 x i8], [2 x i8]* %9, i32 0, i32 0
  %92 = call i8* @strcat(i8* %90, i8* %91) #5
  br label %93

; <label>:93:                                     ; preds = %79, %76
  %94 = load i8*, i8** %10, align 8
  %95 = load i8, i8* %94, align 1
  %96 = sext i8 %95 to i32
  %97 = icmp eq i32 %96, 0
  br i1 %97, label %98, label %99

; <label>:98:                                     ; preds = %93
  br label %103

; <label>:99:                                     ; preds = %93
  %100 = load i8*, i8** %10, align 8
  %101 = getelementptr inbounds i8, i8* %100, i32 1
  store i8* %101, i8** %10, align 8
  br label %102

; <label>:102:                                    ; preds = %99
  br label %14

; <label>:103:                                    ; preds = %98, %14
  %104 = load i8*, i8** %11, align 8
  ret i8* %104
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_escape_regex(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %4 = load i8*, i8** %2, align 8
  %5 = call i8* @mpcf_escape_new(i8* %4, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @mpc_escape_input_raw_re, i32 0, i32 0), i8** getelementptr inbounds ([2 x i8*], [2 x i8*]* @mpc_escape_output_raw_re, i32 0, i32 0))
  store i8* %5, i8** %3, align 8
  %6 = load i8*, i8** %2, align 8
  call void @free(i8* %6) #5
  %7 = load i8*, i8** %3, align 8
  ret i8* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_unescape_regex(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %4 = load i8*, i8** %2, align 8
  %5 = call i8* @mpcf_unescape_new(i8* %4, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @mpc_escape_input_raw_re, i32 0, i32 0), i8** getelementptr inbounds ([2 x i8*], [2 x i8*]* @mpc_escape_output_raw_re, i32 0, i32 0))
  store i8* %5, i8** %3, align 8
  %6 = load i8*, i8** %2, align 8
  call void @free(i8* %6) #5
  %7 = load i8*, i8** %3, align 8
  ret i8* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_escape_string_raw(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %4 = load i8*, i8** %2, align 8
  %5 = call i8* @mpcf_escape_new(i8* %4, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @mpc_escape_input_raw_cstr, i32 0, i32 0), i8** getelementptr inbounds ([2 x i8*], [2 x i8*]* @mpc_escape_output_raw_cstr, i32 0, i32 0))
  store i8* %5, i8** %3, align 8
  %6 = load i8*, i8** %2, align 8
  call void @free(i8* %6) #5
  %7 = load i8*, i8** %3, align 8
  ret i8* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_unescape_string_raw(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %4 = load i8*, i8** %2, align 8
  %5 = call i8* @mpcf_unescape_new(i8* %4, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @mpc_escape_input_raw_cstr, i32 0, i32 0), i8** getelementptr inbounds ([2 x i8*], [2 x i8*]* @mpc_escape_output_raw_cstr, i32 0, i32 0))
  store i8* %5, i8** %3, align 8
  %6 = load i8*, i8** %2, align 8
  call void @free(i8* %6) #5
  %7 = load i8*, i8** %3, align 8
  ret i8* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_escape_char_raw(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %4 = load i8*, i8** %2, align 8
  %5 = call i8* @mpcf_escape_new(i8* %4, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @mpc_escape_input_raw_cchar, i32 0, i32 0), i8** getelementptr inbounds ([2 x i8*], [2 x i8*]* @mpc_escape_output_raw_cchar, i32 0, i32 0))
  store i8* %5, i8** %3, align 8
  %6 = load i8*, i8** %2, align 8
  call void @free(i8* %6) #5
  %7 = load i8*, i8** %3, align 8
  ret i8* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_unescape_char_raw(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %4 = load i8*, i8** %2, align 8
  %5 = call i8* @mpcf_unescape_new(i8* %4, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @mpc_escape_input_raw_cchar, i32 0, i32 0), i8** getelementptr inbounds ([2 x i8*], [2 x i8*]* @mpc_escape_output_raw_cchar, i32 0, i32 0))
  store i8* %5, i8** %3, align 8
  %6 = load i8*, i8** %2, align 8
  call void @free(i8* %6) #5
  %7 = load i8*, i8** %3, align 8
  ret i8* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_null(i32, i8**) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i8**, align 8
  store i32 %0, i32* %3, align 4
  store i8** %1, i8*** %4, align 8
  %5 = load i32, i32* %3, align 4
  %6 = load i8**, i8*** %4, align 8
  ret i8* null
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_trd(i32, i8**) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i8**, align 8
  store i32 %0, i32* %3, align 4
  store i8** %1, i8*** %4, align 8
  %5 = load i32, i32* %3, align 4
  %6 = load i8**, i8*** %4, align 8
  %7 = getelementptr inbounds i8*, i8** %6, i64 2
  %8 = load i8*, i8** %7, align 8
  ret i8* %8
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_fst_free(i32, i8**) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i8**, align 8
  store i32 %0, i32* %3, align 4
  store i8** %1, i8*** %4, align 8
  %5 = load i32, i32* %3, align 4
  %6 = load i8**, i8*** %4, align 8
  %7 = call i8* @mpcf_nth_free(i32 %5, i8** %6, i32 0)
  ret i8* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcf_nth_free(i32, i8**, i32) #0 {
  %4 = alloca i32, align 4
  %5 = alloca i8**, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  store i32 %0, i32* %4, align 4
  store i8** %1, i8*** %5, align 8
  store i32 %2, i32* %6, align 4
  store i32 0, i32* %7, align 4
  br label %8

; <label>:8:                                      ; preds = %23, %3
  %9 = load i32, i32* %7, align 4
  %10 = load i32, i32* %4, align 4
  %11 = icmp slt i32 %9, %10
  br i1 %11, label %12, label %26

; <label>:12:                                     ; preds = %8
  %13 = load i32, i32* %7, align 4
  %14 = load i32, i32* %6, align 4
  %15 = icmp ne i32 %13, %14
  br i1 %15, label %16, label %22

; <label>:16:                                     ; preds = %12
  %17 = load i8**, i8*** %5, align 8
  %18 = load i32, i32* %7, align 4
  %19 = sext i32 %18 to i64
  %20 = getelementptr inbounds i8*, i8** %17, i64 %19
  %21 = load i8*, i8** %20, align 8
  call void @free(i8* %21) #5
  br label %22

; <label>:22:                                     ; preds = %16, %12
  br label %23

; <label>:23:                                     ; preds = %22
  %24 = load i32, i32* %7, align 4
  %25 = add nsw i32 %24, 1
  store i32 %25, i32* %7, align 4
  br label %8

; <label>:26:                                     ; preds = %8
  %27 = load i8**, i8*** %5, align 8
  %28 = load i32, i32* %6, align 4
  %29 = sext i32 %28 to i64
  %30 = getelementptr inbounds i8*, i8** %27, i64 %29
  %31 = load i8*, i8** %30, align 8
  ret i8* %31
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_trd_free(i32, i8**) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i8**, align 8
  store i32 %0, i32* %3, align 4
  store i8** %1, i8*** %4, align 8
  %5 = load i32, i32* %3, align 4
  %6 = load i8**, i8*** %4, align 8
  %7 = call i8* @mpcf_nth_free(i32 %5, i8** %6, i32 2)
  ret i8* %7
}

; Function Attrs: nounwind
declare i8* @strcat(i8*, i8*) #1

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_maths(i32, i8**) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i8**, align 8
  %5 = alloca i32**, align 8
  store i32 %0, i32* %3, align 4
  store i8** %1, i8*** %4, align 8
  %6 = load i8**, i8*** %4, align 8
  %7 = bitcast i8** %6 to i32**
  store i32** %7, i32*** %5, align 8
  %8 = load i32, i32* %3, align 4
  %9 = load i8**, i8*** %4, align 8
  %10 = getelementptr inbounds i8*, i8** %9, i64 1
  %11 = load i8*, i8** %10, align 8
  %12 = call i32 @strcmp(i8* %11, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.74, i32 0, i32 0)) #7
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %14, label %24

; <label>:14:                                     ; preds = %2
  %15 = load i32**, i32*** %5, align 8
  %16 = getelementptr inbounds i32*, i32** %15, i64 2
  %17 = load i32*, i32** %16, align 8
  %18 = load i32, i32* %17, align 4
  %19 = load i32**, i32*** %5, align 8
  %20 = getelementptr inbounds i32*, i32** %19, i64 0
  %21 = load i32*, i32** %20, align 8
  %22 = load i32, i32* %21, align 4
  %23 = mul nsw i32 %22, %18
  store i32 %23, i32* %21, align 4
  br label %24

; <label>:24:                                     ; preds = %14, %2
  %25 = load i8**, i8*** %4, align 8
  %26 = getelementptr inbounds i8*, i8** %25, i64 1
  %27 = load i8*, i8** %26, align 8
  %28 = call i32 @strcmp(i8* %27, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.57, i32 0, i32 0)) #7
  %29 = icmp eq i32 %28, 0
  br i1 %29, label %30, label %40

; <label>:30:                                     ; preds = %24
  %31 = load i32**, i32*** %5, align 8
  %32 = getelementptr inbounds i32*, i32** %31, i64 2
  %33 = load i32*, i32** %32, align 8
  %34 = load i32, i32* %33, align 4
  %35 = load i32**, i32*** %5, align 8
  %36 = getelementptr inbounds i32*, i32** %35, i64 0
  %37 = load i32*, i32** %36, align 8
  %38 = load i32, i32* %37, align 4
  %39 = sdiv i32 %38, %34
  store i32 %39, i32* %37, align 4
  br label %40

; <label>:40:                                     ; preds = %30, %24
  %41 = load i8**, i8*** %4, align 8
  %42 = getelementptr inbounds i8*, i8** %41, i64 1
  %43 = load i8*, i8** %42, align 8
  %44 = call i32 @strcmp(i8* %43, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.75, i32 0, i32 0)) #7
  %45 = icmp eq i32 %44, 0
  br i1 %45, label %46, label %56

; <label>:46:                                     ; preds = %40
  %47 = load i32**, i32*** %5, align 8
  %48 = getelementptr inbounds i32*, i32** %47, i64 2
  %49 = load i32*, i32** %48, align 8
  %50 = load i32, i32* %49, align 4
  %51 = load i32**, i32*** %5, align 8
  %52 = getelementptr inbounds i32*, i32** %51, i64 0
  %53 = load i32*, i32** %52, align 8
  %54 = load i32, i32* %53, align 4
  %55 = srem i32 %54, %50
  store i32 %55, i32* %53, align 4
  br label %56

; <label>:56:                                     ; preds = %46, %40
  %57 = load i8**, i8*** %4, align 8
  %58 = getelementptr inbounds i8*, i8** %57, i64 1
  %59 = load i8*, i8** %58, align 8
  %60 = call i32 @strcmp(i8* %59, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.76, i32 0, i32 0)) #7
  %61 = icmp eq i32 %60, 0
  br i1 %61, label %62, label %72

; <label>:62:                                     ; preds = %56
  %63 = load i32**, i32*** %5, align 8
  %64 = getelementptr inbounds i32*, i32** %63, i64 2
  %65 = load i32*, i32** %64, align 8
  %66 = load i32, i32* %65, align 4
  %67 = load i32**, i32*** %5, align 8
  %68 = getelementptr inbounds i32*, i32** %67, i64 0
  %69 = load i32*, i32** %68, align 8
  %70 = load i32, i32* %69, align 4
  %71 = add nsw i32 %70, %66
  store i32 %71, i32* %69, align 4
  br label %72

; <label>:72:                                     ; preds = %62, %56
  %73 = load i8**, i8*** %4, align 8
  %74 = getelementptr inbounds i8*, i8** %73, i64 1
  %75 = load i8*, i8** %74, align 8
  %76 = call i32 @strcmp(i8* %75, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.77, i32 0, i32 0)) #7
  %77 = icmp eq i32 %76, 0
  br i1 %77, label %78, label %88

; <label>:78:                                     ; preds = %72
  %79 = load i32**, i32*** %5, align 8
  %80 = getelementptr inbounds i32*, i32** %79, i64 2
  %81 = load i32*, i32** %80, align 8
  %82 = load i32, i32* %81, align 4
  %83 = load i32**, i32*** %5, align 8
  %84 = getelementptr inbounds i32*, i32** %83, i64 0
  %85 = load i32*, i32** %84, align 8
  %86 = load i32, i32* %85, align 4
  %87 = sub nsw i32 %86, %82
  store i32 %87, i32* %85, align 4
  br label %88

; <label>:88:                                     ; preds = %78, %72
  %89 = load i8**, i8*** %4, align 8
  %90 = getelementptr inbounds i8*, i8** %89, i64 1
  %91 = load i8*, i8** %90, align 8
  call void @free(i8* %91) #5
  %92 = load i8**, i8*** %4, align 8
  %93 = getelementptr inbounds i8*, i8** %92, i64 2
  %94 = load i8*, i8** %93, align 8
  call void @free(i8* %94) #5
  %95 = load i8**, i8*** %4, align 8
  %96 = getelementptr inbounds i8*, i8** %95, i64 0
  %97 = load i8*, i8** %96, align 8
  ret i8* %97
}

; Function Attrs: nounwind readonly
declare i32 @strcmp(i8*, i8*) #3

; Function Attrs: noinline nounwind optnone uwtable
define void @mpc_print(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  %3 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  call void @mpc_print_unretained(%struct.mpc_parser_t* %3, i32 1)
  %4 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.7, i32 0, i32 0))
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_print_unretained(%struct.mpc_parser_t*, i32) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i8*, align 8
  %7 = alloca i8*, align 8
  %8 = alloca [2 x i8], align 1
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store i32 %1, i32* %4, align 4
  %9 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %10 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %9, i32 0, i32 0
  %11 = load i8, i8* %10, align 8
  %12 = sext i8 %11 to i32
  %13 = icmp ne i32 %12, 0
  br i1 %13, label %14, label %30

; <label>:14:                                     ; preds = %2
  %15 = load i32, i32* %4, align 4
  %16 = icmp ne i32 %15, 0
  br i1 %16, label %30, label %17

; <label>:17:                                     ; preds = %14
  %18 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %19 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %18, i32 0, i32 1
  %20 = load i8*, i8** %19, align 8
  %21 = icmp ne i8* %20, null
  br i1 %21, label %22, label %27

; <label>:22:                                     ; preds = %17
  %23 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %24 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %23, i32 0, i32 1
  %25 = load i8*, i8** %24, align 8
  %26 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.128, i32 0, i32 0), i8* %25)
  br label %29

; <label>:27:                                     ; preds = %17
  %28 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.129, i32 0, i32 0))
  br label %29

; <label>:29:                                     ; preds = %27, %22
  br label %404

; <label>:30:                                     ; preds = %14, %2
  %31 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %32 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %31, i32 0, i32 2
  %33 = load i8, i8* %32, align 8
  %34 = sext i8 %33 to i32
  %35 = icmp eq i32 %34, 0
  br i1 %35, label %36, label %38

; <label>:36:                                     ; preds = %30
  %37 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.130, i32 0, i32 0))
  br label %38

; <label>:38:                                     ; preds = %36, %30
  %39 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %40 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %39, i32 0, i32 2
  %41 = load i8, i8* %40, align 8
  %42 = sext i8 %41 to i32
  %43 = icmp eq i32 %42, 1
  br i1 %43, label %44, label %46

; <label>:44:                                     ; preds = %38
  %45 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.131, i32 0, i32 0))
  br label %46

; <label>:46:                                     ; preds = %44, %38
  %47 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %48 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %47, i32 0, i32 2
  %49 = load i8, i8* %48, align 8
  %50 = sext i8 %49 to i32
  %51 = icmp eq i32 %50, 2
  br i1 %51, label %52, label %54

; <label>:52:                                     ; preds = %46
  %53 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.132, i32 0, i32 0))
  br label %54

; <label>:54:                                     ; preds = %52, %46
  %55 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %56 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %55, i32 0, i32 2
  %57 = load i8, i8* %56, align 8
  %58 = sext i8 %57 to i32
  %59 = icmp eq i32 %58, 3
  br i1 %59, label %60, label %62

; <label>:60:                                     ; preds = %54
  %61 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.133, i32 0, i32 0))
  br label %62

; <label>:62:                                     ; preds = %60, %54
  %63 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %64 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %63, i32 0, i32 2
  %65 = load i8, i8* %64, align 8
  %66 = sext i8 %65 to i32
  %67 = icmp eq i32 %66, 7
  br i1 %67, label %68, label %70

; <label>:68:                                     ; preds = %62
  %69 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.134, i32 0, i32 0))
  br label %70

; <label>:70:                                     ; preds = %68, %62
  %71 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %72 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %71, i32 0, i32 2
  %73 = load i8, i8* %72, align 8
  %74 = sext i8 %73 to i32
  %75 = icmp eq i32 %74, 6
  br i1 %75, label %76, label %78

; <label>:76:                                     ; preds = %70
  %77 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.135, i32 0, i32 0))
  br label %78

; <label>:78:                                     ; preds = %76, %70
  %79 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %80 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %79, i32 0, i32 2
  %81 = load i8, i8* %80, align 8
  %82 = sext i8 %81 to i32
  %83 = icmp eq i32 %82, 5
  br i1 %83, label %84, label %91

; <label>:84:                                     ; preds = %78
  %85 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %86 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %85, i32 0, i32 3
  %87 = bitcast %union.mpc_pdata_t* %86 to %struct.mpc_pdata_expect_t*
  %88 = getelementptr inbounds %struct.mpc_pdata_expect_t, %struct.mpc_pdata_expect_t* %87, i32 0, i32 1
  %89 = load i8*, i8** %88, align 8
  %90 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i8* %89)
  br label %91

; <label>:91:                                     ; preds = %84, %78
  %92 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %93 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %92, i32 0, i32 2
  %94 = load i8, i8* %93, align 8
  %95 = sext i8 %94 to i32
  %96 = icmp eq i32 %95, 8
  br i1 %96, label %97, label %99

; <label>:97:                                     ; preds = %91
  %98 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.136, i32 0, i32 0))
  br label %99

; <label>:99:                                     ; preds = %97, %91
  %100 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %101 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %100, i32 0, i32 2
  %102 = load i8, i8* %101, align 8
  %103 = sext i8 %102 to i32
  %104 = icmp eq i32 %103, 13
  br i1 %104, label %105, label %107

; <label>:105:                                    ; preds = %99
  %106 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.137, i32 0, i32 0))
  br label %107

; <label>:107:                                    ; preds = %105, %99
  %108 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %109 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %108, i32 0, i32 2
  %110 = load i8, i8* %109, align 8
  %111 = sext i8 %110 to i32
  %112 = icmp eq i32 %111, 9
  br i1 %112, label %113, label %126

; <label>:113:                                    ; preds = %107
  %114 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %115 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %114, i32 0, i32 3
  %116 = bitcast %union.mpc_pdata_t* %115 to %struct.mpc_pdata_single_t*
  %117 = getelementptr inbounds %struct.mpc_pdata_single_t, %struct.mpc_pdata_single_t* %116, i32 0, i32 0
  %118 = load i8, i8* %117, align 8
  %119 = getelementptr inbounds [2 x i8], [2 x i8]* %8, i64 0, i64 0
  store i8 %118, i8* %119, align 1
  %120 = getelementptr inbounds [2 x i8], [2 x i8]* %8, i64 0, i64 1
  store i8 0, i8* %120, align 1
  %121 = getelementptr inbounds [2 x i8], [2 x i8]* %8, i32 0, i32 0
  %122 = call i8* @mpcf_escape_new(i8* %121, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @mpc_escape_input_c, i32 0, i32 0), i8** getelementptr inbounds ([12 x i8*], [12 x i8*]* @mpc_escape_output_c, i32 0, i32 0))
  store i8* %122, i8** %6, align 8
  %123 = load i8*, i8** %6, align 8
  %124 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.138, i32 0, i32 0), i8* %123)
  %125 = load i8*, i8** %6, align 8
  call void @free(i8* %125) #5
  br label %126

; <label>:126:                                    ; preds = %113, %107
  %127 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %128 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %127, i32 0, i32 2
  %129 = load i8, i8* %128, align 8
  %130 = sext i8 %129 to i32
  %131 = icmp eq i32 %130, 12
  br i1 %131, label %132, label %156

; <label>:132:                                    ; preds = %126
  %133 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %134 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %133, i32 0, i32 3
  %135 = bitcast %union.mpc_pdata_t* %134 to %struct.mpc_pdata_range_t*
  %136 = getelementptr inbounds %struct.mpc_pdata_range_t, %struct.mpc_pdata_range_t* %135, i32 0, i32 0
  %137 = load i8, i8* %136, align 8
  %138 = getelementptr inbounds [2 x i8], [2 x i8]* %8, i64 0, i64 0
  store i8 %137, i8* %138, align 1
  %139 = getelementptr inbounds [2 x i8], [2 x i8]* %8, i64 0, i64 1
  store i8 0, i8* %139, align 1
  %140 = getelementptr inbounds [2 x i8], [2 x i8]* %8, i32 0, i32 0
  %141 = call i8* @mpcf_escape_new(i8* %140, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @mpc_escape_input_c, i32 0, i32 0), i8** getelementptr inbounds ([12 x i8*], [12 x i8*]* @mpc_escape_output_c, i32 0, i32 0))
  store i8* %141, i8** %6, align 8
  %142 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %143 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %142, i32 0, i32 3
  %144 = bitcast %union.mpc_pdata_t* %143 to %struct.mpc_pdata_range_t*
  %145 = getelementptr inbounds %struct.mpc_pdata_range_t, %struct.mpc_pdata_range_t* %144, i32 0, i32 1
  %146 = load i8, i8* %145, align 1
  %147 = getelementptr inbounds [2 x i8], [2 x i8]* %8, i64 0, i64 0
  store i8 %146, i8* %147, align 1
  %148 = getelementptr inbounds [2 x i8], [2 x i8]* %8, i64 0, i64 1
  store i8 0, i8* %148, align 1
  %149 = getelementptr inbounds [2 x i8], [2 x i8]* %8, i32 0, i32 0
  %150 = call i8* @mpcf_escape_new(i8* %149, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @mpc_escape_input_c, i32 0, i32 0), i8** getelementptr inbounds ([12 x i8*], [12 x i8*]* @mpc_escape_output_c, i32 0, i32 0))
  store i8* %150, i8** %7, align 8
  %151 = load i8*, i8** %6, align 8
  %152 = load i8*, i8** %7, align 8
  %153 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.139, i32 0, i32 0), i8* %151, i8* %152)
  %154 = load i8*, i8** %6, align 8
  call void @free(i8* %154) #5
  %155 = load i8*, i8** %7, align 8
  call void @free(i8* %155) #5
  br label %156

; <label>:156:                                    ; preds = %132, %126
  %157 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %158 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %157, i32 0, i32 2
  %159 = load i8, i8* %158, align 8
  %160 = sext i8 %159 to i32
  %161 = icmp eq i32 %160, 10
  br i1 %161, label %162, label %172

; <label>:162:                                    ; preds = %156
  %163 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %164 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %163, i32 0, i32 3
  %165 = bitcast %union.mpc_pdata_t* %164 to %struct.mpc_pdata_string_t*
  %166 = getelementptr inbounds %struct.mpc_pdata_string_t, %struct.mpc_pdata_string_t* %165, i32 0, i32 0
  %167 = load i8*, i8** %166, align 8
  %168 = call i8* @mpcf_escape_new(i8* %167, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @mpc_escape_input_c, i32 0, i32 0), i8** getelementptr inbounds ([12 x i8*], [12 x i8*]* @mpc_escape_output_c, i32 0, i32 0))
  store i8* %168, i8** %6, align 8
  %169 = load i8*, i8** %6, align 8
  %170 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.140, i32 0, i32 0), i8* %169)
  %171 = load i8*, i8** %6, align 8
  call void @free(i8* %171) #5
  br label %172

; <label>:172:                                    ; preds = %162, %156
  %173 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %174 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %173, i32 0, i32 2
  %175 = load i8, i8* %174, align 8
  %176 = sext i8 %175 to i32
  %177 = icmp eq i32 %176, 11
  br i1 %177, label %178, label %188

; <label>:178:                                    ; preds = %172
  %179 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %180 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %179, i32 0, i32 3
  %181 = bitcast %union.mpc_pdata_t* %180 to %struct.mpc_pdata_string_t*
  %182 = getelementptr inbounds %struct.mpc_pdata_string_t, %struct.mpc_pdata_string_t* %181, i32 0, i32 0
  %183 = load i8*, i8** %182, align 8
  %184 = call i8* @mpcf_escape_new(i8* %183, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @mpc_escape_input_c, i32 0, i32 0), i8** getelementptr inbounds ([12 x i8*], [12 x i8*]* @mpc_escape_output_c, i32 0, i32 0))
  store i8* %184, i8** %6, align 8
  %185 = load i8*, i8** %6, align 8
  %186 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.141, i32 0, i32 0), i8* %185)
  %187 = load i8*, i8** %6, align 8
  call void @free(i8* %187) #5
  br label %188

; <label>:188:                                    ; preds = %178, %172
  %189 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %190 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %189, i32 0, i32 2
  %191 = load i8, i8* %190, align 8
  %192 = sext i8 %191 to i32
  %193 = icmp eq i32 %192, 14
  br i1 %193, label %194, label %204

; <label>:194:                                    ; preds = %188
  %195 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %196 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %195, i32 0, i32 3
  %197 = bitcast %union.mpc_pdata_t* %196 to %struct.mpc_pdata_string_t*
  %198 = getelementptr inbounds %struct.mpc_pdata_string_t, %struct.mpc_pdata_string_t* %197, i32 0, i32 0
  %199 = load i8*, i8** %198, align 8
  %200 = call i8* @mpcf_escape_new(i8* %199, i8* getelementptr inbounds ([11 x i8], [11 x i8]* @mpc_escape_input_c, i32 0, i32 0), i8** getelementptr inbounds ([12 x i8*], [12 x i8*]* @mpc_escape_output_c, i32 0, i32 0))
  store i8* %200, i8** %6, align 8
  %201 = load i8*, i8** %6, align 8
  %202 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.19, i32 0, i32 0), i8* %201)
  %203 = load i8*, i8** %6, align 8
  call void @free(i8* %203) #5
  br label %204

; <label>:204:                                    ; preds = %194, %188
  %205 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %206 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %205, i32 0, i32 2
  %207 = load i8, i8* %206, align 8
  %208 = sext i8 %207 to i32
  %209 = icmp eq i32 %208, 15
  br i1 %209, label %210, label %216

; <label>:210:                                    ; preds = %204
  %211 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %212 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %211, i32 0, i32 3
  %213 = bitcast %union.mpc_pdata_t* %212 to %struct.mpc_pdata_apply_t*
  %214 = getelementptr inbounds %struct.mpc_pdata_apply_t, %struct.mpc_pdata_apply_t* %213, i32 0, i32 0
  %215 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %214, align 8
  call void @mpc_print_unretained(%struct.mpc_parser_t* %215, i32 0)
  br label %216

; <label>:216:                                    ; preds = %210, %204
  %217 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %218 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %217, i32 0, i32 2
  %219 = load i8, i8* %218, align 8
  %220 = sext i8 %219 to i32
  %221 = icmp eq i32 %220, 16
  br i1 %221, label %222, label %228

; <label>:222:                                    ; preds = %216
  %223 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %224 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %223, i32 0, i32 3
  %225 = bitcast %union.mpc_pdata_t* %224 to %struct.mpc_pdata_apply_to_t*
  %226 = getelementptr inbounds %struct.mpc_pdata_apply_to_t, %struct.mpc_pdata_apply_to_t* %225, i32 0, i32 0
  %227 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %226, align 8
  call void @mpc_print_unretained(%struct.mpc_parser_t* %227, i32 0)
  br label %228

; <label>:228:                                    ; preds = %222, %216
  %229 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %230 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %229, i32 0, i32 2
  %231 = load i8, i8* %230, align 8
  %232 = sext i8 %231 to i32
  %233 = icmp eq i32 %232, 17
  br i1 %233, label %234, label %240

; <label>:234:                                    ; preds = %228
  %235 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %236 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %235, i32 0, i32 3
  %237 = bitcast %union.mpc_pdata_t* %236 to %struct.mpc_pdata_predict_t*
  %238 = getelementptr inbounds %struct.mpc_pdata_predict_t, %struct.mpc_pdata_predict_t* %237, i32 0, i32 0
  %239 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %238, align 8
  call void @mpc_print_unretained(%struct.mpc_parser_t* %239, i32 0)
  br label %240

; <label>:240:                                    ; preds = %234, %228
  %241 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %242 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %241, i32 0, i32 2
  %243 = load i8, i8* %242, align 8
  %244 = sext i8 %243 to i32
  %245 = icmp eq i32 %244, 18
  br i1 %245, label %246, label %253

; <label>:246:                                    ; preds = %240
  %247 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %248 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %247, i32 0, i32 3
  %249 = bitcast %union.mpc_pdata_t* %248 to %struct.mpc_pdata_not_t*
  %250 = getelementptr inbounds %struct.mpc_pdata_not_t, %struct.mpc_pdata_not_t* %249, i32 0, i32 0
  %251 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %250, align 8
  call void @mpc_print_unretained(%struct.mpc_parser_t* %251, i32 0)
  %252 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.86, i32 0, i32 0))
  br label %253

; <label>:253:                                    ; preds = %246, %240
  %254 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %255 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %254, i32 0, i32 2
  %256 = load i8, i8* %255, align 8
  %257 = sext i8 %256 to i32
  %258 = icmp eq i32 %257, 19
  br i1 %258, label %259, label %266

; <label>:259:                                    ; preds = %253
  %260 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %261 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %260, i32 0, i32 3
  %262 = bitcast %union.mpc_pdata_t* %261 to %struct.mpc_pdata_not_t*
  %263 = getelementptr inbounds %struct.mpc_pdata_not_t, %struct.mpc_pdata_not_t* %262, i32 0, i32 0
  %264 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %263, align 8
  call void @mpc_print_unretained(%struct.mpc_parser_t* %264, i32 0)
  %265 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.85, i32 0, i32 0))
  br label %266

; <label>:266:                                    ; preds = %259, %253
  %267 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %268 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %267, i32 0, i32 2
  %269 = load i8, i8* %268, align 8
  %270 = sext i8 %269 to i32
  %271 = icmp eq i32 %270, 20
  br i1 %271, label %272, label %279

; <label>:272:                                    ; preds = %266
  %273 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %274 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %273, i32 0, i32 3
  %275 = bitcast %union.mpc_pdata_t* %274 to %struct.mpc_pdata_repeat_t*
  %276 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %275, i32 0, i32 2
  %277 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %276, align 8
  call void @mpc_print_unretained(%struct.mpc_parser_t* %277, i32 0)
  %278 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.74, i32 0, i32 0))
  br label %279

; <label>:279:                                    ; preds = %272, %266
  %280 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %281 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %280, i32 0, i32 2
  %282 = load i8, i8* %281, align 8
  %283 = sext i8 %282 to i32
  %284 = icmp eq i32 %283, 21
  br i1 %284, label %285, label %292

; <label>:285:                                    ; preds = %279
  %286 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %287 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %286, i32 0, i32 3
  %288 = bitcast %union.mpc_pdata_t* %287 to %struct.mpc_pdata_repeat_t*
  %289 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %288, i32 0, i32 2
  %290 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %289, align 8
  call void @mpc_print_unretained(%struct.mpc_parser_t* %290, i32 0)
  %291 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.76, i32 0, i32 0))
  br label %292

; <label>:292:                                    ; preds = %285, %279
  %293 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %294 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %293, i32 0, i32 2
  %295 = load i8, i8* %294, align 8
  %296 = sext i8 %295 to i32
  %297 = icmp eq i32 %296, 22
  br i1 %297, label %298, label %310

; <label>:298:                                    ; preds = %292
  %299 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %300 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %299, i32 0, i32 3
  %301 = bitcast %union.mpc_pdata_t* %300 to %struct.mpc_pdata_repeat_t*
  %302 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %301, i32 0, i32 2
  %303 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %302, align 8
  call void @mpc_print_unretained(%struct.mpc_parser_t* %303, i32 0)
  %304 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %305 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %304, i32 0, i32 3
  %306 = bitcast %union.mpc_pdata_t* %305 to %struct.mpc_pdata_repeat_t*
  %307 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %306, i32 0, i32 0
  %308 = load i32, i32* %307, align 8
  %309 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.142, i32 0, i32 0), i32 %308)
  br label %310

; <label>:310:                                    ; preds = %298, %292
  %311 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %312 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %311, i32 0, i32 2
  %313 = load i8, i8* %312, align 8
  %314 = sext i8 %313 to i32
  %315 = icmp eq i32 %314, 23
  br i1 %315, label %316, label %357

; <label>:316:                                    ; preds = %310
  %317 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.59, i32 0, i32 0))
  store i32 0, i32* %5, align 4
  br label %318

; <label>:318:                                    ; preds = %338, %316
  %319 = load i32, i32* %5, align 4
  %320 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %321 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %320, i32 0, i32 3
  %322 = bitcast %union.mpc_pdata_t* %321 to %struct.mpc_pdata_or_t*
  %323 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %322, i32 0, i32 0
  %324 = load i32, i32* %323, align 8
  %325 = sub nsw i32 %324, 1
  %326 = icmp slt i32 %319, %325
  br i1 %326, label %327, label %341

; <label>:327:                                    ; preds = %318
  %328 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %329 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %328, i32 0, i32 3
  %330 = bitcast %union.mpc_pdata_t* %329 to %struct.mpc_pdata_or_t*
  %331 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %330, i32 0, i32 1
  %332 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %331, align 8
  %333 = load i32, i32* %5, align 4
  %334 = sext i32 %333 to i64
  %335 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %332, i64 %334
  %336 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %335, align 8
  call void @mpc_print_unretained(%struct.mpc_parser_t* %336, i32 0)
  %337 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.143, i32 0, i32 0))
  br label %338

; <label>:338:                                    ; preds = %327
  %339 = load i32, i32* %5, align 4
  %340 = add nsw i32 %339, 1
  store i32 %340, i32* %5, align 4
  br label %318

; <label>:341:                                    ; preds = %318
  %342 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %343 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %342, i32 0, i32 3
  %344 = bitcast %union.mpc_pdata_t* %343 to %struct.mpc_pdata_or_t*
  %345 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %344, i32 0, i32 1
  %346 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %345, align 8
  %347 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %348 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %347, i32 0, i32 3
  %349 = bitcast %union.mpc_pdata_t* %348 to %struct.mpc_pdata_or_t*
  %350 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %349, i32 0, i32 0
  %351 = load i32, i32* %350, align 8
  %352 = sub nsw i32 %351, 1
  %353 = sext i32 %352 to i64
  %354 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %346, i64 %353
  %355 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %354, align 8
  call void @mpc_print_unretained(%struct.mpc_parser_t* %355, i32 0)
  %356 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.60, i32 0, i32 0))
  br label %357

; <label>:357:                                    ; preds = %341, %310
  %358 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %359 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %358, i32 0, i32 2
  %360 = load i8, i8* %359, align 8
  %361 = sext i8 %360 to i32
  %362 = icmp eq i32 %361, 24
  br i1 %362, label %363, label %404

; <label>:363:                                    ; preds = %357
  %364 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.59, i32 0, i32 0))
  store i32 0, i32* %5, align 4
  br label %365

; <label>:365:                                    ; preds = %385, %363
  %366 = load i32, i32* %5, align 4
  %367 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %368 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %367, i32 0, i32 3
  %369 = bitcast %union.mpc_pdata_t* %368 to %struct.mpc_pdata_and_t*
  %370 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %369, i32 0, i32 0
  %371 = load i32, i32* %370, align 8
  %372 = sub nsw i32 %371, 1
  %373 = icmp slt i32 %366, %372
  br i1 %373, label %374, label %388

; <label>:374:                                    ; preds = %365
  %375 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %376 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %375, i32 0, i32 3
  %377 = bitcast %union.mpc_pdata_t* %376 to %struct.mpc_pdata_and_t*
  %378 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %377, i32 0, i32 2
  %379 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %378, align 8
  %380 = load i32, i32* %5, align 4
  %381 = sext i32 %380 to i64
  %382 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %379, i64 %381
  %383 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %382, align 8
  call void @mpc_print_unretained(%struct.mpc_parser_t* %383, i32 0)
  %384 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.144, i32 0, i32 0))
  br label %385

; <label>:385:                                    ; preds = %374
  %386 = load i32, i32* %5, align 4
  %387 = add nsw i32 %386, 1
  store i32 %387, i32* %5, align 4
  br label %365

; <label>:388:                                    ; preds = %365
  %389 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %390 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %389, i32 0, i32 3
  %391 = bitcast %union.mpc_pdata_t* %390 to %struct.mpc_pdata_and_t*
  %392 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %391, i32 0, i32 2
  %393 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %392, align 8
  %394 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %395 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %394, i32 0, i32 3
  %396 = bitcast %union.mpc_pdata_t* %395 to %struct.mpc_pdata_and_t*
  %397 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %396, i32 0, i32 0
  %398 = load i32, i32* %397, align 8
  %399 = sub nsw i32 %398, 1
  %400 = sext i32 %399 to i64
  %401 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %393, i64 %400
  %402 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %401, align 8
  call void @mpc_print_unretained(%struct.mpc_parser_t* %402, i32 0)
  %403 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.60, i32 0, i32 0))
  br label %404

; <label>:404:                                    ; preds = %29, %388, %357
  ret void
}

declare i32 @printf(i8*, ...) #2

; Function Attrs: noinline nounwind optnone uwtable
define i32 @mpc_test_fail(%struct.mpc_parser_t*, i8*, i8*, i32 (i8*, i8*)*, void (i8*)*, void (i8*)*) #0 {
  %7 = alloca i32, align 4
  %8 = alloca %struct.mpc_parser_t*, align 8
  %9 = alloca i8*, align 8
  %10 = alloca i8*, align 8
  %11 = alloca i32 (i8*, i8*)*, align 8
  %12 = alloca void (i8*)*, align 8
  %13 = alloca void (i8*)*, align 8
  %14 = alloca %union.mpc_result_t, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %8, align 8
  store i8* %1, i8** %9, align 8
  store i8* %2, i8** %10, align 8
  store i32 (i8*, i8*)* %3, i32 (i8*, i8*)** %11, align 8
  store void (i8*)* %4, void (i8*)** %12, align 8
  store void (i8*)* %5, void (i8*)** %13, align 8
  %15 = load void (i8*)*, void (i8*)** %13, align 8
  %16 = load i8*, i8** %9, align 8
  %17 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %18 = call i32 @mpc_parse(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.78, i32 0, i32 0), i8* %16, %struct.mpc_parser_t* %17, %union.mpc_result_t* %14)
  %19 = icmp ne i32 %18, 0
  br i1 %19, label %20, label %35

; <label>:20:                                     ; preds = %6
  %21 = load i32 (i8*, i8*)*, i32 (i8*, i8*)** %11, align 8
  %22 = bitcast %union.mpc_result_t* %14 to i8**
  %23 = load i8*, i8** %22, align 8
  %24 = load i8*, i8** %10, align 8
  %25 = call i32 %21(i8* %23, i8* %24)
  %26 = icmp ne i32 %25, 0
  br i1 %26, label %27, label %31

; <label>:27:                                     ; preds = %20
  %28 = load void (i8*)*, void (i8*)** %12, align 8
  %29 = bitcast %union.mpc_result_t* %14 to i8**
  %30 = load i8*, i8** %29, align 8
  call void %28(i8* %30)
  store i32 0, i32* %7, align 4
  br label %38

; <label>:31:                                     ; preds = %20
  %32 = load void (i8*)*, void (i8*)** %12, align 8
  %33 = bitcast %union.mpc_result_t* %14 to i8**
  %34 = load i8*, i8** %33, align 8
  call void %32(i8* %34)
  store i32 1, i32* %7, align 4
  br label %38

; <label>:35:                                     ; preds = %6
  %36 = bitcast %union.mpc_result_t* %14 to %struct.mpc_err_t**
  %37 = load %struct.mpc_err_t*, %struct.mpc_err_t** %36, align 8
  call void @mpc_err_delete(%struct.mpc_err_t* %37)
  store i32 1, i32* %7, align 4
  br label %38

; <label>:38:                                     ; preds = %35, %31, %27
  %39 = load i32, i32* %7, align 4
  ret i32 %39
}

; Function Attrs: noinline nounwind optnone uwtable
define i32 @mpc_test_pass(%struct.mpc_parser_t*, i8*, i8*, i32 (i8*, i8*)*, void (i8*)*, void (i8*)*) #0 {
  %7 = alloca i32, align 4
  %8 = alloca %struct.mpc_parser_t*, align 8
  %9 = alloca i8*, align 8
  %10 = alloca i8*, align 8
  %11 = alloca i32 (i8*, i8*)*, align 8
  %12 = alloca void (i8*)*, align 8
  %13 = alloca void (i8*)*, align 8
  %14 = alloca %union.mpc_result_t, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %8, align 8
  store i8* %1, i8** %9, align 8
  store i8* %2, i8** %10, align 8
  store i32 (i8*, i8*)* %3, i32 (i8*, i8*)** %11, align 8
  store void (i8*)* %4, void (i8*)** %12, align 8
  store void (i8*)* %5, void (i8*)** %13, align 8
  %15 = load i8*, i8** %9, align 8
  %16 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %17 = call i32 @mpc_parse(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.78, i32 0, i32 0), i8* %15, %struct.mpc_parser_t* %16, %union.mpc_result_t* %14)
  %18 = icmp ne i32 %17, 0
  br i1 %18, label %19, label %43

; <label>:19:                                     ; preds = %6
  %20 = load i32 (i8*, i8*)*, i32 (i8*, i8*)** %11, align 8
  %21 = bitcast %union.mpc_result_t* %14 to i8**
  %22 = load i8*, i8** %21, align 8
  %23 = load i8*, i8** %10, align 8
  %24 = call i32 %20(i8* %22, i8* %23)
  %25 = icmp ne i32 %24, 0
  br i1 %25, label %26, label %30

; <label>:26:                                     ; preds = %19
  %27 = load void (i8*)*, void (i8*)** %12, align 8
  %28 = bitcast %union.mpc_result_t* %14 to i8**
  %29 = load i8*, i8** %28, align 8
  call void %27(i8* %29)
  store i32 1, i32* %7, align 4
  br label %48

; <label>:30:                                     ; preds = %19
  %31 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.79, i32 0, i32 0))
  %32 = load void (i8*)*, void (i8*)** %13, align 8
  %33 = bitcast %union.mpc_result_t* %14 to i8**
  %34 = load i8*, i8** %33, align 8
  call void %32(i8* %34)
  %35 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.7, i32 0, i32 0))
  %36 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.80, i32 0, i32 0))
  %37 = load void (i8*)*, void (i8*)** %13, align 8
  %38 = load i8*, i8** %10, align 8
  call void %37(i8* %38)
  %39 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.7, i32 0, i32 0))
  %40 = load void (i8*)*, void (i8*)** %12, align 8
  %41 = bitcast %union.mpc_result_t* %14 to i8**
  %42 = load i8*, i8** %41, align 8
  call void %40(i8* %42)
  store i32 0, i32* %7, align 4
  br label %48

; <label>:43:                                     ; preds = %6
  %44 = bitcast %union.mpc_result_t* %14 to %struct.mpc_err_t**
  %45 = load %struct.mpc_err_t*, %struct.mpc_err_t** %44, align 8
  call void @mpc_err_print(%struct.mpc_err_t* %45)
  %46 = bitcast %union.mpc_result_t* %14 to %struct.mpc_err_t**
  %47 = load %struct.mpc_err_t*, %struct.mpc_err_t** %46, align 8
  call void @mpc_err_delete(%struct.mpc_err_t* %47)
  store i32 0, i32* %7, align 4
  br label %48

; <label>:48:                                     ; preds = %43, %30, %26
  %49 = load i32, i32* %7, align 4
  ret i32 %49
}

; Function Attrs: noinline nounwind optnone uwtable
define void @mpc_ast_delete(%struct.mpc_ast_t*) #0 {
  %2 = alloca %struct.mpc_ast_t*, align 8
  %3 = alloca i32, align 4
  store %struct.mpc_ast_t* %0, %struct.mpc_ast_t** %2, align 8
  %4 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %2, align 8
  %5 = icmp ne %struct.mpc_ast_t* %4, null
  br i1 %5, label %7, label %6

; <label>:6:                                      ; preds = %1
  br label %38

; <label>:7:                                      ; preds = %1
  store i32 0, i32* %3, align 4
  br label %8

; <label>:8:                                      ; preds = %22, %7
  %9 = load i32, i32* %3, align 4
  %10 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %2, align 8
  %11 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %10, i32 0, i32 3
  %12 = load i32, i32* %11, align 8
  %13 = icmp slt i32 %9, %12
  br i1 %13, label %14, label %25

; <label>:14:                                     ; preds = %8
  %15 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %2, align 8
  %16 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %15, i32 0, i32 4
  %17 = load %struct.mpc_ast_t**, %struct.mpc_ast_t*** %16, align 8
  %18 = load i32, i32* %3, align 4
  %19 = sext i32 %18 to i64
  %20 = getelementptr inbounds %struct.mpc_ast_t*, %struct.mpc_ast_t** %17, i64 %19
  %21 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %20, align 8
  call void @mpc_ast_delete(%struct.mpc_ast_t* %21)
  br label %22

; <label>:22:                                     ; preds = %14
  %23 = load i32, i32* %3, align 4
  %24 = add nsw i32 %23, 1
  store i32 %24, i32* %3, align 4
  br label %8

; <label>:25:                                     ; preds = %8
  %26 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %2, align 8
  %27 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %26, i32 0, i32 4
  %28 = load %struct.mpc_ast_t**, %struct.mpc_ast_t*** %27, align 8
  %29 = bitcast %struct.mpc_ast_t** %28 to i8*
  call void @free(i8* %29) #5
  %30 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %2, align 8
  %31 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %30, i32 0, i32 0
  %32 = load i8*, i8** %31, align 8
  call void @free(i8* %32) #5
  %33 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %2, align 8
  %34 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %33, i32 0, i32 1
  %35 = load i8*, i8** %34, align 8
  call void @free(i8* %35) #5
  %36 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %2, align 8
  %37 = bitcast %struct.mpc_ast_t* %36 to i8*
  call void @free(i8* %37) #5
  br label %38

; <label>:38:                                     ; preds = %25, %6
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_ast_t* @mpc_ast_new(i8*, i8*) #0 {
  %3 = alloca i8*, align 8
  %4 = alloca i8*, align 8
  %5 = alloca %struct.mpc_ast_t*, align 8
  %6 = alloca %struct.mpc_state_t, align 8
  store i8* %0, i8** %3, align 8
  store i8* %1, i8** %4, align 8
  %7 = call noalias i8* @malloc(i64 56) #5
  %8 = bitcast i8* %7 to %struct.mpc_ast_t*
  store %struct.mpc_ast_t* %8, %struct.mpc_ast_t** %5, align 8
  %9 = load i8*, i8** %3, align 8
  %10 = call i64 @strlen(i8* %9) #7
  %11 = add i64 %10, 1
  %12 = call noalias i8* @malloc(i64 %11) #5
  %13 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %5, align 8
  %14 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %13, i32 0, i32 0
  store i8* %12, i8** %14, align 8
  %15 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %5, align 8
  %16 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %15, i32 0, i32 0
  %17 = load i8*, i8** %16, align 8
  %18 = load i8*, i8** %3, align 8
  %19 = call i8* @strcpy(i8* %17, i8* %18) #5
  %20 = load i8*, i8** %4, align 8
  %21 = call i64 @strlen(i8* %20) #7
  %22 = add i64 %21, 1
  %23 = call noalias i8* @malloc(i64 %22) #5
  %24 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %5, align 8
  %25 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %24, i32 0, i32 1
  store i8* %23, i8** %25, align 8
  %26 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %5, align 8
  %27 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %26, i32 0, i32 1
  %28 = load i8*, i8** %27, align 8
  %29 = load i8*, i8** %4, align 8
  %30 = call i8* @strcpy(i8* %28, i8* %29) #5
  %31 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %5, align 8
  %32 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %31, i32 0, i32 2
  call void @mpc_state_new(%struct.mpc_state_t* sret %6)
  %33 = bitcast %struct.mpc_state_t* %32 to i8*
  %34 = bitcast %struct.mpc_state_t* %6 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %33, i8* %34, i64 24, i32 8, i1 false)
  %35 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %5, align 8
  %36 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %35, i32 0, i32 3
  store i32 0, i32* %36, align 8
  %37 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %5, align 8
  %38 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %37, i32 0, i32 4
  store %struct.mpc_ast_t** null, %struct.mpc_ast_t*** %38, align 8
  %39 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %5, align 8
  ret %struct.mpc_ast_t* %39
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_state_new(%struct.mpc_state_t* noalias sret) #0 {
  %2 = alloca %struct.mpc_state_t, align 8
  %3 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %2, i32 0, i32 0
  store i64 0, i64* %3, align 8
  %4 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %2, i32 0, i32 1
  store i64 0, i64* %4, align 8
  %5 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %2, i32 0, i32 2
  store i64 0, i64* %5, align 8
  %6 = bitcast %struct.mpc_state_t* %0 to i8*
  %7 = bitcast %struct.mpc_state_t* %2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %6, i8* %7, i64 24, i32 8, i1 false)
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_ast_t* @mpc_ast_build(i32, i8*, ...) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i8*, align 8
  %5 = alloca %struct.mpc_ast_t*, align 8
  %6 = alloca i32, align 4
  %7 = alloca [1 x %struct.__va_list_tag], align 16
  store i32 %0, i32* %3, align 4
  store i8* %1, i8** %4, align 8
  %8 = load i8*, i8** %4, align 8
  %9 = call %struct.mpc_ast_t* @mpc_ast_new(i8* %8, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str.81, i32 0, i32 0))
  store %struct.mpc_ast_t* %9, %struct.mpc_ast_t** %5, align 8
  %10 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %7, i32 0, i32 0
  %11 = bitcast %struct.__va_list_tag* %10 to i8*
  call void @llvm.va_start(i8* %11)
  store i32 0, i32* %6, align 4
  br label %12

; <label>:12:                                     ; preds = %37, %2
  %13 = load i32, i32* %6, align 4
  %14 = load i32, i32* %3, align 4
  %15 = icmp slt i32 %13, %14
  br i1 %15, label %16, label %40

; <label>:16:                                     ; preds = %12
  %17 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %5, align 8
  %18 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %7, i32 0, i32 0
  %19 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %18, i32 0, i32 0
  %20 = load i32, i32* %19, align 16
  %21 = icmp ule i32 %20, 40
  br i1 %21, label %22, label %28

; <label>:22:                                     ; preds = %16
  %23 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %18, i32 0, i32 3
  %24 = load i8*, i8** %23, align 16
  %25 = getelementptr i8, i8* %24, i32 %20
  %26 = bitcast i8* %25 to %struct.mpc_ast_t**
  %27 = add i32 %20, 8
  store i32 %27, i32* %19, align 16
  br label %33

; <label>:28:                                     ; preds = %16
  %29 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %18, i32 0, i32 2
  %30 = load i8*, i8** %29, align 8
  %31 = bitcast i8* %30 to %struct.mpc_ast_t**
  %32 = getelementptr i8, i8* %30, i32 8
  store i8* %32, i8** %29, align 8
  br label %33

; <label>:33:                                     ; preds = %28, %22
  %34 = phi %struct.mpc_ast_t** [ %26, %22 ], [ %31, %28 ]
  %35 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %34, align 8
  %36 = call %struct.mpc_ast_t* @mpc_ast_add_child(%struct.mpc_ast_t* %17, %struct.mpc_ast_t* %35)
  br label %37

; <label>:37:                                     ; preds = %33
  %38 = load i32, i32* %6, align 4
  %39 = add nsw i32 %38, 1
  store i32 %39, i32* %6, align 4
  br label %12

; <label>:40:                                     ; preds = %12
  %41 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %7, i32 0, i32 0
  %42 = bitcast %struct.__va_list_tag* %41 to i8*
  call void @llvm.va_end(i8* %42)
  %43 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %5, align 8
  ret %struct.mpc_ast_t* %43
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_ast_t* @mpc_ast_add_child(%struct.mpc_ast_t*, %struct.mpc_ast_t*) #0 {
  %3 = alloca %struct.mpc_ast_t*, align 8
  %4 = alloca %struct.mpc_ast_t*, align 8
  store %struct.mpc_ast_t* %0, %struct.mpc_ast_t** %3, align 8
  store %struct.mpc_ast_t* %1, %struct.mpc_ast_t** %4, align 8
  %5 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  %6 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %5, i32 0, i32 3
  %7 = load i32, i32* %6, align 8
  %8 = add nsw i32 %7, 1
  store i32 %8, i32* %6, align 8
  %9 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  %10 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %9, i32 0, i32 4
  %11 = load %struct.mpc_ast_t**, %struct.mpc_ast_t*** %10, align 8
  %12 = bitcast %struct.mpc_ast_t** %11 to i8*
  %13 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  %14 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %13, i32 0, i32 3
  %15 = load i32, i32* %14, align 8
  %16 = sext i32 %15 to i64
  %17 = mul i64 8, %16
  %18 = call i8* @realloc(i8* %12, i64 %17) #5
  %19 = bitcast i8* %18 to %struct.mpc_ast_t**
  %20 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  %21 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %20, i32 0, i32 4
  store %struct.mpc_ast_t** %19, %struct.mpc_ast_t*** %21, align 8
  %22 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %23 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  %24 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %23, i32 0, i32 4
  %25 = load %struct.mpc_ast_t**, %struct.mpc_ast_t*** %24, align 8
  %26 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  %27 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %26, i32 0, i32 3
  %28 = load i32, i32* %27, align 8
  %29 = sub nsw i32 %28, 1
  %30 = sext i32 %29 to i64
  %31 = getelementptr inbounds %struct.mpc_ast_t*, %struct.mpc_ast_t** %25, i64 %30
  store %struct.mpc_ast_t* %22, %struct.mpc_ast_t** %31, align 8
  %32 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  ret %struct.mpc_ast_t* %32
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_ast_t* @mpc_ast_add_root(%struct.mpc_ast_t*) #0 {
  %2 = alloca %struct.mpc_ast_t*, align 8
  %3 = alloca %struct.mpc_ast_t*, align 8
  %4 = alloca %struct.mpc_ast_t*, align 8
  store %struct.mpc_ast_t* %0, %struct.mpc_ast_t** %3, align 8
  %5 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  %6 = icmp ne %struct.mpc_ast_t* %5, null
  br i1 %6, label %9, label %7

; <label>:7:                                      ; preds = %1
  %8 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  store %struct.mpc_ast_t* %8, %struct.mpc_ast_t** %2, align 8
  br label %29

; <label>:9:                                      ; preds = %1
  %10 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  %11 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %10, i32 0, i32 3
  %12 = load i32, i32* %11, align 8
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %14, label %16

; <label>:14:                                     ; preds = %9
  %15 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  store %struct.mpc_ast_t* %15, %struct.mpc_ast_t** %2, align 8
  br label %29

; <label>:16:                                     ; preds = %9
  %17 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  %18 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %17, i32 0, i32 3
  %19 = load i32, i32* %18, align 8
  %20 = icmp eq i32 %19, 1
  br i1 %20, label %21, label %23

; <label>:21:                                     ; preds = %16
  %22 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  store %struct.mpc_ast_t* %22, %struct.mpc_ast_t** %2, align 8
  br label %29

; <label>:23:                                     ; preds = %16
  %24 = call %struct.mpc_ast_t* @mpc_ast_new(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.62, i32 0, i32 0), i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str.81, i32 0, i32 0))
  store %struct.mpc_ast_t* %24, %struct.mpc_ast_t** %4, align 8
  %25 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %26 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  %27 = call %struct.mpc_ast_t* @mpc_ast_add_child(%struct.mpc_ast_t* %25, %struct.mpc_ast_t* %26)
  %28 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  store %struct.mpc_ast_t* %28, %struct.mpc_ast_t** %2, align 8
  br label %29

; <label>:29:                                     ; preds = %23, %21, %14, %7
  %30 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %2, align 8
  ret %struct.mpc_ast_t* %30
}

; Function Attrs: noinline nounwind optnone uwtable
define i32 @mpc_ast_eq(%struct.mpc_ast_t*, %struct.mpc_ast_t*) #0 {
  %3 = alloca i32, align 4
  %4 = alloca %struct.mpc_ast_t*, align 8
  %5 = alloca %struct.mpc_ast_t*, align 8
  %6 = alloca i32, align 4
  store %struct.mpc_ast_t* %0, %struct.mpc_ast_t** %4, align 8
  store %struct.mpc_ast_t* %1, %struct.mpc_ast_t** %5, align 8
  %7 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %8 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %7, i32 0, i32 0
  %9 = load i8*, i8** %8, align 8
  %10 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %5, align 8
  %11 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %10, i32 0, i32 0
  %12 = load i8*, i8** %11, align 8
  %13 = call i32 @strcmp(i8* %9, i8* %12) #7
  %14 = icmp ne i32 %13, 0
  br i1 %14, label %15, label %16

; <label>:15:                                     ; preds = %2
  store i32 0, i32* %3, align 4
  br label %65

; <label>:16:                                     ; preds = %2
  %17 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %18 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %17, i32 0, i32 1
  %19 = load i8*, i8** %18, align 8
  %20 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %5, align 8
  %21 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %20, i32 0, i32 1
  %22 = load i8*, i8** %21, align 8
  %23 = call i32 @strcmp(i8* %19, i8* %22) #7
  %24 = icmp ne i32 %23, 0
  br i1 %24, label %25, label %26

; <label>:25:                                     ; preds = %16
  store i32 0, i32* %3, align 4
  br label %65

; <label>:26:                                     ; preds = %16
  %27 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %28 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %27, i32 0, i32 3
  %29 = load i32, i32* %28, align 8
  %30 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %5, align 8
  %31 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %30, i32 0, i32 3
  %32 = load i32, i32* %31, align 8
  %33 = icmp ne i32 %29, %32
  br i1 %33, label %34, label %35

; <label>:34:                                     ; preds = %26
  store i32 0, i32* %3, align 4
  br label %65

; <label>:35:                                     ; preds = %26
  store i32 0, i32* %6, align 4
  br label %36

; <label>:36:                                     ; preds = %61, %35
  %37 = load i32, i32* %6, align 4
  %38 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %39 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %38, i32 0, i32 3
  %40 = load i32, i32* %39, align 8
  %41 = icmp slt i32 %37, %40
  br i1 %41, label %42, label %64

; <label>:42:                                     ; preds = %36
  %43 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %44 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %43, i32 0, i32 4
  %45 = load %struct.mpc_ast_t**, %struct.mpc_ast_t*** %44, align 8
  %46 = load i32, i32* %6, align 4
  %47 = sext i32 %46 to i64
  %48 = getelementptr inbounds %struct.mpc_ast_t*, %struct.mpc_ast_t** %45, i64 %47
  %49 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %48, align 8
  %50 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %5, align 8
  %51 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %50, i32 0, i32 4
  %52 = load %struct.mpc_ast_t**, %struct.mpc_ast_t*** %51, align 8
  %53 = load i32, i32* %6, align 4
  %54 = sext i32 %53 to i64
  %55 = getelementptr inbounds %struct.mpc_ast_t*, %struct.mpc_ast_t** %52, i64 %54
  %56 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %55, align 8
  %57 = call i32 @mpc_ast_eq(%struct.mpc_ast_t* %49, %struct.mpc_ast_t* %56)
  %58 = icmp ne i32 %57, 0
  br i1 %58, label %60, label %59

; <label>:59:                                     ; preds = %42
  store i32 0, i32* %3, align 4
  br label %65

; <label>:60:                                     ; preds = %42
  br label %61

; <label>:61:                                     ; preds = %60
  %62 = load i32, i32* %6, align 4
  %63 = add nsw i32 %62, 1
  store i32 %63, i32* %6, align 4
  br label %36

; <label>:64:                                     ; preds = %36
  store i32 1, i32* %3, align 4
  br label %65

; <label>:65:                                     ; preds = %64, %59, %34, %25, %15
  %66 = load i32, i32* %3, align 4
  ret i32 %66
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_ast_t* @mpc_ast_add_tag(%struct.mpc_ast_t*, i8*) #0 {
  %3 = alloca %struct.mpc_ast_t*, align 8
  %4 = alloca %struct.mpc_ast_t*, align 8
  %5 = alloca i8*, align 8
  store %struct.mpc_ast_t* %0, %struct.mpc_ast_t** %4, align 8
  store i8* %1, i8** %5, align 8
  %6 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %7 = icmp ne %struct.mpc_ast_t* %6, null
  br i1 %7, label %10, label %8

; <label>:8:                                      ; preds = %2
  %9 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  store %struct.mpc_ast_t* %9, %struct.mpc_ast_t** %3, align 8
  br label %54

; <label>:10:                                     ; preds = %2
  %11 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %12 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %11, i32 0, i32 0
  %13 = load i8*, i8** %12, align 8
  %14 = load i8*, i8** %5, align 8
  %15 = call i64 @strlen(i8* %14) #7
  %16 = add i64 %15, 1
  %17 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %18 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %17, i32 0, i32 0
  %19 = load i8*, i8** %18, align 8
  %20 = call i64 @strlen(i8* %19) #7
  %21 = add i64 %16, %20
  %22 = add i64 %21, 1
  %23 = call i8* @realloc(i8* %13, i64 %22) #5
  %24 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %25 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %24, i32 0, i32 0
  store i8* %23, i8** %25, align 8
  %26 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %27 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %26, i32 0, i32 0
  %28 = load i8*, i8** %27, align 8
  %29 = load i8*, i8** %5, align 8
  %30 = call i64 @strlen(i8* %29) #7
  %31 = getelementptr inbounds i8, i8* %28, i64 %30
  %32 = getelementptr inbounds i8, i8* %31, i64 1
  %33 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %34 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %33, i32 0, i32 0
  %35 = load i8*, i8** %34, align 8
  %36 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %37 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %36, i32 0, i32 0
  %38 = load i8*, i8** %37, align 8
  %39 = call i64 @strlen(i8* %38) #7
  %40 = add i64 %39, 1
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %32, i8* %35, i64 %40, i32 1, i1 false)
  %41 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %42 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %41, i32 0, i32 0
  %43 = load i8*, i8** %42, align 8
  %44 = load i8*, i8** %5, align 8
  %45 = load i8*, i8** %5, align 8
  %46 = call i64 @strlen(i8* %45) #7
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %43, i8* %44, i64 %46, i32 1, i1 false)
  %47 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %48 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %47, i32 0, i32 0
  %49 = load i8*, i8** %48, align 8
  %50 = load i8*, i8** %5, align 8
  %51 = call i64 @strlen(i8* %50) #7
  %52 = getelementptr inbounds i8, i8* %49, i64 %51
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %52, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.82, i32 0, i32 0), i64 1, i32 1, i1 false)
  %53 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  store %struct.mpc_ast_t* %53, %struct.mpc_ast_t** %3, align 8
  br label %54

; <label>:54:                                     ; preds = %10, %8
  %55 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  ret %struct.mpc_ast_t* %55
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_ast_t* @mpc_ast_tag(%struct.mpc_ast_t*, i8*) #0 {
  %3 = alloca %struct.mpc_ast_t*, align 8
  %4 = alloca i8*, align 8
  store %struct.mpc_ast_t* %0, %struct.mpc_ast_t** %3, align 8
  store i8* %1, i8** %4, align 8
  %5 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  %6 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %5, i32 0, i32 0
  %7 = load i8*, i8** %6, align 8
  %8 = load i8*, i8** %4, align 8
  %9 = call i64 @strlen(i8* %8) #7
  %10 = add i64 %9, 1
  %11 = call i8* @realloc(i8* %7, i64 %10) #5
  %12 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  %13 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %12, i32 0, i32 0
  store i8* %11, i8** %13, align 8
  %14 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  %15 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %14, i32 0, i32 0
  %16 = load i8*, i8** %15, align 8
  %17 = load i8*, i8** %4, align 8
  %18 = call i8* @strcpy(i8* %16, i8* %17) #5
  %19 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  ret %struct.mpc_ast_t* %19
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_ast_t* @mpc_ast_state(%struct.mpc_ast_t*, %struct.mpc_state_t* byval align 8) #0 {
  %3 = alloca %struct.mpc_ast_t*, align 8
  %4 = alloca %struct.mpc_ast_t*, align 8
  store %struct.mpc_ast_t* %0, %struct.mpc_ast_t** %4, align 8
  %5 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %6 = icmp ne %struct.mpc_ast_t* %5, null
  br i1 %6, label %9, label %7

; <label>:7:                                      ; preds = %2
  %8 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  store %struct.mpc_ast_t* %8, %struct.mpc_ast_t** %3, align 8
  br label %15

; <label>:9:                                      ; preds = %2
  %10 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %11 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %10, i32 0, i32 2
  %12 = bitcast %struct.mpc_state_t* %11 to i8*
  %13 = bitcast %struct.mpc_state_t* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %12, i8* %13, i64 24, i32 8, i1 false)
  %14 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  store %struct.mpc_ast_t* %14, %struct.mpc_ast_t** %3, align 8
  br label %15

; <label>:15:                                     ; preds = %9, %7
  %16 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  ret %struct.mpc_ast_t* %16
}

; Function Attrs: noinline nounwind optnone uwtable
define void @mpc_ast_print(%struct.mpc_ast_t*) #0 {
  %2 = alloca %struct.mpc_ast_t*, align 8
  store %struct.mpc_ast_t* %0, %struct.mpc_ast_t** %2, align 8
  %3 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %2, align 8
  %4 = load %struct._IO_FILE*, %struct._IO_FILE** @stdout, align 8
  call void @mpc_ast_print_depth(%struct.mpc_ast_t* %3, i32 0, %struct._IO_FILE* %4)
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_ast_print_depth(%struct.mpc_ast_t*, i32, %struct._IO_FILE*) #0 {
  %4 = alloca %struct.mpc_ast_t*, align 8
  %5 = alloca i32, align 4
  %6 = alloca %struct._IO_FILE*, align 8
  %7 = alloca i32, align 4
  store %struct.mpc_ast_t* %0, %struct.mpc_ast_t** %4, align 8
  store i32 %1, i32* %5, align 4
  store %struct._IO_FILE* %2, %struct._IO_FILE** %6, align 8
  %8 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %9 = icmp ne %struct.mpc_ast_t* %8, null
  br i1 %9, label %13, label %10

; <label>:10:                                     ; preds = %3
  %11 = load %struct._IO_FILE*, %struct._IO_FILE** %6, align 8
  %12 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %11, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.145, i32 0, i32 0))
  br label %76

; <label>:13:                                     ; preds = %3
  store i32 0, i32* %7, align 4
  br label %14

; <label>:14:                                     ; preds = %21, %13
  %15 = load i32, i32* %7, align 4
  %16 = load i32, i32* %5, align 4
  %17 = icmp slt i32 %15, %16
  br i1 %17, label %18, label %24

; <label>:18:                                     ; preds = %14
  %19 = load %struct._IO_FILE*, %struct._IO_FILE** %6, align 8
  %20 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %19, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.146, i32 0, i32 0))
  br label %21

; <label>:21:                                     ; preds = %18
  %22 = load i32, i32* %7, align 4
  %23 = add nsw i32 %22, 1
  store i32 %23, i32* %7, align 4
  br label %14

; <label>:24:                                     ; preds = %14
  %25 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %26 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %25, i32 0, i32 1
  %27 = load i8*, i8** %26, align 8
  %28 = call i64 @strlen(i8* %27) #7
  %29 = icmp ne i64 %28, 0
  br i1 %29, label %30, label %49

; <label>:30:                                     ; preds = %24
  %31 = load %struct._IO_FILE*, %struct._IO_FILE** %6, align 8
  %32 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %33 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %32, i32 0, i32 0
  %34 = load i8*, i8** %33, align 8
  %35 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %36 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %35, i32 0, i32 2
  %37 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %36, i32 0, i32 1
  %38 = load i64, i64* %37, align 8
  %39 = add nsw i64 %38, 1
  %40 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %41 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %40, i32 0, i32 2
  %42 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %41, i32 0, i32 2
  %43 = load i64, i64* %42, align 8
  %44 = add nsw i64 %43, 1
  %45 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %46 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %45, i32 0, i32 1
  %47 = load i8*, i8** %46, align 8
  %48 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %31, i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.147, i32 0, i32 0), i8* %34, i64 %39, i64 %44, i8* %47)
  br label %55

; <label>:49:                                     ; preds = %24
  %50 = load %struct._IO_FILE*, %struct._IO_FILE** %6, align 8
  %51 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %52 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %51, i32 0, i32 0
  %53 = load i8*, i8** %52, align 8
  %54 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %50, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.148, i32 0, i32 0), i8* %53)
  br label %55

; <label>:55:                                     ; preds = %49, %30
  store i32 0, i32* %7, align 4
  br label %56

; <label>:56:                                     ; preds = %73, %55
  %57 = load i32, i32* %7, align 4
  %58 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %59 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %58, i32 0, i32 3
  %60 = load i32, i32* %59, align 8
  %61 = icmp slt i32 %57, %60
  br i1 %61, label %62, label %76

; <label>:62:                                     ; preds = %56
  %63 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %4, align 8
  %64 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %63, i32 0, i32 4
  %65 = load %struct.mpc_ast_t**, %struct.mpc_ast_t*** %64, align 8
  %66 = load i32, i32* %7, align 4
  %67 = sext i32 %66 to i64
  %68 = getelementptr inbounds %struct.mpc_ast_t*, %struct.mpc_ast_t** %65, i64 %67
  %69 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %68, align 8
  %70 = load i32, i32* %5, align 4
  %71 = add nsw i32 %70, 1
  %72 = load %struct._IO_FILE*, %struct._IO_FILE** %6, align 8
  call void @mpc_ast_print_depth(%struct.mpc_ast_t* %69, i32 %71, %struct._IO_FILE* %72)
  br label %73

; <label>:73:                                     ; preds = %62
  %74 = load i32, i32* %7, align 4
  %75 = add nsw i32 %74, 1
  store i32 %75, i32* %7, align 4
  br label %56

; <label>:76:                                     ; preds = %10, %56
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define void @mpc_ast_print_to(%struct.mpc_ast_t*, %struct._IO_FILE*) #0 {
  %3 = alloca %struct.mpc_ast_t*, align 8
  %4 = alloca %struct._IO_FILE*, align 8
  store %struct.mpc_ast_t* %0, %struct.mpc_ast_t** %3, align 8
  store %struct._IO_FILE* %1, %struct._IO_FILE** %4, align 8
  %5 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  %6 = load %struct._IO_FILE*, %struct._IO_FILE** %4, align 8
  call void @mpc_ast_print_depth(%struct.mpc_ast_t* %5, i32 0, %struct._IO_FILE* %6)
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_fold_ast(i32, i8**) #0 {
  %3 = alloca i8*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i8**, align 8
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca %struct.mpc_ast_t**, align 8
  %9 = alloca %struct.mpc_ast_t*, align 8
  store i32 %0, i32* %4, align 4
  store i8** %1, i8*** %5, align 8
  %10 = load i8**, i8*** %5, align 8
  %11 = bitcast i8** %10 to %struct.mpc_ast_t**
  store %struct.mpc_ast_t** %11, %struct.mpc_ast_t*** %8, align 8
  %12 = load i32, i32* %4, align 4
  %13 = icmp eq i32 %12, 0
  br i1 %13, label %14, label %15

; <label>:14:                                     ; preds = %2
  store i8* null, i8** %3, align 8
  br label %158

; <label>:15:                                     ; preds = %2
  %16 = load i32, i32* %4, align 4
  %17 = icmp eq i32 %16, 1
  br i1 %17, label %18, label %22

; <label>:18:                                     ; preds = %15
  %19 = load i8**, i8*** %5, align 8
  %20 = getelementptr inbounds i8*, i8** %19, i64 0
  %21 = load i8*, i8** %20, align 8
  store i8* %21, i8** %3, align 8
  br label %158

; <label>:22:                                     ; preds = %15
  %23 = load i32, i32* %4, align 4
  %24 = icmp eq i32 %23, 2
  br i1 %24, label %25, label %34

; <label>:25:                                     ; preds = %22
  %26 = load i8**, i8*** %5, align 8
  %27 = getelementptr inbounds i8*, i8** %26, i64 1
  %28 = load i8*, i8** %27, align 8
  %29 = icmp ne i8* %28, null
  br i1 %29, label %34, label %30

; <label>:30:                                     ; preds = %25
  %31 = load i8**, i8*** %5, align 8
  %32 = getelementptr inbounds i8*, i8** %31, i64 0
  %33 = load i8*, i8** %32, align 8
  store i8* %33, i8** %3, align 8
  br label %158

; <label>:34:                                     ; preds = %25, %22
  %35 = load i32, i32* %4, align 4
  %36 = icmp eq i32 %35, 2
  br i1 %36, label %37, label %46

; <label>:37:                                     ; preds = %34
  %38 = load i8**, i8*** %5, align 8
  %39 = getelementptr inbounds i8*, i8** %38, i64 0
  %40 = load i8*, i8** %39, align 8
  %41 = icmp ne i8* %40, null
  br i1 %41, label %46, label %42

; <label>:42:                                     ; preds = %37
  %43 = load i8**, i8*** %5, align 8
  %44 = getelementptr inbounds i8*, i8** %43, i64 1
  %45 = load i8*, i8** %44, align 8
  store i8* %45, i8** %3, align 8
  br label %158

; <label>:46:                                     ; preds = %37, %34
  %47 = call %struct.mpc_ast_t* @mpc_ast_new(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.62, i32 0, i32 0), i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str.81, i32 0, i32 0))
  store %struct.mpc_ast_t* %47, %struct.mpc_ast_t** %9, align 8
  store i32 0, i32* %6, align 4
  br label %48

; <label>:48:                                     ; preds = %136, %46
  %49 = load i32, i32* %6, align 4
  %50 = load i32, i32* %4, align 4
  %51 = icmp slt i32 %49, %50
  br i1 %51, label %52, label %139

; <label>:52:                                     ; preds = %48
  %53 = load %struct.mpc_ast_t**, %struct.mpc_ast_t*** %8, align 8
  %54 = load i32, i32* %6, align 4
  %55 = sext i32 %54 to i64
  %56 = getelementptr inbounds %struct.mpc_ast_t*, %struct.mpc_ast_t** %53, i64 %55
  %57 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %56, align 8
  %58 = icmp ne %struct.mpc_ast_t* %57, null
  br i1 %58, label %60, label %59

; <label>:59:                                     ; preds = %52
  br label %136

; <label>:60:                                     ; preds = %52
  %61 = load %struct.mpc_ast_t**, %struct.mpc_ast_t*** %8, align 8
  %62 = load i32, i32* %6, align 4
  %63 = sext i32 %62 to i64
  %64 = getelementptr inbounds %struct.mpc_ast_t*, %struct.mpc_ast_t** %61, i64 %63
  %65 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %64, align 8
  %66 = icmp ne %struct.mpc_ast_t* %65, null
  br i1 %66, label %67, label %110

; <label>:67:                                     ; preds = %60
  %68 = load %struct.mpc_ast_t**, %struct.mpc_ast_t*** %8, align 8
  %69 = load i32, i32* %6, align 4
  %70 = sext i32 %69 to i64
  %71 = getelementptr inbounds %struct.mpc_ast_t*, %struct.mpc_ast_t** %68, i64 %70
  %72 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %71, align 8
  %73 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %72, i32 0, i32 3
  %74 = load i32, i32* %73, align 8
  %75 = icmp sgt i32 %74, 0
  br i1 %75, label %76, label %110

; <label>:76:                                     ; preds = %67
  store i32 0, i32* %7, align 4
  br label %77

; <label>:77:                                     ; preds = %101, %76
  %78 = load i32, i32* %7, align 4
  %79 = load %struct.mpc_ast_t**, %struct.mpc_ast_t*** %8, align 8
  %80 = load i32, i32* %6, align 4
  %81 = sext i32 %80 to i64
  %82 = getelementptr inbounds %struct.mpc_ast_t*, %struct.mpc_ast_t** %79, i64 %81
  %83 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %82, align 8
  %84 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %83, i32 0, i32 3
  %85 = load i32, i32* %84, align 8
  %86 = icmp slt i32 %78, %85
  br i1 %86, label %87, label %104

; <label>:87:                                     ; preds = %77
  %88 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %9, align 8
  %89 = load %struct.mpc_ast_t**, %struct.mpc_ast_t*** %8, align 8
  %90 = load i32, i32* %6, align 4
  %91 = sext i32 %90 to i64
  %92 = getelementptr inbounds %struct.mpc_ast_t*, %struct.mpc_ast_t** %89, i64 %91
  %93 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %92, align 8
  %94 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %93, i32 0, i32 4
  %95 = load %struct.mpc_ast_t**, %struct.mpc_ast_t*** %94, align 8
  %96 = load i32, i32* %7, align 4
  %97 = sext i32 %96 to i64
  %98 = getelementptr inbounds %struct.mpc_ast_t*, %struct.mpc_ast_t** %95, i64 %97
  %99 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %98, align 8
  %100 = call %struct.mpc_ast_t* @mpc_ast_add_child(%struct.mpc_ast_t* %88, %struct.mpc_ast_t* %99)
  br label %101

; <label>:101:                                    ; preds = %87
  %102 = load i32, i32* %7, align 4
  %103 = add nsw i32 %102, 1
  store i32 %103, i32* %7, align 4
  br label %77

; <label>:104:                                    ; preds = %77
  %105 = load %struct.mpc_ast_t**, %struct.mpc_ast_t*** %8, align 8
  %106 = load i32, i32* %6, align 4
  %107 = sext i32 %106 to i64
  %108 = getelementptr inbounds %struct.mpc_ast_t*, %struct.mpc_ast_t** %105, i64 %107
  %109 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %108, align 8
  call void @mpc_ast_delete_no_children(%struct.mpc_ast_t* %109)
  br label %135

; <label>:110:                                    ; preds = %67, %60
  %111 = load %struct.mpc_ast_t**, %struct.mpc_ast_t*** %8, align 8
  %112 = load i32, i32* %6, align 4
  %113 = sext i32 %112 to i64
  %114 = getelementptr inbounds %struct.mpc_ast_t*, %struct.mpc_ast_t** %111, i64 %113
  %115 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %114, align 8
  %116 = icmp ne %struct.mpc_ast_t* %115, null
  br i1 %116, label %117, label %134

; <label>:117:                                    ; preds = %110
  %118 = load %struct.mpc_ast_t**, %struct.mpc_ast_t*** %8, align 8
  %119 = load i32, i32* %6, align 4
  %120 = sext i32 %119 to i64
  %121 = getelementptr inbounds %struct.mpc_ast_t*, %struct.mpc_ast_t** %118, i64 %120
  %122 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %121, align 8
  %123 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %122, i32 0, i32 3
  %124 = load i32, i32* %123, align 8
  %125 = icmp eq i32 %124, 0
  br i1 %125, label %126, label %134

; <label>:126:                                    ; preds = %117
  %127 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %9, align 8
  %128 = load %struct.mpc_ast_t**, %struct.mpc_ast_t*** %8, align 8
  %129 = load i32, i32* %6, align 4
  %130 = sext i32 %129 to i64
  %131 = getelementptr inbounds %struct.mpc_ast_t*, %struct.mpc_ast_t** %128, i64 %130
  %132 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %131, align 8
  %133 = call %struct.mpc_ast_t* @mpc_ast_add_child(%struct.mpc_ast_t* %127, %struct.mpc_ast_t* %132)
  br label %134

; <label>:134:                                    ; preds = %126, %117, %110
  br label %135

; <label>:135:                                    ; preds = %134, %104
  br label %136

; <label>:136:                                    ; preds = %135, %59
  %137 = load i32, i32* %6, align 4
  %138 = add nsw i32 %137, 1
  store i32 %138, i32* %6, align 4
  br label %48

; <label>:139:                                    ; preds = %48
  %140 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %9, align 8
  %141 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %140, i32 0, i32 3
  %142 = load i32, i32* %141, align 8
  %143 = icmp ne i32 %142, 0
  br i1 %143, label %144, label %155

; <label>:144:                                    ; preds = %139
  %145 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %9, align 8
  %146 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %145, i32 0, i32 2
  %147 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %9, align 8
  %148 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %147, i32 0, i32 4
  %149 = load %struct.mpc_ast_t**, %struct.mpc_ast_t*** %148, align 8
  %150 = getelementptr inbounds %struct.mpc_ast_t*, %struct.mpc_ast_t** %149, i64 0
  %151 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %150, align 8
  %152 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %151, i32 0, i32 2
  %153 = bitcast %struct.mpc_state_t* %146 to i8*
  %154 = bitcast %struct.mpc_state_t* %152 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %153, i8* %154, i64 24, i32 8, i1 false)
  br label %155

; <label>:155:                                    ; preds = %144, %139
  %156 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %9, align 8
  %157 = bitcast %struct.mpc_ast_t* %156 to i8*
  store i8* %157, i8** %3, align 8
  br label %158

; <label>:158:                                    ; preds = %155, %42, %30, %18, %14
  %159 = load i8*, i8** %3, align 8
  ret i8* %159
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_ast_delete_no_children(%struct.mpc_ast_t*) #0 {
  %2 = alloca %struct.mpc_ast_t*, align 8
  store %struct.mpc_ast_t* %0, %struct.mpc_ast_t** %2, align 8
  %3 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %2, align 8
  %4 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %3, i32 0, i32 4
  %5 = load %struct.mpc_ast_t**, %struct.mpc_ast_t*** %4, align 8
  %6 = bitcast %struct.mpc_ast_t** %5 to i8*
  call void @free(i8* %6) #5
  %7 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %2, align 8
  %8 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %7, i32 0, i32 0
  %9 = load i8*, i8** %8, align 8
  call void @free(i8* %9) #5
  %10 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %2, align 8
  %11 = getelementptr inbounds %struct.mpc_ast_t, %struct.mpc_ast_t* %10, i32 0, i32 1
  %12 = load i8*, i8** %11, align 8
  call void @free(i8* %12) #5
  %13 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %2, align 8
  %14 = bitcast %struct.mpc_ast_t* %13 to i8*
  call void @free(i8* %14) #5
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_str_ast(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca %struct.mpc_ast_t*, align 8
  store i8* %0, i8** %2, align 8
  %4 = load i8*, i8** %2, align 8
  %5 = call %struct.mpc_ast_t* @mpc_ast_new(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str.81, i32 0, i32 0), i8* %4)
  store %struct.mpc_ast_t* %5, %struct.mpc_ast_t** %3, align 8
  %6 = load i8*, i8** %2, align 8
  call void @free(i8* %6) #5
  %7 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %3, align 8
  %8 = bitcast %struct.mpc_ast_t* %7 to i8*
  ret i8* %8
}

; Function Attrs: noinline nounwind optnone uwtable
define i8* @mpcf_state_ast(i32, i8**) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i8**, align 8
  %5 = alloca %struct.mpc_state_t*, align 8
  %6 = alloca %struct.mpc_ast_t*, align 8
  store i32 %0, i32* %3, align 4
  store i8** %1, i8*** %4, align 8
  %7 = load i8**, i8*** %4, align 8
  %8 = bitcast i8** %7 to %struct.mpc_state_t**
  %9 = getelementptr inbounds %struct.mpc_state_t*, %struct.mpc_state_t** %8, i64 0
  %10 = load %struct.mpc_state_t*, %struct.mpc_state_t** %9, align 8
  store %struct.mpc_state_t* %10, %struct.mpc_state_t** %5, align 8
  %11 = load i8**, i8*** %4, align 8
  %12 = bitcast i8** %11 to %struct.mpc_ast_t**
  %13 = getelementptr inbounds %struct.mpc_ast_t*, %struct.mpc_ast_t** %12, i64 1
  %14 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %13, align 8
  store %struct.mpc_ast_t* %14, %struct.mpc_ast_t** %6, align 8
  %15 = load i32, i32* %3, align 4
  %16 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %6, align 8
  %17 = load %struct.mpc_state_t*, %struct.mpc_state_t** %5, align 8
  %18 = call %struct.mpc_ast_t* @mpc_ast_state(%struct.mpc_ast_t* %16, %struct.mpc_state_t* byval align 8 %17)
  store %struct.mpc_ast_t* %18, %struct.mpc_ast_t** %6, align 8
  %19 = load %struct.mpc_state_t*, %struct.mpc_state_t** %5, align 8
  %20 = bitcast %struct.mpc_state_t* %19 to i8*
  call void @free(i8* %20) #5
  %21 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %6, align 8
  %22 = bitcast %struct.mpc_ast_t* %21 to i8*
  ret i8* %22
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpca_state(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  %3 = call %struct.mpc_parser_t* @mpc_state()
  %4 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %5 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcf_state_ast, %struct.mpc_parser_t* %3, %struct.mpc_parser_t* %4, void (i8*)* @free)
  ret %struct.mpc_parser_t* %5
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpca_tag(%struct.mpc_parser_t*, i8*) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca i8*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store i8* %1, i8** %4, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = load i8*, i8** %4, align 8
  %7 = call %struct.mpc_parser_t* @mpc_apply_to(%struct.mpc_parser_t* %5, i8* (i8*, i8*)* bitcast (%struct.mpc_ast_t* (%struct.mpc_ast_t*, i8*)* @mpc_ast_tag to i8* (i8*, i8*)*), i8* %6)
  ret %struct.mpc_parser_t* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpca_add_tag(%struct.mpc_parser_t*, i8*) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca i8*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store i8* %1, i8** %4, align 8
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %6 = load i8*, i8** %4, align 8
  %7 = call %struct.mpc_parser_t* @mpc_apply_to(%struct.mpc_parser_t* %5, i8* (i8*, i8*)* bitcast (%struct.mpc_ast_t* (%struct.mpc_ast_t*, i8*)* @mpc_ast_add_tag to i8* (i8*, i8*)*), i8* %6)
  ret %struct.mpc_parser_t* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpca_root(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  %3 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %4 = call %struct.mpc_parser_t* @mpc_apply(%struct.mpc_parser_t* %3, i8* (i8*)* bitcast (%struct.mpc_ast_t* (%struct.mpc_ast_t*)* @mpc_ast_add_root to i8* (i8*)*))
  ret %struct.mpc_parser_t* %4
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpca_not(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  %3 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %4 = call %struct.mpc_parser_t* @mpc_not(%struct.mpc_parser_t* %3, void (i8*)* bitcast (void (%struct.mpc_ast_t*)* @mpc_ast_delete to void (i8*)*))
  ret %struct.mpc_parser_t* %4
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpca_maybe(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  %3 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %4 = call %struct.mpc_parser_t* @mpc_maybe(%struct.mpc_parser_t* %3)
  ret %struct.mpc_parser_t* %4
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpca_many(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  %3 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %4 = call %struct.mpc_parser_t* @mpc_many(i8* (i32, i8**)* @mpcf_fold_ast, %struct.mpc_parser_t* %3)
  ret %struct.mpc_parser_t* %4
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpca_many1(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  %3 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %4 = call %struct.mpc_parser_t* @mpc_many1(i8* (i32, i8**)* @mpcf_fold_ast, %struct.mpc_parser_t* %3)
  ret %struct.mpc_parser_t* %4
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpca_count(i32, %struct.mpc_parser_t*) #0 {
  %3 = alloca i32, align 4
  %4 = alloca %struct.mpc_parser_t*, align 8
  store i32 %0, i32* %3, align 4
  store %struct.mpc_parser_t* %1, %struct.mpc_parser_t** %4, align 8
  %5 = load i32, i32* %3, align 4
  %6 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %7 = call %struct.mpc_parser_t* @mpc_count(i32 %5, i8* (i32, i8**)* @mpcf_fold_ast, %struct.mpc_parser_t* %6, void (i8*)* bitcast (void (%struct.mpc_ast_t*)* @mpc_ast_delete to void (i8*)*))
  ret %struct.mpc_parser_t* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpca_or(i32, ...) #0 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca [1 x %struct.__va_list_tag], align 16
  %5 = alloca %struct.mpc_parser_t*, align 8
  store i32 %0, i32* %2, align 4
  %6 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %6, %struct.mpc_parser_t** %5, align 8
  %7 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %8 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %7, i32 0, i32 2
  store i8 23, i8* %8, align 8
  %9 = load i32, i32* %2, align 4
  %10 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %11 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %10, i32 0, i32 3
  %12 = bitcast %union.mpc_pdata_t* %11 to %struct.mpc_pdata_or_t*
  %13 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %12, i32 0, i32 0
  store i32 %9, i32* %13, align 8
  %14 = load i32, i32* %2, align 4
  %15 = sext i32 %14 to i64
  %16 = mul i64 8, %15
  %17 = call noalias i8* @malloc(i64 %16) #5
  %18 = bitcast i8* %17 to %struct.mpc_parser_t**
  %19 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %20 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %19, i32 0, i32 3
  %21 = bitcast %union.mpc_pdata_t* %20 to %struct.mpc_pdata_or_t*
  %22 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %21, i32 0, i32 1
  store %struct.mpc_parser_t** %18, %struct.mpc_parser_t*** %22, align 8
  %23 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %4, i32 0, i32 0
  %24 = bitcast %struct.__va_list_tag* %23 to i8*
  call void @llvm.va_start(i8* %24)
  store i32 0, i32* %3, align 4
  br label %25

; <label>:25:                                     ; preds = %56, %1
  %26 = load i32, i32* %3, align 4
  %27 = load i32, i32* %2, align 4
  %28 = icmp slt i32 %26, %27
  br i1 %28, label %29, label %59

; <label>:29:                                     ; preds = %25
  %30 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %4, i32 0, i32 0
  %31 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %30, i32 0, i32 0
  %32 = load i32, i32* %31, align 16
  %33 = icmp ule i32 %32, 40
  br i1 %33, label %34, label %40

; <label>:34:                                     ; preds = %29
  %35 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %30, i32 0, i32 3
  %36 = load i8*, i8** %35, align 16
  %37 = getelementptr i8, i8* %36, i32 %32
  %38 = bitcast i8* %37 to %struct.mpc_parser_t**
  %39 = add i32 %32, 8
  store i32 %39, i32* %31, align 16
  br label %45

; <label>:40:                                     ; preds = %29
  %41 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %30, i32 0, i32 2
  %42 = load i8*, i8** %41, align 8
  %43 = bitcast i8* %42 to %struct.mpc_parser_t**
  %44 = getelementptr i8, i8* %42, i32 8
  store i8* %44, i8** %41, align 8
  br label %45

; <label>:45:                                     ; preds = %40, %34
  %46 = phi %struct.mpc_parser_t** [ %38, %34 ], [ %43, %40 ]
  %47 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %46, align 8
  %48 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %49 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %48, i32 0, i32 3
  %50 = bitcast %union.mpc_pdata_t* %49 to %struct.mpc_pdata_or_t*
  %51 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %50, i32 0, i32 1
  %52 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %51, align 8
  %53 = load i32, i32* %3, align 4
  %54 = sext i32 %53 to i64
  %55 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %52, i64 %54
  store %struct.mpc_parser_t* %47, %struct.mpc_parser_t** %55, align 8
  br label %56

; <label>:56:                                     ; preds = %45
  %57 = load i32, i32* %3, align 4
  %58 = add nsw i32 %57, 1
  store i32 %58, i32* %3, align 4
  br label %25

; <label>:59:                                     ; preds = %25
  %60 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %4, i32 0, i32 0
  %61 = bitcast %struct.__va_list_tag* %60 to i8*
  call void @llvm.va_end(i8* %61)
  %62 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  ret %struct.mpc_parser_t* %62
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpca_and(i32, ...) #0 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca [1 x %struct.__va_list_tag], align 16
  %5 = alloca %struct.mpc_parser_t*, align 8
  store i32 %0, i32* %2, align 4
  %6 = call %struct.mpc_parser_t* @mpc_undefined()
  store %struct.mpc_parser_t* %6, %struct.mpc_parser_t** %5, align 8
  %7 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %8 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %7, i32 0, i32 2
  store i8 24, i8* %8, align 8
  %9 = load i32, i32* %2, align 4
  %10 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %11 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %10, i32 0, i32 3
  %12 = bitcast %union.mpc_pdata_t* %11 to %struct.mpc_pdata_and_t*
  %13 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %12, i32 0, i32 0
  store i32 %9, i32* %13, align 8
  %14 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %15 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %14, i32 0, i32 3
  %16 = bitcast %union.mpc_pdata_t* %15 to %struct.mpc_pdata_and_t*
  %17 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %16, i32 0, i32 1
  store i8* (i32, i8**)* @mpcf_fold_ast, i8* (i32, i8**)** %17, align 8
  %18 = load i32, i32* %2, align 4
  %19 = sext i32 %18 to i64
  %20 = mul i64 8, %19
  %21 = call noalias i8* @malloc(i64 %20) #5
  %22 = bitcast i8* %21 to %struct.mpc_parser_t**
  %23 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %24 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %23, i32 0, i32 3
  %25 = bitcast %union.mpc_pdata_t* %24 to %struct.mpc_pdata_and_t*
  %26 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %25, i32 0, i32 2
  store %struct.mpc_parser_t** %22, %struct.mpc_parser_t*** %26, align 8
  %27 = load i32, i32* %2, align 4
  %28 = sub nsw i32 %27, 1
  %29 = sext i32 %28 to i64
  %30 = mul i64 8, %29
  %31 = call noalias i8* @malloc(i64 %30) #5
  %32 = bitcast i8* %31 to void (i8*)**
  %33 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %34 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %33, i32 0, i32 3
  %35 = bitcast %union.mpc_pdata_t* %34 to %struct.mpc_pdata_and_t*
  %36 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %35, i32 0, i32 3
  store void (i8*)** %32, void (i8*)*** %36, align 8
  %37 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %4, i32 0, i32 0
  %38 = bitcast %struct.__va_list_tag* %37 to i8*
  call void @llvm.va_start(i8* %38)
  store i32 0, i32* %3, align 4
  br label %39

; <label>:39:                                     ; preds = %70, %1
  %40 = load i32, i32* %3, align 4
  %41 = load i32, i32* %2, align 4
  %42 = icmp slt i32 %40, %41
  br i1 %42, label %43, label %73

; <label>:43:                                     ; preds = %39
  %44 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %4, i32 0, i32 0
  %45 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %44, i32 0, i32 0
  %46 = load i32, i32* %45, align 16
  %47 = icmp ule i32 %46, 40
  br i1 %47, label %48, label %54

; <label>:48:                                     ; preds = %43
  %49 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %44, i32 0, i32 3
  %50 = load i8*, i8** %49, align 16
  %51 = getelementptr i8, i8* %50, i32 %46
  %52 = bitcast i8* %51 to %struct.mpc_parser_t**
  %53 = add i32 %46, 8
  store i32 %53, i32* %45, align 16
  br label %59

; <label>:54:                                     ; preds = %43
  %55 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %44, i32 0, i32 2
  %56 = load i8*, i8** %55, align 8
  %57 = bitcast i8* %56 to %struct.mpc_parser_t**
  %58 = getelementptr i8, i8* %56, i32 8
  store i8* %58, i8** %55, align 8
  br label %59

; <label>:59:                                     ; preds = %54, %48
  %60 = phi %struct.mpc_parser_t** [ %52, %48 ], [ %57, %54 ]
  %61 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %60, align 8
  %62 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %63 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %62, i32 0, i32 3
  %64 = bitcast %union.mpc_pdata_t* %63 to %struct.mpc_pdata_and_t*
  %65 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %64, i32 0, i32 2
  %66 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %65, align 8
  %67 = load i32, i32* %3, align 4
  %68 = sext i32 %67 to i64
  %69 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %66, i64 %68
  store %struct.mpc_parser_t* %61, %struct.mpc_parser_t** %69, align 8
  br label %70

; <label>:70:                                     ; preds = %59
  %71 = load i32, i32* %3, align 4
  %72 = add nsw i32 %71, 1
  store i32 %72, i32* %3, align 4
  br label %39

; <label>:73:                                     ; preds = %39
  store i32 0, i32* %3, align 4
  br label %74

; <label>:74:                                     ; preds = %88, %73
  %75 = load i32, i32* %3, align 4
  %76 = load i32, i32* %2, align 4
  %77 = sub nsw i32 %76, 1
  %78 = icmp slt i32 %75, %77
  br i1 %78, label %79, label %91

; <label>:79:                                     ; preds = %74
  %80 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  %81 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %80, i32 0, i32 3
  %82 = bitcast %union.mpc_pdata_t* %81 to %struct.mpc_pdata_and_t*
  %83 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %82, i32 0, i32 3
  %84 = load void (i8*)**, void (i8*)*** %83, align 8
  %85 = load i32, i32* %3, align 4
  %86 = sext i32 %85 to i64
  %87 = getelementptr inbounds void (i8*)*, void (i8*)** %84, i64 %86
  store void (i8*)* bitcast (void (%struct.mpc_ast_t*)* @mpc_ast_delete to void (i8*)*), void (i8*)** %87, align 8
  br label %88

; <label>:88:                                     ; preds = %79
  %89 = load i32, i32* %3, align 4
  %90 = add nsw i32 %89, 1
  store i32 %90, i32* %3, align 4
  br label %74

; <label>:91:                                     ; preds = %74
  %92 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %4, i32 0, i32 0
  %93 = bitcast %struct.__va_list_tag* %92 to i8*
  call void @llvm.va_end(i8* %93)
  %94 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %5, align 8
  ret %struct.mpc_parser_t* %94
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpca_total(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  %3 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %4 = call %struct.mpc_parser_t* @mpc_total(%struct.mpc_parser_t* %3, void (i8*)* bitcast (void (%struct.mpc_ast_t*)* @mpc_ast_delete to void (i8*)*))
  ret %struct.mpc_parser_t* %4
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpca_grammar_st(i8*, %struct.mpca_grammar_st_t*) #0 {
  %3 = alloca i8*, align 8
  %4 = alloca %struct.mpca_grammar_st_t*, align 8
  %5 = alloca i8*, align 8
  %6 = alloca %struct.mpc_parser_t*, align 8
  %7 = alloca %union.mpc_result_t, align 8
  %8 = alloca %struct.mpc_parser_t*, align 8
  %9 = alloca %struct.mpc_parser_t*, align 8
  %10 = alloca %struct.mpc_parser_t*, align 8
  %11 = alloca %struct.mpc_parser_t*, align 8
  %12 = alloca %struct.mpc_parser_t*, align 8
  store i8* %0, i8** %3, align 8
  store %struct.mpca_grammar_st_t* %1, %struct.mpca_grammar_st_t** %4, align 8
  %13 = call %struct.mpc_parser_t* @mpc_new(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str.83, i32 0, i32 0))
  store %struct.mpc_parser_t* %13, %struct.mpc_parser_t** %8, align 8
  %14 = call %struct.mpc_parser_t* @mpc_new(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.84, i32 0, i32 0))
  store %struct.mpc_parser_t* %14, %struct.mpc_parser_t** %9, align 8
  %15 = call %struct.mpc_parser_t* @mpc_new(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.67, i32 0, i32 0))
  store %struct.mpc_parser_t* %15, %struct.mpc_parser_t** %10, align 8
  %16 = call %struct.mpc_parser_t* @mpc_new(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.68, i32 0, i32 0))
  store %struct.mpc_parser_t* %16, %struct.mpc_parser_t** %11, align 8
  %17 = call %struct.mpc_parser_t* @mpc_new(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.69, i32 0, i32 0))
  store %struct.mpc_parser_t* %17, %struct.mpc_parser_t** %12, align 8
  %18 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %19 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  %20 = call %struct.mpc_parser_t* @mpc_total(%struct.mpc_parser_t* %19, void (i8*)* @mpc_soft_delete)
  %21 = call %struct.mpc_parser_t* @mpc_predictive(%struct.mpc_parser_t* %20)
  %22 = call %struct.mpc_parser_t* @mpc_define(%struct.mpc_parser_t* %18, %struct.mpc_parser_t* %21)
  %23 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  %24 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %10, align 8
  %25 = call %struct.mpc_parser_t* @mpc_sym(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.82, i32 0, i32 0))
  %26 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  %27 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcf_snd_free, %struct.mpc_parser_t* %25, %struct.mpc_parser_t* %26, void (i8*)* @free)
  %28 = call %struct.mpc_parser_t* @mpc_maybe(%struct.mpc_parser_t* %27)
  %29 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcaf_grammar_or, %struct.mpc_parser_t* %24, %struct.mpc_parser_t* %28, void (i8*)* @mpc_soft_delete)
  %30 = call %struct.mpc_parser_t* @mpc_define(%struct.mpc_parser_t* %23, %struct.mpc_parser_t* %29)
  %31 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %10, align 8
  %32 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %11, align 8
  %33 = call %struct.mpc_parser_t* @mpc_many1(i8* (i32, i8**)* @mpcaf_grammar_and, %struct.mpc_parser_t* %32)
  %34 = call %struct.mpc_parser_t* @mpc_define(%struct.mpc_parser_t* %31, %struct.mpc_parser_t* %33)
  %35 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %11, align 8
  %36 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %12, align 8
  %37 = call %struct.mpc_parser_t* @mpc_sym(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.74, i32 0, i32 0))
  %38 = call %struct.mpc_parser_t* @mpc_sym(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.76, i32 0, i32 0))
  %39 = call %struct.mpc_parser_t* @mpc_sym(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.85, i32 0, i32 0))
  %40 = call %struct.mpc_parser_t* @mpc_sym(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.86, i32 0, i32 0))
  %41 = call %struct.mpc_parser_t* @mpc_int()
  %42 = call %struct.mpc_parser_t* @mpc_tok_brackets(%struct.mpc_parser_t* %41, void (i8*)* @free)
  %43 = call %struct.mpc_parser_t* @mpc_pass()
  %44 = call %struct.mpc_parser_t* (i32, ...) @mpc_or(i32 6, %struct.mpc_parser_t* %37, %struct.mpc_parser_t* %38, %struct.mpc_parser_t* %39, %struct.mpc_parser_t* %40, %struct.mpc_parser_t* %42, %struct.mpc_parser_t* %43)
  %45 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcaf_grammar_repeat, %struct.mpc_parser_t* %36, %struct.mpc_parser_t* %44, void (i8*)* @mpc_soft_delete)
  %46 = call %struct.mpc_parser_t* @mpc_define(%struct.mpc_parser_t* %35, %struct.mpc_parser_t* %45)
  %47 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %12, align 8
  %48 = call %struct.mpc_parser_t* @mpc_string_lit()
  %49 = call %struct.mpc_parser_t* @mpc_tok(%struct.mpc_parser_t* %48)
  %50 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %4, align 8
  %51 = bitcast %struct.mpca_grammar_st_t* %50 to i8*
  %52 = call %struct.mpc_parser_t* @mpc_apply_to(%struct.mpc_parser_t* %49, i8* (i8*, i8*)* @mpcaf_grammar_string, i8* %51)
  %53 = call %struct.mpc_parser_t* @mpc_char_lit()
  %54 = call %struct.mpc_parser_t* @mpc_tok(%struct.mpc_parser_t* %53)
  %55 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %4, align 8
  %56 = bitcast %struct.mpca_grammar_st_t* %55 to i8*
  %57 = call %struct.mpc_parser_t* @mpc_apply_to(%struct.mpc_parser_t* %54, i8* (i8*, i8*)* @mpcaf_grammar_char, i8* %56)
  %58 = call %struct.mpc_parser_t* @mpc_regex_lit()
  %59 = call %struct.mpc_parser_t* @mpc_tok(%struct.mpc_parser_t* %58)
  %60 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %4, align 8
  %61 = bitcast %struct.mpca_grammar_st_t* %60 to i8*
  %62 = call %struct.mpc_parser_t* @mpc_apply_to(%struct.mpc_parser_t* %59, i8* (i8*, i8*)* @mpcaf_grammar_regex, i8* %61)
  %63 = call %struct.mpc_parser_t* @mpc_digits()
  %64 = call %struct.mpc_parser_t* @mpc_ident()
  %65 = call %struct.mpc_parser_t* (i32, ...) @mpc_or(i32 2, %struct.mpc_parser_t* %63, %struct.mpc_parser_t* %64)
  %66 = call %struct.mpc_parser_t* @mpc_tok_braces(%struct.mpc_parser_t* %65, void (i8*)* @free)
  %67 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %4, align 8
  %68 = bitcast %struct.mpca_grammar_st_t* %67 to i8*
  %69 = call %struct.mpc_parser_t* @mpc_apply_to(%struct.mpc_parser_t* %66, i8* (i8*, i8*)* @mpcaf_grammar_id, i8* %68)
  %70 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  %71 = call %struct.mpc_parser_t* @mpc_tok_parens(%struct.mpc_parser_t* %70, void (i8*)* @mpc_soft_delete)
  %72 = call %struct.mpc_parser_t* (i32, ...) @mpc_or(i32 5, %struct.mpc_parser_t* %52, %struct.mpc_parser_t* %57, %struct.mpc_parser_t* %62, %struct.mpc_parser_t* %69, %struct.mpc_parser_t* %71)
  %73 = call %struct.mpc_parser_t* @mpc_define(%struct.mpc_parser_t* %47, %struct.mpc_parser_t* %72)
  %74 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  call void @mpc_optimise(%struct.mpc_parser_t* %74)
  %75 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  call void @mpc_optimise(%struct.mpc_parser_t* %75)
  %76 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %11, align 8
  call void @mpc_optimise(%struct.mpc_parser_t* %76)
  %77 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %10, align 8
  call void @mpc_optimise(%struct.mpc_parser_t* %77)
  %78 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %12, align 8
  call void @mpc_optimise(%struct.mpc_parser_t* %78)
  %79 = load i8*, i8** %3, align 8
  %80 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %81 = call i32 @mpc_parse(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.87, i32 0, i32 0), i8* %79, %struct.mpc_parser_t* %80, %union.mpc_result_t* %7)
  %82 = icmp ne i32 %81, 0
  br i1 %82, label %95, label %83

; <label>:83:                                     ; preds = %2
  %84 = bitcast %union.mpc_result_t* %7 to %struct.mpc_err_t**
  %85 = load %struct.mpc_err_t*, %struct.mpc_err_t** %84, align 8
  %86 = call i8* @mpc_err_string(%struct.mpc_err_t* %85)
  store i8* %86, i8** %5, align 8
  %87 = load i8*, i8** %5, align 8
  %88 = call %struct.mpc_parser_t* (i8*, ...) @mpc_failf(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.88, i32 0, i32 0), i8* %87)
  store %struct.mpc_parser_t* %88, %struct.mpc_parser_t** %6, align 8
  %89 = bitcast %union.mpc_result_t* %7 to %struct.mpc_err_t**
  %90 = load %struct.mpc_err_t*, %struct.mpc_err_t** %89, align 8
  call void @mpc_err_delete(%struct.mpc_err_t* %90)
  %91 = load i8*, i8** %5, align 8
  call void @free(i8* %91) #5
  %92 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %6, align 8
  %93 = bitcast %struct.mpc_parser_t* %92 to i8*
  %94 = bitcast %union.mpc_result_t* %7 to i8**
  store i8* %93, i8** %94, align 8
  br label %95

; <label>:95:                                     ; preds = %83, %2
  %96 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %97 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  %98 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %10, align 8
  %99 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %11, align 8
  %100 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %12, align 8
  call void (i32, ...) @mpc_cleanup(i32 5, %struct.mpc_parser_t* %96, %struct.mpc_parser_t* %97, %struct.mpc_parser_t* %98, %struct.mpc_parser_t* %99, %struct.mpc_parser_t* %100)
  %101 = bitcast %union.mpc_result_t* %7 to i8**
  %102 = load i8*, i8** %101, align 8
  %103 = bitcast i8* %102 to %struct.mpc_parser_t*
  call void @mpc_optimise(%struct.mpc_parser_t* %103)
  %104 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %4, align 8
  %105 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %104, i32 0, i32 3
  %106 = load i32, i32* %105, align 8
  %107 = and i32 %106, 1
  %108 = icmp ne i32 %107, 0
  br i1 %108, label %109, label %115

; <label>:109:                                    ; preds = %95
  %110 = bitcast %union.mpc_result_t* %7 to i8**
  %111 = load i8*, i8** %110, align 8
  %112 = bitcast i8* %111 to %struct.mpc_parser_t*
  %113 = call %struct.mpc_parser_t* @mpc_predictive(%struct.mpc_parser_t* %112)
  %114 = bitcast %struct.mpc_parser_t* %113 to i8*
  br label %118

; <label>:115:                                    ; preds = %95
  %116 = bitcast %union.mpc_result_t* %7 to i8**
  %117 = load i8*, i8** %116, align 8
  br label %118

; <label>:118:                                    ; preds = %115, %109
  %119 = phi i8* [ %114, %109 ], [ %117, %115 ]
  %120 = bitcast i8* %119 to %struct.mpc_parser_t*
  ret %struct.mpc_parser_t* %120
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_soft_delete(i8*) #0 {
  %2 = alloca i8*, align 8
  store i8* %0, i8** %2, align 8
  %3 = load i8*, i8** %2, align 8
  %4 = bitcast i8* %3 to %struct.mpc_parser_t*
  call void @mpc_undefine_unretained(%struct.mpc_parser_t* %4, i32 0)
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcaf_grammar_or(i32, i8**) #0 {
  %3 = alloca i8*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i8**, align 8
  store i32 %0, i32* %4, align 4
  store i8** %1, i8*** %5, align 8
  %6 = load i32, i32* %4, align 4
  %7 = load i8**, i8*** %5, align 8
  %8 = getelementptr inbounds i8*, i8** %7, i64 1
  %9 = load i8*, i8** %8, align 8
  %10 = icmp ne i8* %9, null
  br i1 %10, label %15, label %11

; <label>:11:                                     ; preds = %2
  %12 = load i8**, i8*** %5, align 8
  %13 = getelementptr inbounds i8*, i8** %12, i64 0
  %14 = load i8*, i8** %13, align 8
  store i8* %14, i8** %3, align 8
  br label %24

; <label>:15:                                     ; preds = %2
  %16 = load i8**, i8*** %5, align 8
  %17 = getelementptr inbounds i8*, i8** %16, i64 0
  %18 = load i8*, i8** %17, align 8
  %19 = load i8**, i8*** %5, align 8
  %20 = getelementptr inbounds i8*, i8** %19, i64 1
  %21 = load i8*, i8** %20, align 8
  %22 = call %struct.mpc_parser_t* (i32, ...) @mpca_or(i32 2, i8* %18, i8* %21)
  %23 = bitcast %struct.mpc_parser_t* %22 to i8*
  store i8* %23, i8** %3, align 8
  br label %24

; <label>:24:                                     ; preds = %15, %11
  %25 = load i8*, i8** %3, align 8
  ret i8* %25
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcaf_grammar_and(i32, i8**) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i8**, align 8
  %5 = alloca i32, align 4
  %6 = alloca %struct.mpc_parser_t*, align 8
  store i32 %0, i32* %3, align 4
  store i8** %1, i8*** %4, align 8
  %7 = call %struct.mpc_parser_t* @mpc_pass()
  store %struct.mpc_parser_t* %7, %struct.mpc_parser_t** %6, align 8
  store i32 0, i32* %5, align 4
  br label %8

; <label>:8:                                      ; preds = %28, %2
  %9 = load i32, i32* %5, align 4
  %10 = load i32, i32* %3, align 4
  %11 = icmp slt i32 %9, %10
  br i1 %11, label %12, label %31

; <label>:12:                                     ; preds = %8
  %13 = load i8**, i8*** %4, align 8
  %14 = load i32, i32* %5, align 4
  %15 = sext i32 %14 to i64
  %16 = getelementptr inbounds i8*, i8** %13, i64 %15
  %17 = load i8*, i8** %16, align 8
  %18 = icmp ne i8* %17, null
  br i1 %18, label %19, label %27

; <label>:19:                                     ; preds = %12
  %20 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %6, align 8
  %21 = load i8**, i8*** %4, align 8
  %22 = load i32, i32* %5, align 4
  %23 = sext i32 %22 to i64
  %24 = getelementptr inbounds i8*, i8** %21, i64 %23
  %25 = load i8*, i8** %24, align 8
  %26 = call %struct.mpc_parser_t* (i32, ...) @mpca_and(i32 2, %struct.mpc_parser_t* %20, i8* %25)
  store %struct.mpc_parser_t* %26, %struct.mpc_parser_t** %6, align 8
  br label %27

; <label>:27:                                     ; preds = %19, %12
  br label %28

; <label>:28:                                     ; preds = %27
  %29 = load i32, i32* %5, align 4
  %30 = add nsw i32 %29, 1
  store i32 %30, i32* %5, align 4
  br label %8

; <label>:31:                                     ; preds = %8
  %32 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %6, align 8
  %33 = bitcast %struct.mpc_parser_t* %32 to i8*
  ret i8* %33
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcaf_grammar_repeat(i32, i8**) #0 {
  %3 = alloca i8*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i8**, align 8
  %6 = alloca i32, align 4
  store i32 %0, i32* %4, align 4
  store i8** %1, i8*** %5, align 8
  %7 = load i32, i32* %4, align 4
  %8 = load i8**, i8*** %5, align 8
  %9 = getelementptr inbounds i8*, i8** %8, i64 1
  %10 = load i8*, i8** %9, align 8
  %11 = icmp ne i8* %10, null
  br i1 %11, label %16, label %12

; <label>:12:                                     ; preds = %2
  %13 = load i8**, i8*** %5, align 8
  %14 = getelementptr inbounds i8*, i8** %13, i64 0
  %15 = load i8*, i8** %14, align 8
  store i8* %15, i8** %3, align 8
  br label %96

; <label>:16:                                     ; preds = %2
  %17 = load i8**, i8*** %5, align 8
  %18 = getelementptr inbounds i8*, i8** %17, i64 1
  %19 = load i8*, i8** %18, align 8
  %20 = call i32 @strcmp(i8* %19, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.74, i32 0, i32 0)) #7
  %21 = icmp eq i32 %20, 0
  br i1 %21, label %22, label %32

; <label>:22:                                     ; preds = %16
  %23 = load i8**, i8*** %5, align 8
  %24 = getelementptr inbounds i8*, i8** %23, i64 1
  %25 = load i8*, i8** %24, align 8
  call void @free(i8* %25) #5
  %26 = load i8**, i8*** %5, align 8
  %27 = getelementptr inbounds i8*, i8** %26, i64 0
  %28 = load i8*, i8** %27, align 8
  %29 = bitcast i8* %28 to %struct.mpc_parser_t*
  %30 = call %struct.mpc_parser_t* @mpca_many(%struct.mpc_parser_t* %29)
  %31 = bitcast %struct.mpc_parser_t* %30 to i8*
  store i8* %31, i8** %3, align 8
  br label %96

; <label>:32:                                     ; preds = %16
  %33 = load i8**, i8*** %5, align 8
  %34 = getelementptr inbounds i8*, i8** %33, i64 1
  %35 = load i8*, i8** %34, align 8
  %36 = call i32 @strcmp(i8* %35, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.76, i32 0, i32 0)) #7
  %37 = icmp eq i32 %36, 0
  br i1 %37, label %38, label %48

; <label>:38:                                     ; preds = %32
  %39 = load i8**, i8*** %5, align 8
  %40 = getelementptr inbounds i8*, i8** %39, i64 1
  %41 = load i8*, i8** %40, align 8
  call void @free(i8* %41) #5
  %42 = load i8**, i8*** %5, align 8
  %43 = getelementptr inbounds i8*, i8** %42, i64 0
  %44 = load i8*, i8** %43, align 8
  %45 = bitcast i8* %44 to %struct.mpc_parser_t*
  %46 = call %struct.mpc_parser_t* @mpca_many1(%struct.mpc_parser_t* %45)
  %47 = bitcast %struct.mpc_parser_t* %46 to i8*
  store i8* %47, i8** %3, align 8
  br label %96

; <label>:48:                                     ; preds = %32
  %49 = load i8**, i8*** %5, align 8
  %50 = getelementptr inbounds i8*, i8** %49, i64 1
  %51 = load i8*, i8** %50, align 8
  %52 = call i32 @strcmp(i8* %51, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.85, i32 0, i32 0)) #7
  %53 = icmp eq i32 %52, 0
  br i1 %53, label %54, label %64

; <label>:54:                                     ; preds = %48
  %55 = load i8**, i8*** %5, align 8
  %56 = getelementptr inbounds i8*, i8** %55, i64 1
  %57 = load i8*, i8** %56, align 8
  call void @free(i8* %57) #5
  %58 = load i8**, i8*** %5, align 8
  %59 = getelementptr inbounds i8*, i8** %58, i64 0
  %60 = load i8*, i8** %59, align 8
  %61 = bitcast i8* %60 to %struct.mpc_parser_t*
  %62 = call %struct.mpc_parser_t* @mpca_maybe(%struct.mpc_parser_t* %61)
  %63 = bitcast %struct.mpc_parser_t* %62 to i8*
  store i8* %63, i8** %3, align 8
  br label %96

; <label>:64:                                     ; preds = %48
  %65 = load i8**, i8*** %5, align 8
  %66 = getelementptr inbounds i8*, i8** %65, i64 1
  %67 = load i8*, i8** %66, align 8
  %68 = call i32 @strcmp(i8* %67, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.86, i32 0, i32 0)) #7
  %69 = icmp eq i32 %68, 0
  br i1 %69, label %70, label %80

; <label>:70:                                     ; preds = %64
  %71 = load i8**, i8*** %5, align 8
  %72 = getelementptr inbounds i8*, i8** %71, i64 1
  %73 = load i8*, i8** %72, align 8
  call void @free(i8* %73) #5
  %74 = load i8**, i8*** %5, align 8
  %75 = getelementptr inbounds i8*, i8** %74, i64 0
  %76 = load i8*, i8** %75, align 8
  %77 = bitcast i8* %76 to %struct.mpc_parser_t*
  %78 = call %struct.mpc_parser_t* @mpca_not(%struct.mpc_parser_t* %77)
  %79 = bitcast %struct.mpc_parser_t* %78 to i8*
  store i8* %79, i8** %3, align 8
  br label %96

; <label>:80:                                     ; preds = %64
  %81 = load i8**, i8*** %5, align 8
  %82 = getelementptr inbounds i8*, i8** %81, i64 1
  %83 = load i8*, i8** %82, align 8
  %84 = bitcast i8* %83 to i32*
  %85 = load i32, i32* %84, align 4
  store i32 %85, i32* %6, align 4
  %86 = load i8**, i8*** %5, align 8
  %87 = getelementptr inbounds i8*, i8** %86, i64 1
  %88 = load i8*, i8** %87, align 8
  call void @free(i8* %88) #5
  %89 = load i32, i32* %6, align 4
  %90 = load i8**, i8*** %5, align 8
  %91 = getelementptr inbounds i8*, i8** %90, i64 0
  %92 = load i8*, i8** %91, align 8
  %93 = bitcast i8* %92 to %struct.mpc_parser_t*
  %94 = call %struct.mpc_parser_t* @mpca_count(i32 %89, %struct.mpc_parser_t* %93)
  %95 = bitcast %struct.mpc_parser_t* %94 to i8*
  store i8* %95, i8** %3, align 8
  br label %96

; <label>:96:                                     ; preds = %80, %70, %54, %38, %22, %12
  %97 = load i8*, i8** %3, align 8
  ret i8* %97
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcaf_grammar_string(i8*, i8*) #0 {
  %3 = alloca i8*, align 8
  %4 = alloca i8*, align 8
  %5 = alloca %struct.mpca_grammar_st_t*, align 8
  %6 = alloca i8*, align 8
  %7 = alloca %struct.mpc_parser_t*, align 8
  store i8* %0, i8** %3, align 8
  store i8* %1, i8** %4, align 8
  %8 = load i8*, i8** %4, align 8
  %9 = bitcast i8* %8 to %struct.mpca_grammar_st_t*
  store %struct.mpca_grammar_st_t* %9, %struct.mpca_grammar_st_t** %5, align 8
  %10 = load i8*, i8** %3, align 8
  %11 = call i8* @mpcf_unescape(i8* %10)
  store i8* %11, i8** %6, align 8
  %12 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %13 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %12, i32 0, i32 3
  %14 = load i32, i32* %13, align 8
  %15 = and i32 %14, 2
  %16 = icmp ne i32 %15, 0
  br i1 %16, label %17, label %20

; <label>:17:                                     ; preds = %2
  %18 = load i8*, i8** %6, align 8
  %19 = call %struct.mpc_parser_t* @mpc_string(i8* %18)
  br label %24

; <label>:20:                                     ; preds = %2
  %21 = load i8*, i8** %6, align 8
  %22 = call %struct.mpc_parser_t* @mpc_string(i8* %21)
  %23 = call %struct.mpc_parser_t* @mpc_tok(%struct.mpc_parser_t* %22)
  br label %24

; <label>:24:                                     ; preds = %20, %17
  %25 = phi %struct.mpc_parser_t* [ %19, %17 ], [ %23, %20 ]
  store %struct.mpc_parser_t* %25, %struct.mpc_parser_t** %7, align 8
  %26 = load i8*, i8** %6, align 8
  call void @free(i8* %26) #5
  %27 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %28 = call %struct.mpc_parser_t* @mpc_apply(%struct.mpc_parser_t* %27, i8* (i8*)* @mpcf_str_ast)
  %29 = call %struct.mpc_parser_t* @mpca_tag(%struct.mpc_parser_t* %28, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.56, i32 0, i32 0))
  %30 = call %struct.mpc_parser_t* @mpca_state(%struct.mpc_parser_t* %29)
  %31 = bitcast %struct.mpc_parser_t* %30 to i8*
  ret i8* %31
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcaf_grammar_char(i8*, i8*) #0 {
  %3 = alloca i8*, align 8
  %4 = alloca i8*, align 8
  %5 = alloca %struct.mpca_grammar_st_t*, align 8
  %6 = alloca i8*, align 8
  %7 = alloca %struct.mpc_parser_t*, align 8
  store i8* %0, i8** %3, align 8
  store i8* %1, i8** %4, align 8
  %8 = load i8*, i8** %4, align 8
  %9 = bitcast i8* %8 to %struct.mpca_grammar_st_t*
  store %struct.mpca_grammar_st_t* %9, %struct.mpca_grammar_st_t** %5, align 8
  %10 = load i8*, i8** %3, align 8
  %11 = call i8* @mpcf_unescape(i8* %10)
  store i8* %11, i8** %6, align 8
  %12 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %13 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %12, i32 0, i32 3
  %14 = load i32, i32* %13, align 8
  %15 = and i32 %14, 2
  %16 = icmp ne i32 %15, 0
  br i1 %16, label %17, label %22

; <label>:17:                                     ; preds = %2
  %18 = load i8*, i8** %6, align 8
  %19 = getelementptr inbounds i8, i8* %18, i64 0
  %20 = load i8, i8* %19, align 1
  %21 = call %struct.mpc_parser_t* @mpc_char(i8 signext %20)
  br label %28

; <label>:22:                                     ; preds = %2
  %23 = load i8*, i8** %6, align 8
  %24 = getelementptr inbounds i8, i8* %23, i64 0
  %25 = load i8, i8* %24, align 1
  %26 = call %struct.mpc_parser_t* @mpc_char(i8 signext %25)
  %27 = call %struct.mpc_parser_t* @mpc_tok(%struct.mpc_parser_t* %26)
  br label %28

; <label>:28:                                     ; preds = %22, %17
  %29 = phi %struct.mpc_parser_t* [ %21, %17 ], [ %27, %22 ]
  store %struct.mpc_parser_t* %29, %struct.mpc_parser_t** %7, align 8
  %30 = load i8*, i8** %6, align 8
  call void @free(i8* %30) #5
  %31 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %32 = call %struct.mpc_parser_t* @mpc_apply(%struct.mpc_parser_t* %31, i8* (i8*)* @mpcf_str_ast)
  %33 = call %struct.mpc_parser_t* @mpca_tag(%struct.mpc_parser_t* %32, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.54, i32 0, i32 0))
  %34 = call %struct.mpc_parser_t* @mpca_state(%struct.mpc_parser_t* %33)
  %35 = bitcast %struct.mpc_parser_t* %34 to i8*
  ret i8* %35
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcaf_grammar_regex(i8*, i8*) #0 {
  %3 = alloca i8*, align 8
  %4 = alloca i8*, align 8
  %5 = alloca %struct.mpca_grammar_st_t*, align 8
  %6 = alloca i8*, align 8
  %7 = alloca %struct.mpc_parser_t*, align 8
  store i8* %0, i8** %3, align 8
  store i8* %1, i8** %4, align 8
  %8 = load i8*, i8** %4, align 8
  %9 = bitcast i8* %8 to %struct.mpca_grammar_st_t*
  store %struct.mpca_grammar_st_t* %9, %struct.mpca_grammar_st_t** %5, align 8
  %10 = load i8*, i8** %3, align 8
  %11 = call i8* @mpcf_unescape_regex(i8* %10)
  store i8* %11, i8** %6, align 8
  %12 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %13 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %12, i32 0, i32 3
  %14 = load i32, i32* %13, align 8
  %15 = and i32 %14, 2
  %16 = icmp ne i32 %15, 0
  br i1 %16, label %17, label %20

; <label>:17:                                     ; preds = %2
  %18 = load i8*, i8** %6, align 8
  %19 = call %struct.mpc_parser_t* @mpc_re(i8* %18)
  br label %24

; <label>:20:                                     ; preds = %2
  %21 = load i8*, i8** %6, align 8
  %22 = call %struct.mpc_parser_t* @mpc_re(i8* %21)
  %23 = call %struct.mpc_parser_t* @mpc_tok(%struct.mpc_parser_t* %22)
  br label %24

; <label>:24:                                     ; preds = %20, %17
  %25 = phi %struct.mpc_parser_t* [ %19, %17 ], [ %23, %20 ]
  store %struct.mpc_parser_t* %25, %struct.mpc_parser_t** %7, align 8
  %26 = load i8*, i8** %6, align 8
  call void @free(i8* %26) #5
  %27 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %28 = call %struct.mpc_parser_t* @mpc_apply(%struct.mpc_parser_t* %27, i8* (i8*)* @mpcf_str_ast)
  %29 = call %struct.mpc_parser_t* @mpca_tag(%struct.mpc_parser_t* %28, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.58, i32 0, i32 0))
  %30 = call %struct.mpc_parser_t* @mpca_state(%struct.mpc_parser_t* %29)
  %31 = bitcast %struct.mpc_parser_t* %30 to i8*
  ret i8* %31
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcaf_grammar_id(i8*, i8*) #0 {
  %3 = alloca i8*, align 8
  %4 = alloca i8*, align 8
  %5 = alloca i8*, align 8
  %6 = alloca %struct.mpca_grammar_st_t*, align 8
  %7 = alloca %struct.mpc_parser_t*, align 8
  store i8* %0, i8** %4, align 8
  store i8* %1, i8** %5, align 8
  %8 = load i8*, i8** %5, align 8
  %9 = bitcast i8* %8 to %struct.mpca_grammar_st_t*
  store %struct.mpca_grammar_st_t* %9, %struct.mpca_grammar_st_t** %6, align 8
  %10 = load i8*, i8** %4, align 8
  %11 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %6, align 8
  %12 = call %struct.mpc_parser_t* @mpca_grammar_find_parser(i8* %10, %struct.mpca_grammar_st_t* %11)
  store %struct.mpc_parser_t* %12, %struct.mpc_parser_t** %7, align 8
  %13 = load i8*, i8** %4, align 8
  call void @free(i8* %13) #5
  %14 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %15 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %14, i32 0, i32 1
  %16 = load i8*, i8** %15, align 8
  %17 = icmp ne i8* %16, null
  br i1 %17, label %18, label %27

; <label>:18:                                     ; preds = %2
  %19 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %20 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %21 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %20, i32 0, i32 1
  %22 = load i8*, i8** %21, align 8
  %23 = call %struct.mpc_parser_t* @mpca_add_tag(%struct.mpc_parser_t* %19, i8* %22)
  %24 = call %struct.mpc_parser_t* @mpca_root(%struct.mpc_parser_t* %23)
  %25 = call %struct.mpc_parser_t* @mpca_state(%struct.mpc_parser_t* %24)
  %26 = bitcast %struct.mpc_parser_t* %25 to i8*
  store i8* %26, i8** %3, align 8
  br label %32

; <label>:27:                                     ; preds = %2
  %28 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %29 = call %struct.mpc_parser_t* @mpca_root(%struct.mpc_parser_t* %28)
  %30 = call %struct.mpc_parser_t* @mpca_state(%struct.mpc_parser_t* %29)
  %31 = bitcast %struct.mpc_parser_t* %30 to i8*
  store i8* %31, i8** %3, align 8
  br label %32

; <label>:32:                                     ; preds = %27, %18
  %33 = load i8*, i8** %3, align 8
  ret i8* %33
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_parser_t* @mpca_grammar(i32, i8*, ...) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i8*, align 8
  %5 = alloca %struct.mpca_grammar_st_t, align 8
  %6 = alloca %struct.mpc_parser_t*, align 8
  %7 = alloca [1 x %struct.__va_list_tag], align 16
  store i32 %0, i32* %3, align 4
  store i8* %1, i8** %4, align 8
  %8 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %7, i32 0, i32 0
  %9 = bitcast %struct.__va_list_tag* %8 to i8*
  call void @llvm.va_start(i8* %9)
  %10 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %5, i32 0, i32 0
  store [1 x %struct.__va_list_tag]* %7, [1 x %struct.__va_list_tag]** %10, align 8
  %11 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %5, i32 0, i32 1
  store i32 0, i32* %11, align 8
  %12 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %5, i32 0, i32 2
  store %struct.mpc_parser_t** null, %struct.mpc_parser_t*** %12, align 8
  %13 = load i32, i32* %3, align 4
  %14 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %5, i32 0, i32 3
  store i32 %13, i32* %14, align 8
  %15 = load i8*, i8** %4, align 8
  %16 = call %struct.mpc_parser_t* @mpca_grammar_st(i8* %15, %struct.mpca_grammar_st_t* %5)
  store %struct.mpc_parser_t* %16, %struct.mpc_parser_t** %6, align 8
  %17 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %5, i32 0, i32 2
  %18 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %17, align 8
  %19 = bitcast %struct.mpc_parser_t** %18 to i8*
  call void @free(i8* %19) #5
  %20 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %7, i32 0, i32 0
  %21 = bitcast %struct.__va_list_tag* %20 to i8*
  call void @llvm.va_end(i8* %21)
  %22 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %6, align 8
  ret %struct.mpc_parser_t* %22
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_err_t* @mpca_lang_file(i32, %struct._IO_FILE*, ...) #0 {
  %3 = alloca i32, align 4
  %4 = alloca %struct._IO_FILE*, align 8
  %5 = alloca %struct.mpca_grammar_st_t, align 8
  %6 = alloca %struct.mpc_input_t*, align 8
  %7 = alloca %struct.mpc_err_t*, align 8
  %8 = alloca [1 x %struct.__va_list_tag], align 16
  store i32 %0, i32* %3, align 4
  store %struct._IO_FILE* %1, %struct._IO_FILE** %4, align 8
  %9 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %8, i32 0, i32 0
  %10 = bitcast %struct.__va_list_tag* %9 to i8*
  call void @llvm.va_start(i8* %10)
  %11 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %5, i32 0, i32 0
  store [1 x %struct.__va_list_tag]* %8, [1 x %struct.__va_list_tag]** %11, align 8
  %12 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %5, i32 0, i32 1
  store i32 0, i32* %12, align 8
  %13 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %5, i32 0, i32 2
  store %struct.mpc_parser_t** null, %struct.mpc_parser_t*** %13, align 8
  %14 = load i32, i32* %3, align 4
  %15 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %5, i32 0, i32 3
  store i32 %14, i32* %15, align 8
  %16 = load %struct._IO_FILE*, %struct._IO_FILE** %4, align 8
  %17 = call %struct.mpc_input_t* @mpc_input_new_file(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.89, i32 0, i32 0), %struct._IO_FILE* %16)
  store %struct.mpc_input_t* %17, %struct.mpc_input_t** %6, align 8
  %18 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %19 = call %struct.mpc_err_t* @mpca_lang_st(%struct.mpc_input_t* %18, %struct.mpca_grammar_st_t* %5)
  store %struct.mpc_err_t* %19, %struct.mpc_err_t** %7, align 8
  %20 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  call void @mpc_input_delete(%struct.mpc_input_t* %20)
  %21 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %5, i32 0, i32 2
  %22 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %21, align 8
  %23 = bitcast %struct.mpc_parser_t** %22 to i8*
  call void @free(i8* %23) #5
  %24 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %8, i32 0, i32 0
  %25 = bitcast %struct.__va_list_tag* %24 to i8*
  call void @llvm.va_end(i8* %25)
  %26 = load %struct.mpc_err_t*, %struct.mpc_err_t** %7, align 8
  ret %struct.mpc_err_t* %26
}

; Function Attrs: noinline nounwind optnone uwtable
define internal %struct.mpc_err_t* @mpca_lang_st(%struct.mpc_input_t*, %struct.mpca_grammar_st_t*) #0 {
  %3 = alloca %struct.mpc_input_t*, align 8
  %4 = alloca %struct.mpca_grammar_st_t*, align 8
  %5 = alloca %union.mpc_result_t, align 8
  %6 = alloca %struct.mpc_err_t*, align 8
  %7 = alloca %struct.mpc_parser_t*, align 8
  %8 = alloca %struct.mpc_parser_t*, align 8
  %9 = alloca %struct.mpc_parser_t*, align 8
  %10 = alloca %struct.mpc_parser_t*, align 8
  %11 = alloca %struct.mpc_parser_t*, align 8
  %12 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %3, align 8
  store %struct.mpca_grammar_st_t* %1, %struct.mpca_grammar_st_t** %4, align 8
  %13 = call %struct.mpc_parser_t* @mpc_new(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.151, i32 0, i32 0))
  store %struct.mpc_parser_t* %13, %struct.mpc_parser_t** %7, align 8
  %14 = call %struct.mpc_parser_t* @mpc_new(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.152, i32 0, i32 0))
  store %struct.mpc_parser_t* %14, %struct.mpc_parser_t** %8, align 8
  %15 = call %struct.mpc_parser_t* @mpc_new(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.84, i32 0, i32 0))
  store %struct.mpc_parser_t* %15, %struct.mpc_parser_t** %9, align 8
  %16 = call %struct.mpc_parser_t* @mpc_new(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.67, i32 0, i32 0))
  store %struct.mpc_parser_t* %16, %struct.mpc_parser_t** %10, align 8
  %17 = call %struct.mpc_parser_t* @mpc_new(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.68, i32 0, i32 0))
  store %struct.mpc_parser_t* %17, %struct.mpc_parser_t** %11, align 8
  %18 = call %struct.mpc_parser_t* @mpc_new(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.69, i32 0, i32 0))
  store %struct.mpc_parser_t* %18, %struct.mpc_parser_t** %12, align 8
  %19 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %20 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %21 = call %struct.mpc_parser_t* @mpc_many(i8* (i32, i8**)* @mpca_stmt_fold, %struct.mpc_parser_t* %20)
  %22 = call %struct.mpc_parser_t* @mpc_predictive(%struct.mpc_parser_t* %21)
  %23 = call %struct.mpc_parser_t* @mpc_total(%struct.mpc_parser_t* %22, void (i8*)* @mpca_stmt_list_delete)
  %24 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %4, align 8
  %25 = bitcast %struct.mpca_grammar_st_t* %24 to i8*
  %26 = call %struct.mpc_parser_t* @mpc_apply_to(%struct.mpc_parser_t* %23, i8* (i8*, i8*)* @mpca_stmt_list_apply_to, i8* %25)
  %27 = call %struct.mpc_parser_t* @mpc_define(%struct.mpc_parser_t* %19, %struct.mpc_parser_t* %26)
  %28 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %29 = call %struct.mpc_parser_t* @mpc_ident()
  %30 = call %struct.mpc_parser_t* @mpc_tok(%struct.mpc_parser_t* %29)
  %31 = call %struct.mpc_parser_t* @mpc_string_lit()
  %32 = call %struct.mpc_parser_t* @mpc_tok(%struct.mpc_parser_t* %31)
  %33 = call %struct.mpc_parser_t* @mpc_maybe(%struct.mpc_parser_t* %32)
  %34 = call %struct.mpc_parser_t* @mpc_sym(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.153, i32 0, i32 0))
  %35 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  %36 = call %struct.mpc_parser_t* @mpc_sym(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.154, i32 0, i32 0))
  %37 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 5, i8* (i32, i8**)* @mpca_stmt_afold, %struct.mpc_parser_t* %30, %struct.mpc_parser_t* %33, %struct.mpc_parser_t* %34, %struct.mpc_parser_t* %35, %struct.mpc_parser_t* %36, void (i8*)* @free, void (i8*)* @free, void (i8*)* @free, void (i8*)* @mpc_soft_delete)
  %38 = call %struct.mpc_parser_t* @mpc_define(%struct.mpc_parser_t* %28, %struct.mpc_parser_t* %37)
  %39 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  %40 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %10, align 8
  %41 = call %struct.mpc_parser_t* @mpc_sym(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.82, i32 0, i32 0))
  %42 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  %43 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcf_snd_free, %struct.mpc_parser_t* %41, %struct.mpc_parser_t* %42, void (i8*)* @free)
  %44 = call %struct.mpc_parser_t* @mpc_maybe(%struct.mpc_parser_t* %43)
  %45 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcaf_grammar_or, %struct.mpc_parser_t* %40, %struct.mpc_parser_t* %44, void (i8*)* @mpc_soft_delete)
  %46 = call %struct.mpc_parser_t* @mpc_define(%struct.mpc_parser_t* %39, %struct.mpc_parser_t* %45)
  %47 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %10, align 8
  %48 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %11, align 8
  %49 = call %struct.mpc_parser_t* @mpc_many1(i8* (i32, i8**)* @mpcaf_grammar_and, %struct.mpc_parser_t* %48)
  %50 = call %struct.mpc_parser_t* @mpc_define(%struct.mpc_parser_t* %47, %struct.mpc_parser_t* %49)
  %51 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %11, align 8
  %52 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %12, align 8
  %53 = call %struct.mpc_parser_t* @mpc_sym(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.74, i32 0, i32 0))
  %54 = call %struct.mpc_parser_t* @mpc_sym(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.76, i32 0, i32 0))
  %55 = call %struct.mpc_parser_t* @mpc_sym(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.85, i32 0, i32 0))
  %56 = call %struct.mpc_parser_t* @mpc_sym(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.86, i32 0, i32 0))
  %57 = call %struct.mpc_parser_t* @mpc_int()
  %58 = call %struct.mpc_parser_t* @mpc_tok_brackets(%struct.mpc_parser_t* %57, void (i8*)* @free)
  %59 = call %struct.mpc_parser_t* @mpc_pass()
  %60 = call %struct.mpc_parser_t* (i32, ...) @mpc_or(i32 6, %struct.mpc_parser_t* %53, %struct.mpc_parser_t* %54, %struct.mpc_parser_t* %55, %struct.mpc_parser_t* %56, %struct.mpc_parser_t* %58, %struct.mpc_parser_t* %59)
  %61 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcaf_grammar_repeat, %struct.mpc_parser_t* %52, %struct.mpc_parser_t* %60, void (i8*)* @mpc_soft_delete)
  %62 = call %struct.mpc_parser_t* @mpc_define(%struct.mpc_parser_t* %51, %struct.mpc_parser_t* %61)
  %63 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %12, align 8
  %64 = call %struct.mpc_parser_t* @mpc_string_lit()
  %65 = call %struct.mpc_parser_t* @mpc_tok(%struct.mpc_parser_t* %64)
  %66 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %4, align 8
  %67 = bitcast %struct.mpca_grammar_st_t* %66 to i8*
  %68 = call %struct.mpc_parser_t* @mpc_apply_to(%struct.mpc_parser_t* %65, i8* (i8*, i8*)* @mpcaf_grammar_string, i8* %67)
  %69 = call %struct.mpc_parser_t* @mpc_char_lit()
  %70 = call %struct.mpc_parser_t* @mpc_tok(%struct.mpc_parser_t* %69)
  %71 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %4, align 8
  %72 = bitcast %struct.mpca_grammar_st_t* %71 to i8*
  %73 = call %struct.mpc_parser_t* @mpc_apply_to(%struct.mpc_parser_t* %70, i8* (i8*, i8*)* @mpcaf_grammar_char, i8* %72)
  %74 = call %struct.mpc_parser_t* @mpc_regex_lit()
  %75 = call %struct.mpc_parser_t* @mpc_tok(%struct.mpc_parser_t* %74)
  %76 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %4, align 8
  %77 = bitcast %struct.mpca_grammar_st_t* %76 to i8*
  %78 = call %struct.mpc_parser_t* @mpc_apply_to(%struct.mpc_parser_t* %75, i8* (i8*, i8*)* @mpcaf_grammar_regex, i8* %77)
  %79 = call %struct.mpc_parser_t* @mpc_digits()
  %80 = call %struct.mpc_parser_t* @mpc_ident()
  %81 = call %struct.mpc_parser_t* (i32, ...) @mpc_or(i32 2, %struct.mpc_parser_t* %79, %struct.mpc_parser_t* %80)
  %82 = call %struct.mpc_parser_t* @mpc_tok_braces(%struct.mpc_parser_t* %81, void (i8*)* @free)
  %83 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %4, align 8
  %84 = bitcast %struct.mpca_grammar_st_t* %83 to i8*
  %85 = call %struct.mpc_parser_t* @mpc_apply_to(%struct.mpc_parser_t* %82, i8* (i8*, i8*)* @mpcaf_grammar_id, i8* %84)
  %86 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  %87 = call %struct.mpc_parser_t* @mpc_tok_parens(%struct.mpc_parser_t* %86, void (i8*)* @mpc_soft_delete)
  %88 = call %struct.mpc_parser_t* (i32, ...) @mpc_or(i32 5, %struct.mpc_parser_t* %68, %struct.mpc_parser_t* %73, %struct.mpc_parser_t* %78, %struct.mpc_parser_t* %85, %struct.mpc_parser_t* %87)
  %89 = call %struct.mpc_parser_t* @mpc_define(%struct.mpc_parser_t* %63, %struct.mpc_parser_t* %88)
  %90 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  call void @mpc_optimise(%struct.mpc_parser_t* %90)
  %91 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  call void @mpc_optimise(%struct.mpc_parser_t* %91)
  %92 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  call void @mpc_optimise(%struct.mpc_parser_t* %92)
  %93 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %10, align 8
  call void @mpc_optimise(%struct.mpc_parser_t* %93)
  %94 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %11, align 8
  call void @mpc_optimise(%struct.mpc_parser_t* %94)
  %95 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %12, align 8
  call void @mpc_optimise(%struct.mpc_parser_t* %95)
  %96 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %97 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %98 = call i32 @mpc_parse_input(%struct.mpc_input_t* %96, %struct.mpc_parser_t* %97, %union.mpc_result_t* %5)
  %99 = icmp ne i32 %98, 0
  br i1 %99, label %103, label %100

; <label>:100:                                    ; preds = %2
  %101 = bitcast %union.mpc_result_t* %5 to %struct.mpc_err_t**
  %102 = load %struct.mpc_err_t*, %struct.mpc_err_t** %101, align 8
  store %struct.mpc_err_t* %102, %struct.mpc_err_t** %6, align 8
  br label %104

; <label>:103:                                    ; preds = %2
  store %struct.mpc_err_t* null, %struct.mpc_err_t** %6, align 8
  br label %104

; <label>:104:                                    ; preds = %103, %100
  %105 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %106 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %107 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %9, align 8
  %108 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %10, align 8
  %109 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %11, align 8
  %110 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %12, align 8
  call void (i32, ...) @mpc_cleanup(i32 6, %struct.mpc_parser_t* %105, %struct.mpc_parser_t* %106, %struct.mpc_parser_t* %107, %struct.mpc_parser_t* %108, %struct.mpc_parser_t* %109, %struct.mpc_parser_t* %110)
  %111 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  ret %struct.mpc_err_t* %111
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_err_t* @mpca_lang_pipe(i32, %struct._IO_FILE*, ...) #0 {
  %3 = alloca i32, align 4
  %4 = alloca %struct._IO_FILE*, align 8
  %5 = alloca %struct.mpca_grammar_st_t, align 8
  %6 = alloca %struct.mpc_input_t*, align 8
  %7 = alloca %struct.mpc_err_t*, align 8
  %8 = alloca [1 x %struct.__va_list_tag], align 16
  store i32 %0, i32* %3, align 4
  store %struct._IO_FILE* %1, %struct._IO_FILE** %4, align 8
  %9 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %8, i32 0, i32 0
  %10 = bitcast %struct.__va_list_tag* %9 to i8*
  call void @llvm.va_start(i8* %10)
  %11 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %5, i32 0, i32 0
  store [1 x %struct.__va_list_tag]* %8, [1 x %struct.__va_list_tag]** %11, align 8
  %12 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %5, i32 0, i32 1
  store i32 0, i32* %12, align 8
  %13 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %5, i32 0, i32 2
  store %struct.mpc_parser_t** null, %struct.mpc_parser_t*** %13, align 8
  %14 = load i32, i32* %3, align 4
  %15 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %5, i32 0, i32 3
  store i32 %14, i32* %15, align 8
  %16 = load %struct._IO_FILE*, %struct._IO_FILE** %4, align 8
  %17 = call %struct.mpc_input_t* @mpc_input_new_pipe(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.90, i32 0, i32 0), %struct._IO_FILE* %16)
  store %struct.mpc_input_t* %17, %struct.mpc_input_t** %6, align 8
  %18 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %19 = call %struct.mpc_err_t* @mpca_lang_st(%struct.mpc_input_t* %18, %struct.mpca_grammar_st_t* %5)
  store %struct.mpc_err_t* %19, %struct.mpc_err_t** %7, align 8
  %20 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  call void @mpc_input_delete(%struct.mpc_input_t* %20)
  %21 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %5, i32 0, i32 2
  %22 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %21, align 8
  %23 = bitcast %struct.mpc_parser_t** %22 to i8*
  call void @free(i8* %23) #5
  %24 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %8, i32 0, i32 0
  %25 = bitcast %struct.__va_list_tag* %24 to i8*
  call void @llvm.va_end(i8* %25)
  %26 = load %struct.mpc_err_t*, %struct.mpc_err_t** %7, align 8
  ret %struct.mpc_err_t* %26
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_err_t* @mpca_lang(i32, i8*, ...) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i8*, align 8
  %5 = alloca %struct.mpca_grammar_st_t, align 8
  %6 = alloca %struct.mpc_input_t*, align 8
  %7 = alloca %struct.mpc_err_t*, align 8
  %8 = alloca [1 x %struct.__va_list_tag], align 16
  store i32 %0, i32* %3, align 4
  store i8* %1, i8** %4, align 8
  %9 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %8, i32 0, i32 0
  %10 = bitcast %struct.__va_list_tag* %9 to i8*
  call void @llvm.va_start(i8* %10)
  %11 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %5, i32 0, i32 0
  store [1 x %struct.__va_list_tag]* %8, [1 x %struct.__va_list_tag]** %11, align 8
  %12 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %5, i32 0, i32 1
  store i32 0, i32* %12, align 8
  %13 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %5, i32 0, i32 2
  store %struct.mpc_parser_t** null, %struct.mpc_parser_t*** %13, align 8
  %14 = load i32, i32* %3, align 4
  %15 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %5, i32 0, i32 3
  store i32 %14, i32* %15, align 8
  %16 = load i8*, i8** %4, align 8
  %17 = call %struct.mpc_input_t* @mpc_input_new_string(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.91, i32 0, i32 0), i8* %16)
  store %struct.mpc_input_t* %17, %struct.mpc_input_t** %6, align 8
  %18 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %19 = call %struct.mpc_err_t* @mpca_lang_st(%struct.mpc_input_t* %18, %struct.mpca_grammar_st_t* %5)
  store %struct.mpc_err_t* %19, %struct.mpc_err_t** %7, align 8
  %20 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  call void @mpc_input_delete(%struct.mpc_input_t* %20)
  %21 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %5, i32 0, i32 2
  %22 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %21, align 8
  %23 = bitcast %struct.mpc_parser_t** %22 to i8*
  call void @free(i8* %23) #5
  %24 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %8, i32 0, i32 0
  %25 = bitcast %struct.__va_list_tag* %24 to i8*
  call void @llvm.va_end(i8* %25)
  %26 = load %struct.mpc_err_t*, %struct.mpc_err_t** %7, align 8
  ret %struct.mpc_err_t* %26
}

; Function Attrs: noinline nounwind optnone uwtable
define %struct.mpc_err_t* @mpca_lang_contents(i32, i8*, ...) #0 {
  %3 = alloca %struct.mpc_err_t*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i8*, align 8
  %6 = alloca %struct.mpca_grammar_st_t, align 8
  %7 = alloca %struct.mpc_input_t*, align 8
  %8 = alloca %struct.mpc_err_t*, align 8
  %9 = alloca [1 x %struct.__va_list_tag], align 16
  %10 = alloca %struct._IO_FILE*, align 8
  store i32 %0, i32* %4, align 4
  store i8* %1, i8** %5, align 8
  %11 = load i8*, i8** %5, align 8
  %12 = call %struct._IO_FILE* @fopen(i8* %11, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.9, i32 0, i32 0))
  store %struct._IO_FILE* %12, %struct._IO_FILE** %10, align 8
  %13 = load %struct._IO_FILE*, %struct._IO_FILE** %10, align 8
  %14 = icmp ne %struct._IO_FILE* %13, null
  br i1 %14, label %19, label %15

; <label>:15:                                     ; preds = %2
  %16 = load i8*, i8** %5, align 8
  %17 = call %struct.mpc_err_t* @mpc_err_file(i8* %16, i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.10, i32 0, i32 0))
  store %struct.mpc_err_t* %17, %struct.mpc_err_t** %8, align 8
  %18 = load %struct.mpc_err_t*, %struct.mpc_err_t** %8, align 8
  store %struct.mpc_err_t* %18, %struct.mpc_err_t** %3, align 8
  br label %41

; <label>:19:                                     ; preds = %2
  %20 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %9, i32 0, i32 0
  %21 = bitcast %struct.__va_list_tag* %20 to i8*
  call void @llvm.va_start(i8* %21)
  %22 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %6, i32 0, i32 0
  store [1 x %struct.__va_list_tag]* %9, [1 x %struct.__va_list_tag]** %22, align 8
  %23 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %6, i32 0, i32 1
  store i32 0, i32* %23, align 8
  %24 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %6, i32 0, i32 2
  store %struct.mpc_parser_t** null, %struct.mpc_parser_t*** %24, align 8
  %25 = load i32, i32* %4, align 4
  %26 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %6, i32 0, i32 3
  store i32 %25, i32* %26, align 8
  %27 = load i8*, i8** %5, align 8
  %28 = load %struct._IO_FILE*, %struct._IO_FILE** %10, align 8
  %29 = call %struct.mpc_input_t* @mpc_input_new_file(i8* %27, %struct._IO_FILE* %28)
  store %struct.mpc_input_t* %29, %struct.mpc_input_t** %7, align 8
  %30 = load %struct.mpc_input_t*, %struct.mpc_input_t** %7, align 8
  %31 = call %struct.mpc_err_t* @mpca_lang_st(%struct.mpc_input_t* %30, %struct.mpca_grammar_st_t* %6)
  store %struct.mpc_err_t* %31, %struct.mpc_err_t** %8, align 8
  %32 = load %struct.mpc_input_t*, %struct.mpc_input_t** %7, align 8
  call void @mpc_input_delete(%struct.mpc_input_t* %32)
  %33 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %6, i32 0, i32 2
  %34 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %33, align 8
  %35 = bitcast %struct.mpc_parser_t** %34 to i8*
  call void @free(i8* %35) #5
  %36 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %9, i32 0, i32 0
  %37 = bitcast %struct.__va_list_tag* %36 to i8*
  call void @llvm.va_end(i8* %37)
  %38 = load %struct._IO_FILE*, %struct._IO_FILE** %10, align 8
  %39 = call i32 @fclose(%struct._IO_FILE* %38)
  %40 = load %struct.mpc_err_t*, %struct.mpc_err_t** %8, align 8
  store %struct.mpc_err_t* %40, %struct.mpc_err_t** %3, align 8
  br label %41

; <label>:41:                                     ; preds = %19, %15
  %42 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  ret %struct.mpc_err_t* %42
}

; Function Attrs: noinline nounwind optnone uwtable
define void @mpc_stats(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  %3 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.92, i32 0, i32 0))
  %4 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.93, i32 0, i32 0))
  %5 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %6 = call i32 @mpc_nodecount_unretained(%struct.mpc_parser_t* %5, i32 1)
  %7 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.94, i32 0, i32 0), i32 %6)
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @mpc_nodecount_unretained(%struct.mpc_parser_t*, i32) #0 {
  %3 = alloca i32, align 4
  %4 = alloca %struct.mpc_parser_t*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %4, align 8
  store i32 %1, i32* %5, align 4
  %8 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %9 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %8, i32 0, i32 0
  %10 = load i8, i8* %9, align 8
  %11 = sext i8 %10 to i32
  %12 = icmp ne i32 %11, 0
  br i1 %12, label %13, label %17

; <label>:13:                                     ; preds = %2
  %14 = load i32, i32* %5, align 4
  %15 = icmp ne i32 %14, 0
  br i1 %15, label %17, label %16

; <label>:16:                                     ; preds = %13
  store i32 0, i32* %3, align 4
  br label %210

; <label>:17:                                     ; preds = %13, %2
  %18 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %19 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %18, i32 0, i32 2
  %20 = load i8, i8* %19, align 8
  %21 = sext i8 %20 to i32
  %22 = icmp eq i32 %21, 5
  br i1 %22, label %23, label %31

; <label>:23:                                     ; preds = %17
  %24 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %25 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %24, i32 0, i32 3
  %26 = bitcast %union.mpc_pdata_t* %25 to %struct.mpc_pdata_expect_t*
  %27 = getelementptr inbounds %struct.mpc_pdata_expect_t, %struct.mpc_pdata_expect_t* %26, i32 0, i32 0
  %28 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %27, align 8
  %29 = call i32 @mpc_nodecount_unretained(%struct.mpc_parser_t* %28, i32 0)
  %30 = add nsw i32 1, %29
  store i32 %30, i32* %3, align 4
  br label %210

; <label>:31:                                     ; preds = %17
  %32 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %33 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %32, i32 0, i32 2
  %34 = load i8, i8* %33, align 8
  %35 = sext i8 %34 to i32
  %36 = icmp eq i32 %35, 15
  br i1 %36, label %37, label %45

; <label>:37:                                     ; preds = %31
  %38 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %39 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %38, i32 0, i32 3
  %40 = bitcast %union.mpc_pdata_t* %39 to %struct.mpc_pdata_apply_t*
  %41 = getelementptr inbounds %struct.mpc_pdata_apply_t, %struct.mpc_pdata_apply_t* %40, i32 0, i32 0
  %42 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %41, align 8
  %43 = call i32 @mpc_nodecount_unretained(%struct.mpc_parser_t* %42, i32 0)
  %44 = add nsw i32 1, %43
  store i32 %44, i32* %3, align 4
  br label %210

; <label>:45:                                     ; preds = %31
  %46 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %47 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %46, i32 0, i32 2
  %48 = load i8, i8* %47, align 8
  %49 = sext i8 %48 to i32
  %50 = icmp eq i32 %49, 16
  br i1 %50, label %51, label %59

; <label>:51:                                     ; preds = %45
  %52 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %53 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %52, i32 0, i32 3
  %54 = bitcast %union.mpc_pdata_t* %53 to %struct.mpc_pdata_apply_to_t*
  %55 = getelementptr inbounds %struct.mpc_pdata_apply_to_t, %struct.mpc_pdata_apply_to_t* %54, i32 0, i32 0
  %56 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %55, align 8
  %57 = call i32 @mpc_nodecount_unretained(%struct.mpc_parser_t* %56, i32 0)
  %58 = add nsw i32 1, %57
  store i32 %58, i32* %3, align 4
  br label %210

; <label>:59:                                     ; preds = %45
  %60 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %61 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %60, i32 0, i32 2
  %62 = load i8, i8* %61, align 8
  %63 = sext i8 %62 to i32
  %64 = icmp eq i32 %63, 17
  br i1 %64, label %65, label %73

; <label>:65:                                     ; preds = %59
  %66 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %67 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %66, i32 0, i32 3
  %68 = bitcast %union.mpc_pdata_t* %67 to %struct.mpc_pdata_predict_t*
  %69 = getelementptr inbounds %struct.mpc_pdata_predict_t, %struct.mpc_pdata_predict_t* %68, i32 0, i32 0
  %70 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %69, align 8
  %71 = call i32 @mpc_nodecount_unretained(%struct.mpc_parser_t* %70, i32 0)
  %72 = add nsw i32 1, %71
  store i32 %72, i32* %3, align 4
  br label %210

; <label>:73:                                     ; preds = %59
  %74 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %75 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %74, i32 0, i32 2
  %76 = load i8, i8* %75, align 8
  %77 = sext i8 %76 to i32
  %78 = icmp eq i32 %77, 18
  br i1 %78, label %79, label %87

; <label>:79:                                     ; preds = %73
  %80 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %81 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %80, i32 0, i32 3
  %82 = bitcast %union.mpc_pdata_t* %81 to %struct.mpc_pdata_not_t*
  %83 = getelementptr inbounds %struct.mpc_pdata_not_t, %struct.mpc_pdata_not_t* %82, i32 0, i32 0
  %84 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %83, align 8
  %85 = call i32 @mpc_nodecount_unretained(%struct.mpc_parser_t* %84, i32 0)
  %86 = add nsw i32 1, %85
  store i32 %86, i32* %3, align 4
  br label %210

; <label>:87:                                     ; preds = %73
  %88 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %89 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %88, i32 0, i32 2
  %90 = load i8, i8* %89, align 8
  %91 = sext i8 %90 to i32
  %92 = icmp eq i32 %91, 19
  br i1 %92, label %93, label %101

; <label>:93:                                     ; preds = %87
  %94 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %95 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %94, i32 0, i32 3
  %96 = bitcast %union.mpc_pdata_t* %95 to %struct.mpc_pdata_not_t*
  %97 = getelementptr inbounds %struct.mpc_pdata_not_t, %struct.mpc_pdata_not_t* %96, i32 0, i32 0
  %98 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %97, align 8
  %99 = call i32 @mpc_nodecount_unretained(%struct.mpc_parser_t* %98, i32 0)
  %100 = add nsw i32 1, %99
  store i32 %100, i32* %3, align 4
  br label %210

; <label>:101:                                    ; preds = %87
  %102 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %103 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %102, i32 0, i32 2
  %104 = load i8, i8* %103, align 8
  %105 = sext i8 %104 to i32
  %106 = icmp eq i32 %105, 20
  br i1 %106, label %107, label %115

; <label>:107:                                    ; preds = %101
  %108 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %109 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %108, i32 0, i32 3
  %110 = bitcast %union.mpc_pdata_t* %109 to %struct.mpc_pdata_repeat_t*
  %111 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %110, i32 0, i32 2
  %112 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %111, align 8
  %113 = call i32 @mpc_nodecount_unretained(%struct.mpc_parser_t* %112, i32 0)
  %114 = add nsw i32 1, %113
  store i32 %114, i32* %3, align 4
  br label %210

; <label>:115:                                    ; preds = %101
  %116 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %117 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %116, i32 0, i32 2
  %118 = load i8, i8* %117, align 8
  %119 = sext i8 %118 to i32
  %120 = icmp eq i32 %119, 21
  br i1 %120, label %121, label %129

; <label>:121:                                    ; preds = %115
  %122 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %123 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %122, i32 0, i32 3
  %124 = bitcast %union.mpc_pdata_t* %123 to %struct.mpc_pdata_repeat_t*
  %125 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %124, i32 0, i32 2
  %126 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %125, align 8
  %127 = call i32 @mpc_nodecount_unretained(%struct.mpc_parser_t* %126, i32 0)
  %128 = add nsw i32 1, %127
  store i32 %128, i32* %3, align 4
  br label %210

; <label>:129:                                    ; preds = %115
  %130 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %131 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %130, i32 0, i32 2
  %132 = load i8, i8* %131, align 8
  %133 = sext i8 %132 to i32
  %134 = icmp eq i32 %133, 22
  br i1 %134, label %135, label %143

; <label>:135:                                    ; preds = %129
  %136 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %137 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %136, i32 0, i32 3
  %138 = bitcast %union.mpc_pdata_t* %137 to %struct.mpc_pdata_repeat_t*
  %139 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %138, i32 0, i32 2
  %140 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %139, align 8
  %141 = call i32 @mpc_nodecount_unretained(%struct.mpc_parser_t* %140, i32 0)
  %142 = add nsw i32 1, %141
  store i32 %142, i32* %3, align 4
  br label %210

; <label>:143:                                    ; preds = %129
  %144 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %145 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %144, i32 0, i32 2
  %146 = load i8, i8* %145, align 8
  %147 = sext i8 %146 to i32
  %148 = icmp eq i32 %147, 23
  br i1 %148, label %149, label %176

; <label>:149:                                    ; preds = %143
  store i32 0, i32* %7, align 4
  store i32 0, i32* %6, align 4
  br label %150

; <label>:150:                                    ; preds = %171, %149
  %151 = load i32, i32* %6, align 4
  %152 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %153 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %152, i32 0, i32 3
  %154 = bitcast %union.mpc_pdata_t* %153 to %struct.mpc_pdata_or_t*
  %155 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %154, i32 0, i32 0
  %156 = load i32, i32* %155, align 8
  %157 = icmp slt i32 %151, %156
  br i1 %157, label %158, label %174

; <label>:158:                                    ; preds = %150
  %159 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %160 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %159, i32 0, i32 3
  %161 = bitcast %union.mpc_pdata_t* %160 to %struct.mpc_pdata_or_t*
  %162 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %161, i32 0, i32 1
  %163 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %162, align 8
  %164 = load i32, i32* %6, align 4
  %165 = sext i32 %164 to i64
  %166 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %163, i64 %165
  %167 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %166, align 8
  %168 = call i32 @mpc_nodecount_unretained(%struct.mpc_parser_t* %167, i32 0)
  %169 = load i32, i32* %7, align 4
  %170 = add nsw i32 %169, %168
  store i32 %170, i32* %7, align 4
  br label %171

; <label>:171:                                    ; preds = %158
  %172 = load i32, i32* %6, align 4
  %173 = add nsw i32 %172, 1
  store i32 %173, i32* %6, align 4
  br label %150

; <label>:174:                                    ; preds = %150
  %175 = load i32, i32* %7, align 4
  store i32 %175, i32* %3, align 4
  br label %210

; <label>:176:                                    ; preds = %143
  %177 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %178 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %177, i32 0, i32 2
  %179 = load i8, i8* %178, align 8
  %180 = sext i8 %179 to i32
  %181 = icmp eq i32 %180, 24
  br i1 %181, label %182, label %209

; <label>:182:                                    ; preds = %176
  store i32 0, i32* %7, align 4
  store i32 0, i32* %6, align 4
  br label %183

; <label>:183:                                    ; preds = %204, %182
  %184 = load i32, i32* %6, align 4
  %185 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %186 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %185, i32 0, i32 3
  %187 = bitcast %union.mpc_pdata_t* %186 to %struct.mpc_pdata_and_t*
  %188 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %187, i32 0, i32 0
  %189 = load i32, i32* %188, align 8
  %190 = icmp slt i32 %184, %189
  br i1 %190, label %191, label %207

; <label>:191:                                    ; preds = %183
  %192 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %4, align 8
  %193 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %192, i32 0, i32 3
  %194 = bitcast %union.mpc_pdata_t* %193 to %struct.mpc_pdata_and_t*
  %195 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %194, i32 0, i32 2
  %196 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %195, align 8
  %197 = load i32, i32* %6, align 4
  %198 = sext i32 %197 to i64
  %199 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %196, i64 %198
  %200 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %199, align 8
  %201 = call i32 @mpc_nodecount_unretained(%struct.mpc_parser_t* %200, i32 0)
  %202 = load i32, i32* %7, align 4
  %203 = add nsw i32 %202, %201
  store i32 %203, i32* %7, align 4
  br label %204

; <label>:204:                                    ; preds = %191
  %205 = load i32, i32* %6, align 4
  %206 = add nsw i32 %205, 1
  store i32 %206, i32* %6, align 4
  br label %183

; <label>:207:                                    ; preds = %183
  %208 = load i32, i32* %7, align 4
  store i32 %208, i32* %3, align 4
  br label %210

; <label>:209:                                    ; preds = %176
  store i32 1, i32* %3, align 4
  br label %210

; <label>:210:                                    ; preds = %209, %207, %174, %135, %121, %107, %93, %79, %65, %51, %37, %23, %16
  %211 = load i32, i32* %3, align 4
  ret i32 %211
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_optimise_unretained(%struct.mpc_parser_t*, i32) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca %struct.mpc_parser_t*, align 8
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %3, align 8
  store i32 %1, i32* %4, align 4
  %9 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %10 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %9, i32 0, i32 0
  %11 = load i8, i8* %10, align 8
  %12 = sext i8 %11 to i32
  %13 = icmp ne i32 %12, 0
  br i1 %13, label %14, label %18

; <label>:14:                                     ; preds = %2
  %15 = load i32, i32* %4, align 4
  %16 = icmp ne i32 %15, 0
  br i1 %16, label %18, label %17

; <label>:17:                                     ; preds = %14
  br label %1367

; <label>:18:                                     ; preds = %14, %2
  %19 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %20 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %19, i32 0, i32 2
  %21 = load i8, i8* %20, align 8
  %22 = sext i8 %21 to i32
  %23 = icmp eq i32 %22, 5
  br i1 %23, label %24, label %30

; <label>:24:                                     ; preds = %18
  %25 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %26 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %25, i32 0, i32 3
  %27 = bitcast %union.mpc_pdata_t* %26 to %struct.mpc_pdata_expect_t*
  %28 = getelementptr inbounds %struct.mpc_pdata_expect_t, %struct.mpc_pdata_expect_t* %27, i32 0, i32 0
  %29 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %28, align 8
  call void @mpc_optimise_unretained(%struct.mpc_parser_t* %29, i32 0)
  br label %30

; <label>:30:                                     ; preds = %24, %18
  %31 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %32 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %31, i32 0, i32 2
  %33 = load i8, i8* %32, align 8
  %34 = sext i8 %33 to i32
  %35 = icmp eq i32 %34, 15
  br i1 %35, label %36, label %42

; <label>:36:                                     ; preds = %30
  %37 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %38 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %37, i32 0, i32 3
  %39 = bitcast %union.mpc_pdata_t* %38 to %struct.mpc_pdata_apply_t*
  %40 = getelementptr inbounds %struct.mpc_pdata_apply_t, %struct.mpc_pdata_apply_t* %39, i32 0, i32 0
  %41 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %40, align 8
  call void @mpc_optimise_unretained(%struct.mpc_parser_t* %41, i32 0)
  br label %42

; <label>:42:                                     ; preds = %36, %30
  %43 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %44 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %43, i32 0, i32 2
  %45 = load i8, i8* %44, align 8
  %46 = sext i8 %45 to i32
  %47 = icmp eq i32 %46, 16
  br i1 %47, label %48, label %54

; <label>:48:                                     ; preds = %42
  %49 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %50 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %49, i32 0, i32 3
  %51 = bitcast %union.mpc_pdata_t* %50 to %struct.mpc_pdata_apply_to_t*
  %52 = getelementptr inbounds %struct.mpc_pdata_apply_to_t, %struct.mpc_pdata_apply_to_t* %51, i32 0, i32 0
  %53 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %52, align 8
  call void @mpc_optimise_unretained(%struct.mpc_parser_t* %53, i32 0)
  br label %54

; <label>:54:                                     ; preds = %48, %42
  %55 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %56 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %55, i32 0, i32 2
  %57 = load i8, i8* %56, align 8
  %58 = sext i8 %57 to i32
  %59 = icmp eq i32 %58, 17
  br i1 %59, label %60, label %66

; <label>:60:                                     ; preds = %54
  %61 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %62 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %61, i32 0, i32 3
  %63 = bitcast %union.mpc_pdata_t* %62 to %struct.mpc_pdata_predict_t*
  %64 = getelementptr inbounds %struct.mpc_pdata_predict_t, %struct.mpc_pdata_predict_t* %63, i32 0, i32 0
  %65 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %64, align 8
  call void @mpc_optimise_unretained(%struct.mpc_parser_t* %65, i32 0)
  br label %66

; <label>:66:                                     ; preds = %60, %54
  %67 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %68 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %67, i32 0, i32 2
  %69 = load i8, i8* %68, align 8
  %70 = sext i8 %69 to i32
  %71 = icmp eq i32 %70, 18
  br i1 %71, label %72, label %78

; <label>:72:                                     ; preds = %66
  %73 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %74 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %73, i32 0, i32 3
  %75 = bitcast %union.mpc_pdata_t* %74 to %struct.mpc_pdata_not_t*
  %76 = getelementptr inbounds %struct.mpc_pdata_not_t, %struct.mpc_pdata_not_t* %75, i32 0, i32 0
  %77 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %76, align 8
  call void @mpc_optimise_unretained(%struct.mpc_parser_t* %77, i32 0)
  br label %78

; <label>:78:                                     ; preds = %72, %66
  %79 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %80 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %79, i32 0, i32 2
  %81 = load i8, i8* %80, align 8
  %82 = sext i8 %81 to i32
  %83 = icmp eq i32 %82, 19
  br i1 %83, label %84, label %90

; <label>:84:                                     ; preds = %78
  %85 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %86 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %85, i32 0, i32 3
  %87 = bitcast %union.mpc_pdata_t* %86 to %struct.mpc_pdata_not_t*
  %88 = getelementptr inbounds %struct.mpc_pdata_not_t, %struct.mpc_pdata_not_t* %87, i32 0, i32 0
  %89 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %88, align 8
  call void @mpc_optimise_unretained(%struct.mpc_parser_t* %89, i32 0)
  br label %90

; <label>:90:                                     ; preds = %84, %78
  %91 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %92 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %91, i32 0, i32 2
  %93 = load i8, i8* %92, align 8
  %94 = sext i8 %93 to i32
  %95 = icmp eq i32 %94, 20
  br i1 %95, label %96, label %102

; <label>:96:                                     ; preds = %90
  %97 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %98 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %97, i32 0, i32 3
  %99 = bitcast %union.mpc_pdata_t* %98 to %struct.mpc_pdata_repeat_t*
  %100 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %99, i32 0, i32 2
  %101 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %100, align 8
  call void @mpc_optimise_unretained(%struct.mpc_parser_t* %101, i32 0)
  br label %102

; <label>:102:                                    ; preds = %96, %90
  %103 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %104 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %103, i32 0, i32 2
  %105 = load i8, i8* %104, align 8
  %106 = sext i8 %105 to i32
  %107 = icmp eq i32 %106, 21
  br i1 %107, label %108, label %114

; <label>:108:                                    ; preds = %102
  %109 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %110 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %109, i32 0, i32 3
  %111 = bitcast %union.mpc_pdata_t* %110 to %struct.mpc_pdata_repeat_t*
  %112 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %111, i32 0, i32 2
  %113 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %112, align 8
  call void @mpc_optimise_unretained(%struct.mpc_parser_t* %113, i32 0)
  br label %114

; <label>:114:                                    ; preds = %108, %102
  %115 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %116 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %115, i32 0, i32 2
  %117 = load i8, i8* %116, align 8
  %118 = sext i8 %117 to i32
  %119 = icmp eq i32 %118, 22
  br i1 %119, label %120, label %126

; <label>:120:                                    ; preds = %114
  %121 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %122 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %121, i32 0, i32 3
  %123 = bitcast %union.mpc_pdata_t* %122 to %struct.mpc_pdata_repeat_t*
  %124 = getelementptr inbounds %struct.mpc_pdata_repeat_t, %struct.mpc_pdata_repeat_t* %123, i32 0, i32 2
  %125 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %124, align 8
  call void @mpc_optimise_unretained(%struct.mpc_parser_t* %125, i32 0)
  br label %126

; <label>:126:                                    ; preds = %120, %114
  %127 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %128 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %127, i32 0, i32 2
  %129 = load i8, i8* %128, align 8
  %130 = sext i8 %129 to i32
  %131 = icmp eq i32 %130, 23
  br i1 %131, label %132, label %155

; <label>:132:                                    ; preds = %126
  store i32 0, i32* %5, align 4
  br label %133

; <label>:133:                                    ; preds = %151, %132
  %134 = load i32, i32* %5, align 4
  %135 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %136 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %135, i32 0, i32 3
  %137 = bitcast %union.mpc_pdata_t* %136 to %struct.mpc_pdata_or_t*
  %138 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %137, i32 0, i32 0
  %139 = load i32, i32* %138, align 8
  %140 = icmp slt i32 %134, %139
  br i1 %140, label %141, label %154

; <label>:141:                                    ; preds = %133
  %142 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %143 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %142, i32 0, i32 3
  %144 = bitcast %union.mpc_pdata_t* %143 to %struct.mpc_pdata_or_t*
  %145 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %144, i32 0, i32 1
  %146 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %145, align 8
  %147 = load i32, i32* %5, align 4
  %148 = sext i32 %147 to i64
  %149 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %146, i64 %148
  %150 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %149, align 8
  call void @mpc_optimise_unretained(%struct.mpc_parser_t* %150, i32 0)
  br label %151

; <label>:151:                                    ; preds = %141
  %152 = load i32, i32* %5, align 4
  %153 = add nsw i32 %152, 1
  store i32 %153, i32* %5, align 4
  br label %133

; <label>:154:                                    ; preds = %133
  br label %155

; <label>:155:                                    ; preds = %154, %126
  %156 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %157 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %156, i32 0, i32 2
  %158 = load i8, i8* %157, align 8
  %159 = sext i8 %158 to i32
  %160 = icmp eq i32 %159, 24
  br i1 %160, label %161, label %184

; <label>:161:                                    ; preds = %155
  store i32 0, i32* %5, align 4
  br label %162

; <label>:162:                                    ; preds = %180, %161
  %163 = load i32, i32* %5, align 4
  %164 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %165 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %164, i32 0, i32 3
  %166 = bitcast %union.mpc_pdata_t* %165 to %struct.mpc_pdata_and_t*
  %167 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %166, i32 0, i32 0
  %168 = load i32, i32* %167, align 8
  %169 = icmp slt i32 %163, %168
  br i1 %169, label %170, label %183

; <label>:170:                                    ; preds = %162
  %171 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %172 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %171, i32 0, i32 3
  %173 = bitcast %union.mpc_pdata_t* %172 to %struct.mpc_pdata_and_t*
  %174 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %173, i32 0, i32 2
  %175 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %174, align 8
  %176 = load i32, i32* %5, align 4
  %177 = sext i32 %176 to i64
  %178 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %175, i64 %177
  %179 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %178, align 8
  call void @mpc_optimise_unretained(%struct.mpc_parser_t* %179, i32 0)
  br label %180

; <label>:180:                                    ; preds = %170
  %181 = load i32, i32* %5, align 4
  %182 = add nsw i32 %181, 1
  store i32 %182, i32* %5, align 4
  br label %162

; <label>:183:                                    ; preds = %162
  br label %184

; <label>:184:                                    ; preds = %183, %155
  br label %185

; <label>:185:                                    ; preds = %184, %228, %338, %470, %674, %872, %946, %1150, %1348
  %186 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %187 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %186, i32 0, i32 2
  %188 = load i8, i8* %187, align 8
  %189 = sext i8 %188 to i32
  %190 = icmp eq i32 %189, 23
  br i1 %190, label %191, label %309

; <label>:191:                                    ; preds = %185
  %192 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %193 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %192, i32 0, i32 3
  %194 = bitcast %union.mpc_pdata_t* %193 to %struct.mpc_pdata_or_t*
  %195 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %194, i32 0, i32 1
  %196 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %195, align 8
  %197 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %198 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %197, i32 0, i32 3
  %199 = bitcast %union.mpc_pdata_t* %198 to %struct.mpc_pdata_or_t*
  %200 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %199, i32 0, i32 0
  %201 = load i32, i32* %200, align 8
  %202 = sub nsw i32 %201, 1
  %203 = sext i32 %202 to i64
  %204 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %196, i64 %203
  %205 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %204, align 8
  %206 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %205, i32 0, i32 2
  %207 = load i8, i8* %206, align 8
  %208 = sext i8 %207 to i32
  %209 = icmp eq i32 %208, 23
  br i1 %209, label %210, label %309

; <label>:210:                                    ; preds = %191
  %211 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %212 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %211, i32 0, i32 3
  %213 = bitcast %union.mpc_pdata_t* %212 to %struct.mpc_pdata_or_t*
  %214 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %213, i32 0, i32 1
  %215 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %214, align 8
  %216 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %217 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %216, i32 0, i32 3
  %218 = bitcast %union.mpc_pdata_t* %217 to %struct.mpc_pdata_or_t*
  %219 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %218, i32 0, i32 0
  %220 = load i32, i32* %219, align 8
  %221 = sub nsw i32 %220, 1
  %222 = sext i32 %221 to i64
  %223 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %215, i64 %222
  %224 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %223, align 8
  %225 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %224, i32 0, i32 0
  %226 = load i8, i8* %225, align 8
  %227 = icmp ne i8 %226, 0
  br i1 %227, label %309, label %228

; <label>:228:                                    ; preds = %210
  %229 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %230 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %229, i32 0, i32 3
  %231 = bitcast %union.mpc_pdata_t* %230 to %struct.mpc_pdata_or_t*
  %232 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %231, i32 0, i32 1
  %233 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %232, align 8
  %234 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %235 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %234, i32 0, i32 3
  %236 = bitcast %union.mpc_pdata_t* %235 to %struct.mpc_pdata_or_t*
  %237 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %236, i32 0, i32 0
  %238 = load i32, i32* %237, align 8
  %239 = sub nsw i32 %238, 1
  %240 = sext i32 %239 to i64
  %241 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %233, i64 %240
  %242 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %241, align 8
  store %struct.mpc_parser_t* %242, %struct.mpc_parser_t** %8, align 8
  %243 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %244 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %243, i32 0, i32 3
  %245 = bitcast %union.mpc_pdata_t* %244 to %struct.mpc_pdata_or_t*
  %246 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %245, i32 0, i32 0
  %247 = load i32, i32* %246, align 8
  store i32 %247, i32* %6, align 4
  %248 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %249 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %248, i32 0, i32 3
  %250 = bitcast %union.mpc_pdata_t* %249 to %struct.mpc_pdata_or_t*
  %251 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %250, i32 0, i32 0
  %252 = load i32, i32* %251, align 8
  store i32 %252, i32* %7, align 4
  %253 = load i32, i32* %6, align 4
  %254 = load i32, i32* %7, align 4
  %255 = add nsw i32 %253, %254
  %256 = sub nsw i32 %255, 1
  %257 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %258 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %257, i32 0, i32 3
  %259 = bitcast %union.mpc_pdata_t* %258 to %struct.mpc_pdata_or_t*
  %260 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %259, i32 0, i32 0
  store i32 %256, i32* %260, align 8
  %261 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %262 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %261, i32 0, i32 3
  %263 = bitcast %union.mpc_pdata_t* %262 to %struct.mpc_pdata_or_t*
  %264 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %263, i32 0, i32 1
  %265 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %264, align 8
  %266 = bitcast %struct.mpc_parser_t** %265 to i8*
  %267 = load i32, i32* %6, align 4
  %268 = load i32, i32* %7, align 4
  %269 = add nsw i32 %267, %268
  %270 = sub nsw i32 %269, 1
  %271 = sext i32 %270 to i64
  %272 = mul i64 8, %271
  %273 = call i8* @realloc(i8* %266, i64 %272) #5
  %274 = bitcast i8* %273 to %struct.mpc_parser_t**
  %275 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %276 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %275, i32 0, i32 3
  %277 = bitcast %union.mpc_pdata_t* %276 to %struct.mpc_pdata_or_t*
  %278 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %277, i32 0, i32 1
  store %struct.mpc_parser_t** %274, %struct.mpc_parser_t*** %278, align 8
  %279 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %280 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %279, i32 0, i32 3
  %281 = bitcast %union.mpc_pdata_t* %280 to %struct.mpc_pdata_or_t*
  %282 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %281, i32 0, i32 1
  %283 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %282, align 8
  %284 = load i32, i32* %6, align 4
  %285 = sext i32 %284 to i64
  %286 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %283, i64 %285
  %287 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %286, i64 -1
  %288 = bitcast %struct.mpc_parser_t** %287 to i8*
  %289 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %290 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %289, i32 0, i32 3
  %291 = bitcast %union.mpc_pdata_t* %290 to %struct.mpc_pdata_or_t*
  %292 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %291, i32 0, i32 1
  %293 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %292, align 8
  %294 = bitcast %struct.mpc_parser_t** %293 to i8*
  %295 = load i32, i32* %7, align 4
  %296 = sext i32 %295 to i64
  %297 = mul i64 %296, 8
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %288, i8* %294, i64 %297, i32 8, i1 false)
  %298 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %299 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %298, i32 0, i32 3
  %300 = bitcast %union.mpc_pdata_t* %299 to %struct.mpc_pdata_or_t*
  %301 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %300, i32 0, i32 1
  %302 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %301, align 8
  %303 = bitcast %struct.mpc_parser_t** %302 to i8*
  call void @free(i8* %303) #5
  %304 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %305 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %304, i32 0, i32 1
  %306 = load i8*, i8** %305, align 8
  call void @free(i8* %306) #5
  %307 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %308 = bitcast %struct.mpc_parser_t* %307 to i8*
  call void @free(i8* %308) #5
  br label %185

; <label>:309:                                    ; preds = %210, %191, %185
  %310 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %311 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %310, i32 0, i32 2
  %312 = load i8, i8* %311, align 8
  %313 = sext i8 %312 to i32
  %314 = icmp eq i32 %313, 23
  br i1 %314, label %315, label %427

; <label>:315:                                    ; preds = %309
  %316 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %317 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %316, i32 0, i32 3
  %318 = bitcast %union.mpc_pdata_t* %317 to %struct.mpc_pdata_or_t*
  %319 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %318, i32 0, i32 1
  %320 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %319, align 8
  %321 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %320, i64 0
  %322 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %321, align 8
  %323 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %322, i32 0, i32 2
  %324 = load i8, i8* %323, align 8
  %325 = sext i8 %324 to i32
  %326 = icmp eq i32 %325, 23
  br i1 %326, label %327, label %427

; <label>:327:                                    ; preds = %315
  %328 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %329 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %328, i32 0, i32 3
  %330 = bitcast %union.mpc_pdata_t* %329 to %struct.mpc_pdata_or_t*
  %331 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %330, i32 0, i32 1
  %332 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %331, align 8
  %333 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %332, i64 0
  %334 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %333, align 8
  %335 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %334, i32 0, i32 0
  %336 = load i8, i8* %335, align 8
  %337 = icmp ne i8 %336, 0
  br i1 %337, label %427, label %338

; <label>:338:                                    ; preds = %327
  %339 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %340 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %339, i32 0, i32 3
  %341 = bitcast %union.mpc_pdata_t* %340 to %struct.mpc_pdata_or_t*
  %342 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %341, i32 0, i32 1
  %343 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %342, align 8
  %344 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %343, i64 0
  %345 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %344, align 8
  store %struct.mpc_parser_t* %345, %struct.mpc_parser_t** %8, align 8
  %346 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %347 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %346, i32 0, i32 3
  %348 = bitcast %union.mpc_pdata_t* %347 to %struct.mpc_pdata_or_t*
  %349 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %348, i32 0, i32 0
  %350 = load i32, i32* %349, align 8
  store i32 %350, i32* %6, align 4
  %351 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %352 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %351, i32 0, i32 3
  %353 = bitcast %union.mpc_pdata_t* %352 to %struct.mpc_pdata_or_t*
  %354 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %353, i32 0, i32 0
  %355 = load i32, i32* %354, align 8
  store i32 %355, i32* %7, align 4
  %356 = load i32, i32* %6, align 4
  %357 = load i32, i32* %7, align 4
  %358 = add nsw i32 %356, %357
  %359 = sub nsw i32 %358, 1
  %360 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %361 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %360, i32 0, i32 3
  %362 = bitcast %union.mpc_pdata_t* %361 to %struct.mpc_pdata_or_t*
  %363 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %362, i32 0, i32 0
  store i32 %359, i32* %363, align 8
  %364 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %365 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %364, i32 0, i32 3
  %366 = bitcast %union.mpc_pdata_t* %365 to %struct.mpc_pdata_or_t*
  %367 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %366, i32 0, i32 1
  %368 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %367, align 8
  %369 = bitcast %struct.mpc_parser_t** %368 to i8*
  %370 = load i32, i32* %6, align 4
  %371 = load i32, i32* %7, align 4
  %372 = add nsw i32 %370, %371
  %373 = sub nsw i32 %372, 1
  %374 = sext i32 %373 to i64
  %375 = mul i64 8, %374
  %376 = call i8* @realloc(i8* %369, i64 %375) #5
  %377 = bitcast i8* %376 to %struct.mpc_parser_t**
  %378 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %379 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %378, i32 0, i32 3
  %380 = bitcast %union.mpc_pdata_t* %379 to %struct.mpc_pdata_or_t*
  %381 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %380, i32 0, i32 1
  store %struct.mpc_parser_t** %377, %struct.mpc_parser_t*** %381, align 8
  %382 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %383 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %382, i32 0, i32 3
  %384 = bitcast %union.mpc_pdata_t* %383 to %struct.mpc_pdata_or_t*
  %385 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %384, i32 0, i32 1
  %386 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %385, align 8
  %387 = load i32, i32* %7, align 4
  %388 = sext i32 %387 to i64
  %389 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %386, i64 %388
  %390 = bitcast %struct.mpc_parser_t** %389 to i8*
  %391 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %392 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %391, i32 0, i32 3
  %393 = bitcast %union.mpc_pdata_t* %392 to %struct.mpc_pdata_or_t*
  %394 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %393, i32 0, i32 1
  %395 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %394, align 8
  %396 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %395, i64 1
  %397 = bitcast %struct.mpc_parser_t** %396 to i8*
  %398 = load i32, i32* %6, align 4
  %399 = sext i32 %398 to i64
  %400 = mul i64 %399, 8
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %390, i8* %397, i64 %400, i32 8, i1 false)
  %401 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %402 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %401, i32 0, i32 3
  %403 = bitcast %union.mpc_pdata_t* %402 to %struct.mpc_pdata_or_t*
  %404 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %403, i32 0, i32 1
  %405 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %404, align 8
  %406 = bitcast %struct.mpc_parser_t** %405 to i8*
  %407 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %408 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %407, i32 0, i32 3
  %409 = bitcast %union.mpc_pdata_t* %408 to %struct.mpc_pdata_or_t*
  %410 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %409, i32 0, i32 1
  %411 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %410, align 8
  %412 = bitcast %struct.mpc_parser_t** %411 to i8*
  %413 = load i32, i32* %7, align 4
  %414 = sext i32 %413 to i64
  %415 = mul i64 %414, 8
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %406, i8* %412, i64 %415, i32 8, i1 false)
  %416 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %417 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %416, i32 0, i32 3
  %418 = bitcast %union.mpc_pdata_t* %417 to %struct.mpc_pdata_or_t*
  %419 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %418, i32 0, i32 1
  %420 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %419, align 8
  %421 = bitcast %struct.mpc_parser_t** %420 to i8*
  call void @free(i8* %421) #5
  %422 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %423 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %422, i32 0, i32 1
  %424 = load i8*, i8** %423, align 8
  call void @free(i8* %424) #5
  %425 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %426 = bitcast %struct.mpc_parser_t* %425 to i8*
  call void @free(i8* %426) #5
  br label %185

; <label>:427:                                    ; preds = %327, %315, %309
  %428 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %429 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %428, i32 0, i32 2
  %430 = load i8, i8* %429, align 8
  %431 = sext i8 %430 to i32
  %432 = icmp eq i32 %431, 24
  br i1 %432, label %433, label %506

; <label>:433:                                    ; preds = %427
  %434 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %435 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %434, i32 0, i32 3
  %436 = bitcast %union.mpc_pdata_t* %435 to %struct.mpc_pdata_and_t*
  %437 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %436, i32 0, i32 0
  %438 = load i32, i32* %437, align 8
  %439 = icmp eq i32 %438, 2
  br i1 %439, label %440, label %506

; <label>:440:                                    ; preds = %433
  %441 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %442 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %441, i32 0, i32 3
  %443 = bitcast %union.mpc_pdata_t* %442 to %struct.mpc_pdata_and_t*
  %444 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %443, i32 0, i32 2
  %445 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %444, align 8
  %446 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %445, i64 0
  %447 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %446, align 8
  %448 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %447, i32 0, i32 2
  %449 = load i8, i8* %448, align 8
  %450 = sext i8 %449 to i32
  %451 = icmp eq i32 %450, 1
  br i1 %451, label %452, label %506

; <label>:452:                                    ; preds = %440
  %453 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %454 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %453, i32 0, i32 3
  %455 = bitcast %union.mpc_pdata_t* %454 to %struct.mpc_pdata_and_t*
  %456 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %455, i32 0, i32 2
  %457 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %456, align 8
  %458 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %457, i64 0
  %459 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %458, align 8
  %460 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %459, i32 0, i32 0
  %461 = load i8, i8* %460, align 8
  %462 = icmp ne i8 %461, 0
  br i1 %462, label %506, label %463

; <label>:463:                                    ; preds = %452
  %464 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %465 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %464, i32 0, i32 3
  %466 = bitcast %union.mpc_pdata_t* %465 to %struct.mpc_pdata_and_t*
  %467 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %466, i32 0, i32 1
  %468 = load i8* (i32, i8**)*, i8* (i32, i8**)** %467, align 8
  %469 = icmp eq i8* (i32, i8**)* %468, @mpcf_fold_ast
  br i1 %469, label %470, label %506

; <label>:470:                                    ; preds = %463
  %471 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %472 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %471, i32 0, i32 3
  %473 = bitcast %union.mpc_pdata_t* %472 to %struct.mpc_pdata_and_t*
  %474 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %473, i32 0, i32 2
  %475 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %474, align 8
  %476 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %475, i64 1
  %477 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %476, align 8
  store %struct.mpc_parser_t* %477, %struct.mpc_parser_t** %8, align 8
  %478 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %479 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %478, i32 0, i32 3
  %480 = bitcast %union.mpc_pdata_t* %479 to %struct.mpc_pdata_and_t*
  %481 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %480, i32 0, i32 2
  %482 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %481, align 8
  %483 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %482, i64 0
  %484 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %483, align 8
  call void @mpc_delete(%struct.mpc_parser_t* %484)
  %485 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %486 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %485, i32 0, i32 3
  %487 = bitcast %union.mpc_pdata_t* %486 to %struct.mpc_pdata_and_t*
  %488 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %487, i32 0, i32 2
  %489 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %488, align 8
  %490 = bitcast %struct.mpc_parser_t** %489 to i8*
  call void @free(i8* %490) #5
  %491 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %492 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %491, i32 0, i32 3
  %493 = bitcast %union.mpc_pdata_t* %492 to %struct.mpc_pdata_and_t*
  %494 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %493, i32 0, i32 3
  %495 = load void (i8*)**, void (i8*)*** %494, align 8
  %496 = bitcast void (i8*)** %495 to i8*
  call void @free(i8* %496) #5
  %497 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %498 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %497, i32 0, i32 1
  %499 = load i8*, i8** %498, align 8
  call void @free(i8* %499) #5
  %500 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %501 = bitcast %struct.mpc_parser_t* %500 to i8*
  %502 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %503 = bitcast %struct.mpc_parser_t* %502 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %501, i8* %503, i64 56, i32 8, i1 false)
  %504 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %505 = bitcast %struct.mpc_parser_t* %504 to i8*
  call void @free(i8* %505) #5
  br label %185

; <label>:506:                                    ; preds = %463, %452, %440, %433, %427
  %507 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %508 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %507, i32 0, i32 2
  %509 = load i8, i8* %508, align 8
  %510 = sext i8 %509 to i32
  %511 = icmp eq i32 %510, 24
  br i1 %511, label %512, label %692

; <label>:512:                                    ; preds = %506
  %513 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %514 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %513, i32 0, i32 3
  %515 = bitcast %union.mpc_pdata_t* %514 to %struct.mpc_pdata_and_t*
  %516 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %515, i32 0, i32 1
  %517 = load i8* (i32, i8**)*, i8* (i32, i8**)** %516, align 8
  %518 = icmp eq i8* (i32, i8**)* %517, @mpcf_fold_ast
  br i1 %518, label %519, label %692

; <label>:519:                                    ; preds = %512
  %520 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %521 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %520, i32 0, i32 3
  %522 = bitcast %union.mpc_pdata_t* %521 to %struct.mpc_pdata_and_t*
  %523 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %522, i32 0, i32 2
  %524 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %523, align 8
  %525 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %524, i64 0
  %526 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %525, align 8
  %527 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %526, i32 0, i32 2
  %528 = load i8, i8* %527, align 8
  %529 = sext i8 %528 to i32
  %530 = icmp eq i32 %529, 24
  br i1 %530, label %531, label %692

; <label>:531:                                    ; preds = %519
  %532 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %533 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %532, i32 0, i32 3
  %534 = bitcast %union.mpc_pdata_t* %533 to %struct.mpc_pdata_and_t*
  %535 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %534, i32 0, i32 2
  %536 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %535, align 8
  %537 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %536, i64 0
  %538 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %537, align 8
  %539 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %538, i32 0, i32 0
  %540 = load i8, i8* %539, align 8
  %541 = icmp ne i8 %540, 0
  br i1 %541, label %692, label %542

; <label>:542:                                    ; preds = %531
  %543 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %544 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %543, i32 0, i32 3
  %545 = bitcast %union.mpc_pdata_t* %544 to %struct.mpc_pdata_and_t*
  %546 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %545, i32 0, i32 2
  %547 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %546, align 8
  %548 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %547, i64 0
  %549 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %548, align 8
  %550 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %549, i32 0, i32 3
  %551 = bitcast %union.mpc_pdata_t* %550 to %struct.mpc_pdata_and_t*
  %552 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %551, i32 0, i32 1
  %553 = load i8* (i32, i8**)*, i8* (i32, i8**)** %552, align 8
  %554 = icmp eq i8* (i32, i8**)* %553, @mpcf_fold_ast
  br i1 %554, label %555, label %692

; <label>:555:                                    ; preds = %542
  %556 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %557 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %556, i32 0, i32 3
  %558 = bitcast %union.mpc_pdata_t* %557 to %struct.mpc_pdata_and_t*
  %559 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %558, i32 0, i32 2
  %560 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %559, align 8
  %561 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %560, i64 0
  %562 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %561, align 8
  store %struct.mpc_parser_t* %562, %struct.mpc_parser_t** %8, align 8
  %563 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %564 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %563, i32 0, i32 3
  %565 = bitcast %union.mpc_pdata_t* %564 to %struct.mpc_pdata_and_t*
  %566 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %565, i32 0, i32 0
  %567 = load i32, i32* %566, align 8
  store i32 %567, i32* %6, align 4
  %568 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %569 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %568, i32 0, i32 3
  %570 = bitcast %union.mpc_pdata_t* %569 to %struct.mpc_pdata_and_t*
  %571 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %570, i32 0, i32 0
  %572 = load i32, i32* %571, align 8
  store i32 %572, i32* %7, align 4
  %573 = load i32, i32* %6, align 4
  %574 = load i32, i32* %7, align 4
  %575 = add nsw i32 %573, %574
  %576 = sub nsw i32 %575, 1
  %577 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %578 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %577, i32 0, i32 3
  %579 = bitcast %union.mpc_pdata_t* %578 to %struct.mpc_pdata_and_t*
  %580 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %579, i32 0, i32 0
  store i32 %576, i32* %580, align 8
  %581 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %582 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %581, i32 0, i32 3
  %583 = bitcast %union.mpc_pdata_t* %582 to %struct.mpc_pdata_and_t*
  %584 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %583, i32 0, i32 2
  %585 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %584, align 8
  %586 = bitcast %struct.mpc_parser_t** %585 to i8*
  %587 = load i32, i32* %6, align 4
  %588 = load i32, i32* %7, align 4
  %589 = add nsw i32 %587, %588
  %590 = sub nsw i32 %589, 1
  %591 = sext i32 %590 to i64
  %592 = mul i64 8, %591
  %593 = call i8* @realloc(i8* %586, i64 %592) #5
  %594 = bitcast i8* %593 to %struct.mpc_parser_t**
  %595 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %596 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %595, i32 0, i32 3
  %597 = bitcast %union.mpc_pdata_t* %596 to %struct.mpc_pdata_and_t*
  %598 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %597, i32 0, i32 2
  store %struct.mpc_parser_t** %594, %struct.mpc_parser_t*** %598, align 8
  %599 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %600 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %599, i32 0, i32 3
  %601 = bitcast %union.mpc_pdata_t* %600 to %struct.mpc_pdata_and_t*
  %602 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %601, i32 0, i32 3
  %603 = load void (i8*)**, void (i8*)*** %602, align 8
  %604 = bitcast void (i8*)** %603 to i8*
  %605 = load i32, i32* %6, align 4
  %606 = load i32, i32* %7, align 4
  %607 = add nsw i32 %605, %606
  %608 = sub nsw i32 %607, 1
  %609 = sub nsw i32 %608, 1
  %610 = sext i32 %609 to i64
  %611 = mul i64 8, %610
  %612 = call i8* @realloc(i8* %604, i64 %611) #5
  %613 = bitcast i8* %612 to void (i8*)**
  %614 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %615 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %614, i32 0, i32 3
  %616 = bitcast %union.mpc_pdata_t* %615 to %struct.mpc_pdata_and_t*
  %617 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %616, i32 0, i32 3
  store void (i8*)** %613, void (i8*)*** %617, align 8
  %618 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %619 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %618, i32 0, i32 3
  %620 = bitcast %union.mpc_pdata_t* %619 to %struct.mpc_pdata_and_t*
  %621 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %620, i32 0, i32 2
  %622 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %621, align 8
  %623 = load i32, i32* %7, align 4
  %624 = sext i32 %623 to i64
  %625 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %622, i64 %624
  %626 = bitcast %struct.mpc_parser_t** %625 to i8*
  %627 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %628 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %627, i32 0, i32 3
  %629 = bitcast %union.mpc_pdata_t* %628 to %struct.mpc_pdata_and_t*
  %630 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %629, i32 0, i32 2
  %631 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %630, align 8
  %632 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %631, i64 1
  %633 = bitcast %struct.mpc_parser_t** %632 to i8*
  %634 = load i32, i32* %6, align 4
  %635 = sub nsw i32 %634, 1
  %636 = sext i32 %635 to i64
  %637 = mul i64 %636, 8
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %626, i8* %633, i64 %637, i32 8, i1 false)
  %638 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %639 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %638, i32 0, i32 3
  %640 = bitcast %union.mpc_pdata_t* %639 to %struct.mpc_pdata_and_t*
  %641 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %640, i32 0, i32 2
  %642 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %641, align 8
  %643 = bitcast %struct.mpc_parser_t** %642 to i8*
  %644 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %645 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %644, i32 0, i32 3
  %646 = bitcast %union.mpc_pdata_t* %645 to %struct.mpc_pdata_and_t*
  %647 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %646, i32 0, i32 2
  %648 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %647, align 8
  %649 = bitcast %struct.mpc_parser_t** %648 to i8*
  %650 = load i32, i32* %7, align 4
  %651 = sext i32 %650 to i64
  %652 = mul i64 %651, 8
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %643, i8* %649, i64 %652, i32 8, i1 false)
  store i32 0, i32* %5, align 4
  br label %653

; <label>:653:                                    ; preds = %671, %555
  %654 = load i32, i32* %5, align 4
  %655 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %656 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %655, i32 0, i32 3
  %657 = bitcast %union.mpc_pdata_t* %656 to %struct.mpc_pdata_and_t*
  %658 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %657, i32 0, i32 0
  %659 = load i32, i32* %658, align 8
  %660 = sub nsw i32 %659, 1
  %661 = icmp slt i32 %654, %660
  br i1 %661, label %662, label %674

; <label>:662:                                    ; preds = %653
  %663 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %664 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %663, i32 0, i32 3
  %665 = bitcast %union.mpc_pdata_t* %664 to %struct.mpc_pdata_and_t*
  %666 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %665, i32 0, i32 3
  %667 = load void (i8*)**, void (i8*)*** %666, align 8
  %668 = load i32, i32* %5, align 4
  %669 = sext i32 %668 to i64
  %670 = getelementptr inbounds void (i8*)*, void (i8*)** %667, i64 %669
  store void (i8*)* bitcast (void (%struct.mpc_ast_t*)* @mpc_ast_delete to void (i8*)*), void (i8*)** %670, align 8
  br label %671

; <label>:671:                                    ; preds = %662
  %672 = load i32, i32* %5, align 4
  %673 = add nsw i32 %672, 1
  store i32 %673, i32* %5, align 4
  br label %653

; <label>:674:                                    ; preds = %653
  %675 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %676 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %675, i32 0, i32 3
  %677 = bitcast %union.mpc_pdata_t* %676 to %struct.mpc_pdata_and_t*
  %678 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %677, i32 0, i32 2
  %679 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %678, align 8
  %680 = bitcast %struct.mpc_parser_t** %679 to i8*
  call void @free(i8* %680) #5
  %681 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %682 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %681, i32 0, i32 3
  %683 = bitcast %union.mpc_pdata_t* %682 to %struct.mpc_pdata_and_t*
  %684 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %683, i32 0, i32 3
  %685 = load void (i8*)**, void (i8*)*** %684, align 8
  %686 = bitcast void (i8*)** %685 to i8*
  call void @free(i8* %686) #5
  %687 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %688 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %687, i32 0, i32 1
  %689 = load i8*, i8** %688, align 8
  call void @free(i8* %689) #5
  %690 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %691 = bitcast %struct.mpc_parser_t* %690 to i8*
  call void @free(i8* %691) #5
  br label %185

; <label>:692:                                    ; preds = %542, %531, %519, %512, %506
  %693 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %694 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %693, i32 0, i32 2
  %695 = load i8, i8* %694, align 8
  %696 = sext i8 %695 to i32
  %697 = icmp eq i32 %696, 24
  br i1 %697, label %698, label %890

; <label>:698:                                    ; preds = %692
  %699 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %700 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %699, i32 0, i32 3
  %701 = bitcast %union.mpc_pdata_t* %700 to %struct.mpc_pdata_and_t*
  %702 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %701, i32 0, i32 1
  %703 = load i8* (i32, i8**)*, i8* (i32, i8**)** %702, align 8
  %704 = icmp eq i8* (i32, i8**)* %703, @mpcf_fold_ast
  br i1 %704, label %705, label %890

; <label>:705:                                    ; preds = %698
  %706 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %707 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %706, i32 0, i32 3
  %708 = bitcast %union.mpc_pdata_t* %707 to %struct.mpc_pdata_and_t*
  %709 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %708, i32 0, i32 2
  %710 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %709, align 8
  %711 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %712 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %711, i32 0, i32 3
  %713 = bitcast %union.mpc_pdata_t* %712 to %struct.mpc_pdata_and_t*
  %714 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %713, i32 0, i32 0
  %715 = load i32, i32* %714, align 8
  %716 = sub nsw i32 %715, 1
  %717 = sext i32 %716 to i64
  %718 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %710, i64 %717
  %719 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %718, align 8
  %720 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %719, i32 0, i32 2
  %721 = load i8, i8* %720, align 8
  %722 = sext i8 %721 to i32
  %723 = icmp eq i32 %722, 24
  br i1 %723, label %724, label %890

; <label>:724:                                    ; preds = %705
  %725 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %726 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %725, i32 0, i32 3
  %727 = bitcast %union.mpc_pdata_t* %726 to %struct.mpc_pdata_and_t*
  %728 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %727, i32 0, i32 2
  %729 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %728, align 8
  %730 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %731 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %730, i32 0, i32 3
  %732 = bitcast %union.mpc_pdata_t* %731 to %struct.mpc_pdata_and_t*
  %733 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %732, i32 0, i32 0
  %734 = load i32, i32* %733, align 8
  %735 = sub nsw i32 %734, 1
  %736 = sext i32 %735 to i64
  %737 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %729, i64 %736
  %738 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %737, align 8
  %739 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %738, i32 0, i32 0
  %740 = load i8, i8* %739, align 8
  %741 = icmp ne i8 %740, 0
  br i1 %741, label %890, label %742

; <label>:742:                                    ; preds = %724
  %743 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %744 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %743, i32 0, i32 3
  %745 = bitcast %union.mpc_pdata_t* %744 to %struct.mpc_pdata_and_t*
  %746 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %745, i32 0, i32 2
  %747 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %746, align 8
  %748 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %749 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %748, i32 0, i32 3
  %750 = bitcast %union.mpc_pdata_t* %749 to %struct.mpc_pdata_and_t*
  %751 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %750, i32 0, i32 0
  %752 = load i32, i32* %751, align 8
  %753 = sub nsw i32 %752, 1
  %754 = sext i32 %753 to i64
  %755 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %747, i64 %754
  %756 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %755, align 8
  %757 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %756, i32 0, i32 3
  %758 = bitcast %union.mpc_pdata_t* %757 to %struct.mpc_pdata_and_t*
  %759 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %758, i32 0, i32 1
  %760 = load i8* (i32, i8**)*, i8* (i32, i8**)** %759, align 8
  %761 = icmp eq i8* (i32, i8**)* %760, @mpcf_fold_ast
  br i1 %761, label %762, label %890

; <label>:762:                                    ; preds = %742
  %763 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %764 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %763, i32 0, i32 3
  %765 = bitcast %union.mpc_pdata_t* %764 to %struct.mpc_pdata_and_t*
  %766 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %765, i32 0, i32 2
  %767 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %766, align 8
  %768 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %769 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %768, i32 0, i32 3
  %770 = bitcast %union.mpc_pdata_t* %769 to %struct.mpc_pdata_and_t*
  %771 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %770, i32 0, i32 0
  %772 = load i32, i32* %771, align 8
  %773 = sub nsw i32 %772, 1
  %774 = sext i32 %773 to i64
  %775 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %767, i64 %774
  %776 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %775, align 8
  store %struct.mpc_parser_t* %776, %struct.mpc_parser_t** %8, align 8
  %777 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %778 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %777, i32 0, i32 3
  %779 = bitcast %union.mpc_pdata_t* %778 to %struct.mpc_pdata_and_t*
  %780 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %779, i32 0, i32 0
  %781 = load i32, i32* %780, align 8
  store i32 %781, i32* %6, align 4
  %782 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %783 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %782, i32 0, i32 3
  %784 = bitcast %union.mpc_pdata_t* %783 to %struct.mpc_pdata_and_t*
  %785 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %784, i32 0, i32 0
  %786 = load i32, i32* %785, align 8
  store i32 %786, i32* %7, align 4
  %787 = load i32, i32* %6, align 4
  %788 = load i32, i32* %7, align 4
  %789 = add nsw i32 %787, %788
  %790 = sub nsw i32 %789, 1
  %791 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %792 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %791, i32 0, i32 3
  %793 = bitcast %union.mpc_pdata_t* %792 to %struct.mpc_pdata_and_t*
  %794 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %793, i32 0, i32 0
  store i32 %790, i32* %794, align 8
  %795 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %796 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %795, i32 0, i32 3
  %797 = bitcast %union.mpc_pdata_t* %796 to %struct.mpc_pdata_and_t*
  %798 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %797, i32 0, i32 2
  %799 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %798, align 8
  %800 = bitcast %struct.mpc_parser_t** %799 to i8*
  %801 = load i32, i32* %6, align 4
  %802 = load i32, i32* %7, align 4
  %803 = add nsw i32 %801, %802
  %804 = sub nsw i32 %803, 1
  %805 = sext i32 %804 to i64
  %806 = mul i64 8, %805
  %807 = call i8* @realloc(i8* %800, i64 %806) #5
  %808 = bitcast i8* %807 to %struct.mpc_parser_t**
  %809 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %810 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %809, i32 0, i32 3
  %811 = bitcast %union.mpc_pdata_t* %810 to %struct.mpc_pdata_and_t*
  %812 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %811, i32 0, i32 2
  store %struct.mpc_parser_t** %808, %struct.mpc_parser_t*** %812, align 8
  %813 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %814 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %813, i32 0, i32 3
  %815 = bitcast %union.mpc_pdata_t* %814 to %struct.mpc_pdata_and_t*
  %816 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %815, i32 0, i32 3
  %817 = load void (i8*)**, void (i8*)*** %816, align 8
  %818 = bitcast void (i8*)** %817 to i8*
  %819 = load i32, i32* %6, align 4
  %820 = load i32, i32* %7, align 4
  %821 = add nsw i32 %819, %820
  %822 = sub nsw i32 %821, 1
  %823 = sub nsw i32 %822, 1
  %824 = sext i32 %823 to i64
  %825 = mul i64 8, %824
  %826 = call i8* @realloc(i8* %818, i64 %825) #5
  %827 = bitcast i8* %826 to void (i8*)**
  %828 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %829 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %828, i32 0, i32 3
  %830 = bitcast %union.mpc_pdata_t* %829 to %struct.mpc_pdata_and_t*
  %831 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %830, i32 0, i32 3
  store void (i8*)** %827, void (i8*)*** %831, align 8
  %832 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %833 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %832, i32 0, i32 3
  %834 = bitcast %union.mpc_pdata_t* %833 to %struct.mpc_pdata_and_t*
  %835 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %834, i32 0, i32 2
  %836 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %835, align 8
  %837 = load i32, i32* %6, align 4
  %838 = sext i32 %837 to i64
  %839 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %836, i64 %838
  %840 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %839, i64 -1
  %841 = bitcast %struct.mpc_parser_t** %840 to i8*
  %842 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %843 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %842, i32 0, i32 3
  %844 = bitcast %union.mpc_pdata_t* %843 to %struct.mpc_pdata_and_t*
  %845 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %844, i32 0, i32 2
  %846 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %845, align 8
  %847 = bitcast %struct.mpc_parser_t** %846 to i8*
  %848 = load i32, i32* %7, align 4
  %849 = sext i32 %848 to i64
  %850 = mul i64 %849, 8
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %841, i8* %847, i64 %850, i32 8, i1 false)
  store i32 0, i32* %5, align 4
  br label %851

; <label>:851:                                    ; preds = %869, %762
  %852 = load i32, i32* %5, align 4
  %853 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %854 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %853, i32 0, i32 3
  %855 = bitcast %union.mpc_pdata_t* %854 to %struct.mpc_pdata_and_t*
  %856 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %855, i32 0, i32 0
  %857 = load i32, i32* %856, align 8
  %858 = sub nsw i32 %857, 1
  %859 = icmp slt i32 %852, %858
  br i1 %859, label %860, label %872

; <label>:860:                                    ; preds = %851
  %861 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %862 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %861, i32 0, i32 3
  %863 = bitcast %union.mpc_pdata_t* %862 to %struct.mpc_pdata_and_t*
  %864 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %863, i32 0, i32 3
  %865 = load void (i8*)**, void (i8*)*** %864, align 8
  %866 = load i32, i32* %5, align 4
  %867 = sext i32 %866 to i64
  %868 = getelementptr inbounds void (i8*)*, void (i8*)** %865, i64 %867
  store void (i8*)* bitcast (void (%struct.mpc_ast_t*)* @mpc_ast_delete to void (i8*)*), void (i8*)** %868, align 8
  br label %869

; <label>:869:                                    ; preds = %860
  %870 = load i32, i32* %5, align 4
  %871 = add nsw i32 %870, 1
  store i32 %871, i32* %5, align 4
  br label %851

; <label>:872:                                    ; preds = %851
  %873 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %874 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %873, i32 0, i32 3
  %875 = bitcast %union.mpc_pdata_t* %874 to %struct.mpc_pdata_and_t*
  %876 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %875, i32 0, i32 2
  %877 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %876, align 8
  %878 = bitcast %struct.mpc_parser_t** %877 to i8*
  call void @free(i8* %878) #5
  %879 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %880 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %879, i32 0, i32 3
  %881 = bitcast %union.mpc_pdata_t* %880 to %struct.mpc_pdata_and_t*
  %882 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %881, i32 0, i32 3
  %883 = load void (i8*)**, void (i8*)*** %882, align 8
  %884 = bitcast void (i8*)** %883 to i8*
  call void @free(i8* %884) #5
  %885 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %886 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %885, i32 0, i32 1
  %887 = load i8*, i8** %886, align 8
  call void @free(i8* %887) #5
  %888 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %889 = bitcast %struct.mpc_parser_t* %888 to i8*
  call void @free(i8* %889) #5
  br label %185

; <label>:890:                                    ; preds = %742, %724, %705, %698, %692
  %891 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %892 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %891, i32 0, i32 2
  %893 = load i8, i8* %892, align 8
  %894 = sext i8 %893 to i32
  %895 = icmp eq i32 %894, 24
  br i1 %895, label %896, label %982

; <label>:896:                                    ; preds = %890
  %897 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %898 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %897, i32 0, i32 3
  %899 = bitcast %union.mpc_pdata_t* %898 to %struct.mpc_pdata_and_t*
  %900 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %899, i32 0, i32 0
  %901 = load i32, i32* %900, align 8
  %902 = icmp eq i32 %901, 2
  br i1 %902, label %903, label %982

; <label>:903:                                    ; preds = %896
  %904 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %905 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %904, i32 0, i32 3
  %906 = bitcast %union.mpc_pdata_t* %905 to %struct.mpc_pdata_and_t*
  %907 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %906, i32 0, i32 2
  %908 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %907, align 8
  %909 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %908, i64 0
  %910 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %909, align 8
  %911 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %910, i32 0, i32 2
  %912 = load i8, i8* %911, align 8
  %913 = sext i8 %912 to i32
  %914 = icmp eq i32 %913, 3
  br i1 %914, label %915, label %982

; <label>:915:                                    ; preds = %903
  %916 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %917 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %916, i32 0, i32 3
  %918 = bitcast %union.mpc_pdata_t* %917 to %struct.mpc_pdata_and_t*
  %919 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %918, i32 0, i32 2
  %920 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %919, align 8
  %921 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %920, i64 0
  %922 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %921, align 8
  %923 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %922, i32 0, i32 3
  %924 = bitcast %union.mpc_pdata_t* %923 to %struct.mpc_pdata_lift_t*
  %925 = getelementptr inbounds %struct.mpc_pdata_lift_t, %struct.mpc_pdata_lift_t* %924, i32 0, i32 0
  %926 = load i8* ()*, i8* ()** %925, align 8
  %927 = icmp eq i8* ()* %926, @mpcf_ctor_str
  br i1 %927, label %928, label %982

; <label>:928:                                    ; preds = %915
  %929 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %930 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %929, i32 0, i32 3
  %931 = bitcast %union.mpc_pdata_t* %930 to %struct.mpc_pdata_and_t*
  %932 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %931, i32 0, i32 2
  %933 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %932, align 8
  %934 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %933, i64 0
  %935 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %934, align 8
  %936 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %935, i32 0, i32 0
  %937 = load i8, i8* %936, align 8
  %938 = icmp ne i8 %937, 0
  br i1 %938, label %982, label %939

; <label>:939:                                    ; preds = %928
  %940 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %941 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %940, i32 0, i32 3
  %942 = bitcast %union.mpc_pdata_t* %941 to %struct.mpc_pdata_and_t*
  %943 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %942, i32 0, i32 1
  %944 = load i8* (i32, i8**)*, i8* (i32, i8**)** %943, align 8
  %945 = icmp eq i8* (i32, i8**)* %944, @mpcf_strfold
  br i1 %945, label %946, label %982

; <label>:946:                                    ; preds = %939
  %947 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %948 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %947, i32 0, i32 3
  %949 = bitcast %union.mpc_pdata_t* %948 to %struct.mpc_pdata_and_t*
  %950 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %949, i32 0, i32 2
  %951 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %950, align 8
  %952 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %951, i64 1
  %953 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %952, align 8
  store %struct.mpc_parser_t* %953, %struct.mpc_parser_t** %8, align 8
  %954 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %955 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %954, i32 0, i32 3
  %956 = bitcast %union.mpc_pdata_t* %955 to %struct.mpc_pdata_and_t*
  %957 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %956, i32 0, i32 2
  %958 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %957, align 8
  %959 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %958, i64 0
  %960 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %959, align 8
  call void @mpc_delete(%struct.mpc_parser_t* %960)
  %961 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %962 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %961, i32 0, i32 3
  %963 = bitcast %union.mpc_pdata_t* %962 to %struct.mpc_pdata_and_t*
  %964 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %963, i32 0, i32 2
  %965 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %964, align 8
  %966 = bitcast %struct.mpc_parser_t** %965 to i8*
  call void @free(i8* %966) #5
  %967 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %968 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %967, i32 0, i32 3
  %969 = bitcast %union.mpc_pdata_t* %968 to %struct.mpc_pdata_and_t*
  %970 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %969, i32 0, i32 3
  %971 = load void (i8*)**, void (i8*)*** %970, align 8
  %972 = bitcast void (i8*)** %971 to i8*
  call void @free(i8* %972) #5
  %973 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %974 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %973, i32 0, i32 1
  %975 = load i8*, i8** %974, align 8
  call void @free(i8* %975) #5
  %976 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %977 = bitcast %struct.mpc_parser_t* %976 to i8*
  %978 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %979 = bitcast %struct.mpc_parser_t* %978 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %977, i8* %979, i64 56, i32 8, i1 false)
  %980 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %981 = bitcast %struct.mpc_parser_t* %980 to i8*
  call void @free(i8* %981) #5
  br label %185

; <label>:982:                                    ; preds = %939, %928, %915, %903, %896, %890
  %983 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %984 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %983, i32 0, i32 2
  %985 = load i8, i8* %984, align 8
  %986 = sext i8 %985 to i32
  %987 = icmp eq i32 %986, 24
  br i1 %987, label %988, label %1168

; <label>:988:                                    ; preds = %982
  %989 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %990 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %989, i32 0, i32 3
  %991 = bitcast %union.mpc_pdata_t* %990 to %struct.mpc_pdata_and_t*
  %992 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %991, i32 0, i32 1
  %993 = load i8* (i32, i8**)*, i8* (i32, i8**)** %992, align 8
  %994 = icmp eq i8* (i32, i8**)* %993, @mpcf_strfold
  br i1 %994, label %995, label %1168

; <label>:995:                                    ; preds = %988
  %996 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %997 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %996, i32 0, i32 3
  %998 = bitcast %union.mpc_pdata_t* %997 to %struct.mpc_pdata_and_t*
  %999 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %998, i32 0, i32 2
  %1000 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %999, align 8
  %1001 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %1000, i64 0
  %1002 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1001, align 8
  %1003 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1002, i32 0, i32 2
  %1004 = load i8, i8* %1003, align 8
  %1005 = sext i8 %1004 to i32
  %1006 = icmp eq i32 %1005, 24
  br i1 %1006, label %1007, label %1168

; <label>:1007:                                   ; preds = %995
  %1008 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1009 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1008, i32 0, i32 3
  %1010 = bitcast %union.mpc_pdata_t* %1009 to %struct.mpc_pdata_and_t*
  %1011 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1010, i32 0, i32 2
  %1012 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %1011, align 8
  %1013 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %1012, i64 0
  %1014 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1013, align 8
  %1015 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1014, i32 0, i32 0
  %1016 = load i8, i8* %1015, align 8
  %1017 = icmp ne i8 %1016, 0
  br i1 %1017, label %1168, label %1018

; <label>:1018:                                   ; preds = %1007
  %1019 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1020 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1019, i32 0, i32 3
  %1021 = bitcast %union.mpc_pdata_t* %1020 to %struct.mpc_pdata_and_t*
  %1022 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1021, i32 0, i32 2
  %1023 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %1022, align 8
  %1024 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %1023, i64 0
  %1025 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1024, align 8
  %1026 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1025, i32 0, i32 3
  %1027 = bitcast %union.mpc_pdata_t* %1026 to %struct.mpc_pdata_and_t*
  %1028 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1027, i32 0, i32 1
  %1029 = load i8* (i32, i8**)*, i8* (i32, i8**)** %1028, align 8
  %1030 = icmp eq i8* (i32, i8**)* %1029, @mpcf_strfold
  br i1 %1030, label %1031, label %1168

; <label>:1031:                                   ; preds = %1018
  %1032 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1033 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1032, i32 0, i32 3
  %1034 = bitcast %union.mpc_pdata_t* %1033 to %struct.mpc_pdata_and_t*
  %1035 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1034, i32 0, i32 2
  %1036 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %1035, align 8
  %1037 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %1036, i64 0
  %1038 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1037, align 8
  store %struct.mpc_parser_t* %1038, %struct.mpc_parser_t** %8, align 8
  %1039 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1040 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1039, i32 0, i32 3
  %1041 = bitcast %union.mpc_pdata_t* %1040 to %struct.mpc_pdata_and_t*
  %1042 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1041, i32 0, i32 0
  %1043 = load i32, i32* %1042, align 8
  store i32 %1043, i32* %6, align 4
  %1044 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %1045 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1044, i32 0, i32 3
  %1046 = bitcast %union.mpc_pdata_t* %1045 to %struct.mpc_pdata_and_t*
  %1047 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1046, i32 0, i32 0
  %1048 = load i32, i32* %1047, align 8
  store i32 %1048, i32* %7, align 4
  %1049 = load i32, i32* %6, align 4
  %1050 = load i32, i32* %7, align 4
  %1051 = add nsw i32 %1049, %1050
  %1052 = sub nsw i32 %1051, 1
  %1053 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1054 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1053, i32 0, i32 3
  %1055 = bitcast %union.mpc_pdata_t* %1054 to %struct.mpc_pdata_and_t*
  %1056 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1055, i32 0, i32 0
  store i32 %1052, i32* %1056, align 8
  %1057 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1058 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1057, i32 0, i32 3
  %1059 = bitcast %union.mpc_pdata_t* %1058 to %struct.mpc_pdata_and_t*
  %1060 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1059, i32 0, i32 2
  %1061 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %1060, align 8
  %1062 = bitcast %struct.mpc_parser_t** %1061 to i8*
  %1063 = load i32, i32* %6, align 4
  %1064 = load i32, i32* %7, align 4
  %1065 = add nsw i32 %1063, %1064
  %1066 = sub nsw i32 %1065, 1
  %1067 = sext i32 %1066 to i64
  %1068 = mul i64 8, %1067
  %1069 = call i8* @realloc(i8* %1062, i64 %1068) #5
  %1070 = bitcast i8* %1069 to %struct.mpc_parser_t**
  %1071 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1072 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1071, i32 0, i32 3
  %1073 = bitcast %union.mpc_pdata_t* %1072 to %struct.mpc_pdata_and_t*
  %1074 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1073, i32 0, i32 2
  store %struct.mpc_parser_t** %1070, %struct.mpc_parser_t*** %1074, align 8
  %1075 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1076 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1075, i32 0, i32 3
  %1077 = bitcast %union.mpc_pdata_t* %1076 to %struct.mpc_pdata_and_t*
  %1078 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1077, i32 0, i32 3
  %1079 = load void (i8*)**, void (i8*)*** %1078, align 8
  %1080 = bitcast void (i8*)** %1079 to i8*
  %1081 = load i32, i32* %6, align 4
  %1082 = load i32, i32* %7, align 4
  %1083 = add nsw i32 %1081, %1082
  %1084 = sub nsw i32 %1083, 1
  %1085 = sub nsw i32 %1084, 1
  %1086 = sext i32 %1085 to i64
  %1087 = mul i64 8, %1086
  %1088 = call i8* @realloc(i8* %1080, i64 %1087) #5
  %1089 = bitcast i8* %1088 to void (i8*)**
  %1090 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1091 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1090, i32 0, i32 3
  %1092 = bitcast %union.mpc_pdata_t* %1091 to %struct.mpc_pdata_and_t*
  %1093 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1092, i32 0, i32 3
  store void (i8*)** %1089, void (i8*)*** %1093, align 8
  %1094 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1095 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1094, i32 0, i32 3
  %1096 = bitcast %union.mpc_pdata_t* %1095 to %struct.mpc_pdata_and_t*
  %1097 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1096, i32 0, i32 2
  %1098 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %1097, align 8
  %1099 = load i32, i32* %7, align 4
  %1100 = sext i32 %1099 to i64
  %1101 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %1098, i64 %1100
  %1102 = bitcast %struct.mpc_parser_t** %1101 to i8*
  %1103 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1104 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1103, i32 0, i32 3
  %1105 = bitcast %union.mpc_pdata_t* %1104 to %struct.mpc_pdata_and_t*
  %1106 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1105, i32 0, i32 2
  %1107 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %1106, align 8
  %1108 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %1107, i64 1
  %1109 = bitcast %struct.mpc_parser_t** %1108 to i8*
  %1110 = load i32, i32* %6, align 4
  %1111 = sub nsw i32 %1110, 1
  %1112 = sext i32 %1111 to i64
  %1113 = mul i64 %1112, 8
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %1102, i8* %1109, i64 %1113, i32 8, i1 false)
  %1114 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1115 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1114, i32 0, i32 3
  %1116 = bitcast %union.mpc_pdata_t* %1115 to %struct.mpc_pdata_and_t*
  %1117 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1116, i32 0, i32 2
  %1118 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %1117, align 8
  %1119 = bitcast %struct.mpc_parser_t** %1118 to i8*
  %1120 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %1121 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1120, i32 0, i32 3
  %1122 = bitcast %union.mpc_pdata_t* %1121 to %struct.mpc_pdata_and_t*
  %1123 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1122, i32 0, i32 2
  %1124 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %1123, align 8
  %1125 = bitcast %struct.mpc_parser_t** %1124 to i8*
  %1126 = load i32, i32* %7, align 4
  %1127 = sext i32 %1126 to i64
  %1128 = mul i64 %1127, 8
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %1119, i8* %1125, i64 %1128, i32 8, i1 false)
  store i32 0, i32* %5, align 4
  br label %1129

; <label>:1129:                                   ; preds = %1147, %1031
  %1130 = load i32, i32* %5, align 4
  %1131 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1132 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1131, i32 0, i32 3
  %1133 = bitcast %union.mpc_pdata_t* %1132 to %struct.mpc_pdata_and_t*
  %1134 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1133, i32 0, i32 0
  %1135 = load i32, i32* %1134, align 8
  %1136 = sub nsw i32 %1135, 1
  %1137 = icmp slt i32 %1130, %1136
  br i1 %1137, label %1138, label %1150

; <label>:1138:                                   ; preds = %1129
  %1139 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1140 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1139, i32 0, i32 3
  %1141 = bitcast %union.mpc_pdata_t* %1140 to %struct.mpc_pdata_and_t*
  %1142 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1141, i32 0, i32 3
  %1143 = load void (i8*)**, void (i8*)*** %1142, align 8
  %1144 = load i32, i32* %5, align 4
  %1145 = sext i32 %1144 to i64
  %1146 = getelementptr inbounds void (i8*)*, void (i8*)** %1143, i64 %1145
  store void (i8*)* @free, void (i8*)** %1146, align 8
  br label %1147

; <label>:1147:                                   ; preds = %1138
  %1148 = load i32, i32* %5, align 4
  %1149 = add nsw i32 %1148, 1
  store i32 %1149, i32* %5, align 4
  br label %1129

; <label>:1150:                                   ; preds = %1129
  %1151 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %1152 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1151, i32 0, i32 3
  %1153 = bitcast %union.mpc_pdata_t* %1152 to %struct.mpc_pdata_and_t*
  %1154 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1153, i32 0, i32 2
  %1155 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %1154, align 8
  %1156 = bitcast %struct.mpc_parser_t** %1155 to i8*
  call void @free(i8* %1156) #5
  %1157 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %1158 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1157, i32 0, i32 3
  %1159 = bitcast %union.mpc_pdata_t* %1158 to %struct.mpc_pdata_and_t*
  %1160 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1159, i32 0, i32 3
  %1161 = load void (i8*)**, void (i8*)*** %1160, align 8
  %1162 = bitcast void (i8*)** %1161 to i8*
  call void @free(i8* %1162) #5
  %1163 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %1164 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1163, i32 0, i32 1
  %1165 = load i8*, i8** %1164, align 8
  call void @free(i8* %1165) #5
  %1166 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %1167 = bitcast %struct.mpc_parser_t* %1166 to i8*
  call void @free(i8* %1167) #5
  br label %185

; <label>:1168:                                   ; preds = %1018, %1007, %995, %988, %982
  %1169 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1170 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1169, i32 0, i32 2
  %1171 = load i8, i8* %1170, align 8
  %1172 = sext i8 %1171 to i32
  %1173 = icmp eq i32 %1172, 24
  br i1 %1173, label %1174, label %1366

; <label>:1174:                                   ; preds = %1168
  %1175 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1176 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1175, i32 0, i32 3
  %1177 = bitcast %union.mpc_pdata_t* %1176 to %struct.mpc_pdata_and_t*
  %1178 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1177, i32 0, i32 1
  %1179 = load i8* (i32, i8**)*, i8* (i32, i8**)** %1178, align 8
  %1180 = icmp eq i8* (i32, i8**)* %1179, @mpcf_strfold
  br i1 %1180, label %1181, label %1366

; <label>:1181:                                   ; preds = %1174
  %1182 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1183 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1182, i32 0, i32 3
  %1184 = bitcast %union.mpc_pdata_t* %1183 to %struct.mpc_pdata_and_t*
  %1185 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1184, i32 0, i32 2
  %1186 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %1185, align 8
  %1187 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1188 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1187, i32 0, i32 3
  %1189 = bitcast %union.mpc_pdata_t* %1188 to %struct.mpc_pdata_and_t*
  %1190 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1189, i32 0, i32 0
  %1191 = load i32, i32* %1190, align 8
  %1192 = sub nsw i32 %1191, 1
  %1193 = sext i32 %1192 to i64
  %1194 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %1186, i64 %1193
  %1195 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1194, align 8
  %1196 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1195, i32 0, i32 2
  %1197 = load i8, i8* %1196, align 8
  %1198 = sext i8 %1197 to i32
  %1199 = icmp eq i32 %1198, 24
  br i1 %1199, label %1200, label %1366

; <label>:1200:                                   ; preds = %1181
  %1201 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1202 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1201, i32 0, i32 3
  %1203 = bitcast %union.mpc_pdata_t* %1202 to %struct.mpc_pdata_and_t*
  %1204 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1203, i32 0, i32 2
  %1205 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %1204, align 8
  %1206 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1207 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1206, i32 0, i32 3
  %1208 = bitcast %union.mpc_pdata_t* %1207 to %struct.mpc_pdata_and_t*
  %1209 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1208, i32 0, i32 0
  %1210 = load i32, i32* %1209, align 8
  %1211 = sub nsw i32 %1210, 1
  %1212 = sext i32 %1211 to i64
  %1213 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %1205, i64 %1212
  %1214 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1213, align 8
  %1215 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1214, i32 0, i32 0
  %1216 = load i8, i8* %1215, align 8
  %1217 = icmp ne i8 %1216, 0
  br i1 %1217, label %1366, label %1218

; <label>:1218:                                   ; preds = %1200
  %1219 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1220 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1219, i32 0, i32 3
  %1221 = bitcast %union.mpc_pdata_t* %1220 to %struct.mpc_pdata_and_t*
  %1222 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1221, i32 0, i32 2
  %1223 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %1222, align 8
  %1224 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1225 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1224, i32 0, i32 3
  %1226 = bitcast %union.mpc_pdata_t* %1225 to %struct.mpc_pdata_and_t*
  %1227 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1226, i32 0, i32 0
  %1228 = load i32, i32* %1227, align 8
  %1229 = sub nsw i32 %1228, 1
  %1230 = sext i32 %1229 to i64
  %1231 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %1223, i64 %1230
  %1232 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1231, align 8
  %1233 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1232, i32 0, i32 3
  %1234 = bitcast %union.mpc_pdata_t* %1233 to %struct.mpc_pdata_and_t*
  %1235 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1234, i32 0, i32 1
  %1236 = load i8* (i32, i8**)*, i8* (i32, i8**)** %1235, align 8
  %1237 = icmp eq i8* (i32, i8**)* %1236, @mpcf_strfold
  br i1 %1237, label %1238, label %1366

; <label>:1238:                                   ; preds = %1218
  %1239 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1240 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1239, i32 0, i32 3
  %1241 = bitcast %union.mpc_pdata_t* %1240 to %struct.mpc_pdata_and_t*
  %1242 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1241, i32 0, i32 2
  %1243 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %1242, align 8
  %1244 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1245 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1244, i32 0, i32 3
  %1246 = bitcast %union.mpc_pdata_t* %1245 to %struct.mpc_pdata_and_t*
  %1247 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1246, i32 0, i32 0
  %1248 = load i32, i32* %1247, align 8
  %1249 = sub nsw i32 %1248, 1
  %1250 = sext i32 %1249 to i64
  %1251 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %1243, i64 %1250
  %1252 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %1251, align 8
  store %struct.mpc_parser_t* %1252, %struct.mpc_parser_t** %8, align 8
  %1253 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1254 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1253, i32 0, i32 3
  %1255 = bitcast %union.mpc_pdata_t* %1254 to %struct.mpc_pdata_and_t*
  %1256 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1255, i32 0, i32 0
  %1257 = load i32, i32* %1256, align 8
  store i32 %1257, i32* %6, align 4
  %1258 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %1259 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1258, i32 0, i32 3
  %1260 = bitcast %union.mpc_pdata_t* %1259 to %struct.mpc_pdata_and_t*
  %1261 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1260, i32 0, i32 0
  %1262 = load i32, i32* %1261, align 8
  store i32 %1262, i32* %7, align 4
  %1263 = load i32, i32* %6, align 4
  %1264 = load i32, i32* %7, align 4
  %1265 = add nsw i32 %1263, %1264
  %1266 = sub nsw i32 %1265, 1
  %1267 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1268 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1267, i32 0, i32 3
  %1269 = bitcast %union.mpc_pdata_t* %1268 to %struct.mpc_pdata_and_t*
  %1270 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1269, i32 0, i32 0
  store i32 %1266, i32* %1270, align 8
  %1271 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1272 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1271, i32 0, i32 3
  %1273 = bitcast %union.mpc_pdata_t* %1272 to %struct.mpc_pdata_and_t*
  %1274 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1273, i32 0, i32 2
  %1275 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %1274, align 8
  %1276 = bitcast %struct.mpc_parser_t** %1275 to i8*
  %1277 = load i32, i32* %6, align 4
  %1278 = load i32, i32* %7, align 4
  %1279 = add nsw i32 %1277, %1278
  %1280 = sub nsw i32 %1279, 1
  %1281 = sext i32 %1280 to i64
  %1282 = mul i64 8, %1281
  %1283 = call i8* @realloc(i8* %1276, i64 %1282) #5
  %1284 = bitcast i8* %1283 to %struct.mpc_parser_t**
  %1285 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1286 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1285, i32 0, i32 3
  %1287 = bitcast %union.mpc_pdata_t* %1286 to %struct.mpc_pdata_and_t*
  %1288 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1287, i32 0, i32 2
  store %struct.mpc_parser_t** %1284, %struct.mpc_parser_t*** %1288, align 8
  %1289 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1290 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1289, i32 0, i32 3
  %1291 = bitcast %union.mpc_pdata_t* %1290 to %struct.mpc_pdata_and_t*
  %1292 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1291, i32 0, i32 3
  %1293 = load void (i8*)**, void (i8*)*** %1292, align 8
  %1294 = bitcast void (i8*)** %1293 to i8*
  %1295 = load i32, i32* %6, align 4
  %1296 = load i32, i32* %7, align 4
  %1297 = add nsw i32 %1295, %1296
  %1298 = sub nsw i32 %1297, 1
  %1299 = sub nsw i32 %1298, 1
  %1300 = sext i32 %1299 to i64
  %1301 = mul i64 8, %1300
  %1302 = call i8* @realloc(i8* %1294, i64 %1301) #5
  %1303 = bitcast i8* %1302 to void (i8*)**
  %1304 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1305 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1304, i32 0, i32 3
  %1306 = bitcast %union.mpc_pdata_t* %1305 to %struct.mpc_pdata_and_t*
  %1307 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1306, i32 0, i32 3
  store void (i8*)** %1303, void (i8*)*** %1307, align 8
  %1308 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1309 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1308, i32 0, i32 3
  %1310 = bitcast %union.mpc_pdata_t* %1309 to %struct.mpc_pdata_and_t*
  %1311 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1310, i32 0, i32 2
  %1312 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %1311, align 8
  %1313 = load i32, i32* %6, align 4
  %1314 = sext i32 %1313 to i64
  %1315 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %1312, i64 %1314
  %1316 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %1315, i64 -1
  %1317 = bitcast %struct.mpc_parser_t** %1316 to i8*
  %1318 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %1319 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1318, i32 0, i32 3
  %1320 = bitcast %union.mpc_pdata_t* %1319 to %struct.mpc_pdata_and_t*
  %1321 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1320, i32 0, i32 2
  %1322 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %1321, align 8
  %1323 = bitcast %struct.mpc_parser_t** %1322 to i8*
  %1324 = load i32, i32* %7, align 4
  %1325 = sext i32 %1324 to i64
  %1326 = mul i64 %1325, 8
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %1317, i8* %1323, i64 %1326, i32 8, i1 false)
  store i32 0, i32* %5, align 4
  br label %1327

; <label>:1327:                                   ; preds = %1345, %1238
  %1328 = load i32, i32* %5, align 4
  %1329 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1330 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1329, i32 0, i32 3
  %1331 = bitcast %union.mpc_pdata_t* %1330 to %struct.mpc_pdata_and_t*
  %1332 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1331, i32 0, i32 0
  %1333 = load i32, i32* %1332, align 8
  %1334 = sub nsw i32 %1333, 1
  %1335 = icmp slt i32 %1328, %1334
  br i1 %1335, label %1336, label %1348

; <label>:1336:                                   ; preds = %1327
  %1337 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  %1338 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1337, i32 0, i32 3
  %1339 = bitcast %union.mpc_pdata_t* %1338 to %struct.mpc_pdata_and_t*
  %1340 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1339, i32 0, i32 3
  %1341 = load void (i8*)**, void (i8*)*** %1340, align 8
  %1342 = load i32, i32* %5, align 4
  %1343 = sext i32 %1342 to i64
  %1344 = getelementptr inbounds void (i8*)*, void (i8*)** %1341, i64 %1343
  store void (i8*)* @free, void (i8*)** %1344, align 8
  br label %1345

; <label>:1345:                                   ; preds = %1336
  %1346 = load i32, i32* %5, align 4
  %1347 = add nsw i32 %1346, 1
  store i32 %1347, i32* %5, align 4
  br label %1327

; <label>:1348:                                   ; preds = %1327
  %1349 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %1350 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1349, i32 0, i32 3
  %1351 = bitcast %union.mpc_pdata_t* %1350 to %struct.mpc_pdata_and_t*
  %1352 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1351, i32 0, i32 2
  %1353 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %1352, align 8
  %1354 = bitcast %struct.mpc_parser_t** %1353 to i8*
  call void @free(i8* %1354) #5
  %1355 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %1356 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1355, i32 0, i32 3
  %1357 = bitcast %union.mpc_pdata_t* %1356 to %struct.mpc_pdata_and_t*
  %1358 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %1357, i32 0, i32 3
  %1359 = load void (i8*)**, void (i8*)*** %1358, align 8
  %1360 = bitcast void (i8*)** %1359 to i8*
  call void @free(i8* %1360) #5
  %1361 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %1362 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %1361, i32 0, i32 1
  %1363 = load i8*, i8** %1362, align 8
  call void @free(i8* %1363) #5
  %1364 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %1365 = bitcast %struct.mpc_parser_t* %1364 to i8*
  call void @free(i8* %1365) #5
  br label %185

; <label>:1366:                                   ; preds = %1218, %1200, %1181, %1174, %1168
  br label %1367

; <label>:1367:                                   ; preds = %1366, %17
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpc_malloc(%struct.mpc_input_t*, i64) #0 {
  %3 = alloca i8*, align 8
  %4 = alloca %struct.mpc_input_t*, align 8
  %5 = alloca i64, align 8
  %6 = alloca i64, align 8
  %7 = alloca i8*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %4, align 8
  store i64 %1, i64* %5, align 8
  %8 = load i64, i64* %5, align 8
  %9 = icmp ugt i64 %8, 64
  br i1 %9, label %10, label %13

; <label>:10:                                     ; preds = %2
  %11 = load i64, i64* %5, align 8
  %12 = call noalias i8* @malloc(i64 %11) #5
  store i8* %12, i8** %3, align 8
  br label %66

; <label>:13:                                     ; preds = %2
  %14 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %15 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %14, i32 0, i32 13
  %16 = load i64, i64* %15, align 8
  store i64 %16, i64* %6, align 8
  br label %17

; <label>:17:                                     ; preds = %57, %13
  %18 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %19 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %18, i32 0, i32 14
  %20 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %21 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %20, i32 0, i32 13
  %22 = load i64, i64* %21, align 8
  %23 = getelementptr inbounds [512 x i8], [512 x i8]* %19, i64 0, i64 %22
  %24 = load i8, i8* %23, align 1
  %25 = icmp ne i8 %24, 0
  br i1 %25, label %49, label %26

; <label>:26:                                     ; preds = %17
  %27 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %28 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %27, i32 0, i32 15
  %29 = getelementptr inbounds [512 x %struct.mpc_mem_t], [512 x %struct.mpc_mem_t]* %28, i32 0, i32 0
  %30 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %31 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %30, i32 0, i32 13
  %32 = load i64, i64* %31, align 8
  %33 = getelementptr inbounds %struct.mpc_mem_t, %struct.mpc_mem_t* %29, i64 %32
  %34 = bitcast %struct.mpc_mem_t* %33 to i8*
  store i8* %34, i8** %7, align 8
  %35 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %36 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %35, i32 0, i32 14
  %37 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %38 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %37, i32 0, i32 13
  %39 = load i64, i64* %38, align 8
  %40 = getelementptr inbounds [512 x i8], [512 x i8]* %36, i64 0, i64 %39
  store i8 1, i8* %40, align 1
  %41 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %42 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %41, i32 0, i32 13
  %43 = load i64, i64* %42, align 8
  %44 = add i64 %43, 1
  %45 = urem i64 %44, 512
  %46 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %47 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %46, i32 0, i32 13
  store i64 %45, i64* %47, align 8
  %48 = load i8*, i8** %7, align 8
  store i8* %48, i8** %3, align 8
  br label %66

; <label>:49:                                     ; preds = %17
  %50 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %51 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %50, i32 0, i32 13
  %52 = load i64, i64* %51, align 8
  %53 = add i64 %52, 1
  %54 = urem i64 %53, 512
  %55 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %56 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %55, i32 0, i32 13
  store i64 %54, i64* %56, align 8
  br label %57

; <label>:57:                                     ; preds = %49
  %58 = load i64, i64* %6, align 8
  %59 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %60 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %59, i32 0, i32 13
  %61 = load i64, i64* %60, align 8
  %62 = icmp ne i64 %58, %61
  br i1 %62, label %17, label %63

; <label>:63:                                     ; preds = %57
  %64 = load i64, i64* %5, align 8
  %65 = call noalias i8* @malloc(i64 %64) #5
  store i8* %65, i8** %3, align 8
  br label %66

; <label>:66:                                     ; preds = %63, %26, %10
  %67 = load i8*, i8** %3, align 8
  ret i8* %67
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @mpc_input_any(%struct.mpc_input_t*, i8**) #0 {
  %3 = alloca i32, align 4
  %4 = alloca %struct.mpc_input_t*, align 8
  %5 = alloca i8**, align 8
  %6 = alloca i8, align 1
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %4, align 8
  store i8** %1, i8*** %5, align 8
  %7 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %8 = call signext i8 @mpc_input_getc(%struct.mpc_input_t* %7)
  store i8 %8, i8* %6, align 1
  %9 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %10 = call i32 @mpc_input_terminated(%struct.mpc_input_t* %9)
  %11 = icmp ne i32 %10, 0
  br i1 %11, label %12, label %13

; <label>:12:                                     ; preds = %2
  store i32 0, i32* %3, align 4
  br label %18

; <label>:13:                                     ; preds = %2
  %14 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %15 = load i8, i8* %6, align 1
  %16 = load i8**, i8*** %5, align 8
  %17 = call i32 @mpc_input_success(%struct.mpc_input_t* %14, i8 signext %15, i8** %16)
  store i32 %17, i32* %3, align 4
  br label %18

; <label>:18:                                     ; preds = %13, %12
  %19 = load i32, i32* %3, align 4
  ret i32 %19
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @mpc_input_char(%struct.mpc_input_t*, i8 signext, i8**) #0 {
  %4 = alloca i32, align 4
  %5 = alloca %struct.mpc_input_t*, align 8
  %6 = alloca i8, align 1
  %7 = alloca i8**, align 8
  %8 = alloca i8, align 1
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %5, align 8
  store i8 %1, i8* %6, align 1
  store i8** %2, i8*** %7, align 8
  %9 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %10 = call signext i8 @mpc_input_getc(%struct.mpc_input_t* %9)
  store i8 %10, i8* %8, align 1
  %11 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %12 = call i32 @mpc_input_terminated(%struct.mpc_input_t* %11)
  %13 = icmp ne i32 %12, 0
  br i1 %13, label %14, label %15

; <label>:14:                                     ; preds = %3
  store i32 0, i32* %4, align 4
  br label %32

; <label>:15:                                     ; preds = %3
  %16 = load i8, i8* %8, align 1
  %17 = sext i8 %16 to i32
  %18 = load i8, i8* %6, align 1
  %19 = sext i8 %18 to i32
  %20 = icmp eq i32 %17, %19
  br i1 %20, label %21, label %26

; <label>:21:                                     ; preds = %15
  %22 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %23 = load i8, i8* %8, align 1
  %24 = load i8**, i8*** %7, align 8
  %25 = call i32 @mpc_input_success(%struct.mpc_input_t* %22, i8 signext %23, i8** %24)
  br label %30

; <label>:26:                                     ; preds = %15
  %27 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %28 = load i8, i8* %8, align 1
  %29 = call i32 @mpc_input_failure(%struct.mpc_input_t* %27, i8 signext %28)
  br label %30

; <label>:30:                                     ; preds = %26, %21
  %31 = phi i32 [ %25, %21 ], [ %29, %26 ]
  store i32 %31, i32* %4, align 4
  br label %32

; <label>:32:                                     ; preds = %30, %14
  %33 = load i32, i32* %4, align 4
  ret i32 %33
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @mpc_input_range(%struct.mpc_input_t*, i8 signext, i8 signext, i8**) #0 {
  %5 = alloca i32, align 4
  %6 = alloca %struct.mpc_input_t*, align 8
  %7 = alloca i8, align 1
  %8 = alloca i8, align 1
  %9 = alloca i8**, align 8
  %10 = alloca i8, align 1
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %6, align 8
  store i8 %1, i8* %7, align 1
  store i8 %2, i8* %8, align 1
  store i8** %3, i8*** %9, align 8
  %11 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %12 = call signext i8 @mpc_input_getc(%struct.mpc_input_t* %11)
  store i8 %12, i8* %10, align 1
  %13 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %14 = call i32 @mpc_input_terminated(%struct.mpc_input_t* %13)
  %15 = icmp ne i32 %14, 0
  br i1 %15, label %16, label %17

; <label>:16:                                     ; preds = %4
  store i32 0, i32* %5, align 4
  br label %40

; <label>:17:                                     ; preds = %4
  %18 = load i8, i8* %10, align 1
  %19 = sext i8 %18 to i32
  %20 = load i8, i8* %7, align 1
  %21 = sext i8 %20 to i32
  %22 = icmp sge i32 %19, %21
  br i1 %22, label %23, label %34

; <label>:23:                                     ; preds = %17
  %24 = load i8, i8* %10, align 1
  %25 = sext i8 %24 to i32
  %26 = load i8, i8* %8, align 1
  %27 = sext i8 %26 to i32
  %28 = icmp sle i32 %25, %27
  br i1 %28, label %29, label %34

; <label>:29:                                     ; preds = %23
  %30 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %31 = load i8, i8* %10, align 1
  %32 = load i8**, i8*** %9, align 8
  %33 = call i32 @mpc_input_success(%struct.mpc_input_t* %30, i8 signext %31, i8** %32)
  br label %38

; <label>:34:                                     ; preds = %23, %17
  %35 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %36 = load i8, i8* %10, align 1
  %37 = call i32 @mpc_input_failure(%struct.mpc_input_t* %35, i8 signext %36)
  br label %38

; <label>:38:                                     ; preds = %34, %29
  %39 = phi i32 [ %33, %29 ], [ %37, %34 ]
  store i32 %39, i32* %5, align 4
  br label %40

; <label>:40:                                     ; preds = %38, %16
  %41 = load i32, i32* %5, align 4
  ret i32 %41
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @mpc_input_oneof(%struct.mpc_input_t*, i8*, i8**) #0 {
  %4 = alloca i32, align 4
  %5 = alloca %struct.mpc_input_t*, align 8
  %6 = alloca i8*, align 8
  %7 = alloca i8**, align 8
  %8 = alloca i8, align 1
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %5, align 8
  store i8* %1, i8** %6, align 8
  store i8** %2, i8*** %7, align 8
  %9 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %10 = call signext i8 @mpc_input_getc(%struct.mpc_input_t* %9)
  store i8 %10, i8* %8, align 1
  %11 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %12 = call i32 @mpc_input_terminated(%struct.mpc_input_t* %11)
  %13 = icmp ne i32 %12, 0
  br i1 %13, label %14, label %15

; <label>:14:                                     ; preds = %3
  store i32 0, i32* %4, align 4
  br label %32

; <label>:15:                                     ; preds = %3
  %16 = load i8*, i8** %6, align 8
  %17 = load i8, i8* %8, align 1
  %18 = sext i8 %17 to i32
  %19 = call i8* @strchr(i8* %16, i32 %18) #7
  %20 = icmp ne i8* %19, null
  br i1 %20, label %21, label %26

; <label>:21:                                     ; preds = %15
  %22 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %23 = load i8, i8* %8, align 1
  %24 = load i8**, i8*** %7, align 8
  %25 = call i32 @mpc_input_success(%struct.mpc_input_t* %22, i8 signext %23, i8** %24)
  br label %30

; <label>:26:                                     ; preds = %15
  %27 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %28 = load i8, i8* %8, align 1
  %29 = call i32 @mpc_input_failure(%struct.mpc_input_t* %27, i8 signext %28)
  br label %30

; <label>:30:                                     ; preds = %26, %21
  %31 = phi i32 [ %25, %21 ], [ %29, %26 ]
  store i32 %31, i32* %4, align 4
  br label %32

; <label>:32:                                     ; preds = %30, %14
  %33 = load i32, i32* %4, align 4
  ret i32 %33
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @mpc_input_noneof(%struct.mpc_input_t*, i8*, i8**) #0 {
  %4 = alloca i32, align 4
  %5 = alloca %struct.mpc_input_t*, align 8
  %6 = alloca i8*, align 8
  %7 = alloca i8**, align 8
  %8 = alloca i8, align 1
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %5, align 8
  store i8* %1, i8** %6, align 8
  store i8** %2, i8*** %7, align 8
  %9 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %10 = call signext i8 @mpc_input_getc(%struct.mpc_input_t* %9)
  store i8 %10, i8* %8, align 1
  %11 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %12 = call i32 @mpc_input_terminated(%struct.mpc_input_t* %11)
  %13 = icmp ne i32 %12, 0
  br i1 %13, label %14, label %15

; <label>:14:                                     ; preds = %3
  store i32 0, i32* %4, align 4
  br label %32

; <label>:15:                                     ; preds = %3
  %16 = load i8*, i8** %6, align 8
  %17 = load i8, i8* %8, align 1
  %18 = sext i8 %17 to i32
  %19 = call i8* @strchr(i8* %16, i32 %18) #7
  %20 = icmp eq i8* %19, null
  br i1 %20, label %21, label %26

; <label>:21:                                     ; preds = %15
  %22 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %23 = load i8, i8* %8, align 1
  %24 = load i8**, i8*** %7, align 8
  %25 = call i32 @mpc_input_success(%struct.mpc_input_t* %22, i8 signext %23, i8** %24)
  br label %30

; <label>:26:                                     ; preds = %15
  %27 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %28 = load i8, i8* %8, align 1
  %29 = call i32 @mpc_input_failure(%struct.mpc_input_t* %27, i8 signext %28)
  br label %30

; <label>:30:                                     ; preds = %26, %21
  %31 = phi i32 [ %25, %21 ], [ %29, %26 ]
  store i32 %31, i32* %4, align 4
  br label %32

; <label>:32:                                     ; preds = %30, %14
  %33 = load i32, i32* %4, align 4
  ret i32 %33
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @mpc_input_satisfy(%struct.mpc_input_t*, i32 (i8)*, i8**) #0 {
  %4 = alloca i32, align 4
  %5 = alloca %struct.mpc_input_t*, align 8
  %6 = alloca i32 (i8)*, align 8
  %7 = alloca i8**, align 8
  %8 = alloca i8, align 1
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %5, align 8
  store i32 (i8)* %1, i32 (i8)** %6, align 8
  store i8** %2, i8*** %7, align 8
  %9 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %10 = call signext i8 @mpc_input_getc(%struct.mpc_input_t* %9)
  store i8 %10, i8* %8, align 1
  %11 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %12 = call i32 @mpc_input_terminated(%struct.mpc_input_t* %11)
  %13 = icmp ne i32 %12, 0
  br i1 %13, label %14, label %15

; <label>:14:                                     ; preds = %3
  store i32 0, i32* %4, align 4
  br label %31

; <label>:15:                                     ; preds = %3
  %16 = load i32 (i8)*, i32 (i8)** %6, align 8
  %17 = load i8, i8* %8, align 1
  %18 = call i32 %16(i8 signext %17)
  %19 = icmp ne i32 %18, 0
  br i1 %19, label %20, label %25

; <label>:20:                                     ; preds = %15
  %21 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %22 = load i8, i8* %8, align 1
  %23 = load i8**, i8*** %7, align 8
  %24 = call i32 @mpc_input_success(%struct.mpc_input_t* %21, i8 signext %22, i8** %23)
  br label %29

; <label>:25:                                     ; preds = %15
  %26 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %27 = load i8, i8* %8, align 1
  %28 = call i32 @mpc_input_failure(%struct.mpc_input_t* %26, i8 signext %27)
  br label %29

; <label>:29:                                     ; preds = %25, %20
  %30 = phi i32 [ %24, %20 ], [ %28, %25 ]
  store i32 %30, i32* %4, align 4
  br label %31

; <label>:31:                                     ; preds = %29, %14
  %32 = load i32, i32* %4, align 4
  ret i32 %32
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @mpc_input_string(%struct.mpc_input_t*, i8*, i8**) #0 {
  %4 = alloca i32, align 4
  %5 = alloca %struct.mpc_input_t*, align 8
  %6 = alloca i8*, align 8
  %7 = alloca i8**, align 8
  %8 = alloca i8*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %5, align 8
  store i8* %1, i8** %6, align 8
  store i8** %2, i8*** %7, align 8
  %9 = load i8*, i8** %6, align 8
  store i8* %9, i8** %8, align 8
  %10 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  call void @mpc_input_mark(%struct.mpc_input_t* %10)
  br label %11

; <label>:11:                                     ; preds = %23, %3
  %12 = load i8*, i8** %8, align 8
  %13 = load i8, i8* %12, align 1
  %14 = icmp ne i8 %13, 0
  br i1 %14, label %15, label %26

; <label>:15:                                     ; preds = %11
  %16 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %17 = load i8*, i8** %8, align 8
  %18 = load i8, i8* %17, align 1
  %19 = call i32 @mpc_input_char(%struct.mpc_input_t* %16, i8 signext %18, i8** null)
  %20 = icmp ne i32 %19, 0
  br i1 %20, label %23, label %21

; <label>:21:                                     ; preds = %15
  %22 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  call void @mpc_input_rewind(%struct.mpc_input_t* %22)
  store i32 0, i32* %4, align 4
  br label %38

; <label>:23:                                     ; preds = %15
  %24 = load i8*, i8** %8, align 8
  %25 = getelementptr inbounds i8, i8* %24, i32 1
  store i8* %25, i8** %8, align 8
  br label %11

; <label>:26:                                     ; preds = %11
  %27 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  call void @mpc_input_unmark(%struct.mpc_input_t* %27)
  %28 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %29 = load i8*, i8** %6, align 8
  %30 = call i64 @strlen(i8* %29) #7
  %31 = add i64 %30, 1
  %32 = call i8* @mpc_malloc(%struct.mpc_input_t* %28, i64 %31)
  %33 = load i8**, i8*** %7, align 8
  store i8* %32, i8** %33, align 8
  %34 = load i8**, i8*** %7, align 8
  %35 = load i8*, i8** %34, align 8
  %36 = load i8*, i8** %6, align 8
  %37 = call i8* @strcpy(i8* %35, i8* %36) #5
  store i32 1, i32* %4, align 4
  br label %38

; <label>:38:                                     ; preds = %26, %21
  %39 = load i32, i32* %4, align 4
  ret i32 %39
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @mpc_input_anchor(%struct.mpc_input_t*, i32 (i8, i8)*, i8**) #0 {
  %4 = alloca %struct.mpc_input_t*, align 8
  %5 = alloca i32 (i8, i8)*, align 8
  %6 = alloca i8**, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %4, align 8
  store i32 (i8, i8)* %1, i32 (i8, i8)** %5, align 8
  store i8** %2, i8*** %6, align 8
  %7 = load i8**, i8*** %6, align 8
  store i8* null, i8** %7, align 8
  %8 = load i32 (i8, i8)*, i32 (i8, i8)** %5, align 8
  %9 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %10 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %9, i32 0, i32 12
  %11 = load i8, i8* %10, align 8
  %12 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %13 = call signext i8 @mpc_input_peekc(%struct.mpc_input_t* %12)
  %14 = call i32 %8(i8 signext %11, i8 signext %13)
  ret i32 %14
}

; Function Attrs: noinline nounwind optnone uwtable
define internal %struct.mpc_state_t* @mpc_input_state_copy(%struct.mpc_input_t*) #0 {
  %2 = alloca %struct.mpc_input_t*, align 8
  %3 = alloca %struct.mpc_state_t*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %2, align 8
  %4 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %5 = call i8* @mpc_malloc(%struct.mpc_input_t* %4, i64 24)
  %6 = bitcast i8* %5 to %struct.mpc_state_t*
  store %struct.mpc_state_t* %6, %struct.mpc_state_t** %3, align 8
  %7 = load %struct.mpc_state_t*, %struct.mpc_state_t** %3, align 8
  %8 = bitcast %struct.mpc_state_t* %7 to i8*
  %9 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %10 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %9, i32 0, i32 2
  %11 = bitcast %struct.mpc_state_t* %10 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %8, i8* %11, i64 24, i32 8, i1 false)
  %12 = load %struct.mpc_state_t*, %struct.mpc_state_t** %3, align 8
  ret %struct.mpc_state_t* %12
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpc_parse_apply(%struct.mpc_input_t*, i8* (i8*)*, i8*) #0 {
  %4 = alloca i8*, align 8
  %5 = alloca %struct.mpc_input_t*, align 8
  %6 = alloca i8* (i8*)*, align 8
  %7 = alloca i8*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %5, align 8
  store i8* (i8*)* %1, i8* (i8*)** %6, align 8
  store i8* %2, i8** %7, align 8
  %8 = load i8* (i8*)*, i8* (i8*)** %6, align 8
  %9 = icmp eq i8* (i8*)* %8, @mpcf_free
  br i1 %9, label %10, label %14

; <label>:10:                                     ; preds = %3
  %11 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %12 = load i8*, i8** %7, align 8
  %13 = call i8* @mpcf_input_free(%struct.mpc_input_t* %11, i8* %12)
  store i8* %13, i8** %4, align 8
  br label %27

; <label>:14:                                     ; preds = %3
  %15 = load i8* (i8*)*, i8* (i8*)** %6, align 8
  %16 = icmp eq i8* (i8*)* %15, @mpcf_str_ast
  br i1 %16, label %17, label %21

; <label>:17:                                     ; preds = %14
  %18 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %19 = load i8*, i8** %7, align 8
  %20 = call i8* @mpcf_input_str_ast(%struct.mpc_input_t* %18, i8* %19)
  store i8* %20, i8** %4, align 8
  br label %27

; <label>:21:                                     ; preds = %14
  %22 = load i8* (i8*)*, i8* (i8*)** %6, align 8
  %23 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %24 = load i8*, i8** %7, align 8
  %25 = call i8* @mpc_export(%struct.mpc_input_t* %23, i8* %24)
  %26 = call i8* %22(i8* %25)
  store i8* %26, i8** %4, align 8
  br label %27

; <label>:27:                                     ; preds = %21, %17, %10
  %28 = load i8*, i8** %4, align 8
  ret i8* %28
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpc_parse_apply_to(%struct.mpc_input_t*, i8* (i8*, i8*)*, i8*, i8*) #0 {
  %5 = alloca %struct.mpc_input_t*, align 8
  %6 = alloca i8* (i8*, i8*)*, align 8
  %7 = alloca i8*, align 8
  %8 = alloca i8*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %5, align 8
  store i8* (i8*, i8*)* %1, i8* (i8*, i8*)** %6, align 8
  store i8* %2, i8** %7, align 8
  store i8* %3, i8** %8, align 8
  %9 = load i8* (i8*, i8*)*, i8* (i8*, i8*)** %6, align 8
  %10 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %11 = load i8*, i8** %7, align 8
  %12 = call i8* @mpc_export(%struct.mpc_input_t* %10, i8* %11)
  %13 = load i8*, i8** %8, align 8
  %14 = call i8* %9(i8* %12, i8* %13)
  ret i8* %14
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_input_suppress_enable(%struct.mpc_input_t*) #0 {
  %2 = alloca %struct.mpc_input_t*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %2, align 8
  %3 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %4 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %3, i32 0, i32 6
  %5 = load i32, i32* %4, align 8
  %6 = add nsw i32 %5, 1
  store i32 %6, i32* %4, align 8
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_input_suppress_disable(%struct.mpc_input_t*) #0 {
  %2 = alloca %struct.mpc_input_t*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %2, align 8
  %3 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %4 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %3, i32 0, i32 6
  %5 = load i32, i32* %4, align 8
  %6 = add nsw i32 %5, -1
  store i32 %6, i32* %4, align 8
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal %struct.mpc_err_t* @mpc_err_new(%struct.mpc_input_t*, i8*) #0 {
  %3 = alloca %struct.mpc_err_t*, align 8
  %4 = alloca %struct.mpc_input_t*, align 8
  %5 = alloca i8*, align 8
  %6 = alloca %struct.mpc_err_t*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %4, align 8
  store i8* %1, i8** %5, align 8
  %7 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %8 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %7, i32 0, i32 6
  %9 = load i32, i32* %8, align 8
  %10 = icmp ne i32 %9, 0
  br i1 %10, label %11, label %12

; <label>:11:                                     ; preds = %2
  store %struct.mpc_err_t* null, %struct.mpc_err_t** %3, align 8
  br label %68

; <label>:12:                                     ; preds = %2
  %13 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %14 = call i8* @mpc_malloc(%struct.mpc_input_t* %13, i64 64)
  %15 = bitcast i8* %14 to %struct.mpc_err_t*
  store %struct.mpc_err_t* %15, %struct.mpc_err_t** %6, align 8
  %16 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %17 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %18 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %17, i32 0, i32 1
  %19 = load i8*, i8** %18, align 8
  %20 = call i64 @strlen(i8* %19) #7
  %21 = add i64 %20, 1
  %22 = call i8* @mpc_malloc(%struct.mpc_input_t* %16, i64 %21)
  %23 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %24 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %23, i32 0, i32 2
  store i8* %22, i8** %24, align 8
  %25 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %26 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %25, i32 0, i32 2
  %27 = load i8*, i8** %26, align 8
  %28 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %29 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %28, i32 0, i32 1
  %30 = load i8*, i8** %29, align 8
  %31 = call i8* @strcpy(i8* %27, i8* %30) #5
  %32 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %33 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %32, i32 0, i32 0
  %34 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %35 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %34, i32 0, i32 2
  %36 = bitcast %struct.mpc_state_t* %33 to i8*
  %37 = bitcast %struct.mpc_state_t* %35 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %36, i8* %37, i64 24, i32 8, i1 false)
  %38 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %39 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %38, i32 0, i32 1
  store i32 1, i32* %39, align 8
  %40 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %41 = call i8* @mpc_malloc(%struct.mpc_input_t* %40, i64 8)
  %42 = bitcast i8* %41 to i8**
  %43 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %44 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %43, i32 0, i32 4
  store i8** %42, i8*** %44, align 8
  %45 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %46 = load i8*, i8** %5, align 8
  %47 = call i64 @strlen(i8* %46) #7
  %48 = add i64 %47, 1
  %49 = call i8* @mpc_malloc(%struct.mpc_input_t* %45, i64 %48)
  %50 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %51 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %50, i32 0, i32 4
  %52 = load i8**, i8*** %51, align 8
  %53 = getelementptr inbounds i8*, i8** %52, i64 0
  store i8* %49, i8** %53, align 8
  %54 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %55 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %54, i32 0, i32 4
  %56 = load i8**, i8*** %55, align 8
  %57 = getelementptr inbounds i8*, i8** %56, i64 0
  %58 = load i8*, i8** %57, align 8
  %59 = load i8*, i8** %5, align 8
  %60 = call i8* @strcpy(i8* %58, i8* %59) #5
  %61 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %62 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %61, i32 0, i32 3
  store i8* null, i8** %62, align 8
  %63 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %64 = call signext i8 @mpc_input_peekc(%struct.mpc_input_t* %63)
  %65 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %66 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %65, i32 0, i32 5
  store i8 %64, i8* %66, align 8
  %67 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  store %struct.mpc_err_t* %67, %struct.mpc_err_t** %3, align 8
  br label %68

; <label>:68:                                     ; preds = %12, %11
  %69 = load %struct.mpc_err_t*, %struct.mpc_err_t** %3, align 8
  ret %struct.mpc_err_t* %69
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_input_backtrack_disable(%struct.mpc_input_t*) #0 {
  %2 = alloca %struct.mpc_input_t*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %2, align 8
  %3 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %4 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %3, i32 0, i32 7
  %5 = load i32, i32* %4, align 4
  %6 = add nsw i32 %5, -1
  store i32 %6, i32* %4, align 4
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_input_backtrack_enable(%struct.mpc_input_t*) #0 {
  %2 = alloca %struct.mpc_input_t*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %2, align 8
  %3 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %4 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %3, i32 0, i32 7
  %5 = load i32, i32* %4, align 4
  %6 = add nsw i32 %5, 1
  store i32 %6, i32* %4, align 4
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_input_mark(%struct.mpc_input_t*) #0 {
  %2 = alloca %struct.mpc_input_t*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %2, align 8
  %3 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %4 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %3, i32 0, i32 7
  %5 = load i32, i32* %4, align 4
  %6 = icmp slt i32 %5, 1
  br i1 %6, label %7, label %8

; <label>:7:                                      ; preds = %1
  br label %94

; <label>:8:                                      ; preds = %1
  %9 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %10 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %9, i32 0, i32 9
  %11 = load i32, i32* %10, align 4
  %12 = add nsw i32 %11, 1
  store i32 %12, i32* %10, align 4
  %13 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %14 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %13, i32 0, i32 9
  %15 = load i32, i32* %14, align 4
  %16 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %17 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %16, i32 0, i32 8
  %18 = load i32, i32* %17, align 8
  %19 = icmp sgt i32 %15, %18
  br i1 %19, label %20, label %55

; <label>:20:                                     ; preds = %8
  %21 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %22 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %21, i32 0, i32 9
  %23 = load i32, i32* %22, align 4
  %24 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %25 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %24, i32 0, i32 9
  %26 = load i32, i32* %25, align 4
  %27 = sdiv i32 %26, 2
  %28 = add nsw i32 %23, %27
  %29 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %30 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %29, i32 0, i32 8
  store i32 %28, i32* %30, align 8
  %31 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %32 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %31, i32 0, i32 10
  %33 = load %struct.mpc_state_t*, %struct.mpc_state_t** %32, align 8
  %34 = bitcast %struct.mpc_state_t* %33 to i8*
  %35 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %36 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %35, i32 0, i32 8
  %37 = load i32, i32* %36, align 8
  %38 = sext i32 %37 to i64
  %39 = mul i64 24, %38
  %40 = call i8* @realloc(i8* %34, i64 %39) #5
  %41 = bitcast i8* %40 to %struct.mpc_state_t*
  %42 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %43 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %42, i32 0, i32 10
  store %struct.mpc_state_t* %41, %struct.mpc_state_t** %43, align 8
  %44 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %45 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %44, i32 0, i32 11
  %46 = load i8*, i8** %45, align 8
  %47 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %48 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %47, i32 0, i32 8
  %49 = load i32, i32* %48, align 8
  %50 = sext i32 %49 to i64
  %51 = mul i64 1, %50
  %52 = call i8* @realloc(i8* %46, i64 %51) #5
  %53 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %54 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %53, i32 0, i32 11
  store i8* %52, i8** %54, align 8
  br label %55

; <label>:55:                                     ; preds = %20, %8
  %56 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %57 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %56, i32 0, i32 10
  %58 = load %struct.mpc_state_t*, %struct.mpc_state_t** %57, align 8
  %59 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %60 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %59, i32 0, i32 9
  %61 = load i32, i32* %60, align 4
  %62 = sub nsw i32 %61, 1
  %63 = sext i32 %62 to i64
  %64 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %58, i64 %63
  %65 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %66 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %65, i32 0, i32 2
  %67 = bitcast %struct.mpc_state_t* %64 to i8*
  %68 = bitcast %struct.mpc_state_t* %66 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %67, i8* %68, i64 24, i32 8, i1 false)
  %69 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %70 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %69, i32 0, i32 12
  %71 = load i8, i8* %70, align 8
  %72 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %73 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %72, i32 0, i32 11
  %74 = load i8*, i8** %73, align 8
  %75 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %76 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %75, i32 0, i32 9
  %77 = load i32, i32* %76, align 4
  %78 = sub nsw i32 %77, 1
  %79 = sext i32 %78 to i64
  %80 = getelementptr inbounds i8, i8* %74, i64 %79
  store i8 %71, i8* %80, align 1
  %81 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %82 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %81, i32 0, i32 0
  %83 = load i32, i32* %82, align 8
  %84 = icmp eq i32 %83, 2
  br i1 %84, label %85, label %94

; <label>:85:                                     ; preds = %55
  %86 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %87 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %86, i32 0, i32 9
  %88 = load i32, i32* %87, align 4
  %89 = icmp eq i32 %88, 1
  br i1 %89, label %90, label %94

; <label>:90:                                     ; preds = %85
  %91 = call noalias i8* @calloc(i64 1, i64 1) #5
  %92 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %93 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %92, i32 0, i32 4
  store i8* %91, i8** %93, align 8
  br label %94

; <label>:94:                                     ; preds = %7, %90, %85, %55
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_input_rewind(%struct.mpc_input_t*) #0 {
  %2 = alloca %struct.mpc_input_t*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %2, align 8
  %3 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %4 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %3, i32 0, i32 7
  %5 = load i32, i32* %4, align 4
  %6 = icmp slt i32 %5, 1
  br i1 %6, label %7, label %8

; <label>:7:                                      ; preds = %1
  br label %49

; <label>:8:                                      ; preds = %1
  %9 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %10 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %9, i32 0, i32 2
  %11 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %12 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %11, i32 0, i32 10
  %13 = load %struct.mpc_state_t*, %struct.mpc_state_t** %12, align 8
  %14 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %15 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %14, i32 0, i32 9
  %16 = load i32, i32* %15, align 4
  %17 = sub nsw i32 %16, 1
  %18 = sext i32 %17 to i64
  %19 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %13, i64 %18
  %20 = bitcast %struct.mpc_state_t* %10 to i8*
  %21 = bitcast %struct.mpc_state_t* %19 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %20, i8* %21, i64 24, i32 8, i1 false)
  %22 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %23 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %22, i32 0, i32 11
  %24 = load i8*, i8** %23, align 8
  %25 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %26 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %25, i32 0, i32 9
  %27 = load i32, i32* %26, align 4
  %28 = sub nsw i32 %27, 1
  %29 = sext i32 %28 to i64
  %30 = getelementptr inbounds i8, i8* %24, i64 %29
  %31 = load i8, i8* %30, align 1
  %32 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %33 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %32, i32 0, i32 12
  store i8 %31, i8* %33, align 8
  %34 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %35 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %34, i32 0, i32 0
  %36 = load i32, i32* %35, align 8
  %37 = icmp eq i32 %36, 1
  br i1 %37, label %38, label %47

; <label>:38:                                     ; preds = %8
  %39 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %40 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %39, i32 0, i32 5
  %41 = load %struct._IO_FILE*, %struct._IO_FILE** %40, align 8
  %42 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %43 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %42, i32 0, i32 2
  %44 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %43, i32 0, i32 0
  %45 = load i64, i64* %44, align 8
  %46 = call i32 @fseek(%struct._IO_FILE* %41, i64 %45, i32 0)
  br label %47

; <label>:47:                                     ; preds = %38, %8
  %48 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  call void @mpc_input_unmark(%struct.mpc_input_t* %48)
  br label %49

; <label>:49:                                     ; preds = %47, %7
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_parse_dtor(%struct.mpc_input_t*, void (i8*)*, i8*) #0 {
  %4 = alloca %struct.mpc_input_t*, align 8
  %5 = alloca void (i8*)*, align 8
  %6 = alloca i8*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %4, align 8
  store void (i8*)* %1, void (i8*)** %5, align 8
  store i8* %2, i8** %6, align 8
  %7 = load void (i8*)*, void (i8*)** %5, align 8
  %8 = icmp eq void (i8*)* %7, @free
  br i1 %8, label %9, label %12

; <label>:9:                                      ; preds = %3
  %10 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %11 = load i8*, i8** %6, align 8
  call void @mpc_free(%struct.mpc_input_t* %10, i8* %11)
  br label %17

; <label>:12:                                     ; preds = %3
  %13 = load void (i8*)*, void (i8*)** %5, align 8
  %14 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %15 = load i8*, i8** %6, align 8
  %16 = call i8* @mpc_export(%struct.mpc_input_t* %14, i8* %15)
  call void %13(i8* %16)
  br label %17

; <label>:17:                                     ; preds = %12, %9
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_input_unmark(%struct.mpc_input_t*) #0 {
  %2 = alloca %struct.mpc_input_t*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %2, align 8
  %3 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %4 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %3, i32 0, i32 7
  %5 = load i32, i32* %4, align 4
  %6 = icmp slt i32 %5, 1
  br i1 %6, label %7, label %8

; <label>:7:                                      ; preds = %1
  br label %84

; <label>:8:                                      ; preds = %1
  %9 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %10 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %9, i32 0, i32 9
  %11 = load i32, i32* %10, align 4
  %12 = add nsw i32 %11, -1
  store i32 %12, i32* %10, align 4
  %13 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %14 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %13, i32 0, i32 8
  %15 = load i32, i32* %14, align 8
  %16 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %17 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %16, i32 0, i32 9
  %18 = load i32, i32* %17, align 4
  %19 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %20 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %19, i32 0, i32 9
  %21 = load i32, i32* %20, align 4
  %22 = sdiv i32 %21, 2
  %23 = add nsw i32 %18, %22
  %24 = icmp sgt i32 %15, %23
  br i1 %24, label %25, label %68

; <label>:25:                                     ; preds = %8
  %26 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %27 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %26, i32 0, i32 8
  %28 = load i32, i32* %27, align 8
  %29 = icmp sgt i32 %28, 32
  br i1 %29, label %30, label %68

; <label>:30:                                     ; preds = %25
  %31 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %32 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %31, i32 0, i32 9
  %33 = load i32, i32* %32, align 4
  %34 = icmp sgt i32 %33, 32
  br i1 %34, label %35, label %39

; <label>:35:                                     ; preds = %30
  %36 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %37 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %36, i32 0, i32 9
  %38 = load i32, i32* %37, align 4
  br label %40

; <label>:39:                                     ; preds = %30
  br label %40

; <label>:40:                                     ; preds = %39, %35
  %41 = phi i32 [ %38, %35 ], [ 32, %39 ]
  %42 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %43 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %42, i32 0, i32 8
  store i32 %41, i32* %43, align 8
  %44 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %45 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %44, i32 0, i32 10
  %46 = load %struct.mpc_state_t*, %struct.mpc_state_t** %45, align 8
  %47 = bitcast %struct.mpc_state_t* %46 to i8*
  %48 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %49 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %48, i32 0, i32 8
  %50 = load i32, i32* %49, align 8
  %51 = sext i32 %50 to i64
  %52 = mul i64 24, %51
  %53 = call i8* @realloc(i8* %47, i64 %52) #5
  %54 = bitcast i8* %53 to %struct.mpc_state_t*
  %55 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %56 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %55, i32 0, i32 10
  store %struct.mpc_state_t* %54, %struct.mpc_state_t** %56, align 8
  %57 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %58 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %57, i32 0, i32 11
  %59 = load i8*, i8** %58, align 8
  %60 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %61 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %60, i32 0, i32 8
  %62 = load i32, i32* %61, align 8
  %63 = sext i32 %62 to i64
  %64 = mul i64 1, %63
  %65 = call i8* @realloc(i8* %59, i64 %64) #5
  %66 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %67 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %66, i32 0, i32 11
  store i8* %65, i8** %67, align 8
  br label %68

; <label>:68:                                     ; preds = %40, %25, %8
  %69 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %70 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %69, i32 0, i32 0
  %71 = load i32, i32* %70, align 8
  %72 = icmp eq i32 %71, 2
  br i1 %72, label %73, label %84

; <label>:73:                                     ; preds = %68
  %74 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %75 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %74, i32 0, i32 9
  %76 = load i32, i32* %75, align 4
  %77 = icmp eq i32 %76, 0
  br i1 %77, label %78, label %84

; <label>:78:                                     ; preds = %73
  %79 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %80 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %79, i32 0, i32 4
  %81 = load i8*, i8** %80, align 8
  call void @free(i8* %81) #5
  %82 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %83 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %82, i32 0, i32 4
  store i8* null, i8** %83, align 8
  br label %84

; <label>:84:                                     ; preds = %7, %78, %73, %68
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpc_realloc(%struct.mpc_input_t*, i8*, i64) #0 {
  %4 = alloca i8*, align 8
  %5 = alloca %struct.mpc_input_t*, align 8
  %6 = alloca i8*, align 8
  %7 = alloca i64, align 8
  %8 = alloca i8*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %5, align 8
  store i8* %1, i8** %6, align 8
  store i64 %2, i64* %7, align 8
  store i8* null, i8** %8, align 8
  %9 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %10 = load i8*, i8** %6, align 8
  %11 = call i32 @mpc_mem_ptr(%struct.mpc_input_t* %9, i8* %10)
  %12 = icmp ne i32 %11, 0
  br i1 %12, label %17, label %13

; <label>:13:                                     ; preds = %3
  %14 = load i8*, i8** %6, align 8
  %15 = load i64, i64* %7, align 8
  %16 = call i8* @realloc(i8* %14, i64 %15) #5
  store i8* %16, i8** %4, align 8
  br label %30

; <label>:17:                                     ; preds = %3
  %18 = load i64, i64* %7, align 8
  %19 = icmp ugt i64 %18, 64
  br i1 %19, label %20, label %28

; <label>:20:                                     ; preds = %17
  %21 = load i64, i64* %7, align 8
  %22 = call noalias i8* @malloc(i64 %21) #5
  store i8* %22, i8** %8, align 8
  %23 = load i8*, i8** %8, align 8
  %24 = load i8*, i8** %6, align 8
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %23, i8* %24, i64 64, i32 1, i1 false)
  %25 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %26 = load i8*, i8** %6, align 8
  call void @mpc_free(%struct.mpc_input_t* %25, i8* %26)
  %27 = load i8*, i8** %8, align 8
  store i8* %27, i8** %4, align 8
  br label %30

; <label>:28:                                     ; preds = %17
  %29 = load i8*, i8** %6, align 8
  store i8* %29, i8** %4, align 8
  br label %30

; <label>:30:                                     ; preds = %28, %20, %13
  %31 = load i8*, i8** %4, align 8
  ret i8* %31
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpc_parse_fold(%struct.mpc_input_t*, i8* (i32, i8**)*, i32, i8**) #0 {
  %5 = alloca i8*, align 8
  %6 = alloca %struct.mpc_input_t*, align 8
  %7 = alloca i8* (i32, i8**)*, align 8
  %8 = alloca i32, align 4
  %9 = alloca i8**, align 8
  %10 = alloca i32, align 4
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %6, align 8
  store i8* (i32, i8**)* %1, i8* (i32, i8**)** %7, align 8
  store i32 %2, i32* %8, align 4
  store i8** %3, i8*** %9, align 8
  %11 = load i8* (i32, i8**)*, i8* (i32, i8**)** %7, align 8
  %12 = icmp eq i8* (i32, i8**)* %11, @mpcf_null
  br i1 %12, label %13, label %17

; <label>:13:                                     ; preds = %4
  %14 = load i32, i32* %8, align 4
  %15 = load i8**, i8*** %9, align 8
  %16 = call i8* @mpcf_null(i32 %14, i8** %15)
  store i8* %16, i8** %5, align 8
  br label %103

; <label>:17:                                     ; preds = %4
  %18 = load i8* (i32, i8**)*, i8* (i32, i8**)** %7, align 8
  %19 = icmp eq i8* (i32, i8**)* %18, @mpcf_fst
  br i1 %19, label %20, label %24

; <label>:20:                                     ; preds = %17
  %21 = load i32, i32* %8, align 4
  %22 = load i8**, i8*** %9, align 8
  %23 = call i8* @mpcf_fst(i32 %21, i8** %22)
  store i8* %23, i8** %5, align 8
  br label %103

; <label>:24:                                     ; preds = %17
  %25 = load i8* (i32, i8**)*, i8* (i32, i8**)** %7, align 8
  %26 = icmp eq i8* (i32, i8**)* %25, @mpcf_snd
  br i1 %26, label %27, label %31

; <label>:27:                                     ; preds = %24
  %28 = load i32, i32* %8, align 4
  %29 = load i8**, i8*** %9, align 8
  %30 = call i8* @mpcf_snd(i32 %28, i8** %29)
  store i8* %30, i8** %5, align 8
  br label %103

; <label>:31:                                     ; preds = %24
  %32 = load i8* (i32, i8**)*, i8* (i32, i8**)** %7, align 8
  %33 = icmp eq i8* (i32, i8**)* %32, @mpcf_trd
  br i1 %33, label %34, label %38

; <label>:34:                                     ; preds = %31
  %35 = load i32, i32* %8, align 4
  %36 = load i8**, i8*** %9, align 8
  %37 = call i8* @mpcf_trd(i32 %35, i8** %36)
  store i8* %37, i8** %5, align 8
  br label %103

; <label>:38:                                     ; preds = %31
  %39 = load i8* (i32, i8**)*, i8* (i32, i8**)** %7, align 8
  %40 = icmp eq i8* (i32, i8**)* %39, @mpcf_fst_free
  br i1 %40, label %41, label %46

; <label>:41:                                     ; preds = %38
  %42 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %43 = load i32, i32* %8, align 4
  %44 = load i8**, i8*** %9, align 8
  %45 = call i8* @mpcf_input_fst_free(%struct.mpc_input_t* %42, i32 %43, i8** %44)
  store i8* %45, i8** %5, align 8
  br label %103

; <label>:46:                                     ; preds = %38
  %47 = load i8* (i32, i8**)*, i8* (i32, i8**)** %7, align 8
  %48 = icmp eq i8* (i32, i8**)* %47, @mpcf_snd_free
  br i1 %48, label %49, label %54

; <label>:49:                                     ; preds = %46
  %50 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %51 = load i32, i32* %8, align 4
  %52 = load i8**, i8*** %9, align 8
  %53 = call i8* @mpcf_input_snd_free(%struct.mpc_input_t* %50, i32 %51, i8** %52)
  store i8* %53, i8** %5, align 8
  br label %103

; <label>:54:                                     ; preds = %46
  %55 = load i8* (i32, i8**)*, i8* (i32, i8**)** %7, align 8
  %56 = icmp eq i8* (i32, i8**)* %55, @mpcf_trd_free
  br i1 %56, label %57, label %62

; <label>:57:                                     ; preds = %54
  %58 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %59 = load i32, i32* %8, align 4
  %60 = load i8**, i8*** %9, align 8
  %61 = call i8* @mpcf_input_trd_free(%struct.mpc_input_t* %58, i32 %59, i8** %60)
  store i8* %61, i8** %5, align 8
  br label %103

; <label>:62:                                     ; preds = %54
  %63 = load i8* (i32, i8**)*, i8* (i32, i8**)** %7, align 8
  %64 = icmp eq i8* (i32, i8**)* %63, @mpcf_strfold
  br i1 %64, label %65, label %70

; <label>:65:                                     ; preds = %62
  %66 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %67 = load i32, i32* %8, align 4
  %68 = load i8**, i8*** %9, align 8
  %69 = call i8* @mpcf_input_strfold(%struct.mpc_input_t* %66, i32 %67, i8** %68)
  store i8* %69, i8** %5, align 8
  br label %103

; <label>:70:                                     ; preds = %62
  %71 = load i8* (i32, i8**)*, i8* (i32, i8**)** %7, align 8
  %72 = icmp eq i8* (i32, i8**)* %71, @mpcf_state_ast
  br i1 %72, label %73, label %78

; <label>:73:                                     ; preds = %70
  %74 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %75 = load i32, i32* %8, align 4
  %76 = load i8**, i8*** %9, align 8
  %77 = call i8* @mpcf_input_state_ast(%struct.mpc_input_t* %74, i32 %75, i8** %76)
  store i8* %77, i8** %5, align 8
  br label %103

; <label>:78:                                     ; preds = %70
  store i32 0, i32* %10, align 4
  br label %79

; <label>:79:                                     ; preds = %95, %78
  %80 = load i32, i32* %10, align 4
  %81 = load i32, i32* %8, align 4
  %82 = icmp slt i32 %80, %81
  br i1 %82, label %83, label %98

; <label>:83:                                     ; preds = %79
  %84 = load %struct.mpc_input_t*, %struct.mpc_input_t** %6, align 8
  %85 = load i8**, i8*** %9, align 8
  %86 = load i32, i32* %10, align 4
  %87 = sext i32 %86 to i64
  %88 = getelementptr inbounds i8*, i8** %85, i64 %87
  %89 = load i8*, i8** %88, align 8
  %90 = call i8* @mpc_export(%struct.mpc_input_t* %84, i8* %89)
  %91 = load i8**, i8*** %9, align 8
  %92 = load i32, i32* %10, align 4
  %93 = sext i32 %92 to i64
  %94 = getelementptr inbounds i8*, i8** %91, i64 %93
  store i8* %90, i8** %94, align 8
  br label %95

; <label>:95:                                     ; preds = %83
  %96 = load i32, i32* %10, align 4
  %97 = add nsw i32 %96, 1
  store i32 %97, i32* %10, align 4
  br label %79

; <label>:98:                                     ; preds = %79
  %99 = load i8* (i32, i8**)*, i8* (i32, i8**)** %7, align 8
  %100 = load i32, i32* %10, align 4
  %101 = load i8**, i8*** %9, align 8
  %102 = call i8* %99(i32 %100, i8** %101)
  store i8* %102, i8** %5, align 8
  br label %103

; <label>:103:                                    ; preds = %98, %73, %65, %57, %49, %41, %34, %27, %20, %13
  %104 = load i8*, i8** %5, align 8
  ret i8* %104
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_free(%struct.mpc_input_t*, i8*) #0 {
  %3 = alloca %struct.mpc_input_t*, align 8
  %4 = alloca i8*, align 8
  %5 = alloca i64, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %3, align 8
  store i8* %1, i8** %4, align 8
  %6 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %7 = load i8*, i8** %4, align 8
  %8 = call i32 @mpc_mem_ptr(%struct.mpc_input_t* %6, i8* %7)
  %9 = icmp ne i32 %8, 0
  br i1 %9, label %12, label %10

; <label>:10:                                     ; preds = %2
  %11 = load i8*, i8** %4, align 8
  call void @free(i8* %11) #5
  br label %26

; <label>:12:                                     ; preds = %2
  %13 = load i8*, i8** %4, align 8
  %14 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %15 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %14, i32 0, i32 15
  %16 = getelementptr inbounds [512 x %struct.mpc_mem_t], [512 x %struct.mpc_mem_t]* %15, i32 0, i32 0
  %17 = bitcast %struct.mpc_mem_t* %16 to i8*
  %18 = ptrtoint i8* %13 to i64
  %19 = ptrtoint i8* %17 to i64
  %20 = sub i64 %18, %19
  %21 = udiv i64 %20, 64
  store i64 %21, i64* %5, align 8
  %22 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %23 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %22, i32 0, i32 14
  %24 = load i64, i64* %5, align 8
  %25 = getelementptr inbounds [512 x i8], [512 x i8]* %23, i64 0, i64 %24
  store i8 0, i8* %25, align 1
  br label %26

; <label>:26:                                     ; preds = %12, %10
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal %struct.mpc_err_t* @mpc_err_many1(%struct.mpc_input_t*, %struct.mpc_err_t*) #0 {
  %3 = alloca %struct.mpc_input_t*, align 8
  %4 = alloca %struct.mpc_err_t*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %3, align 8
  store %struct.mpc_err_t* %1, %struct.mpc_err_t** %4, align 8
  %5 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %6 = load %struct.mpc_err_t*, %struct.mpc_err_t** %4, align 8
  %7 = call %struct.mpc_err_t* @mpc_err_repeat(%struct.mpc_input_t* %5, %struct.mpc_err_t* %6, i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.104, i32 0, i32 0))
  ret %struct.mpc_err_t* %7
}

; Function Attrs: noinline nounwind optnone uwtable
define internal %struct.mpc_err_t* @mpc_err_count(%struct.mpc_input_t*, %struct.mpc_err_t*, i32) #0 {
  %4 = alloca %struct.mpc_input_t*, align 8
  %5 = alloca %struct.mpc_err_t*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %struct.mpc_err_t*, align 8
  %8 = alloca i32, align 4
  %9 = alloca i8*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %4, align 8
  store %struct.mpc_err_t* %1, %struct.mpc_err_t** %5, align 8
  store i32 %2, i32* %6, align 4
  %10 = load i32, i32* %6, align 4
  %11 = sdiv i32 %10, 10
  %12 = add nsw i32 %11, 1
  store i32 %12, i32* %8, align 4
  %13 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %14 = load i32, i32* %8, align 4
  %15 = sext i32 %14 to i64
  %16 = add i64 %15, 4
  %17 = add i64 %16, 1
  %18 = call i8* @mpc_malloc(%struct.mpc_input_t* %13, i64 %17)
  store i8* %18, i8** %9, align 8
  %19 = load i8*, i8** %9, align 8
  %20 = load i32, i32* %6, align 4
  %21 = call i32 (i8*, i8*, ...) @sprintf(i8* %19, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.107, i32 0, i32 0), i32 %20) #5
  %22 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %23 = load %struct.mpc_err_t*, %struct.mpc_err_t** %5, align 8
  %24 = load i8*, i8** %9, align 8
  %25 = call %struct.mpc_err_t* @mpc_err_repeat(%struct.mpc_input_t* %22, %struct.mpc_err_t* %23, i8* %24)
  store %struct.mpc_err_t* %25, %struct.mpc_err_t** %7, align 8
  %26 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %27 = load i8*, i8** %9, align 8
  call void @mpc_free(%struct.mpc_input_t* %26, i8* %27)
  %28 = load %struct.mpc_err_t*, %struct.mpc_err_t** %7, align 8
  ret %struct.mpc_err_t* %28
}

; Function Attrs: noinline nounwind optnone uwtable
define internal signext i8 @mpc_input_getc(%struct.mpc_input_t*) #0 {
  %2 = alloca i8, align 1
  %3 = alloca %struct.mpc_input_t*, align 8
  %4 = alloca i8, align 1
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %3, align 8
  store i8 0, i8* %4, align 1
  %5 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %6 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %5, i32 0, i32 0
  %7 = load i32, i32* %6, align 8
  switch i32 %7, label %57 [
    i32 0, label %8
    i32 1, label %18
    i32 2, label %25
  ]

; <label>:8:                                      ; preds = %1
  %9 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %10 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %9, i32 0, i32 3
  %11 = load i8*, i8** %10, align 8
  %12 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %13 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %12, i32 0, i32 2
  %14 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %13, i32 0, i32 0
  %15 = load i64, i64* %14, align 8
  %16 = getelementptr inbounds i8, i8* %11, i64 %15
  %17 = load i8, i8* %16, align 1
  store i8 %17, i8* %2, align 1
  br label %59

; <label>:18:                                     ; preds = %1
  %19 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %20 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %19, i32 0, i32 5
  %21 = load %struct._IO_FILE*, %struct._IO_FILE** %20, align 8
  %22 = call i32 @fgetc(%struct._IO_FILE* %21)
  %23 = trunc i32 %22 to i8
  store i8 %23, i8* %4, align 1
  %24 = load i8, i8* %4, align 1
  store i8 %24, i8* %2, align 1
  br label %59

; <label>:25:                                     ; preds = %1
  %26 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %27 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %26, i32 0, i32 4
  %28 = load i8*, i8** %27, align 8
  %29 = icmp ne i8* %28, null
  br i1 %29, label %37, label %30

; <label>:30:                                     ; preds = %25
  %31 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %32 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %31, i32 0, i32 5
  %33 = load %struct._IO_FILE*, %struct._IO_FILE** %32, align 8
  %34 = call i32 @_IO_getc(%struct._IO_FILE* %33)
  %35 = trunc i32 %34 to i8
  store i8 %35, i8* %4, align 1
  %36 = load i8, i8* %4, align 1
  store i8 %36, i8* %2, align 1
  br label %59

; <label>:37:                                     ; preds = %25
  %38 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %39 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %38, i32 0, i32 4
  %40 = load i8*, i8** %39, align 8
  %41 = icmp ne i8* %40, null
  br i1 %41, label %42, label %50

; <label>:42:                                     ; preds = %37
  %43 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %44 = call i32 @mpc_input_buffer_in_range(%struct.mpc_input_t* %43)
  %45 = icmp ne i32 %44, 0
  br i1 %45, label %46, label %50

; <label>:46:                                     ; preds = %42
  %47 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %48 = call signext i8 @mpc_input_buffer_get(%struct.mpc_input_t* %47)
  store i8 %48, i8* %4, align 1
  %49 = load i8, i8* %4, align 1
  store i8 %49, i8* %2, align 1
  br label %59

; <label>:50:                                     ; preds = %42, %37
  %51 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %52 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %51, i32 0, i32 5
  %53 = load %struct._IO_FILE*, %struct._IO_FILE** %52, align 8
  %54 = call i32 @_IO_getc(%struct._IO_FILE* %53)
  %55 = trunc i32 %54 to i8
  store i8 %55, i8* %4, align 1
  %56 = load i8, i8* %4, align 1
  store i8 %56, i8* %2, align 1
  br label %59

; <label>:57:                                     ; preds = %1
  %58 = load i8, i8* %4, align 1
  store i8 %58, i8* %2, align 1
  br label %59

; <label>:59:                                     ; preds = %57, %50, %46, %30, %18, %8
  %60 = load i8, i8* %2, align 1
  ret i8 %60
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @mpc_input_terminated(%struct.mpc_input_t*) #0 {
  %2 = alloca i32, align 4
  %3 = alloca %struct.mpc_input_t*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %3, align 8
  %4 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %5 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %4, i32 0, i32 0
  %6 = load i32, i32* %5, align 8
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %8, label %19

; <label>:8:                                      ; preds = %1
  %9 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %10 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %9, i32 0, i32 2
  %11 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %10, i32 0, i32 0
  %12 = load i64, i64* %11, align 8
  %13 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %14 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %13, i32 0, i32 3
  %15 = load i8*, i8** %14, align 8
  %16 = call i64 @strlen(i8* %15) #7
  %17 = icmp eq i64 %12, %16
  br i1 %17, label %18, label %19

; <label>:18:                                     ; preds = %8
  store i32 1, i32* %2, align 4
  br label %44

; <label>:19:                                     ; preds = %8, %1
  %20 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %21 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %20, i32 0, i32 0
  %22 = load i32, i32* %21, align 8
  %23 = icmp eq i32 %22, 1
  br i1 %23, label %24, label %31

; <label>:24:                                     ; preds = %19
  %25 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %26 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %25, i32 0, i32 5
  %27 = load %struct._IO_FILE*, %struct._IO_FILE** %26, align 8
  %28 = call i32 @feof(%struct._IO_FILE* %27) #5
  %29 = icmp ne i32 %28, 0
  br i1 %29, label %30, label %31

; <label>:30:                                     ; preds = %24
  store i32 1, i32* %2, align 4
  br label %44

; <label>:31:                                     ; preds = %24, %19
  %32 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %33 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %32, i32 0, i32 0
  %34 = load i32, i32* %33, align 8
  %35 = icmp eq i32 %34, 2
  br i1 %35, label %36, label %43

; <label>:36:                                     ; preds = %31
  %37 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %38 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %37, i32 0, i32 5
  %39 = load %struct._IO_FILE*, %struct._IO_FILE** %38, align 8
  %40 = call i32 @feof(%struct._IO_FILE* %39) #5
  %41 = icmp ne i32 %40, 0
  br i1 %41, label %42, label %43

; <label>:42:                                     ; preds = %36
  store i32 1, i32* %2, align 4
  br label %44

; <label>:43:                                     ; preds = %36, %31
  store i32 0, i32* %2, align 4
  br label %44

; <label>:44:                                     ; preds = %43, %42, %30, %18
  %45 = load i32, i32* %2, align 4
  ret i32 %45
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @mpc_input_success(%struct.mpc_input_t*, i8 signext, i8**) #0 {
  %4 = alloca %struct.mpc_input_t*, align 8
  %5 = alloca i8, align 1
  %6 = alloca i8**, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %4, align 8
  store i8 %1, i8* %5, align 1
  store i8** %2, i8*** %6, align 8
  %7 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %8 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %7, i32 0, i32 0
  %9 = load i32, i32* %8, align 8
  %10 = icmp eq i32 %9, 2
  br i1 %10, label %11, label %51

; <label>:11:                                     ; preds = %3
  %12 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %13 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %12, i32 0, i32 4
  %14 = load i8*, i8** %13, align 8
  %15 = icmp ne i8* %14, null
  br i1 %15, label %16, label %51

; <label>:16:                                     ; preds = %11
  %17 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %18 = call i32 @mpc_input_buffer_in_range(%struct.mpc_input_t* %17)
  %19 = icmp ne i32 %18, 0
  br i1 %19, label %51, label %20

; <label>:20:                                     ; preds = %16
  %21 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %22 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %21, i32 0, i32 4
  %23 = load i8*, i8** %22, align 8
  %24 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %25 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %24, i32 0, i32 4
  %26 = load i8*, i8** %25, align 8
  %27 = call i64 @strlen(i8* %26) #7
  %28 = add i64 %27, 2
  %29 = call i8* @realloc(i8* %23, i64 %28) #5
  %30 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %31 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %30, i32 0, i32 4
  store i8* %29, i8** %31, align 8
  %32 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %33 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %32, i32 0, i32 4
  %34 = load i8*, i8** %33, align 8
  %35 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %36 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %35, i32 0, i32 4
  %37 = load i8*, i8** %36, align 8
  %38 = call i64 @strlen(i8* %37) #7
  %39 = add i64 %38, 1
  %40 = getelementptr inbounds i8, i8* %34, i64 %39
  store i8 0, i8* %40, align 1
  %41 = load i8, i8* %5, align 1
  %42 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %43 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %42, i32 0, i32 4
  %44 = load i8*, i8** %43, align 8
  %45 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %46 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %45, i32 0, i32 4
  %47 = load i8*, i8** %46, align 8
  %48 = call i64 @strlen(i8* %47) #7
  %49 = add i64 %48, 0
  %50 = getelementptr inbounds i8, i8* %44, i64 %49
  store i8 %41, i8* %50, align 1
  br label %51

; <label>:51:                                     ; preds = %20, %16, %11, %3
  %52 = load i8, i8* %5, align 1
  %53 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %54 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %53, i32 0, i32 12
  store i8 %52, i8* %54, align 8
  %55 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %56 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %55, i32 0, i32 2
  %57 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %56, i32 0, i32 0
  %58 = load i64, i64* %57, align 8
  %59 = add nsw i64 %58, 1
  store i64 %59, i64* %57, align 8
  %60 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %61 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %60, i32 0, i32 2
  %62 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %61, i32 0, i32 2
  %63 = load i64, i64* %62, align 8
  %64 = add nsw i64 %63, 1
  store i64 %64, i64* %62, align 8
  %65 = load i8, i8* %5, align 1
  %66 = sext i8 %65 to i32
  %67 = icmp eq i32 %66, 10
  br i1 %67, label %68, label %77

; <label>:68:                                     ; preds = %51
  %69 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %70 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %69, i32 0, i32 2
  %71 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %70, i32 0, i32 2
  store i64 0, i64* %71, align 8
  %72 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %73 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %72, i32 0, i32 2
  %74 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %73, i32 0, i32 1
  %75 = load i64, i64* %74, align 8
  %76 = add nsw i64 %75, 1
  store i64 %76, i64* %74, align 8
  br label %77

; <label>:77:                                     ; preds = %68, %51
  %78 = load i8**, i8*** %6, align 8
  %79 = icmp ne i8** %78, null
  br i1 %79, label %80, label %91

; <label>:80:                                     ; preds = %77
  %81 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %82 = call i8* @mpc_malloc(%struct.mpc_input_t* %81, i64 2)
  %83 = load i8**, i8*** %6, align 8
  store i8* %82, i8** %83, align 8
  %84 = load i8, i8* %5, align 1
  %85 = load i8**, i8*** %6, align 8
  %86 = load i8*, i8** %85, align 8
  %87 = getelementptr inbounds i8, i8* %86, i64 0
  store i8 %84, i8* %87, align 1
  %88 = load i8**, i8*** %6, align 8
  %89 = load i8*, i8** %88, align 8
  %90 = getelementptr inbounds i8, i8* %89, i64 1
  store i8 0, i8* %90, align 1
  br label %91

; <label>:91:                                     ; preds = %80, %77
  ret i32 1
}

declare i32 @fgetc(%struct._IO_FILE*) #2

declare i32 @_IO_getc(%struct._IO_FILE*) #2

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @mpc_input_buffer_in_range(%struct.mpc_input_t*) #0 {
  %2 = alloca %struct.mpc_input_t*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %2, align 8
  %3 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %4 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %3, i32 0, i32 2
  %5 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %4, i32 0, i32 0
  %6 = load i64, i64* %5, align 8
  %7 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %8 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %7, i32 0, i32 4
  %9 = load i8*, i8** %8, align 8
  %10 = call i64 @strlen(i8* %9) #7
  %11 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %12 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %11, i32 0, i32 10
  %13 = load %struct.mpc_state_t*, %struct.mpc_state_t** %12, align 8
  %14 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %13, i64 0
  %15 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %14, i32 0, i32 0
  %16 = load i64, i64* %15, align 8
  %17 = add i64 %10, %16
  %18 = icmp slt i64 %6, %17
  %19 = zext i1 %18 to i32
  ret i32 %19
}

; Function Attrs: noinline nounwind optnone uwtable
define internal signext i8 @mpc_input_buffer_get(%struct.mpc_input_t*) #0 {
  %2 = alloca %struct.mpc_input_t*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %2, align 8
  %3 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %4 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %3, i32 0, i32 4
  %5 = load i8*, i8** %4, align 8
  %6 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %7 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %6, i32 0, i32 2
  %8 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %7, i32 0, i32 0
  %9 = load i64, i64* %8, align 8
  %10 = load %struct.mpc_input_t*, %struct.mpc_input_t** %2, align 8
  %11 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %10, i32 0, i32 10
  %12 = load %struct.mpc_state_t*, %struct.mpc_state_t** %11, align 8
  %13 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %12, i64 0
  %14 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %13, i32 0, i32 0
  %15 = load i64, i64* %14, align 8
  %16 = sub nsw i64 %9, %15
  %17 = getelementptr inbounds i8, i8* %5, i64 %16
  %18 = load i8, i8* %17, align 1
  ret i8 %18
}

; Function Attrs: nounwind
declare i32 @feof(%struct._IO_FILE*) #1

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @mpc_input_failure(%struct.mpc_input_t*, i8 signext) #0 {
  %3 = alloca %struct.mpc_input_t*, align 8
  %4 = alloca i8, align 1
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %3, align 8
  store i8 %1, i8* %4, align 1
  %5 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %6 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %5, i32 0, i32 0
  %7 = load i32, i32* %6, align 8
  switch i32 %7, label %44 [
    i32 0, label %8
    i32 1, label %9
    i32 2, label %14
  ]

; <label>:8:                                      ; preds = %2
  br label %45

; <label>:9:                                      ; preds = %2
  %10 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %11 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %10, i32 0, i32 5
  %12 = load %struct._IO_FILE*, %struct._IO_FILE** %11, align 8
  %13 = call i32 @fseek(%struct._IO_FILE* %12, i64 -1, i32 1)
  br label %45

; <label>:14:                                     ; preds = %2
  %15 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %16 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %15, i32 0, i32 4
  %17 = load i8*, i8** %16, align 8
  %18 = icmp ne i8* %17, null
  br i1 %18, label %26, label %19

; <label>:19:                                     ; preds = %14
  %20 = load i8, i8* %4, align 1
  %21 = sext i8 %20 to i32
  %22 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %23 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %22, i32 0, i32 5
  %24 = load %struct._IO_FILE*, %struct._IO_FILE** %23, align 8
  %25 = call i32 @ungetc(i32 %21, %struct._IO_FILE* %24)
  br label %45

; <label>:26:                                     ; preds = %14
  %27 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %28 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %27, i32 0, i32 4
  %29 = load i8*, i8** %28, align 8
  %30 = icmp ne i8* %29, null
  br i1 %30, label %31, label %36

; <label>:31:                                     ; preds = %26
  %32 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %33 = call i32 @mpc_input_buffer_in_range(%struct.mpc_input_t* %32)
  %34 = icmp ne i32 %33, 0
  br i1 %34, label %35, label %36

; <label>:35:                                     ; preds = %31
  br label %45

; <label>:36:                                     ; preds = %31, %26
  %37 = load i8, i8* %4, align 1
  %38 = sext i8 %37 to i32
  %39 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %40 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %39, i32 0, i32 5
  %41 = load %struct._IO_FILE*, %struct._IO_FILE** %40, align 8
  %42 = call i32 @ungetc(i32 %38, %struct._IO_FILE* %41)
  br label %43

; <label>:43:                                     ; preds = %36
  br label %44

; <label>:44:                                     ; preds = %2, %43
  br label %45

; <label>:45:                                     ; preds = %44, %35, %19, %9, %8
  ret i32 0
}

declare i32 @fseek(%struct._IO_FILE*, i64, i32) #2

declare i32 @ungetc(i32, %struct._IO_FILE*) #2

; Function Attrs: nounwind readonly
declare i8* @strchr(i8*, i32) #3

; Function Attrs: noinline nounwind optnone uwtable
define internal signext i8 @mpc_input_peekc(%struct.mpc_input_t*) #0 {
  %2 = alloca i8, align 1
  %3 = alloca %struct.mpc_input_t*, align 8
  %4 = alloca i8, align 1
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %3, align 8
  store i8 0, i8* %4, align 1
  %5 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %6 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %5, i32 0, i32 0
  %7 = load i32, i32* %6, align 8
  switch i32 %7, label %93 [
    i32 0, label %8
    i32 1, label %18
    i32 2, label %36
  ]

; <label>:8:                                      ; preds = %1
  %9 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %10 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %9, i32 0, i32 3
  %11 = load i8*, i8** %10, align 8
  %12 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %13 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %12, i32 0, i32 2
  %14 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %13, i32 0, i32 0
  %15 = load i64, i64* %14, align 8
  %16 = getelementptr inbounds i8, i8* %11, i64 %15
  %17 = load i8, i8* %16, align 1
  store i8 %17, i8* %2, align 1
  br label %95

; <label>:18:                                     ; preds = %1
  %19 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %20 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %19, i32 0, i32 5
  %21 = load %struct._IO_FILE*, %struct._IO_FILE** %20, align 8
  %22 = call i32 @fgetc(%struct._IO_FILE* %21)
  %23 = trunc i32 %22 to i8
  store i8 %23, i8* %4, align 1
  %24 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %25 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %24, i32 0, i32 5
  %26 = load %struct._IO_FILE*, %struct._IO_FILE** %25, align 8
  %27 = call i32 @feof(%struct._IO_FILE* %26) #5
  %28 = icmp ne i32 %27, 0
  br i1 %28, label %29, label %30

; <label>:29:                                     ; preds = %18
  store i8 0, i8* %2, align 1
  br label %95

; <label>:30:                                     ; preds = %18
  %31 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %32 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %31, i32 0, i32 5
  %33 = load %struct._IO_FILE*, %struct._IO_FILE** %32, align 8
  %34 = call i32 @fseek(%struct._IO_FILE* %33, i64 -1, i32 1)
  %35 = load i8, i8* %4, align 1
  store i8 %35, i8* %2, align 1
  br label %95

; <label>:36:                                     ; preds = %1
  %37 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %38 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %37, i32 0, i32 4
  %39 = load i8*, i8** %38, align 8
  %40 = icmp ne i8* %39, null
  br i1 %40, label %61, label %41

; <label>:41:                                     ; preds = %36
  %42 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %43 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %42, i32 0, i32 5
  %44 = load %struct._IO_FILE*, %struct._IO_FILE** %43, align 8
  %45 = call i32 @_IO_getc(%struct._IO_FILE* %44)
  %46 = trunc i32 %45 to i8
  store i8 %46, i8* %4, align 1
  %47 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %48 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %47, i32 0, i32 5
  %49 = load %struct._IO_FILE*, %struct._IO_FILE** %48, align 8
  %50 = call i32 @feof(%struct._IO_FILE* %49) #5
  %51 = icmp ne i32 %50, 0
  br i1 %51, label %52, label %53

; <label>:52:                                     ; preds = %41
  store i8 0, i8* %2, align 1
  br label %95

; <label>:53:                                     ; preds = %41
  %54 = load i8, i8* %4, align 1
  %55 = sext i8 %54 to i32
  %56 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %57 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %56, i32 0, i32 5
  %58 = load %struct._IO_FILE*, %struct._IO_FILE** %57, align 8
  %59 = call i32 @ungetc(i32 %55, %struct._IO_FILE* %58)
  %60 = load i8, i8* %4, align 1
  store i8 %60, i8* %2, align 1
  br label %95

; <label>:61:                                     ; preds = %36
  %62 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %63 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %62, i32 0, i32 4
  %64 = load i8*, i8** %63, align 8
  %65 = icmp ne i8* %64, null
  br i1 %65, label %66, label %73

; <label>:66:                                     ; preds = %61
  %67 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %68 = call i32 @mpc_input_buffer_in_range(%struct.mpc_input_t* %67)
  %69 = icmp ne i32 %68, 0
  br i1 %69, label %70, label %73

; <label>:70:                                     ; preds = %66
  %71 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %72 = call signext i8 @mpc_input_buffer_get(%struct.mpc_input_t* %71)
  store i8 %72, i8* %2, align 1
  br label %95

; <label>:73:                                     ; preds = %66, %61
  %74 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %75 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %74, i32 0, i32 5
  %76 = load %struct._IO_FILE*, %struct._IO_FILE** %75, align 8
  %77 = call i32 @_IO_getc(%struct._IO_FILE* %76)
  %78 = trunc i32 %77 to i8
  store i8 %78, i8* %4, align 1
  %79 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %80 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %79, i32 0, i32 5
  %81 = load %struct._IO_FILE*, %struct._IO_FILE** %80, align 8
  %82 = call i32 @feof(%struct._IO_FILE* %81) #5
  %83 = icmp ne i32 %82, 0
  br i1 %83, label %84, label %85

; <label>:84:                                     ; preds = %73
  store i8 0, i8* %2, align 1
  br label %95

; <label>:85:                                     ; preds = %73
  %86 = load i8, i8* %4, align 1
  %87 = sext i8 %86 to i32
  %88 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %89 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %88, i32 0, i32 5
  %90 = load %struct._IO_FILE*, %struct._IO_FILE** %89, align 8
  %91 = call i32 @ungetc(i32 %87, %struct._IO_FILE* %90)
  %92 = load i8, i8* %4, align 1
  store i8 %92, i8* %2, align 1
  br label %95

; <label>:93:                                     ; preds = %1
  %94 = load i8, i8* %4, align 1
  store i8 %94, i8* %2, align 1
  br label %95

; <label>:95:                                     ; preds = %93, %85, %84, %70, %53, %52, %30, %29, %8
  %96 = load i8, i8* %2, align 1
  ret i8 %96
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcf_input_free(%struct.mpc_input_t*, i8*) #0 {
  %3 = alloca %struct.mpc_input_t*, align 8
  %4 = alloca i8*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %3, align 8
  store i8* %1, i8** %4, align 8
  %5 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %6 = load i8*, i8** %4, align 8
  call void @mpc_free(%struct.mpc_input_t* %5, i8* %6)
  ret i8* null
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcf_input_str_ast(%struct.mpc_input_t*, i8*) #0 {
  %3 = alloca %struct.mpc_input_t*, align 8
  %4 = alloca i8*, align 8
  %5 = alloca %struct.mpc_ast_t*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %3, align 8
  store i8* %1, i8** %4, align 8
  %6 = load i8*, i8** %4, align 8
  %7 = call %struct.mpc_ast_t* @mpc_ast_new(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @.str.81, i32 0, i32 0), i8* %6)
  store %struct.mpc_ast_t* %7, %struct.mpc_ast_t** %5, align 8
  %8 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %9 = load i8*, i8** %4, align 8
  call void @mpc_free(%struct.mpc_input_t* %8, i8* %9)
  %10 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %5, align 8
  %11 = bitcast %struct.mpc_ast_t* %10 to i8*
  ret i8* %11
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @mpc_mem_ptr(%struct.mpc_input_t*, i8*) #0 {
  %3 = alloca %struct.mpc_input_t*, align 8
  %4 = alloca i8*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %3, align 8
  store i8* %1, i8** %4, align 8
  %5 = load i8*, i8** %4, align 8
  %6 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %7 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %6, i32 0, i32 15
  %8 = getelementptr inbounds [512 x %struct.mpc_mem_t], [512 x %struct.mpc_mem_t]* %7, i32 0, i32 0
  %9 = bitcast %struct.mpc_mem_t* %8 to i8*
  %10 = icmp uge i8* %5, %9
  br i1 %10, label %11, label %19

; <label>:11:                                     ; preds = %2
  %12 = load i8*, i8** %4, align 8
  %13 = load %struct.mpc_input_t*, %struct.mpc_input_t** %3, align 8
  %14 = getelementptr inbounds %struct.mpc_input_t, %struct.mpc_input_t* %13, i32 0, i32 15
  %15 = getelementptr inbounds [512 x %struct.mpc_mem_t], [512 x %struct.mpc_mem_t]* %14, i32 0, i32 0
  %16 = bitcast %struct.mpc_mem_t* %15 to i8*
  %17 = getelementptr inbounds i8, i8* %16, i64 32768
  %18 = icmp ult i8* %12, %17
  br label %19

; <label>:19:                                     ; preds = %11, %2
  %20 = phi i1 [ false, %2 ], [ %18, %11 ]
  %21 = zext i1 %20 to i32
  ret i32 %21
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcf_input_fst_free(%struct.mpc_input_t*, i32, i8**) #0 {
  %4 = alloca %struct.mpc_input_t*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i8**, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %4, align 8
  store i32 %1, i32* %5, align 4
  store i8** %2, i8*** %6, align 8
  %7 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %8 = load i32, i32* %5, align 4
  %9 = load i8**, i8*** %6, align 8
  %10 = call i8* @mpcf_input_nth_free(%struct.mpc_input_t* %7, i32 %8, i8** %9, i32 0)
  ret i8* %10
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcf_input_snd_free(%struct.mpc_input_t*, i32, i8**) #0 {
  %4 = alloca %struct.mpc_input_t*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i8**, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %4, align 8
  store i32 %1, i32* %5, align 4
  store i8** %2, i8*** %6, align 8
  %7 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %8 = load i32, i32* %5, align 4
  %9 = load i8**, i8*** %6, align 8
  %10 = call i8* @mpcf_input_nth_free(%struct.mpc_input_t* %7, i32 %8, i8** %9, i32 1)
  ret i8* %10
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcf_input_trd_free(%struct.mpc_input_t*, i32, i8**) #0 {
  %4 = alloca %struct.mpc_input_t*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i8**, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %4, align 8
  store i32 %1, i32* %5, align 4
  store i8** %2, i8*** %6, align 8
  %7 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %8 = load i32, i32* %5, align 4
  %9 = load i8**, i8*** %6, align 8
  %10 = call i8* @mpcf_input_nth_free(%struct.mpc_input_t* %7, i32 %8, i8** %9, i32 2)
  ret i8* %10
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcf_input_strfold(%struct.mpc_input_t*, i32, i8**) #0 {
  %4 = alloca i8*, align 8
  %5 = alloca %struct.mpc_input_t*, align 8
  %6 = alloca i32, align 4
  %7 = alloca i8**, align 8
  %8 = alloca i32, align 4
  %9 = alloca i64, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %5, align 8
  store i32 %1, i32* %6, align 4
  store i8** %2, i8*** %7, align 8
  store i64 0, i64* %9, align 8
  %10 = load i32, i32* %6, align 4
  %11 = icmp eq i32 %10, 0
  br i1 %11, label %12, label %15

; <label>:12:                                     ; preds = %3
  %13 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %14 = call i8* @mpc_calloc(%struct.mpc_input_t* %13, i64 1, i64 1)
  store i8* %14, i8** %4, align 8
  br label %69

; <label>:15:                                     ; preds = %3
  store i32 0, i32* %8, align 4
  br label %16

; <label>:16:                                     ; preds = %29, %15
  %17 = load i32, i32* %8, align 4
  %18 = load i32, i32* %6, align 4
  %19 = icmp slt i32 %17, %18
  br i1 %19, label %20, label %32

; <label>:20:                                     ; preds = %16
  %21 = load i8**, i8*** %7, align 8
  %22 = load i32, i32* %8, align 4
  %23 = sext i32 %22 to i64
  %24 = getelementptr inbounds i8*, i8** %21, i64 %23
  %25 = load i8*, i8** %24, align 8
  %26 = call i64 @strlen(i8* %25) #7
  %27 = load i64, i64* %9, align 8
  %28 = add i64 %27, %26
  store i64 %28, i64* %9, align 8
  br label %29

; <label>:29:                                     ; preds = %20
  %30 = load i32, i32* %8, align 4
  %31 = add nsw i32 %30, 1
  store i32 %31, i32* %8, align 4
  br label %16

; <label>:32:                                     ; preds = %16
  %33 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %34 = load i8**, i8*** %7, align 8
  %35 = getelementptr inbounds i8*, i8** %34, i64 0
  %36 = load i8*, i8** %35, align 8
  %37 = load i64, i64* %9, align 8
  %38 = add i64 %37, 1
  %39 = call i8* @mpc_realloc(%struct.mpc_input_t* %33, i8* %36, i64 %38)
  %40 = load i8**, i8*** %7, align 8
  %41 = getelementptr inbounds i8*, i8** %40, i64 0
  store i8* %39, i8** %41, align 8
  store i32 1, i32* %8, align 4
  br label %42

; <label>:42:                                     ; preds = %62, %32
  %43 = load i32, i32* %8, align 4
  %44 = load i32, i32* %6, align 4
  %45 = icmp slt i32 %43, %44
  br i1 %45, label %46, label %65

; <label>:46:                                     ; preds = %42
  %47 = load i8**, i8*** %7, align 8
  %48 = getelementptr inbounds i8*, i8** %47, i64 0
  %49 = load i8*, i8** %48, align 8
  %50 = load i8**, i8*** %7, align 8
  %51 = load i32, i32* %8, align 4
  %52 = sext i32 %51 to i64
  %53 = getelementptr inbounds i8*, i8** %50, i64 %52
  %54 = load i8*, i8** %53, align 8
  %55 = call i8* @strcat(i8* %49, i8* %54) #5
  %56 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %57 = load i8**, i8*** %7, align 8
  %58 = load i32, i32* %8, align 4
  %59 = sext i32 %58 to i64
  %60 = getelementptr inbounds i8*, i8** %57, i64 %59
  %61 = load i8*, i8** %60, align 8
  call void @mpc_free(%struct.mpc_input_t* %56, i8* %61)
  br label %62

; <label>:62:                                     ; preds = %46
  %63 = load i32, i32* %8, align 4
  %64 = add nsw i32 %63, 1
  store i32 %64, i32* %8, align 4
  br label %42

; <label>:65:                                     ; preds = %42
  %66 = load i8**, i8*** %7, align 8
  %67 = getelementptr inbounds i8*, i8** %66, i64 0
  %68 = load i8*, i8** %67, align 8
  store i8* %68, i8** %4, align 8
  br label %69

; <label>:69:                                     ; preds = %65, %12
  %70 = load i8*, i8** %4, align 8
  ret i8* %70
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcf_input_state_ast(%struct.mpc_input_t*, i32, i8**) #0 {
  %4 = alloca %struct.mpc_input_t*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i8**, align 8
  %7 = alloca %struct.mpc_state_t*, align 8
  %8 = alloca %struct.mpc_ast_t*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %4, align 8
  store i32 %1, i32* %5, align 4
  store i8** %2, i8*** %6, align 8
  %9 = load i8**, i8*** %6, align 8
  %10 = bitcast i8** %9 to %struct.mpc_state_t**
  %11 = getelementptr inbounds %struct.mpc_state_t*, %struct.mpc_state_t** %10, i64 0
  %12 = load %struct.mpc_state_t*, %struct.mpc_state_t** %11, align 8
  store %struct.mpc_state_t* %12, %struct.mpc_state_t** %7, align 8
  %13 = load i8**, i8*** %6, align 8
  %14 = bitcast i8** %13 to %struct.mpc_ast_t**
  %15 = getelementptr inbounds %struct.mpc_ast_t*, %struct.mpc_ast_t** %14, i64 1
  %16 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %15, align 8
  store %struct.mpc_ast_t* %16, %struct.mpc_ast_t** %8, align 8
  %17 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %8, align 8
  %18 = load %struct.mpc_state_t*, %struct.mpc_state_t** %7, align 8
  %19 = call %struct.mpc_ast_t* @mpc_ast_state(%struct.mpc_ast_t* %17, %struct.mpc_state_t* byval align 8 %18)
  store %struct.mpc_ast_t* %19, %struct.mpc_ast_t** %8, align 8
  %20 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %21 = load %struct.mpc_state_t*, %struct.mpc_state_t** %7, align 8
  %22 = bitcast %struct.mpc_state_t* %21 to i8*
  call void @mpc_free(%struct.mpc_input_t* %20, i8* %22)
  %23 = load i32, i32* %5, align 4
  %24 = load %struct.mpc_ast_t*, %struct.mpc_ast_t** %8, align 8
  %25 = bitcast %struct.mpc_ast_t* %24 to i8*
  ret i8* %25
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpcf_input_nth_free(%struct.mpc_input_t*, i32, i8**, i32) #0 {
  %5 = alloca %struct.mpc_input_t*, align 8
  %6 = alloca i32, align 4
  %7 = alloca i8**, align 8
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %5, align 8
  store i32 %1, i32* %6, align 4
  store i8** %2, i8*** %7, align 8
  store i32 %3, i32* %8, align 4
  store i32 0, i32* %9, align 4
  br label %10

; <label>:10:                                     ; preds = %26, %4
  %11 = load i32, i32* %9, align 4
  %12 = load i32, i32* %6, align 4
  %13 = icmp slt i32 %11, %12
  br i1 %13, label %14, label %29

; <label>:14:                                     ; preds = %10
  %15 = load i32, i32* %9, align 4
  %16 = load i32, i32* %8, align 4
  %17 = icmp ne i32 %15, %16
  br i1 %17, label %18, label %25

; <label>:18:                                     ; preds = %14
  %19 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %20 = load i8**, i8*** %7, align 8
  %21 = load i32, i32* %9, align 4
  %22 = sext i32 %21 to i64
  %23 = getelementptr inbounds i8*, i8** %20, i64 %22
  %24 = load i8*, i8** %23, align 8
  call void @mpc_free(%struct.mpc_input_t* %19, i8* %24)
  br label %25

; <label>:25:                                     ; preds = %18, %14
  br label %26

; <label>:26:                                     ; preds = %25
  %27 = load i32, i32* %9, align 4
  %28 = add nsw i32 %27, 1
  store i32 %28, i32* %9, align 4
  br label %10

; <label>:29:                                     ; preds = %10
  %30 = load i8**, i8*** %7, align 8
  %31 = load i32, i32* %8, align 4
  %32 = sext i32 %31 to i64
  %33 = getelementptr inbounds i8*, i8** %30, i64 %32
  %34 = load i8*, i8** %33, align 8
  ret i8* %34
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpc_calloc(%struct.mpc_input_t*, i64, i64) #0 {
  %4 = alloca %struct.mpc_input_t*, align 8
  %5 = alloca i64, align 8
  %6 = alloca i64, align 8
  %7 = alloca i8*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %4, align 8
  store i64 %1, i64* %5, align 8
  store i64 %2, i64* %6, align 8
  %8 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %9 = load i64, i64* %5, align 8
  %10 = load i64, i64* %6, align 8
  %11 = mul i64 %9, %10
  %12 = call i8* @mpc_malloc(%struct.mpc_input_t* %8, i64 %11)
  store i8* %12, i8** %7, align 8
  %13 = load i8*, i8** %7, align 8
  %14 = load i64, i64* %5, align 8
  %15 = load i64, i64* %6, align 8
  %16 = mul i64 %14, %15
  call void @llvm.memset.p0i8.i64(i8* %13, i8 0, i64 %16, i32 1, i1 false)
  %17 = load i8*, i8** %7, align 8
  ret i8* %17
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #4

; Function Attrs: noinline nounwind optnone uwtable
define internal %struct.mpc_err_t* @mpc_err_repeat(%struct.mpc_input_t*, %struct.mpc_err_t*, i8*) #0 {
  %4 = alloca %struct.mpc_err_t*, align 8
  %5 = alloca %struct.mpc_input_t*, align 8
  %6 = alloca %struct.mpc_err_t*, align 8
  %7 = alloca i8*, align 8
  %8 = alloca i32, align 4
  %9 = alloca i64, align 8
  %10 = alloca i8*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %5, align 8
  store %struct.mpc_err_t* %1, %struct.mpc_err_t** %6, align 8
  store i8* %2, i8** %7, align 8
  store i32 0, i32* %8, align 4
  store i64 0, i64* %9, align 8
  store i8* null, i8** %10, align 8
  %11 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %12 = icmp ne %struct.mpc_err_t* %11, null
  br i1 %12, label %14, label %13

; <label>:13:                                     ; preds = %3
  store %struct.mpc_err_t* null, %struct.mpc_err_t** %4, align 8
  br label %245

; <label>:14:                                     ; preds = %3
  %15 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %16 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %15, i32 0, i32 1
  %17 = load i32, i32* %16, align 8
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %19, label %44

; <label>:19:                                     ; preds = %14
  %20 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %21 = call i8* @mpc_calloc(%struct.mpc_input_t* %20, i64 1, i64 1)
  store i8* %21, i8** %10, align 8
  %22 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %23 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %22, i32 0, i32 1
  store i32 1, i32* %23, align 8
  %24 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %25 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %26 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %25, i32 0, i32 4
  %27 = load i8**, i8*** %26, align 8
  %28 = bitcast i8** %27 to i8*
  %29 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %30 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %29, i32 0, i32 1
  %31 = load i32, i32* %30, align 8
  %32 = sext i32 %31 to i64
  %33 = mul i64 8, %32
  %34 = call i8* @mpc_realloc(%struct.mpc_input_t* %24, i8* %28, i64 %33)
  %35 = bitcast i8* %34 to i8**
  %36 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %37 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %36, i32 0, i32 4
  store i8** %35, i8*** %37, align 8
  %38 = load i8*, i8** %10, align 8
  %39 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %40 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %39, i32 0, i32 4
  %41 = load i8**, i8*** %40, align 8
  %42 = getelementptr inbounds i8*, i8** %41, i64 0
  store i8* %38, i8** %42, align 8
  %43 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  store %struct.mpc_err_t* %43, %struct.mpc_err_t** %4, align 8
  br label %245

; <label>:44:                                     ; preds = %14
  %45 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %46 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %45, i32 0, i32 1
  %47 = load i32, i32* %46, align 8
  %48 = icmp eq i32 %47, 1
  br i1 %48, label %49, label %84

; <label>:49:                                     ; preds = %44
  %50 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %51 = load i8*, i8** %7, align 8
  %52 = call i64 @strlen(i8* %51) #7
  %53 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %54 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %53, i32 0, i32 4
  %55 = load i8**, i8*** %54, align 8
  %56 = getelementptr inbounds i8*, i8** %55, i64 0
  %57 = load i8*, i8** %56, align 8
  %58 = call i64 @strlen(i8* %57) #7
  %59 = add i64 %52, %58
  %60 = add i64 %59, 1
  %61 = call i8* @mpc_malloc(%struct.mpc_input_t* %50, i64 %60)
  store i8* %61, i8** %10, align 8
  %62 = load i8*, i8** %10, align 8
  %63 = load i8*, i8** %7, align 8
  %64 = call i8* @strcpy(i8* %62, i8* %63) #5
  %65 = load i8*, i8** %10, align 8
  %66 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %67 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %66, i32 0, i32 4
  %68 = load i8**, i8*** %67, align 8
  %69 = getelementptr inbounds i8*, i8** %68, i64 0
  %70 = load i8*, i8** %69, align 8
  %71 = call i8* @strcat(i8* %65, i8* %70) #5
  %72 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %73 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %74 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %73, i32 0, i32 4
  %75 = load i8**, i8*** %74, align 8
  %76 = getelementptr inbounds i8*, i8** %75, i64 0
  %77 = load i8*, i8** %76, align 8
  call void @mpc_free(%struct.mpc_input_t* %72, i8* %77)
  %78 = load i8*, i8** %10, align 8
  %79 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %80 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %79, i32 0, i32 4
  %81 = load i8**, i8*** %80, align 8
  %82 = getelementptr inbounds i8*, i8** %81, i64 0
  store i8* %78, i8** %82, align 8
  %83 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  store %struct.mpc_err_t* %83, %struct.mpc_err_t** %4, align 8
  br label %245

; <label>:84:                                     ; preds = %44
  %85 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %86 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %85, i32 0, i32 1
  %87 = load i32, i32* %86, align 8
  %88 = icmp sgt i32 %87, 1
  br i1 %88, label %89, label %242

; <label>:89:                                     ; preds = %84
  %90 = load i8*, i8** %7, align 8
  %91 = call i64 @strlen(i8* %90) #7
  %92 = load i64, i64* %9, align 8
  %93 = add i64 %92, %91
  store i64 %93, i64* %9, align 8
  store i32 0, i32* %8, align 4
  br label %94

; <label>:94:                                     ; preds = %113, %89
  %95 = load i32, i32* %8, align 4
  %96 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %97 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %96, i32 0, i32 1
  %98 = load i32, i32* %97, align 8
  %99 = sub nsw i32 %98, 2
  %100 = icmp slt i32 %95, %99
  br i1 %100, label %101, label %116

; <label>:101:                                    ; preds = %94
  %102 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %103 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %102, i32 0, i32 4
  %104 = load i8**, i8*** %103, align 8
  %105 = load i32, i32* %8, align 4
  %106 = sext i32 %105 to i64
  %107 = getelementptr inbounds i8*, i8** %104, i64 %106
  %108 = load i8*, i8** %107, align 8
  %109 = call i64 @strlen(i8* %108) #7
  %110 = add i64 %109, 2
  %111 = load i64, i64* %9, align 8
  %112 = add i64 %111, %110
  store i64 %112, i64* %9, align 8
  br label %113

; <label>:113:                                    ; preds = %101
  %114 = load i32, i32* %8, align 4
  %115 = add nsw i32 %114, 1
  store i32 %115, i32* %8, align 4
  br label %94

; <label>:116:                                    ; preds = %94
  %117 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %118 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %117, i32 0, i32 4
  %119 = load i8**, i8*** %118, align 8
  %120 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %121 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %120, i32 0, i32 1
  %122 = load i32, i32* %121, align 8
  %123 = sub nsw i32 %122, 2
  %124 = sext i32 %123 to i64
  %125 = getelementptr inbounds i8*, i8** %119, i64 %124
  %126 = load i8*, i8** %125, align 8
  %127 = call i64 @strlen(i8* %126) #7
  %128 = load i64, i64* %9, align 8
  %129 = add i64 %128, %127
  store i64 %129, i64* %9, align 8
  %130 = load i64, i64* %9, align 8
  %131 = add i64 %130, 4
  store i64 %131, i64* %9, align 8
  %132 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %133 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %132, i32 0, i32 4
  %134 = load i8**, i8*** %133, align 8
  %135 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %136 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %135, i32 0, i32 1
  %137 = load i32, i32* %136, align 8
  %138 = sub nsw i32 %137, 1
  %139 = sext i32 %138 to i64
  %140 = getelementptr inbounds i8*, i8** %134, i64 %139
  %141 = load i8*, i8** %140, align 8
  %142 = call i64 @strlen(i8* %141) #7
  %143 = load i64, i64* %9, align 8
  %144 = add i64 %143, %142
  store i64 %144, i64* %9, align 8
  %145 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %146 = load i64, i64* %9, align 8
  %147 = add i64 %146, 1
  %148 = call i8* @mpc_malloc(%struct.mpc_input_t* %145, i64 %147)
  store i8* %148, i8** %10, align 8
  %149 = load i8*, i8** %10, align 8
  %150 = load i8*, i8** %7, align 8
  %151 = call i8* @strcpy(i8* %149, i8* %150) #5
  store i32 0, i32* %8, align 4
  br label %152

; <label>:152:                                    ; preds = %171, %116
  %153 = load i32, i32* %8, align 4
  %154 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %155 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %154, i32 0, i32 1
  %156 = load i32, i32* %155, align 8
  %157 = sub nsw i32 %156, 2
  %158 = icmp slt i32 %153, %157
  br i1 %158, label %159, label %174

; <label>:159:                                    ; preds = %152
  %160 = load i8*, i8** %10, align 8
  %161 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %162 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %161, i32 0, i32 4
  %163 = load i8**, i8*** %162, align 8
  %164 = load i32, i32* %8, align 4
  %165 = sext i32 %164 to i64
  %166 = getelementptr inbounds i8*, i8** %163, i64 %165
  %167 = load i8*, i8** %166, align 8
  %168 = call i8* @strcat(i8* %160, i8* %167) #5
  %169 = load i8*, i8** %10, align 8
  %170 = call i8* @strcat(i8* %169, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.105, i32 0, i32 0)) #5
  br label %171

; <label>:171:                                    ; preds = %159
  %172 = load i32, i32* %8, align 4
  %173 = add nsw i32 %172, 1
  store i32 %173, i32* %8, align 4
  br label %152

; <label>:174:                                    ; preds = %152
  %175 = load i8*, i8** %10, align 8
  %176 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %177 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %176, i32 0, i32 4
  %178 = load i8**, i8*** %177, align 8
  %179 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %180 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %179, i32 0, i32 1
  %181 = load i32, i32* %180, align 8
  %182 = sub nsw i32 %181, 2
  %183 = sext i32 %182 to i64
  %184 = getelementptr inbounds i8*, i8** %178, i64 %183
  %185 = load i8*, i8** %184, align 8
  %186 = call i8* @strcat(i8* %175, i8* %185) #5
  %187 = load i8*, i8** %10, align 8
  %188 = call i8* @strcat(i8* %187, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.106, i32 0, i32 0)) #5
  %189 = load i8*, i8** %10, align 8
  %190 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %191 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %190, i32 0, i32 4
  %192 = load i8**, i8*** %191, align 8
  %193 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %194 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %193, i32 0, i32 1
  %195 = load i32, i32* %194, align 8
  %196 = sub nsw i32 %195, 1
  %197 = sext i32 %196 to i64
  %198 = getelementptr inbounds i8*, i8** %192, i64 %197
  %199 = load i8*, i8** %198, align 8
  %200 = call i8* @strcat(i8* %189, i8* %199) #5
  store i32 0, i32* %8, align 4
  br label %201

; <label>:201:                                    ; preds = %216, %174
  %202 = load i32, i32* %8, align 4
  %203 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %204 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %203, i32 0, i32 1
  %205 = load i32, i32* %204, align 8
  %206 = icmp slt i32 %202, %205
  br i1 %206, label %207, label %219

; <label>:207:                                    ; preds = %201
  %208 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %209 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %210 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %209, i32 0, i32 4
  %211 = load i8**, i8*** %210, align 8
  %212 = load i32, i32* %8, align 4
  %213 = sext i32 %212 to i64
  %214 = getelementptr inbounds i8*, i8** %211, i64 %213
  %215 = load i8*, i8** %214, align 8
  call void @mpc_free(%struct.mpc_input_t* %208, i8* %215)
  br label %216

; <label>:216:                                    ; preds = %207
  %217 = load i32, i32* %8, align 4
  %218 = add nsw i32 %217, 1
  store i32 %218, i32* %8, align 4
  br label %201

; <label>:219:                                    ; preds = %201
  %220 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %221 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %220, i32 0, i32 1
  store i32 1, i32* %221, align 8
  %222 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %223 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %224 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %223, i32 0, i32 4
  %225 = load i8**, i8*** %224, align 8
  %226 = bitcast i8** %225 to i8*
  %227 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %228 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %227, i32 0, i32 1
  %229 = load i32, i32* %228, align 8
  %230 = sext i32 %229 to i64
  %231 = mul i64 8, %230
  %232 = call i8* @mpc_realloc(%struct.mpc_input_t* %222, i8* %226, i64 %231)
  %233 = bitcast i8* %232 to i8**
  %234 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %235 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %234, i32 0, i32 4
  store i8** %233, i8*** %235, align 8
  %236 = load i8*, i8** %10, align 8
  %237 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %238 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %237, i32 0, i32 4
  %239 = load i8**, i8*** %238, align 8
  %240 = getelementptr inbounds i8*, i8** %239, i64 0
  store i8* %236, i8** %240, align 8
  %241 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  store %struct.mpc_err_t* %241, %struct.mpc_err_t** %4, align 8
  br label %245

; <label>:242:                                    ; preds = %84
  br label %243

; <label>:243:                                    ; preds = %242
  br label %244

; <label>:244:                                    ; preds = %243
  store %struct.mpc_err_t* null, %struct.mpc_err_t** %4, align 8
  br label %245

; <label>:245:                                    ; preds = %244, %219, %49, %19, %13
  %246 = load %struct.mpc_err_t*, %struct.mpc_err_t** %4, align 8
  ret %struct.mpc_err_t* %246
}

; Function Attrs: nounwind
declare i32 @sprintf(i8*, i8*, ...) #1

; Function Attrs: noinline nounwind optnone uwtable
define internal %struct.mpc_err_t* @mpc_err_or(%struct.mpc_input_t*, %struct.mpc_err_t**, i32) #0 {
  %4 = alloca %struct.mpc_err_t*, align 8
  %5 = alloca %struct.mpc_input_t*, align 8
  %6 = alloca %struct.mpc_err_t**, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca %struct.mpc_err_t*, align 8
  %12 = alloca %struct.mpc_state_t, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %5, align 8
  store %struct.mpc_err_t** %1, %struct.mpc_err_t*** %6, align 8
  store i32 %2, i32* %7, align 4
  store i32 -1, i32* %10, align 4
  store i32 0, i32* %8, align 4
  br label %13

; <label>:13:                                     ; preds = %27, %3
  %14 = load i32, i32* %8, align 4
  %15 = load i32, i32* %7, align 4
  %16 = icmp slt i32 %14, %15
  br i1 %16, label %17, label %30

; <label>:17:                                     ; preds = %13
  %18 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %6, align 8
  %19 = load i32, i32* %8, align 4
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds %struct.mpc_err_t*, %struct.mpc_err_t** %18, i64 %20
  %22 = load %struct.mpc_err_t*, %struct.mpc_err_t** %21, align 8
  %23 = icmp ne %struct.mpc_err_t* %22, null
  br i1 %23, label %24, label %26

; <label>:24:                                     ; preds = %17
  %25 = load i32, i32* %8, align 4
  store i32 %25, i32* %10, align 4
  br label %26

; <label>:26:                                     ; preds = %24, %17
  br label %27

; <label>:27:                                     ; preds = %26
  %28 = load i32, i32* %8, align 4
  %29 = add nsw i32 %28, 1
  store i32 %29, i32* %8, align 4
  br label %13

; <label>:30:                                     ; preds = %13
  %31 = load i32, i32* %10, align 4
  %32 = icmp eq i32 %31, -1
  br i1 %32, label %33, label %34

; <label>:33:                                     ; preds = %30
  store %struct.mpc_err_t* null, %struct.mpc_err_t** %4, align 8
  br label %258

; <label>:34:                                     ; preds = %30
  %35 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %36 = call i8* @mpc_malloc(%struct.mpc_input_t* %35, i64 64)
  %37 = bitcast i8* %36 to %struct.mpc_err_t*
  store %struct.mpc_err_t* %37, %struct.mpc_err_t** %11, align 8
  %38 = load %struct.mpc_err_t*, %struct.mpc_err_t** %11, align 8
  %39 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %38, i32 0, i32 0
  call void @mpc_state_invalid(%struct.mpc_state_t* sret %12)
  %40 = bitcast %struct.mpc_state_t* %39 to i8*
  %41 = bitcast %struct.mpc_state_t* %12 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %40, i8* %41, i64 24, i32 8, i1 false)
  %42 = load %struct.mpc_err_t*, %struct.mpc_err_t** %11, align 8
  %43 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %42, i32 0, i32 1
  store i32 0, i32* %43, align 8
  %44 = load %struct.mpc_err_t*, %struct.mpc_err_t** %11, align 8
  %45 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %44, i32 0, i32 4
  store i8** null, i8*** %45, align 8
  %46 = load %struct.mpc_err_t*, %struct.mpc_err_t** %11, align 8
  %47 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %46, i32 0, i32 3
  store i8* null, i8** %47, align 8
  %48 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %49 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %6, align 8
  %50 = load i32, i32* %10, align 4
  %51 = sext i32 %50 to i64
  %52 = getelementptr inbounds %struct.mpc_err_t*, %struct.mpc_err_t** %49, i64 %51
  %53 = load %struct.mpc_err_t*, %struct.mpc_err_t** %52, align 8
  %54 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %53, i32 0, i32 2
  %55 = load i8*, i8** %54, align 8
  %56 = call i64 @strlen(i8* %55) #7
  %57 = add i64 %56, 1
  %58 = call i8* @mpc_malloc(%struct.mpc_input_t* %48, i64 %57)
  %59 = load %struct.mpc_err_t*, %struct.mpc_err_t** %11, align 8
  %60 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %59, i32 0, i32 2
  store i8* %58, i8** %60, align 8
  %61 = load %struct.mpc_err_t*, %struct.mpc_err_t** %11, align 8
  %62 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %61, i32 0, i32 2
  %63 = load i8*, i8** %62, align 8
  %64 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %6, align 8
  %65 = load i32, i32* %10, align 4
  %66 = sext i32 %65 to i64
  %67 = getelementptr inbounds %struct.mpc_err_t*, %struct.mpc_err_t** %64, i64 %66
  %68 = load %struct.mpc_err_t*, %struct.mpc_err_t** %67, align 8
  %69 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %68, i32 0, i32 2
  %70 = load i8*, i8** %69, align 8
  %71 = call i8* @strcpy(i8* %63, i8* %70) #5
  store i32 0, i32* %8, align 4
  br label %72

; <label>:72:                                     ; preds = %110, %34
  %73 = load i32, i32* %8, align 4
  %74 = load i32, i32* %7, align 4
  %75 = icmp slt i32 %73, %74
  br i1 %75, label %76, label %113

; <label>:76:                                     ; preds = %72
  %77 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %6, align 8
  %78 = load i32, i32* %8, align 4
  %79 = sext i32 %78 to i64
  %80 = getelementptr inbounds %struct.mpc_err_t*, %struct.mpc_err_t** %77, i64 %79
  %81 = load %struct.mpc_err_t*, %struct.mpc_err_t** %80, align 8
  %82 = icmp ne %struct.mpc_err_t* %81, null
  br i1 %82, label %84, label %83

; <label>:83:                                     ; preds = %76
  br label %110

; <label>:84:                                     ; preds = %76
  %85 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %6, align 8
  %86 = load i32, i32* %8, align 4
  %87 = sext i32 %86 to i64
  %88 = getelementptr inbounds %struct.mpc_err_t*, %struct.mpc_err_t** %85, i64 %87
  %89 = load %struct.mpc_err_t*, %struct.mpc_err_t** %88, align 8
  %90 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %89, i32 0, i32 0
  %91 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %90, i32 0, i32 0
  %92 = load i64, i64* %91, align 8
  %93 = load %struct.mpc_err_t*, %struct.mpc_err_t** %11, align 8
  %94 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %93, i32 0, i32 0
  %95 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %94, i32 0, i32 0
  %96 = load i64, i64* %95, align 8
  %97 = icmp sgt i64 %92, %96
  br i1 %97, label %98, label %109

; <label>:98:                                     ; preds = %84
  %99 = load %struct.mpc_err_t*, %struct.mpc_err_t** %11, align 8
  %100 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %99, i32 0, i32 0
  %101 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %6, align 8
  %102 = load i32, i32* %8, align 4
  %103 = sext i32 %102 to i64
  %104 = getelementptr inbounds %struct.mpc_err_t*, %struct.mpc_err_t** %101, i64 %103
  %105 = load %struct.mpc_err_t*, %struct.mpc_err_t** %104, align 8
  %106 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %105, i32 0, i32 0
  %107 = bitcast %struct.mpc_state_t* %100 to i8*
  %108 = bitcast %struct.mpc_state_t* %106 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %107, i8* %108, i64 24, i32 8, i1 false)
  br label %109

; <label>:109:                                    ; preds = %98, %84
  br label %110

; <label>:110:                                    ; preds = %109, %83
  %111 = load i32, i32* %8, align 4
  %112 = add nsw i32 %111, 1
  store i32 %112, i32* %8, align 4
  br label %72

; <label>:113:                                    ; preds = %72
  store i32 0, i32* %8, align 4
  br label %114

; <label>:114:                                    ; preds = %230, %113
  %115 = load i32, i32* %8, align 4
  %116 = load i32, i32* %7, align 4
  %117 = icmp slt i32 %115, %116
  br i1 %117, label %118, label %233

; <label>:118:                                    ; preds = %114
  %119 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %6, align 8
  %120 = load i32, i32* %8, align 4
  %121 = sext i32 %120 to i64
  %122 = getelementptr inbounds %struct.mpc_err_t*, %struct.mpc_err_t** %119, i64 %121
  %123 = load %struct.mpc_err_t*, %struct.mpc_err_t** %122, align 8
  %124 = icmp ne %struct.mpc_err_t* %123, null
  br i1 %124, label %126, label %125

; <label>:125:                                    ; preds = %118
  br label %230

; <label>:126:                                    ; preds = %118
  %127 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %6, align 8
  %128 = load i32, i32* %8, align 4
  %129 = sext i32 %128 to i64
  %130 = getelementptr inbounds %struct.mpc_err_t*, %struct.mpc_err_t** %127, i64 %129
  %131 = load %struct.mpc_err_t*, %struct.mpc_err_t** %130, align 8
  %132 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %131, i32 0, i32 0
  %133 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %132, i32 0, i32 0
  %134 = load i64, i64* %133, align 8
  %135 = load %struct.mpc_err_t*, %struct.mpc_err_t** %11, align 8
  %136 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %135, i32 0, i32 0
  %137 = getelementptr inbounds %struct.mpc_state_t, %struct.mpc_state_t* %136, i32 0, i32 0
  %138 = load i64, i64* %137, align 8
  %139 = icmp slt i64 %134, %138
  br i1 %139, label %140, label %141

; <label>:140:                                    ; preds = %126
  br label %230

; <label>:141:                                    ; preds = %126
  %142 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %6, align 8
  %143 = load i32, i32* %8, align 4
  %144 = sext i32 %143 to i64
  %145 = getelementptr inbounds %struct.mpc_err_t*, %struct.mpc_err_t** %142, i64 %144
  %146 = load %struct.mpc_err_t*, %struct.mpc_err_t** %145, align 8
  %147 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %146, i32 0, i32 3
  %148 = load i8*, i8** %147, align 8
  %149 = icmp ne i8* %148, null
  br i1 %149, label %150, label %175

; <label>:150:                                    ; preds = %141
  %151 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %152 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %6, align 8
  %153 = load i32, i32* %8, align 4
  %154 = sext i32 %153 to i64
  %155 = getelementptr inbounds %struct.mpc_err_t*, %struct.mpc_err_t** %152, i64 %154
  %156 = load %struct.mpc_err_t*, %struct.mpc_err_t** %155, align 8
  %157 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %156, i32 0, i32 3
  %158 = load i8*, i8** %157, align 8
  %159 = call i64 @strlen(i8* %158) #7
  %160 = add i64 %159, 1
  %161 = call i8* @mpc_malloc(%struct.mpc_input_t* %151, i64 %160)
  %162 = load %struct.mpc_err_t*, %struct.mpc_err_t** %11, align 8
  %163 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %162, i32 0, i32 3
  store i8* %161, i8** %163, align 8
  %164 = load %struct.mpc_err_t*, %struct.mpc_err_t** %11, align 8
  %165 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %164, i32 0, i32 3
  %166 = load i8*, i8** %165, align 8
  %167 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %6, align 8
  %168 = load i32, i32* %8, align 4
  %169 = sext i32 %168 to i64
  %170 = getelementptr inbounds %struct.mpc_err_t*, %struct.mpc_err_t** %167, i64 %169
  %171 = load %struct.mpc_err_t*, %struct.mpc_err_t** %170, align 8
  %172 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %171, i32 0, i32 3
  %173 = load i8*, i8** %172, align 8
  %174 = call i8* @strcpy(i8* %166, i8* %173) #5
  br label %233

; <label>:175:                                    ; preds = %141
  %176 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %6, align 8
  %177 = load i32, i32* %8, align 4
  %178 = sext i32 %177 to i64
  %179 = getelementptr inbounds %struct.mpc_err_t*, %struct.mpc_err_t** %176, i64 %178
  %180 = load %struct.mpc_err_t*, %struct.mpc_err_t** %179, align 8
  %181 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %180, i32 0, i32 5
  %182 = load i8, i8* %181, align 8
  %183 = load %struct.mpc_err_t*, %struct.mpc_err_t** %11, align 8
  %184 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %183, i32 0, i32 5
  store i8 %182, i8* %184, align 8
  store i32 0, i32* %9, align 4
  br label %185

; <label>:185:                                    ; preds = %226, %175
  %186 = load i32, i32* %9, align 4
  %187 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %6, align 8
  %188 = load i32, i32* %8, align 4
  %189 = sext i32 %188 to i64
  %190 = getelementptr inbounds %struct.mpc_err_t*, %struct.mpc_err_t** %187, i64 %189
  %191 = load %struct.mpc_err_t*, %struct.mpc_err_t** %190, align 8
  %192 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %191, i32 0, i32 1
  %193 = load i32, i32* %192, align 8
  %194 = icmp slt i32 %186, %193
  br i1 %194, label %195, label %229

; <label>:195:                                    ; preds = %185
  %196 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %197 = load %struct.mpc_err_t*, %struct.mpc_err_t** %11, align 8
  %198 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %6, align 8
  %199 = load i32, i32* %8, align 4
  %200 = sext i32 %199 to i64
  %201 = getelementptr inbounds %struct.mpc_err_t*, %struct.mpc_err_t** %198, i64 %200
  %202 = load %struct.mpc_err_t*, %struct.mpc_err_t** %201, align 8
  %203 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %202, i32 0, i32 4
  %204 = load i8**, i8*** %203, align 8
  %205 = load i32, i32* %9, align 4
  %206 = sext i32 %205 to i64
  %207 = getelementptr inbounds i8*, i8** %204, i64 %206
  %208 = load i8*, i8** %207, align 8
  %209 = call i32 @mpc_err_contains_expected(%struct.mpc_input_t* %196, %struct.mpc_err_t* %197, i8* %208)
  %210 = icmp ne i32 %209, 0
  br i1 %210, label %225, label %211

; <label>:211:                                    ; preds = %195
  %212 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %213 = load %struct.mpc_err_t*, %struct.mpc_err_t** %11, align 8
  %214 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %6, align 8
  %215 = load i32, i32* %8, align 4
  %216 = sext i32 %215 to i64
  %217 = getelementptr inbounds %struct.mpc_err_t*, %struct.mpc_err_t** %214, i64 %216
  %218 = load %struct.mpc_err_t*, %struct.mpc_err_t** %217, align 8
  %219 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %218, i32 0, i32 4
  %220 = load i8**, i8*** %219, align 8
  %221 = load i32, i32* %9, align 4
  %222 = sext i32 %221 to i64
  %223 = getelementptr inbounds i8*, i8** %220, i64 %222
  %224 = load i8*, i8** %223, align 8
  call void @mpc_err_add_expected(%struct.mpc_input_t* %212, %struct.mpc_err_t* %213, i8* %224)
  br label %225

; <label>:225:                                    ; preds = %211, %195
  br label %226

; <label>:226:                                    ; preds = %225
  %227 = load i32, i32* %9, align 4
  %228 = add nsw i32 %227, 1
  store i32 %228, i32* %9, align 4
  br label %185

; <label>:229:                                    ; preds = %185
  br label %230

; <label>:230:                                    ; preds = %229, %140, %125
  %231 = load i32, i32* %8, align 4
  %232 = add nsw i32 %231, 1
  store i32 %232, i32* %8, align 4
  br label %114

; <label>:233:                                    ; preds = %150, %114
  store i32 0, i32* %8, align 4
  br label %234

; <label>:234:                                    ; preds = %253, %233
  %235 = load i32, i32* %8, align 4
  %236 = load i32, i32* %7, align 4
  %237 = icmp slt i32 %235, %236
  br i1 %237, label %238, label %256

; <label>:238:                                    ; preds = %234
  %239 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %6, align 8
  %240 = load i32, i32* %8, align 4
  %241 = sext i32 %240 to i64
  %242 = getelementptr inbounds %struct.mpc_err_t*, %struct.mpc_err_t** %239, i64 %241
  %243 = load %struct.mpc_err_t*, %struct.mpc_err_t** %242, align 8
  %244 = icmp ne %struct.mpc_err_t* %243, null
  br i1 %244, label %246, label %245

; <label>:245:                                    ; preds = %238
  br label %253

; <label>:246:                                    ; preds = %238
  %247 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  %248 = load %struct.mpc_err_t**, %struct.mpc_err_t*** %6, align 8
  %249 = load i32, i32* %8, align 4
  %250 = sext i32 %249 to i64
  %251 = getelementptr inbounds %struct.mpc_err_t*, %struct.mpc_err_t** %248, i64 %250
  %252 = load %struct.mpc_err_t*, %struct.mpc_err_t** %251, align 8
  call void @mpc_err_delete_internal(%struct.mpc_input_t* %247, %struct.mpc_err_t* %252)
  br label %253

; <label>:253:                                    ; preds = %246, %245
  %254 = load i32, i32* %8, align 4
  %255 = add nsw i32 %254, 1
  store i32 %255, i32* %8, align 4
  br label %234

; <label>:256:                                    ; preds = %234
  %257 = load %struct.mpc_err_t*, %struct.mpc_err_t** %11, align 8
  store %struct.mpc_err_t* %257, %struct.mpc_err_t** %4, align 8
  br label %258

; <label>:258:                                    ; preds = %256, %33
  %259 = load %struct.mpc_err_t*, %struct.mpc_err_t** %4, align 8
  ret %struct.mpc_err_t* %259
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @mpc_err_contains_expected(%struct.mpc_input_t*, %struct.mpc_err_t*, i8*) #0 {
  %4 = alloca i32, align 4
  %5 = alloca %struct.mpc_input_t*, align 8
  %6 = alloca %struct.mpc_err_t*, align 8
  %7 = alloca i8*, align 8
  %8 = alloca i32, align 4
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %5, align 8
  store %struct.mpc_err_t* %1, %struct.mpc_err_t** %6, align 8
  store i8* %2, i8** %7, align 8
  %9 = load %struct.mpc_input_t*, %struct.mpc_input_t** %5, align 8
  store i32 0, i32* %8, align 4
  br label %10

; <label>:10:                                     ; preds = %29, %3
  %11 = load i32, i32* %8, align 4
  %12 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %13 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %12, i32 0, i32 1
  %14 = load i32, i32* %13, align 8
  %15 = icmp slt i32 %11, %14
  br i1 %15, label %16, label %32

; <label>:16:                                     ; preds = %10
  %17 = load %struct.mpc_err_t*, %struct.mpc_err_t** %6, align 8
  %18 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %17, i32 0, i32 4
  %19 = load i8**, i8*** %18, align 8
  %20 = load i32, i32* %8, align 4
  %21 = sext i32 %20 to i64
  %22 = getelementptr inbounds i8*, i8** %19, i64 %21
  %23 = load i8*, i8** %22, align 8
  %24 = load i8*, i8** %7, align 8
  %25 = call i32 @strcmp(i8* %23, i8* %24) #7
  %26 = icmp eq i32 %25, 0
  br i1 %26, label %27, label %28

; <label>:27:                                     ; preds = %16
  store i32 1, i32* %4, align 4
  br label %33

; <label>:28:                                     ; preds = %16
  br label %29

; <label>:29:                                     ; preds = %28
  %30 = load i32, i32* %8, align 4
  %31 = add nsw i32 %30, 1
  store i32 %31, i32* %8, align 4
  br label %10

; <label>:32:                                     ; preds = %10
  store i32 0, i32* %4, align 4
  br label %33

; <label>:33:                                     ; preds = %32, %27
  %34 = load i32, i32* %4, align 4
  ret i32 %34
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_err_add_expected(%struct.mpc_input_t*, %struct.mpc_err_t*, i8*) #0 {
  %4 = alloca %struct.mpc_input_t*, align 8
  %5 = alloca %struct.mpc_err_t*, align 8
  %6 = alloca i8*, align 8
  store %struct.mpc_input_t* %0, %struct.mpc_input_t** %4, align 8
  store %struct.mpc_err_t* %1, %struct.mpc_err_t** %5, align 8
  store i8* %2, i8** %6, align 8
  %7 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %8 = load %struct.mpc_err_t*, %struct.mpc_err_t** %5, align 8
  %9 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %8, i32 0, i32 1
  %10 = load i32, i32* %9, align 8
  %11 = add nsw i32 %10, 1
  store i32 %11, i32* %9, align 8
  %12 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %13 = load %struct.mpc_err_t*, %struct.mpc_err_t** %5, align 8
  %14 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %13, i32 0, i32 4
  %15 = load i8**, i8*** %14, align 8
  %16 = bitcast i8** %15 to i8*
  %17 = load %struct.mpc_err_t*, %struct.mpc_err_t** %5, align 8
  %18 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %17, i32 0, i32 1
  %19 = load i32, i32* %18, align 8
  %20 = sext i32 %19 to i64
  %21 = mul i64 8, %20
  %22 = call i8* @mpc_realloc(%struct.mpc_input_t* %12, i8* %16, i64 %21)
  %23 = bitcast i8* %22 to i8**
  %24 = load %struct.mpc_err_t*, %struct.mpc_err_t** %5, align 8
  %25 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %24, i32 0, i32 4
  store i8** %23, i8*** %25, align 8
  %26 = load %struct.mpc_input_t*, %struct.mpc_input_t** %4, align 8
  %27 = load i8*, i8** %6, align 8
  %28 = call i64 @strlen(i8* %27) #7
  %29 = add i64 %28, 1
  %30 = call i8* @mpc_malloc(%struct.mpc_input_t* %26, i64 %29)
  %31 = load %struct.mpc_err_t*, %struct.mpc_err_t** %5, align 8
  %32 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %31, i32 0, i32 4
  %33 = load i8**, i8*** %32, align 8
  %34 = load %struct.mpc_err_t*, %struct.mpc_err_t** %5, align 8
  %35 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %34, i32 0, i32 1
  %36 = load i32, i32* %35, align 8
  %37 = sub nsw i32 %36, 1
  %38 = sext i32 %37 to i64
  %39 = getelementptr inbounds i8*, i8** %33, i64 %38
  store i8* %30, i8** %39, align 8
  %40 = load %struct.mpc_err_t*, %struct.mpc_err_t** %5, align 8
  %41 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %40, i32 0, i32 4
  %42 = load i8**, i8*** %41, align 8
  %43 = load %struct.mpc_err_t*, %struct.mpc_err_t** %5, align 8
  %44 = getelementptr inbounds %struct.mpc_err_t, %struct.mpc_err_t* %43, i32 0, i32 1
  %45 = load i32, i32* %44, align 8
  %46 = sub nsw i32 %45, 1
  %47 = sext i32 %46 to i64
  %48 = getelementptr inbounds i8*, i8** %42, i64 %47
  %49 = load i8*, i8** %48, align 8
  %50 = load i8*, i8** %6, align 8
  %51 = call i8* @strcpy(i8* %49, i8* %50) #5
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_undefine_or(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  %3 = alloca i32, align 4
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  store i32 0, i32* %3, align 4
  br label %4

; <label>:4:                                      ; preds = %22, %1
  %5 = load i32, i32* %3, align 4
  %6 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %7 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %6, i32 0, i32 3
  %8 = bitcast %union.mpc_pdata_t* %7 to %struct.mpc_pdata_or_t*
  %9 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %8, i32 0, i32 0
  %10 = load i32, i32* %9, align 8
  %11 = icmp slt i32 %5, %10
  br i1 %11, label %12, label %25

; <label>:12:                                     ; preds = %4
  %13 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %14 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %13, i32 0, i32 3
  %15 = bitcast %union.mpc_pdata_t* %14 to %struct.mpc_pdata_or_t*
  %16 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %15, i32 0, i32 1
  %17 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %16, align 8
  %18 = load i32, i32* %3, align 4
  %19 = sext i32 %18 to i64
  %20 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %17, i64 %19
  %21 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %20, align 8
  call void @mpc_undefine_unretained(%struct.mpc_parser_t* %21, i32 0)
  br label %22

; <label>:22:                                     ; preds = %12
  %23 = load i32, i32* %3, align 4
  %24 = add nsw i32 %23, 1
  store i32 %24, i32* %3, align 4
  br label %4

; <label>:25:                                     ; preds = %4
  %26 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %27 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %26, i32 0, i32 3
  %28 = bitcast %union.mpc_pdata_t* %27 to %struct.mpc_pdata_or_t*
  %29 = getelementptr inbounds %struct.mpc_pdata_or_t, %struct.mpc_pdata_or_t* %28, i32 0, i32 1
  %30 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %29, align 8
  %31 = bitcast %struct.mpc_parser_t** %30 to i8*
  call void @free(i8* %31) #5
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpc_undefine_and(%struct.mpc_parser_t*) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  %3 = alloca i32, align 4
  store %struct.mpc_parser_t* %0, %struct.mpc_parser_t** %2, align 8
  store i32 0, i32* %3, align 4
  br label %4

; <label>:4:                                      ; preds = %22, %1
  %5 = load i32, i32* %3, align 4
  %6 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %7 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %6, i32 0, i32 3
  %8 = bitcast %union.mpc_pdata_t* %7 to %struct.mpc_pdata_and_t*
  %9 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %8, i32 0, i32 0
  %10 = load i32, i32* %9, align 8
  %11 = icmp slt i32 %5, %10
  br i1 %11, label %12, label %25

; <label>:12:                                     ; preds = %4
  %13 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %14 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %13, i32 0, i32 3
  %15 = bitcast %union.mpc_pdata_t* %14 to %struct.mpc_pdata_and_t*
  %16 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %15, i32 0, i32 2
  %17 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %16, align 8
  %18 = load i32, i32* %3, align 4
  %19 = sext i32 %18 to i64
  %20 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %17, i64 %19
  %21 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %20, align 8
  call void @mpc_undefine_unretained(%struct.mpc_parser_t* %21, i32 0)
  br label %22

; <label>:22:                                     ; preds = %12
  %23 = load i32, i32* %3, align 4
  %24 = add nsw i32 %23, 1
  store i32 %24, i32* %3, align 4
  br label %4

; <label>:25:                                     ; preds = %4
  %26 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %27 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %26, i32 0, i32 3
  %28 = bitcast %union.mpc_pdata_t* %27 to %struct.mpc_pdata_and_t*
  %29 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %28, i32 0, i32 2
  %30 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %29, align 8
  %31 = bitcast %struct.mpc_parser_t** %30 to i8*
  call void @free(i8* %31) #5
  %32 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  %33 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %32, i32 0, i32 3
  %34 = bitcast %union.mpc_pdata_t* %33 to %struct.mpc_pdata_and_t*
  %35 = getelementptr inbounds %struct.mpc_pdata_and_t, %struct.mpc_pdata_and_t* %34, i32 0, i32 3
  %36 = load void (i8*)**, void (i8*)*** %35, align 8
  %37 = bitcast void (i8*)** %36 to i8*
  call void @free(i8* %37) #5
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal %struct.mpc_parser_t* @mpc_re_escape_char(i8 signext) #0 {
  %2 = alloca %struct.mpc_parser_t*, align 8
  %3 = alloca i8, align 1
  store i8 %0, i8* %3, align 1
  %4 = load i8, i8* %3, align 1
  %5 = sext i8 %4 to i32
  switch i32 %5, label %48 [
    i32 97, label %6
    i32 102, label %8
    i32 110, label %10
    i32 114, label %12
    i32 116, label %14
    i32 118, label %16
    i32 98, label %18
    i32 66, label %22
    i32 65, label %25
    i32 90, label %29
    i32 100, label %33
    i32 68, label %35
    i32 115, label %38
    i32 83, label %40
    i32 119, label %43
    i32 87, label %45
  ]

; <label>:6:                                      ; preds = %1
  %7 = call %struct.mpc_parser_t* @mpc_char(i8 signext 7)
  store %struct.mpc_parser_t* %7, %struct.mpc_parser_t** %2, align 8
  br label %49

; <label>:8:                                      ; preds = %1
  %9 = call %struct.mpc_parser_t* @mpc_char(i8 signext 12)
  store %struct.mpc_parser_t* %9, %struct.mpc_parser_t** %2, align 8
  br label %49

; <label>:10:                                     ; preds = %1
  %11 = call %struct.mpc_parser_t* @mpc_char(i8 signext 10)
  store %struct.mpc_parser_t* %11, %struct.mpc_parser_t** %2, align 8
  br label %49

; <label>:12:                                     ; preds = %1
  %13 = call %struct.mpc_parser_t* @mpc_char(i8 signext 13)
  store %struct.mpc_parser_t* %13, %struct.mpc_parser_t** %2, align 8
  br label %49

; <label>:14:                                     ; preds = %1
  %15 = call %struct.mpc_parser_t* @mpc_char(i8 signext 9)
  store %struct.mpc_parser_t* %15, %struct.mpc_parser_t** %2, align 8
  br label %49

; <label>:16:                                     ; preds = %1
  %17 = call %struct.mpc_parser_t* @mpc_char(i8 signext 11)
  store %struct.mpc_parser_t* %17, %struct.mpc_parser_t** %2, align 8
  br label %49

; <label>:18:                                     ; preds = %1
  %19 = call %struct.mpc_parser_t* @mpc_boundary()
  %20 = call %struct.mpc_parser_t* @mpc_lift(i8* ()* @mpcf_ctor_str)
  %21 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcf_snd, %struct.mpc_parser_t* %19, %struct.mpc_parser_t* %20, void (i8*)* @free)
  store %struct.mpc_parser_t* %21, %struct.mpc_parser_t** %2, align 8
  br label %49

; <label>:22:                                     ; preds = %1
  %23 = call %struct.mpc_parser_t* @mpc_boundary()
  %24 = call %struct.mpc_parser_t* @mpc_not_lift(%struct.mpc_parser_t* %23, void (i8*)* @free, i8* ()* @mpcf_ctor_str)
  store %struct.mpc_parser_t* %24, %struct.mpc_parser_t** %2, align 8
  br label %49

; <label>:25:                                     ; preds = %1
  %26 = call %struct.mpc_parser_t* @mpc_soi()
  %27 = call %struct.mpc_parser_t* @mpc_lift(i8* ()* @mpcf_ctor_str)
  %28 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcf_snd, %struct.mpc_parser_t* %26, %struct.mpc_parser_t* %27, void (i8*)* @free)
  store %struct.mpc_parser_t* %28, %struct.mpc_parser_t** %2, align 8
  br label %49

; <label>:29:                                     ; preds = %1
  %30 = call %struct.mpc_parser_t* @mpc_eoi()
  %31 = call %struct.mpc_parser_t* @mpc_lift(i8* ()* @mpcf_ctor_str)
  %32 = call %struct.mpc_parser_t* (i32, i8* (i32, i8**)*, ...) @mpc_and(i32 2, i8* (i32, i8**)* @mpcf_snd, %struct.mpc_parser_t* %30, %struct.mpc_parser_t* %31, void (i8*)* @free)
  store %struct.mpc_parser_t* %32, %struct.mpc_parser_t** %2, align 8
  br label %49

; <label>:33:                                     ; preds = %1
  %34 = call %struct.mpc_parser_t* @mpc_digit()
  store %struct.mpc_parser_t* %34, %struct.mpc_parser_t** %2, align 8
  br label %49

; <label>:35:                                     ; preds = %1
  %36 = call %struct.mpc_parser_t* @mpc_digit()
  %37 = call %struct.mpc_parser_t* @mpc_not_lift(%struct.mpc_parser_t* %36, void (i8*)* @free, i8* ()* @mpcf_ctor_str)
  store %struct.mpc_parser_t* %37, %struct.mpc_parser_t** %2, align 8
  br label %49

; <label>:38:                                     ; preds = %1
  %39 = call %struct.mpc_parser_t* @mpc_whitespace()
  store %struct.mpc_parser_t* %39, %struct.mpc_parser_t** %2, align 8
  br label %49

; <label>:40:                                     ; preds = %1
  %41 = call %struct.mpc_parser_t* @mpc_whitespace()
  %42 = call %struct.mpc_parser_t* @mpc_not_lift(%struct.mpc_parser_t* %41, void (i8*)* @free, i8* ()* @mpcf_ctor_str)
  store %struct.mpc_parser_t* %42, %struct.mpc_parser_t** %2, align 8
  br label %49

; <label>:43:                                     ; preds = %1
  %44 = call %struct.mpc_parser_t* @mpc_alphanum()
  store %struct.mpc_parser_t* %44, %struct.mpc_parser_t** %2, align 8
  br label %49

; <label>:45:                                     ; preds = %1
  %46 = call %struct.mpc_parser_t* @mpc_alphanum()
  %47 = call %struct.mpc_parser_t* @mpc_not_lift(%struct.mpc_parser_t* %46, void (i8*)* @free, i8* ()* @mpcf_ctor_str)
  store %struct.mpc_parser_t* %47, %struct.mpc_parser_t** %2, align 8
  br label %49

; <label>:48:                                     ; preds = %1
  store %struct.mpc_parser_t* null, %struct.mpc_parser_t** %2, align 8
  br label %49

; <label>:49:                                     ; preds = %48, %45, %43, %40, %38, %35, %33, %29, %25, %22, %18, %16, %14, %12, %10, %8, %6
  %50 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %2, align 8
  ret %struct.mpc_parser_t* %50
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpc_re_range_escape_char(i8 signext) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca i8, align 1
  store i8 %0, i8* %3, align 1
  %4 = load i8, i8* %3, align 1
  %5 = sext i8 %4 to i32
  switch i32 %5, label %17 [
    i32 45, label %6
    i32 97, label %7
    i32 102, label %8
    i32 110, label %9
    i32 114, label %10
    i32 116, label %11
    i32 118, label %12
    i32 98, label %13
    i32 100, label %14
    i32 115, label %15
    i32 119, label %16
  ]

; <label>:6:                                      ; preds = %1
  store i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.77, i32 0, i32 0), i8** %2, align 8
  br label %18

; <label>:7:                                      ; preds = %1
  store i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.110, i32 0, i32 0), i8** %2, align 8
  br label %18

; <label>:8:                                      ; preds = %1
  store i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.111, i32 0, i32 0), i8** %2, align 8
  br label %18

; <label>:9:                                      ; preds = %1
  store i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.7, i32 0, i32 0), i8** %2, align 8
  br label %18

; <label>:10:                                     ; preds = %1
  store i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.112, i32 0, i32 0), i8** %2, align 8
  br label %18

; <label>:11:                                     ; preds = %1
  store i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.113, i32 0, i32 0), i8** %2, align 8
  br label %18

; <label>:12:                                     ; preds = %1
  store i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.114, i32 0, i32 0), i8** %2, align 8
  br label %18

; <label>:13:                                     ; preds = %1
  store i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.115, i32 0, i32 0), i8** %2, align 8
  br label %18

; <label>:14:                                     ; preds = %1
  store i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.28, i32 0, i32 0), i8** %2, align 8
  br label %18

; <label>:15:                                     ; preds = %1
  store i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.23, i32 0, i32 0), i8** %2, align 8
  br label %18

; <label>:16:                                     ; preds = %1
  store i8* getelementptr inbounds ([64 x i8], [64 x i8]* @.str.108, i32 0, i32 0), i8** %2, align 8
  br label %18

; <label>:17:                                     ; preds = %1
  store i8* null, i8** %2, align 8
  br label %18

; <label>:18:                                     ; preds = %17, %16, %15, %14, %13, %12, %11, %10, %9, %8, %7, %6
  %19 = load i8*, i8** %2, align 8
  ret i8* %19
}

; Function Attrs: noinline nounwind optnone uwtable
define internal %struct.mpc_parser_t* @mpca_grammar_find_parser(i8*, %struct.mpca_grammar_st_t*) #0 {
  %3 = alloca %struct.mpc_parser_t*, align 8
  %4 = alloca i8*, align 8
  %5 = alloca %struct.mpca_grammar_st_t*, align 8
  %6 = alloca i32, align 4
  %7 = alloca %struct.mpc_parser_t*, align 8
  %8 = alloca %struct.mpc_parser_t*, align 8
  store i8* %0, i8** %4, align 8
  store %struct.mpca_grammar_st_t* %1, %struct.mpca_grammar_st_t** %5, align 8
  %9 = load i8*, i8** %4, align 8
  %10 = call i32 @is_number(i8* %9)
  %11 = icmp ne i32 %10, 0
  br i1 %11, label %12, label %99

; <label>:12:                                     ; preds = %2
  %13 = load i8*, i8** %4, align 8
  %14 = call i64 @strtol(i8* %13, i8** null, i32 10) #5
  %15 = trunc i64 %14 to i32
  store i32 %15, i32* %6, align 4
  br label %16

; <label>:16:                                     ; preds = %87, %12
  %17 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %18 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %17, i32 0, i32 1
  %19 = load i32, i32* %18, align 8
  %20 = load i32, i32* %6, align 4
  %21 = icmp sle i32 %19, %20
  br i1 %21, label %22, label %88

; <label>:22:                                     ; preds = %16
  %23 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %24 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %23, i32 0, i32 1
  %25 = load i32, i32* %24, align 8
  %26 = add nsw i32 %25, 1
  store i32 %26, i32* %24, align 8
  %27 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %28 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %27, i32 0, i32 2
  %29 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %28, align 8
  %30 = bitcast %struct.mpc_parser_t** %29 to i8*
  %31 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %32 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %31, i32 0, i32 1
  %33 = load i32, i32* %32, align 8
  %34 = sext i32 %33 to i64
  %35 = mul i64 8, %34
  %36 = call i8* @realloc(i8* %30, i64 %35) #5
  %37 = bitcast i8* %36 to %struct.mpc_parser_t**
  %38 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %39 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %38, i32 0, i32 2
  store %struct.mpc_parser_t** %37, %struct.mpc_parser_t*** %39, align 8
  %40 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %41 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %40, i32 0, i32 0
  %42 = load [1 x %struct.__va_list_tag]*, [1 x %struct.__va_list_tag]** %41, align 8
  %43 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %42, i32 0, i32 0
  %44 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %43, i32 0, i32 0
  %45 = load i32, i32* %44, align 8
  %46 = icmp ule i32 %45, 40
  br i1 %46, label %47, label %53

; <label>:47:                                     ; preds = %22
  %48 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %43, i32 0, i32 3
  %49 = load i8*, i8** %48, align 8
  %50 = getelementptr i8, i8* %49, i32 %45
  %51 = bitcast i8* %50 to %struct.mpc_parser_t**
  %52 = add i32 %45, 8
  store i32 %52, i32* %44, align 8
  br label %58

; <label>:53:                                     ; preds = %22
  %54 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %43, i32 0, i32 2
  %55 = load i8*, i8** %54, align 8
  %56 = bitcast i8* %55 to %struct.mpc_parser_t**
  %57 = getelementptr i8, i8* %55, i32 8
  store i8* %57, i8** %54, align 8
  br label %58

; <label>:58:                                     ; preds = %53, %47
  %59 = phi %struct.mpc_parser_t** [ %51, %47 ], [ %56, %53 ]
  %60 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %59, align 8
  %61 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %62 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %61, i32 0, i32 2
  %63 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %62, align 8
  %64 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %65 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %64, i32 0, i32 1
  %66 = load i32, i32* %65, align 8
  %67 = sub nsw i32 %66, 1
  %68 = sext i32 %67 to i64
  %69 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %63, i64 %68
  store %struct.mpc_parser_t* %60, %struct.mpc_parser_t** %69, align 8
  %70 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %71 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %70, i32 0, i32 2
  %72 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %71, align 8
  %73 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %74 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %73, i32 0, i32 1
  %75 = load i32, i32* %74, align 8
  %76 = sub nsw i32 %75, 1
  %77 = sext i32 %76 to i64
  %78 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %72, i64 %77
  %79 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %78, align 8
  %80 = icmp ne %struct.mpc_parser_t* %79, null
  br i1 %80, label %87, label %81

; <label>:81:                                     ; preds = %58
  %82 = load i32, i32* %6, align 4
  %83 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %84 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %83, i32 0, i32 1
  %85 = load i32, i32* %84, align 8
  %86 = call %struct.mpc_parser_t* (i8*, ...) @mpc_failf(i8* getelementptr inbounds ([52 x i8], [52 x i8]* @.str.149, i32 0, i32 0), i32 %82, i32 %85)
  store %struct.mpc_parser_t* %86, %struct.mpc_parser_t** %3, align 8
  br label %207

; <label>:87:                                     ; preds = %58
  br label %16

; <label>:88:                                     ; preds = %16
  %89 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %90 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %89, i32 0, i32 2
  %91 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %90, align 8
  %92 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %93 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %92, i32 0, i32 1
  %94 = load i32, i32* %93, align 8
  %95 = sub nsw i32 %94, 1
  %96 = sext i32 %95 to i64
  %97 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %91, i64 %96
  %98 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %97, align 8
  store %struct.mpc_parser_t* %98, %struct.mpc_parser_t** %3, align 8
  br label %207

; <label>:99:                                     ; preds = %2
  store i32 0, i32* %6, align 4
  br label %100

; <label>:100:                                    ; preds = %134, %99
  %101 = load i32, i32* %6, align 4
  %102 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %103 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %102, i32 0, i32 1
  %104 = load i32, i32* %103, align 8
  %105 = icmp slt i32 %101, %104
  br i1 %105, label %106, label %137

; <label>:106:                                    ; preds = %100
  %107 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %108 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %107, i32 0, i32 2
  %109 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %108, align 8
  %110 = load i32, i32* %6, align 4
  %111 = sext i32 %110 to i64
  %112 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %109, i64 %111
  %113 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %112, align 8
  store %struct.mpc_parser_t* %113, %struct.mpc_parser_t** %8, align 8
  %114 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %115 = icmp ne %struct.mpc_parser_t* %114, null
  br i1 %115, label %119, label %116

; <label>:116:                                    ; preds = %106
  %117 = load i8*, i8** %4, align 8
  %118 = call %struct.mpc_parser_t* (i8*, ...) @mpc_failf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.150, i32 0, i32 0), i8* %117)
  store %struct.mpc_parser_t* %118, %struct.mpc_parser_t** %3, align 8
  br label %207

; <label>:119:                                    ; preds = %106
  %120 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %121 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %120, i32 0, i32 1
  %122 = load i8*, i8** %121, align 8
  %123 = icmp ne i8* %122, null
  br i1 %123, label %124, label %133

; <label>:124:                                    ; preds = %119
  %125 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %126 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %125, i32 0, i32 1
  %127 = load i8*, i8** %126, align 8
  %128 = load i8*, i8** %4, align 8
  %129 = call i32 @strcmp(i8* %127, i8* %128) #7
  %130 = icmp eq i32 %129, 0
  br i1 %130, label %131, label %133

; <label>:131:                                    ; preds = %124
  %132 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  store %struct.mpc_parser_t* %132, %struct.mpc_parser_t** %3, align 8
  br label %207

; <label>:133:                                    ; preds = %124, %119
  br label %134

; <label>:134:                                    ; preds = %133
  %135 = load i32, i32* %6, align 4
  %136 = add nsw i32 %135, 1
  store i32 %136, i32* %6, align 4
  br label %100

; <label>:137:                                    ; preds = %100
  br label %138

; <label>:138:                                    ; preds = %137, %206
  %139 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %140 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %139, i32 0, i32 0
  %141 = load [1 x %struct.__va_list_tag]*, [1 x %struct.__va_list_tag]** %140, align 8
  %142 = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %141, i32 0, i32 0
  %143 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %142, i32 0, i32 0
  %144 = load i32, i32* %143, align 8
  %145 = icmp ule i32 %144, 40
  br i1 %145, label %146, label %152

; <label>:146:                                    ; preds = %138
  %147 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %142, i32 0, i32 3
  %148 = load i8*, i8** %147, align 8
  %149 = getelementptr i8, i8* %148, i32 %144
  %150 = bitcast i8* %149 to %struct.mpc_parser_t**
  %151 = add i32 %144, 8
  store i32 %151, i32* %143, align 8
  br label %157

; <label>:152:                                    ; preds = %138
  %153 = getelementptr inbounds %struct.__va_list_tag, %struct.__va_list_tag* %142, i32 0, i32 2
  %154 = load i8*, i8** %153, align 8
  %155 = bitcast i8* %154 to %struct.mpc_parser_t**
  %156 = getelementptr i8, i8* %154, i32 8
  store i8* %156, i8** %153, align 8
  br label %157

; <label>:157:                                    ; preds = %152, %146
  %158 = phi %struct.mpc_parser_t** [ %150, %146 ], [ %155, %152 ]
  %159 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %158, align 8
  store %struct.mpc_parser_t* %159, %struct.mpc_parser_t** %7, align 8
  %160 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %161 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %160, i32 0, i32 1
  %162 = load i32, i32* %161, align 8
  %163 = add nsw i32 %162, 1
  store i32 %163, i32* %161, align 8
  %164 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %165 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %164, i32 0, i32 2
  %166 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %165, align 8
  %167 = bitcast %struct.mpc_parser_t** %166 to i8*
  %168 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %169 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %168, i32 0, i32 1
  %170 = load i32, i32* %169, align 8
  %171 = sext i32 %170 to i64
  %172 = mul i64 8, %171
  %173 = call i8* @realloc(i8* %167, i64 %172) #5
  %174 = bitcast i8* %173 to %struct.mpc_parser_t**
  %175 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %176 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %175, i32 0, i32 2
  store %struct.mpc_parser_t** %174, %struct.mpc_parser_t*** %176, align 8
  %177 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %178 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %179 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %178, i32 0, i32 2
  %180 = load %struct.mpc_parser_t**, %struct.mpc_parser_t*** %179, align 8
  %181 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %182 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %181, i32 0, i32 1
  %183 = load i32, i32* %182, align 8
  %184 = sub nsw i32 %183, 1
  %185 = sext i32 %184 to i64
  %186 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %180, i64 %185
  store %struct.mpc_parser_t* %177, %struct.mpc_parser_t** %186, align 8
  %187 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %188 = icmp ne %struct.mpc_parser_t* %187, null
  br i1 %188, label %192, label %189

; <label>:189:                                    ; preds = %157
  %190 = load i8*, i8** %4, align 8
  %191 = call %struct.mpc_parser_t* (i8*, ...) @mpc_failf(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.150, i32 0, i32 0), i8* %190)
  store %struct.mpc_parser_t* %191, %struct.mpc_parser_t** %3, align 8
  br label %207

; <label>:192:                                    ; preds = %157
  %193 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %194 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %193, i32 0, i32 1
  %195 = load i8*, i8** %194, align 8
  %196 = icmp ne i8* %195, null
  br i1 %196, label %197, label %206

; <label>:197:                                    ; preds = %192
  %198 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  %199 = getelementptr inbounds %struct.mpc_parser_t, %struct.mpc_parser_t* %198, i32 0, i32 1
  %200 = load i8*, i8** %199, align 8
  %201 = load i8*, i8** %4, align 8
  %202 = call i32 @strcmp(i8* %200, i8* %201) #7
  %203 = icmp eq i32 %202, 0
  br i1 %203, label %204, label %206

; <label>:204:                                    ; preds = %197
  %205 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %7, align 8
  store %struct.mpc_parser_t* %205, %struct.mpc_parser_t** %3, align 8
  br label %207

; <label>:206:                                    ; preds = %197, %192
  br label %138

; <label>:207:                                    ; preds = %204, %189, %131, %116, %88, %81
  %208 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %3, align 8
  ret %struct.mpc_parser_t* %208
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @is_number(i8*) #0 {
  %2 = alloca i32, align 4
  %3 = alloca i8*, align 8
  %4 = alloca i64, align 8
  store i8* %0, i8** %3, align 8
  store i64 0, i64* %4, align 8
  br label %5

; <label>:5:                                      ; preds = %20, %1
  %6 = load i64, i64* %4, align 8
  %7 = load i8*, i8** %3, align 8
  %8 = call i64 @strlen(i8* %7) #7
  %9 = icmp ult i64 %6, %8
  br i1 %9, label %10, label %23

; <label>:10:                                     ; preds = %5
  %11 = load i8*, i8** %3, align 8
  %12 = load i64, i64* %4, align 8
  %13 = getelementptr inbounds i8, i8* %11, i64 %12
  %14 = load i8, i8* %13, align 1
  %15 = sext i8 %14 to i32
  %16 = call i8* @strchr(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.28, i32 0, i32 0), i32 %15) #7
  %17 = icmp ne i8* %16, null
  br i1 %17, label %19, label %18

; <label>:18:                                     ; preds = %10
  store i32 0, i32* %2, align 4
  br label %24

; <label>:19:                                     ; preds = %10
  br label %20

; <label>:20:                                     ; preds = %19
  %21 = load i64, i64* %4, align 8
  %22 = add i64 %21, 1
  store i64 %22, i64* %4, align 8
  br label %5

; <label>:23:                                     ; preds = %5
  store i32 1, i32* %2, align 4
  br label %24

; <label>:24:                                     ; preds = %23, %18
  %25 = load i32, i32* %2, align 4
  ret i32 %25
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpca_stmt_fold(i32, i8**) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i8**, align 8
  %5 = alloca i32, align 4
  %6 = alloca %struct.mpca_stmt_t**, align 8
  store i32 %0, i32* %3, align 4
  store i8** %1, i8*** %4, align 8
  %7 = load i32, i32* %3, align 4
  %8 = add nsw i32 %7, 1
  %9 = sext i32 %8 to i64
  %10 = mul i64 8, %9
  %11 = call noalias i8* @malloc(i64 %10) #5
  %12 = bitcast i8* %11 to %struct.mpca_stmt_t**
  store %struct.mpca_stmt_t** %12, %struct.mpca_stmt_t*** %6, align 8
  store i32 0, i32* %5, align 4
  br label %13

; <label>:13:                                     ; preds = %28, %2
  %14 = load i32, i32* %5, align 4
  %15 = load i32, i32* %3, align 4
  %16 = icmp slt i32 %14, %15
  br i1 %16, label %17, label %31

; <label>:17:                                     ; preds = %13
  %18 = load i8**, i8*** %4, align 8
  %19 = load i32, i32* %5, align 4
  %20 = sext i32 %19 to i64
  %21 = getelementptr inbounds i8*, i8** %18, i64 %20
  %22 = load i8*, i8** %21, align 8
  %23 = bitcast i8* %22 to %struct.mpca_stmt_t*
  %24 = load %struct.mpca_stmt_t**, %struct.mpca_stmt_t*** %6, align 8
  %25 = load i32, i32* %5, align 4
  %26 = sext i32 %25 to i64
  %27 = getelementptr inbounds %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %24, i64 %26
  store %struct.mpca_stmt_t* %23, %struct.mpca_stmt_t** %27, align 8
  br label %28

; <label>:28:                                     ; preds = %17
  %29 = load i32, i32* %5, align 4
  %30 = add nsw i32 %29, 1
  store i32 %30, i32* %5, align 4
  br label %13

; <label>:31:                                     ; preds = %13
  %32 = load %struct.mpca_stmt_t**, %struct.mpca_stmt_t*** %6, align 8
  %33 = load i32, i32* %3, align 4
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %32, i64 %34
  store %struct.mpca_stmt_t* null, %struct.mpca_stmt_t** %35, align 8
  %36 = load %struct.mpca_stmt_t**, %struct.mpca_stmt_t*** %6, align 8
  %37 = bitcast %struct.mpca_stmt_t** %36 to i8*
  ret i8* %37
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @mpca_stmt_list_delete(i8*) #0 {
  %2 = alloca i8*, align 8
  %3 = alloca %struct.mpca_stmt_t**, align 8
  %4 = alloca %struct.mpca_stmt_t*, align 8
  store i8* %0, i8** %2, align 8
  %5 = load i8*, i8** %2, align 8
  %6 = bitcast i8* %5 to %struct.mpca_stmt_t**
  store %struct.mpca_stmt_t** %6, %struct.mpca_stmt_t*** %3, align 8
  br label %7

; <label>:7:                                      ; preds = %11, %1
  %8 = load %struct.mpca_stmt_t**, %struct.mpca_stmt_t*** %3, align 8
  %9 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %8, align 8
  %10 = icmp ne %struct.mpca_stmt_t* %9, null
  br i1 %10, label %11, label %28

; <label>:11:                                     ; preds = %7
  %12 = load %struct.mpca_stmt_t**, %struct.mpca_stmt_t*** %3, align 8
  %13 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %12, align 8
  store %struct.mpca_stmt_t* %13, %struct.mpca_stmt_t** %4, align 8
  %14 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %4, align 8
  %15 = getelementptr inbounds %struct.mpca_stmt_t, %struct.mpca_stmt_t* %14, i32 0, i32 0
  %16 = load i8*, i8** %15, align 8
  call void @free(i8* %16) #5
  %17 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %4, align 8
  %18 = getelementptr inbounds %struct.mpca_stmt_t, %struct.mpca_stmt_t* %17, i32 0, i32 1
  %19 = load i8*, i8** %18, align 8
  call void @free(i8* %19) #5
  %20 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %4, align 8
  %21 = getelementptr inbounds %struct.mpca_stmt_t, %struct.mpca_stmt_t* %20, i32 0, i32 2
  %22 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %21, align 8
  %23 = bitcast %struct.mpc_parser_t* %22 to i8*
  call void @mpc_soft_delete(i8* %23)
  %24 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %4, align 8
  %25 = bitcast %struct.mpca_stmt_t* %24 to i8*
  call void @free(i8* %25) #5
  %26 = load %struct.mpca_stmt_t**, %struct.mpca_stmt_t*** %3, align 8
  %27 = getelementptr inbounds %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %26, i32 1
  store %struct.mpca_stmt_t** %27, %struct.mpca_stmt_t*** %3, align 8
  br label %7

; <label>:28:                                     ; preds = %7
  %29 = load i8*, i8** %2, align 8
  call void @free(i8* %29) #5
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpca_stmt_list_apply_to(i8*, i8*) #0 {
  %3 = alloca i8*, align 8
  %4 = alloca i8*, align 8
  %5 = alloca %struct.mpca_grammar_st_t*, align 8
  %6 = alloca %struct.mpca_stmt_t*, align 8
  %7 = alloca %struct.mpca_stmt_t**, align 8
  %8 = alloca %struct.mpc_parser_t*, align 8
  store i8* %0, i8** %3, align 8
  store i8* %1, i8** %4, align 8
  %9 = load i8*, i8** %4, align 8
  %10 = bitcast i8* %9 to %struct.mpca_grammar_st_t*
  store %struct.mpca_grammar_st_t* %10, %struct.mpca_grammar_st_t** %5, align 8
  %11 = load i8*, i8** %3, align 8
  %12 = bitcast i8* %11 to %struct.mpca_stmt_t**
  store %struct.mpca_stmt_t** %12, %struct.mpca_stmt_t*** %7, align 8
  br label %13

; <label>:13:                                     ; preds = %52, %2
  %14 = load %struct.mpca_stmt_t**, %struct.mpca_stmt_t*** %7, align 8
  %15 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %14, align 8
  %16 = icmp ne %struct.mpca_stmt_t* %15, null
  br i1 %16, label %17, label %71

; <label>:17:                                     ; preds = %13
  %18 = load %struct.mpca_stmt_t**, %struct.mpca_stmt_t*** %7, align 8
  %19 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %18, align 8
  store %struct.mpca_stmt_t* %19, %struct.mpca_stmt_t** %6, align 8
  %20 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %6, align 8
  %21 = getelementptr inbounds %struct.mpca_stmt_t, %struct.mpca_stmt_t* %20, i32 0, i32 0
  %22 = load i8*, i8** %21, align 8
  %23 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %24 = call %struct.mpc_parser_t* @mpca_grammar_find_parser(i8* %22, %struct.mpca_grammar_st_t* %23)
  store %struct.mpc_parser_t* %24, %struct.mpc_parser_t** %8, align 8
  %25 = load %struct.mpca_grammar_st_t*, %struct.mpca_grammar_st_t** %5, align 8
  %26 = getelementptr inbounds %struct.mpca_grammar_st_t, %struct.mpca_grammar_st_t* %25, i32 0, i32 3
  %27 = load i32, i32* %26, align 8
  %28 = and i32 %27, 1
  %29 = icmp ne i32 %28, 0
  br i1 %29, label %30, label %37

; <label>:30:                                     ; preds = %17
  %31 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %6, align 8
  %32 = getelementptr inbounds %struct.mpca_stmt_t, %struct.mpca_stmt_t* %31, i32 0, i32 2
  %33 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %32, align 8
  %34 = call %struct.mpc_parser_t* @mpc_predictive(%struct.mpc_parser_t* %33)
  %35 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %6, align 8
  %36 = getelementptr inbounds %struct.mpca_stmt_t, %struct.mpca_stmt_t* %35, i32 0, i32 2
  store %struct.mpc_parser_t* %34, %struct.mpc_parser_t** %36, align 8
  br label %37

; <label>:37:                                     ; preds = %30, %17
  %38 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %6, align 8
  %39 = getelementptr inbounds %struct.mpca_stmt_t, %struct.mpca_stmt_t* %38, i32 0, i32 1
  %40 = load i8*, i8** %39, align 8
  %41 = icmp ne i8* %40, null
  br i1 %41, label %42, label %52

; <label>:42:                                     ; preds = %37
  %43 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %6, align 8
  %44 = getelementptr inbounds %struct.mpca_stmt_t, %struct.mpca_stmt_t* %43, i32 0, i32 2
  %45 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %44, align 8
  %46 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %6, align 8
  %47 = getelementptr inbounds %struct.mpca_stmt_t, %struct.mpca_stmt_t* %46, i32 0, i32 1
  %48 = load i8*, i8** %47, align 8
  %49 = call %struct.mpc_parser_t* @mpc_expect(%struct.mpc_parser_t* %45, i8* %48)
  %50 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %6, align 8
  %51 = getelementptr inbounds %struct.mpca_stmt_t, %struct.mpca_stmt_t* %50, i32 0, i32 2
  store %struct.mpc_parser_t* %49, %struct.mpc_parser_t** %51, align 8
  br label %52

; <label>:52:                                     ; preds = %42, %37
  %53 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %6, align 8
  %54 = getelementptr inbounds %struct.mpca_stmt_t, %struct.mpca_stmt_t* %53, i32 0, i32 2
  %55 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %54, align 8
  call void @mpc_optimise(%struct.mpc_parser_t* %55)
  %56 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %8, align 8
  %57 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %6, align 8
  %58 = getelementptr inbounds %struct.mpca_stmt_t, %struct.mpca_stmt_t* %57, i32 0, i32 2
  %59 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %58, align 8
  %60 = call %struct.mpc_parser_t* @mpc_define(%struct.mpc_parser_t* %56, %struct.mpc_parser_t* %59)
  %61 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %6, align 8
  %62 = getelementptr inbounds %struct.mpca_stmt_t, %struct.mpca_stmt_t* %61, i32 0, i32 0
  %63 = load i8*, i8** %62, align 8
  call void @free(i8* %63) #5
  %64 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %6, align 8
  %65 = getelementptr inbounds %struct.mpca_stmt_t, %struct.mpca_stmt_t* %64, i32 0, i32 1
  %66 = load i8*, i8** %65, align 8
  call void @free(i8* %66) #5
  %67 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %6, align 8
  %68 = bitcast %struct.mpca_stmt_t* %67 to i8*
  call void @free(i8* %68) #5
  %69 = load %struct.mpca_stmt_t**, %struct.mpca_stmt_t*** %7, align 8
  %70 = getelementptr inbounds %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %69, i32 1
  store %struct.mpca_stmt_t** %70, %struct.mpca_stmt_t*** %7, align 8
  br label %13

; <label>:71:                                     ; preds = %13
  %72 = load i8*, i8** %3, align 8
  call void @free(i8* %72) #5
  ret i8* null
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i8* @mpca_stmt_afold(i32, i8**) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i8**, align 8
  %5 = alloca %struct.mpca_stmt_t*, align 8
  store i32 %0, i32* %3, align 4
  store i8** %1, i8*** %4, align 8
  %6 = call noalias i8* @malloc(i64 24) #5
  %7 = bitcast i8* %6 to %struct.mpca_stmt_t*
  store %struct.mpca_stmt_t* %7, %struct.mpca_stmt_t** %5, align 8
  %8 = load i8**, i8*** %4, align 8
  %9 = getelementptr inbounds i8*, i8** %8, i64 0
  %10 = load i8*, i8** %9, align 8
  %11 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %5, align 8
  %12 = getelementptr inbounds %struct.mpca_stmt_t, %struct.mpca_stmt_t* %11, i32 0, i32 0
  store i8* %10, i8** %12, align 8
  %13 = load i8**, i8*** %4, align 8
  %14 = getelementptr inbounds i8*, i8** %13, i64 1
  %15 = load i8*, i8** %14, align 8
  %16 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %5, align 8
  %17 = getelementptr inbounds %struct.mpca_stmt_t, %struct.mpca_stmt_t* %16, i32 0, i32 1
  store i8* %15, i8** %17, align 8
  %18 = load i8**, i8*** %4, align 8
  %19 = bitcast i8** %18 to %struct.mpc_parser_t**
  %20 = getelementptr inbounds %struct.mpc_parser_t*, %struct.mpc_parser_t** %19, i64 3
  %21 = load %struct.mpc_parser_t*, %struct.mpc_parser_t** %20, align 8
  %22 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %5, align 8
  %23 = getelementptr inbounds %struct.mpca_stmt_t, %struct.mpca_stmt_t* %22, i32 0, i32 2
  store %struct.mpc_parser_t* %21, %struct.mpc_parser_t** %23, align 8
  %24 = load i32, i32* %3, align 4
  %25 = load i8**, i8*** %4, align 8
  %26 = getelementptr inbounds i8*, i8** %25, i64 2
  %27 = load i8*, i8** %26, align 8
  call void @free(i8* %27) #5
  %28 = load i8**, i8*** %4, align 8
  %29 = getelementptr inbounds i8*, i8** %28, i64 4
  %30 = load i8*, i8** %29, align 8
  call void @free(i8* %30) #5
  %31 = load %struct.mpca_stmt_t*, %struct.mpca_stmt_t** %5, align 8
  %32 = bitcast %struct.mpca_stmt_t* %31 to i8*
  ret i8* %32
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { argmemonly nounwind }
attributes #5 = { nounwind }
attributes #6 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { nounwind readonly }
attributes #8 = { nounwind readnone }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.0 (tags/RELEASE_600/final)"}
