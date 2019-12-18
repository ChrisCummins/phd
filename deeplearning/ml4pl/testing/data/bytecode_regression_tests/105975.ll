; ModuleID = '-'
source_filename = "-"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.LSH_query_t = type { %struct.LSH_t*, %struct.cass_dataset_t*, i32, i32, i32, %struct.bitmap_t*, i32**, i32*, %struct.ptb_vec_t**, i32*, %struct.ptb_vec_t*, %struct.ptb_vec_t**, %struct.cass_list_entry_t*, %struct.cass_list_entry_t**, i32*, i32*, float*, i32, float, float, float }
%struct.bitmap_t = type { i32, i32, i8* }
%struct.ptb_vec_t = type { i64, float, i32, i32, float }
%struct.cass_list_entry_t = type { i32, float }
%struct.LSH_t = type { i32, i32, i32, i32, i32, float*, float**, float*, i32**, %struct.ohash_t*, i32**, i32*, %struct.LSH_est_t*, %struct.LSH_recall_t }
%struct.ohash_t = type { i32, %struct.bucket_t* }
%struct.bucket_t = type { i32, i32, i32, i32* }
%struct.LSH_est_t = type { i32, i32, double, double, double, double, double, double, double**, double** }
%struct.LSH_recall_t = type { i32, i32, float, float, float** }
%struct.cass_dataset_t = type { i32, i32, i32, i32, i32, i32, i8*, i32, i32, %struct._cass_vecset_t* }
%struct._cass_vecset_t = type { i32, i32 }
%struct._cass_vec_t = type { float, i32, %union.anon }
%union.anon = type { [14 x float] }

@.str = private unnamed_addr constant [19 x i8] c"query->tmp != NULL\00", align 1
@.str.1 = private unnamed_addr constant [8 x i8] c"<stdin>\00", align 1
@__PRETTY_FUNCTION__.LSH_query_init = private unnamed_addr constant [101 x i8] c"void LSH_query_init(LSH_query_t *, LSH_t *, cass_dataset_t *, cass_size_t, cass_size_t, cass_size_t)\00", align 1
@.str.2 = private unnamed_addr constant [20 x i8] c"query->tmp2 != NULL\00", align 1
@.str.3 = private unnamed_addr constant [19 x i8] c"query->ptb != NULL\00", align 1
@.str.4 = private unnamed_addr constant [12 x i8] c"scr != NULL\00", align 1
@.str.5 = private unnamed_addr constant [23 x i8] c"query->ptb_set != NULL\00", align 1
@.str.6 = private unnamed_addr constant [23 x i8] c"query->ptb_vec != NULL\00", align 1
@.str.7 = private unnamed_addr constant [24 x i8] c"query->ptb_step != NULL\00", align 1
@.str.8 = private unnamed_addr constant [20 x i8] c"query->topk != NULL\00", align 1
@.str.9 = private unnamed_addr constant [21 x i8] c"query->_topk != NULL\00", align 1
@.str.10 = private unnamed_addr constant [17 x i8] c"query->C != NULL\00", align 1
@.str.11 = private unnamed_addr constant [17 x i8] c"query->H != NULL\00", align 1
@.str.12 = private unnamed_addr constant [17 x i8] c"query->S != NULL\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define void @LSH_query_init(%struct.LSH_query_t*, %struct.LSH_t*, %struct.cass_dataset_t*, i32, i32, i32) #0 {
  %7 = alloca %struct.LSH_query_t*, align 8
  %8 = alloca %struct.LSH_t*, align 8
  %9 = alloca %struct.cass_dataset_t*, align 8
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca %struct.ptb_vec_t*, align 8
  store %struct.LSH_query_t* %0, %struct.LSH_query_t** %7, align 8
  store %struct.LSH_t* %1, %struct.LSH_t** %8, align 8
  store %struct.cass_dataset_t* %2, %struct.cass_dataset_t** %9, align 8
  store i32 %3, i32* %10, align 4
  store i32 %4, i32* %11, align 4
  store i32 %5, i32* %12, align 4
  %14 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %15 = bitcast %struct.LSH_query_t* %14 to i8*
  call void @llvm.memset.p0i8.i64(i8* %15, i8 0, i64 144, i32 8, i1 false)
  %16 = load %struct.LSH_t*, %struct.LSH_t** %8, align 8
  %17 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %18 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %17, i32 0, i32 0
  store %struct.LSH_t* %16, %struct.LSH_t** %18, align 8
  %19 = load %struct.cass_dataset_t*, %struct.cass_dataset_t** %9, align 8
  %20 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %21 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %20, i32 0, i32 1
  store %struct.cass_dataset_t* %19, %struct.cass_dataset_t** %21, align 8
  %22 = load i32, i32* %10, align 4
  %23 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %24 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %23, i32 0, i32 2
  store i32 %22, i32* %24, align 8
  %25 = load i32, i32* %11, align 4
  %26 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %27 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %26, i32 0, i32 3
  store i32 %25, i32* %27, align 4
  %28 = load i32, i32* %12, align 4
  %29 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %30 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %29, i32 0, i32 4
  store i32 %28, i32* %30, align 8
  %31 = load %struct.LSH_t*, %struct.LSH_t** %8, align 8
  %32 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %31, i32 0, i32 4
  %33 = load i32, i32* %32, align 8
  %34 = call %struct.bitmap_t* @bitmap_new(i32 %33)
  %35 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %36 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %35, i32 0, i32 5
  store %struct.bitmap_t* %34, %struct.bitmap_t** %36, align 8
  %37 = load %struct.LSH_t*, %struct.LSH_t** %8, align 8
  %38 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %37, i32 0, i32 2
  %39 = load i32, i32* %38, align 8
  %40 = load %struct.LSH_t*, %struct.LSH_t** %8, align 8
  %41 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %40, i32 0, i32 1
  %42 = load i32, i32* %41, align 4
  %43 = call i8** @__matrix_alloc(i32 %39, i32 %42, i32 4)
  %44 = bitcast i8** %43 to i32**
  %45 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %46 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %45, i32 0, i32 6
  store i32** %44, i32*** %46, align 8
  %47 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %48 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %47, i32 0, i32 6
  %49 = load i32**, i32*** %48, align 8
  %50 = icmp ne i32** %49, null
  br i1 %50, label %51, label %52

; <label>:51:                                     ; preds = %6
  br label %54

; <label>:52:                                     ; preds = %6
  call void @__assert_fail(i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.1, i32 0, i32 0), i32 1861, i8* getelementptr inbounds ([101 x i8], [101 x i8]* @__PRETTY_FUNCTION__.LSH_query_init, i32 0, i32 0)) #7
  unreachable
                                                  ; No predecessors!
  br label %54

; <label>:54:                                     ; preds = %53, %51
  %55 = load %struct.LSH_t*, %struct.LSH_t** %8, align 8
  %56 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %55, i32 0, i32 2
  %57 = load i32, i32* %56, align 8
  %58 = zext i32 %57 to i64
  %59 = mul i64 4, %58
  %60 = call noalias i8* @malloc(i64 %59) #6
  %61 = bitcast i8* %60 to i32*
  %62 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %63 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %62, i32 0, i32 7
  store i32* %61, i32** %63, align 8
  %64 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %65 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %64, i32 0, i32 7
  %66 = load i32*, i32** %65, align 8
  %67 = icmp ne i32* %66, null
  br i1 %67, label %68, label %69

; <label>:68:                                     ; preds = %54
  br label %71

; <label>:69:                                     ; preds = %54
  call void @__assert_fail(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.2, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.1, i32 0, i32 0), i32 1863, i8* getelementptr inbounds ([101 x i8], [101 x i8]* @__PRETTY_FUNCTION__.LSH_query_init, i32 0, i32 0)) #7
  unreachable
                                                  ; No predecessors!
  br label %71

; <label>:71:                                     ; preds = %70, %68
  %72 = load i32, i32* %11, align 4
  %73 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %74 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %73, i32 0, i32 0
  %75 = load %struct.LSH_t*, %struct.LSH_t** %74, align 8
  %76 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %75, i32 0, i32 1
  %77 = load i32, i32* %76, align 4
  %78 = mul i32 %77, 2
  %79 = call i8** @__matrix_alloc(i32 %72, i32 %78, i32 24)
  %80 = bitcast i8** %79 to %struct.ptb_vec_t**
  %81 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %82 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %81, i32 0, i32 8
  store %struct.ptb_vec_t** %80, %struct.ptb_vec_t*** %82, align 8
  %83 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %84 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %83, i32 0, i32 8
  %85 = load %struct.ptb_vec_t**, %struct.ptb_vec_t*** %84, align 8
  %86 = icmp ne %struct.ptb_vec_t** %85, null
  br i1 %86, label %87, label %88

; <label>:87:                                     ; preds = %71
  br label %90

; <label>:88:                                     ; preds = %71
  call void @__assert_fail(i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.3, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.1, i32 0, i32 0), i32 1866, i8* getelementptr inbounds ([101 x i8], [101 x i8]* @__PRETTY_FUNCTION__.LSH_query_init, i32 0, i32 0)) #7
  unreachable
                                                  ; No predecessors!
  br label %90

; <label>:90:                                     ; preds = %89, %87
  %91 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %92 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %91, i32 0, i32 0
  %93 = load %struct.LSH_t*, %struct.LSH_t** %92, align 8
  %94 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %93, i32 0, i32 1
  %95 = load i32, i32* %94, align 4
  %96 = call %struct.ptb_vec_t* @gen_score(i32 %95)
  store %struct.ptb_vec_t* %96, %struct.ptb_vec_t** %13, align 8
  %97 = load %struct.ptb_vec_t*, %struct.ptb_vec_t** %13, align 8
  %98 = icmp ne %struct.ptb_vec_t* %97, null
  br i1 %98, label %99, label %100

; <label>:99:                                     ; preds = %90
  br label %102

; <label>:100:                                    ; preds = %90
  call void @__assert_fail(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.4, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.1, i32 0, i32 0), i32 1877, i8* getelementptr inbounds ([101 x i8], [101 x i8]* @__PRETTY_FUNCTION__.LSH_query_init, i32 0, i32 0)) #7
  unreachable
                                                  ; No predecessors!
  br label %102

; <label>:102:                                    ; preds = %101, %99
  %103 = load i32, i32* %12, align 4
  %104 = zext i32 %103 to i64
  %105 = call noalias i8* @calloc(i64 24, i64 %104) #6
  %106 = bitcast i8* %105 to %struct.ptb_vec_t*
  %107 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %108 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %107, i32 0, i32 10
  store %struct.ptb_vec_t* %106, %struct.ptb_vec_t** %108, align 8
  %109 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %110 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %109, i32 0, i32 10
  %111 = load %struct.ptb_vec_t*, %struct.ptb_vec_t** %110, align 8
  %112 = icmp ne %struct.ptb_vec_t* %111, null
  br i1 %112, label %113, label %114

; <label>:113:                                    ; preds = %102
  br label %116

; <label>:114:                                    ; preds = %102
  call void @__assert_fail(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.5, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.1, i32 0, i32 0), i32 1879, i8* getelementptr inbounds ([101 x i8], [101 x i8]* @__PRETTY_FUNCTION__.LSH_query_init, i32 0, i32 0)) #7
  unreachable
                                                  ; No predecessors!
  br label %116

; <label>:116:                                    ; preds = %115, %113
  %117 = load %struct.ptb_vec_t*, %struct.ptb_vec_t** %13, align 8
  %118 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %119 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %118, i32 0, i32 10
  %120 = load %struct.ptb_vec_t*, %struct.ptb_vec_t** %119, align 8
  %121 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %122 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %121, i32 0, i32 0
  %123 = load %struct.LSH_t*, %struct.LSH_t** %122, align 8
  %124 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %123, i32 0, i32 1
  %125 = load i32, i32* %124, align 4
  %126 = load i32, i32* %12, align 4
  %127 = call i32 @gen_perturb_set(%struct.ptb_vec_t* %117, %struct.ptb_vec_t* %120, i32 %125, i32 %126)
  %128 = load i32, i32* %11, align 4
  %129 = load i32, i32* %12, align 4
  %130 = call i8** @__matrix_alloc(i32 %128, i32 %129, i32 24)
  %131 = bitcast i8** %130 to %struct.ptb_vec_t**
  %132 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %133 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %132, i32 0, i32 11
  store %struct.ptb_vec_t** %131, %struct.ptb_vec_t*** %133, align 8
  %134 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %135 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %134, i32 0, i32 11
  %136 = load %struct.ptb_vec_t**, %struct.ptb_vec_t*** %135, align 8
  %137 = icmp ne %struct.ptb_vec_t** %136, null
  br i1 %137, label %138, label %139

; <label>:138:                                    ; preds = %116
  br label %141

; <label>:139:                                    ; preds = %116
  call void @__assert_fail(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.6, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.1, i32 0, i32 0), i32 1882, i8* getelementptr inbounds ([101 x i8], [101 x i8]* @__PRETTY_FUNCTION__.LSH_query_init, i32 0, i32 0)) #7
  unreachable
                                                  ; No predecessors!
  br label %141

; <label>:141:                                    ; preds = %140, %138
  %142 = load i32, i32* %11, align 4
  %143 = zext i32 %142 to i64
  %144 = call noalias i8* @calloc(i64 4, i64 %143) #6
  %145 = bitcast i8* %144 to i32*
  %146 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %147 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %146, i32 0, i32 9
  store i32* %145, i32** %147, align 8
  %148 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %149 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %148, i32 0, i32 9
  %150 = load i32*, i32** %149, align 8
  %151 = icmp ne i32* %150, null
  br i1 %151, label %152, label %153

; <label>:152:                                    ; preds = %141
  br label %155

; <label>:153:                                    ; preds = %141
  call void @__assert_fail(i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str.7, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.1, i32 0, i32 0), i32 1884, i8* getelementptr inbounds ([101 x i8], [101 x i8]* @__PRETTY_FUNCTION__.LSH_query_init, i32 0, i32 0)) #7
  unreachable
                                                  ; No predecessors!
  br label %155

; <label>:155:                                    ; preds = %154, %152
  %156 = load %struct.ptb_vec_t*, %struct.ptb_vec_t** %13, align 8
  %157 = bitcast %struct.ptb_vec_t* %156 to i8*
  call void @free(i8* %157) #6
  %158 = load i32, i32* %10, align 4
  %159 = zext i32 %158 to i64
  %160 = call noalias i8* @calloc(i64 8, i64 %159) #6
  %161 = bitcast i8* %160 to %struct.cass_list_entry_t*
  %162 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %163 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %162, i32 0, i32 12
  store %struct.cass_list_entry_t* %161, %struct.cass_list_entry_t** %163, align 8
  %164 = load i32, i32* %11, align 4
  %165 = load i32, i32* %10, align 4
  %166 = call i8** @__matrix_alloc(i32 %164, i32 %165, i32 8)
  %167 = bitcast i8** %166 to %struct.cass_list_entry_t**
  %168 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %169 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %168, i32 0, i32 13
  store %struct.cass_list_entry_t** %167, %struct.cass_list_entry_t*** %169, align 8
  %170 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %171 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %170, i32 0, i32 12
  %172 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %171, align 8
  %173 = icmp ne %struct.cass_list_entry_t* %172, null
  br i1 %173, label %174, label %175

; <label>:174:                                    ; preds = %155
  br label %177

; <label>:175:                                    ; preds = %155
  call void @__assert_fail(i8* getelementptr inbounds ([20 x i8], [20 x i8]* @.str.8, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.1, i32 0, i32 0), i32 1891, i8* getelementptr inbounds ([101 x i8], [101 x i8]* @__PRETTY_FUNCTION__.LSH_query_init, i32 0, i32 0)) #7
  unreachable
                                                  ; No predecessors!
  br label %177

; <label>:177:                                    ; preds = %176, %174
  %178 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %179 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %178, i32 0, i32 13
  %180 = load %struct.cass_list_entry_t**, %struct.cass_list_entry_t*** %179, align 8
  %181 = icmp ne %struct.cass_list_entry_t** %180, null
  br i1 %181, label %182, label %183

; <label>:182:                                    ; preds = %177
  br label %185

; <label>:183:                                    ; preds = %177
  call void @__assert_fail(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @.str.9, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.1, i32 0, i32 0), i32 1892, i8* getelementptr inbounds ([101 x i8], [101 x i8]* @__PRETTY_FUNCTION__.LSH_query_init, i32 0, i32 0)) #7
  unreachable
                                                  ; No predecessors!
  br label %185

; <label>:185:                                    ; preds = %184, %182
  %186 = load i32, i32* %11, align 4
  %187 = zext i32 %186 to i64
  %188 = mul i64 %187, 4
  %189 = call noalias i8* @malloc(i64 %188) #6
  %190 = bitcast i8* %189 to i32*
  %191 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %192 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %191, i32 0, i32 14
  store i32* %190, i32** %192, align 8
  %193 = load i32, i32* %11, align 4
  %194 = zext i32 %193 to i64
  %195 = mul i64 %194, 4
  %196 = call noalias i8* @malloc(i64 %195) #6
  %197 = bitcast i8* %196 to i32*
  %198 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %199 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %198, i32 0, i32 15
  store i32* %197, i32** %199, align 8
  %200 = load i32, i32* %11, align 4
  %201 = zext i32 %200 to i64
  %202 = mul i64 %201, 4
  %203 = call noalias i8* @malloc(i64 %202) #6
  %204 = bitcast i8* %203 to float*
  %205 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %206 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %205, i32 0, i32 16
  store float* %204, float** %206, align 8
  %207 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %208 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %207, i32 0, i32 14
  %209 = load i32*, i32** %208, align 8
  %210 = icmp ne i32* %209, null
  br i1 %210, label %211, label %212

; <label>:211:                                    ; preds = %185
  br label %214

; <label>:212:                                    ; preds = %185
  call void @__assert_fail(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.10, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.1, i32 0, i32 0), i32 1898, i8* getelementptr inbounds ([101 x i8], [101 x i8]* @__PRETTY_FUNCTION__.LSH_query_init, i32 0, i32 0)) #7
  unreachable
                                                  ; No predecessors!
  br label %214

; <label>:214:                                    ; preds = %213, %211
  %215 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %216 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %215, i32 0, i32 15
  %217 = load i32*, i32** %216, align 8
  %218 = icmp ne i32* %217, null
  br i1 %218, label %219, label %220

; <label>:219:                                    ; preds = %214
  br label %222

; <label>:220:                                    ; preds = %214
  call void @__assert_fail(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.11, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.1, i32 0, i32 0), i32 1899, i8* getelementptr inbounds ([101 x i8], [101 x i8]* @__PRETTY_FUNCTION__.LSH_query_init, i32 0, i32 0)) #7
  unreachable
                                                  ; No predecessors!
  br label %222

; <label>:222:                                    ; preds = %221, %219
  %223 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %224 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %223, i32 0, i32 16
  %225 = load float*, float** %224, align 8
  %226 = icmp ne float* %225, null
  br i1 %226, label %227, label %228

; <label>:227:                                    ; preds = %222
  br label %230

; <label>:228:                                    ; preds = %222
  call void @__assert_fail(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str.12, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.1, i32 0, i32 0), i32 1900, i8* getelementptr inbounds ([101 x i8], [101 x i8]* @__PRETTY_FUNCTION__.LSH_query_init, i32 0, i32 0)) #7
  unreachable
                                                  ; No predecessors!
  br label %230

; <label>:230:                                    ; preds = %229, %227
  %231 = load %struct.LSH_query_t*, %struct.LSH_query_t** %7, align 8
  %232 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %231, i32 0, i32 20
  store float 1.000000e+00, float* %232, align 4
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #1

declare %struct.bitmap_t* @bitmap_new(i32) #2

declare i8** @__matrix_alloc(i32, i32, i32) #2

; Function Attrs: noreturn nounwind
declare void @__assert_fail(i8*, i8*, i32, i8*) #3

; Function Attrs: nounwind
declare noalias i8* @malloc(i64) #4

declare %struct.ptb_vec_t* @gen_score(i32) #2

; Function Attrs: nounwind
declare noalias i8* @calloc(i64, i64) #4

declare i32 @gen_perturb_set(%struct.ptb_vec_t*, %struct.ptb_vec_t*, i32, i32) #2

; Function Attrs: nounwind
declare void @free(i8*) #4

; Function Attrs: noinline nounwind optnone uwtable
define void @LSH_query_cleanup(%struct.LSH_query_t*) #0 {
  %2 = alloca %struct.LSH_query_t*, align 8
  store %struct.LSH_query_t* %0, %struct.LSH_query_t** %2, align 8
  %3 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %4 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %3, i32 0, i32 5
  %5 = call i32 @bitmap_free(%struct.bitmap_t** %4)
  %6 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %7 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %6, i32 0, i32 6
  %8 = load i32**, i32*** %7, align 8
  %9 = bitcast i32** %8 to i8**
  call void @__matrix_free(i8** %9)
  %10 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %11 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %10, i32 0, i32 13
  %12 = load %struct.cass_list_entry_t**, %struct.cass_list_entry_t*** %11, align 8
  %13 = bitcast %struct.cass_list_entry_t** %12 to i8**
  call void @__matrix_free(i8** %13)
  %14 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %15 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %14, i32 0, i32 12
  %16 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %15, align 8
  %17 = bitcast %struct.cass_list_entry_t* %16 to i8*
  call void @free(i8* %17) #6
  %18 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %19 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %18, i32 0, i32 7
  %20 = load i32*, i32** %19, align 8
  %21 = bitcast i32* %20 to i8*
  call void @free(i8* %21) #6
  %22 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %23 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %22, i32 0, i32 9
  %24 = load i32*, i32** %23, align 8
  %25 = bitcast i32* %24 to i8*
  call void @free(i8* %25) #6
  %26 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %27 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %26, i32 0, i32 10
  %28 = load %struct.ptb_vec_t*, %struct.ptb_vec_t** %27, align 8
  %29 = bitcast %struct.ptb_vec_t* %28 to i8*
  call void @free(i8* %29) #6
  %30 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %31 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %30, i32 0, i32 11
  %32 = load %struct.ptb_vec_t**, %struct.ptb_vec_t*** %31, align 8
  %33 = bitcast %struct.ptb_vec_t** %32 to i8**
  call void @__matrix_free(i8** %33)
  %34 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %35 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %34, i32 0, i32 8
  %36 = load %struct.ptb_vec_t**, %struct.ptb_vec_t*** %35, align 8
  %37 = bitcast %struct.ptb_vec_t** %36 to i8**
  call void @__matrix_free(i8** %37)
  %38 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %39 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %38, i32 0, i32 14
  %40 = load i32*, i32** %39, align 8
  %41 = bitcast i32* %40 to i8*
  call void @free(i8* %41) #6
  %42 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %43 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %42, i32 0, i32 15
  %44 = load i32*, i32** %43, align 8
  %45 = bitcast i32* %44 to i8*
  call void @free(i8* %45) #6
  %46 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %47 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %46, i32 0, i32 16
  %48 = load float*, float** %47, align 8
  %49 = bitcast float* %48 to i8*
  call void @free(i8* %49) #6
  ret void
}

declare i32 @bitmap_free(%struct.bitmap_t**) #2

declare void @__matrix_free(i8**) #2

; Function Attrs: noinline nounwind optnone uwtable
define void @LSH_hash_score(%struct.LSH_t*, i32, float*, i32**, %struct.ptb_vec_t**) #0 {
  %6 = alloca %struct.LSH_t*, align 8
  %7 = alloca i32, align 4
  %8 = alloca float*, align 8
  %9 = alloca i32**, align 8
  %10 = alloca %struct.ptb_vec_t**, align 8
  %11 = alloca float, align 4
  %12 = alloca float, align 4
  %13 = alloca i32, align 4
  %14 = alloca i32, align 4
  %15 = alloca i32, align 4
  %16 = alloca i32, align 4
  %17 = alloca i32, align 4
  store %struct.LSH_t* %0, %struct.LSH_t** %6, align 8
  store i32 %1, i32* %7, align 4
  store float* %2, float** %8, align 8
  store i32** %3, i32*** %9, align 8
  store %struct.ptb_vec_t** %4, %struct.ptb_vec_t*** %10, align 8
  store i32 0, i32* %17, align 4
  store i32 0, i32* %13, align 4
  br label %18

; <label>:18:                                     ; preds = %187, %5
  %19 = load i32, i32* %13, align 4
  %20 = load i32, i32* %7, align 4
  %21 = icmp slt i32 %19, %20
  br i1 %21, label %22, label %190

; <label>:22:                                     ; preds = %18
  store i32 0, i32* %16, align 4
  store i32 0, i32* %14, align 4
  br label %23

; <label>:23:                                     ; preds = %183, %22
  %24 = load i32, i32* %14, align 4
  %25 = load %struct.LSH_t*, %struct.LSH_t** %6, align 8
  %26 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %25, i32 0, i32 1
  %27 = load i32, i32* %26, align 4
  %28 = icmp ult i32 %24, %27
  br i1 %28, label %29, label %186

; <label>:29:                                     ; preds = %23
  %30 = load %struct.LSH_t*, %struct.LSH_t** %6, align 8
  %31 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %30, i32 0, i32 7
  %32 = load float*, float** %31, align 8
  %33 = load i32, i32* %17, align 4
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds float, float* %32, i64 %34
  %36 = load float, float* %35, align 4
  store float %36, float* %11, align 4
  store i32 0, i32* %15, align 4
  br label %37

; <label>:37:                                     ; preds = %63, %29
  %38 = load i32, i32* %15, align 4
  %39 = load %struct.LSH_t*, %struct.LSH_t** %6, align 8
  %40 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %39, i32 0, i32 0
  %41 = load i32, i32* %40, align 8
  %42 = icmp ult i32 %38, %41
  br i1 %42, label %43, label %66

; <label>:43:                                     ; preds = %37
  %44 = load float*, float** %8, align 8
  %45 = load i32, i32* %15, align 4
  %46 = sext i32 %45 to i64
  %47 = getelementptr inbounds float, float* %44, i64 %46
  %48 = load float, float* %47, align 4
  %49 = load %struct.LSH_t*, %struct.LSH_t** %6, align 8
  %50 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %49, i32 0, i32 6
  %51 = load float**, float*** %50, align 8
  %52 = load i32, i32* %17, align 4
  %53 = sext i32 %52 to i64
  %54 = getelementptr inbounds float*, float** %51, i64 %53
  %55 = load float*, float** %54, align 8
  %56 = load i32, i32* %15, align 4
  %57 = sext i32 %56 to i64
  %58 = getelementptr inbounds float, float* %55, i64 %57
  %59 = load float, float* %58, align 4
  %60 = fmul float %48, %59
  %61 = load float, float* %11, align 4
  %62 = fadd float %61, %60
  store float %62, float* %11, align 4
  br label %63

; <label>:63:                                     ; preds = %43
  %64 = load i32, i32* %15, align 4
  %65 = add nsw i32 %64, 1
  store i32 %65, i32* %15, align 4
  br label %37

; <label>:66:                                     ; preds = %37
  %67 = load float, float* %11, align 4
  %68 = load %struct.LSH_t*, %struct.LSH_t** %6, align 8
  %69 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %68, i32 0, i32 5
  %70 = load float*, float** %69, align 8
  %71 = load i32, i32* %13, align 4
  %72 = sext i32 %71 to i64
  %73 = getelementptr inbounds float, float* %70, i64 %72
  %74 = load float, float* %73, align 4
  %75 = fdiv float %67, %74
  %76 = fpext float %75 to double
  %77 = call double @llvm.floor.f64(double %76)
  %78 = fptrunc double %77 to float
  store float %78, float* %12, align 4
  %79 = load float, float* %12, align 4
  %80 = fptoui float %79 to i32
  %81 = load i32**, i32*** %9, align 8
  %82 = load i32, i32* %13, align 4
  %83 = sext i32 %82 to i64
  %84 = getelementptr inbounds i32*, i32** %81, i64 %83
  %85 = load i32*, i32** %84, align 8
  %86 = load i32, i32* %14, align 4
  %87 = sext i32 %86 to i64
  %88 = getelementptr inbounds i32, i32* %85, i64 %87
  store i32 %80, i32* %88, align 4
  %89 = load float, float* %11, align 4
  %90 = load float, float* %12, align 4
  %91 = load %struct.LSH_t*, %struct.LSH_t** %6, align 8
  %92 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %91, i32 0, i32 5
  %93 = load float*, float** %92, align 8
  %94 = load i32, i32* %13, align 4
  %95 = sext i32 %94 to i64
  %96 = getelementptr inbounds float, float* %93, i64 %95
  %97 = load float, float* %96, align 4
  %98 = fmul float %90, %97
  %99 = fsub float %89, %98
  store float %99, float* %12, align 4
  %100 = load i32, i32* %14, align 4
  %101 = shl i32 1, %100
  %102 = sext i32 %101 to i64
  %103 = load %struct.ptb_vec_t**, %struct.ptb_vec_t*** %10, align 8
  %104 = load i32, i32* %13, align 4
  %105 = sext i32 %104 to i64
  %106 = getelementptr inbounds %struct.ptb_vec_t*, %struct.ptb_vec_t** %103, i64 %105
  %107 = load %struct.ptb_vec_t*, %struct.ptb_vec_t** %106, align 8
  %108 = load i32, i32* %16, align 4
  %109 = sext i32 %108 to i64
  %110 = getelementptr inbounds %struct.ptb_vec_t, %struct.ptb_vec_t* %107, i64 %109
  %111 = getelementptr inbounds %struct.ptb_vec_t, %struct.ptb_vec_t* %110, i32 0, i32 0
  store i64 %102, i64* %111, align 8
  %112 = load %struct.ptb_vec_t**, %struct.ptb_vec_t*** %10, align 8
  %113 = load i32, i32* %13, align 4
  %114 = sext i32 %113 to i64
  %115 = getelementptr inbounds %struct.ptb_vec_t*, %struct.ptb_vec_t** %112, i64 %114
  %116 = load %struct.ptb_vec_t*, %struct.ptb_vec_t** %115, align 8
  %117 = load i32, i32* %16, align 4
  %118 = sext i32 %117 to i64
  %119 = getelementptr inbounds %struct.ptb_vec_t, %struct.ptb_vec_t* %116, i64 %118
  %120 = getelementptr inbounds %struct.ptb_vec_t, %struct.ptb_vec_t* %119, i32 0, i32 2
  store i32 0, i32* %120, align 4
  %121 = load float, float* %12, align 4
  %122 = load float, float* %12, align 4
  %123 = fmul float %121, %122
  %124 = load %struct.ptb_vec_t**, %struct.ptb_vec_t*** %10, align 8
  %125 = load i32, i32* %13, align 4
  %126 = sext i32 %125 to i64
  %127 = getelementptr inbounds %struct.ptb_vec_t*, %struct.ptb_vec_t** %124, i64 %126
  %128 = load %struct.ptb_vec_t*, %struct.ptb_vec_t** %127, align 8
  %129 = load i32, i32* %16, align 4
  %130 = sext i32 %129 to i64
  %131 = getelementptr inbounds %struct.ptb_vec_t, %struct.ptb_vec_t* %128, i64 %130
  %132 = getelementptr inbounds %struct.ptb_vec_t, %struct.ptb_vec_t* %131, i32 0, i32 1
  store float %123, float* %132, align 8
  %133 = load i32, i32* %16, align 4
  %134 = add nsw i32 %133, 1
  store i32 %134, i32* %16, align 4
  %135 = load %struct.LSH_t*, %struct.LSH_t** %6, align 8
  %136 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %135, i32 0, i32 5
  %137 = load float*, float** %136, align 8
  %138 = load i32, i32* %13, align 4
  %139 = sext i32 %138 to i64
  %140 = getelementptr inbounds float, float* %137, i64 %139
  %141 = load float, float* %140, align 4
  %142 = load float, float* %12, align 4
  %143 = fsub float %141, %142
  store float %143, float* %12, align 4
  %144 = load i32, i32* %14, align 4
  %145 = shl i32 1, %144
  %146 = sext i32 %145 to i64
  %147 = load %struct.ptb_vec_t**, %struct.ptb_vec_t*** %10, align 8
  %148 = load i32, i32* %13, align 4
  %149 = sext i32 %148 to i64
  %150 = getelementptr inbounds %struct.ptb_vec_t*, %struct.ptb_vec_t** %147, i64 %149
  %151 = load %struct.ptb_vec_t*, %struct.ptb_vec_t** %150, align 8
  %152 = load i32, i32* %16, align 4
  %153 = sext i32 %152 to i64
  %154 = getelementptr inbounds %struct.ptb_vec_t, %struct.ptb_vec_t* %151, i64 %153
  %155 = getelementptr inbounds %struct.ptb_vec_t, %struct.ptb_vec_t* %154, i32 0, i32 0
  store i64 %146, i64* %155, align 8
  %156 = load i32, i32* %14, align 4
  %157 = shl i32 1, %156
  %158 = load %struct.ptb_vec_t**, %struct.ptb_vec_t*** %10, align 8
  %159 = load i32, i32* %13, align 4
  %160 = sext i32 %159 to i64
  %161 = getelementptr inbounds %struct.ptb_vec_t*, %struct.ptb_vec_t** %158, i64 %160
  %162 = load %struct.ptb_vec_t*, %struct.ptb_vec_t** %161, align 8
  %163 = load i32, i32* %16, align 4
  %164 = sext i32 %163 to i64
  %165 = getelementptr inbounds %struct.ptb_vec_t, %struct.ptb_vec_t* %162, i64 %164
  %166 = getelementptr inbounds %struct.ptb_vec_t, %struct.ptb_vec_t* %165, i32 0, i32 2
  store i32 %157, i32* %166, align 4
  %167 = load float, float* %12, align 4
  %168 = load float, float* %12, align 4
  %169 = fmul float %167, %168
  %170 = load %struct.ptb_vec_t**, %struct.ptb_vec_t*** %10, align 8
  %171 = load i32, i32* %13, align 4
  %172 = sext i32 %171 to i64
  %173 = getelementptr inbounds %struct.ptb_vec_t*, %struct.ptb_vec_t** %170, i64 %172
  %174 = load %struct.ptb_vec_t*, %struct.ptb_vec_t** %173, align 8
  %175 = load i32, i32* %16, align 4
  %176 = sext i32 %175 to i64
  %177 = getelementptr inbounds %struct.ptb_vec_t, %struct.ptb_vec_t* %174, i64 %176
  %178 = getelementptr inbounds %struct.ptb_vec_t, %struct.ptb_vec_t* %177, i32 0, i32 1
  store float %169, float* %178, align 8
  %179 = load i32, i32* %16, align 4
  %180 = add nsw i32 %179, 1
  store i32 %180, i32* %16, align 4
  %181 = load i32, i32* %17, align 4
  %182 = add nsw i32 %181, 1
  store i32 %182, i32* %17, align 4
  br label %183

; <label>:183:                                    ; preds = %66
  %184 = load i32, i32* %14, align 4
  %185 = add nsw i32 %184, 1
  store i32 %185, i32* %14, align 4
  br label %23

; <label>:186:                                    ; preds = %23
  br label %187

; <label>:187:                                    ; preds = %186
  %188 = load i32, i32* %13, align 4
  %189 = add nsw i32 %188, 1
  store i32 %189, i32* %13, align 4
  br label %18

; <label>:190:                                    ; preds = %18
  ret void
}

; Function Attrs: nounwind readnone speculatable
declare double @llvm.floor.f64(double) #5

; Function Attrs: noinline nounwind optnone uwtable
define void @LSH_hash2_noperturb(%struct.LSH_t*, i32**, i32*, i32) #0 {
  %5 = alloca %struct.LSH_t*, align 8
  %6 = alloca i32**, align 8
  %7 = alloca i32*, align 8
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  store %struct.LSH_t* %0, %struct.LSH_t** %5, align 8
  store i32** %1, i32*** %6, align 8
  store i32* %2, i32** %7, align 8
  store i32 %3, i32* %8, align 4
  store i32 0, i32* %9, align 4
  br label %12

; <label>:12:                                     ; preds = %60, %4
  %13 = load i32, i32* %9, align 4
  %14 = load i32, i32* %8, align 4
  %15 = icmp slt i32 %13, %14
  br i1 %15, label %16, label %63

; <label>:16:                                     ; preds = %12
  store i32 0, i32* %11, align 4
  store i32 0, i32* %10, align 4
  br label %17

; <label>:17:                                     ; preds = %47, %16
  %18 = load i32, i32* %10, align 4
  %19 = load %struct.LSH_t*, %struct.LSH_t** %5, align 8
  %20 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %19, i32 0, i32 1
  %21 = load i32, i32* %20, align 4
  %22 = icmp ult i32 %18, %21
  br i1 %22, label %23, label %50

; <label>:23:                                     ; preds = %17
  %24 = load %struct.LSH_t*, %struct.LSH_t** %5, align 8
  %25 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %24, i32 0, i32 8
  %26 = load i32**, i32*** %25, align 8
  %27 = load i32, i32* %9, align 4
  %28 = sext i32 %27 to i64
  %29 = getelementptr inbounds i32*, i32** %26, i64 %28
  %30 = load i32*, i32** %29, align 8
  %31 = load i32, i32* %10, align 4
  %32 = sext i32 %31 to i64
  %33 = getelementptr inbounds i32, i32* %30, i64 %32
  %34 = load i32, i32* %33, align 4
  %35 = load i32**, i32*** %6, align 8
  %36 = load i32, i32* %9, align 4
  %37 = sext i32 %36 to i64
  %38 = getelementptr inbounds i32*, i32** %35, i64 %37
  %39 = load i32*, i32** %38, align 8
  %40 = load i32, i32* %10, align 4
  %41 = sext i32 %40 to i64
  %42 = getelementptr inbounds i32, i32* %39, i64 %41
  %43 = load i32, i32* %42, align 4
  %44 = mul i32 %34, %43
  %45 = load i32, i32* %11, align 4
  %46 = add i32 %45, %44
  store i32 %46, i32* %11, align 4
  br label %47

; <label>:47:                                     ; preds = %23
  %48 = load i32, i32* %10, align 4
  %49 = add nsw i32 %48, 1
  store i32 %49, i32* %10, align 4
  br label %17

; <label>:50:                                     ; preds = %17
  %51 = load i32, i32* %11, align 4
  %52 = load %struct.LSH_t*, %struct.LSH_t** %5, align 8
  %53 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %52, i32 0, i32 3
  %54 = load i32, i32* %53, align 4
  %55 = urem i32 %51, %54
  %56 = load i32*, i32** %7, align 8
  %57 = load i32, i32* %9, align 4
  %58 = sext i32 %57 to i64
  %59 = getelementptr inbounds i32, i32* %56, i64 %58
  store i32 %55, i32* %59, align 4
  br label %60

; <label>:60:                                     ; preds = %50
  %61 = load i32, i32* %9, align 4
  %62 = add nsw i32 %61, 1
  store i32 %62, i32* %9, align 4
  br label %12

; <label>:63:                                     ; preds = %12
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define void @LSH_hash2_perturb(%struct.LSH_t*, i32**, i32*, %struct.ptb_vec_t*, i32) #0 {
  %6 = alloca %struct.LSH_t*, align 8
  %7 = alloca i32**, align 8
  %8 = alloca i32*, align 8
  %9 = alloca %struct.ptb_vec_t*, align 8
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca i32, align 4
  %14 = alloca i32, align 4
  %15 = alloca i32, align 4
  store %struct.LSH_t* %0, %struct.LSH_t** %6, align 8
  store i32** %1, i32*** %7, align 8
  store i32* %2, i32** %8, align 8
  store %struct.ptb_vec_t* %3, %struct.ptb_vec_t** %9, align 8
  store i32 %4, i32* %10, align 4
  %16 = load %struct.ptb_vec_t*, %struct.ptb_vec_t** %9, align 8
  %17 = getelementptr inbounds %struct.ptb_vec_t, %struct.ptb_vec_t* %16, i32 0, i32 0
  %18 = load i64, i64* %17, align 8
  %19 = trunc i64 %18 to i32
  store i32 %19, i32* %13, align 4
  %20 = load %struct.ptb_vec_t*, %struct.ptb_vec_t** %9, align 8
  %21 = getelementptr inbounds %struct.ptb_vec_t, %struct.ptb_vec_t* %20, i32 0, i32 2
  %22 = load i32, i32* %21, align 4
  store i32 %22, i32* %14, align 4
  store i32 1, i32* %12, align 4
  store i32 0, i32* %15, align 4
  store i32 0, i32* %11, align 4
  br label %23

; <label>:23:                                     ; preds = %117, %5
  %24 = load i32, i32* %11, align 4
  %25 = load %struct.LSH_t*, %struct.LSH_t** %6, align 8
  %26 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %25, i32 0, i32 1
  %27 = load i32, i32* %26, align 4
  %28 = icmp ult i32 %24, %27
  br i1 %28, label %29, label %120

; <label>:29:                                     ; preds = %23
  %30 = load i32, i32* %13, align 4
  %31 = load i32, i32* %12, align 4
  %32 = and i32 %30, %31
  %33 = icmp ne i32 %32, 0
  br i1 %33, label %34, label %90

; <label>:34:                                     ; preds = %29
  %35 = load i32, i32* %14, align 4
  %36 = load i32, i32* %12, align 4
  %37 = and i32 %35, %36
  %38 = icmp ne i32 %37, 0
  br i1 %38, label %39, label %64

; <label>:39:                                     ; preds = %34
  %40 = load %struct.LSH_t*, %struct.LSH_t** %6, align 8
  %41 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %40, i32 0, i32 8
  %42 = load i32**, i32*** %41, align 8
  %43 = load i32, i32* %10, align 4
  %44 = sext i32 %43 to i64
  %45 = getelementptr inbounds i32*, i32** %42, i64 %44
  %46 = load i32*, i32** %45, align 8
  %47 = load i32, i32* %11, align 4
  %48 = zext i32 %47 to i64
  %49 = getelementptr inbounds i32, i32* %46, i64 %48
  %50 = load i32, i32* %49, align 4
  %51 = load i32**, i32*** %7, align 8
  %52 = load i32, i32* %10, align 4
  %53 = sext i32 %52 to i64
  %54 = getelementptr inbounds i32*, i32** %51, i64 %53
  %55 = load i32*, i32** %54, align 8
  %56 = load i32, i32* %11, align 4
  %57 = zext i32 %56 to i64
  %58 = getelementptr inbounds i32, i32* %55, i64 %57
  %59 = load i32, i32* %58, align 4
  %60 = add i32 %59, 1
  %61 = mul i32 %50, %60
  %62 = load i32, i32* %15, align 4
  %63 = add i32 %62, %61
  store i32 %63, i32* %15, align 4
  br label %89

; <label>:64:                                     ; preds = %34
  %65 = load %struct.LSH_t*, %struct.LSH_t** %6, align 8
  %66 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %65, i32 0, i32 8
  %67 = load i32**, i32*** %66, align 8
  %68 = load i32, i32* %10, align 4
  %69 = sext i32 %68 to i64
  %70 = getelementptr inbounds i32*, i32** %67, i64 %69
  %71 = load i32*, i32** %70, align 8
  %72 = load i32, i32* %11, align 4
  %73 = zext i32 %72 to i64
  %74 = getelementptr inbounds i32, i32* %71, i64 %73
  %75 = load i32, i32* %74, align 4
  %76 = load i32**, i32*** %7, align 8
  %77 = load i32, i32* %10, align 4
  %78 = sext i32 %77 to i64
  %79 = getelementptr inbounds i32*, i32** %76, i64 %78
  %80 = load i32*, i32** %79, align 8
  %81 = load i32, i32* %11, align 4
  %82 = zext i32 %81 to i64
  %83 = getelementptr inbounds i32, i32* %80, i64 %82
  %84 = load i32, i32* %83, align 4
  %85 = sub i32 %84, 1
  %86 = mul i32 %75, %85
  %87 = load i32, i32* %15, align 4
  %88 = add i32 %87, %86
  store i32 %88, i32* %15, align 4
  br label %89

; <label>:89:                                     ; preds = %64, %39
  br label %114

; <label>:90:                                     ; preds = %29
  %91 = load %struct.LSH_t*, %struct.LSH_t** %6, align 8
  %92 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %91, i32 0, i32 8
  %93 = load i32**, i32*** %92, align 8
  %94 = load i32, i32* %10, align 4
  %95 = sext i32 %94 to i64
  %96 = getelementptr inbounds i32*, i32** %93, i64 %95
  %97 = load i32*, i32** %96, align 8
  %98 = load i32, i32* %11, align 4
  %99 = zext i32 %98 to i64
  %100 = getelementptr inbounds i32, i32* %97, i64 %99
  %101 = load i32, i32* %100, align 4
  %102 = load i32**, i32*** %7, align 8
  %103 = load i32, i32* %10, align 4
  %104 = sext i32 %103 to i64
  %105 = getelementptr inbounds i32*, i32** %102, i64 %104
  %106 = load i32*, i32** %105, align 8
  %107 = load i32, i32* %11, align 4
  %108 = zext i32 %107 to i64
  %109 = getelementptr inbounds i32, i32* %106, i64 %108
  %110 = load i32, i32* %109, align 4
  %111 = mul i32 %101, %110
  %112 = load i32, i32* %15, align 4
  %113 = add i32 %112, %111
  store i32 %113, i32* %15, align 4
  br label %114

; <label>:114:                                    ; preds = %90, %89
  %115 = load i32, i32* %12, align 4
  %116 = shl i32 %115, 1
  store i32 %116, i32* %12, align 4
  br label %117

; <label>:117:                                    ; preds = %114
  %118 = load i32, i32* %11, align 4
  %119 = add i32 %118, 1
  store i32 %119, i32* %11, align 4
  br label %23

; <label>:120:                                    ; preds = %23
  %121 = load i32, i32* %15, align 4
  %122 = load %struct.LSH_t*, %struct.LSH_t** %6, align 8
  %123 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %122, i32 0, i32 3
  %124 = load i32, i32* %123, align 4
  %125 = urem i32 %121, %124
  %126 = load i32*, i32** %8, align 8
  store i32 %125, i32* %126, align 4
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define void @LSH_query_merge(%struct.LSH_query_t*) #0 {
  %2 = alloca %struct.LSH_query_t*, align 8
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca %struct.cass_list_entry_t**, align 8
  %6 = alloca %struct.cass_list_entry_t*, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  store %struct.LSH_query_t* %0, %struct.LSH_query_t** %2, align 8
  %12 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %13 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %12, i32 0, i32 2
  %14 = load i32, i32* %13, align 8
  store i32 %14, i32* %3, align 4
  %15 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %16 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %15, i32 0, i32 3
  %17 = load i32, i32* %16, align 4
  store i32 %17, i32* %4, align 4
  %18 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %19 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %18, i32 0, i32 13
  %20 = load %struct.cass_list_entry_t**, %struct.cass_list_entry_t*** %19, align 8
  store %struct.cass_list_entry_t** %20, %struct.cass_list_entry_t*** %5, align 8
  %21 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %22 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %21, i32 0, i32 12
  %23 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %22, align 8
  store %struct.cass_list_entry_t* %23, %struct.cass_list_entry_t** %6, align 8
  %24 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %6, align 8
  %25 = bitcast %struct.cass_list_entry_t* %24 to i8*
  %26 = load i32, i32* %3, align 4
  %27 = zext i32 %26 to i64
  %28 = mul i64 8, %27
  call void @llvm.memset.p0i8.i64(i8* %25, i8 -1, i64 %28, i32 4, i1 false)
  br label %29

; <label>:29:                                     ; preds = %1
  store i32 0, i32* %9, align 4
  br label %30

; <label>:30:                                     ; preds = %40, %29
  %31 = load i32, i32* %9, align 4
  %32 = load i32, i32* %3, align 4
  %33 = icmp ult i32 %31, %32
  br i1 %33, label %34, label %43

; <label>:34:                                     ; preds = %30
  %35 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %6, align 8
  %36 = load i32, i32* %9, align 4
  %37 = sext i32 %36 to i64
  %38 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %35, i64 %37
  %39 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %38, i32 0, i32 1
  store float 0x7FF0000000000000, float* %39, align 4
  br label %40

; <label>:40:                                     ; preds = %34
  %41 = load i32, i32* %9, align 4
  %42 = add nsw i32 %41, 1
  store i32 %42, i32* %9, align 4
  br label %30

; <label>:43:                                     ; preds = %30
  br label %44

; <label>:44:                                     ; preds = %43
  store i32 0, i32* %7, align 4
  br label %45

; <label>:45:                                     ; preds = %150, %44
  %46 = load i32, i32* %7, align 4
  %47 = load i32, i32* %4, align 4
  %48 = icmp ult i32 %46, %47
  br i1 %48, label %49, label %153

; <label>:49:                                     ; preds = %45
  store i32 0, i32* %8, align 4
  br label %50

; <label>:50:                                     ; preds = %146, %49
  %51 = load i32, i32* %8, align 4
  %52 = load i32, i32* %3, align 4
  %53 = icmp ult i32 %51, %52
  br i1 %53, label %54, label %149

; <label>:54:                                     ; preds = %50
  br label %55

; <label>:55:                                     ; preds = %54
  store i32 0, i32* %11, align 4
  store i32 0, i32* %10, align 4
  br label %56

; <label>:56:                                     ; preds = %98, %55
  %57 = load i32, i32* %10, align 4
  %58 = load i32, i32* %3, align 4
  %59 = icmp ult i32 %57, %58
  br i1 %59, label %60, label %101

; <label>:60:                                     ; preds = %56
  %61 = load %struct.cass_list_entry_t**, %struct.cass_list_entry_t*** %5, align 8
  %62 = load i32, i32* %7, align 4
  %63 = sext i32 %62 to i64
  %64 = getelementptr inbounds %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %61, i64 %63
  %65 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %64, align 8
  %66 = load i32, i32* %8, align 4
  %67 = sext i32 %66 to i64
  %68 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %65, i64 %67
  %69 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %68, i32 0, i32 1
  %70 = load float, float* %69, align 4
  %71 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %6, align 8
  %72 = load i32, i32* %10, align 4
  %73 = sext i32 %72 to i64
  %74 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %71, i64 %73
  %75 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %74, i32 0, i32 1
  %76 = load float, float* %75, align 4
  %77 = fcmp ogt float %70, %76
  br i1 %77, label %78, label %79

; <label>:78:                                     ; preds = %60
  br label %101

; <label>:79:                                     ; preds = %60
  %80 = load %struct.cass_list_entry_t**, %struct.cass_list_entry_t*** %5, align 8
  %81 = load i32, i32* %7, align 4
  %82 = sext i32 %81 to i64
  %83 = getelementptr inbounds %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %80, i64 %82
  %84 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %83, align 8
  %85 = load i32, i32* %8, align 4
  %86 = sext i32 %85 to i64
  %87 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %84, i64 %86
  %88 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %87, i32 0, i32 0
  %89 = load i32, i32* %88, align 4
  %90 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %6, align 8
  %91 = load i32, i32* %10, align 4
  %92 = sext i32 %91 to i64
  %93 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %90, i64 %92
  %94 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %93, i32 0, i32 0
  %95 = load i32, i32* %94, align 4
  %96 = icmp eq i32 %89, %95
  br i1 %96, label %97, label %98

; <label>:97:                                     ; preds = %79
  store i32 1, i32* %11, align 4
  br label %101

; <label>:98:                                     ; preds = %79
  %99 = load i32, i32* %10, align 4
  %100 = add nsw i32 %99, 1
  store i32 %100, i32* %10, align 4
  br label %56

; <label>:101:                                    ; preds = %97, %78, %56
  %102 = load i32, i32* %11, align 4
  %103 = icmp ne i32 %102, 0
  br i1 %103, label %104, label %105

; <label>:104:                                    ; preds = %101
  br label %145

; <label>:105:                                    ; preds = %101
  %106 = load i32, i32* %10, align 4
  %107 = icmp eq i32 %106, 0
  br i1 %107, label %108, label %109

; <label>:108:                                    ; preds = %105
  br label %145

; <label>:109:                                    ; preds = %105
  store i32 0, i32* %11, align 4
  br label %110

; <label>:110:                                    ; preds = %127, %109
  %111 = load i32, i32* %11, align 4
  %112 = load i32, i32* %10, align 4
  %113 = sub nsw i32 %112, 1
  %114 = icmp slt i32 %111, %113
  br i1 %114, label %115, label %130

; <label>:115:                                    ; preds = %110
  %116 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %6, align 8
  %117 = load i32, i32* %11, align 4
  %118 = sext i32 %117 to i64
  %119 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %116, i64 %118
  %120 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %6, align 8
  %121 = load i32, i32* %11, align 4
  %122 = add nsw i32 %121, 1
  %123 = sext i32 %122 to i64
  %124 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %120, i64 %123
  %125 = bitcast %struct.cass_list_entry_t* %119 to i8*
  %126 = bitcast %struct.cass_list_entry_t* %124 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %125, i8* %126, i64 8, i32 4, i1 false)
  br label %127

; <label>:127:                                    ; preds = %115
  %128 = load i32, i32* %11, align 4
  %129 = add nsw i32 %128, 1
  store i32 %129, i32* %11, align 4
  br label %110

; <label>:130:                                    ; preds = %110
  %131 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %6, align 8
  %132 = load i32, i32* %11, align 4
  %133 = sext i32 %132 to i64
  %134 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %131, i64 %133
  %135 = load %struct.cass_list_entry_t**, %struct.cass_list_entry_t*** %5, align 8
  %136 = load i32, i32* %7, align 4
  %137 = sext i32 %136 to i64
  %138 = getelementptr inbounds %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %135, i64 %137
  %139 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %138, align 8
  %140 = load i32, i32* %8, align 4
  %141 = sext i32 %140 to i64
  %142 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %139, i64 %141
  %143 = bitcast %struct.cass_list_entry_t* %134 to i8*
  %144 = bitcast %struct.cass_list_entry_t* %142 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %143, i8* %144, i64 8, i32 4, i1 false)
  br label %145

; <label>:145:                                    ; preds = %130, %108, %104
  br label %146

; <label>:146:                                    ; preds = %145
  %147 = load i32, i32* %8, align 4
  %148 = add nsw i32 %147, 1
  store i32 %148, i32* %8, align 4
  br label %50

; <label>:149:                                    ; preds = %50
  br label %150

; <label>:150:                                    ; preds = %149
  %151 = load i32, i32* %7, align 4
  %152 = add nsw i32 %151, 1
  store i32 %152, i32* %7, align 4
  br label %45

; <label>:153:                                    ; preds = %45
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #1

; Function Attrs: noinline nounwind optnone uwtable
define void @LSH_query(%struct.LSH_query_t*, float*) #0 {
  %3 = alloca %struct.LSH_query_t*, align 8
  %4 = alloca float*, align 8
  %5 = alloca %struct.LSH_est_t*, align 8
  %6 = alloca %struct.cass_list_entry_t*, align 8
  %7 = alloca float, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  store %struct.LSH_query_t* %0, %struct.LSH_query_t** %3, align 8
  store float* %1, float** %4, align 8
  %12 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %13 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %12, i32 0, i32 0
  %14 = load %struct.LSH_t*, %struct.LSH_t** %13, align 8
  %15 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %14, i32 0, i32 12
  %16 = load %struct.LSH_est_t*, %struct.LSH_est_t** %15, align 8
  store %struct.LSH_est_t* %16, %struct.LSH_est_t** %5, align 8
  %17 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %18 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %17, i32 0, i32 12
  %19 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %18, align 8
  store %struct.cass_list_entry_t* %19, %struct.cass_list_entry_t** %6, align 8
  %20 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %21 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %20, i32 0, i32 18
  %22 = load float, float* %21, align 4
  store float %22, float* %7, align 4
  %23 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %24 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %23, i32 0, i32 4
  %25 = load i32, i32* %24, align 8
  store i32 %25, i32* %8, align 4
  %26 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %27 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %26, i32 0, i32 3
  %28 = load i32, i32* %27, align 4
  store i32 %28, i32* %9, align 4
  %29 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %30 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %29, i32 0, i32 17
  store i32 0, i32* %30, align 8
  %31 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %32 = load float*, float** %4, align 8
  call void @LSH_query_bootstrap(%struct.LSH_query_t* %31, float* %32)
  store i32 0, i32* %10, align 4
  br label %33

; <label>:33:                                     ; preds = %62, %2
  %34 = load i32, i32* %10, align 4
  %35 = load i32, i32* %9, align 4
  %36 = icmp ult i32 %34, %35
  br i1 %36, label %37, label %65

; <label>:37:                                     ; preds = %33
  store i32 0, i32* %11, align 4
  br label %38

; <label>:38:                                     ; preds = %58, %37
  %39 = load i32, i32* %11, align 4
  %40 = load i32, i32* %8, align 4
  %41 = icmp ult i32 %39, %40
  br i1 %41, label %42, label %61

; <label>:42:                                     ; preds = %38
  %43 = load %struct.LSH_est_t*, %struct.LSH_est_t** %5, align 8
  %44 = icmp ne %struct.LSH_est_t* %43, null
  br i1 %44, label %45, label %54

; <label>:45:                                     ; preds = %42
  %46 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %6, align 8
  %47 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %46, i64 0
  %48 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %47, i32 0, i32 1
  %49 = load float, float* %48, align 4
  %50 = load float, float* %7, align 4
  %51 = fcmp ole float %49, %50
  br i1 %51, label %52, label %53

; <label>:52:                                     ; preds = %45
  br label %67

; <label>:53:                                     ; preds = %45
  br label %54

; <label>:54:                                     ; preds = %53, %42
  %55 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %56 = load float*, float** %4, align 8
  %57 = load i32, i32* %10, align 4
  call void @LSH_query_probe(%struct.LSH_query_t* %55, float* %56, i32 %57, i32 0)
  br label %58

; <label>:58:                                     ; preds = %54
  %59 = load i32, i32* %11, align 4
  %60 = add nsw i32 %59, 1
  store i32 %60, i32* %11, align 4
  br label %38

; <label>:61:                                     ; preds = %38
  br label %62

; <label>:62:                                     ; preds = %61
  %63 = load i32, i32* %10, align 4
  %64 = add nsw i32 %63, 1
  store i32 %64, i32* %10, align 4
  br label %33

; <label>:65:                                     ; preds = %33
  %66 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  call void @LSH_query_merge(%struct.LSH_query_t* %66)
  br label %67

; <label>:67:                                     ; preds = %65, %52
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @LSH_query_bootstrap(%struct.LSH_query_t*, float*) #0 {
  %3 = alloca %struct.LSH_query_t*, align 8
  %4 = alloca float*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca %struct.LSH_t*, align 8
  %9 = alloca %struct.ptb_vec_t**, align 8
  %10 = alloca i32**, align 8
  %11 = alloca i32*, align 8
  %12 = alloca %struct.cass_list_entry_t**, align 8
  %13 = alloca %struct.cass_list_entry_t, align 4
  %14 = alloca i32*, align 8
  %15 = alloca i32*, align 8
  %16 = alloca i32, align 4
  %17 = alloca i32, align 4
  %18 = alloca i32, align 4
  %19 = alloca i32, align 4
  %20 = alloca %struct._cass_vec_t*, align 8
  %21 = alloca i32, align 4
  %22 = alloca i32, align 4
  store %struct.LSH_query_t* %0, %struct.LSH_query_t** %3, align 8
  store float* %1, float** %4, align 8
  %23 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %24 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %23, i32 0, i32 0
  %25 = load %struct.LSH_t*, %struct.LSH_t** %24, align 8
  %26 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %25, i32 0, i32 0
  %27 = load i32, i32* %26, align 8
  store i32 %27, i32* %5, align 4
  %28 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %29 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %28, i32 0, i32 2
  %30 = load i32, i32* %29, align 8
  store i32 %30, i32* %6, align 4
  %31 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %32 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %31, i32 0, i32 3
  %33 = load i32, i32* %32, align 4
  store i32 %33, i32* %7, align 4
  %34 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %35 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %34, i32 0, i32 0
  %36 = load %struct.LSH_t*, %struct.LSH_t** %35, align 8
  store %struct.LSH_t* %36, %struct.LSH_t** %8, align 8
  %37 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %38 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %37, i32 0, i32 8
  %39 = load %struct.ptb_vec_t**, %struct.ptb_vec_t*** %38, align 8
  store %struct.ptb_vec_t** %39, %struct.ptb_vec_t*** %9, align 8
  %40 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %41 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %40, i32 0, i32 6
  %42 = load i32**, i32*** %41, align 8
  store i32** %42, i32*** %10, align 8
  %43 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %44 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %43, i32 0, i32 7
  %45 = load i32*, i32** %44, align 8
  store i32* %45, i32** %11, align 8
  %46 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %47 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %46, i32 0, i32 13
  %48 = load %struct.cass_list_entry_t**, %struct.cass_list_entry_t*** %47, align 8
  store %struct.cass_list_entry_t** %48, %struct.cass_list_entry_t*** %12, align 8
  %49 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %50 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %49, i32 0, i32 14
  %51 = load i32*, i32** %50, align 8
  store i32* %51, i32** %14, align 8
  %52 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %53 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %52, i32 0, i32 15
  %54 = load i32*, i32** %53, align 8
  store i32* %54, i32** %15, align 8
  %55 = load i32*, i32** %14, align 8
  %56 = bitcast i32* %55 to i8*
  %57 = load i32, i32* %7, align 4
  %58 = zext i32 %57 to i64
  %59 = mul i64 %58, 4
  call void @llvm.memset.p0i8.i64(i8* %56, i8 0, i64 %59, i32 4, i1 false)
  %60 = load i32*, i32** %15, align 8
  %61 = bitcast i32* %60 to i8*
  %62 = load i32, i32* %7, align 4
  %63 = zext i32 %62 to i64
  %64 = mul i64 %63, 4
  call void @llvm.memset.p0i8.i64(i8* %61, i8 0, i64 %64, i32 4, i1 false)
  %65 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %66 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %65, i32 0, i32 16
  %67 = load float*, float** %66, align 8
  %68 = bitcast float* %67 to i8*
  %69 = load i32, i32* %7, align 4
  %70 = zext i32 %69 to i64
  %71 = mul i64 %70, 4
  call void @llvm.memset.p0i8.i64(i8* %68, i8 0, i64 %71, i32 4, i1 false)
  %72 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %73 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %72, i32 0, i32 5
  %74 = load %struct.bitmap_t*, %struct.bitmap_t** %73, align 8
  %75 = call i32 @bitmap_clear(%struct.bitmap_t* %74)
  %76 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %77 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %76, i32 0, i32 0
  %78 = load %struct.LSH_t*, %struct.LSH_t** %77, align 8
  %79 = load i32, i32* %7, align 4
  %80 = load float*, float** %4, align 8
  %81 = load i32**, i32*** %10, align 8
  %82 = load %struct.ptb_vec_t**, %struct.ptb_vec_t*** %9, align 8
  call void @LSH_hash_score(%struct.LSH_t* %78, i32 %79, float* %80, i32** %81, %struct.ptb_vec_t** %82)
  %83 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %84 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %83, i32 0, i32 0
  %85 = load %struct.LSH_t*, %struct.LSH_t** %84, align 8
  %86 = load i32**, i32*** %10, align 8
  %87 = load i32*, i32** %11, align 8
  %88 = load i32, i32* %7, align 4
  call void @LSH_hash2_noperturb(%struct.LSH_t* %85, i32** %86, i32* %87, i32 %88)
  store i32 0, i32* %16, align 4
  br label %89

; <label>:89:                                     ; preds = %349, %2
  %90 = load i32, i32* %16, align 4
  %91 = load i32, i32* %7, align 4
  %92 = icmp ult i32 %90, %91
  br i1 %92, label %93, label %352

; <label>:93:                                     ; preds = %89
  %94 = load %struct.cass_list_entry_t**, %struct.cass_list_entry_t*** %12, align 8
  %95 = load i32, i32* %16, align 4
  %96 = sext i32 %95 to i64
  %97 = getelementptr inbounds %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %94, i64 %96
  %98 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %97, align 8
  %99 = bitcast %struct.cass_list_entry_t* %98 to i8*
  %100 = load i32, i32* %6, align 4
  %101 = zext i32 %100 to i64
  %102 = mul i64 8, %101
  call void @llvm.memset.p0i8.i64(i8* %99, i8 -1, i64 %102, i32 4, i1 false)
  br label %103

; <label>:103:                                    ; preds = %93
  store i32 0, i32* %17, align 4
  br label %104

; <label>:104:                                    ; preds = %118, %103
  %105 = load i32, i32* %17, align 4
  %106 = load i32, i32* %6, align 4
  %107 = icmp ult i32 %105, %106
  br i1 %107, label %108, label %121

; <label>:108:                                    ; preds = %104
  %109 = load %struct.cass_list_entry_t**, %struct.cass_list_entry_t*** %12, align 8
  %110 = load i32, i32* %16, align 4
  %111 = sext i32 %110 to i64
  %112 = getelementptr inbounds %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %109, i64 %111
  %113 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %112, align 8
  %114 = load i32, i32* %17, align 4
  %115 = sext i32 %114 to i64
  %116 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %113, i64 %115
  %117 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %116, i32 0, i32 1
  store float 0x7FF0000000000000, float* %117, align 4
  br label %118

; <label>:118:                                    ; preds = %108
  %119 = load i32, i32* %17, align 4
  %120 = add nsw i32 %119, 1
  store i32 %120, i32* %17, align 4
  br label %104

; <label>:121:                                    ; preds = %104
  br label %122

; <label>:122:                                    ; preds = %121
  br label %123

; <label>:123:                                    ; preds = %122
  store i32 0, i32* %18, align 4
  br label %124

; <label>:124:                                    ; preds = %307, %123
  %125 = load i32, i32* %18, align 4
  %126 = load %struct.LSH_t*, %struct.LSH_t** %8, align 8
  %127 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %126, i32 0, i32 9
  %128 = load %struct.ohash_t*, %struct.ohash_t** %127, align 8
  %129 = load i32, i32* %16, align 4
  %130 = sext i32 %129 to i64
  %131 = getelementptr inbounds %struct.ohash_t, %struct.ohash_t* %128, i64 %130
  %132 = getelementptr inbounds %struct.ohash_t, %struct.ohash_t* %131, i32 0, i32 1
  %133 = load %struct.bucket_t*, %struct.bucket_t** %132, align 8
  %134 = load i32*, i32** %11, align 8
  %135 = load i32, i32* %16, align 4
  %136 = sext i32 %135 to i64
  %137 = getelementptr inbounds i32, i32* %134, i64 %136
  %138 = load i32, i32* %137, align 4
  %139 = zext i32 %138 to i64
  %140 = getelementptr inbounds %struct.bucket_t, %struct.bucket_t* %133, i64 %139
  %141 = getelementptr inbounds %struct.bucket_t, %struct.bucket_t* %140, i32 0, i32 2
  %142 = load i32, i32* %141, align 8
  %143 = icmp ult i32 %125, %142
  br i1 %143, label %144, label %310

; <label>:144:                                    ; preds = %124
  %145 = load %struct.LSH_t*, %struct.LSH_t** %8, align 8
  %146 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %145, i32 0, i32 9
  %147 = load %struct.ohash_t*, %struct.ohash_t** %146, align 8
  %148 = load i32, i32* %16, align 4
  %149 = sext i32 %148 to i64
  %150 = getelementptr inbounds %struct.ohash_t, %struct.ohash_t* %147, i64 %149
  %151 = getelementptr inbounds %struct.ohash_t, %struct.ohash_t* %150, i32 0, i32 1
  %152 = load %struct.bucket_t*, %struct.bucket_t** %151, align 8
  %153 = load i32*, i32** %11, align 8
  %154 = load i32, i32* %16, align 4
  %155 = sext i32 %154 to i64
  %156 = getelementptr inbounds i32, i32* %153, i64 %155
  %157 = load i32, i32* %156, align 4
  %158 = zext i32 %157 to i64
  %159 = getelementptr inbounds %struct.bucket_t, %struct.bucket_t* %152, i64 %158
  %160 = getelementptr inbounds %struct.bucket_t, %struct.bucket_t* %159, i32 0, i32 3
  %161 = load i32*, i32** %160, align 8
  %162 = load i32, i32* %18, align 4
  %163 = zext i32 %162 to i64
  %164 = getelementptr inbounds i32, i32* %161, i64 %163
  %165 = load i32, i32* %164, align 4
  store i32 %165, i32* %19, align 4
  %166 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %167 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %166, i32 0, i32 5
  %168 = load %struct.bitmap_t*, %struct.bitmap_t** %167, align 8
  %169 = load i32, i32* %19, align 4
  %170 = call i32 @bitmap_contain(%struct.bitmap_t* %168, i32 %169)
  %171 = icmp ne i32 %170, 0
  br i1 %171, label %306, label %172

; <label>:172:                                    ; preds = %144
  %173 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %174 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %173, i32 0, i32 5
  %175 = load %struct.bitmap_t*, %struct.bitmap_t** %174, align 8
  %176 = load i32, i32* %19, align 4
  %177 = call i32 @bitmap_insert(%struct.bitmap_t* %175, i32 %176)
  %178 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %179 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %178, i32 0, i32 1
  %180 = load %struct.cass_dataset_t*, %struct.cass_dataset_t** %179, align 8
  %181 = getelementptr inbounds %struct.cass_dataset_t, %struct.cass_dataset_t* %180, i32 0, i32 6
  %182 = load i8*, i8** %181, align 8
  %183 = load i32, i32* %19, align 4
  %184 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %185 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %184, i32 0, i32 1
  %186 = load %struct.cass_dataset_t*, %struct.cass_dataset_t** %185, align 8
  %187 = getelementptr inbounds %struct.cass_dataset_t, %struct.cass_dataset_t* %186, i32 0, i32 2
  %188 = load i32, i32* %187, align 8
  %189 = mul i32 %183, %188
  %190 = zext i32 %189 to i64
  %191 = getelementptr inbounds i8, i8* %182, i64 %190
  %192 = bitcast i8* %191 to %struct._cass_vec_t*
  store %struct._cass_vec_t* %192, %struct._cass_vec_t** %20, align 8
  %193 = load i32, i32* %19, align 4
  %194 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %13, i32 0, i32 0
  store i32 %193, i32* %194, align 4
  %195 = load i32, i32* %5, align 4
  %196 = load %struct._cass_vec_t*, %struct._cass_vec_t** %20, align 8
  %197 = getelementptr inbounds %struct._cass_vec_t, %struct._cass_vec_t* %196, i32 0, i32 2
  %198 = bitcast %union.anon* %197 to [14 x float]*
  %199 = getelementptr inbounds [14 x float], [14 x float]* %198, i32 0, i32 0
  %200 = load float*, float** %4, align 8
  %201 = call float @dist_L2_float(i32 %195, float* %199, float* %200)
  %202 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %13, i32 0, i32 1
  store float %201, float* %202, align 4
  %203 = load i32*, i32** %14, align 8
  %204 = load i32, i32* %16, align 4
  %205 = sext i32 %204 to i64
  %206 = getelementptr inbounds i32, i32* %203, i64 %205
  %207 = load i32, i32* %206, align 4
  %208 = add nsw i32 %207, 1
  store i32 %208, i32* %206, align 4
  %209 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %210 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %209, i32 0, i32 17
  %211 = load i32, i32* %210, align 8
  %212 = add nsw i32 %211, 1
  store i32 %212, i32* %210, align 8
  br label %213

; <label>:213:                                    ; preds = %172
  store i32 0, i32* %22, align 4
  store i32 0, i32* %21, align 4
  br label %214

; <label>:214:                                    ; preds = %248, %213
  %215 = load i32, i32* %21, align 4
  %216 = load i32, i32* %6, align 4
  %217 = icmp ult i32 %215, %216
  br i1 %217, label %218, label %251

; <label>:218:                                    ; preds = %214
  %219 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %13, i32 0, i32 1
  %220 = load float, float* %219, align 4
  %221 = load %struct.cass_list_entry_t**, %struct.cass_list_entry_t*** %12, align 8
  %222 = load i32, i32* %16, align 4
  %223 = sext i32 %222 to i64
  %224 = getelementptr inbounds %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %221, i64 %223
  %225 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %224, align 8
  %226 = load i32, i32* %21, align 4
  %227 = sext i32 %226 to i64
  %228 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %225, i64 %227
  %229 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %228, i32 0, i32 1
  %230 = load float, float* %229, align 4
  %231 = fcmp ogt float %220, %230
  br i1 %231, label %232, label %233

; <label>:232:                                    ; preds = %218
  br label %251

; <label>:233:                                    ; preds = %218
  %234 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %13, i32 0, i32 0
  %235 = load i32, i32* %234, align 4
  %236 = load %struct.cass_list_entry_t**, %struct.cass_list_entry_t*** %12, align 8
  %237 = load i32, i32* %16, align 4
  %238 = sext i32 %237 to i64
  %239 = getelementptr inbounds %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %236, i64 %238
  %240 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %239, align 8
  %241 = load i32, i32* %21, align 4
  %242 = sext i32 %241 to i64
  %243 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %240, i64 %242
  %244 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %243, i32 0, i32 0
  %245 = load i32, i32* %244, align 4
  %246 = icmp eq i32 %235, %245
  br i1 %246, label %247, label %248

; <label>:247:                                    ; preds = %233
  store i32 1, i32* %22, align 4
  br label %251

; <label>:248:                                    ; preds = %233
  %249 = load i32, i32* %21, align 4
  %250 = add nsw i32 %249, 1
  store i32 %250, i32* %21, align 4
  br label %214

; <label>:251:                                    ; preds = %247, %232, %214
  %252 = load i32, i32* %22, align 4
  %253 = icmp ne i32 %252, 0
  br i1 %253, label %254, label %255

; <label>:254:                                    ; preds = %251
  br label %305

; <label>:255:                                    ; preds = %251
  %256 = load i32, i32* %21, align 4
  %257 = icmp eq i32 %256, 0
  br i1 %257, label %258, label %259

; <label>:258:                                    ; preds = %255
  br label %305

; <label>:259:                                    ; preds = %255
  store i32 0, i32* %22, align 4
  br label %260

; <label>:260:                                    ; preds = %285, %259
  %261 = load i32, i32* %22, align 4
  %262 = load i32, i32* %21, align 4
  %263 = sub nsw i32 %262, 1
  %264 = icmp slt i32 %261, %263
  br i1 %264, label %265, label %288

; <label>:265:                                    ; preds = %260
  %266 = load %struct.cass_list_entry_t**, %struct.cass_list_entry_t*** %12, align 8
  %267 = load i32, i32* %16, align 4
  %268 = sext i32 %267 to i64
  %269 = getelementptr inbounds %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %266, i64 %268
  %270 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %269, align 8
  %271 = load i32, i32* %22, align 4
  %272 = sext i32 %271 to i64
  %273 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %270, i64 %272
  %274 = load %struct.cass_list_entry_t**, %struct.cass_list_entry_t*** %12, align 8
  %275 = load i32, i32* %16, align 4
  %276 = sext i32 %275 to i64
  %277 = getelementptr inbounds %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %274, i64 %276
  %278 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %277, align 8
  %279 = load i32, i32* %22, align 4
  %280 = add nsw i32 %279, 1
  %281 = sext i32 %280 to i64
  %282 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %278, i64 %281
  %283 = bitcast %struct.cass_list_entry_t* %273 to i8*
  %284 = bitcast %struct.cass_list_entry_t* %282 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %283, i8* %284, i64 8, i32 4, i1 false)
  br label %285

; <label>:285:                                    ; preds = %265
  %286 = load i32, i32* %22, align 4
  %287 = add nsw i32 %286, 1
  store i32 %287, i32* %22, align 4
  br label %260

; <label>:288:                                    ; preds = %260
  %289 = load %struct.cass_list_entry_t**, %struct.cass_list_entry_t*** %12, align 8
  %290 = load i32, i32* %16, align 4
  %291 = sext i32 %290 to i64
  %292 = getelementptr inbounds %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %289, i64 %291
  %293 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %292, align 8
  %294 = load i32, i32* %22, align 4
  %295 = sext i32 %294 to i64
  %296 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %293, i64 %295
  %297 = bitcast %struct.cass_list_entry_t* %296 to i8*
  %298 = bitcast %struct.cass_list_entry_t* %13 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %297, i8* %298, i64 8, i32 4, i1 false)
  %299 = load i32*, i32** %15, align 8
  %300 = load i32, i32* %16, align 4
  %301 = sext i32 %300 to i64
  %302 = getelementptr inbounds i32, i32* %299, i64 %301
  %303 = load i32, i32* %302, align 4
  %304 = add nsw i32 %303, 1
  store i32 %304, i32* %302, align 4
  br label %305

; <label>:305:                                    ; preds = %288, %258, %254
  br label %306

; <label>:306:                                    ; preds = %305, %144
  br label %307

; <label>:307:                                    ; preds = %306
  %308 = load i32, i32* %18, align 4
  %309 = add i32 %308, 1
  store i32 %309, i32* %18, align 4
  br label %124

; <label>:310:                                    ; preds = %124
  br label %311

; <label>:311:                                    ; preds = %310
  %312 = load %struct.ptb_vec_t**, %struct.ptb_vec_t*** %9, align 8
  %313 = load i32, i32* %16, align 4
  %314 = sext i32 %313 to i64
  %315 = getelementptr inbounds %struct.ptb_vec_t*, %struct.ptb_vec_t** %312, i64 %314
  %316 = load %struct.ptb_vec_t*, %struct.ptb_vec_t** %315, align 8
  %317 = load %struct.LSH_t*, %struct.LSH_t** %8, align 8
  %318 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %317, i32 0, i32 1
  %319 = load i32, i32* %318, align 4
  %320 = mul i32 %319, 2
  call void @ptb_qsort(%struct.ptb_vec_t* %316, i32 %320)
  %321 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %322 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %321, i32 0, i32 9
  %323 = load i32*, i32** %322, align 8
  %324 = load i32, i32* %16, align 4
  %325 = sext i32 %324 to i64
  %326 = getelementptr inbounds i32, i32* %323, i64 %325
  store i32 0, i32* %326, align 4
  %327 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %328 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %327, i32 0, i32 10
  %329 = load %struct.ptb_vec_t*, %struct.ptb_vec_t** %328, align 8
  %330 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %331 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %330, i32 0, i32 11
  %332 = load %struct.ptb_vec_t**, %struct.ptb_vec_t*** %331, align 8
  %333 = load i32, i32* %16, align 4
  %334 = sext i32 %333 to i64
  %335 = getelementptr inbounds %struct.ptb_vec_t*, %struct.ptb_vec_t** %332, i64 %334
  %336 = load %struct.ptb_vec_t*, %struct.ptb_vec_t** %335, align 8
  %337 = load %struct.ptb_vec_t**, %struct.ptb_vec_t*** %9, align 8
  %338 = load i32, i32* %16, align 4
  %339 = sext i32 %338 to i64
  %340 = getelementptr inbounds %struct.ptb_vec_t*, %struct.ptb_vec_t** %337, i64 %339
  %341 = load %struct.ptb_vec_t*, %struct.ptb_vec_t** %340, align 8
  %342 = load %struct.LSH_t*, %struct.LSH_t** %8, align 8
  %343 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %342, i32 0, i32 1
  %344 = load i32, i32* %343, align 4
  %345 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %346 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %345, i32 0, i32 4
  %347 = load i32, i32* %346, align 8
  %348 = call i32 @map_perturb_vector(%struct.ptb_vec_t* %329, %struct.ptb_vec_t* %336, %struct.ptb_vec_t* %341, i32 %344, i32 %347)
  br label %349

; <label>:349:                                    ; preds = %311
  %350 = load i32, i32* %16, align 4
  %351 = add nsw i32 %350, 1
  store i32 %351, i32* %16, align 4
  br label %89

; <label>:352:                                    ; preds = %89
  %353 = load %struct.LSH_t*, %struct.LSH_t** %8, align 8
  %354 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %353, i32 0, i32 12
  %355 = load %struct.LSH_est_t*, %struct.LSH_est_t** %354, align 8
  %356 = icmp ne %struct.LSH_est_t* %355, null
  br i1 %356, label %357, label %360

; <label>:357:                                    ; preds = %352
  %358 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  call void @LSH_query_merge(%struct.LSH_query_t* %358)
  %359 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  call void @LSH_query_local(%struct.LSH_query_t* %359)
  br label %360

; <label>:360:                                    ; preds = %357, %352
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @LSH_query_probe(%struct.LSH_query_t*, float*, i32, i32) #0 {
  %5 = alloca %struct.LSH_query_t*, align 8
  %6 = alloca float*, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca %struct.LSH_t*, align 8
  %12 = alloca i32**, align 8
  %13 = alloca %struct.cass_list_entry_t*, align 8
  %14 = alloca %struct.cass_list_entry_t, align 4
  %15 = alloca %struct.ptb_vec_t, align 8
  %16 = alloca i32*, align 8
  %17 = alloca i32*, align 8
  %18 = alloca i32, align 4
  %19 = alloca i32, align 4
  %20 = alloca i32, align 4
  %21 = alloca %struct._cass_vec_t*, align 8
  %22 = alloca i32, align 4
  %23 = alloca i32, align 4
  store %struct.LSH_query_t* %0, %struct.LSH_query_t** %5, align 8
  store float* %1, float** %6, align 8
  store i32 %2, i32* %7, align 4
  store i32 %3, i32* %8, align 4
  %24 = load %struct.LSH_query_t*, %struct.LSH_query_t** %5, align 8
  %25 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %24, i32 0, i32 0
  %26 = load %struct.LSH_t*, %struct.LSH_t** %25, align 8
  %27 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %26, i32 0, i32 0
  %28 = load i32, i32* %27, align 8
  store i32 %28, i32* %9, align 4
  %29 = load %struct.LSH_query_t*, %struct.LSH_query_t** %5, align 8
  %30 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %29, i32 0, i32 2
  %31 = load i32, i32* %30, align 8
  store i32 %31, i32* %10, align 4
  %32 = load %struct.LSH_query_t*, %struct.LSH_query_t** %5, align 8
  %33 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %32, i32 0, i32 0
  %34 = load %struct.LSH_t*, %struct.LSH_t** %33, align 8
  store %struct.LSH_t* %34, %struct.LSH_t** %11, align 8
  %35 = load %struct.LSH_query_t*, %struct.LSH_query_t** %5, align 8
  %36 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %35, i32 0, i32 6
  %37 = load i32**, i32*** %36, align 8
  store i32** %37, i32*** %12, align 8
  %38 = load i32, i32* %8, align 4
  %39 = icmp eq i32 %38, 0
  br i1 %39, label %40, label %48

; <label>:40:                                     ; preds = %4
  %41 = load %struct.LSH_query_t*, %struct.LSH_query_t** %5, align 8
  %42 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %41, i32 0, i32 13
  %43 = load %struct.cass_list_entry_t**, %struct.cass_list_entry_t*** %42, align 8
  %44 = load i32, i32* %7, align 4
  %45 = sext i32 %44 to i64
  %46 = getelementptr inbounds %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %43, i64 %45
  %47 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %46, align 8
  br label %52

; <label>:48:                                     ; preds = %4
  %49 = load %struct.LSH_query_t*, %struct.LSH_query_t** %5, align 8
  %50 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %49, i32 0, i32 12
  %51 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %50, align 8
  br label %52

; <label>:52:                                     ; preds = %48, %40
  %53 = phi %struct.cass_list_entry_t* [ %47, %40 ], [ %51, %48 ]
  store %struct.cass_list_entry_t* %53, %struct.cass_list_entry_t** %13, align 8
  %54 = load %struct.LSH_query_t*, %struct.LSH_query_t** %5, align 8
  %55 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %54, i32 0, i32 14
  %56 = load i32*, i32** %55, align 8
  store i32* %56, i32** %16, align 8
  %57 = load %struct.LSH_query_t*, %struct.LSH_query_t** %5, align 8
  %58 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %57, i32 0, i32 15
  %59 = load i32*, i32** %58, align 8
  store i32* %59, i32** %17, align 8
  %60 = load %struct.LSH_query_t*, %struct.LSH_query_t** %5, align 8
  %61 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %60, i32 0, i32 11
  %62 = load %struct.ptb_vec_t**, %struct.ptb_vec_t*** %61, align 8
  %63 = load i32, i32* %7, align 4
  %64 = sext i32 %63 to i64
  %65 = getelementptr inbounds %struct.ptb_vec_t*, %struct.ptb_vec_t** %62, i64 %64
  %66 = load %struct.ptb_vec_t*, %struct.ptb_vec_t** %65, align 8
  %67 = load %struct.LSH_query_t*, %struct.LSH_query_t** %5, align 8
  %68 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %67, i32 0, i32 9
  %69 = load i32*, i32** %68, align 8
  %70 = load i32, i32* %7, align 4
  %71 = sext i32 %70 to i64
  %72 = getelementptr inbounds i32, i32* %69, i64 %71
  %73 = load i32, i32* %72, align 4
  %74 = add nsw i32 %73, 1
  store i32 %74, i32* %72, align 4
  %75 = sext i32 %73 to i64
  %76 = getelementptr inbounds %struct.ptb_vec_t, %struct.ptb_vec_t* %66, i64 %75
  %77 = bitcast %struct.ptb_vec_t* %15 to i8*
  %78 = bitcast %struct.ptb_vec_t* %76 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %77, i8* %78, i64 24, i32 8, i1 false)
  %79 = load %struct.LSH_query_t*, %struct.LSH_query_t** %5, align 8
  %80 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %79, i32 0, i32 0
  %81 = load %struct.LSH_t*, %struct.LSH_t** %80, align 8
  %82 = load i32**, i32*** %12, align 8
  %83 = load i32, i32* %7, align 4
  call void @LSH_hash2_perturb(%struct.LSH_t* %81, i32** %82, i32* %18, %struct.ptb_vec_t* %15, i32 %83)
  br label %84

; <label>:84:                                     ; preds = %52
  store i32 0, i32* %19, align 4
  br label %85

; <label>:85:                                     ; preds = %240, %84
  %86 = load i32, i32* %19, align 4
  %87 = load %struct.LSH_t*, %struct.LSH_t** %11, align 8
  %88 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %87, i32 0, i32 9
  %89 = load %struct.ohash_t*, %struct.ohash_t** %88, align 8
  %90 = load i32, i32* %7, align 4
  %91 = sext i32 %90 to i64
  %92 = getelementptr inbounds %struct.ohash_t, %struct.ohash_t* %89, i64 %91
  %93 = getelementptr inbounds %struct.ohash_t, %struct.ohash_t* %92, i32 0, i32 1
  %94 = load %struct.bucket_t*, %struct.bucket_t** %93, align 8
  %95 = load i32, i32* %18, align 4
  %96 = zext i32 %95 to i64
  %97 = getelementptr inbounds %struct.bucket_t, %struct.bucket_t* %94, i64 %96
  %98 = getelementptr inbounds %struct.bucket_t, %struct.bucket_t* %97, i32 0, i32 2
  %99 = load i32, i32* %98, align 8
  %100 = icmp ult i32 %86, %99
  br i1 %100, label %101, label %243

; <label>:101:                                    ; preds = %85
  %102 = load %struct.LSH_t*, %struct.LSH_t** %11, align 8
  %103 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %102, i32 0, i32 9
  %104 = load %struct.ohash_t*, %struct.ohash_t** %103, align 8
  %105 = load i32, i32* %7, align 4
  %106 = sext i32 %105 to i64
  %107 = getelementptr inbounds %struct.ohash_t, %struct.ohash_t* %104, i64 %106
  %108 = getelementptr inbounds %struct.ohash_t, %struct.ohash_t* %107, i32 0, i32 1
  %109 = load %struct.bucket_t*, %struct.bucket_t** %108, align 8
  %110 = load i32, i32* %18, align 4
  %111 = zext i32 %110 to i64
  %112 = getelementptr inbounds %struct.bucket_t, %struct.bucket_t* %109, i64 %111
  %113 = getelementptr inbounds %struct.bucket_t, %struct.bucket_t* %112, i32 0, i32 3
  %114 = load i32*, i32** %113, align 8
  %115 = load i32, i32* %19, align 4
  %116 = zext i32 %115 to i64
  %117 = getelementptr inbounds i32, i32* %114, i64 %116
  %118 = load i32, i32* %117, align 4
  store i32 %118, i32* %20, align 4
  %119 = load %struct.LSH_query_t*, %struct.LSH_query_t** %5, align 8
  %120 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %119, i32 0, i32 5
  %121 = load %struct.bitmap_t*, %struct.bitmap_t** %120, align 8
  %122 = load i32, i32* %20, align 4
  %123 = call i32 @bitmap_contain(%struct.bitmap_t* %121, i32 %122)
  %124 = icmp ne i32 %123, 0
  br i1 %124, label %239, label %125

; <label>:125:                                    ; preds = %101
  %126 = load %struct.LSH_query_t*, %struct.LSH_query_t** %5, align 8
  %127 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %126, i32 0, i32 5
  %128 = load %struct.bitmap_t*, %struct.bitmap_t** %127, align 8
  %129 = load i32, i32* %20, align 4
  %130 = call i32 @bitmap_insert(%struct.bitmap_t* %128, i32 %129)
  %131 = load %struct.LSH_query_t*, %struct.LSH_query_t** %5, align 8
  %132 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %131, i32 0, i32 1
  %133 = load %struct.cass_dataset_t*, %struct.cass_dataset_t** %132, align 8
  %134 = getelementptr inbounds %struct.cass_dataset_t, %struct.cass_dataset_t* %133, i32 0, i32 6
  %135 = load i8*, i8** %134, align 8
  %136 = load i32, i32* %20, align 4
  %137 = load %struct.LSH_query_t*, %struct.LSH_query_t** %5, align 8
  %138 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %137, i32 0, i32 1
  %139 = load %struct.cass_dataset_t*, %struct.cass_dataset_t** %138, align 8
  %140 = getelementptr inbounds %struct.cass_dataset_t, %struct.cass_dataset_t* %139, i32 0, i32 2
  %141 = load i32, i32* %140, align 8
  %142 = mul i32 %136, %141
  %143 = zext i32 %142 to i64
  %144 = getelementptr inbounds i8, i8* %135, i64 %143
  %145 = bitcast i8* %144 to %struct._cass_vec_t*
  store %struct._cass_vec_t* %145, %struct._cass_vec_t** %21, align 8
  %146 = load i32*, i32** %16, align 8
  %147 = load i32, i32* %7, align 4
  %148 = sext i32 %147 to i64
  %149 = getelementptr inbounds i32, i32* %146, i64 %148
  %150 = load i32, i32* %149, align 4
  %151 = add nsw i32 %150, 1
  store i32 %151, i32* %149, align 4
  %152 = load %struct.LSH_query_t*, %struct.LSH_query_t** %5, align 8
  %153 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %152, i32 0, i32 17
  %154 = load i32, i32* %153, align 8
  %155 = add nsw i32 %154, 1
  store i32 %155, i32* %153, align 8
  %156 = load i32, i32* %20, align 4
  %157 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %14, i32 0, i32 0
  store i32 %156, i32* %157, align 4
  %158 = load i32, i32* %9, align 4
  %159 = load %struct._cass_vec_t*, %struct._cass_vec_t** %21, align 8
  %160 = getelementptr inbounds %struct._cass_vec_t, %struct._cass_vec_t* %159, i32 0, i32 2
  %161 = bitcast %union.anon* %160 to [14 x float]*
  %162 = getelementptr inbounds [14 x float], [14 x float]* %161, i32 0, i32 0
  %163 = load float*, float** %6, align 8
  %164 = call float @dist_L2_float(i32 %158, float* %162, float* %163)
  %165 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %14, i32 0, i32 1
  store float %164, float* %165, align 4
  br label %166

; <label>:166:                                    ; preds = %125
  store i32 0, i32* %23, align 4
  store i32 0, i32* %22, align 4
  br label %167

; <label>:167:                                    ; preds = %193, %166
  %168 = load i32, i32* %22, align 4
  %169 = load i32, i32* %10, align 4
  %170 = icmp ult i32 %168, %169
  br i1 %170, label %171, label %196

; <label>:171:                                    ; preds = %167
  %172 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %14, i32 0, i32 1
  %173 = load float, float* %172, align 4
  %174 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %13, align 8
  %175 = load i32, i32* %22, align 4
  %176 = sext i32 %175 to i64
  %177 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %174, i64 %176
  %178 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %177, i32 0, i32 1
  %179 = load float, float* %178, align 4
  %180 = fcmp ogt float %173, %179
  br i1 %180, label %181, label %182

; <label>:181:                                    ; preds = %171
  br label %196

; <label>:182:                                    ; preds = %171
  %183 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %14, i32 0, i32 0
  %184 = load i32, i32* %183, align 4
  %185 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %13, align 8
  %186 = load i32, i32* %22, align 4
  %187 = sext i32 %186 to i64
  %188 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %185, i64 %187
  %189 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %188, i32 0, i32 0
  %190 = load i32, i32* %189, align 4
  %191 = icmp eq i32 %184, %190
  br i1 %191, label %192, label %193

; <label>:192:                                    ; preds = %182
  store i32 1, i32* %23, align 4
  br label %196

; <label>:193:                                    ; preds = %182
  %194 = load i32, i32* %22, align 4
  %195 = add nsw i32 %194, 1
  store i32 %195, i32* %22, align 4
  br label %167

; <label>:196:                                    ; preds = %192, %181, %167
  %197 = load i32, i32* %23, align 4
  %198 = icmp ne i32 %197, 0
  br i1 %198, label %199, label %200

; <label>:199:                                    ; preds = %196
  br label %238

; <label>:200:                                    ; preds = %196
  %201 = load i32, i32* %22, align 4
  %202 = icmp eq i32 %201, 0
  br i1 %202, label %203, label %204

; <label>:203:                                    ; preds = %200
  br label %238

; <label>:204:                                    ; preds = %200
  store i32 0, i32* %23, align 4
  br label %205

; <label>:205:                                    ; preds = %222, %204
  %206 = load i32, i32* %23, align 4
  %207 = load i32, i32* %22, align 4
  %208 = sub nsw i32 %207, 1
  %209 = icmp slt i32 %206, %208
  br i1 %209, label %210, label %225

; <label>:210:                                    ; preds = %205
  %211 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %13, align 8
  %212 = load i32, i32* %23, align 4
  %213 = sext i32 %212 to i64
  %214 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %211, i64 %213
  %215 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %13, align 8
  %216 = load i32, i32* %23, align 4
  %217 = add nsw i32 %216, 1
  %218 = sext i32 %217 to i64
  %219 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %215, i64 %218
  %220 = bitcast %struct.cass_list_entry_t* %214 to i8*
  %221 = bitcast %struct.cass_list_entry_t* %219 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %220, i8* %221, i64 8, i32 4, i1 false)
  br label %222

; <label>:222:                                    ; preds = %210
  %223 = load i32, i32* %23, align 4
  %224 = add nsw i32 %223, 1
  store i32 %224, i32* %23, align 4
  br label %205

; <label>:225:                                    ; preds = %205
  %226 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %13, align 8
  %227 = load i32, i32* %23, align 4
  %228 = sext i32 %227 to i64
  %229 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %226, i64 %228
  %230 = bitcast %struct.cass_list_entry_t* %229 to i8*
  %231 = bitcast %struct.cass_list_entry_t* %14 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %230, i8* %231, i64 8, i32 4, i1 false)
  %232 = load i32*, i32** %17, align 8
  %233 = load i32, i32* %7, align 4
  %234 = sext i32 %233 to i64
  %235 = getelementptr inbounds i32, i32* %232, i64 %234
  %236 = load i32, i32* %235, align 4
  %237 = add nsw i32 %236, 1
  store i32 %237, i32* %235, align 4
  br label %238

; <label>:238:                                    ; preds = %225, %203, %199
  br label %239

; <label>:239:                                    ; preds = %238, %101
  br label %240

; <label>:240:                                    ; preds = %239
  %241 = load i32, i32* %19, align 4
  %242 = add i32 %241, 1
  store i32 %242, i32* %19, align 4
  br label %85

; <label>:243:                                    ; preds = %85
  br label %244

; <label>:244:                                    ; preds = %243
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define void @LSH_query_recall(%struct.LSH_query_t*, float*, float) #0 {
  %4 = alloca %struct.LSH_query_t*, align 8
  %5 = alloca float*, align 8
  %6 = alloca float, align 4
  %7 = alloca %struct.LSH_recall_t*, align 8
  %8 = alloca %struct.cass_list_entry_t*, align 8
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca i32, align 4
  %14 = alloca float, align 4
  store %struct.LSH_query_t* %0, %struct.LSH_query_t** %4, align 8
  store float* %1, float** %5, align 8
  store float %2, float* %6, align 4
  %15 = load %struct.LSH_query_t*, %struct.LSH_query_t** %4, align 8
  %16 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %15, i32 0, i32 0
  %17 = load %struct.LSH_t*, %struct.LSH_t** %16, align 8
  %18 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %17, i32 0, i32 13
  store %struct.LSH_recall_t* %18, %struct.LSH_recall_t** %7, align 8
  %19 = load %struct.LSH_query_t*, %struct.LSH_query_t** %4, align 8
  %20 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %19, i32 0, i32 12
  %21 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %20, align 8
  store %struct.cass_list_entry_t* %21, %struct.cass_list_entry_t** %8, align 8
  %22 = load %struct.LSH_query_t*, %struct.LSH_query_t** %4, align 8
  %23 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %22, i32 0, i32 4
  %24 = load i32, i32* %23, align 8
  store i32 %24, i32* %9, align 4
  %25 = load %struct.LSH_query_t*, %struct.LSH_query_t** %4, align 8
  %26 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %25, i32 0, i32 3
  %27 = load i32, i32* %26, align 4
  store i32 %27, i32* %10, align 4
  %28 = load %struct.LSH_query_t*, %struct.LSH_query_t** %4, align 8
  %29 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %28, i32 0, i32 2
  %30 = load i32, i32* %29, align 8
  store i32 %30, i32* %11, align 4
  %31 = load %struct.LSH_query_t*, %struct.LSH_query_t** %4, align 8
  %32 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %31, i32 0, i32 17
  store i32 0, i32* %32, align 8
  %33 = load %struct.LSH_query_t*, %struct.LSH_query_t** %4, align 8
  %34 = load float*, float** %5, align 8
  call void @LSH_query_bootstrap(%struct.LSH_query_t* %33, float* %34)
  %35 = load %struct.LSH_query_t*, %struct.LSH_query_t** %4, align 8
  call void @LSH_query_merge(%struct.LSH_query_t* %35)
  store i32 0, i32* %12, align 4
  br label %36

; <label>:36:                                     ; preds = %86, %3
  %37 = load i32, i32* %12, align 4
  %38 = load i32, i32* %9, align 4
  %39 = icmp ult i32 %37, %38
  br i1 %39, label %40, label %89

; <label>:40:                                     ; preds = %36
  %41 = load %struct.LSH_recall_t*, %struct.LSH_recall_t** %7, align 8
  %42 = icmp ne %struct.LSH_recall_t* %41, null
  br i1 %42, label %43, label %73

; <label>:43:                                     ; preds = %40
  store float 0.000000e+00, float* %14, align 4
  store i32 0, i32* %13, align 4
  br label %44

; <label>:44:                                     ; preds = %60, %43
  %45 = load i32, i32* %13, align 4
  %46 = load i32, i32* %11, align 4
  %47 = icmp slt i32 %45, %46
  br i1 %47, label %48, label %63

; <label>:48:                                     ; preds = %44
  %49 = load %struct.LSH_recall_t*, %struct.LSH_recall_t** %7, align 8
  %50 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %8, align 8
  %51 = load i32, i32* %13, align 4
  %52 = sext i32 %51 to i64
  %53 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %50, i64 %52
  %54 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %53, i32 0, i32 1
  %55 = load float, float* %54, align 4
  %56 = load i32, i32* %12, align 4
  %57 = call float @LSH_recall(%struct.LSH_recall_t* %49, float %55, i32 %56)
  %58 = load float, float* %14, align 4
  %59 = fadd float %58, %57
  store float %59, float* %14, align 4
  br label %60

; <label>:60:                                     ; preds = %48
  %61 = load i32, i32* %13, align 4
  %62 = add nsw i32 %61, 1
  store i32 %62, i32* %13, align 4
  br label %44

; <label>:63:                                     ; preds = %44
  %64 = load i32, i32* %11, align 4
  %65 = sitofp i32 %64 to float
  %66 = load float, float* %14, align 4
  %67 = fdiv float %66, %65
  store float %67, float* %14, align 4
  %68 = load float, float* %14, align 4
  %69 = load float, float* %6, align 4
  %70 = fcmp oge float %68, %69
  br i1 %70, label %71, label %72

; <label>:71:                                     ; preds = %63
  br label %89

; <label>:72:                                     ; preds = %63
  br label %73

; <label>:73:                                     ; preds = %72, %40
  store i32 0, i32* %13, align 4
  br label %74

; <label>:74:                                     ; preds = %82, %73
  %75 = load i32, i32* %13, align 4
  %76 = load i32, i32* %10, align 4
  %77 = icmp ult i32 %75, %76
  br i1 %77, label %78, label %85

; <label>:78:                                     ; preds = %74
  %79 = load %struct.LSH_query_t*, %struct.LSH_query_t** %4, align 8
  %80 = load float*, float** %5, align 8
  %81 = load i32, i32* %13, align 4
  call void @LSH_query_probe(%struct.LSH_query_t* %79, float* %80, i32 %81, i32 1)
  br label %82

; <label>:82:                                     ; preds = %78
  %83 = load i32, i32* %13, align 4
  %84 = add nsw i32 %83, 1
  store i32 %84, i32* %13, align 4
  br label %74

; <label>:85:                                     ; preds = %74
  br label %86

; <label>:86:                                     ; preds = %85
  %87 = load i32, i32* %12, align 4
  %88 = add nsw i32 %87, 1
  store i32 %88, i32* %12, align 4
  br label %36

; <label>:89:                                     ; preds = %71, %36
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal float @LSH_recall(%struct.LSH_recall_t*, float, i32) #0 {
  %4 = alloca float, align 4
  %5 = alloca %struct.LSH_recall_t*, align 8
  %6 = alloca float, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  store %struct.LSH_recall_t* %0, %struct.LSH_recall_t** %5, align 8
  store float %1, float* %6, align 4
  store i32 %2, i32* %7, align 4
  %9 = load float, float* %6, align 4
  %10 = load %struct.LSH_recall_t*, %struct.LSH_recall_t** %5, align 8
  %11 = getelementptr inbounds %struct.LSH_recall_t, %struct.LSH_recall_t* %10, i32 0, i32 2
  %12 = load float, float* %11, align 8
  %13 = fcmp olt float %9, %12
  br i1 %13, label %14, label %15

; <label>:14:                                     ; preds = %3
  store float 1.000000e+00, float* %4, align 4
  br label %53

; <label>:15:                                     ; preds = %3
  %16 = load float, float* %6, align 4
  %17 = load %struct.LSH_recall_t*, %struct.LSH_recall_t** %5, align 8
  %18 = getelementptr inbounds %struct.LSH_recall_t, %struct.LSH_recall_t* %17, i32 0, i32 3
  %19 = load float, float* %18, align 4
  %20 = fcmp ogt float %16, %19
  br i1 %20, label %21, label %22

; <label>:21:                                     ; preds = %15
  store float 0.000000e+00, float* %4, align 4
  br label %53

; <label>:22:                                     ; preds = %15
  %23 = load float, float* %6, align 4
  %24 = load %struct.LSH_recall_t*, %struct.LSH_recall_t** %5, align 8
  %25 = getelementptr inbounds %struct.LSH_recall_t, %struct.LSH_recall_t* %24, i32 0, i32 2
  %26 = load float, float* %25, align 8
  %27 = fsub float %23, %26
  %28 = load %struct.LSH_recall_t*, %struct.LSH_recall_t** %5, align 8
  %29 = getelementptr inbounds %struct.LSH_recall_t, %struct.LSH_recall_t* %28, i32 0, i32 0
  %30 = load i32, i32* %29, align 8
  %31 = uitofp i32 %30 to float
  %32 = fmul float %27, %31
  %33 = load %struct.LSH_recall_t*, %struct.LSH_recall_t** %5, align 8
  %34 = getelementptr inbounds %struct.LSH_recall_t, %struct.LSH_recall_t* %33, i32 0, i32 3
  %35 = load float, float* %34, align 4
  %36 = load %struct.LSH_recall_t*, %struct.LSH_recall_t** %5, align 8
  %37 = getelementptr inbounds %struct.LSH_recall_t, %struct.LSH_recall_t* %36, i32 0, i32 2
  %38 = load float, float* %37, align 8
  %39 = fsub float %35, %38
  %40 = fdiv float %32, %39
  %41 = fptosi float %40 to i32
  store i32 %41, i32* %8, align 4
  %42 = load %struct.LSH_recall_t*, %struct.LSH_recall_t** %5, align 8
  %43 = getelementptr inbounds %struct.LSH_recall_t, %struct.LSH_recall_t* %42, i32 0, i32 4
  %44 = load float**, float*** %43, align 8
  %45 = load i32, i32* %7, align 4
  %46 = sext i32 %45 to i64
  %47 = getelementptr inbounds float*, float** %44, i64 %46
  %48 = load float*, float** %47, align 8
  %49 = load i32, i32* %8, align 4
  %50 = sext i32 %49 to i64
  %51 = getelementptr inbounds float, float* %48, i64 %50
  %52 = load float, float* %51, align 4
  store float %52, float* %4, align 4
  br label %53

; <label>:53:                                     ; preds = %22, %21, %14
  %54 = load float, float* %4, align 4
  ret float %54
}

; Function Attrs: noinline nounwind optnone uwtable
define i32 @LSH_query_select(%struct.LSH_query_t*, i32) #0 {
  %3 = alloca i32, align 4
  %4 = alloca %struct.LSH_query_t*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i8*, align 8
  %7 = alloca i32, align 4
  %8 = alloca i32
  store %struct.LSH_query_t* %0, %struct.LSH_query_t** %4, align 8
  store i32 %1, i32* %5, align 4
  %9 = load %struct.LSH_query_t*, %struct.LSH_query_t** %4, align 8
  %10 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %9, i32 0, i32 3
  %11 = load i32, i32* %10, align 4
  %12 = zext i32 %11 to i64
  %13 = call i8* @llvm.stacksave()
  store i8* %13, i8** %6, align 8
  %14 = alloca %struct.cass_list_entry_t, i64 %12, align 16
  store i32 0, i32* %7, align 4
  br label %15

; <label>:15:                                     ; preds = %38, %2
  %16 = load i32, i32* %7, align 4
  %17 = load %struct.LSH_query_t*, %struct.LSH_query_t** %4, align 8
  %18 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %17, i32 0, i32 3
  %19 = load i32, i32* %18, align 4
  %20 = icmp ult i32 %16, %19
  br i1 %20, label %21, label %41

; <label>:21:                                     ; preds = %15
  %22 = load i32, i32* %7, align 4
  %23 = load i32, i32* %7, align 4
  %24 = sext i32 %23 to i64
  %25 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %14, i64 %24
  %26 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %25, i32 0, i32 0
  store i32 %22, i32* %26, align 8
  %27 = load %struct.LSH_query_t*, %struct.LSH_query_t** %4, align 8
  %28 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %27, i32 0, i32 16
  %29 = load float*, float** %28, align 8
  %30 = load i32, i32* %7, align 4
  %31 = sext i32 %30 to i64
  %32 = getelementptr inbounds float, float* %29, i64 %31
  %33 = load float, float* %32, align 4
  %34 = load i32, i32* %7, align 4
  %35 = sext i32 %34 to i64
  %36 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %14, i64 %35
  %37 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %36, i32 0, i32 1
  store float %33, float* %37, align 4
  br label %38

; <label>:38:                                     ; preds = %21
  %39 = load i32, i32* %7, align 4
  %40 = add nsw i32 %39, 1
  store i32 %40, i32* %7, align 4
  br label %15

; <label>:41:                                     ; preds = %15
  %42 = load %struct.LSH_query_t*, %struct.LSH_query_t** %4, align 8
  %43 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %42, i32 0, i32 3
  %44 = load i32, i32* %43, align 4
  call void @__cass_list_entry_qsort(%struct.cass_list_entry_t* %14, i32 %44)
  store i32 0, i32* %7, align 4
  br label %45

; <label>:45:                                     ; preds = %63, %41
  %46 = load i32, i32* %7, align 4
  %47 = load %struct.LSH_query_t*, %struct.LSH_query_t** %4, align 8
  %48 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %47, i32 0, i32 3
  %49 = load i32, i32* %48, align 4
  %50 = sub i32 %49, 1
  %51 = icmp ult i32 %46, %50
  br i1 %51, label %52, label %66

; <label>:52:                                     ; preds = %45
  %53 = call i32 @rand() #6
  %54 = srem i32 %53, 3
  %55 = icmp slt i32 %54, 2
  br i1 %55, label %56, label %62

; <label>:56:                                     ; preds = %52
  %57 = load i32, i32* %7, align 4
  %58 = sext i32 %57 to i64
  %59 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %14, i64 %58
  %60 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %59, i32 0, i32 0
  %61 = load i32, i32* %60, align 8
  store i32 %61, i32* %3, align 4
  store i32 1, i32* %8, align 4
  br label %72

; <label>:62:                                     ; preds = %52
  br label %63

; <label>:63:                                     ; preds = %62
  %64 = load i32, i32* %7, align 4
  %65 = add nsw i32 %64, 1
  store i32 %65, i32* %7, align 4
  br label %45

; <label>:66:                                     ; preds = %45
  %67 = load i32, i32* %7, align 4
  %68 = sext i32 %67 to i64
  %69 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %14, i64 %68
  %70 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %69, i32 0, i32 0
  %71 = load i32, i32* %70, align 8
  store i32 %71, i32* %3, align 4
  store i32 1, i32* %8, align 4
  br label %72

; <label>:72:                                     ; preds = %66, %56
  %73 = load i8*, i8** %6, align 8
  call void @llvm.stackrestore(i8* %73)
  %74 = load i32, i32* %3, align 4
  ret i32 %74
}

; Function Attrs: nounwind
declare i8* @llvm.stacksave() #6

declare void @__cass_list_entry_qsort(%struct.cass_list_entry_t*, i32) #2

; Function Attrs: nounwind
declare i32 @rand() #4

; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*) #6

; Function Attrs: noinline nounwind optnone uwtable
define void @LSH_query_boost(%struct.LSH_query_t*, float*) #0 {
  %3 = alloca %struct.LSH_query_t*, align 8
  %4 = alloca float*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  store %struct.LSH_query_t* %0, %struct.LSH_query_t** %3, align 8
  store float* %1, float** %4, align 8
  %8 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %9 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %8, i32 0, i32 4
  %10 = load i32, i32* %9, align 8
  store i32 %10, i32* %5, align 4
  store i32 0, i32* %7, align 4
  %11 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %12 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %11, i32 0, i32 17
  store i32 0, i32* %12, align 8
  %13 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %14 = load float*, float** %4, align 8
  call void @LSH_query_bootstrap(%struct.LSH_query_t* %13, float* %14)
  store i32 0, i32* %6, align 4
  br label %15

; <label>:15:                                     ; preds = %83, %2
  %16 = load i32, i32* %6, align 4
  %17 = load i32, i32* %5, align 4
  %18 = icmp ult i32 %16, %17
  br i1 %18, label %19, label %86

; <label>:19:                                     ; preds = %15
  %20 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %21 = load i32, i32* %7, align 4
  %22 = call i32 @LSH_query_select(%struct.LSH_query_t* %20, i32 %21)
  store i32 %22, i32* %7, align 4
  %23 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %24 = load float*, float** %4, align 8
  %25 = load i32, i32* %7, align 4
  call void @LSH_query_probe(%struct.LSH_query_t* %23, float* %24, i32 %25, i32 0)
  %26 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %27 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %26, i32 0, i32 14
  %28 = load i32*, i32** %27, align 8
  %29 = load i32, i32* %7, align 4
  %30 = sext i32 %29 to i64
  %31 = getelementptr inbounds i32, i32* %28, i64 %30
  %32 = load i32, i32* %31, align 4
  %33 = icmp sgt i32 %32, 0
  br i1 %33, label %34, label %75

; <label>:34:                                     ; preds = %19
  %35 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %36 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %35, i32 0, i32 15
  %37 = load i32*, i32** %36, align 8
  %38 = load i32, i32* %7, align 4
  %39 = sext i32 %38 to i64
  %40 = getelementptr inbounds i32, i32* %37, i64 %39
  %41 = load i32, i32* %40, align 4
  %42 = sitofp i32 %41 to double
  %43 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %44 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %43, i32 0, i32 14
  %45 = load i32*, i32** %44, align 8
  %46 = load i32, i32* %7, align 4
  %47 = sext i32 %46 to i64
  %48 = getelementptr inbounds i32, i32* %45, i64 %47
  %49 = load i32, i32* %48, align 4
  %50 = sitofp i32 %49 to double
  %51 = fdiv double %42, %50
  %52 = fsub double 1.000000e+00, %51
  %53 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %54 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %53, i32 0, i32 13
  %55 = load %struct.cass_list_entry_t**, %struct.cass_list_entry_t*** %54, align 8
  %56 = load i32, i32* %7, align 4
  %57 = sext i32 %56 to i64
  %58 = getelementptr inbounds %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %55, i64 %57
  %59 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %58, align 8
  %60 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %59, i64 0
  %61 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %60, i32 0, i32 1
  %62 = load float, float* %61, align 4
  %63 = fpext float %62 to double
  %64 = fmul double %52, %63
  %65 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %66 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %65, i32 0, i32 16
  %67 = load float*, float** %66, align 8
  %68 = load i32, i32* %7, align 4
  %69 = sext i32 %68 to i64
  %70 = getelementptr inbounds float, float* %67, i64 %69
  %71 = load float, float* %70, align 4
  %72 = fpext float %71 to double
  %73 = fadd double %72, %64
  %74 = fptrunc double %73 to float
  store float %74, float* %70, align 4
  br label %82

; <label>:75:                                     ; preds = %19
  %76 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  %77 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %76, i32 0, i32 16
  %78 = load float*, float** %77, align 8
  %79 = load i32, i32* %7, align 4
  %80 = sext i32 %79 to i64
  %81 = getelementptr inbounds float, float* %78, i64 %80
  store float 0.000000e+00, float* %81, align 4
  br label %82

; <label>:82:                                     ; preds = %75, %34
  br label %83

; <label>:83:                                     ; preds = %82
  %84 = load i32, i32* %6, align 4
  %85 = add nsw i32 %84, 1
  store i32 %85, i32* %6, align 4
  br label %15

; <label>:86:                                     ; preds = %15
  %87 = load %struct.LSH_query_t*, %struct.LSH_query_t** %3, align 8
  call void @LSH_query_merge(%struct.LSH_query_t* %87)
  ret void
}

declare i32 @bitmap_clear(%struct.bitmap_t*) #2

declare i32 @bitmap_contain(%struct.bitmap_t*, i32) #2

declare i32 @bitmap_insert(%struct.bitmap_t*, i32) #2

; Function Attrs: noinline nounwind optnone uwtable
define internal float @dist_L2_float(i32, float*, float*) #0 {
  %4 = alloca i32, align 4
  %5 = alloca float*, align 8
  %6 = alloca float*, align 8
  %7 = alloca float, align 4
  %8 = alloca float, align 4
  %9 = alloca i32, align 4
  store i32 %0, i32* %4, align 4
  store float* %1, float** %5, align 8
  store float* %2, float** %6, align 8
  store float 0.000000e+00, float* %7, align 4
  store i32 0, i32* %9, align 4
  br label %10

; <label>:10:                                     ; preds = %32, %3
  %11 = load i32, i32* %9, align 4
  %12 = load i32, i32* %4, align 4
  %13 = icmp ult i32 %11, %12
  br i1 %13, label %14, label %35

; <label>:14:                                     ; preds = %10
  %15 = load float*, float** %5, align 8
  %16 = load i32, i32* %9, align 4
  %17 = zext i32 %16 to i64
  %18 = getelementptr inbounds float, float* %15, i64 %17
  %19 = load float, float* %18, align 4
  %20 = load float*, float** %6, align 8
  %21 = load i32, i32* %9, align 4
  %22 = zext i32 %21 to i64
  %23 = getelementptr inbounds float, float* %20, i64 %22
  %24 = load float, float* %23, align 4
  %25 = fsub float %19, %24
  store float %25, float* %8, align 4
  %26 = load float, float* %8, align 4
  %27 = load float, float* %8, align 4
  %28 = fmul float %27, %26
  store float %28, float* %8, align 4
  %29 = load float, float* %8, align 4
  %30 = load float, float* %7, align 4
  %31 = fadd float %30, %29
  store float %31, float* %7, align 4
  br label %32

; <label>:32:                                     ; preds = %14
  %33 = load i32, i32* %9, align 4
  %34 = add i32 %33, 1
  store i32 %34, i32* %9, align 4
  br label %10

; <label>:35:                                     ; preds = %10
  %36 = load float, float* %7, align 4
  %37 = fpext float %36 to double
  %38 = call double @sqrt(double %37) #6
  %39 = fptrunc double %38 to float
  ret float %39
}

declare void @ptb_qsort(%struct.ptb_vec_t*, i32) #2

declare i32 @map_perturb_vector(%struct.ptb_vec_t*, %struct.ptb_vec_t*, %struct.ptb_vec_t*, i32, i32) #2

; Function Attrs: noinline nounwind optnone uwtable
define internal void @LSH_query_local(%struct.LSH_query_t*) #0 {
  %2 = alloca %struct.LSH_query_t*, align 8
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  %5 = alloca double, align 8
  %6 = alloca double, align 8
  %7 = alloca double, align 8
  %8 = alloca double, align 8
  %9 = alloca double, align 8
  %10 = alloca double, align 8
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  store %struct.LSH_query_t* %0, %struct.LSH_query_t** %2, align 8
  %13 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %14 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %13, i32 0, i32 2
  %15 = load i32, i32* %14, align 8
  store i32 %15, i32* %12, align 4
  store double 0.000000e+00, double* %6, align 8
  store double 0.000000e+00, double* %5, align 8
  store double 0.000000e+00, double* %4, align 8
  store double 0.000000e+00, double* %3, align 8
  store i32 0, i32* %11, align 4
  br label %16

; <label>:16:                                     ; preds = %70, %1
  %17 = load i32, i32* %11, align 4
  %18 = load i32, i32* %12, align 4
  %19 = sub nsw i32 %18, 1
  %20 = icmp slt i32 %17, %19
  br i1 %20, label %21, label %73

; <label>:21:                                     ; preds = %16
  %22 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %23 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %22, i32 0, i32 12
  %24 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %23, align 8
  %25 = load i32, i32* %12, align 4
  %26 = load i32, i32* %11, align 4
  %27 = sub nsw i32 %25, %26
  %28 = sub nsw i32 %27, 2
  %29 = sext i32 %28 to i64
  %30 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %24, i64 %29
  %31 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %30, i32 0, i32 1
  %32 = load float, float* %31, align 4
  %33 = fpext float %32 to double
  %34 = fcmp oge double %33, 0x7FF0000000000000
  br i1 %34, label %35, label %36

; <label>:35:                                     ; preds = %21
  br label %73

; <label>:36:                                     ; preds = %21
  %37 = load i32, i32* %11, align 4
  %38 = add nsw i32 %37, 1
  %39 = sitofp i32 %38 to double
  %40 = call double @log(double %39) #6
  store double %40, double* %8, align 8
  %41 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %42 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %41, i32 0, i32 12
  %43 = load %struct.cass_list_entry_t*, %struct.cass_list_entry_t** %42, align 8
  %44 = load i32, i32* %12, align 4
  %45 = load i32, i32* %11, align 4
  %46 = sub nsw i32 %44, %45
  %47 = sub nsw i32 %46, 2
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %43, i64 %48
  %50 = getelementptr inbounds %struct.cass_list_entry_t, %struct.cass_list_entry_t* %49, i32 0, i32 1
  %51 = load float, float* %50, align 4
  %52 = fpext float %51 to double
  %53 = call double @log(double %52) #6
  store double %53, double* %7, align 8
  %54 = load double, double* %8, align 8
  %55 = load double, double* %3, align 8
  %56 = fadd double %55, %54
  store double %56, double* %3, align 8
  %57 = load double, double* %7, align 8
  %58 = load double, double* %4, align 8
  %59 = fadd double %58, %57
  store double %59, double* %4, align 8
  %60 = load double, double* %8, align 8
  %61 = load double, double* %8, align 8
  %62 = fmul double %60, %61
  %63 = load double, double* %5, align 8
  %64 = fadd double %63, %62
  store double %64, double* %5, align 8
  %65 = load double, double* %8, align 8
  %66 = load double, double* %7, align 8
  %67 = fmul double %65, %66
  %68 = load double, double* %6, align 8
  %69 = fadd double %68, %67
  store double %69, double* %6, align 8
  br label %70

; <label>:70:                                     ; preds = %36
  %71 = load i32, i32* %11, align 4
  %72 = add nsw i32 %71, 1
  store i32 %72, i32* %11, align 4
  br label %16

; <label>:73:                                     ; preds = %35, %16
  %74 = load i32, i32* %11, align 4
  %75 = load double, double* %5, align 8
  %76 = load double, double* %6, align 8
  %77 = load double, double* %3, align 8
  %78 = load double, double* %4, align 8
  call void @least_squares(double* %9, double* %10, i32 %74, double %75, double %76, double %77, double %78)
  %79 = load double, double* %9, align 8
  %80 = call double @exp(double %79) #6
  store double %80, double* %9, align 8
  %81 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %82 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %81, i32 0, i32 0
  %83 = load %struct.LSH_t*, %struct.LSH_t** %82, align 8
  %84 = getelementptr inbounds %struct.LSH_t, %struct.LSH_t* %83, i32 0, i32 12
  %85 = load %struct.LSH_est_t*, %struct.LSH_est_t** %84, align 8
  %86 = load double, double* %9, align 8
  %87 = load double, double* %10, align 8
  %88 = load i32, i32* %12, align 4
  %89 = sub nsw i32 %88, 1
  %90 = call double @LSH_est(%struct.LSH_est_t* %85, double %86, double %87, i32 %89)
  %91 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %92 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %91, i32 0, i32 20
  %93 = load float, float* %92, align 4
  %94 = fpext float %93 to double
  %95 = fmul double %90, %94
  %96 = fptrunc double %95 to float
  %97 = load %struct.LSH_query_t*, %struct.LSH_query_t** %2, align 8
  %98 = getelementptr inbounds %struct.LSH_query_t, %struct.LSH_query_t* %97, i32 0, i32 18
  store float %96, float* %98, align 4
  ret void
}

; Function Attrs: nounwind
declare double @sqrt(double) #4

; Function Attrs: nounwind
declare double @log(double) #4

; Function Attrs: noinline nounwind optnone uwtable
define internal void @least_squares(double*, double*, i32, double, double, double, double) #0 {
  %8 = alloca double*, align 8
  %9 = alloca double*, align 8
  %10 = alloca i32, align 4
  %11 = alloca double, align 8
  %12 = alloca double, align 8
  %13 = alloca double, align 8
  %14 = alloca double, align 8
  store double* %0, double** %8, align 8
  store double* %1, double** %9, align 8
  store i32 %2, i32* %10, align 4
  store double %3, double* %11, align 8
  store double %4, double* %12, align 8
  store double %5, double* %13, align 8
  store double %6, double* %14, align 8
  %15 = load i32, i32* %10, align 4
  %16 = sitofp i32 %15 to double
  %17 = load double, double* %12, align 8
  %18 = fmul double %16, %17
  %19 = load double, double* %13, align 8
  %20 = load double, double* %14, align 8
  %21 = fmul double %19, %20
  %22 = fsub double %18, %21
  %23 = load i32, i32* %10, align 4
  %24 = sitofp i32 %23 to double
  %25 = load double, double* %11, align 8
  %26 = fmul double %24, %25
  %27 = load double, double* %13, align 8
  %28 = load double, double* %13, align 8
  %29 = fmul double %27, %28
  %30 = fsub double %26, %29
  %31 = fdiv double %22, %30
  %32 = load double*, double** %9, align 8
  store double %31, double* %32, align 8
  %33 = load double, double* %14, align 8
  %34 = load double*, double** %9, align 8
  %35 = load double, double* %34, align 8
  %36 = load double, double* %13, align 8
  %37 = fmul double %35, %36
  %38 = fsub double %33, %37
  %39 = load i32, i32* %10, align 4
  %40 = sitofp i32 %39 to double
  %41 = fdiv double %38, %40
  %42 = load double*, double** %8, align 8
  store double %41, double* %42, align 8
  ret void
}

; Function Attrs: nounwind
declare double @exp(double) #4

declare double @LSH_est(%struct.LSH_est_t*, double, double, i32) #2

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { noreturn nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind readnone speculatable }
attributes #6 = { nounwind }
attributes #7 = { noreturn nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.0 (tags/RELEASE_600/final)"}
