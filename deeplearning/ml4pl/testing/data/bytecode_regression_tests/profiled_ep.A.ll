; ModuleID = './ep.A.bc'
source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, i8*, i8*, i8*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type { %struct._IO_marker*, %struct._IO_FILE*, i32 }
%struct.timeval = type { i64, i64 }
%struct.timezone = type { i32, i32 }

@main.dum = private unnamed_addr constant [3 x double] [double 1.000000e+00, double 1.000000e+00, double 1.000000e+00], align 16
@.str = private unnamed_addr constant [11 x i8] c"timer.flag\00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"r\00", align 1
@.str.2 = private unnamed_addr constant [8 x i8] c"%15.0lf\00", align 1
@str = private unnamed_addr constant [57 x i8] c"\0A\0A NAS Parallel Benchmarks (NPB3.3-SER-C) - EP Benchmark\00"
@.str.4 = private unnamed_addr constant [44 x i8] c"\0A Number of random numbers generated: %15s\0A\00", align 1
@x = internal global [131072 x double] zeroinitializer, align 16, !dbg !0
@q = internal unnamed_addr global [10 x double] zeroinitializer, align 16, !dbg !13
@str.26 = private unnamed_addr constant [24 x i8] c"\0AEP Benchmark Results:\0A\00"
@.str.6 = private unnamed_addr constant [19 x i8] c"CPU Time =%10.4lf\0A\00", align 1
@.str.7 = private unnamed_addr constant [11 x i8] c"N = 2^%5d\0A\00", align 1
@.str.8 = private unnamed_addr constant [30 x i8] c"No. Gaussian Pairs = %15.0lf\0A\00", align 1
@.str.9 = private unnamed_addr constant [26 x i8] c"Sums = %25.15lE %25.15lE\0A\00", align 1
@str.27 = private unnamed_addr constant [9 x i8] c"Counts: \00"
@.str.11 = private unnamed_addr constant [12 x i8] c"%3d%15.0lf\0A\00", align 1
@.str.12 = private unnamed_addr constant [3 x i8] c"EP\00", align 1
@.str.13 = private unnamed_addr constant [25 x i8] c"Random numbers generated\00", align 1
@.str.14 = private unnamed_addr constant [6 x i8] c"3.3.1\00", align 1
@.str.15 = private unnamed_addr constant [12 x i8] c"12 Feb 2020\00", align 1
@.str.16 = private unnamed_addr constant [45 x i8] c"/users/mcopik/projects/mlcode/wrappers/clang\00", align 1
@.str.17 = private unnamed_addr constant [6 x i8] c"$(CC)\00", align 1
@.str.18 = private unnamed_addr constant [4 x i8] c"-lm\00", align 1
@.str.19 = private unnamed_addr constant [12 x i8] c"-I../common\00", align 1
@.str.20 = private unnamed_addr constant [47 x i8] c"-g -Wall -O3 -mcmodel=medium -fprofile-inst...\00", align 1
@.str.21 = private unnamed_addr constant [47 x i8] c"-O3 -mcmodel=medium -fprofile-instr-use=${P...\00", align 1
@.str.22 = private unnamed_addr constant [7 x i8] c"randdp\00", align 1
@.str.23 = private unnamed_addr constant [34 x i8] c"\0ATotal time:     %9.3lf (%6.2lf)\0A\00", align 1
@.str.24 = private unnamed_addr constant [33 x i8] c"Gaussian pairs: %9.3lf (%6.2lf)\0A\00", align 1
@.str.25 = private unnamed_addr constant [33 x i8] c"Random numbers: %9.3lf (%6.2lf)\0A\00", align 1
@.str.3 = private unnamed_addr constant [28 x i8] c"\0A\0A %s Benchmark Completed.\0A\00", align 1
@.str.1.4 = private unnamed_addr constant [37 x i8] c" Class           =             %12c\0A\00", align 1
@.str.2.5 = private unnamed_addr constant [8 x i8] c"%15.0lf\00", align 1
@.str.3.6 = private unnamed_addr constant [34 x i8] c" Size            =          %15s\0A\00", align 1
@.str.4.7 = private unnamed_addr constant [37 x i8] c" Size            =             %12d\0A\00", align 1
@.str.5 = private unnamed_addr constant [42 x i8] c" Size            =           %4dx%4dx%4d\0A\00", align 1
@.str.6.8 = private unnamed_addr constant [37 x i8] c" Iterations      =             %12d\0A\00", align 1
@.str.7.9 = private unnamed_addr constant [40 x i8] c" Time in seconds =             %12.2lf\0A\00", align 1
@.str.8.10 = private unnamed_addr constant [37 x i8] c" Mop/s total     =          %15.2lf\0A\00", align 1
@.str.9.11 = private unnamed_addr constant [25 x i8] c" Operation type  = %24s\0A\00", align 1
@.str.10 = private unnamed_addr constant [37 x i8] c" Verification    =             %12s\0A\00", align 1
@.str.11.12 = private unnamed_addr constant [11 x i8] c"SUCCESSFUL\00", align 1
@.str.12.13 = private unnamed_addr constant [13 x i8] c"UNSUCCESSFUL\00", align 1
@.str.13.14 = private unnamed_addr constant [37 x i8] c" Version         =             %12s\0A\00", align 1
@.str.14.15 = private unnamed_addr constant [37 x i8] c" Compile date    =             %12s\0A\00", align 1
@.str.15.16 = private unnamed_addr constant [42 x i8] c"\0A Compile options:\0A    CC           = %s\0A\00", align 1
@.str.16.17 = private unnamed_addr constant [23 x i8] c"    CLINK        = %s\0A\00", align 1
@.str.17.18 = private unnamed_addr constant [23 x i8] c"    C_LIB        = %s\0A\00", align 1
@.str.18.19 = private unnamed_addr constant [23 x i8] c"    C_INC        = %s\0A\00", align 1
@.str.19.20 = private unnamed_addr constant [23 x i8] c"    CFLAGS       = %s\0A\00", align 1
@.str.20.21 = private unnamed_addr constant [23 x i8] c"    CLINKFLAGS   = %s\0A\00", align 1
@.str.21.22 = private unnamed_addr constant [23 x i8] c"    RAND         = %s\0A\00", align 1
@str.23 = private unnamed_addr constant [194 x i8] c"\0A--------------------------------------\0A Please send all errors/feedbacks to:\0A Center for Manycore Programming\0A cmp@aces.snu.ac.kr\0A http://aces.snu.ac.kr\0A--------------------------------------\0A\00"
@elapsed = internal unnamed_addr global [64 x double] zeroinitializer, align 16, !dbg !22
@start = internal unnamed_addr global [64 x double] zeroinitializer, align 16, !dbg !28
@wtime_.sec = internal unnamed_addr global i32 -1, align 4, !dbg !34

; Function Attrs: nounwind uwtable
define i32 @main() local_unnamed_addr #0 !dbg !95 !prof !196 {
  %1 = alloca double, align 8
  %2 = alloca double, align 8
  %3 = alloca [3 x double], align 16
  %4 = alloca [16 x i8], align 16
  %5 = bitcast double* %1 to i8*, !dbg !197
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %5) #4, !dbg !197
  %6 = bitcast double* %2 to i8*, !dbg !197
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %6) #4, !dbg !197
  %7 = bitcast [3 x double]* %3 to i8*, !dbg !198
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %7) #4, !dbg !198
  call void @llvm.dbg.declare(metadata [3 x double]* %3, metadata !127, metadata !DIExpression()), !dbg !199
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull %7, i8* bitcast ([3 x double]* @main.dum to i8*), i64 24, i32 16, i1 false), !dbg !199
  %8 = getelementptr inbounds [16 x i8], [16 x i8]* %4, i64 0, i64 0, !dbg !200
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %8) #4, !dbg !200
  call void @llvm.dbg.declare(metadata [16 x i8]* %4, metadata !131, metadata !DIExpression()), !dbg !201
  %9 = tail call %struct._IO_FILE* @fopen(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0)), !dbg !202
  call void @llvm.dbg.value(metadata %struct._IO_FILE* %9, metadata !136, metadata !DIExpression()), !dbg !204
  %10 = icmp eq %struct._IO_FILE* %9, null, !dbg !205
  br i1 %10, label %13, label %11, !dbg !206, !prof !207

; <label>:11:                                     ; preds = %0
  call void @llvm.dbg.value(metadata i32 1, metadata !126, metadata !DIExpression()), !dbg !208
  %12 = tail call i32 @fclose(%struct._IO_FILE* nonnull %9), !dbg !209
  br label %13

; <label>:13:                                     ; preds = %0, %11
  %14 = phi i32 [ 1, %11 ], [ 0, %0 ]
  call void @llvm.dbg.value(metadata i32 %14, metadata !126, metadata !DIExpression()), !dbg !208
  %15 = call i32 (i8*, i8*, ...) @sprintf(i8* nonnull %8, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.2, i64 0, i64 0), double 0x41C0000000000000) #4, !dbg !211
  call void @llvm.dbg.value(metadata i32 14, metadata !123, metadata !DIExpression()), !dbg !212
  %16 = getelementptr inbounds [16 x i8], [16 x i8]* %4, i64 0, i64 14, !dbg !213
  %17 = load i8, i8* %16, align 2, !dbg !213, !tbaa !215
  %18 = icmp eq i8 %17, 46, !dbg !218
  %19 = select i1 %18, i64 14, i64 15, !dbg !219, !prof !220
  %20 = getelementptr inbounds [16 x i8], [16 x i8]* %4, i64 0, i64 %19, !dbg !221
  store i8 0, i8* %20, align 1, !dbg !222, !tbaa !215
  %21 = tail call i32 @puts(i8* getelementptr inbounds ([57 x i8], [57 x i8]* @str, i64 0, i64 0)), !dbg !223
  %22 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([44 x i8], [44 x i8]* @.str.4, i64 0, i64 0), i8* nonnull %8), !dbg !224
  call void @llvm.dbg.value(metadata i32 0, metadata !124, metadata !DIExpression()), !dbg !225
  call void @llvm.dbg.value(metadata i32 4096, metadata !115, metadata !DIExpression()), !dbg !226
  %23 = getelementptr inbounds [3 x double], [3 x double]* %3, i64 0, i64 0, !dbg !227
  %24 = getelementptr inbounds [3 x double], [3 x double]* %3, i64 0, i64 1, !dbg !228
  %25 = getelementptr inbounds [3 x double], [3 x double]* %3, i64 0, i64 2, !dbg !229
  call void @vranlc(i32 0, double* nonnull %23, double 1.000000e+00, double* nonnull %25) #4, !dbg !230
  %26 = load double, double* %25, align 16, !dbg !231, !tbaa !232
  %27 = call double @randlc(double* nonnull %24, double %26) #4, !dbg !234
  store double %27, double* %23, align 16, !dbg !235, !tbaa !232
  call void @llvm.dbg.value(metadata i32 0, metadata !116, metadata !DIExpression()), !dbg !236
  br label %28, !dbg !237

; <label>:28:                                     ; preds = %28, %13
  %29 = phi i64 [ 0, %13 ], [ %49, %28 ], !dbg !239
  %30 = getelementptr inbounds [131072 x double], [131072 x double]* @x, i64 0, i64 %29, !dbg !241
  %31 = bitcast double* %30 to <2 x double>*, !dbg !243
  store <2 x double> <double 0xD47D42AEA2879F2E, double 0xD47D42AEA2879F2E>, <2 x double>* %31, align 16, !dbg !243, !tbaa !232
  %32 = getelementptr double, double* %30, i64 2, !dbg !243
  %33 = bitcast double* %32 to <2 x double>*, !dbg !243
  store <2 x double> <double 0xD47D42AEA2879F2E, double 0xD47D42AEA2879F2E>, <2 x double>* %33, align 16, !dbg !243, !tbaa !232
  %34 = or i64 %29, 4, !dbg !239
  %35 = getelementptr inbounds [131072 x double], [131072 x double]* @x, i64 0, i64 %34, !dbg !241
  %36 = bitcast double* %35 to <2 x double>*, !dbg !243
  store <2 x double> <double 0xD47D42AEA2879F2E, double 0xD47D42AEA2879F2E>, <2 x double>* %36, align 16, !dbg !243, !tbaa !232
  %37 = getelementptr double, double* %35, i64 2, !dbg !243
  %38 = bitcast double* %37 to <2 x double>*, !dbg !243
  store <2 x double> <double 0xD47D42AEA2879F2E, double 0xD47D42AEA2879F2E>, <2 x double>* %38, align 16, !dbg !243, !tbaa !232
  %39 = or i64 %29, 8, !dbg !239
  %40 = getelementptr inbounds [131072 x double], [131072 x double]* @x, i64 0, i64 %39, !dbg !241
  %41 = bitcast double* %40 to <2 x double>*, !dbg !243
  store <2 x double> <double 0xD47D42AEA2879F2E, double 0xD47D42AEA2879F2E>, <2 x double>* %41, align 16, !dbg !243, !tbaa !232
  %42 = getelementptr double, double* %40, i64 2, !dbg !243
  %43 = bitcast double* %42 to <2 x double>*, !dbg !243
  store <2 x double> <double 0xD47D42AEA2879F2E, double 0xD47D42AEA2879F2E>, <2 x double>* %43, align 16, !dbg !243, !tbaa !232
  %44 = or i64 %29, 12, !dbg !239
  %45 = getelementptr inbounds [131072 x double], [131072 x double]* @x, i64 0, i64 %44, !dbg !241
  %46 = bitcast double* %45 to <2 x double>*, !dbg !243
  store <2 x double> <double 0xD47D42AEA2879F2E, double 0xD47D42AEA2879F2E>, <2 x double>* %46, align 16, !dbg !243, !tbaa !232
  %47 = getelementptr double, double* %45, i64 2, !dbg !243
  %48 = bitcast double* %47 to <2 x double>*, !dbg !243
  store <2 x double> <double 0xD47D42AEA2879F2E, double 0xD47D42AEA2879F2E>, <2 x double>* %48, align 16, !dbg !243, !tbaa !232
  %49 = add nuw nsw i64 %29, 16, !dbg !239
  %50 = icmp eq i64 %49, 131072, !dbg !239
  br i1 %50, label %51, label %28, !dbg !239, !llvm.loop !244

; <label>:51:                                     ; preds = %28
  call void @timer_clear(i32 0) #4, !dbg !247
  call void @timer_clear(i32 1) #4, !dbg !248
  call void @timer_clear(i32 2) #4, !dbg !249
  call void @timer_start(i32 0) #4, !dbg !250
  call void @llvm.dbg.value(metadata double 0x41D2309CE5400000, metadata !99, metadata !DIExpression()), !dbg !251
  store double 0x41D2309CE5400000, double* %1, align 8, !dbg !252, !tbaa !232
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  call void @vranlc(i32 0, double* nonnull %1, double 0x41D2309CE5400000, double* getelementptr inbounds ([131072 x double], [131072 x double]* @x, i64 0, i64 0)) #4, !dbg !253
  call void @llvm.dbg.value(metadata double 0x41D2309CE5400000, metadata !99, metadata !DIExpression()), !dbg !251
  store double 0x41D2309CE5400000, double* %1, align 8, !dbg !254, !tbaa !232
  call void @llvm.dbg.value(metadata i32 0, metadata !116, metadata !DIExpression()), !dbg !236
  call void @llvm.dbg.value(metadata double 0x41D2309CE5400000, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata i32 0, metadata !116, metadata !DIExpression()), !dbg !236
  %52 = call double @randlc(double* nonnull %1, double 0x41D2309CE5400000) #4, !dbg !255
  call void @llvm.dbg.value(metadata double %52, metadata !100, metadata !DIExpression()), !dbg !259
  store double %52, double* %2, align 8, !dbg !260, !tbaa !232
  call void @llvm.dbg.value(metadata i32 1, metadata !116, metadata !DIExpression()), !dbg !236
  %53 = load double, double* %1, align 8, !tbaa !232
  call void @llvm.dbg.value(metadata double %53, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata i32 1, metadata !116, metadata !DIExpression()), !dbg !236
  %54 = call double @randlc(double* nonnull %1, double %53) #4, !dbg !255
  call void @llvm.dbg.value(metadata double %54, metadata !100, metadata !DIExpression()), !dbg !259
  store double %54, double* %2, align 8, !dbg !260, !tbaa !232
  call void @llvm.dbg.value(metadata i32 2, metadata !116, metadata !DIExpression()), !dbg !236
  %55 = load double, double* %1, align 8, !tbaa !232
  call void @llvm.dbg.value(metadata double %55, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata i32 2, metadata !116, metadata !DIExpression()), !dbg !236
  %56 = call double @randlc(double* nonnull %1, double %55) #4, !dbg !255
  call void @llvm.dbg.value(metadata double %56, metadata !100, metadata !DIExpression()), !dbg !259
  store double %56, double* %2, align 8, !dbg !260, !tbaa !232
  call void @llvm.dbg.value(metadata i32 3, metadata !116, metadata !DIExpression()), !dbg !236
  %57 = load double, double* %1, align 8, !tbaa !232
  call void @llvm.dbg.value(metadata double %57, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata i32 3, metadata !116, metadata !DIExpression()), !dbg !236
  %58 = call double @randlc(double* nonnull %1, double %57) #4, !dbg !255
  call void @llvm.dbg.value(metadata double %58, metadata !100, metadata !DIExpression()), !dbg !259
  store double %58, double* %2, align 8, !dbg !260, !tbaa !232
  call void @llvm.dbg.value(metadata i32 4, metadata !116, metadata !DIExpression()), !dbg !236
  %59 = load double, double* %1, align 8, !tbaa !232
  call void @llvm.dbg.value(metadata double %59, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata i32 4, metadata !116, metadata !DIExpression()), !dbg !236
  %60 = call double @randlc(double* nonnull %1, double %59) #4, !dbg !255
  call void @llvm.dbg.value(metadata double %60, metadata !100, metadata !DIExpression()), !dbg !259
  store double %60, double* %2, align 8, !dbg !260, !tbaa !232
  call void @llvm.dbg.value(metadata i32 5, metadata !116, metadata !DIExpression()), !dbg !236
  %61 = load double, double* %1, align 8, !tbaa !232
  call void @llvm.dbg.value(metadata double %61, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata i32 5, metadata !116, metadata !DIExpression()), !dbg !236
  %62 = call double @randlc(double* nonnull %1, double %61) #4, !dbg !255
  call void @llvm.dbg.value(metadata double %62, metadata !100, metadata !DIExpression()), !dbg !259
  store double %62, double* %2, align 8, !dbg !260, !tbaa !232
  call void @llvm.dbg.value(metadata i32 6, metadata !116, metadata !DIExpression()), !dbg !236
  %63 = load double, double* %1, align 8, !tbaa !232
  call void @llvm.dbg.value(metadata double %63, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata i32 6, metadata !116, metadata !DIExpression()), !dbg !236
  %64 = call double @randlc(double* nonnull %1, double %63) #4, !dbg !255
  call void @llvm.dbg.value(metadata double %64, metadata !100, metadata !DIExpression()), !dbg !259
  store double %64, double* %2, align 8, !dbg !260, !tbaa !232
  call void @llvm.dbg.value(metadata i32 7, metadata !116, metadata !DIExpression()), !dbg !236
  %65 = load double, double* %1, align 8, !tbaa !232
  call void @llvm.dbg.value(metadata double %65, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata i32 7, metadata !116, metadata !DIExpression()), !dbg !236
  %66 = call double @randlc(double* nonnull %1, double %65) #4, !dbg !255
  call void @llvm.dbg.value(metadata double %66, metadata !100, metadata !DIExpression()), !dbg !259
  store double %66, double* %2, align 8, !dbg !260, !tbaa !232
  call void @llvm.dbg.value(metadata i32 8, metadata !116, metadata !DIExpression()), !dbg !236
  %67 = load double, double* %1, align 8, !tbaa !232
  call void @llvm.dbg.value(metadata double %67, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata i32 8, metadata !116, metadata !DIExpression()), !dbg !236
  %68 = call double @randlc(double* nonnull %1, double %67) #4, !dbg !255
  call void @llvm.dbg.value(metadata double %68, metadata !100, metadata !DIExpression()), !dbg !259
  store double %68, double* %2, align 8, !dbg !260, !tbaa !232
  call void @llvm.dbg.value(metadata i32 9, metadata !116, metadata !DIExpression()), !dbg !236
  %69 = load double, double* %1, align 8, !tbaa !232
  call void @llvm.dbg.value(metadata double %69, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata i32 9, metadata !116, metadata !DIExpression()), !dbg !236
  %70 = call double @randlc(double* nonnull %1, double %69) #4, !dbg !255
  call void @llvm.dbg.value(metadata double %70, metadata !100, metadata !DIExpression()), !dbg !259
  store double %70, double* %2, align 8, !dbg !260, !tbaa !232
  call void @llvm.dbg.value(metadata i32 10, metadata !116, metadata !DIExpression()), !dbg !236
  %71 = load double, double* %1, align 8, !tbaa !232
  call void @llvm.dbg.value(metadata double %71, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata i32 10, metadata !116, metadata !DIExpression()), !dbg !236
  %72 = call double @randlc(double* nonnull %1, double %71) #4, !dbg !255
  call void @llvm.dbg.value(metadata double %72, metadata !100, metadata !DIExpression()), !dbg !259
  store double %72, double* %2, align 8, !dbg !260, !tbaa !232
  call void @llvm.dbg.value(metadata i32 11, metadata !116, metadata !DIExpression()), !dbg !236
  %73 = load double, double* %1, align 8, !tbaa !232
  call void @llvm.dbg.value(metadata double %73, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata i32 11, metadata !116, metadata !DIExpression()), !dbg !236
  %74 = call double @randlc(double* nonnull %1, double %73) #4, !dbg !255
  call void @llvm.dbg.value(metadata double %74, metadata !100, metadata !DIExpression()), !dbg !259
  store double %74, double* %2, align 8, !dbg !260, !tbaa !232
  call void @llvm.dbg.value(metadata i32 12, metadata !116, metadata !DIExpression()), !dbg !236
  %75 = load double, double* %1, align 8, !tbaa !232
  call void @llvm.dbg.value(metadata double %75, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata i32 12, metadata !116, metadata !DIExpression()), !dbg !236
  %76 = call double @randlc(double* nonnull %1, double %75) #4, !dbg !255
  call void @llvm.dbg.value(metadata double %76, metadata !100, metadata !DIExpression()), !dbg !259
  store double %76, double* %2, align 8, !dbg !260, !tbaa !232
  call void @llvm.dbg.value(metadata i32 13, metadata !116, metadata !DIExpression()), !dbg !236
  %77 = load double, double* %1, align 8, !tbaa !232
  call void @llvm.dbg.value(metadata double %77, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata i32 13, metadata !116, metadata !DIExpression()), !dbg !236
  %78 = call double @randlc(double* nonnull %1, double %77) #4, !dbg !255
  call void @llvm.dbg.value(metadata double %78, metadata !100, metadata !DIExpression()), !dbg !259
  store double %78, double* %2, align 8, !dbg !260, !tbaa !232
  call void @llvm.dbg.value(metadata i32 14, metadata !116, metadata !DIExpression()), !dbg !236
  %79 = load double, double* %1, align 8, !tbaa !232
  call void @llvm.dbg.value(metadata double %79, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata i32 14, metadata !116, metadata !DIExpression()), !dbg !236
  %80 = call double @randlc(double* nonnull %1, double %79) #4, !dbg !255
  call void @llvm.dbg.value(metadata double %80, metadata !100, metadata !DIExpression()), !dbg !259
  store double %80, double* %2, align 8, !dbg !260, !tbaa !232
  call void @llvm.dbg.value(metadata i32 15, metadata !116, metadata !DIExpression()), !dbg !236
  %81 = load double, double* %1, align 8, !tbaa !232
  call void @llvm.dbg.value(metadata double %81, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata i32 15, metadata !116, metadata !DIExpression()), !dbg !236
  %82 = call double @randlc(double* nonnull %1, double %81) #4, !dbg !255
  call void @llvm.dbg.value(metadata double %82, metadata !100, metadata !DIExpression()), !dbg !259
  store double %82, double* %2, align 8, !dbg !260, !tbaa !232
  call void @llvm.dbg.value(metadata i32 16, metadata !116, metadata !DIExpression()), !dbg !236
  %83 = load double, double* %1, align 8, !tbaa !232
  call void @llvm.dbg.value(metadata double %83, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  call void @llvm.dbg.value(metadata i32 16, metadata !116, metadata !DIExpression()), !dbg !236
  %84 = call double @randlc(double* nonnull %1, double %83) #4, !dbg !255
  call void @llvm.dbg.value(metadata double %84, metadata !100, metadata !DIExpression()), !dbg !259
  store double %84, double* %2, align 8, !dbg !260, !tbaa !232
  call void @llvm.dbg.value(metadata i32 17, metadata !116, metadata !DIExpression()), !dbg !236
  %85 = bitcast double* %1 to i64*
  %86 = load i64, i64* %85, align 8, !tbaa !232
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression(DW_OP_deref)), !dbg !251
  call void @llvm.dbg.value(metadata double undef, metadata !108, metadata !DIExpression()), !dbg !261
  call void @llvm.dbg.value(metadata double 0x41B033C4D7000000, metadata !109, metadata !DIExpression()), !dbg !262
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !105, metadata !DIExpression()), !dbg !264
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !106, metadata !DIExpression()), !dbg !265
  call void @llvm.dbg.value(metadata i32 0, metadata !116, metadata !DIExpression()), !dbg !236
  call void @llvm.memset.p0i8.i64(i8* bitcast ([10 x double]* @q to i8*), i8 0, i64 80, i32 16, i1 false), !dbg !266
  call void @llvm.dbg.value(metadata i32 -1, metadata !122, metadata !DIExpression()), !dbg !270
  call void @llvm.dbg.value(metadata i32 1, metadata !120, metadata !DIExpression()), !dbg !271
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !105, metadata !DIExpression()), !dbg !264
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !106, metadata !DIExpression()), !dbg !265
  %87 = icmp eq i32 %14, 0
  %88 = bitcast double* %2 to i64*
  br i1 %87, label %90, label %89, !dbg !272, !prof !274

; <label>:89:                                     ; preds = %51
  br label %91, !dbg !275

; <label>:90:                                     ; preds = %51
  br label %154, !dbg !275

; <label>:91:                                     ; preds = %89, %113
  %92 = phi i32 [ %114, %113 ], [ 1, %89 ]
  %93 = phi <2 x double> [ %151, %113 ], [ zeroinitializer, %89 ]
  call void @llvm.dbg.value(metadata double undef, metadata !106, metadata !DIExpression()), !dbg !265
  call void @llvm.dbg.value(metadata i32 %92, metadata !120, metadata !DIExpression()), !dbg !271
  call void @llvm.dbg.value(metadata double undef, metadata !105, metadata !DIExpression()), !dbg !264
  %94 = add nsw i32 %92, -1, !dbg !275
  call void @llvm.dbg.value(metadata i32 %94, metadata !118, metadata !DIExpression()), !dbg !278
  call void @llvm.dbg.value(metadata double 0x41B033C4D7000000, metadata !99, metadata !DIExpression()), !dbg !251
  store double 0x41B033C4D7000000, double* %1, align 8, !dbg !279, !tbaa !232
  call void @llvm.dbg.value(metadata double undef, metadata !100, metadata !DIExpression()), !dbg !259
  store i64 %86, i64* %88, align 8, !dbg !280, !tbaa !232
  call void @llvm.dbg.value(metadata i32 1, metadata !116, metadata !DIExpression()), !dbg !236
  call void @llvm.dbg.value(metadata i32 %94, metadata !118, metadata !DIExpression()), !dbg !278
  br label %95, !dbg !281

; <label>:95:                                     ; preds = %107, %91
  %96 = phi i32 [ %94, %91 ], [ %98, %107 ]
  %97 = phi i32 [ 1, %91 ], [ %110, %107 ]
  call void @llvm.dbg.value(metadata i32 %97, metadata !116, metadata !DIExpression()), !dbg !236
  call void @llvm.dbg.value(metadata i32 %96, metadata !118, metadata !DIExpression()), !dbg !278
  %98 = sdiv i32 %96, 2, !dbg !283
  call void @llvm.dbg.value(metadata i32 %98, metadata !117, metadata !DIExpression()), !dbg !286
  %99 = shl nsw i32 %98, 1, !dbg !287
  %100 = icmp eq i32 %99, %96, !dbg !289
  br i1 %100, label %104, label %101, !dbg !290, !prof !291

; <label>:101:                                    ; preds = %95
  %102 = load double, double* %2, align 8, !dbg !292, !tbaa !232
  call void @llvm.dbg.value(metadata double %102, metadata !100, metadata !DIExpression()), !dbg !259
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  %103 = call double @randlc(double* nonnull %1, double %102) #4, !dbg !293
  call void @llvm.dbg.value(metadata double %103, metadata !101, metadata !DIExpression()), !dbg !294
  br label %104, !dbg !295

; <label>:104:                                    ; preds = %101, %95
  %105 = add i32 %96, 1, !dbg !296
  %106 = icmp ult i32 %105, 3, !dbg !296
  br i1 %106, label %112, label %107, !dbg !298, !prof !299

; <label>:107:                                    ; preds = %104
  %108 = load double, double* %2, align 8, !dbg !300, !tbaa !232
  call void @llvm.dbg.value(metadata double %108, metadata !100, metadata !DIExpression()), !dbg !259
  call void @llvm.dbg.value(metadata double* %2, metadata !100, metadata !DIExpression()), !dbg !259
  %109 = call double @randlc(double* nonnull %2, double %108) #4, !dbg !301
  call void @llvm.dbg.value(metadata double %109, metadata !101, metadata !DIExpression()), !dbg !294
  %110 = add nuw nsw i32 %97, 1, !dbg !302
  call void @llvm.dbg.value(metadata i32 %98, metadata !118, metadata !DIExpression()), !dbg !278
  call void @llvm.dbg.value(metadata i32 %110, metadata !116, metadata !DIExpression()), !dbg !236
  %111 = icmp ult i32 %110, 101, !dbg !303
  br i1 %111, label %95, label %112, !dbg !281, !prof !304, !llvm.loop !305

; <label>:112:                                    ; preds = %104, %107
  call void @timer_start(i32 2) #4, !dbg !307
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  call void @vranlc(i32 131072, double* nonnull %1, double 0x41D2309CE5400000, double* getelementptr inbounds ([131072 x double], [131072 x double]* @x, i64 0, i64 0)) #4, !dbg !309
  call void @timer_stop(i32 2) #4, !dbg !310
  call void @timer_start(i32 1) #4, !dbg !312
  call void @llvm.dbg.value(metadata i32 0, metadata !116, metadata !DIExpression()), !dbg !236
  call void @llvm.dbg.value(metadata double undef, metadata !105, metadata !DIExpression()), !dbg !264
  call void @llvm.dbg.value(metadata double undef, metadata !106, metadata !DIExpression()), !dbg !265
  br label %116, !dbg !314

; <label>:113:                                    ; preds = %150
  call void @timer_stop(i32 1) #4, !dbg !316
  %114 = add nuw nsw i32 %92, 1, !dbg !318
  call void @llvm.dbg.value(metadata double undef, metadata !105, metadata !DIExpression()), !dbg !264
  call void @llvm.dbg.value(metadata i32 %114, metadata !120, metadata !DIExpression()), !dbg !271
  call void @llvm.dbg.value(metadata double undef, metadata !106, metadata !DIExpression()), !dbg !265
  %115 = icmp eq i32 %114, 4097, !dbg !319
  br i1 %115, label %217, label %91, !dbg !272, !prof !320, !llvm.loop !321

; <label>:116:                                    ; preds = %150, %112
  %117 = phi i64 [ %152, %150 ], [ 0, %112 ]
  %118 = phi <2 x double> [ %151, %150 ], [ %93, %112 ]
  call void @llvm.dbg.value(metadata i64 %117, metadata !116, metadata !DIExpression()), !dbg !236
  call void @llvm.dbg.value(metadata double undef, metadata !106, metadata !DIExpression()), !dbg !265
  call void @llvm.dbg.value(metadata double undef, metadata !105, metadata !DIExpression()), !dbg !264
  %119 = shl nuw nsw i64 %117, 1, !dbg !323
  %120 = getelementptr inbounds [131072 x double], [131072 x double]* @x, i64 0, i64 %119, !dbg !326
  call void @llvm.dbg.value(metadata double undef, metadata !103, metadata !DIExpression()), !dbg !327
  %121 = bitcast double* %120 to <2 x double>*, !dbg !326
  %122 = load <2 x double>, <2 x double>* %121, align 16, !dbg !326, !tbaa !232
  %123 = fmul <2 x double> %122, <double 2.000000e+00, double 2.000000e+00>, !dbg !328
  %124 = fadd <2 x double> %123, <double -1.000000e+00, double -1.000000e+00>, !dbg !329
  call void @llvm.dbg.value(metadata double undef, metadata !104, metadata !DIExpression()), !dbg !330
  %125 = fmul <2 x double> %124, %124, !dbg !331
  %126 = extractelement <2 x double> %125, i32 0, !dbg !332
  %127 = extractelement <2 x double> %125, i32 1, !dbg !332
  %128 = fadd double %126, %127, !dbg !332
  call void @llvm.dbg.value(metadata double %128, metadata !99, metadata !DIExpression()), !dbg !251
  store double %128, double* %1, align 8, !dbg !333, !tbaa !232
  %129 = fcmp ugt double %128, 1.000000e+00, !dbg !334
  br i1 %129, label %150, label %130, !dbg !336, !prof !337

; <label>:130:                                    ; preds = %116
  %131 = call double @log(double %128) #4, !dbg !338
  %132 = fmul double %131, -2.000000e+00, !dbg !340
  %133 = load double, double* %1, align 8, !dbg !341, !tbaa !232
  call void @llvm.dbg.value(metadata double %133, metadata !99, metadata !DIExpression()), !dbg !251
  %134 = fdiv double %132, %133, !dbg !342
  %135 = call double @sqrt(double %134) #4, !dbg !343
  call void @llvm.dbg.value(metadata double %135, metadata !100, metadata !DIExpression()), !dbg !259
  store double %135, double* %2, align 8, !dbg !344, !tbaa !232
  call void @llvm.dbg.value(metadata double undef, metadata !101, metadata !DIExpression()), !dbg !294
  %136 = insertelement <2 x double> undef, double %135, i32 0, !dbg !345
  %137 = shufflevector <2 x double> %136, <2 x double> undef, <2 x i32> zeroinitializer, !dbg !345
  %138 = fmul <2 x double> %124, %137, !dbg !345
  call void @llvm.dbg.value(metadata double undef, metadata !102, metadata !DIExpression()), !dbg !346
  %139 = call <2 x double> @llvm.fabs.v2f64(<2 x double> %138), !dbg !347
  %140 = extractelement <2 x double> %139, i32 0, !dbg !347
  %141 = extractelement <2 x double> %139, i32 1, !dbg !347
  %142 = fcmp ogt double %140, %141, !dbg !347
  %143 = select i1 %142, double %140, double %141, !dbg !347, !prof !348
  %144 = fptosi double %143 to i32, !dbg !347
  call void @llvm.dbg.value(metadata i32 %144, metadata !119, metadata !DIExpression()), !dbg !349
  %145 = sext i32 %144 to i64, !dbg !350
  %146 = getelementptr inbounds [10 x double], [10 x double]* @q, i64 0, i64 %145, !dbg !350
  %147 = load double, double* %146, align 8, !dbg !350, !tbaa !232
  %148 = fadd double %147, 1.000000e+00, !dbg !351
  store double %148, double* %146, align 8, !dbg !352, !tbaa !232
  call void @llvm.dbg.value(metadata double undef, metadata !105, metadata !DIExpression()), !dbg !264
  %149 = fadd <2 x double> %118, %138, !dbg !353
  call void @llvm.dbg.value(metadata double undef, metadata !106, metadata !DIExpression()), !dbg !265
  br label %150, !dbg !354

; <label>:150:                                    ; preds = %130, %116
  %151 = phi <2 x double> [ %149, %130 ], [ %118, %116 ]
  %152 = add nuw nsw i64 %117, 1, !dbg !355
  call void @llvm.dbg.value(metadata double undef, metadata !105, metadata !DIExpression()), !dbg !264
  call void @llvm.dbg.value(metadata double undef, metadata !106, metadata !DIExpression()), !dbg !265
  %153 = icmp eq i64 %152, 65536, !dbg !356
  br i1 %153, label %113, label %116, !dbg !314, !prof !357, !llvm.loop !358

; <label>:154:                                    ; preds = %90, %214
  %155 = phi i32 [ %215, %214 ], [ 1, %90 ]
  %156 = phi <2 x double> [ %211, %214 ], [ zeroinitializer, %90 ]
  call void @llvm.dbg.value(metadata double undef, metadata !106, metadata !DIExpression()), !dbg !265
  call void @llvm.dbg.value(metadata i32 %155, metadata !120, metadata !DIExpression()), !dbg !271
  call void @llvm.dbg.value(metadata double undef, metadata !105, metadata !DIExpression()), !dbg !264
  %157 = add nsw i32 %155, -1, !dbg !275
  call void @llvm.dbg.value(metadata i32 %157, metadata !118, metadata !DIExpression()), !dbg !278
  call void @llvm.dbg.value(metadata double 0x41B033C4D7000000, metadata !99, metadata !DIExpression()), !dbg !251
  store double 0x41B033C4D7000000, double* %1, align 8, !dbg !279, !tbaa !232
  call void @llvm.dbg.value(metadata double undef, metadata !100, metadata !DIExpression()), !dbg !259
  store i64 %86, i64* %88, align 8, !dbg !280, !tbaa !232
  call void @llvm.dbg.value(metadata i32 1, metadata !116, metadata !DIExpression()), !dbg !236
  call void @llvm.dbg.value(metadata i32 %157, metadata !118, metadata !DIExpression()), !dbg !278
  br label %158, !dbg !281

; <label>:158:                                    ; preds = %154, %170
  %159 = phi i32 [ %157, %154 ], [ %161, %170 ]
  %160 = phi i32 [ 1, %154 ], [ %173, %170 ]
  call void @llvm.dbg.value(metadata i32 %160, metadata !116, metadata !DIExpression()), !dbg !236
  call void @llvm.dbg.value(metadata i32 %159, metadata !118, metadata !DIExpression()), !dbg !278
  %161 = sdiv i32 %159, 2, !dbg !283
  call void @llvm.dbg.value(metadata i32 %161, metadata !117, metadata !DIExpression()), !dbg !286
  %162 = shl nsw i32 %161, 1, !dbg !287
  %163 = icmp eq i32 %162, %159, !dbg !289
  br i1 %163, label %167, label %164, !dbg !290, !prof !291

; <label>:164:                                    ; preds = %158
  %165 = load double, double* %2, align 8, !dbg !292, !tbaa !232
  call void @llvm.dbg.value(metadata double %165, metadata !100, metadata !DIExpression()), !dbg !259
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  %166 = call double @randlc(double* nonnull %1, double %165) #4, !dbg !293
  call void @llvm.dbg.value(metadata double %166, metadata !101, metadata !DIExpression()), !dbg !294
  br label %167, !dbg !295

; <label>:167:                                    ; preds = %158, %164
  %168 = add i32 %159, 1, !dbg !296
  %169 = icmp ult i32 %168, 3, !dbg !296
  br i1 %169, label %175, label %170, !dbg !298, !prof !299

; <label>:170:                                    ; preds = %167
  %171 = load double, double* %2, align 8, !dbg !300, !tbaa !232
  call void @llvm.dbg.value(metadata double %171, metadata !100, metadata !DIExpression()), !dbg !259
  call void @llvm.dbg.value(metadata double* %2, metadata !100, metadata !DIExpression()), !dbg !259
  %172 = call double @randlc(double* nonnull %2, double %171) #4, !dbg !301
  call void @llvm.dbg.value(metadata double %172, metadata !101, metadata !DIExpression()), !dbg !294
  %173 = add nuw nsw i32 %160, 1, !dbg !302
  call void @llvm.dbg.value(metadata i32 %161, metadata !118, metadata !DIExpression()), !dbg !278
  call void @llvm.dbg.value(metadata i32 %173, metadata !116, metadata !DIExpression()), !dbg !236
  %174 = icmp ult i32 %173, 101, !dbg !303
  br i1 %174, label %158, label %175, !dbg !281, !prof !304, !llvm.loop !305

; <label>:175:                                    ; preds = %170, %167
  call void @llvm.dbg.value(metadata double* %1, metadata !99, metadata !DIExpression()), !dbg !251
  call void @vranlc(i32 131072, double* nonnull %1, double 0x41D2309CE5400000, double* getelementptr inbounds ([131072 x double], [131072 x double]* @x, i64 0, i64 0)) #4, !dbg !309
  call void @llvm.dbg.value(metadata i32 0, metadata !116, metadata !DIExpression()), !dbg !236
  call void @llvm.dbg.value(metadata double undef, metadata !105, metadata !DIExpression()), !dbg !264
  call void @llvm.dbg.value(metadata double undef, metadata !106, metadata !DIExpression()), !dbg !265
  br label %176, !dbg !314

; <label>:176:                                    ; preds = %210, %175
  %177 = phi i64 [ 0, %175 ], [ %212, %210 ]
  %178 = phi <2 x double> [ %156, %175 ], [ %211, %210 ]
  call void @llvm.dbg.value(metadata i64 %177, metadata !116, metadata !DIExpression()), !dbg !236
  call void @llvm.dbg.value(metadata double undef, metadata !106, metadata !DIExpression()), !dbg !265
  call void @llvm.dbg.value(metadata double undef, metadata !105, metadata !DIExpression()), !dbg !264
  %179 = shl nuw nsw i64 %177, 1, !dbg !323
  %180 = getelementptr inbounds [131072 x double], [131072 x double]* @x, i64 0, i64 %179, !dbg !326
  call void @llvm.dbg.value(metadata double undef, metadata !103, metadata !DIExpression()), !dbg !327
  %181 = bitcast double* %180 to <2 x double>*, !dbg !326
  %182 = load <2 x double>, <2 x double>* %181, align 16, !dbg !326, !tbaa !232
  %183 = fmul <2 x double> %182, <double 2.000000e+00, double 2.000000e+00>, !dbg !328
  %184 = fadd <2 x double> %183, <double -1.000000e+00, double -1.000000e+00>, !dbg !329
  call void @llvm.dbg.value(metadata double undef, metadata !104, metadata !DIExpression()), !dbg !330
  %185 = fmul <2 x double> %184, %184, !dbg !331
  %186 = extractelement <2 x double> %185, i32 0, !dbg !332
  %187 = extractelement <2 x double> %185, i32 1, !dbg !332
  %188 = fadd double %186, %187, !dbg !332
  call void @llvm.dbg.value(metadata double %188, metadata !99, metadata !DIExpression()), !dbg !251
  store double %188, double* %1, align 8, !dbg !333, !tbaa !232
  %189 = fcmp ugt double %188, 1.000000e+00, !dbg !334
  br i1 %189, label %210, label %190, !dbg !336, !prof !337

; <label>:190:                                    ; preds = %176
  %191 = call double @log(double %188) #4, !dbg !338
  %192 = fmul double %191, -2.000000e+00, !dbg !340
  %193 = load double, double* %1, align 8, !dbg !341, !tbaa !232
  call void @llvm.dbg.value(metadata double %193, metadata !99, metadata !DIExpression()), !dbg !251
  %194 = fdiv double %192, %193, !dbg !342
  %195 = call double @sqrt(double %194) #4, !dbg !343
  call void @llvm.dbg.value(metadata double %195, metadata !100, metadata !DIExpression()), !dbg !259
  store double %195, double* %2, align 8, !dbg !344, !tbaa !232
  call void @llvm.dbg.value(metadata double undef, metadata !101, metadata !DIExpression()), !dbg !294
  %196 = insertelement <2 x double> undef, double %195, i32 0, !dbg !345
  %197 = shufflevector <2 x double> %196, <2 x double> undef, <2 x i32> zeroinitializer, !dbg !345
  %198 = fmul <2 x double> %184, %197, !dbg !345
  call void @llvm.dbg.value(metadata double undef, metadata !102, metadata !DIExpression()), !dbg !346
  %199 = call <2 x double> @llvm.fabs.v2f64(<2 x double> %198), !dbg !347
  %200 = extractelement <2 x double> %199, i32 0, !dbg !347
  %201 = extractelement <2 x double> %199, i32 1, !dbg !347
  %202 = fcmp ogt double %200, %201, !dbg !347
  %203 = select i1 %202, double %200, double %201, !dbg !347, !prof !348
  %204 = fptosi double %203 to i32, !dbg !347
  call void @llvm.dbg.value(metadata i32 %204, metadata !119, metadata !DIExpression()), !dbg !349
  %205 = sext i32 %204 to i64, !dbg !350
  %206 = getelementptr inbounds [10 x double], [10 x double]* @q, i64 0, i64 %205, !dbg !350
  %207 = load double, double* %206, align 8, !dbg !350, !tbaa !232
  %208 = fadd double %207, 1.000000e+00, !dbg !351
  store double %208, double* %206, align 8, !dbg !352, !tbaa !232
  call void @llvm.dbg.value(metadata double undef, metadata !105, metadata !DIExpression()), !dbg !264
  %209 = fadd <2 x double> %178, %198, !dbg !353
  call void @llvm.dbg.value(metadata double undef, metadata !106, metadata !DIExpression()), !dbg !265
  br label %210, !dbg !354

; <label>:210:                                    ; preds = %176, %190
  %211 = phi <2 x double> [ %209, %190 ], [ %178, %176 ]
  %212 = add nuw nsw i64 %177, 1, !dbg !355
  call void @llvm.dbg.value(metadata double undef, metadata !105, metadata !DIExpression()), !dbg !264
  call void @llvm.dbg.value(metadata double undef, metadata !106, metadata !DIExpression()), !dbg !265
  %213 = icmp eq i64 %212, 65536, !dbg !356
  br i1 %213, label %214, label %176, !dbg !314, !prof !357, !llvm.loop !358

; <label>:214:                                    ; preds = %210
  %215 = add nuw nsw i32 %155, 1, !dbg !318
  call void @llvm.dbg.value(metadata double undef, metadata !105, metadata !DIExpression()), !dbg !264
  call void @llvm.dbg.value(metadata i32 %215, metadata !120, metadata !DIExpression()), !dbg !271
  call void @llvm.dbg.value(metadata double undef, metadata !106, metadata !DIExpression()), !dbg !265
  %216 = icmp eq i32 %215, 4097, !dbg !319
  br i1 %216, label %217, label %154, !dbg !272, !prof !320, !llvm.loop !321

; <label>:217:                                    ; preds = %113, %214
  %218 = phi <2 x double> [ %211, %214 ], [ %151, %113 ]
  call void @llvm.dbg.value(metadata i32 0, metadata !116, metadata !DIExpression()), !dbg !236
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata double 0.000000e+00, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata i64 0, metadata !116, metadata !DIExpression()), !dbg !236
  %219 = load double, double* getelementptr inbounds ([10 x double], [10 x double]* @q, i64 0, i64 0), align 16, !dbg !360, !tbaa !232
  %220 = fadd double %219, 0.000000e+00, !dbg !364
  call void @llvm.dbg.value(metadata double %220, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata double %220, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata i64 1, metadata !116, metadata !DIExpression()), !dbg !236
  %221 = load double, double* getelementptr inbounds ([10 x double], [10 x double]* @q, i64 0, i64 1), align 8, !dbg !360, !tbaa !232
  %222 = fadd double %220, %221, !dbg !364
  call void @llvm.dbg.value(metadata double %222, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata double %222, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata i64 2, metadata !116, metadata !DIExpression()), !dbg !236
  %223 = load double, double* getelementptr inbounds ([10 x double], [10 x double]* @q, i64 0, i64 2), align 16, !dbg !360, !tbaa !232
  %224 = fadd double %222, %223, !dbg !364
  call void @llvm.dbg.value(metadata double %224, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata double %224, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata i64 3, metadata !116, metadata !DIExpression()), !dbg !236
  %225 = load double, double* getelementptr inbounds ([10 x double], [10 x double]* @q, i64 0, i64 3), align 8, !dbg !360, !tbaa !232
  %226 = fadd double %224, %225, !dbg !364
  call void @llvm.dbg.value(metadata double %226, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata double %226, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata i64 4, metadata !116, metadata !DIExpression()), !dbg !236
  %227 = load double, double* getelementptr inbounds ([10 x double], [10 x double]* @q, i64 0, i64 4), align 16, !dbg !360, !tbaa !232
  %228 = fadd double %226, %227, !dbg !364
  call void @llvm.dbg.value(metadata double %228, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata double %228, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata i64 5, metadata !116, metadata !DIExpression()), !dbg !236
  %229 = load double, double* getelementptr inbounds ([10 x double], [10 x double]* @q, i64 0, i64 5), align 8, !dbg !360, !tbaa !232
  %230 = fadd double %228, %229, !dbg !364
  call void @llvm.dbg.value(metadata double %230, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata double %230, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata i64 6, metadata !116, metadata !DIExpression()), !dbg !236
  %231 = load double, double* getelementptr inbounds ([10 x double], [10 x double]* @q, i64 0, i64 6), align 16, !dbg !360, !tbaa !232
  %232 = fadd double %230, %231, !dbg !364
  call void @llvm.dbg.value(metadata double %232, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata double %232, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata i64 7, metadata !116, metadata !DIExpression()), !dbg !236
  %233 = load double, double* getelementptr inbounds ([10 x double], [10 x double]* @q, i64 0, i64 7), align 8, !dbg !360, !tbaa !232
  %234 = fadd double %232, %233, !dbg !364
  call void @llvm.dbg.value(metadata double %234, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata double %234, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata i64 8, metadata !116, metadata !DIExpression()), !dbg !236
  %235 = load double, double* getelementptr inbounds ([10 x double], [10 x double]* @q, i64 0, i64 8), align 16, !dbg !360, !tbaa !232
  %236 = fadd double %234, %235, !dbg !364
  call void @llvm.dbg.value(metadata double %236, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata double %236, metadata !110, metadata !DIExpression()), !dbg !263
  call void @llvm.dbg.value(metadata i64 9, metadata !116, metadata !DIExpression()), !dbg !236
  %237 = load double, double* getelementptr inbounds ([10 x double], [10 x double]* @q, i64 0, i64 9), align 8, !dbg !360, !tbaa !232
  %238 = fadd double %236, %237, !dbg !364
  call void @llvm.dbg.value(metadata double %238, metadata !110, metadata !DIExpression()), !dbg !263
  call void @timer_stop(i32 0) #4, !dbg !365
  %239 = call double @timer_read(i32 0) #4, !dbg !366
  call void @llvm.dbg.value(metadata double %239, metadata !107, metadata !DIExpression()), !dbg !367
  call void @llvm.dbg.value(metadata i32 0, metadata !121, metadata !DIExpression()), !dbg !368
  call void @llvm.dbg.value(metadata i32 1, metadata !124, metadata !DIExpression()), !dbg !225
  call void @llvm.dbg.value(metadata double 0xC0B0C7E00ADACEF8, metadata !111, metadata !DIExpression()), !dbg !369
  call void @llvm.dbg.value(metadata double 0xC0CEDFA9B1BE31DC, metadata !112, metadata !DIExpression()), !dbg !370
  %240 = extractelement <2 x double> %218, i32 0, !dbg !371
  call void @llvm.dbg.value(metadata double undef, metadata !113, metadata !DIExpression()), !dbg !374
  %241 = extractelement <2 x double> %218, i32 1, !dbg !375
  %242 = fadd <2 x double> %218, <double 0x40B0C7E00ADACEF8, double 0x40CEDFA9B1BE31DC>, !dbg !375
  %243 = fdiv <2 x double> %242, <double 0xC0B0C7E00ADACEF8, double 0xC0CEDFA9B1BE31DC>, !dbg !376
  %244 = shufflevector <2 x double> %243, <2 x double> undef, <2 x i32> <i32 1, i32 0>, !dbg !376
  %245 = call <2 x double> @llvm.fabs.v2f64(<2 x double> %244), !dbg !377
  call void @llvm.dbg.value(metadata double undef, metadata !114, metadata !DIExpression()), !dbg !378
  %246 = fcmp ole <2 x double> %245, <double 1.000000e-08, double 1.000000e-08>, !dbg !379
  %247 = extractelement <2 x i1> %246, i32 0, !dbg !380
  %248 = extractelement <2 x i1> %246, i32 1, !dbg !380
  %249 = and i1 %247, %248, !dbg !380
  call void @llvm.dbg.value(metadata i32 %277, metadata !124, metadata !DIExpression()), !dbg !225
  call void @llvm.dbg.value(metadata i32 %277, metadata !124, metadata !DIExpression()), !dbg !225
  %250 = fdiv double 0x41C0000000000000, %239, !dbg !381
  call void @llvm.dbg.value(metadata double %278, metadata !98, metadata !DIExpression()), !dbg !382
  %251 = call i32 @puts(i8* getelementptr inbounds ([24 x i8], [24 x i8]* @str.26, i64 0, i64 0)), !dbg !383
  %252 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str.6, i64 0, i64 0), double %239), !dbg !384
  %253 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.7, i64 0, i64 0), i32 28), !dbg !385
  %254 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([30 x i8], [30 x i8]* @.str.8, i64 0, i64 0), double %238), !dbg !386
  %255 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str.9, i64 0, i64 0), double %240, double %241), !dbg !387
  %256 = call i32 @puts(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @str.27, i64 0, i64 0)), !dbg !388
  call void @llvm.dbg.value(metadata i32 0, metadata !116, metadata !DIExpression()), !dbg !236
  call void @llvm.dbg.value(metadata i64 0, metadata !116, metadata !DIExpression()), !dbg !236
  %257 = load double, double* getelementptr inbounds ([10 x double], [10 x double]* @q, i64 0, i64 0), align 16, !dbg !389, !tbaa !232
  %258 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.11, i64 0, i64 0), i32 0, double %257), !dbg !393
  call void @llvm.dbg.value(metadata i64 1, metadata !116, metadata !DIExpression()), !dbg !236
  %259 = load double, double* getelementptr inbounds ([10 x double], [10 x double]* @q, i64 0, i64 1), align 8, !dbg !389, !tbaa !232
  %260 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.11, i64 0, i64 0), i32 1, double %259), !dbg !393
  call void @llvm.dbg.value(metadata i64 2, metadata !116, metadata !DIExpression()), !dbg !236
  %261 = load double, double* getelementptr inbounds ([10 x double], [10 x double]* @q, i64 0, i64 2), align 16, !dbg !389, !tbaa !232
  %262 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.11, i64 0, i64 0), i32 2, double %261), !dbg !393
  call void @llvm.dbg.value(metadata i64 3, metadata !116, metadata !DIExpression()), !dbg !236
  %263 = load double, double* getelementptr inbounds ([10 x double], [10 x double]* @q, i64 0, i64 3), align 8, !dbg !389, !tbaa !232
  %264 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.11, i64 0, i64 0), i32 3, double %263), !dbg !393
  call void @llvm.dbg.value(metadata i64 4, metadata !116, metadata !DIExpression()), !dbg !236
  %265 = load double, double* getelementptr inbounds ([10 x double], [10 x double]* @q, i64 0, i64 4), align 16, !dbg !389, !tbaa !232
  %266 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.11, i64 0, i64 0), i32 4, double %265), !dbg !393
  call void @llvm.dbg.value(metadata i64 5, metadata !116, metadata !DIExpression()), !dbg !236
  %267 = load double, double* getelementptr inbounds ([10 x double], [10 x double]* @q, i64 0, i64 5), align 8, !dbg !389, !tbaa !232
  %268 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.11, i64 0, i64 0), i32 5, double %267), !dbg !393
  call void @llvm.dbg.value(metadata i64 6, metadata !116, metadata !DIExpression()), !dbg !236
  %269 = load double, double* getelementptr inbounds ([10 x double], [10 x double]* @q, i64 0, i64 6), align 16, !dbg !389, !tbaa !232
  %270 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.11, i64 0, i64 0), i32 6, double %269), !dbg !393
  call void @llvm.dbg.value(metadata i64 7, metadata !116, metadata !DIExpression()), !dbg !236
  %271 = load double, double* getelementptr inbounds ([10 x double], [10 x double]* @q, i64 0, i64 7), align 8, !dbg !389, !tbaa !232
  %272 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.11, i64 0, i64 0), i32 7, double %271), !dbg !393
  call void @llvm.dbg.value(metadata i64 8, metadata !116, metadata !DIExpression()), !dbg !236
  %273 = load double, double* getelementptr inbounds ([10 x double], [10 x double]* @q, i64 0, i64 8), align 16, !dbg !389, !tbaa !232
  %274 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.11, i64 0, i64 0), i32 8, double %273), !dbg !393
  call void @llvm.dbg.value(metadata i64 9, metadata !116, metadata !DIExpression()), !dbg !236
  %275 = load double, double* getelementptr inbounds ([10 x double], [10 x double]* @q, i64 0, i64 9), align 8, !dbg !389, !tbaa !232
  %276 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.11, i64 0, i64 0), i32 9, double %275), !dbg !393
  %277 = zext i1 %249 to i32, !dbg !380
  %278 = fdiv double %250, 1.000000e+06, !dbg !394
  call void @print_results(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.12, i64 0, i64 0), i8 signext 65, i32 29, i32 0, i32 0, i32 0, double %239, double %278, i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.13, i64 0, i64 0), i32 %277, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.14, i64 0, i64 0), i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.15, i64 0, i64 0), i8* getelementptr inbounds ([45 x i8], [45 x i8]* @.str.16, i64 0, i64 0), i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str.17, i64 0, i64 0), i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.18, i64 0, i64 0), i8* getelementptr inbounds ([12 x i8], [12 x i8]* @.str.19, i64 0, i64 0), i8* getelementptr inbounds ([47 x i8], [47 x i8]* @.str.20, i64 0, i64 0), i8* getelementptr inbounds ([47 x i8], [47 x i8]* @.str.21, i64 0, i64 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.22, i64 0, i64 0)) #4, !dbg !395
  br i1 %87, label %294, label %279, !dbg !396, !prof !207

; <label>:279:                                    ; preds = %217
  %280 = fcmp ugt double %239, 0.000000e+00, !dbg !397
  call void @llvm.dbg.value(metadata double 1.000000e+00, metadata !107, metadata !DIExpression()), !dbg !367
  %281 = select i1 %280, double %239, double 1.000000e+00, !dbg !401
  call void @llvm.dbg.value(metadata double %281, metadata !107, metadata !DIExpression()), !dbg !367
  %282 = call double @timer_read(i32 0) #4, !dbg !402
  call void @llvm.dbg.value(metadata double %282, metadata !109, metadata !DIExpression()), !dbg !262
  %283 = fmul double %282, 1.000000e+02, !dbg !403
  %284 = fdiv double %283, %281, !dbg !404
  %285 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([34 x i8], [34 x i8]* @.str.23, i64 0, i64 0), double %282, double %284), !dbg !405
  %286 = call double @timer_read(i32 1) #4, !dbg !406
  call void @llvm.dbg.value(metadata double %286, metadata !109, metadata !DIExpression()), !dbg !262
  %287 = fmul double %286, 1.000000e+02, !dbg !407
  %288 = fdiv double %287, %281, !dbg !408
  %289 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([33 x i8], [33 x i8]* @.str.24, i64 0, i64 0), double %286, double %288), !dbg !409
  %290 = call double @timer_read(i32 2) #4, !dbg !410
  call void @llvm.dbg.value(metadata double %290, metadata !109, metadata !DIExpression()), !dbg !262
  %291 = fmul double %290, 1.000000e+02, !dbg !411
  %292 = fdiv double %291, %281, !dbg !412
  %293 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([33 x i8], [33 x i8]* @.str.25, i64 0, i64 0), double %290, double %292), !dbg !413
  br label %294, !dbg !414

; <label>:294:                                    ; preds = %217, %279
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %8) #4, !dbg !415
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %7) #4, !dbg !415
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %6) #4, !dbg !415
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %5) #4, !dbg !415
  ret i32 0, !dbg !416
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #1

; Function Attrs: nounwind
declare noalias %struct._IO_FILE* @fopen(i8* nocapture readonly, i8* nocapture readonly) local_unnamed_addr #3

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

; Function Attrs: nounwind
declare i32 @fclose(%struct._IO_FILE* nocapture) local_unnamed_addr #3

; Function Attrs: nounwind
declare i32 @sprintf(i8* nocapture, i8* nocapture readonly, ...) local_unnamed_addr #3

; Function Attrs: nounwind
declare i32 @puts(i8* nocapture readonly) local_unnamed_addr #4

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...) local_unnamed_addr #3

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #1

; Function Attrs: nounwind
declare double @log(double) local_unnamed_addr #3

; Function Attrs: nounwind
declare double @sqrt(double) local_unnamed_addr #3

; Function Attrs: nounwind readnone speculatable
declare <2 x double> @llvm.fabs.v2f64(<2 x double>) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define void @print_results(i8*, i8 signext, i32, i32, i32, i32, double, double, i8*, i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*) local_unnamed_addr #0 !dbg !417 !prof !196 {
  %20 = alloca [16 x i8], align 16
  call void @llvm.dbg.value(metadata i8* %0, metadata !423, metadata !DIExpression()), !dbg !444
  call void @llvm.dbg.value(metadata i8 %1, metadata !424, metadata !DIExpression()), !dbg !445
  call void @llvm.dbg.value(metadata i32 %2, metadata !425, metadata !DIExpression()), !dbg !446
  call void @llvm.dbg.value(metadata i32 %3, metadata !426, metadata !DIExpression()), !dbg !447
  call void @llvm.dbg.value(metadata i32 %4, metadata !427, metadata !DIExpression()), !dbg !448
  call void @llvm.dbg.value(metadata i32 %5, metadata !428, metadata !DIExpression()), !dbg !449
  call void @llvm.dbg.value(metadata double %6, metadata !429, metadata !DIExpression()), !dbg !450
  call void @llvm.dbg.value(metadata double %7, metadata !430, metadata !DIExpression()), !dbg !451
  call void @llvm.dbg.value(metadata i8* %8, metadata !431, metadata !DIExpression()), !dbg !452
  call void @llvm.dbg.value(metadata i32 %9, metadata !432, metadata !DIExpression()), !dbg !453
  call void @llvm.dbg.value(metadata i8* %10, metadata !433, metadata !DIExpression()), !dbg !454
  call void @llvm.dbg.value(metadata i8* %11, metadata !434, metadata !DIExpression()), !dbg !455
  call void @llvm.dbg.value(metadata i8* %12, metadata !435, metadata !DIExpression()), !dbg !456
  call void @llvm.dbg.value(metadata i8* %13, metadata !436, metadata !DIExpression()), !dbg !457
  call void @llvm.dbg.value(metadata i8* %14, metadata !437, metadata !DIExpression()), !dbg !458
  call void @llvm.dbg.value(metadata i8* %15, metadata !438, metadata !DIExpression()), !dbg !459
  call void @llvm.dbg.value(metadata i8* %16, metadata !439, metadata !DIExpression()), !dbg !460
  call void @llvm.dbg.value(metadata i8* %17, metadata !440, metadata !DIExpression()), !dbg !461
  call void @llvm.dbg.value(metadata i8* %18, metadata !441, metadata !DIExpression()), !dbg !462
  %21 = getelementptr inbounds [16 x i8], [16 x i8]* %20, i64 0, i64 0, !dbg !463
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %21) #4, !dbg !463
  call void @llvm.dbg.declare(metadata [16 x i8]* %20, metadata !442, metadata !DIExpression()), !dbg !464
  %22 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([28 x i8], [28 x i8]* @.str.3, i64 0, i64 0), i8* %0), !dbg !465
  %23 = sext i8 %1 to i32, !dbg !466
  %24 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.1.4, i64 0, i64 0), i32 %23), !dbg !467
  %25 = or i32 %4, %3, !dbg !468
  %26 = icmp eq i32 %25, 0, !dbg !468
  br i1 %26, label %27, label %47, !dbg !468, !prof !470

; <label>:27:                                     ; preds = %19
  %28 = load i8, i8* %0, align 1, !dbg !471, !tbaa !215
  %29 = icmp eq i8 %28, 69, !dbg !474
  br i1 %29, label %30, label %45, !dbg !475, !prof !207

; <label>:30:                                     ; preds = %27
  %31 = getelementptr inbounds i8, i8* %0, i64 1, !dbg !476
  %32 = load i8, i8* %31, align 1, !dbg !476, !tbaa !215
  %33 = icmp eq i8 %32, 80, !dbg !477
  br i1 %33, label %34, label %45, !dbg !478, !prof !207

; <label>:34:                                     ; preds = %30
  %35 = tail call double @ldexp(double 1.000000e+00, i32 %2) #4, !dbg !479
  %36 = call i32 (i8*, i8*, ...) @sprintf(i8* nonnull %21, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.2.5, i64 0, i64 0), double %35) #4, !dbg !481
  call void @llvm.dbg.value(metadata i32 14, metadata !443, metadata !DIExpression()), !dbg !482
  %37 = getelementptr inbounds [16 x i8], [16 x i8]* %20, i64 0, i64 14, !dbg !483
  %38 = load i8, i8* %37, align 2, !dbg !483, !tbaa !215
  %39 = icmp eq i8 %38, 46, !dbg !485
  br i1 %39, label %40, label %41, !dbg !486, !prof !220

; <label>:40:                                     ; preds = %34
  store i8 32, i8* %37, align 2, !dbg !487, !tbaa !215
  call void @llvm.dbg.value(metadata i32 13, metadata !443, metadata !DIExpression()), !dbg !482
  br label %41, !dbg !489

; <label>:41:                                     ; preds = %34, %40
  %42 = phi i64 [ 14, %40 ], [ 15, %34 ]
  %43 = getelementptr inbounds [16 x i8], [16 x i8]* %20, i64 0, i64 %42, !dbg !490
  store i8 0, i8* %43, align 1, !dbg !491, !tbaa !215
  %44 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([34 x i8], [34 x i8]* @.str.3.6, i64 0, i64 0), i8* nonnull %21), !dbg !492
  br label %49, !dbg !493

; <label>:45:                                     ; preds = %30, %27
  %46 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.4.7, i64 0, i64 0), i32 %2), !dbg !494
  br label %49

; <label>:47:                                     ; preds = %19
  %48 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.5, i64 0, i64 0), i32 %2, i32 %3, i32 %4), !dbg !496
  br label %49

; <label>:49:                                     ; preds = %41, %45, %47
  %50 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.6.8, i64 0, i64 0), i32 %5), !dbg !498
  %51 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @.str.7.9, i64 0, i64 0), double %6), !dbg !499
  %52 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.8.10, i64 0, i64 0), double %7), !dbg !500
  %53 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([25 x i8], [25 x i8]* @.str.9.11, i64 0, i64 0), i8* %8), !dbg !501
  %54 = icmp eq i32 %9, 0, !dbg !502
  br i1 %54, label %57, label %55, !dbg !504, !prof !220

; <label>:55:                                     ; preds = %49
  %56 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.10, i64 0, i64 0), i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str.11.12, i64 0, i64 0)), !dbg !505
  br label %59, !dbg !505

; <label>:57:                                     ; preds = %49
  %58 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.10, i64 0, i64 0), i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str.12.13, i64 0, i64 0)), !dbg !506
  br label %59

; <label>:59:                                     ; preds = %57, %55
  %60 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.13.14, i64 0, i64 0), i8* %10), !dbg !507
  %61 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([37 x i8], [37 x i8]* @.str.14.15, i64 0, i64 0), i8* %11), !dbg !508
  %62 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([42 x i8], [42 x i8]* @.str.15.16, i64 0, i64 0), i8* %12), !dbg !509
  %63 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.16.17, i64 0, i64 0), i8* %13), !dbg !510
  %64 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.17.18, i64 0, i64 0), i8* %14), !dbg !511
  %65 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.18.19, i64 0, i64 0), i8* %15), !dbg !512
  %66 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.19.20, i64 0, i64 0), i8* %16), !dbg !513
  %67 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.20.21, i64 0, i64 0), i8* %17), !dbg !514
  %68 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.21.22, i64 0, i64 0), i8* %18), !dbg !515
  %69 = call i32 @puts(i8* getelementptr inbounds ([194 x i8], [194 x i8]* @str.23, i64 0, i64 0)), !dbg !516
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %21) #4, !dbg !517
  ret void, !dbg !517
}

declare double @ldexp(double, i32) local_unnamed_addr

; Function Attrs: nounwind uwtable
define double @randlc(double* nocapture, double) local_unnamed_addr #0 !dbg !518 !prof !540 {
  call void @llvm.dbg.value(metadata double* %0, metadata !523, metadata !DIExpression()), !dbg !541
  call void @llvm.dbg.value(metadata double %1, metadata !524, metadata !DIExpression()), !dbg !542
  call void @llvm.dbg.value(metadata double 0x3E80000000000000, metadata !525, metadata !DIExpression()), !dbg !543
  call void @llvm.dbg.value(metadata double 0x3D10000000000000, metadata !527, metadata !DIExpression()), !dbg !544
  call void @llvm.dbg.value(metadata double 0x4160000000000000, metadata !528, metadata !DIExpression()), !dbg !545
  call void @llvm.dbg.value(metadata double 0x42D0000000000000, metadata !529, metadata !DIExpression()), !dbg !546
  %3 = fmul double %1, 0x3E80000000000000, !dbg !547
  call void @llvm.dbg.value(metadata double %3, metadata !530, metadata !DIExpression()), !dbg !548
  %4 = fptosi double %3 to i32, !dbg !549
  %5 = sitofp i32 %4 to double, !dbg !549
  call void @llvm.dbg.value(metadata double %5, metadata !534, metadata !DIExpression()), !dbg !550
  %6 = fmul double %5, 0x4160000000000000, !dbg !551
  %7 = fsub double %1, %6, !dbg !552
  call void @llvm.dbg.value(metadata double %7, metadata !535, metadata !DIExpression()), !dbg !553
  %8 = load double, double* %0, align 8, !dbg !554, !tbaa !232
  %9 = fmul double %8, 0x3E80000000000000, !dbg !555
  call void @llvm.dbg.value(metadata double %9, metadata !530, metadata !DIExpression()), !dbg !548
  %10 = fptosi double %9 to i32, !dbg !556
  %11 = sitofp i32 %10 to double, !dbg !556
  call void @llvm.dbg.value(metadata double %11, metadata !536, metadata !DIExpression()), !dbg !557
  %12 = fmul double %11, 0x4160000000000000, !dbg !558
  %13 = fsub double %8, %12, !dbg !559
  call void @llvm.dbg.value(metadata double %13, metadata !537, metadata !DIExpression()), !dbg !560
  %14 = fmul double %13, %5, !dbg !561
  %15 = fmul double %7, %11, !dbg !562
  %16 = fadd double %15, %14, !dbg !563
  call void @llvm.dbg.value(metadata double %16, metadata !530, metadata !DIExpression()), !dbg !548
  %17 = fmul double %16, 0x3E80000000000000, !dbg !564
  %18 = fptosi double %17 to i32, !dbg !565
  %19 = sitofp i32 %18 to double, !dbg !565
  call void @llvm.dbg.value(metadata double %19, metadata !531, metadata !DIExpression()), !dbg !566
  %20 = fmul double %19, 0x4160000000000000, !dbg !567
  %21 = fsub double %16, %20, !dbg !568
  call void @llvm.dbg.value(metadata double %21, metadata !538, metadata !DIExpression()), !dbg !569
  %22 = fmul double %21, 0x4160000000000000, !dbg !570
  %23 = fmul double %7, %13, !dbg !571
  %24 = fadd double %23, %22, !dbg !572
  call void @llvm.dbg.value(metadata double %24, metadata !532, metadata !DIExpression()), !dbg !573
  %25 = fmul double %24, 0x3D10000000000000, !dbg !574
  %26 = fptosi double %25 to i32, !dbg !575
  %27 = sitofp i32 %26 to double, !dbg !575
  call void @llvm.dbg.value(metadata double %27, metadata !533, metadata !DIExpression()), !dbg !576
  %28 = fmul double %27, 0x42D0000000000000, !dbg !577
  %29 = fsub double %24, %28, !dbg !578
  store double %29, double* %0, align 8, !dbg !579, !tbaa !232
  %30 = fmul double %29, 0x3D10000000000000, !dbg !580
  call void @llvm.dbg.value(metadata double %30, metadata !539, metadata !DIExpression()), !dbg !581
  ret double %30, !dbg !582
}

; Function Attrs: nounwind uwtable
define void @vranlc(i32, double* nocapture, double, double* nocapture) local_unnamed_addr #0 !dbg !583 !prof !605 {
  call void @llvm.dbg.value(metadata i32 %0, metadata !587, metadata !DIExpression()), !dbg !606
  call void @llvm.dbg.value(metadata double* %1, metadata !588, metadata !DIExpression()), !dbg !607
  call void @llvm.dbg.value(metadata double %2, metadata !589, metadata !DIExpression()), !dbg !608
  call void @llvm.dbg.value(metadata double* %3, metadata !590, metadata !DIExpression()), !dbg !609
  call void @llvm.dbg.value(metadata double 0x3E80000000000000, metadata !591, metadata !DIExpression()), !dbg !610
  call void @llvm.dbg.value(metadata double 0x3D10000000000000, metadata !592, metadata !DIExpression()), !dbg !611
  call void @llvm.dbg.value(metadata double 0x4160000000000000, metadata !593, metadata !DIExpression()), !dbg !612
  call void @llvm.dbg.value(metadata double 0x42D0000000000000, metadata !594, metadata !DIExpression()), !dbg !613
  %5 = fmul double %2, 0x3E80000000000000, !dbg !614
  call void @llvm.dbg.value(metadata double %5, metadata !595, metadata !DIExpression()), !dbg !615
  %6 = fptosi double %5 to i32, !dbg !616
  %7 = sitofp i32 %6 to double, !dbg !616
  call void @llvm.dbg.value(metadata double %7, metadata !599, metadata !DIExpression()), !dbg !617
  %8 = fmul double %7, 0x4160000000000000, !dbg !618
  %9 = fsub double %2, %8, !dbg !619
  call void @llvm.dbg.value(metadata double %9, metadata !600, metadata !DIExpression()), !dbg !620
  call void @llvm.dbg.value(metadata i32 0, metadata !604, metadata !DIExpression()), !dbg !621
  %10 = icmp sgt i32 %0, 0, !dbg !622
  br i1 %10, label %11, label %41, !dbg !625, !prof !626

; <label>:11:                                     ; preds = %4
  %12 = zext i32 %0 to i64
  br label %13, !dbg !625

; <label>:13:                                     ; preds = %13, %11
  %14 = phi i64 [ 0, %11 ], [ %39, %13 ]
  call void @llvm.dbg.value(metadata i64 %14, metadata !604, metadata !DIExpression()), !dbg !621
  %15 = load double, double* %1, align 8, !dbg !627, !tbaa !232
  %16 = fmul double %15, 0x3E80000000000000, !dbg !629
  call void @llvm.dbg.value(metadata double %16, metadata !595, metadata !DIExpression()), !dbg !615
  %17 = fptosi double %16 to i32, !dbg !630
  %18 = sitofp i32 %17 to double, !dbg !630
  call void @llvm.dbg.value(metadata double %18, metadata !601, metadata !DIExpression()), !dbg !631
  %19 = fmul double %18, 0x4160000000000000, !dbg !632
  %20 = fsub double %15, %19, !dbg !633
  call void @llvm.dbg.value(metadata double %20, metadata !602, metadata !DIExpression()), !dbg !634
  %21 = fmul double %20, %7, !dbg !635
  %22 = fmul double %9, %18, !dbg !636
  %23 = fadd double %22, %21, !dbg !637
  call void @llvm.dbg.value(metadata double %23, metadata !595, metadata !DIExpression()), !dbg !615
  %24 = fmul double %23, 0x3E80000000000000, !dbg !638
  %25 = fptosi double %24 to i32, !dbg !639
  %26 = sitofp i32 %25 to double, !dbg !639
  call void @llvm.dbg.value(metadata double %26, metadata !596, metadata !DIExpression()), !dbg !640
  %27 = fmul double %26, 0x4160000000000000, !dbg !641
  %28 = fsub double %23, %27, !dbg !642
  call void @llvm.dbg.value(metadata double %28, metadata !603, metadata !DIExpression()), !dbg !643
  %29 = fmul double %28, 0x4160000000000000, !dbg !644
  %30 = fmul double %9, %20, !dbg !645
  %31 = fadd double %30, %29, !dbg !646
  call void @llvm.dbg.value(metadata double %31, metadata !597, metadata !DIExpression()), !dbg !647
  %32 = fmul double %31, 0x3D10000000000000, !dbg !648
  %33 = fptosi double %32 to i32, !dbg !649
  %34 = sitofp i32 %33 to double, !dbg !649
  call void @llvm.dbg.value(metadata double %34, metadata !598, metadata !DIExpression()), !dbg !650
  %35 = fmul double %34, 0x42D0000000000000, !dbg !651
  %36 = fsub double %31, %35, !dbg !652
  store double %36, double* %1, align 8, !dbg !653, !tbaa !232
  %37 = fmul double %36, 0x3D10000000000000, !dbg !654
  %38 = getelementptr inbounds double, double* %3, i64 %14, !dbg !655
  store double %37, double* %38, align 8, !dbg !656, !tbaa !232
  %39 = add nuw nsw i64 %14, 1, !dbg !657
  %40 = icmp eq i64 %39, %12, !dbg !622
  br i1 %40, label %41, label %13, !dbg !625, !prof !658, !llvm.loop !659

; <label>:41:                                     ; preds = %13, %4
  ret void, !dbg !661
}

; Function Attrs: nounwind uwtable
define void @timer_clear(i32) local_unnamed_addr #0 !dbg !662 !prof !667 {
  call void @llvm.dbg.value(metadata i32 %0, metadata !666, metadata !DIExpression()), !dbg !668
  %2 = sext i32 %0 to i64, !dbg !669
  %3 = getelementptr inbounds [64 x double], [64 x double]* @elapsed, i64 0, i64 %2, !dbg !669
  store double 0.000000e+00, double* %3, align 8, !dbg !670, !tbaa !232
  ret void, !dbg !671
}

; Function Attrs: nounwind uwtable
define void @timer_start(i32) local_unnamed_addr #0 !dbg !672 !prof !196 {
  %2 = alloca double, align 8
  call void @llvm.dbg.value(metadata i32 %0, metadata !674, metadata !DIExpression()), !dbg !675
  %3 = bitcast double* %2 to i8*, !dbg !676
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3) #4, !dbg !676
  call void @llvm.dbg.value(metadata double* %2, metadata !681, metadata !DIExpression()) #4, !dbg !683
  call void @wtime_(double* nonnull %2) #4, !dbg !684
  %4 = bitcast double* %2 to i64*, !dbg !685
  %5 = load i64, i64* %4, align 8, !dbg !685, !tbaa !232
  call void @llvm.dbg.value(metadata double* %2, metadata !681, metadata !DIExpression(DW_OP_deref)) #4, !dbg !683
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3) #4, !dbg !686
  %6 = sext i32 %0 to i64, !dbg !687
  %7 = getelementptr inbounds [64 x double], [64 x double]* @start, i64 0, i64 %6, !dbg !687
  %8 = bitcast double* %7 to i64*, !dbg !688
  store i64 %5, i64* %8, align 8, !dbg !688, !tbaa !232
  ret void, !dbg !689
}

; Function Attrs: nounwind uwtable
define void @timer_stop(i32) local_unnamed_addr #0 !dbg !690 !prof !196 {
  %2 = alloca double, align 8
  call void @llvm.dbg.value(metadata i32 %0, metadata !692, metadata !DIExpression()), !dbg !695
  %3 = bitcast double* %2 to i8*, !dbg !696
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3) #4, !dbg !696
  call void @llvm.dbg.value(metadata double* %2, metadata !681, metadata !DIExpression()) #4, !dbg !698
  call void @wtime_(double* nonnull %2) #4, !dbg !699
  %4 = load double, double* %2, align 8, !dbg !700, !tbaa !232
  call void @llvm.dbg.value(metadata double %4, metadata !681, metadata !DIExpression()) #4, !dbg !698
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3) #4, !dbg !701
  call void @llvm.dbg.value(metadata double %4, metadata !694, metadata !DIExpression()), !dbg !702
  %5 = sext i32 %0 to i64, !dbg !703
  %6 = getelementptr inbounds [64 x double], [64 x double]* @start, i64 0, i64 %5, !dbg !703
  %7 = load double, double* %6, align 8, !dbg !703, !tbaa !232
  %8 = fsub double %4, %7, !dbg !704
  call void @llvm.dbg.value(metadata double %8, metadata !693, metadata !DIExpression()), !dbg !705
  %9 = getelementptr inbounds [64 x double], [64 x double]* @elapsed, i64 0, i64 %5, !dbg !706
  %10 = load double, double* %9, align 8, !dbg !707, !tbaa !232
  %11 = fadd double %10, %8, !dbg !707
  store double %11, double* %9, align 8, !dbg !707, !tbaa !232
  ret void, !dbg !708
}

; Function Attrs: nounwind readonly uwtable
define double @timer_read(i32) local_unnamed_addr #5 !dbg !709 !prof !196 {
  call void @llvm.dbg.value(metadata i32 %0, metadata !713, metadata !DIExpression()), !dbg !714
  %2 = sext i32 %0 to i64, !dbg !715
  %3 = getelementptr inbounds [64 x double], [64 x double]* @elapsed, i64 0, i64 %2, !dbg !715
  %4 = load double, double* %3, align 8, !dbg !715, !tbaa !232
  ret double %4, !dbg !716
}

; Function Attrs: nounwind uwtable
define void @wtime_(double* nocapture) local_unnamed_addr #0 !dbg !36 !prof !717 {
  %2 = alloca %struct.timeval, align 8
  call void @llvm.dbg.value(metadata double* %0, metadata !44, metadata !DIExpression()), !dbg !718
  %3 = bitcast %struct.timeval* %2 to i8*, !dbg !719
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %3) #4, !dbg !719
  call void @llvm.dbg.value(metadata %struct.timeval* %2, metadata !45, metadata !DIExpression()), !dbg !720
  %4 = call i32 @gettimeofday(%struct.timeval* nonnull %2, %struct.timezone* null) #4, !dbg !721
  %5 = load i32, i32* @wtime_.sec, align 4, !dbg !722, !tbaa !724
  %6 = icmp slt i32 %5, 0, !dbg !726
  %7 = getelementptr inbounds %struct.timeval, %struct.timeval* %2, i64 0, i32 0
  %8 = load i64, i64* %7, align 8, !tbaa !727
  br i1 %6, label %9, label %11, !dbg !730, !prof !731

; <label>:9:                                      ; preds = %1
  %10 = trunc i64 %8 to i32, !dbg !732
  store i32 %10, i32* @wtime_.sec, align 4, !dbg !733, !tbaa !724
  br label %11, !dbg !734

; <label>:11:                                     ; preds = %1, %9
  %12 = phi i32 [ %10, %9 ], [ %5, %1 ], !dbg !735
  %13 = sext i32 %12 to i64, !dbg !735
  %14 = sub nsw i64 %8, %13, !dbg !736
  %15 = sitofp i64 %14 to double, !dbg !737
  %16 = getelementptr inbounds %struct.timeval, %struct.timeval* %2, i64 0, i32 1, !dbg !738
  %17 = load i64, i64* %16, align 8, !dbg !738, !tbaa !739
  %18 = sitofp i64 %17 to double, !dbg !740
  %19 = fmul double %18, 0x3EB0C6F7A0B5ED8D, !dbg !741
  %20 = fadd double %19, %15, !dbg !742
  store double %20, double* %0, align 8, !dbg !743, !tbaa !232
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %3) #4, !dbg !744
  ret void, !dbg !744
}

; Function Attrs: nounwind
declare i32 @gettimeofday(%struct.timeval* nocapture, %struct.timezone* nocapture) local_unnamed_addr #3

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }
attributes #5 = { nounwind readonly uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2, !56, !61, !24, !41}
!llvm.ident = !{!64, !64, !64, !64, !64}
!llvm.module.flags = !{!65, !92, !93, !94}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 68, type: !19, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 6.0.0 (tags/RELEASE_600/final)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !10, globals: !12)
!3 = !DIFile(filename: "ep.c", directory: "/users/mcopik/projects/mlcode/nas_benchmarks/SNU_NPB-1.0.3/NPB3.3-SER-C/EP")
!4 = !{!5}
!5 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !6, line: 4, size: 32, elements: !7)
!6 = !DIFile(filename: "../common/type.h", directory: "/users/mcopik/projects/mlcode/nas_benchmarks/SNU_NPB-1.0.3/NPB3.3-SER-C/EP")
!7 = !{!8, !9}
!8 = !DIEnumerator(name: "false", value: 0)
!9 = !DIEnumerator(name: "true", value: 1)
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!12 = !{!0, !13}
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(name: "q", scope: !2, file: !3, line: 69, type: !15, isLocal: true, isDefinition: true)
!15 = !DICompositeType(tag: DW_TAG_array_type, baseType: !16, size: 640, elements: !17)
!16 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!17 = !{!18}
!18 = !DISubrange(count: 10)
!19 = !DICompositeType(tag: DW_TAG_array_type, baseType: !16, size: 8388608, elements: !20)
!20 = !{!21}
!21 = !DISubrange(count: 131072)
!22 = !DIGlobalVariableExpression(var: !23, expr: !DIExpression())
!23 = distinct !DIGlobalVariable(name: "elapsed", scope: !24, file: !30, line: 20, type: !31, isLocal: true, isDefinition: true)
!24 = distinct !DICompileUnit(language: DW_LANG_C99, file: !25, producer: "clang version 6.0.0 (tags/RELEASE_600/final)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !26, globals: !27)
!25 = !DIFile(filename: "../common/c_timers.c", directory: "/users/mcopik/projects/mlcode/nas_benchmarks/SNU_NPB-1.0.3/NPB3.3-SER-C/common")
!26 = !{}
!27 = !{!28, !22}
!28 = !DIGlobalVariableExpression(var: !29, expr: !DIExpression())
!29 = distinct !DIGlobalVariable(name: "start", scope: !24, file: !30, line: 20, type: !31, isLocal: true, isDefinition: true)
!30 = !DIFile(filename: "c_timers.c", directory: "/users/mcopik/projects/mlcode/nas_benchmarks/SNU_NPB-1.0.3/NPB3.3-SER-C/common")
!31 = !DICompositeType(tag: DW_TAG_array_type, baseType: !16, size: 4096, elements: !32)
!32 = !{!33}
!33 = !DISubrange(count: 64)
!34 = !DIGlobalVariableExpression(var: !35, expr: !DIExpression())
!35 = distinct !DIGlobalVariable(name: "sec", scope: !36, file: !37, line: 9, type: !55, isLocal: true, isDefinition: true)
!36 = distinct !DISubprogram(name: "wtime_", scope: !37, file: !37, line: 7, type: !38, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: true, unit: !41, variables: !43)
!37 = !DIFile(filename: "../common/wtime.c", directory: "/users/mcopik/projects/mlcode/nas_benchmarks/SNU_NPB-1.0.3/NPB3.3-SER-C/common")
!38 = !DISubroutineType(types: !39)
!39 = !{null, !40}
!40 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64)
!41 = distinct !DICompileUnit(language: DW_LANG_C99, file: !37, producer: "clang version 6.0.0 (tags/RELEASE_600/final)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !26, globals: !42)
!42 = !{!34}
!43 = !{!44, !45}
!44 = !DILocalVariable(name: "t", arg: 1, scope: !36, file: !37, line: 7, type: !40)
!45 = !DILocalVariable(name: "tv", scope: !36, file: !37, line: 10, type: !46)
!46 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "timeval", file: !47, line: 30, size: 128, elements: !48)
!47 = !DIFile(filename: "/usr/include/bits/time.h", directory: "/users/mcopik/projects/mlcode/nas_benchmarks/SNU_NPB-1.0.3/NPB3.3-SER-C/common")
!48 = !{!49, !53}
!49 = !DIDerivedType(tag: DW_TAG_member, name: "tv_sec", scope: !46, file: !47, line: 32, baseType: !50, size: 64)
!50 = !DIDerivedType(tag: DW_TAG_typedef, name: "__time_t", file: !51, line: 148, baseType: !52)
!51 = !DIFile(filename: "/usr/include/bits/types.h", directory: "/users/mcopik/projects/mlcode/nas_benchmarks/SNU_NPB-1.0.3/NPB3.3-SER-C/common")
!52 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!53 = !DIDerivedType(tag: DW_TAG_member, name: "tv_usec", scope: !46, file: !47, line: 33, baseType: !54, size: 64, offset: 64)
!54 = !DIDerivedType(tag: DW_TAG_typedef, name: "__suseconds_t", file: !51, line: 150, baseType: !52)
!55 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!56 = distinct !DICompileUnit(language: DW_LANG_C99, file: !57, producer: "clang version 6.0.0 (tags/RELEASE_600/final)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !58)
!57 = !DIFile(filename: "../common/print_results.c", directory: "/users/mcopik/projects/mlcode/nas_benchmarks/SNU_NPB-1.0.3/NPB3.3-SER-C/common")
!58 = !{!59}
!59 = !DICompositeType(tag: DW_TAG_enumeration_type, file: !60, line: 4, size: 32, elements: !7)
!60 = !DIFile(filename: "../common/type.h", directory: "/users/mcopik/projects/mlcode/nas_benchmarks/SNU_NPB-1.0.3/NPB3.3-SER-C/common")
!61 = distinct !DICompileUnit(language: DW_LANG_C99, file: !62, producer: "clang version 6.0.0 (tags/RELEASE_600/final)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !26, retainedTypes: !63)
!62 = !DIFile(filename: "../common/randdp.c", directory: "/users/mcopik/projects/mlcode/nas_benchmarks/SNU_NPB-1.0.3/NPB3.3-SER-C/common")
!63 = !{!55}
!64 = !{!"clang version 6.0.0 (tags/RELEASE_600/final)"}
!65 = !{i32 1, !"ProfileSummary", !66}
!66 = !{!67, !68, !69, !70, !71, !72, !73, !74}
!67 = !{!"ProfileFormat", !"InstrProf"}
!68 = !{!"TotalCount", i64 1121842441}
!69 = !{!"MaxCount", i64 536870912}
!70 = !{!"MaxInternalCount", i64 536870912}
!71 = !{!"MaxFunctionCount", i64 65556}
!72 = !{!"NumCounts", i64 48}
!73 = !{!"NumFunctions", i64 10}
!74 = !{!"DetailedSummary", !75}
!75 = !{!76, !77, !78, !79, !80, !81, !81, !82, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91}
!76 = !{i32 10000, i64 536870912, i32 1}
!77 = !{i32 100000, i64 536870912, i32 1}
!78 = !{i32 200000, i64 536870912, i32 1}
!79 = !{i32 300000, i64 536870912, i32 1}
!80 = !{i32 400000, i64 536870912, i32 1}
!81 = !{i32 500000, i64 268435456, i32 2}
!82 = !{i32 600000, i64 268435456, i32 2}
!83 = !{i32 700000, i64 268435456, i32 2}
!84 = !{i32 800000, i64 210832767, i32 3}
!85 = !{i32 900000, i64 210832767, i32 3}
!86 = !{i32 950000, i64 105424685, i32 4}
!87 = !{i32 990000, i64 105424685, i32 4}
!88 = !{i32 999000, i64 105424685, i32 4}
!89 = !{i32 999900, i64 65556, i32 6}
!90 = !{i32 999990, i64 4098, i32 9}
!91 = !{i32 999999, i64 4096, i32 11}
!92 = !{i32 2, !"Dwarf Version", i32 4}
!93 = !{i32 2, !"Debug Info Version", i32 3}
!94 = !{i32 1, !"wchar_size", i32 4}
!95 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 72, type: !96, isLocal: false, isDefinition: true, scopeLine: 73, isOptimized: true, unit: !2, variables: !97)
!96 = !DISubroutineType(types: !63)
!97 = !{!98, !99, !100, !101, !102, !103, !104, !105, !106, !107, !108, !109, !110, !111, !112, !113, !114, !115, !116, !117, !118, !119, !120, !121, !122, !123, !124, !126, !127, !131, !136}
!98 = !DILocalVariable(name: "Mops", scope: !95, file: !3, line: 74, type: !16)
!99 = !DILocalVariable(name: "t1", scope: !95, file: !3, line: 74, type: !16)
!100 = !DILocalVariable(name: "t2", scope: !95, file: !3, line: 74, type: !16)
!101 = !DILocalVariable(name: "t3", scope: !95, file: !3, line: 74, type: !16)
!102 = !DILocalVariable(name: "t4", scope: !95, file: !3, line: 74, type: !16)
!103 = !DILocalVariable(name: "x1", scope: !95, file: !3, line: 74, type: !16)
!104 = !DILocalVariable(name: "x2", scope: !95, file: !3, line: 74, type: !16)
!105 = !DILocalVariable(name: "sx", scope: !95, file: !3, line: 75, type: !16)
!106 = !DILocalVariable(name: "sy", scope: !95, file: !3, line: 75, type: !16)
!107 = !DILocalVariable(name: "tm", scope: !95, file: !3, line: 75, type: !16)
!108 = !DILocalVariable(name: "an", scope: !95, file: !3, line: 75, type: !16)
!109 = !DILocalVariable(name: "tt", scope: !95, file: !3, line: 75, type: !16)
!110 = !DILocalVariable(name: "gc", scope: !95, file: !3, line: 75, type: !16)
!111 = !DILocalVariable(name: "sx_verify_value", scope: !95, file: !3, line: 76, type: !16)
!112 = !DILocalVariable(name: "sy_verify_value", scope: !95, file: !3, line: 76, type: !16)
!113 = !DILocalVariable(name: "sx_err", scope: !95, file: !3, line: 76, type: !16)
!114 = !DILocalVariable(name: "sy_err", scope: !95, file: !3, line: 76, type: !16)
!115 = !DILocalVariable(name: "np", scope: !95, file: !3, line: 77, type: !55)
!116 = !DILocalVariable(name: "i", scope: !95, file: !3, line: 78, type: !55)
!117 = !DILocalVariable(name: "ik", scope: !95, file: !3, line: 78, type: !55)
!118 = !DILocalVariable(name: "kk", scope: !95, file: !3, line: 78, type: !55)
!119 = !DILocalVariable(name: "l", scope: !95, file: !3, line: 78, type: !55)
!120 = !DILocalVariable(name: "k", scope: !95, file: !3, line: 78, type: !55)
!121 = !DILocalVariable(name: "nit", scope: !95, file: !3, line: 78, type: !55)
!122 = !DILocalVariable(name: "k_offset", scope: !95, file: !3, line: 79, type: !55)
!123 = !DILocalVariable(name: "j", scope: !95, file: !3, line: 79, type: !55)
!124 = !DILocalVariable(name: "verified", scope: !95, file: !3, line: 80, type: !125)
!125 = !DIDerivedType(tag: DW_TAG_typedef, name: "logical", file: !6, line: 4, baseType: !5)
!126 = !DILocalVariable(name: "timers_enabled", scope: !95, file: !3, line: 80, type: !125)
!127 = !DILocalVariable(name: "dum", scope: !95, file: !3, line: 82, type: !128)
!128 = !DICompositeType(tag: DW_TAG_array_type, baseType: !16, size: 192, elements: !129)
!129 = !{!130}
!130 = !DISubrange(count: 3)
!131 = !DILocalVariable(name: "size", scope: !95, file: !3, line: 83, type: !132)
!132 = !DICompositeType(tag: DW_TAG_array_type, baseType: !133, size: 128, elements: !134)
!133 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!134 = !{!135}
!135 = !DISubrange(count: 16)
!136 = !DILocalVariable(name: "fp", scope: !95, file: !3, line: 85, type: !137)
!137 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !138, size: 64)
!138 = !DIDerivedType(tag: DW_TAG_typedef, name: "FILE", file: !139, line: 48, baseType: !140)
!139 = !DIFile(filename: "/usr/include/stdio.h", directory: "/users/mcopik/projects/mlcode/nas_benchmarks/SNU_NPB-1.0.3/NPB3.3-SER-C/EP")
!140 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "_IO_FILE", file: !141, line: 246, size: 1728, elements: !142)
!141 = !DIFile(filename: "/usr/include/libio.h", directory: "/users/mcopik/projects/mlcode/nas_benchmarks/SNU_NPB-1.0.3/NPB3.3-SER-C/EP")
!142 = !{!143, !144, !146, !147, !148, !149, !150, !151, !152, !153, !154, !155, !156, !164, !165, !166, !167, !170, !172, !174, !178, !181, !183, !184, !185, !186, !187, !191, !192}
!143 = !DIDerivedType(tag: DW_TAG_member, name: "_flags", scope: !140, file: !141, line: 247, baseType: !55, size: 32)
!144 = !DIDerivedType(tag: DW_TAG_member, name: "_IO_read_ptr", scope: !140, file: !141, line: 252, baseType: !145, size: 64, offset: 64)
!145 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !133, size: 64)
!146 = !DIDerivedType(tag: DW_TAG_member, name: "_IO_read_end", scope: !140, file: !141, line: 253, baseType: !145, size: 64, offset: 128)
!147 = !DIDerivedType(tag: DW_TAG_member, name: "_IO_read_base", scope: !140, file: !141, line: 254, baseType: !145, size: 64, offset: 192)
!148 = !DIDerivedType(tag: DW_TAG_member, name: "_IO_write_base", scope: !140, file: !141, line: 255, baseType: !145, size: 64, offset: 256)
!149 = !DIDerivedType(tag: DW_TAG_member, name: "_IO_write_ptr", scope: !140, file: !141, line: 256, baseType: !145, size: 64, offset: 320)
!150 = !DIDerivedType(tag: DW_TAG_member, name: "_IO_write_end", scope: !140, file: !141, line: 257, baseType: !145, size: 64, offset: 384)
!151 = !DIDerivedType(tag: DW_TAG_member, name: "_IO_buf_base", scope: !140, file: !141, line: 258, baseType: !145, size: 64, offset: 448)
!152 = !DIDerivedType(tag: DW_TAG_member, name: "_IO_buf_end", scope: !140, file: !141, line: 259, baseType: !145, size: 64, offset: 512)
!153 = !DIDerivedType(tag: DW_TAG_member, name: "_IO_save_base", scope: !140, file: !141, line: 261, baseType: !145, size: 64, offset: 576)
!154 = !DIDerivedType(tag: DW_TAG_member, name: "_IO_backup_base", scope: !140, file: !141, line: 262, baseType: !145, size: 64, offset: 640)
!155 = !DIDerivedType(tag: DW_TAG_member, name: "_IO_save_end", scope: !140, file: !141, line: 263, baseType: !145, size: 64, offset: 704)
!156 = !DIDerivedType(tag: DW_TAG_member, name: "_markers", scope: !140, file: !141, line: 265, baseType: !157, size: 64, offset: 768)
!157 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !158, size: 64)
!158 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "_IO_marker", file: !141, line: 161, size: 192, elements: !159)
!159 = !{!160, !161, !163}
!160 = !DIDerivedType(tag: DW_TAG_member, name: "_next", scope: !158, file: !141, line: 162, baseType: !157, size: 64)
!161 = !DIDerivedType(tag: DW_TAG_member, name: "_sbuf", scope: !158, file: !141, line: 163, baseType: !162, size: 64, offset: 64)
!162 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !140, size: 64)
!163 = !DIDerivedType(tag: DW_TAG_member, name: "_pos", scope: !158, file: !141, line: 167, baseType: !55, size: 32, offset: 128)
!164 = !DIDerivedType(tag: DW_TAG_member, name: "_chain", scope: !140, file: !141, line: 267, baseType: !162, size: 64, offset: 832)
!165 = !DIDerivedType(tag: DW_TAG_member, name: "_fileno", scope: !140, file: !141, line: 269, baseType: !55, size: 32, offset: 896)
!166 = !DIDerivedType(tag: DW_TAG_member, name: "_flags2", scope: !140, file: !141, line: 273, baseType: !55, size: 32, offset: 928)
!167 = !DIDerivedType(tag: DW_TAG_member, name: "_old_offset", scope: !140, file: !141, line: 275, baseType: !168, size: 64, offset: 960)
!168 = !DIDerivedType(tag: DW_TAG_typedef, name: "__off_t", file: !169, line: 140, baseType: !52)
!169 = !DIFile(filename: "/usr/include/bits/types.h", directory: "/users/mcopik/projects/mlcode/nas_benchmarks/SNU_NPB-1.0.3/NPB3.3-SER-C/EP")
!170 = !DIDerivedType(tag: DW_TAG_member, name: "_cur_column", scope: !140, file: !141, line: 279, baseType: !171, size: 16, offset: 1024)
!171 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!172 = !DIDerivedType(tag: DW_TAG_member, name: "_vtable_offset", scope: !140, file: !141, line: 280, baseType: !173, size: 8, offset: 1040)
!173 = !DIBasicType(name: "signed char", size: 8, encoding: DW_ATE_signed_char)
!174 = !DIDerivedType(tag: DW_TAG_member, name: "_shortbuf", scope: !140, file: !141, line: 281, baseType: !175, size: 8, offset: 1048)
!175 = !DICompositeType(tag: DW_TAG_array_type, baseType: !133, size: 8, elements: !176)
!176 = !{!177}
!177 = !DISubrange(count: 1)
!178 = !DIDerivedType(tag: DW_TAG_member, name: "_lock", scope: !140, file: !141, line: 285, baseType: !179, size: 64, offset: 1088)
!179 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !180, size: 64)
!180 = !DIDerivedType(tag: DW_TAG_typedef, name: "_IO_lock_t", file: !141, line: 155, baseType: null)
!181 = !DIDerivedType(tag: DW_TAG_member, name: "_offset", scope: !140, file: !141, line: 294, baseType: !182, size: 64, offset: 1152)
!182 = !DIDerivedType(tag: DW_TAG_typedef, name: "__off64_t", file: !169, line: 141, baseType: !52)
!183 = !DIDerivedType(tag: DW_TAG_member, name: "__pad1", scope: !140, file: !141, line: 303, baseType: !11, size: 64, offset: 1216)
!184 = !DIDerivedType(tag: DW_TAG_member, name: "__pad2", scope: !140, file: !141, line: 304, baseType: !11, size: 64, offset: 1280)
!185 = !DIDerivedType(tag: DW_TAG_member, name: "__pad3", scope: !140, file: !141, line: 305, baseType: !11, size: 64, offset: 1344)
!186 = !DIDerivedType(tag: DW_TAG_member, name: "__pad4", scope: !140, file: !141, line: 306, baseType: !11, size: 64, offset: 1408)
!187 = !DIDerivedType(tag: DW_TAG_member, name: "__pad5", scope: !140, file: !141, line: 307, baseType: !188, size: 64, offset: 1472)
!188 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !189, line: 62, baseType: !190)
!189 = !DIFile(filename: "/users2/mcopik/projects/clang-6.0/install/lib/clang/6.0.0/include/stddef.h", directory: "/users/mcopik/projects/mlcode/nas_benchmarks/SNU_NPB-1.0.3/NPB3.3-SER-C/EP")
!190 = !DIBasicType(name: "long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!191 = !DIDerivedType(tag: DW_TAG_member, name: "_mode", scope: !140, file: !141, line: 309, baseType: !55, size: 32, offset: 1536)
!192 = !DIDerivedType(tag: DW_TAG_member, name: "_unused2", scope: !140, file: !141, line: 311, baseType: !193, size: 160, offset: 1568)
!193 = !DICompositeType(tag: DW_TAG_array_type, baseType: !133, size: 160, elements: !194)
!194 = !{!195}
!195 = !DISubrange(count: 20)
!196 = !{!"function_entry_count", i64 1}
!197 = !DILocation(line: 74, column: 3, scope: !95)
!198 = !DILocation(line: 82, column: 3, scope: !95)
!199 = !DILocation(line: 82, column: 10, scope: !95)
!200 = !DILocation(line: 83, column: 3, scope: !95)
!201 = !DILocation(line: 83, column: 10, scope: !95)
!202 = !DILocation(line: 87, column: 13, scope: !203)
!203 = distinct !DILexicalBlock(scope: !95, file: !3, line: 87, column: 7)
!204 = !DILocation(line: 85, column: 9, scope: !95)
!205 = !DILocation(line: 87, column: 39, scope: !203)
!206 = !DILocation(line: 87, column: 7, scope: !95)
!207 = !{!"branch_weights", i32 2, i32 1}
!208 = !DILocation(line: 80, column: 21, scope: !95)
!209 = !DILocation(line: 91, column: 5, scope: !210)
!210 = distinct !DILexicalBlock(scope: !203, file: !3, line: 89, column: 10)
!211 = !DILocation(line: 101, column: 3, scope: !95)
!212 = !DILocation(line: 79, column: 20, scope: !95)
!213 = !DILocation(line: 103, column: 7, scope: !214)
!214 = distinct !DILexicalBlock(scope: !95, file: !3, line: 103, column: 7)
!215 = !{!216, !216, i64 0}
!216 = !{!"omnipotent char", !217, i64 0}
!217 = !{!"Simple C/C++ TBAA"}
!218 = !DILocation(line: 103, column: 15, scope: !214)
!219 = !DILocation(line: 103, column: 7, scope: !95)
!220 = !{!"branch_weights", i32 1, i32 2}
!221 = !DILocation(line: 104, column: 3, scope: !95)
!222 = !DILocation(line: 104, column: 13, scope: !95)
!223 = !DILocation(line: 105, column: 3, scope: !95)
!224 = !DILocation(line: 106, column: 3, scope: !95)
!225 = !DILocation(line: 80, column: 11, scope: !95)
!226 = !DILocation(line: 77, column: 10, scope: !95)
!227 = !DILocation(line: 125, column: 14, scope: !95)
!228 = !DILocation(line: 125, column: 22, scope: !95)
!229 = !DILocation(line: 125, column: 31, scope: !95)
!230 = !DILocation(line: 125, column: 3, scope: !95)
!231 = !DILocation(line: 126, column: 28, scope: !95)
!232 = !{!233, !233, i64 0}
!233 = !{!"double", !216, i64 0}
!234 = !DILocation(line: 126, column: 12, scope: !95)
!235 = !DILocation(line: 126, column: 10, scope: !95)
!236 = !DILocation(line: 78, column: 10, scope: !95)
!237 = !DILocation(line: 127, column: 3, scope: !238)
!238 = distinct !DILexicalBlock(scope: !95, file: !3, line: 127, column: 3)
!239 = !DILocation(line: 127, column: 28, scope: !240)
!240 = distinct !DILexicalBlock(scope: !238, file: !3, line: 127, column: 3)
!241 = !DILocation(line: 128, column: 5, scope: !242)
!242 = distinct !DILexicalBlock(scope: !240, file: !3, line: 127, column: 32)
!243 = !DILocation(line: 128, column: 10, scope: !242)
!244 = distinct !{!244, !237, !245, !246}
!245 = !DILocation(line: 129, column: 3, scope: !238)
!246 = !{!"llvm.loop.isvectorized", i32 1}
!247 = !DILocation(line: 132, column: 3, scope: !95)
!248 = !DILocation(line: 133, column: 3, scope: !95)
!249 = !DILocation(line: 134, column: 3, scope: !95)
!250 = !DILocation(line: 135, column: 3, scope: !95)
!251 = !DILocation(line: 74, column: 16, scope: !95)
!252 = !DILocation(line: 137, column: 6, scope: !95)
!253 = !DILocation(line: 138, column: 3, scope: !95)
!254 = !DILocation(line: 144, column: 6, scope: !95)
!255 = !DILocation(line: 147, column: 10, scope: !256)
!256 = distinct !DILexicalBlock(scope: !257, file: !3, line: 146, column: 32)
!257 = distinct !DILexicalBlock(scope: !258, file: !3, line: 146, column: 3)
!258 = distinct !DILexicalBlock(scope: !95, file: !3, line: 146, column: 3)
!259 = !DILocation(line: 74, column: 20, scope: !95)
!260 = !DILocation(line: 147, column: 8, scope: !256)
!261 = !DILocation(line: 75, column: 22, scope: !95)
!262 = !DILocation(line: 75, column: 26, scope: !95)
!263 = !DILocation(line: 75, column: 30, scope: !95)
!264 = !DILocation(line: 75, column: 10, scope: !95)
!265 = !DILocation(line: 75, column: 14, scope: !95)
!266 = !DILocation(line: 157, column: 10, scope: !267)
!267 = distinct !DILexicalBlock(scope: !268, file: !3, line: 156, column: 28)
!268 = distinct !DILexicalBlock(scope: !269, file: !3, line: 156, column: 3)
!269 = distinct !DILexicalBlock(scope: !95, file: !3, line: 156, column: 3)
!270 = !DILocation(line: 79, column: 10, scope: !95)
!271 = !DILocation(line: 78, column: 24, scope: !95)
!272 = !DILocation(line: 168, column: 3, scope: !273)
!273 = distinct !DILexicalBlock(scope: !95, file: !3, line: 168, column: 3)
!274 = !{!"branch_weights", i32 4097, i32 1}
!275 = !DILocation(line: 169, column: 19, scope: !276)
!276 = distinct !DILexicalBlock(scope: !277, file: !3, line: 168, column: 29)
!277 = distinct !DILexicalBlock(scope: !273, file: !3, line: 168, column: 3)
!278 = !DILocation(line: 78, column: 17, scope: !95)
!279 = !DILocation(line: 170, column: 8, scope: !276)
!280 = !DILocation(line: 171, column: 8, scope: !276)
!281 = !DILocation(line: 175, column: 5, scope: !282)
!282 = distinct !DILexicalBlock(scope: !276, file: !3, line: 175, column: 5)
!283 = !DILocation(line: 176, column: 15, scope: !284)
!284 = distinct !DILexicalBlock(scope: !285, file: !3, line: 175, column: 32)
!285 = distinct !DILexicalBlock(scope: !282, file: !3, line: 175, column: 5)
!286 = !DILocation(line: 78, column: 13, scope: !95)
!287 = !DILocation(line: 177, column: 14, scope: !288)
!288 = distinct !DILexicalBlock(scope: !284, file: !3, line: 177, column: 11)
!289 = !DILocation(line: 177, column: 20, scope: !288)
!290 = !DILocation(line: 177, column: 11, scope: !284)
!291 = !{!"branch_weights", i32 20483, i32 24577}
!292 = !DILocation(line: 177, column: 44, scope: !288)
!293 = !DILocation(line: 177, column: 32, scope: !288)
!294 = !DILocation(line: 74, column: 24, scope: !95)
!295 = !DILocation(line: 177, column: 27, scope: !288)
!296 = !DILocation(line: 178, column: 14, scope: !297)
!297 = distinct !DILexicalBlock(scope: !284, file: !3, line: 178, column: 11)
!298 = !DILocation(line: 178, column: 11, scope: !284)
!299 = !{!"branch_weights", i32 4097, i32 40963}
!300 = !DILocation(line: 179, column: 24, scope: !284)
!301 = !DILocation(line: 179, column: 12, scope: !284)
!302 = !DILocation(line: 175, column: 28, scope: !285)
!303 = !DILocation(line: 175, column: 19, scope: !285)
!304 = !{!"branch_weights", i32 45059, i32 1}
!305 = distinct !{!305, !281, !306}
!306 = !DILocation(line: 181, column: 5, scope: !282)
!307 = !DILocation(line: 186, column: 25, scope: !308)
!308 = distinct !DILexicalBlock(scope: !276, file: !3, line: 186, column: 9)
!309 = !DILocation(line: 187, column: 5, scope: !276)
!310 = !DILocation(line: 188, column: 25, scope: !311)
!311 = distinct !DILexicalBlock(scope: !276, file: !3, line: 188, column: 9)
!312 = !DILocation(line: 195, column: 25, scope: !313)
!313 = distinct !DILexicalBlock(scope: !276, file: !3, line: 195, column: 9)
!314 = !DILocation(line: 197, column: 5, scope: !315)
!315 = distinct !DILexicalBlock(scope: !276, file: !3, line: 197, column: 5)
!316 = !DILocation(line: 212, column: 25, scope: !317)
!317 = distinct !DILexicalBlock(scope: !276, file: !3, line: 212, column: 9)
!318 = !DILocation(line: 168, column: 25, scope: !277)
!319 = !DILocation(line: 168, column: 17, scope: !277)
!320 = !{!"branch_weights", i32 2, i32 4097}
!321 = distinct !{!321, !272, !322}
!322 = !DILocation(line: 213, column: 3, scope: !273)
!323 = !DILocation(line: 198, column: 21, scope: !324)
!324 = distinct !DILexicalBlock(scope: !325, file: !3, line: 197, column: 30)
!325 = distinct !DILexicalBlock(scope: !315, file: !3, line: 197, column: 5)
!326 = !DILocation(line: 198, column: 18, scope: !324)
!327 = !DILocation(line: 74, column: 32, scope: !95)
!328 = !DILocation(line: 198, column: 16, scope: !324)
!329 = !DILocation(line: 198, column: 25, scope: !324)
!330 = !DILocation(line: 74, column: 36, scope: !95)
!331 = !DILocation(line: 200, column: 15, scope: !324)
!332 = !DILocation(line: 200, column: 20, scope: !324)
!333 = !DILocation(line: 200, column: 10, scope: !324)
!334 = !DILocation(line: 201, column: 14, scope: !335)
!335 = distinct !DILexicalBlock(scope: !324, file: !3, line: 201, column: 11)
!336 = !DILocation(line: 201, column: 11, scope: !324)
!337 = !{!"branch_weights", i32 57602690, i32 210832768}
!338 = !DILocation(line: 202, column: 28, scope: !339)
!339 = distinct !DILexicalBlock(scope: !335, file: !3, line: 201, column: 22)
!340 = !DILocation(line: 202, column: 26, scope: !339)
!341 = !DILocation(line: 202, column: 38, scope: !339)
!342 = !DILocation(line: 202, column: 36, scope: !339)
!343 = !DILocation(line: 202, column: 16, scope: !339)
!344 = !DILocation(line: 202, column: 14, scope: !339)
!345 = !DILocation(line: 203, column: 20, scope: !339)
!346 = !DILocation(line: 74, column: 28, scope: !95)
!347 = !DILocation(line: 205, column: 16, scope: !339)
!348 = !{!"branch_weights", i32 105424686, i32 105408083}
!349 = !DILocation(line: 78, column: 21, scope: !95)
!350 = !DILocation(line: 206, column: 16, scope: !339)
!351 = !DILocation(line: 206, column: 21, scope: !339)
!352 = !DILocation(line: 206, column: 14, scope: !339)
!353 = !DILocation(line: 207, column: 19, scope: !339)
!354 = !DILocation(line: 209, column: 7, scope: !339)
!355 = !DILocation(line: 197, column: 26, scope: !325)
!356 = !DILocation(line: 197, column: 19, scope: !325)
!357 = !{!"branch_weights", i32 4097, i32 268435457}
!358 = distinct !{!358, !314, !359}
!359 = !DILocation(line: 210, column: 5, scope: !315)
!360 = !DILocation(line: 216, column: 15, scope: !361)
!361 = distinct !DILexicalBlock(scope: !362, file: !3, line: 215, column: 28)
!362 = distinct !DILexicalBlock(scope: !363, file: !3, line: 215, column: 3)
!363 = distinct !DILexicalBlock(scope: !95, file: !3, line: 215, column: 3)
!364 = !DILocation(line: 216, column: 13, scope: !361)
!365 = !DILocation(line: 219, column: 3, scope: !95)
!366 = !DILocation(line: 220, column: 8, scope: !95)
!367 = !DILocation(line: 75, column: 18, scope: !95)
!368 = !DILocation(line: 78, column: 27, scope: !95)
!369 = !DILocation(line: 76, column: 10, scope: !95)
!370 = !DILocation(line: 76, column: 27, scope: !95)
!371 = !DILocation(line: 250, column: 23, scope: !372)
!372 = distinct !DILexicalBlock(scope: !373, file: !3, line: 249, column: 17)
!373 = distinct !DILexicalBlock(scope: !95, file: !3, line: 249, column: 7)
!374 = !DILocation(line: 76, column: 44, scope: !95)
!375 = !DILocation(line: 251, column: 23, scope: !372)
!376 = !DILocation(line: 251, column: 42, scope: !372)
!377 = !DILocation(line: 251, column: 14, scope: !372)
!378 = !DILocation(line: 76, column: 52, scope: !95)
!379 = !DILocation(line: 252, column: 48, scope: !372)
!380 = !DILocation(line: 252, column: 37, scope: !372)
!381 = !DILocation(line: 255, column: 24, scope: !95)
!382 = !DILocation(line: 74, column: 10, scope: !95)
!383 = !DILocation(line: 257, column: 3, scope: !95)
!384 = !DILocation(line: 258, column: 3, scope: !95)
!385 = !DILocation(line: 259, column: 3, scope: !95)
!386 = !DILocation(line: 260, column: 3, scope: !95)
!387 = !DILocation(line: 261, column: 3, scope: !95)
!388 = !DILocation(line: 262, column: 3, scope: !95)
!389 = !DILocation(line: 264, column: 31, scope: !390)
!390 = distinct !DILexicalBlock(scope: !391, file: !3, line: 263, column: 28)
!391 = distinct !DILexicalBlock(scope: !392, file: !3, line: 263, column: 3)
!392 = distinct !DILexicalBlock(scope: !95, file: !3, line: 263, column: 3)
!393 = !DILocation(line: 264, column: 5, scope: !390)
!394 = !DILocation(line: 255, column: 29, scope: !95)
!395 = !DILocation(line: 267, column: 3, scope: !95)
!396 = !DILocation(line: 273, column: 7, scope: !95)
!397 = !DILocation(line: 274, column: 12, scope: !398)
!398 = distinct !DILexicalBlock(scope: !399, file: !3, line: 274, column: 9)
!399 = distinct !DILexicalBlock(scope: !400, file: !3, line: 273, column: 23)
!400 = distinct !DILexicalBlock(scope: !95, file: !3, line: 273, column: 7)
!401 = !DILocation(line: 274, column: 9, scope: !399)
!402 = !DILocation(line: 275, column: 10, scope: !399)
!403 = !DILocation(line: 276, column: 57, scope: !399)
!404 = !DILocation(line: 276, column: 63, scope: !399)
!405 = !DILocation(line: 276, column: 5, scope: !399)
!406 = !DILocation(line: 277, column: 10, scope: !399)
!407 = !DILocation(line: 278, column: 55, scope: !399)
!408 = !DILocation(line: 278, column: 61, scope: !399)
!409 = !DILocation(line: 278, column: 5, scope: !399)
!410 = !DILocation(line: 279, column: 10, scope: !399)
!411 = !DILocation(line: 280, column: 55, scope: !399)
!412 = !DILocation(line: 280, column: 61, scope: !399)
!413 = !DILocation(line: 280, column: 5, scope: !399)
!414 = !DILocation(line: 281, column: 3, scope: !399)
!415 = !DILocation(line: 284, column: 1, scope: !95)
!416 = !DILocation(line: 283, column: 3, scope: !95)
!417 = distinct !DISubprogram(name: "print_results", scope: !418, file: !418, line: 6, type: !419, isLocal: false, isDefinition: true, scopeLine: 10, flags: DIFlagPrototyped, isOptimized: true, unit: !56, variables: !422)
!418 = !DIFile(filename: "print_results.c", directory: "/users/mcopik/projects/mlcode/nas_benchmarks/SNU_NPB-1.0.3/NPB3.3-SER-C/common")
!419 = !DISubroutineType(types: !420)
!420 = !{null, !145, !133, !55, !55, !55, !55, !16, !16, !145, !421, !145, !145, !145, !145, !145, !145, !145, !145, !145}
!421 = !DIDerivedType(tag: DW_TAG_typedef, name: "logical", file: !60, line: 4, baseType: !59)
!422 = !{!423, !424, !425, !426, !427, !428, !429, !430, !431, !432, !433, !434, !435, !436, !437, !438, !439, !440, !441, !442, !443}
!423 = !DILocalVariable(name: "name", arg: 1, scope: !417, file: !418, line: 6, type: !145)
!424 = !DILocalVariable(name: "class", arg: 2, scope: !417, file: !418, line: 6, type: !133)
!425 = !DILocalVariable(name: "n1", arg: 3, scope: !417, file: !418, line: 6, type: !55)
!426 = !DILocalVariable(name: "n2", arg: 4, scope: !417, file: !418, line: 6, type: !55)
!427 = !DILocalVariable(name: "n3", arg: 5, scope: !417, file: !418, line: 6, type: !55)
!428 = !DILocalVariable(name: "niter", arg: 6, scope: !417, file: !418, line: 6, type: !55)
!429 = !DILocalVariable(name: "t", arg: 7, scope: !417, file: !418, line: 7, type: !16)
!430 = !DILocalVariable(name: "mops", arg: 8, scope: !417, file: !418, line: 7, type: !16)
!431 = !DILocalVariable(name: "optype", arg: 9, scope: !417, file: !418, line: 7, type: !145)
!432 = !DILocalVariable(name: "verified", arg: 10, scope: !417, file: !418, line: 7, type: !421)
!433 = !DILocalVariable(name: "npbversion", arg: 11, scope: !417, file: !418, line: 7, type: !145)
!434 = !DILocalVariable(name: "compiletime", arg: 12, scope: !417, file: !418, line: 8, type: !145)
!435 = !DILocalVariable(name: "cs1", arg: 13, scope: !417, file: !418, line: 8, type: !145)
!436 = !DILocalVariable(name: "cs2", arg: 14, scope: !417, file: !418, line: 8, type: !145)
!437 = !DILocalVariable(name: "cs3", arg: 15, scope: !417, file: !418, line: 8, type: !145)
!438 = !DILocalVariable(name: "cs4", arg: 16, scope: !417, file: !418, line: 8, type: !145)
!439 = !DILocalVariable(name: "cs5", arg: 17, scope: !417, file: !418, line: 8, type: !145)
!440 = !DILocalVariable(name: "cs6", arg: 18, scope: !417, file: !418, line: 9, type: !145)
!441 = !DILocalVariable(name: "cs7", arg: 19, scope: !417, file: !418, line: 9, type: !145)
!442 = !DILocalVariable(name: "size", scope: !417, file: !418, line: 11, type: !132)
!443 = !DILocalVariable(name: "j", scope: !417, file: !418, line: 12, type: !55)
!444 = !DILocation(line: 6, column: 26, scope: !417)
!445 = !DILocation(line: 6, column: 37, scope: !417)
!446 = !DILocation(line: 6, column: 48, scope: !417)
!447 = !DILocation(line: 6, column: 56, scope: !417)
!448 = !DILocation(line: 6, column: 64, scope: !417)
!449 = !DILocation(line: 6, column: 72, scope: !417)
!450 = !DILocation(line: 7, column: 12, scope: !417)
!451 = !DILocation(line: 7, column: 22, scope: !417)
!452 = !DILocation(line: 7, column: 34, scope: !417)
!453 = !DILocation(line: 7, column: 50, scope: !417)
!454 = !DILocation(line: 7, column: 66, scope: !417)
!455 = !DILocation(line: 8, column: 11, scope: !417)
!456 = !DILocation(line: 8, column: 30, scope: !417)
!457 = !DILocation(line: 8, column: 41, scope: !417)
!458 = !DILocation(line: 8, column: 52, scope: !417)
!459 = !DILocation(line: 8, column: 63, scope: !417)
!460 = !DILocation(line: 8, column: 74, scope: !417)
!461 = !DILocation(line: 9, column: 11, scope: !417)
!462 = !DILocation(line: 9, column: 22, scope: !417)
!463 = !DILocation(line: 11, column: 3, scope: !417)
!464 = !DILocation(line: 11, column: 8, scope: !417)
!465 = !DILocation(line: 14, column: 3, scope: !417)
!466 = !DILocation(line: 15, column: 52, scope: !417)
!467 = !DILocation(line: 15, column: 3, scope: !417)
!468 = !DILocation(line: 22, column: 20, scope: !469)
!469 = distinct !DILexicalBlock(scope: !417, file: !418, line: 22, column: 8)
!470 = !{!"branch_weights", i32 4, i32 5}
!471 = !DILocation(line: 23, column: 12, scope: !472)
!472 = distinct !DILexicalBlock(scope: !473, file: !418, line: 23, column: 10)
!473 = distinct !DILexicalBlock(scope: !469, file: !418, line: 22, column: 37)
!474 = !DILocation(line: 23, column: 20, scope: !472)
!475 = !DILocation(line: 23, column: 29, scope: !472)
!476 = !DILocation(line: 23, column: 34, scope: !472)
!477 = !DILocation(line: 23, column: 42, scope: !472)
!478 = !DILocation(line: 23, column: 10, scope: !473)
!479 = !DILocation(line: 24, column: 33, scope: !480)
!480 = distinct !DILexicalBlock(scope: !472, file: !418, line: 23, column: 53)
!481 = !DILocation(line: 24, column: 7, scope: !480)
!482 = !DILocation(line: 12, column: 7, scope: !417)
!483 = !DILocation(line: 26, column: 12, scope: !484)
!484 = distinct !DILexicalBlock(scope: !480, file: !418, line: 26, column: 12)
!485 = !DILocation(line: 26, column: 20, scope: !484)
!486 = !DILocation(line: 26, column: 12, scope: !480)
!487 = !DILocation(line: 27, column: 17, scope: !488)
!488 = distinct !DILexicalBlock(scope: !484, file: !418, line: 26, column: 29)
!489 = !DILocation(line: 29, column: 7, scope: !488)
!490 = !DILocation(line: 30, column: 7, scope: !480)
!491 = !DILocation(line: 30, column: 17, scope: !480)
!492 = !DILocation(line: 31, column: 7, scope: !480)
!493 = !DILocation(line: 32, column: 5, scope: !480)
!494 = !DILocation(line: 33, column: 7, scope: !495)
!495 = distinct !DILexicalBlock(scope: !472, file: !418, line: 32, column: 12)
!496 = !DILocation(line: 36, column: 5, scope: !497)
!497 = distinct !DILexicalBlock(scope: !469, file: !418, line: 35, column: 10)
!498 = !DILocation(line: 39, column: 3, scope: !417)
!499 = !DILocation(line: 40, column: 3, scope: !417)
!500 = !DILocation(line: 41, column: 3, scope: !417)
!501 = !DILocation(line: 42, column: 3, scope: !417)
!502 = !DILocation(line: 43, column: 8, scope: !503)
!503 = distinct !DILexicalBlock(scope: !417, file: !418, line: 43, column: 8)
!504 = !DILocation(line: 43, column: 8, scope: !417)
!505 = !DILocation(line: 44, column: 5, scope: !503)
!506 = !DILocation(line: 46, column: 5, scope: !503)
!507 = !DILocation(line: 47, column: 3, scope: !417)
!508 = !DILocation(line: 48, column: 3, scope: !417)
!509 = !DILocation(line: 50, column: 3, scope: !417)
!510 = !DILocation(line: 52, column: 3, scope: !417)
!511 = !DILocation(line: 53, column: 3, scope: !417)
!512 = !DILocation(line: 54, column: 3, scope: !417)
!513 = !DILocation(line: 55, column: 3, scope: !417)
!514 = !DILocation(line: 56, column: 3, scope: !417)
!515 = !DILocation(line: 57, column: 3, scope: !417)
!516 = !DILocation(line: 59, column: 3, scope: !417)
!517 = !DILocation(line: 65, column: 1, scope: !417)
!518 = distinct !DISubprogram(name: "randlc", scope: !519, file: !519, line: 4, type: !520, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: true, unit: !61, variables: !522)
!519 = !DIFile(filename: "randdp.c", directory: "/users/mcopik/projects/mlcode/nas_benchmarks/SNU_NPB-1.0.3/NPB3.3-SER-C/common")
!520 = !DISubroutineType(types: !521)
!521 = !{!16, !40, !16}
!522 = !{!523, !524, !525, !527, !528, !529, !530, !531, !532, !533, !534, !535, !536, !537, !538, !539}
!523 = !DILocalVariable(name: "x", arg: 1, scope: !518, file: !519, line: 4, type: !40)
!524 = !DILocalVariable(name: "a", arg: 2, scope: !518, file: !519, line: 4, type: !16)
!525 = !DILocalVariable(name: "r23", scope: !518, file: !519, line: 36, type: !526)
!526 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !16)
!527 = !DILocalVariable(name: "r46", scope: !518, file: !519, line: 37, type: !526)
!528 = !DILocalVariable(name: "t23", scope: !518, file: !519, line: 38, type: !526)
!529 = !DILocalVariable(name: "t46", scope: !518, file: !519, line: 39, type: !526)
!530 = !DILocalVariable(name: "t1", scope: !518, file: !519, line: 41, type: !16)
!531 = !DILocalVariable(name: "t2", scope: !518, file: !519, line: 41, type: !16)
!532 = !DILocalVariable(name: "t3", scope: !518, file: !519, line: 41, type: !16)
!533 = !DILocalVariable(name: "t4", scope: !518, file: !519, line: 41, type: !16)
!534 = !DILocalVariable(name: "a1", scope: !518, file: !519, line: 41, type: !16)
!535 = !DILocalVariable(name: "a2", scope: !518, file: !519, line: 41, type: !16)
!536 = !DILocalVariable(name: "x1", scope: !518, file: !519, line: 41, type: !16)
!537 = !DILocalVariable(name: "x2", scope: !518, file: !519, line: 41, type: !16)
!538 = !DILocalVariable(name: "z", scope: !518, file: !519, line: 41, type: !16)
!539 = !DILocalVariable(name: "r", scope: !518, file: !519, line: 42, type: !16)
!540 = !{!"function_entry_count", i64 65556}
!541 = !DILocation(line: 4, column: 24, scope: !518)
!542 = !DILocation(line: 4, column: 34, scope: !518)
!543 = !DILocation(line: 36, column: 16, scope: !518)
!544 = !DILocation(line: 37, column: 16, scope: !518)
!545 = !DILocation(line: 38, column: 16, scope: !518)
!546 = !DILocation(line: 39, column: 16, scope: !518)
!547 = !DILocation(line: 47, column: 12, scope: !518)
!548 = !DILocation(line: 41, column: 10, scope: !518)
!549 = !DILocation(line: 48, column: 8, scope: !518)
!550 = !DILocation(line: 41, column: 26, scope: !518)
!551 = !DILocation(line: 49, column: 16, scope: !518)
!552 = !DILocation(line: 49, column: 10, scope: !518)
!553 = !DILocation(line: 41, column: 30, scope: !518)
!554 = !DILocation(line: 56, column: 15, scope: !518)
!555 = !DILocation(line: 56, column: 12, scope: !518)
!556 = !DILocation(line: 57, column: 8, scope: !518)
!557 = !DILocation(line: 41, column: 34, scope: !518)
!558 = !DILocation(line: 58, column: 17, scope: !518)
!559 = !DILocation(line: 58, column: 11, scope: !518)
!560 = !DILocation(line: 41, column: 38, scope: !518)
!561 = !DILocation(line: 59, column: 11, scope: !518)
!562 = !DILocation(line: 59, column: 21, scope: !518)
!563 = !DILocation(line: 59, column: 16, scope: !518)
!564 = !DILocation(line: 60, column: 19, scope: !518)
!565 = !DILocation(line: 60, column: 8, scope: !518)
!566 = !DILocation(line: 41, column: 14, scope: !518)
!567 = !DILocation(line: 61, column: 16, scope: !518)
!568 = !DILocation(line: 61, column: 10, scope: !518)
!569 = !DILocation(line: 41, column: 42, scope: !518)
!570 = !DILocation(line: 62, column: 12, scope: !518)
!571 = !DILocation(line: 62, column: 21, scope: !518)
!572 = !DILocation(line: 62, column: 16, scope: !518)
!573 = !DILocation(line: 41, column: 18, scope: !518)
!574 = !DILocation(line: 63, column: 19, scope: !518)
!575 = !DILocation(line: 63, column: 8, scope: !518)
!576 = !DILocation(line: 41, column: 22, scope: !518)
!577 = !DILocation(line: 64, column: 17, scope: !518)
!578 = !DILocation(line: 64, column: 11, scope: !518)
!579 = !DILocation(line: 64, column: 6, scope: !518)
!580 = !DILocation(line: 65, column: 11, scope: !518)
!581 = !DILocation(line: 42, column: 10, scope: !518)
!582 = !DILocation(line: 67, column: 3, scope: !518)
!583 = distinct !DISubprogram(name: "vranlc", scope: !519, file: !519, line: 71, type: !584, isLocal: false, isDefinition: true, scopeLine: 72, flags: DIFlagPrototyped, isOptimized: true, unit: !61, variables: !586)
!584 = !DISubroutineType(types: !585)
!585 = !{null, !55, !40, !16, !40}
!586 = !{!587, !588, !589, !590, !591, !592, !593, !594, !595, !596, !597, !598, !599, !600, !601, !602, !603, !604}
!587 = !DILocalVariable(name: "n", arg: 1, scope: !583, file: !519, line: 71, type: !55)
!588 = !DILocalVariable(name: "x", arg: 2, scope: !583, file: !519, line: 71, type: !40)
!589 = !DILocalVariable(name: "a", arg: 3, scope: !583, file: !519, line: 71, type: !16)
!590 = !DILocalVariable(name: "y", arg: 4, scope: !583, file: !519, line: 71, type: !40)
!591 = !DILocalVariable(name: "r23", scope: !583, file: !519, line: 103, type: !526)
!592 = !DILocalVariable(name: "r46", scope: !583, file: !519, line: 104, type: !526)
!593 = !DILocalVariable(name: "t23", scope: !583, file: !519, line: 105, type: !526)
!594 = !DILocalVariable(name: "t46", scope: !583, file: !519, line: 106, type: !526)
!595 = !DILocalVariable(name: "t1", scope: !583, file: !519, line: 108, type: !16)
!596 = !DILocalVariable(name: "t2", scope: !583, file: !519, line: 108, type: !16)
!597 = !DILocalVariable(name: "t3", scope: !583, file: !519, line: 108, type: !16)
!598 = !DILocalVariable(name: "t4", scope: !583, file: !519, line: 108, type: !16)
!599 = !DILocalVariable(name: "a1", scope: !583, file: !519, line: 108, type: !16)
!600 = !DILocalVariable(name: "a2", scope: !583, file: !519, line: 108, type: !16)
!601 = !DILocalVariable(name: "x1", scope: !583, file: !519, line: 108, type: !16)
!602 = !DILocalVariable(name: "x2", scope: !583, file: !519, line: 108, type: !16)
!603 = !DILocalVariable(name: "z", scope: !583, file: !519, line: 108, type: !16)
!604 = !DILocalVariable(name: "i", scope: !583, file: !519, line: 110, type: !55)
!605 = !{!"function_entry_count", i64 4098}
!606 = !DILocation(line: 71, column: 18, scope: !583)
!607 = !DILocation(line: 71, column: 29, scope: !583)
!608 = !DILocation(line: 71, column: 39, scope: !583)
!609 = !DILocation(line: 71, column: 49, scope: !583)
!610 = !DILocation(line: 103, column: 16, scope: !583)
!611 = !DILocation(line: 104, column: 16, scope: !583)
!612 = !DILocation(line: 105, column: 16, scope: !583)
!613 = !DILocation(line: 106, column: 16, scope: !583)
!614 = !DILocation(line: 115, column: 12, scope: !583)
!615 = !DILocation(line: 108, column: 10, scope: !583)
!616 = !DILocation(line: 116, column: 8, scope: !583)
!617 = !DILocation(line: 108, column: 26, scope: !583)
!618 = !DILocation(line: 117, column: 16, scope: !583)
!619 = !DILocation(line: 117, column: 10, scope: !583)
!620 = !DILocation(line: 108, column: 30, scope: !583)
!621 = !DILocation(line: 110, column: 7, scope: !583)
!622 = !DILocation(line: 122, column: 18, scope: !623)
!623 = distinct !DILexicalBlock(scope: !624, file: !519, line: 122, column: 3)
!624 = distinct !DILexicalBlock(scope: !583, file: !519, line: 122, column: 3)
!625 = !DILocation(line: 122, column: 3, scope: !624)
!626 = !{!"branch_weights", i32 536870913, i32 4099}
!627 = !DILocation(line: 128, column: 17, scope: !628)
!628 = distinct !DILexicalBlock(scope: !623, file: !519, line: 122, column: 29)
!629 = !DILocation(line: 128, column: 14, scope: !628)
!630 = !DILocation(line: 129, column: 10, scope: !628)
!631 = !DILocation(line: 108, column: 34, scope: !583)
!632 = !DILocation(line: 130, column: 19, scope: !628)
!633 = !DILocation(line: 130, column: 13, scope: !628)
!634 = !DILocation(line: 108, column: 38, scope: !583)
!635 = !DILocation(line: 131, column: 13, scope: !628)
!636 = !DILocation(line: 131, column: 23, scope: !628)
!637 = !DILocation(line: 131, column: 18, scope: !628)
!638 = !DILocation(line: 132, column: 21, scope: !628)
!639 = !DILocation(line: 132, column: 10, scope: !628)
!640 = !DILocation(line: 108, column: 14, scope: !583)
!641 = !DILocation(line: 133, column: 18, scope: !628)
!642 = !DILocation(line: 133, column: 12, scope: !628)
!643 = !DILocation(line: 108, column: 42, scope: !583)
!644 = !DILocation(line: 134, column: 14, scope: !628)
!645 = !DILocation(line: 134, column: 23, scope: !628)
!646 = !DILocation(line: 134, column: 18, scope: !628)
!647 = !DILocation(line: 108, column: 18, scope: !583)
!648 = !DILocation(line: 135, column: 21, scope: !628)
!649 = !DILocation(line: 135, column: 10, scope: !628)
!650 = !DILocation(line: 108, column: 22, scope: !583)
!651 = !DILocation(line: 136, column: 19, scope: !628)
!652 = !DILocation(line: 136, column: 13, scope: !628)
!653 = !DILocation(line: 136, column: 8, scope: !628)
!654 = !DILocation(line: 137, column: 16, scope: !628)
!655 = !DILocation(line: 137, column: 5, scope: !628)
!656 = !DILocation(line: 137, column: 10, scope: !628)
!657 = !DILocation(line: 122, column: 24, scope: !623)
!658 = !{!"branch_weights", i32 4099, i32 536870913}
!659 = distinct !{!659, !625, !660}
!660 = !DILocation(line: 138, column: 3, scope: !624)
!661 = !DILocation(line: 141, column: 1, scope: !583)
!662 = distinct !DISubprogram(name: "timer_clear", scope: !30, file: !30, line: 25, type: !663, isLocal: false, isDefinition: true, scopeLine: 26, flags: DIFlagPrototyped, isOptimized: true, unit: !24, variables: !665)
!663 = !DISubroutineType(types: !664)
!664 = !{null, !55}
!665 = !{!666}
!666 = !DILocalVariable(name: "n", arg: 1, scope: !662, file: !30, line: 25, type: !55)
!667 = !{!"function_entry_count", i64 3}
!668 = !DILocation(line: 25, column: 23, scope: !662)
!669 = !DILocation(line: 27, column: 5, scope: !662)
!670 = !DILocation(line: 27, column: 16, scope: !662)
!671 = !DILocation(line: 28, column: 1, scope: !662)
!672 = distinct !DISubprogram(name: "timer_start", scope: !30, file: !30, line: 34, type: !663, isLocal: false, isDefinition: true, scopeLine: 35, flags: DIFlagPrototyped, isOptimized: true, unit: !24, variables: !673)
!673 = !{!674}
!674 = !DILocalVariable(name: "n", arg: 1, scope: !672, file: !30, line: 34, type: !55)
!675 = !DILocation(line: 34, column: 23, scope: !672)
!676 = !DILocation(line: 13, column: 5, scope: !677, inlinedAt: !682)
!677 = distinct !DISubprogram(name: "elapsed_time", scope: !30, file: !30, line: 11, type: !678, isLocal: true, isDefinition: true, scopeLine: 12, flags: DIFlagPrototyped, isOptimized: true, unit: !24, variables: !680)
!678 = !DISubroutineType(types: !679)
!679 = !{!16}
!680 = !{!681}
!681 = !DILocalVariable(name: "t", scope: !677, file: !30, line: 13, type: !16)
!682 = distinct !DILocation(line: 36, column: 16, scope: !672)
!683 = !DILocation(line: 13, column: 12, scope: !677, inlinedAt: !682)
!684 = !DILocation(line: 15, column: 5, scope: !677, inlinedAt: !682)
!685 = !DILocation(line: 16, column: 13, scope: !677, inlinedAt: !682)
!686 = !DILocation(line: 17, column: 1, scope: !677, inlinedAt: !682)
!687 = !DILocation(line: 36, column: 5, scope: !672)
!688 = !DILocation(line: 36, column: 14, scope: !672)
!689 = !DILocation(line: 37, column: 1, scope: !672)
!690 = distinct !DISubprogram(name: "timer_stop", scope: !30, file: !30, line: 43, type: !663, isLocal: false, isDefinition: true, scopeLine: 44, flags: DIFlagPrototyped, isOptimized: true, unit: !24, variables: !691)
!691 = !{!692, !693, !694}
!692 = !DILocalVariable(name: "n", arg: 1, scope: !690, file: !30, line: 43, type: !55)
!693 = !DILocalVariable(name: "t", scope: !690, file: !30, line: 45, type: !16)
!694 = !DILocalVariable(name: "now", scope: !690, file: !30, line: 45, type: !16)
!695 = !DILocation(line: 43, column: 22, scope: !690)
!696 = !DILocation(line: 13, column: 5, scope: !677, inlinedAt: !697)
!697 = distinct !DILocation(line: 47, column: 11, scope: !690)
!698 = !DILocation(line: 13, column: 12, scope: !677, inlinedAt: !697)
!699 = !DILocation(line: 15, column: 5, scope: !677, inlinedAt: !697)
!700 = !DILocation(line: 16, column: 13, scope: !677, inlinedAt: !697)
!701 = !DILocation(line: 17, column: 1, scope: !677, inlinedAt: !697)
!702 = !DILocation(line: 45, column: 15, scope: !690)
!703 = !DILocation(line: 48, column: 15, scope: !690)
!704 = !DILocation(line: 48, column: 13, scope: !690)
!705 = !DILocation(line: 45, column: 12, scope: !690)
!706 = !DILocation(line: 49, column: 5, scope: !690)
!707 = !DILocation(line: 49, column: 16, scope: !690)
!708 = !DILocation(line: 51, column: 1, scope: !690)
!709 = distinct !DISubprogram(name: "timer_read", scope: !30, file: !30, line: 57, type: !710, isLocal: false, isDefinition: true, scopeLine: 58, flags: DIFlagPrototyped, isOptimized: true, unit: !24, variables: !712)
!710 = !DISubroutineType(types: !711)
!711 = !{!16, !55}
!712 = !{!713}
!713 = !DILocalVariable(name: "n", arg: 1, scope: !709, file: !30, line: 57, type: !55)
!714 = !DILocation(line: 57, column: 24, scope: !709)
!715 = !DILocation(line: 59, column: 13, scope: !709)
!716 = !DILocation(line: 59, column: 5, scope: !709)
!717 = !{!"function_entry_count", i64 2}
!718 = !DILocation(line: 7, column: 20, scope: !36)
!719 = !DILocation(line: 10, column: 3, scope: !36)
!720 = !DILocation(line: 10, column: 18, scope: !36)
!721 = !DILocation(line: 11, column: 3, scope: !36)
!722 = !DILocation(line: 12, column: 7, scope: !723)
!723 = distinct !DILexicalBlock(scope: !36, file: !37, line: 12, column: 7)
!724 = !{!725, !725, i64 0}
!725 = !{!"int", !216, i64 0}
!726 = !DILocation(line: 12, column: 11, scope: !723)
!727 = !{!728, !729, i64 0}
!728 = !{!"timeval", !729, i64 0, !729, i64 8}
!729 = !{!"long", !216, i64 0}
!730 = !DILocation(line: 12, column: 7, scope: !36)
!731 = !{!"branch_weights", i32 2, i32 2}
!732 = !DILocation(line: 12, column: 22, scope: !723)
!733 = !DILocation(line: 12, column: 20, scope: !723)
!734 = !DILocation(line: 12, column: 16, scope: !723)
!735 = !DILocation(line: 13, column: 21, scope: !36)
!736 = !DILocation(line: 13, column: 19, scope: !36)
!737 = !DILocation(line: 13, column: 8, scope: !36)
!738 = !DILocation(line: 13, column: 38, scope: !36)
!739 = !{!728, !729, i64 8}
!740 = !DILocation(line: 13, column: 35, scope: !36)
!741 = !DILocation(line: 13, column: 34, scope: !36)
!742 = !DILocation(line: 13, column: 26, scope: !36)
!743 = !DILocation(line: 13, column: 6, scope: !36)
!744 = !DILocation(line: 14, column: 1, scope: !36)
