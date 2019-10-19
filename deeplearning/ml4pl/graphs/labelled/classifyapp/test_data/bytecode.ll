; ModuleID = '/scratch/talbn/classifyapp_code/train//38/746.txt.cpp'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external global i8
@.str = private unnamed_addr constant [3 x i8] c"%d\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"%lf\00", align 1
@.str.2 = private unnamed_addr constant [7 x i8] c"%.5lf\0A\00", align 1
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_746.txt.cpp, i8* null }]

define internal void @__cxx_global_var_init() #0 section ".text.startup" {
entry:
  call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* @_ZStL8__ioinit)
  %0 = call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit, i32 0, i32 0), i8* @__dso_handle) #2
  ret void
}

declare void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) #0

; Function Attrs: nounwind
declare void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) #1

; Function Attrs: nounwind
declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) #2

; Function Attrs: uwtable
define i32 @main() #3 {
entry:
  %retval = alloca i32, align 4
  %n = alloca i32, align 4
  %k = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %a = alloca [100 x [100 x double]], align 16
  %b = alloca double, align 8
  %c = alloca double, align 8
  %e = alloca double, align 8
  %f = alloca [100 x double], align 16
  %sum = alloca double, align 8
  %d = alloca double, align 8
  store i32 0, i32* %retval
  %call = call i32 (i8*, ...) @scanf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i32* %k)
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc.41, %entry
  %0 = load i32, i32* %i, align 4
  %1 = load i32, i32* %k, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end.43

for.body:                                         ; preds = %for.cond
  store double 0.000000e+00, double* %sum, align 8
  store double 0.000000e+00, double* %d, align 8
  %call1 = call i32 (i8*, ...) @scanf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i32 0, i32 0), i32* %n)
  store i32 0, i32* %j, align 4
  br label %for.cond.2

for.cond.2:                                       ; preds = %for.inc, %for.body
  %2 = load i32, i32* %j, align 4
  %3 = load i32, i32* %n, align 4
  %cmp3 = icmp slt i32 %2, %3
  br i1 %cmp3, label %for.body.4, label %for.end

for.body.4:                                       ; preds = %for.cond.2
  %4 = load i32, i32* %j, align 4
  %idxprom = sext i32 %4 to i64
  %5 = load i32, i32* %i, align 4
  %idxprom5 = sext i32 %5 to i64
  %arrayidx = getelementptr inbounds [100 x [100 x double]], [100 x [100 x double]]* %a, i32 0, i64 %idxprom5
  %arrayidx6 = getelementptr inbounds [100 x double], [100 x double]* %arrayidx, i32 0, i64 %idxprom
  %call7 = call i32 (i8*, ...) @scanf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str.1, i32 0, i32 0), double* %arrayidx6)
  br label %for.inc

for.inc:                                          ; preds = %for.body.4
  %6 = load i32, i32* %j, align 4
  %inc = add nsw i32 %6, 1
  store i32 %inc, i32* %j, align 4
  br label %for.cond.2

for.end:                                          ; preds = %for.cond.2
  store i32 0, i32* %j, align 4
  br label %for.cond.8

for.cond.8:                                       ; preds = %for.inc.15, %for.end
  %7 = load i32, i32* %j, align 4
  %8 = load i32, i32* %n, align 4
  %cmp9 = icmp slt i32 %7, %8
  br i1 %cmp9, label %for.body.10, label %for.end.17

for.body.10:                                      ; preds = %for.cond.8
  %9 = load i32, i32* %j, align 4
  %idxprom11 = sext i32 %9 to i64
  %10 = load i32, i32* %i, align 4
  %idxprom12 = sext i32 %10 to i64
  %arrayidx13 = getelementptr inbounds [100 x [100 x double]], [100 x [100 x double]]* %a, i32 0, i64 %idxprom12
  %arrayidx14 = getelementptr inbounds [100 x double], [100 x double]* %arrayidx13, i32 0, i64 %idxprom11
  %11 = load double, double* %arrayidx14, align 8
  %12 = load double, double* %sum, align 8
  %add = fadd double %12, %11
  store double %add, double* %sum, align 8
  br label %for.inc.15

for.inc.15:                                       ; preds = %for.body.10
  %13 = load i32, i32* %j, align 4
  %inc16 = add nsw i32 %13, 1
  store i32 %inc16, i32* %j, align 4
  br label %for.cond.8

for.end.17:                                       ; preds = %for.cond.8
  %14 = load double, double* %sum, align 8
  %mul = fmul double 1.000000e+00, %14
  %15 = load i32, i32* %n, align 4
  %conv = sitofp i32 %15 to double
  %div = fdiv double %mul, %conv
  store double %div, double* %b, align 8
  store i32 0, i32* %j, align 4
  br label %for.cond.18

for.cond.18:                                      ; preds = %for.inc.32, %for.end.17
  %16 = load i32, i32* %j, align 4
  %17 = load i32, i32* %n, align 4
  %cmp19 = icmp slt i32 %16, %17
  br i1 %cmp19, label %for.body.20, label %for.end.34

for.body.20:                                      ; preds = %for.cond.18
  %18 = load i32, i32* %j, align 4
  %idxprom21 = sext i32 %18 to i64
  %19 = load i32, i32* %i, align 4
  %idxprom22 = sext i32 %19 to i64
  %arrayidx23 = getelementptr inbounds [100 x [100 x double]], [100 x [100 x double]]* %a, i32 0, i64 %idxprom22
  %arrayidx24 = getelementptr inbounds [100 x double], [100 x double]* %arrayidx23, i32 0, i64 %idxprom21
  %20 = load double, double* %arrayidx24, align 8
  %21 = load double, double* %b, align 8
  %sub = fsub double %20, %21
  %22 = load i32, i32* %j, align 4
  %idxprom25 = sext i32 %22 to i64
  %23 = load i32, i32* %i, align 4
  %idxprom26 = sext i32 %23 to i64
  %arrayidx27 = getelementptr inbounds [100 x [100 x double]], [100 x [100 x double]]* %a, i32 0, i64 %idxprom26
  %arrayidx28 = getelementptr inbounds [100 x double], [100 x double]* %arrayidx27, i32 0, i64 %idxprom25
  %24 = load double, double* %arrayidx28, align 8
  %25 = load double, double* %b, align 8
  %sub29 = fsub double %24, %25
  %mul30 = fmul double %sub, %sub29
  store double %mul30, double* %c, align 8
  %26 = load double, double* %c, align 8
  %27 = load double, double* %d, align 8
  %add31 = fadd double %27, %26
  store double %add31, double* %d, align 8
  br label %for.inc.32

for.inc.32:                                       ; preds = %for.body.20
  %28 = load i32, i32* %j, align 4
  %inc33 = add nsw i32 %28, 1
  store i32 %inc33, i32* %j, align 4
  br label %for.cond.18

for.end.34:                                       ; preds = %for.cond.18
  %29 = load double, double* %d, align 8
  %mul35 = fmul double 1.000000e+00, %29
  %30 = load i32, i32* %n, align 4
  %conv36 = sitofp i32 %30 to double
  %div37 = fdiv double %mul35, %conv36
  %call38 = call double @sqrt(double %div37) #2
  store double %call38, double* %e, align 8
  %31 = load double, double* %e, align 8
  %32 = load i32, i32* %i, align 4
  %idxprom39 = sext i32 %32 to i64
  %arrayidx40 = getelementptr inbounds [100 x double], [100 x double]* %f, i32 0, i64 %idxprom39
  store double %31, double* %arrayidx40, align 8
  br label %for.inc.41

for.inc.41:                                       ; preds = %for.end.34
  %33 = load i32, i32* %i, align 4
  %inc42 = add nsw i32 %33, 1
  store i32 %inc42, i32* %i, align 4
  br label %for.cond

for.end.43:                                       ; preds = %for.cond
  store i32 0, i32* %i, align 4
  br label %for.cond.44

for.cond.44:                                      ; preds = %for.inc.50, %for.end.43
  %34 = load i32, i32* %i, align 4
  %35 = load i32, i32* %k, align 4
  %cmp45 = icmp slt i32 %34, %35
  br i1 %cmp45, label %for.body.46, label %for.end.52

for.body.46:                                      ; preds = %for.cond.44
  %36 = load i32, i32* %i, align 4
  %idxprom47 = sext i32 %36 to i64
  %arrayidx48 = getelementptr inbounds [100 x double], [100 x double]* %f, i32 0, i64 %idxprom47
  %37 = load double, double* %arrayidx48, align 8
  %call49 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.2, i32 0, i32 0), double %37)
  br label %for.inc.50

for.inc.50:                                       ; preds = %for.body.46
  %38 = load i32, i32* %i, align 4
  %inc51 = add nsw i32 %38, 1
  store i32 %inc51, i32* %i, align 4
  br label %for.cond.44

for.end.52:                                       ; preds = %for.cond.44
  ret i32 0
}

declare i32 @scanf(i8*, ...) #0

; Function Attrs: nounwind
declare double @sqrt(double) #1

declare i32 @printf(i8*, ...) #0

define internal void @_GLOBAL__sub_I_746.txt.cpp() #0 section ".text.startup" {
entry:
  call void @__cxx_global_var_init()
  ret void
}

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="knl" "target-features"="+adx,+aes,+avx,+avx2,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+hle,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+prfchw,+rdrnd,+rdseed,+rtm,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,-avx512er,-avx512pf,-fma4,-sha,-sse4a,-tbm,-xop" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="knl" "target-features"="+adx,+aes,+avx,+avx2,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+hle,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+prfchw,+rdrnd,+rdseed,+rtm,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,-avx512er,-avx512pf,-fma4,-sha,-sse4a,-tbm,-xop" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="knl" "target-features"="+adx,+aes,+avx,+avx2,+avx512bw,+avx512cd,+avx512dq,+avx512f,+avx512vl,+bmi,+bmi2,+cmov,+cx16,+f16c,+fma,+fsgsbase,+hle,+lzcnt,+mmx,+movbe,+pclmul,+popcnt,+prfchw,+rdrnd,+rdseed,+rtm,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,-avx512er,-avx512pf,-fma4,-sha,-sse4a,-tbm,-xop" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.1 (tags/RELEASE_371/final)"}
