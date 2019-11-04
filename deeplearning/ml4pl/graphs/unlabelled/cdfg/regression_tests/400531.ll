; ModuleID = '/tmp/phd_import_hutzl4s7/file.bc'
source_filename = "file.bc"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%swift.type = type { i64 }
%swift.type_descriptor = type opaque
%swift.method_descriptor = type { i32, i32 }
%swift.method_override_descriptor = type { i32, i32, i32 }
%swift.opaque = type opaque
%T4file13KcptunProfileC = type <{ %swift.refcounted, %TSS, %TSS, %TSS, %TSb, [3 x i8], %Ts6UInt32V, %Ts6UInt32V, %Ts6UInt32V, %TSS }>
%swift.refcounted = type { %swift.type*, i64 }
%TSb = type <{ i1 }>
%Ts6UInt32V = type <{ i32 }>
%TSS = type <{ %Ts11_StringGutsV }>
%Ts11_StringGutsV = type <{ %Ts13_StringObjectV }>
%Ts13_StringObjectV = type <{ %Ts6UInt64V, %swift.bridge* }>
%Ts6UInt64V = type <{ i64 }>
%swift.bridge = type opaque
%Any = type { [24 x i8], %swift.type* }
%T10Foundation8NSObjectC = type <{ %swift.refcounted }>
%TypSg = type <{ [32 x i8] }>
%T10Foundation7NSCoderC = type opaque
%T10Foundation15NSKeyedArchiverC = type opaque
%Ts28__ContiguousArrayStorageBaseC = type opaque
%swift.protocol_conformance_descriptor = type { i32, i32, i32, i32 }
%swift.full_type = type { i8**, %swift.type }
%swift.protocol = type { i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i32, i32, i32, i32, i32, i32 }
%swift.type_metadata_record = type { i32 }
%"$s4file13KcptunProfileC4modeSSvM.Frame" = type { [24 x i8] }
%"$s4file13KcptunProfileC3keySSvM.Frame" = type { [24 x i8] }
%"$s4file13KcptunProfileC5cryptSSvM.Frame" = type { [24 x i8] }
%"$s4file13KcptunProfileC6nocompSbvM.Frame" = type { [24 x i8] }
%"$s4file13KcptunProfileC9datashards6UInt32VvM.Frame" = type { [24 x i8] }
%"$s4file13KcptunProfileC11parityshards6UInt32VvM.Frame" = type { [24 x i8] }
%"$s4file13KcptunProfileC3mtus6UInt32VvM.Frame" = type { [24 x i8] }
%"$s4file13KcptunProfileC9argumentsSSvM.Frame" = type { [24 x i8] }
%swift.metadata_response = type { %swift.type*, i64 }
%AnyObject = type { %swift.refcounted* }
%T10Foundation8NSStringC = type opaque
%T10Foundation8NSNumberC = type opaque
%Ts26DefaultStringInterpolationV = type <{ %TSS }>
%TSi = type <{ i64 }>
%T10Foundation12UserDefaultsC = type opaque
%T10Foundation12URLQueryItemV = type <{ %T10Foundation14NSURLQueryItemC* }>
%T10Foundation14NSURLQueryItemC = type opaque
%Ts16IndexingIteratorVySay10Foundation12URLQueryItemVGG = type <{ %TSa, %TSi }>
%TSa = type <{ %Ts22_ContiguousArrayBufferV }>
%Ts22_ContiguousArrayBufferV = type <{ %Ts28__ContiguousArrayStorageBaseC* }>
%T10Foundation12URLQueryItemVSg = type <{ [8 x i8] }>
%Ts6UInt32VSg = type <{ [4 x i8], [1 x i8] }>
%Ts16IndexingIteratorV = type <{}>
%TSq = type <{}>
%Ts16IndexingIteratorV.0 = type <{}>
%TSq.1 = type <{}>

@"$s4file13KcptunProfileC4modeSSvpWvd" = dso_local hidden constant i64 16, align 8
@"$s4file13KcptunProfileC3keySSvpWvd" = dso_local hidden constant i64 32, align 8
@"$s4file13KcptunProfileC5cryptSSvpWvd" = dso_local hidden constant i64 48, align 8
@"$s4file13KcptunProfileC6nocompSbvpWvd" = dso_local hidden constant i64 64, align 8
@"$s4file13KcptunProfileC9datashards6UInt32VvpWvd" = dso_local hidden constant i64 68, align 8
@"$s4file13KcptunProfileC11parityshards6UInt32VvpWvd" = dso_local hidden constant i64 72, align 8
@"$s4file13KcptunProfileC3mtus6UInt32VvpWvd" = dso_local hidden constant i64 76, align 8
@"$s4file13KcptunProfileC9argumentsSSvpWvd" = dso_local hidden constant i64 80, align 8
@"$sBoWV" = external global i8*, align 8
@"$s10Foundation8NSObjectCN" = external global %swift.type, align 8
@0 = private dso_local constant [5 x i8] c"file\00"
@"$s4fileMXM" = linkonce_odr dso_local hidden constant <{ i32, i32, i32 }> <{ i32 0, i32 0, i32 trunc (i64 sub (i64 ptrtoint ([5 x i8]* @0 to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32 }>, <{ i32, i32, i32 }>* @"$s4fileMXM", i32 0, i32 2) to i64)) to i32) }>, section ".rodata", align 4
@1 = private dso_local constant [14 x i8] c"KcptunProfile\00"
@"symbolic 10Foundation8NSObjectC" = linkonce_odr dso_local hidden constant <{ [22 x i8], i8 }> <{ [22 x i8] c"10Foundation8NSObjectC", i8 0 }>, section "swift5_typeref", align 2
@"$s10Foundation8NSObjectCMn" = external global %swift.type_descriptor, align 4
@"got.$s10Foundation8NSObjectCMn" = private dso_local unnamed_addr constant %swift.type_descriptor* @"$s10Foundation8NSObjectCMn"
@"$s10Foundation8NSObjectCACycfCTq" = external global %swift.method_descriptor, align 4
@"got.$s10Foundation8NSObjectCACycfCTq" = private dso_local unnamed_addr constant %swift.method_descriptor* @"$s10Foundation8NSObjectCACycfCTq"
@"$s4file13KcptunProfileCMn" = dso_local hidden constant <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }> <{ i32 -1073741744, i32 trunc (i64 sub (i64 ptrtoint (<{ i32, i32, i32 }>* @"$s4fileMXM" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 1) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint ([14 x i8]* @1 to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (%swift.metadata_response (i64)* @"$s4file13KcptunProfileCMa" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 3) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 4) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (<{ [22 x i8], i8 }>* @"symbolic 10Foundation8NSObjectC" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 5) to i64)) to i32), i32 2, i32 68, i32 37, i32 8, i32 31, i32 39, i32 29, %swift.method_descriptor { i32 18, i32 trunc (i64 sub (i64 ptrtoint ({ i64, %swift.bridge* } (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC4modeSSvg" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 13, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 19, i32 trunc (i64 sub (i64 ptrtoint (void (i64, %swift.bridge*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC4modeSSvs" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 14, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 20, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC4modeSSvM" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 15, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 18, i32 trunc (i64 sub (i64 ptrtoint ({ i64, %swift.bridge* } (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC3keySSvg" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 16, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 19, i32 trunc (i64 sub (i64 ptrtoint (void (i64, %swift.bridge*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC3keySSvs" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 17, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 20, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC3keySSvM" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 18, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 18, i32 trunc (i64 sub (i64 ptrtoint ({ i64, %swift.bridge* } (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC5cryptSSvg" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 19, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 19, i32 trunc (i64 sub (i64 ptrtoint (void (i64, %swift.bridge*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC5cryptSSvs" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 20, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 20, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC5cryptSSvM" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 21, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 18, i32 trunc (i64 sub (i64 ptrtoint (i1 (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC6nocompSbvg" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 22, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 19, i32 trunc (i64 sub (i64 ptrtoint (void (i1, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC6nocompSbvs" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 23, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 20, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, %TSb* } (i8*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC6nocompSbvM" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 24, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 18, i32 trunc (i64 sub (i64 ptrtoint (i32 (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC9datashards6UInt32Vvg" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 25, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 19, i32 trunc (i64 sub (i64 ptrtoint (void (i32, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC9datashards6UInt32Vvs" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 26, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 20, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC9datashards6UInt32VvM" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 27, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 18, i32 trunc (i64 sub (i64 ptrtoint (i32 (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC11parityshards6UInt32Vvg" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 28, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 19, i32 trunc (i64 sub (i64 ptrtoint (void (i32, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC11parityshards6UInt32Vvs" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 29, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 20, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC11parityshards6UInt32VvM" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 30, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 18, i32 trunc (i64 sub (i64 ptrtoint (i32 (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC3mtus6UInt32Vvg" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 31, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 19, i32 trunc (i64 sub (i64 ptrtoint (void (i32, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC3mtus6UInt32Vvs" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 32, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 20, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC3mtus6UInt32VvM" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 33, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 18, i32 trunc (i64 sub (i64 ptrtoint ({ i64, %swift.bridge* } (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC9argumentsSSvg" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 34, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 19, i32 trunc (i64 sub (i64 ptrtoint (void (i64, %swift.bridge*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC9argumentsSSvs" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 35, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 20, i32 trunc (i64 sub (i64 ptrtoint ({ i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC9argumentsSSvM" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 36, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 16, i32 trunc (i64 sub (i64 ptrtoint (void (%Any*, i8, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC4copy4withyp10Foundation6NSZoneVSg_tF" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 37, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 16, i32 trunc (i64 sub (i64 ptrtoint (%swift.bridge* (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC12toDictionarySDySSyXlGyF" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 38, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 16, i32 trunc (i64 sub (i64 ptrtoint (%swift.bridge* (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC12toJsonConfigSDySSyXlGyF" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 39, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 16, i32 trunc (i64 sub (i64 ptrtoint (%Ts28__ContiguousArrayStorageBaseC* (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC13urlQueryItemsSay10Foundation12URLQueryItemVGyF" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 40, i32 1) to i64)) to i32) }, %swift.method_descriptor { i32 16, i32 trunc (i64 sub (i64 ptrtoint (void (%Ts28__ContiguousArrayStorageBaseC*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC17loadUrlQueryItems5itemsySay10Foundation12URLQueryItemVG_tF" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 41, i32 1) to i64)) to i32) }, i32 1, %swift.method_override_descriptor { i32 add (i32 trunc (i64 sub (i64 ptrtoint (%swift.type_descriptor** @"got.$s10Foundation8NSObjectCMn" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 43, i32 0) to i64)) to i32), i32 1), i32 add (i32 trunc (i64 sub (i64 ptrtoint (%swift.method_descriptor** @"got.$s10Foundation8NSObjectCACycfCTq" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 43, i32 1) to i64)) to i32), i32 1), i32 trunc (i64 sub (i64 ptrtoint (%T4file13KcptunProfileC* (%swift.type*)* @"$s4file13KcptunProfileCACycfC" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 43, i32 2) to i64)) to i32) } }>, section ".rodata", align 4
@"$s4file13KcptunProfileCML" = internal dso_local global %swift.type* null, align 8
@"$s4file13KcptunProfileCMf" = internal dso_local global <{ void (%T4file13KcptunProfileC*)*, i8**, i64, %swift.type*, %swift.opaque*, %swift.opaque*, i64, i32, i32, i32, i16, i16, i32, i32, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>*, void (%T4file13KcptunProfileC*)*, %T4file13KcptunProfileC* (%swift.type*)*, void (%Any*, %T10Foundation8NSObjectC*)*, void (%Any*, %T10Foundation8NSObjectC*)*, i1 (%TypSg*, %T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, %T10Foundation8NSObjectC* (%T10Foundation8NSObjectC*)*, i1 (%T10Foundation8NSObjectC*)*, { i64, %swift.bridge* } (%T10Foundation8NSObjectC*)*, { i64, %swift.bridge* } (%T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, %swift.type* (%T10Foundation8NSObjectC*)*, void (%TypSg*, %T10Foundation7NSCoderC*, %T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, void (%TypSg*, %T10Foundation15NSKeyedArchiverC*, %T10Foundation8NSObjectC*)*, %Ts28__ContiguousArrayStorageBaseC* (%swift.type*)*, %swift.type* (%swift.type*)*, i64 (%swift.type*)*, i64 (%swift.type*)*, i1 (%swift.type*, %swift.type*)*, i1 (%swift.type*, %T10Foundation8NSObjectC*)*, i1 (%swift.type*, %T10Foundation8NSObjectC*)*, i64, i64, i64, i64, i64, i64, i64, i64, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, i1 (%T4file13KcptunProfileC*)*, void (i1, %T4file13KcptunProfileC*)*, { i8*, %TSb* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, void (%Any*, i8, %T4file13KcptunProfileC*)*, %swift.bridge* (%T4file13KcptunProfileC*)*, %swift.bridge* (%T4file13KcptunProfileC*)*, %Ts28__ContiguousArrayStorageBaseC* (%T4file13KcptunProfileC*)*, void (%Ts28__ContiguousArrayStorageBaseC*, %T4file13KcptunProfileC*)* }> <{ void (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileCfD", i8** @"$sBoWV", i64 0, %swift.type* @"$s10Foundation8NSObjectCN", %swift.opaque* null, %swift.opaque* null, i64 1, i32 2, i32 0, i32 96, i16 7, i16 0, i32 560, i32 16, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", void (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileCfE", %T4file13KcptunProfileC* (%swift.type*)* @"$s4file13KcptunProfileCACycfC", void (%Any*, %T10Foundation8NSObjectC*)* @"$s10Foundation8NSObjectC4copyypyF", void (%Any*, %T10Foundation8NSObjectC*)* @"$s10Foundation8NSObjectC11mutableCopyypyF", i1 (%TypSg*, %T10Foundation8NSObjectC*)* @"$s10Foundation8NSObjectC7isEqualySbypSgF", i64 (%T10Foundation8NSObjectC*)* @"$s10Foundation8NSObjectC4hashSivg", %T10Foundation8NSObjectC* (%T10Foundation8NSObjectC*)* @"$s10Foundation8NSObjectC4selfACXDyF", i1 (%T10Foundation8NSObjectC*)* @"$s10Foundation8NSObjectC7isProxySbyF", { i64, %swift.bridge* } (%T10Foundation8NSObjectC*)* @"$s10Foundation8NSObjectC11descriptionSSvg", { i64, %swift.bridge* } (%T10Foundation8NSObjectC*)* @"$s10Foundation8NSObjectC16debugDescriptionSSvg", i64 (%T10Foundation8NSObjectC*)* @"$s10Foundation8NSObjectC9_cfTypeIDSuvg", %swift.type* (%T10Foundation8NSObjectC*)* @"$s10Foundation8NSObjectC13classForCoderyXlXpvg", void (%TypSg*, %T10Foundation7NSCoderC*, %T10Foundation8NSObjectC*)* @"$s10Foundation8NSObjectC17replacementObject3forypSgAA7NSCoderC_tF", i64 (%T10Foundation8NSObjectC*)* @"$s10Foundation8NSObjectC21classForKeyedArchiveryXlXpSgvg", void (%TypSg*, %T10Foundation15NSKeyedArchiverC*, %T10Foundation8NSObjectC*)* @"$s10Foundation8NSObjectC17replacementObject3forypSgAA15NSKeyedArchiverC_tF", %Ts28__ContiguousArrayStorageBaseC* (%swift.type*)* @"$s10Foundation8NSObjectC30classFallbacksForKeyedArchiverSaySSGyFZ", %swift.type* (%swift.type*)* @"$s10Foundation8NSObjectC23classForKeyedUnarchiveryXlXpyFZ", i64 (%swift.type*)* @"$s10Foundation8NSObjectC18nsObjectSuperclass33_6DA0945A07226B3278459E9368612FF4LLACmSgvgZ", i64 (%swift.type*)* @"$s10Foundation8NSObjectC10superclassyXlXpSgvgZ", i1 (%swift.type*, %swift.type*)* @"$s10Foundation8NSObjectC10isSubclass2ofSbyXlXp_tFZ", i1 (%swift.type*, %T10Foundation8NSObjectC*)* @"$s10Foundation8NSObjectC8isMember2ofSbyXlXp_tF", i1 (%swift.type*, %T10Foundation8NSObjectC*)* @"$s10Foundation8NSObjectC6isKind2ofSbyXlXp_tF", i64 16, i64 32, i64 48, i64 64, i64 68, i64 72, i64 76, i64 80, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC4modeSSvg", void (i64, %swift.bridge*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC4modeSSvs", { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC4modeSSvM", { i64, %swift.bridge* } (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC3keySSvg", void (i64, %swift.bridge*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC3keySSvs", { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC3keySSvM", { i64, %swift.bridge* } (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC5cryptSSvg", void (i64, %swift.bridge*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC5cryptSSvs", { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC5cryptSSvM", i1 (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC6nocompSbvg", void (i1, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC6nocompSbvs", { i8*, %TSb* } (i8*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC6nocompSbvM", i32 (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC9datashards6UInt32Vvg", void (i32, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC9datashards6UInt32Vvs", { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC9datashards6UInt32VvM", i32 (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC11parityshards6UInt32Vvg", void (i32, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC11parityshards6UInt32Vvs", { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC11parityshards6UInt32VvM", i32 (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC3mtus6UInt32Vvg", void (i32, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC3mtus6UInt32Vvs", { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC3mtus6UInt32VvM", { i64, %swift.bridge* } (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC9argumentsSSvg", void (i64, %swift.bridge*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC9argumentsSSvs", { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC9argumentsSSvM", void (%Any*, i8, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC4copy4withyp10Foundation6NSZoneVSg_tF", %swift.bridge* (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC12toDictionarySDySSyXlGyF", %swift.bridge* (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC12toJsonConfigSDySSyXlGyF", %Ts28__ContiguousArrayStorageBaseC* (%T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC13urlQueryItemsSay10Foundation12URLQueryItemVGyF", void (%Ts28__ContiguousArrayStorageBaseC*, %T4file13KcptunProfileC*)* @"$s4file13KcptunProfileC17loadUrlQueryItems5itemsySay10Foundation12URLQueryItemVG_tF" }>, align 8
@"symbolic _____ 4file13KcptunProfileC" = linkonce_odr dso_local hidden constant <{ i8, i32, i8 }> <{ i8 1, i32 trunc (i64 sub (i64 ptrtoint (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn" to i64), i64 ptrtoint (i32* getelementptr inbounds (<{ i8, i32, i8 }>, <{ i8, i32, i8 }>* @"symbolic _____ 4file13KcptunProfileC", i32 0, i32 1) to i64)) to i32), i8 0 }>, section "swift5_typeref", align 2
@"symbolic SS" = linkonce_odr dso_local hidden constant <{ [2 x i8], i8 }> <{ [2 x i8] c"SS", i8 0 }>, section "swift5_typeref", align 2
@2 = private dso_local constant [5 x i8] c"mode\00", section "swift5_reflstr"
@3 = private dso_local constant [4 x i8] c"key\00", section "swift5_reflstr"
@4 = private dso_local constant [6 x i8] c"crypt\00", section "swift5_reflstr"
@"symbolic Sb" = linkonce_odr dso_local hidden constant <{ [2 x i8], i8 }> <{ [2 x i8] c"Sb", i8 0 }>, section "swift5_typeref", align 2
@5 = private dso_local constant [7 x i8] c"nocomp\00", section "swift5_reflstr"
@"symbolic s6UInt32V" = linkonce_odr dso_local hidden constant <{ [9 x i8], i8 }> <{ [9 x i8] c"s6UInt32V", i8 0 }>, section "swift5_typeref", align 2
@6 = private dso_local constant [10 x i8] c"datashard\00", section "swift5_reflstr"
@7 = private dso_local constant [12 x i8] c"parityshard\00", section "swift5_reflstr"
@8 = private dso_local constant [4 x i8] c"mtu\00", section "swift5_reflstr"
@9 = private dso_local constant [10 x i8] c"arguments\00", section "swift5_reflstr"
@"$s4file13KcptunProfileCMF" = internal dso_local constant { i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } { i32 trunc (i64 sub (i64 ptrtoint (<{ i8, i32, i8 }>* @"symbolic _____ 4file13KcptunProfileC" to i64), i64 ptrtoint ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF" to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (<{ [22 x i8], i8 }>* @"symbolic 10Foundation8NSObjectC" to i64), i64 ptrtoint (i32* getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF", i32 0, i32 1) to i64)) to i32), i16 1, i16 12, i32 8, i32 2, i32 trunc (i64 sub (i64 ptrtoint (<{ [2 x i8], i8 }>* @"symbolic SS" to i64), i64 ptrtoint (i32* getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF", i32 0, i32 6) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint ([5 x i8]* @2 to i64), i64 ptrtoint (i32* getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF", i32 0, i32 7) to i64)) to i32), i32 2, i32 trunc (i64 sub (i64 ptrtoint (<{ [2 x i8], i8 }>* @"symbolic SS" to i64), i64 ptrtoint (i32* getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF", i32 0, i32 9) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint ([4 x i8]* @3 to i64), i64 ptrtoint (i32* getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF", i32 0, i32 10) to i64)) to i32), i32 2, i32 trunc (i64 sub (i64 ptrtoint (<{ [2 x i8], i8 }>* @"symbolic SS" to i64), i64 ptrtoint (i32* getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF", i32 0, i32 12) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint ([6 x i8]* @4 to i64), i64 ptrtoint (i32* getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF", i32 0, i32 13) to i64)) to i32), i32 2, i32 trunc (i64 sub (i64 ptrtoint (<{ [2 x i8], i8 }>* @"symbolic Sb" to i64), i64 ptrtoint (i32* getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF", i32 0, i32 15) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint ([7 x i8]* @5 to i64), i64 ptrtoint (i32* getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF", i32 0, i32 16) to i64)) to i32), i32 2, i32 trunc (i64 sub (i64 ptrtoint (<{ [9 x i8], i8 }>* @"symbolic s6UInt32V" to i64), i64 ptrtoint (i32* getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF", i32 0, i32 18) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint ([10 x i8]* @6 to i64), i64 ptrtoint (i32* getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF", i32 0, i32 19) to i64)) to i32), i32 2, i32 trunc (i64 sub (i64 ptrtoint (<{ [9 x i8], i8 }>* @"symbolic s6UInt32V" to i64), i64 ptrtoint (i32* getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF", i32 0, i32 21) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint ([12 x i8]* @7 to i64), i64 ptrtoint (i32* getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF", i32 0, i32 22) to i64)) to i32), i32 2, i32 trunc (i64 sub (i64 ptrtoint (<{ [9 x i8], i8 }>* @"symbolic s6UInt32V" to i64), i64 ptrtoint (i32* getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF", i32 0, i32 24) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint ([4 x i8]* @8 to i64), i64 ptrtoint (i32* getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF", i32 0, i32 25) to i64)) to i32), i32 2, i32 trunc (i64 sub (i64 ptrtoint (<{ [2 x i8], i8 }>* @"symbolic SS" to i64), i64 ptrtoint (i32* getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF", i32 0, i32 27) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint ([10 x i8]* @9 to i64), i64 ptrtoint (i32* getelementptr inbounds ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }, { i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF", i32 0, i32 28) to i64)) to i32) }, section "swift5_fieldmd", align 4
@"_swift_FORCE_LOAD_$_swiftGlibc_$_file" = weak_odr dso_local hidden constant void ()* @"_swift_FORCE_LOAD_$_swiftGlibc"
@"$s4file13KcptunProfileC10Foundation9NSCopyingAAMc" = dso_local hidden constant %swift.protocol_conformance_descriptor { i32 add (i32 trunc (i64 sub (i64 ptrtoint (%swift.protocol** @"got.$s10Foundation9NSCopyingMp" to i64), i64 ptrtoint (%swift.protocol_conformance_descriptor* @"$s4file13KcptunProfileC10Foundation9NSCopyingAAMc" to i64)) to i32), i32 1), i32 trunc (i64 sub (i64 ptrtoint (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn" to i64), i64 ptrtoint (i32* getelementptr inbounds (%swift.protocol_conformance_descriptor, %swift.protocol_conformance_descriptor* @"$s4file13KcptunProfileC10Foundation9NSCopyingAAMc", i32 0, i32 1) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint ([2 x i8*]* @"$s4file13KcptunProfileC10Foundation9NSCopyingAAWP" to i64), i64 ptrtoint (i32* getelementptr inbounds (%swift.protocol_conformance_descriptor, %swift.protocol_conformance_descriptor* @"$s4file13KcptunProfileC10Foundation9NSCopyingAAMc", i32 0, i32 2) to i64)) to i32), i32 0 }, section ".rodata", align 4
@"$s4file13KcptunProfileC10Foundation9NSCopyingAAWP" = dso_local hidden constant [2 x i8*] [i8* bitcast (%swift.protocol_conformance_descriptor* @"$s4file13KcptunProfileC10Foundation9NSCopyingAAMc" to i8*), i8* bitcast (void (%Any*, i8, %T4file13KcptunProfileC**, %swift.type*, i8**)* @"$s4file13KcptunProfileC10Foundation9NSCopyingAadEP4copy4withypAD6NSZoneVSg_tFTW" to i8*)], align 8
@10 = private dso_local unnamed_addr constant [5 x i8] c"fast\00"
@11 = private dso_local unnamed_addr constant [15 x i8] c"it's a secrect\00"
@12 = private dso_local unnamed_addr constant [4 x i8] c"aes\00"
@13 = private dso_local unnamed_addr constant [1 x i8] zeroinitializer
@"$sSay10Foundation12URLQueryItemVGML" = linkonce_odr dso_local hidden global %swift.type* null, align 8
@"$s10Foundation12URLQueryItemVN" = external global %swift.type, align 8
@"$sSay10Foundation12URLQueryItemVGSayxGSlsWL" = linkonce_odr dso_local hidden global i8** null, align 8
@"$sSayxGSlsMc" = external global %swift.protocol_conformance_descriptor, align 4
@"$ss16IndexingIteratorVySay10Foundation12URLQueryItemVGGML" = linkonce_odr dso_local hidden global %swift.type* null, align 8
@14 = private dso_local unnamed_addr constant [5 x i8] c"mode\00"
@15 = private dso_local unnamed_addr constant [4 x i8] c"key\00"
@16 = private dso_local unnamed_addr constant [6 x i8] c"crypt\00"
@17 = private dso_local unnamed_addr constant [10 x i8] c"datashard\00"
@18 = private dso_local unnamed_addr constant [12 x i8] c"parityshard\00"
@19 = private dso_local unnamed_addr constant [7 x i8] c"nocomp\00"
@20 = private dso_local unnamed_addr constant [4 x i8] c"mtu\00"
@21 = private dso_local unnamed_addr constant [10 x i8] c"arguments\00"
@"$ss6UInt32VN" = external global %swift.type, align 8
@"$ss6UInt32VABs17FixedWidthIntegersWL" = linkonce_odr dso_local hidden global i8** null, align 8
@"$ss6UInt32Vs17FixedWidthIntegersMc" = external global %swift.protocol_conformance_descriptor, align 4
@"$ss6UInt32Vs23CustomStringConvertiblesWP" = external global i8*, align 8
@22 = private dso_local unnamed_addr constant [17 x i8] c"Kcptun.LocalHost\00"
@23 = private dso_local unnamed_addr constant [11 x i8] c"file.swift\00"
@24 = private dso_local unnamed_addr constant [12 x i8] c"Fatal error\00"
@25 = private dso_local unnamed_addr constant [58 x i8] c"Unexpectedly found nil while unwrapping an Optional value\00"
@26 = private dso_local unnamed_addr constant [17 x i8] c"Kcptun.LocalPort\00"
@27 = private dso_local unnamed_addr constant [12 x i8] c"Kcptun.Conn\00"
@"$sSS_yXltML" = linkonce_odr dso_local hidden global %swift.type* null, align 8
@"$sSSN" = external global %swift.type, align 8
@"$syXlN" = external global %swift.full_type
@28 = private dso_local unnamed_addr constant [10 x i8] c"localaddr\00"
@"$sSSs23CustomStringConvertiblesWP" = external global i8*, align 8
@"$sSSs20TextOutputStreamablesWP" = external global i8*, align 8
@29 = private dso_local unnamed_addr constant [2 x i8] c":\00"
@"$sSiN" = external global %swift.type, align 8
@"$sSis23CustomStringConvertiblesWP" = external global i8*, align 8
@30 = private dso_local unnamed_addr constant [5 x i8] c"conn\00"
@"$sSSSHsWP" = external global i8*, align 8
@"$s10Foundation9NSCopyingMp" = external global %swift.protocol, align 4
@"got.$s10Foundation9NSCopyingMp" = private dso_local unnamed_addr constant %swift.protocol* @"$s10Foundation9NSCopyingMp"
@"\01l_protocol_conformances" = private dso_local constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (%swift.protocol_conformance_descriptor* @"$s4file13KcptunProfileC10Foundation9NSCopyingAAMc" to i64), i64 ptrtoint ([1 x i32]* @"\01l_protocol_conformances" to i64)) to i32)], section "swift5_protocol_conformances", align 4
@"\01l_type_metadata_table" = private dso_local constant [1 x %swift.type_metadata_record] [%swift.type_metadata_record { i32 trunc (i64 sub (i64 ptrtoint (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn" to i64), i64 ptrtoint ([1 x %swift.type_metadata_record]* @"\01l_type_metadata_table" to i64)) to i32) }], section "swift5_type_metadata", align 4
@__swift_reflection_version = linkonce_odr dso_local hidden constant i16 3
@_swift1_autolink_entries = private dso_local constant [132 x i8] c"-lFoundation\00-lswiftCore\00-lswiftGlibc\00-lpthread\00-lutil\00-ldl\00-lm\00-lswiftDispatch\00-ldispatch\00-lBlocksRuntime\00-lswiftSwiftOnoneSupport\00", section ".swift1_autolink_entries", align 8
@llvm.used = appending global [6 x i8*] [i8* bitcast ({ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF" to i8*), i8* bitcast (void ()** @"_swift_FORCE_LOAD_$_swiftGlibc_$_file" to i8*), i8* bitcast ([1 x i32]* @"\01l_protocol_conformances" to i8*), i8* bitcast ([1 x %swift.type_metadata_record]* @"\01l_type_metadata_table" to i8*), i8* bitcast (i16* @__swift_reflection_version to i8*), i8* getelementptr inbounds ([132 x i8], [132 x i8]* @_swift1_autolink_entries, i32 0, i32 0)], section "llvm.metadata", align 8

@"$s4file13KcptunProfileC4modeSSvgTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 13)
@"$s4file13KcptunProfileC4modeSSvsTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 14)
@"$s4file13KcptunProfileC4modeSSvMTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 15)
@"$s4file13KcptunProfileC3keySSvgTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 16)
@"$s4file13KcptunProfileC3keySSvsTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 17)
@"$s4file13KcptunProfileC3keySSvMTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 18)
@"$s4file13KcptunProfileC5cryptSSvgTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 19)
@"$s4file13KcptunProfileC5cryptSSvsTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 20)
@"$s4file13KcptunProfileC5cryptSSvMTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 21)
@"$s4file13KcptunProfileC6nocompSbvgTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 22)
@"$s4file13KcptunProfileC6nocompSbvsTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 23)
@"$s4file13KcptunProfileC6nocompSbvMTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 24)
@"$s4file13KcptunProfileC9datashards6UInt32VvgTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 25)
@"$s4file13KcptunProfileC9datashards6UInt32VvsTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 26)
@"$s4file13KcptunProfileC9datashards6UInt32VvMTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 27)
@"$s4file13KcptunProfileC11parityshards6UInt32VvgTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 28)
@"$s4file13KcptunProfileC11parityshards6UInt32VvsTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 29)
@"$s4file13KcptunProfileC11parityshards6UInt32VvMTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 30)
@"$s4file13KcptunProfileC3mtus6UInt32VvgTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 31)
@"$s4file13KcptunProfileC3mtus6UInt32VvsTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 32)
@"$s4file13KcptunProfileC3mtus6UInt32VvMTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 33)
@"$s4file13KcptunProfileC9argumentsSSvgTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 34)
@"$s4file13KcptunProfileC9argumentsSSvsTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 35)
@"$s4file13KcptunProfileC9argumentsSSvMTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 36)
@"$s4file13KcptunProfileC4copy4withyp10Foundation6NSZoneVSg_tFTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 37)
@"$s4file13KcptunProfileC12toDictionarySDySSyXlGyFTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 38)
@"$s4file13KcptunProfileC12toJsonConfigSDySSyXlGyFTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 39)
@"$s4file13KcptunProfileC13urlQueryItemsSay10Foundation12URLQueryItemVGyFTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 40)
@"$s4file13KcptunProfileC17loadUrlQueryItems5itemsySay10Foundation12URLQueryItemVG_tFTq" = dso_local hidden alias %swift.method_descriptor, getelementptr inbounds (<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", i32 0, i32 41)
@"$s4file13KcptunProfileCN" = dso_local hidden alias %swift.type, bitcast (i64* getelementptr inbounds (<{ void (%T4file13KcptunProfileC*)*, i8**, i64, %swift.type*, %swift.opaque*, %swift.opaque*, i64, i32, i32, i32, i16, i16, i32, i32, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>*, void (%T4file13KcptunProfileC*)*, %T4file13KcptunProfileC* (%swift.type*)*, void (%Any*, %T10Foundation8NSObjectC*)*, void (%Any*, %T10Foundation8NSObjectC*)*, i1 (%TypSg*, %T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, %T10Foundation8NSObjectC* (%T10Foundation8NSObjectC*)*, i1 (%T10Foundation8NSObjectC*)*, { i64, %swift.bridge* } (%T10Foundation8NSObjectC*)*, { i64, %swift.bridge* } (%T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, %swift.type* (%T10Foundation8NSObjectC*)*, void (%TypSg*, %T10Foundation7NSCoderC*, %T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, void (%TypSg*, %T10Foundation15NSKeyedArchiverC*, %T10Foundation8NSObjectC*)*, %Ts28__ContiguousArrayStorageBaseC* (%swift.type*)*, %swift.type* (%swift.type*)*, i64 (%swift.type*)*, i64 (%swift.type*)*, i1 (%swift.type*, %swift.type*)*, i1 (%swift.type*, %T10Foundation8NSObjectC*)*, i1 (%swift.type*, %T10Foundation8NSObjectC*)*, i64, i64, i64, i64, i64, i64, i64, i64, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, i1 (%T4file13KcptunProfileC*)*, void (i1, %T4file13KcptunProfileC*)*, { i8*, %TSb* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, void (%Any*, i8, %T4file13KcptunProfileC*)*, %swift.bridge* (%T4file13KcptunProfileC*)*, %swift.bridge* (%T4file13KcptunProfileC*)*, %Ts28__ContiguousArrayStorageBaseC* (%T4file13KcptunProfileC*)*, void (%Ts28__ContiguousArrayStorageBaseC*, %T4file13KcptunProfileC*)* }>, <{ void (%T4file13KcptunProfileC*)*, i8**, i64, %swift.type*, %swift.opaque*, %swift.opaque*, i64, i32, i32, i32, i16, i16, i32, i32, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>*, void (%T4file13KcptunProfileC*)*, %T4file13KcptunProfileC* (%swift.type*)*, void (%Any*, %T10Foundation8NSObjectC*)*, void (%Any*, %T10Foundation8NSObjectC*)*, i1 (%TypSg*, %T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, %T10Foundation8NSObjectC* (%T10Foundation8NSObjectC*)*, i1 (%T10Foundation8NSObjectC*)*, { i64, %swift.bridge* } (%T10Foundation8NSObjectC*)*, { i64, %swift.bridge* } (%T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, %swift.type* (%T10Foundation8NSObjectC*)*, void (%TypSg*, %T10Foundation7NSCoderC*, %T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, void (%TypSg*, %T10Foundation15NSKeyedArchiverC*, %T10Foundation8NSObjectC*)*, %Ts28__ContiguousArrayStorageBaseC* (%swift.type*)*, %swift.type* (%swift.type*)*, i64 (%swift.type*)*, i64 (%swift.type*)*, i1 (%swift.type*, %swift.type*)*, i1 (%swift.type*, %T10Foundation8NSObjectC*)*, i1 (%swift.type*, %T10Foundation8NSObjectC*)*, i64, i64, i64, i64, i64, i64, i64, i64, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, i1 (%T4file13KcptunProfileC*)*, void (i1, %T4file13KcptunProfileC*)*, { i8*, %TSb* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, void (%Any*, i8, %T4file13KcptunProfileC*)*, %swift.bridge* (%T4file13KcptunProfileC*)*, %swift.bridge* (%T4file13KcptunProfileC*)*, %Ts28__ContiguousArrayStorageBaseC* (%T4file13KcptunProfileC*)*, void (%Ts28__ContiguousArrayStorageBaseC*, %T4file13KcptunProfileC*)* }>* @"$s4file13KcptunProfileCMf", i32 0, i32 2) to %swift.type*)

define dso_local protected i32 @main(i32, i8**) #0 {
  %3 = bitcast i8** %1 to i8*
  ret i32 0
}

define dso_local hidden swiftcc { i64, %swift.bridge* } @"$s4file13KcptunProfileC4modeSSvg"(%T4file13KcptunProfileC* swiftself) #0 {
  %2 = alloca [24 x i8], align 8
  %3 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 1
  %4 = bitcast [24 x i8]* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %4)
  %5 = bitcast %TSS* %3 to i8*
  call void @swift_beginAccess(i8* %5, [24 x i8]* %2, i64 32, i8* null) #6
  %6 = getelementptr inbounds %TSS, %TSS* %3, i32 0, i32 0
  %7 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %6, i32 0, i32 0
  %8 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %7, i32 0, i32 0
  %9 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %8, i32 0, i32 0
  %10 = load i64, i64* %9, align 8
  %11 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %7, i32 0, i32 1
  %12 = load %swift.bridge*, %swift.bridge** %11, align 8
  %13 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %12) #6
  call void @swift_endAccess([24 x i8]* %2) #6
  %14 = bitcast [24 x i8]* %2 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %14)
  %15 = insertvalue { i64, %swift.bridge* } undef, i64 %10, 0
  %16 = insertvalue { i64, %swift.bridge* } %15, %swift.bridge* %12, 1
  ret { i64, %swift.bridge* } %16
}

define dso_local hidden swiftcc void @"$s4file13KcptunProfileC4modeSSvs"(i64, %swift.bridge*, %T4file13KcptunProfileC* swiftself) #0 {
  %4 = alloca [24 x i8], align 8
  %5 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %1) #6
  %6 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %2, i32 0, i32 1
  %7 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %7)
  %8 = bitcast %TSS* %6 to i8*
  call void @swift_beginAccess(i8* %8, [24 x i8]* %4, i64 33, i8* null) #6
  %9 = getelementptr inbounds %TSS, %TSS* %6, i32 0, i32 0
  %10 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %9, i32 0, i32 0
  %11 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %10, i32 0, i32 0
  %12 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %11, i32 0, i32 0
  %13 = load i64, i64* %12, align 8
  %14 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %10, i32 0, i32 1
  %15 = load %swift.bridge*, %swift.bridge** %14, align 8
  %16 = getelementptr inbounds %TSS, %TSS* %6, i32 0, i32 0
  %17 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %16, i32 0, i32 0
  %18 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %17, i32 0, i32 0
  %19 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %18, i32 0, i32 0
  store i64 %0, i64* %19, align 8
  %20 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %17, i32 0, i32 1
  store %swift.bridge* %1, %swift.bridge** %20, align 8
  call void @swift_bridgeObjectRelease(%swift.bridge* %15) #6
  call void @swift_endAccess([24 x i8]* %4) #6
  %21 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %21)
  call void @swift_bridgeObjectRelease(%swift.bridge* %1) #6
  ret void
}

; Function Attrs: noinline
define dso_local hidden swiftcc { i8*, %TSS* } @"$s4file13KcptunProfileC4modeSSvM"(i8* noalias dereferenceable(32), %T4file13KcptunProfileC* swiftself) #1 {
  %3 = bitcast i8* %0 to %"$s4file13KcptunProfileC4modeSSvM.Frame"*
  %4 = getelementptr inbounds %"$s4file13KcptunProfileC4modeSSvM.Frame", %"$s4file13KcptunProfileC4modeSSvM.Frame"* %3, i32 0, i32 0
  %5 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %1, i32 0, i32 1
  %6 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %6)
  %7 = bitcast %TSS* %5 to i8*
  call void @swift_beginAccess(i8* %7, [24 x i8]* %4, i64 33, i8* null) #6
  %8 = insertvalue { i8*, %TSS* } { i8* bitcast (void (i8*, i1)* @"$s4file13KcptunProfileC4modeSSvM.resume.0" to i8*), %TSS* undef }, %TSS* %5, 1
  ret { i8*, %TSS* } %8
}

define internal dso_local swiftcc void @"$s4file13KcptunProfileC4modeSSvM.resume.0"(i8* noalias nonnull dereferenceable(32), i1) #0 {
  %3 = bitcast i8* %0 to %"$s4file13KcptunProfileC4modeSSvM.Frame"*
  %4 = bitcast %"$s4file13KcptunProfileC4modeSSvM.Frame"* %3 to i8*
  %5 = getelementptr inbounds %"$s4file13KcptunProfileC4modeSSvM.Frame", %"$s4file13KcptunProfileC4modeSSvM.Frame"* %3, i32 0, i32 0
  %6 = getelementptr inbounds %"$s4file13KcptunProfileC4modeSSvM.Frame", %"$s4file13KcptunProfileC4modeSSvM.Frame"* %3, i32 0, i32 0
  call void @swift_endAccess([24 x i8]* %6) #6
  %7 = bitcast [24 x i8]* %6 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %7)
  ret void
}

define dso_local hidden swiftcc { i64, %swift.bridge* } @"$s4file13KcptunProfileC3keySSvg"(%T4file13KcptunProfileC* swiftself) #0 {
  %2 = alloca [24 x i8], align 8
  %3 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 2
  %4 = bitcast [24 x i8]* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %4)
  %5 = bitcast %TSS* %3 to i8*
  call void @swift_beginAccess(i8* %5, [24 x i8]* %2, i64 32, i8* null) #6
  %6 = getelementptr inbounds %TSS, %TSS* %3, i32 0, i32 0
  %7 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %6, i32 0, i32 0
  %8 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %7, i32 0, i32 0
  %9 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %8, i32 0, i32 0
  %10 = load i64, i64* %9, align 8
  %11 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %7, i32 0, i32 1
  %12 = load %swift.bridge*, %swift.bridge** %11, align 8
  %13 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %12) #6
  call void @swift_endAccess([24 x i8]* %2) #6
  %14 = bitcast [24 x i8]* %2 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %14)
  %15 = insertvalue { i64, %swift.bridge* } undef, i64 %10, 0
  %16 = insertvalue { i64, %swift.bridge* } %15, %swift.bridge* %12, 1
  ret { i64, %swift.bridge* } %16
}

define dso_local hidden swiftcc void @"$s4file13KcptunProfileC3keySSvs"(i64, %swift.bridge*, %T4file13KcptunProfileC* swiftself) #0 {
  %4 = alloca [24 x i8], align 8
  %5 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %1) #6
  %6 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %2, i32 0, i32 2
  %7 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %7)
  %8 = bitcast %TSS* %6 to i8*
  call void @swift_beginAccess(i8* %8, [24 x i8]* %4, i64 33, i8* null) #6
  %9 = getelementptr inbounds %TSS, %TSS* %6, i32 0, i32 0
  %10 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %9, i32 0, i32 0
  %11 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %10, i32 0, i32 0
  %12 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %11, i32 0, i32 0
  %13 = load i64, i64* %12, align 8
  %14 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %10, i32 0, i32 1
  %15 = load %swift.bridge*, %swift.bridge** %14, align 8
  %16 = getelementptr inbounds %TSS, %TSS* %6, i32 0, i32 0
  %17 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %16, i32 0, i32 0
  %18 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %17, i32 0, i32 0
  %19 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %18, i32 0, i32 0
  store i64 %0, i64* %19, align 8
  %20 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %17, i32 0, i32 1
  store %swift.bridge* %1, %swift.bridge** %20, align 8
  call void @swift_bridgeObjectRelease(%swift.bridge* %15) #6
  call void @swift_endAccess([24 x i8]* %4) #6
  %21 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %21)
  call void @swift_bridgeObjectRelease(%swift.bridge* %1) #6
  ret void
}

; Function Attrs: noinline
define dso_local hidden swiftcc { i8*, %TSS* } @"$s4file13KcptunProfileC3keySSvM"(i8* noalias dereferenceable(32), %T4file13KcptunProfileC* swiftself) #1 {
  %3 = bitcast i8* %0 to %"$s4file13KcptunProfileC3keySSvM.Frame"*
  %4 = getelementptr inbounds %"$s4file13KcptunProfileC3keySSvM.Frame", %"$s4file13KcptunProfileC3keySSvM.Frame"* %3, i32 0, i32 0
  %5 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %1, i32 0, i32 2
  %6 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %6)
  %7 = bitcast %TSS* %5 to i8*
  call void @swift_beginAccess(i8* %7, [24 x i8]* %4, i64 33, i8* null) #6
  %8 = insertvalue { i8*, %TSS* } { i8* bitcast (void (i8*, i1)* @"$s4file13KcptunProfileC3keySSvM.resume.0" to i8*), %TSS* undef }, %TSS* %5, 1
  ret { i8*, %TSS* } %8
}

define internal dso_local swiftcc void @"$s4file13KcptunProfileC3keySSvM.resume.0"(i8* noalias nonnull dereferenceable(32), i1) #0 {
  %3 = bitcast i8* %0 to %"$s4file13KcptunProfileC3keySSvM.Frame"*
  %4 = bitcast %"$s4file13KcptunProfileC3keySSvM.Frame"* %3 to i8*
  %5 = getelementptr inbounds %"$s4file13KcptunProfileC3keySSvM.Frame", %"$s4file13KcptunProfileC3keySSvM.Frame"* %3, i32 0, i32 0
  %6 = getelementptr inbounds %"$s4file13KcptunProfileC3keySSvM.Frame", %"$s4file13KcptunProfileC3keySSvM.Frame"* %3, i32 0, i32 0
  call void @swift_endAccess([24 x i8]* %6) #6
  %7 = bitcast [24 x i8]* %6 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %7)
  ret void
}

define dso_local hidden swiftcc { i64, %swift.bridge* } @"$s4file13KcptunProfileC5cryptSSvg"(%T4file13KcptunProfileC* swiftself) #0 {
  %2 = alloca [24 x i8], align 8
  %3 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 3
  %4 = bitcast [24 x i8]* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %4)
  %5 = bitcast %TSS* %3 to i8*
  call void @swift_beginAccess(i8* %5, [24 x i8]* %2, i64 32, i8* null) #6
  %6 = getelementptr inbounds %TSS, %TSS* %3, i32 0, i32 0
  %7 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %6, i32 0, i32 0
  %8 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %7, i32 0, i32 0
  %9 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %8, i32 0, i32 0
  %10 = load i64, i64* %9, align 8
  %11 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %7, i32 0, i32 1
  %12 = load %swift.bridge*, %swift.bridge** %11, align 8
  %13 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %12) #6
  call void @swift_endAccess([24 x i8]* %2) #6
  %14 = bitcast [24 x i8]* %2 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %14)
  %15 = insertvalue { i64, %swift.bridge* } undef, i64 %10, 0
  %16 = insertvalue { i64, %swift.bridge* } %15, %swift.bridge* %12, 1
  ret { i64, %swift.bridge* } %16
}

define dso_local hidden swiftcc void @"$s4file13KcptunProfileC5cryptSSvs"(i64, %swift.bridge*, %T4file13KcptunProfileC* swiftself) #0 {
  %4 = alloca [24 x i8], align 8
  %5 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %1) #6
  %6 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %2, i32 0, i32 3
  %7 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %7)
  %8 = bitcast %TSS* %6 to i8*
  call void @swift_beginAccess(i8* %8, [24 x i8]* %4, i64 33, i8* null) #6
  %9 = getelementptr inbounds %TSS, %TSS* %6, i32 0, i32 0
  %10 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %9, i32 0, i32 0
  %11 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %10, i32 0, i32 0
  %12 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %11, i32 0, i32 0
  %13 = load i64, i64* %12, align 8
  %14 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %10, i32 0, i32 1
  %15 = load %swift.bridge*, %swift.bridge** %14, align 8
  %16 = getelementptr inbounds %TSS, %TSS* %6, i32 0, i32 0
  %17 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %16, i32 0, i32 0
  %18 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %17, i32 0, i32 0
  %19 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %18, i32 0, i32 0
  store i64 %0, i64* %19, align 8
  %20 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %17, i32 0, i32 1
  store %swift.bridge* %1, %swift.bridge** %20, align 8
  call void @swift_bridgeObjectRelease(%swift.bridge* %15) #6
  call void @swift_endAccess([24 x i8]* %4) #6
  %21 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %21)
  call void @swift_bridgeObjectRelease(%swift.bridge* %1) #6
  ret void
}

; Function Attrs: noinline
define dso_local hidden swiftcc { i8*, %TSS* } @"$s4file13KcptunProfileC5cryptSSvM"(i8* noalias dereferenceable(32), %T4file13KcptunProfileC* swiftself) #1 {
  %3 = bitcast i8* %0 to %"$s4file13KcptunProfileC5cryptSSvM.Frame"*
  %4 = getelementptr inbounds %"$s4file13KcptunProfileC5cryptSSvM.Frame", %"$s4file13KcptunProfileC5cryptSSvM.Frame"* %3, i32 0, i32 0
  %5 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %1, i32 0, i32 3
  %6 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %6)
  %7 = bitcast %TSS* %5 to i8*
  call void @swift_beginAccess(i8* %7, [24 x i8]* %4, i64 33, i8* null) #6
  %8 = insertvalue { i8*, %TSS* } { i8* bitcast (void (i8*, i1)* @"$s4file13KcptunProfileC5cryptSSvM.resume.0" to i8*), %TSS* undef }, %TSS* %5, 1
  ret { i8*, %TSS* } %8
}

define internal dso_local swiftcc void @"$s4file13KcptunProfileC5cryptSSvM.resume.0"(i8* noalias nonnull dereferenceable(32), i1) #0 {
  %3 = bitcast i8* %0 to %"$s4file13KcptunProfileC5cryptSSvM.Frame"*
  %4 = bitcast %"$s4file13KcptunProfileC5cryptSSvM.Frame"* %3 to i8*
  %5 = getelementptr inbounds %"$s4file13KcptunProfileC5cryptSSvM.Frame", %"$s4file13KcptunProfileC5cryptSSvM.Frame"* %3, i32 0, i32 0
  %6 = getelementptr inbounds %"$s4file13KcptunProfileC5cryptSSvM.Frame", %"$s4file13KcptunProfileC5cryptSSvM.Frame"* %3, i32 0, i32 0
  call void @swift_endAccess([24 x i8]* %6) #6
  %7 = bitcast [24 x i8]* %6 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %7)
  ret void
}

define dso_local hidden swiftcc i1 @"$s4file13KcptunProfileC6nocompSbvg"(%T4file13KcptunProfileC* swiftself) #0 {
  %2 = alloca [24 x i8], align 8
  %3 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 4
  %4 = bitcast [24 x i8]* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %4)
  %5 = bitcast %TSb* %3 to i8*
  call void @swift_beginAccess(i8* %5, [24 x i8]* %2, i64 32, i8* null) #6
  %6 = getelementptr inbounds %TSb, %TSb* %3, i32 0, i32 0
  %7 = load i1, i1* %6, align 8
  call void @swift_endAccess([24 x i8]* %2) #6
  %8 = bitcast [24 x i8]* %2 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %8)
  ret i1 %7
}

define dso_local hidden swiftcc void @"$s4file13KcptunProfileC6nocompSbvs"(i1, %T4file13KcptunProfileC* swiftself) #0 {
  %3 = alloca [24 x i8], align 8
  %4 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %1, i32 0, i32 4
  %5 = bitcast [24 x i8]* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %5)
  %6 = bitcast %TSb* %4 to i8*
  call void @swift_beginAccess(i8* %6, [24 x i8]* %3, i64 33, i8* null) #6
  %7 = getelementptr inbounds %TSb, %TSb* %4, i32 0, i32 0
  store i1 %0, i1* %7, align 8
  call void @swift_endAccess([24 x i8]* %3) #6
  %8 = bitcast [24 x i8]* %3 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %8)
  ret void
}

; Function Attrs: noinline
define dso_local hidden swiftcc { i8*, %TSb* } @"$s4file13KcptunProfileC6nocompSbvM"(i8* noalias dereferenceable(32), %T4file13KcptunProfileC* swiftself) #1 {
  %3 = bitcast i8* %0 to %"$s4file13KcptunProfileC6nocompSbvM.Frame"*
  %4 = getelementptr inbounds %"$s4file13KcptunProfileC6nocompSbvM.Frame", %"$s4file13KcptunProfileC6nocompSbvM.Frame"* %3, i32 0, i32 0
  %5 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %1, i32 0, i32 4
  %6 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %6)
  %7 = bitcast %TSb* %5 to i8*
  call void @swift_beginAccess(i8* %7, [24 x i8]* %4, i64 33, i8* null) #6
  %8 = insertvalue { i8*, %TSb* } { i8* bitcast (void (i8*, i1)* @"$s4file13KcptunProfileC6nocompSbvM.resume.0" to i8*), %TSb* undef }, %TSb* %5, 1
  ret { i8*, %TSb* } %8
}

define internal dso_local swiftcc void @"$s4file13KcptunProfileC6nocompSbvM.resume.0"(i8* noalias nonnull dereferenceable(32), i1) #0 {
  %3 = bitcast i8* %0 to %"$s4file13KcptunProfileC6nocompSbvM.Frame"*
  %4 = bitcast %"$s4file13KcptunProfileC6nocompSbvM.Frame"* %3 to i8*
  %5 = getelementptr inbounds %"$s4file13KcptunProfileC6nocompSbvM.Frame", %"$s4file13KcptunProfileC6nocompSbvM.Frame"* %3, i32 0, i32 0
  %6 = getelementptr inbounds %"$s4file13KcptunProfileC6nocompSbvM.Frame", %"$s4file13KcptunProfileC6nocompSbvM.Frame"* %3, i32 0, i32 0
  call void @swift_endAccess([24 x i8]* %6) #6
  %7 = bitcast [24 x i8]* %6 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %7)
  ret void
}

define dso_local hidden swiftcc i32 @"$s4file13KcptunProfileC9datashards6UInt32Vvg"(%T4file13KcptunProfileC* swiftself) #0 {
  %2 = alloca [24 x i8], align 8
  %3 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 6
  %4 = bitcast [24 x i8]* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %4)
  %5 = bitcast %Ts6UInt32V* %3 to i8*
  call void @swift_beginAccess(i8* %5, [24 x i8]* %2, i64 32, i8* null) #6
  %6 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %3, i32 0, i32 0
  %7 = load i32, i32* %6, align 4
  call void @swift_endAccess([24 x i8]* %2) #6
  %8 = bitcast [24 x i8]* %2 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %8)
  ret i32 %7
}

define dso_local hidden swiftcc void @"$s4file13KcptunProfileC9datashards6UInt32Vvs"(i32, %T4file13KcptunProfileC* swiftself) #0 {
  %3 = alloca [24 x i8], align 8
  %4 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %1, i32 0, i32 6
  %5 = bitcast [24 x i8]* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %5)
  %6 = bitcast %Ts6UInt32V* %4 to i8*
  call void @swift_beginAccess(i8* %6, [24 x i8]* %3, i64 33, i8* null) #6
  %7 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %4, i32 0, i32 0
  store i32 %0, i32* %7, align 4
  call void @swift_endAccess([24 x i8]* %3) #6
  %8 = bitcast [24 x i8]* %3 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %8)
  ret void
}

; Function Attrs: noinline
define dso_local hidden swiftcc { i8*, %Ts6UInt32V* } @"$s4file13KcptunProfileC9datashards6UInt32VvM"(i8* noalias dereferenceable(32), %T4file13KcptunProfileC* swiftself) #1 {
  %3 = bitcast i8* %0 to %"$s4file13KcptunProfileC9datashards6UInt32VvM.Frame"*
  %4 = getelementptr inbounds %"$s4file13KcptunProfileC9datashards6UInt32VvM.Frame", %"$s4file13KcptunProfileC9datashards6UInt32VvM.Frame"* %3, i32 0, i32 0
  %5 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %1, i32 0, i32 6
  %6 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %6)
  %7 = bitcast %Ts6UInt32V* %5 to i8*
  call void @swift_beginAccess(i8* %7, [24 x i8]* %4, i64 33, i8* null) #6
  %8 = insertvalue { i8*, %Ts6UInt32V* } { i8* bitcast (void (i8*, i1)* @"$s4file13KcptunProfileC9datashards6UInt32VvM.resume.0" to i8*), %Ts6UInt32V* undef }, %Ts6UInt32V* %5, 1
  ret { i8*, %Ts6UInt32V* } %8
}

define internal dso_local swiftcc void @"$s4file13KcptunProfileC9datashards6UInt32VvM.resume.0"(i8* noalias nonnull dereferenceable(32), i1) #0 {
  %3 = bitcast i8* %0 to %"$s4file13KcptunProfileC9datashards6UInt32VvM.Frame"*
  %4 = bitcast %"$s4file13KcptunProfileC9datashards6UInt32VvM.Frame"* %3 to i8*
  %5 = getelementptr inbounds %"$s4file13KcptunProfileC9datashards6UInt32VvM.Frame", %"$s4file13KcptunProfileC9datashards6UInt32VvM.Frame"* %3, i32 0, i32 0
  %6 = getelementptr inbounds %"$s4file13KcptunProfileC9datashards6UInt32VvM.Frame", %"$s4file13KcptunProfileC9datashards6UInt32VvM.Frame"* %3, i32 0, i32 0
  call void @swift_endAccess([24 x i8]* %6) #6
  %7 = bitcast [24 x i8]* %6 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %7)
  ret void
}

define dso_local hidden swiftcc i32 @"$s4file13KcptunProfileC11parityshards6UInt32Vvg"(%T4file13KcptunProfileC* swiftself) #0 {
  %2 = alloca [24 x i8], align 8
  %3 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 7
  %4 = bitcast [24 x i8]* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %4)
  %5 = bitcast %Ts6UInt32V* %3 to i8*
  call void @swift_beginAccess(i8* %5, [24 x i8]* %2, i64 32, i8* null) #6
  %6 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %3, i32 0, i32 0
  %7 = load i32, i32* %6, align 8
  call void @swift_endAccess([24 x i8]* %2) #6
  %8 = bitcast [24 x i8]* %2 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %8)
  ret i32 %7
}

define dso_local hidden swiftcc void @"$s4file13KcptunProfileC11parityshards6UInt32Vvs"(i32, %T4file13KcptunProfileC* swiftself) #0 {
  %3 = alloca [24 x i8], align 8
  %4 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %1, i32 0, i32 7
  %5 = bitcast [24 x i8]* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %5)
  %6 = bitcast %Ts6UInt32V* %4 to i8*
  call void @swift_beginAccess(i8* %6, [24 x i8]* %3, i64 33, i8* null) #6
  %7 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %4, i32 0, i32 0
  store i32 %0, i32* %7, align 8
  call void @swift_endAccess([24 x i8]* %3) #6
  %8 = bitcast [24 x i8]* %3 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %8)
  ret void
}

; Function Attrs: noinline
define dso_local hidden swiftcc { i8*, %Ts6UInt32V* } @"$s4file13KcptunProfileC11parityshards6UInt32VvM"(i8* noalias dereferenceable(32), %T4file13KcptunProfileC* swiftself) #1 {
  %3 = bitcast i8* %0 to %"$s4file13KcptunProfileC11parityshards6UInt32VvM.Frame"*
  %4 = getelementptr inbounds %"$s4file13KcptunProfileC11parityshards6UInt32VvM.Frame", %"$s4file13KcptunProfileC11parityshards6UInt32VvM.Frame"* %3, i32 0, i32 0
  %5 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %1, i32 0, i32 7
  %6 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %6)
  %7 = bitcast %Ts6UInt32V* %5 to i8*
  call void @swift_beginAccess(i8* %7, [24 x i8]* %4, i64 33, i8* null) #6
  %8 = insertvalue { i8*, %Ts6UInt32V* } { i8* bitcast (void (i8*, i1)* @"$s4file13KcptunProfileC11parityshards6UInt32VvM.resume.0" to i8*), %Ts6UInt32V* undef }, %Ts6UInt32V* %5, 1
  ret { i8*, %Ts6UInt32V* } %8
}

define internal dso_local swiftcc void @"$s4file13KcptunProfileC11parityshards6UInt32VvM.resume.0"(i8* noalias nonnull dereferenceable(32), i1) #0 {
  %3 = bitcast i8* %0 to %"$s4file13KcptunProfileC11parityshards6UInt32VvM.Frame"*
  %4 = bitcast %"$s4file13KcptunProfileC11parityshards6UInt32VvM.Frame"* %3 to i8*
  %5 = getelementptr inbounds %"$s4file13KcptunProfileC11parityshards6UInt32VvM.Frame", %"$s4file13KcptunProfileC11parityshards6UInt32VvM.Frame"* %3, i32 0, i32 0
  %6 = getelementptr inbounds %"$s4file13KcptunProfileC11parityshards6UInt32VvM.Frame", %"$s4file13KcptunProfileC11parityshards6UInt32VvM.Frame"* %3, i32 0, i32 0
  call void @swift_endAccess([24 x i8]* %6) #6
  %7 = bitcast [24 x i8]* %6 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %7)
  ret void
}

define dso_local hidden swiftcc i32 @"$s4file13KcptunProfileC3mtus6UInt32Vvg"(%T4file13KcptunProfileC* swiftself) #0 {
  %2 = alloca [24 x i8], align 8
  %3 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 8
  %4 = bitcast [24 x i8]* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %4)
  %5 = bitcast %Ts6UInt32V* %3 to i8*
  call void @swift_beginAccess(i8* %5, [24 x i8]* %2, i64 32, i8* null) #6
  %6 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %3, i32 0, i32 0
  %7 = load i32, i32* %6, align 4
  call void @swift_endAccess([24 x i8]* %2) #6
  %8 = bitcast [24 x i8]* %2 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %8)
  ret i32 %7
}

define dso_local hidden swiftcc void @"$s4file13KcptunProfileC3mtus6UInt32Vvs"(i32, %T4file13KcptunProfileC* swiftself) #0 {
  %3 = alloca [24 x i8], align 8
  %4 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %1, i32 0, i32 8
  %5 = bitcast [24 x i8]* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %5)
  %6 = bitcast %Ts6UInt32V* %4 to i8*
  call void @swift_beginAccess(i8* %6, [24 x i8]* %3, i64 33, i8* null) #6
  %7 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %4, i32 0, i32 0
  store i32 %0, i32* %7, align 4
  call void @swift_endAccess([24 x i8]* %3) #6
  %8 = bitcast [24 x i8]* %3 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %8)
  ret void
}

; Function Attrs: noinline
define dso_local hidden swiftcc { i8*, %Ts6UInt32V* } @"$s4file13KcptunProfileC3mtus6UInt32VvM"(i8* noalias dereferenceable(32), %T4file13KcptunProfileC* swiftself) #1 {
  %3 = bitcast i8* %0 to %"$s4file13KcptunProfileC3mtus6UInt32VvM.Frame"*
  %4 = getelementptr inbounds %"$s4file13KcptunProfileC3mtus6UInt32VvM.Frame", %"$s4file13KcptunProfileC3mtus6UInt32VvM.Frame"* %3, i32 0, i32 0
  %5 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %1, i32 0, i32 8
  %6 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %6)
  %7 = bitcast %Ts6UInt32V* %5 to i8*
  call void @swift_beginAccess(i8* %7, [24 x i8]* %4, i64 33, i8* null) #6
  %8 = insertvalue { i8*, %Ts6UInt32V* } { i8* bitcast (void (i8*, i1)* @"$s4file13KcptunProfileC3mtus6UInt32VvM.resume.0" to i8*), %Ts6UInt32V* undef }, %Ts6UInt32V* %5, 1
  ret { i8*, %Ts6UInt32V* } %8
}

define internal dso_local swiftcc void @"$s4file13KcptunProfileC3mtus6UInt32VvM.resume.0"(i8* noalias nonnull dereferenceable(32), i1) #0 {
  %3 = bitcast i8* %0 to %"$s4file13KcptunProfileC3mtus6UInt32VvM.Frame"*
  %4 = bitcast %"$s4file13KcptunProfileC3mtus6UInt32VvM.Frame"* %3 to i8*
  %5 = getelementptr inbounds %"$s4file13KcptunProfileC3mtus6UInt32VvM.Frame", %"$s4file13KcptunProfileC3mtus6UInt32VvM.Frame"* %3, i32 0, i32 0
  %6 = getelementptr inbounds %"$s4file13KcptunProfileC3mtus6UInt32VvM.Frame", %"$s4file13KcptunProfileC3mtus6UInt32VvM.Frame"* %3, i32 0, i32 0
  call void @swift_endAccess([24 x i8]* %6) #6
  %7 = bitcast [24 x i8]* %6 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %7)
  ret void
}

define dso_local hidden swiftcc { i64, %swift.bridge* } @"$s4file13KcptunProfileC9argumentsSSvg"(%T4file13KcptunProfileC* swiftself) #0 {
  %2 = alloca [24 x i8], align 8
  %3 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 9
  %4 = bitcast [24 x i8]* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %4)
  %5 = bitcast %TSS* %3 to i8*
  call void @swift_beginAccess(i8* %5, [24 x i8]* %2, i64 32, i8* null) #6
  %6 = getelementptr inbounds %TSS, %TSS* %3, i32 0, i32 0
  %7 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %6, i32 0, i32 0
  %8 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %7, i32 0, i32 0
  %9 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %8, i32 0, i32 0
  %10 = load i64, i64* %9, align 8
  %11 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %7, i32 0, i32 1
  %12 = load %swift.bridge*, %swift.bridge** %11, align 8
  %13 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %12) #6
  call void @swift_endAccess([24 x i8]* %2) #6
  %14 = bitcast [24 x i8]* %2 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %14)
  %15 = insertvalue { i64, %swift.bridge* } undef, i64 %10, 0
  %16 = insertvalue { i64, %swift.bridge* } %15, %swift.bridge* %12, 1
  ret { i64, %swift.bridge* } %16
}

define dso_local hidden swiftcc void @"$s4file13KcptunProfileC9argumentsSSvs"(i64, %swift.bridge*, %T4file13KcptunProfileC* swiftself) #0 {
  %4 = alloca [24 x i8], align 8
  %5 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %1) #6
  %6 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %2, i32 0, i32 9
  %7 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %7)
  %8 = bitcast %TSS* %6 to i8*
  call void @swift_beginAccess(i8* %8, [24 x i8]* %4, i64 33, i8* null) #6
  %9 = getelementptr inbounds %TSS, %TSS* %6, i32 0, i32 0
  %10 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %9, i32 0, i32 0
  %11 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %10, i32 0, i32 0
  %12 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %11, i32 0, i32 0
  %13 = load i64, i64* %12, align 8
  %14 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %10, i32 0, i32 1
  %15 = load %swift.bridge*, %swift.bridge** %14, align 8
  %16 = getelementptr inbounds %TSS, %TSS* %6, i32 0, i32 0
  %17 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %16, i32 0, i32 0
  %18 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %17, i32 0, i32 0
  %19 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %18, i32 0, i32 0
  store i64 %0, i64* %19, align 8
  %20 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %17, i32 0, i32 1
  store %swift.bridge* %1, %swift.bridge** %20, align 8
  call void @swift_bridgeObjectRelease(%swift.bridge* %15) #6
  call void @swift_endAccess([24 x i8]* %4) #6
  %21 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %21)
  call void @swift_bridgeObjectRelease(%swift.bridge* %1) #6
  ret void
}

; Function Attrs: noinline
define dso_local hidden swiftcc { i8*, %TSS* } @"$s4file13KcptunProfileC9argumentsSSvM"(i8* noalias dereferenceable(32), %T4file13KcptunProfileC* swiftself) #1 {
  %3 = bitcast i8* %0 to %"$s4file13KcptunProfileC9argumentsSSvM.Frame"*
  %4 = getelementptr inbounds %"$s4file13KcptunProfileC9argumentsSSvM.Frame", %"$s4file13KcptunProfileC9argumentsSSvM.Frame"* %3, i32 0, i32 0
  %5 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %1, i32 0, i32 9
  %6 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %6)
  %7 = bitcast %TSS* %5 to i8*
  call void @swift_beginAccess(i8* %7, [24 x i8]* %4, i64 33, i8* null) #6
  %8 = insertvalue { i8*, %TSS* } { i8* bitcast (void (i8*, i1)* @"$s4file13KcptunProfileC9argumentsSSvM.resume.0" to i8*), %TSS* undef }, %TSS* %5, 1
  ret { i8*, %TSS* } %8
}

define internal dso_local swiftcc void @"$s4file13KcptunProfileC9argumentsSSvM.resume.0"(i8* noalias nonnull dereferenceable(32), i1) #0 {
  %3 = bitcast i8* %0 to %"$s4file13KcptunProfileC9argumentsSSvM.Frame"*
  %4 = bitcast %"$s4file13KcptunProfileC9argumentsSSvM.Frame"* %3 to i8*
  %5 = getelementptr inbounds %"$s4file13KcptunProfileC9argumentsSSvM.Frame", %"$s4file13KcptunProfileC9argumentsSSvM.Frame"* %3, i32 0, i32 0
  %6 = getelementptr inbounds %"$s4file13KcptunProfileC9argumentsSSvM.Frame", %"$s4file13KcptunProfileC9argumentsSSvM.Frame"* %3, i32 0, i32 0
  call void @swift_endAccess([24 x i8]* %6) #6
  %7 = bitcast [24 x i8]* %6 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %7)
  ret void
}

define dso_local hidden swiftcc void @"$s4file13KcptunProfileC4copy4withyp10Foundation6NSZoneVSg_tF"(%Any* noalias nocapture sret, i8, %T4file13KcptunProfileC* swiftself) #0 {
  %4 = alloca i1, align 8
  %5 = bitcast i1* %4 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %5, i8 0, i64 1, i1 false)
  %6 = alloca %T4file13KcptunProfileC*, align 8
  %7 = bitcast %T4file13KcptunProfileC** %6 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %7, i8 0, i64 8, i1 false)
  %8 = alloca [24 x i8], align 8
  %9 = alloca [24 x i8], align 8
  %10 = alloca [24 x i8], align 8
  %11 = alloca [24 x i8], align 8
  %12 = alloca [24 x i8], align 8
  %13 = alloca [24 x i8], align 8
  %14 = alloca [24 x i8], align 8
  %15 = alloca [24 x i8], align 8
  %16 = alloca [24 x i8], align 8
  %17 = alloca [24 x i8], align 8
  %18 = alloca [24 x i8], align 8
  %19 = alloca [24 x i8], align 8
  %20 = alloca [24 x i8], align 8
  %21 = alloca [24 x i8], align 8
  %22 = trunc i8 %1 to i1
  store i1 %22, i1* %4, align 8
  store %T4file13KcptunProfileC* %2, %T4file13KcptunProfileC** %6, align 8
  %23 = call swiftcc %swift.metadata_response @"$s4file13KcptunProfileCMa"(i64 0) #8
  %24 = extractvalue %swift.metadata_response %23, 0
  %25 = call swiftcc %T4file13KcptunProfileC* @"$s4file13KcptunProfileCACycfC"(%swift.type* swiftself %24)
  call void asm sideeffect "", "r"(%T4file13KcptunProfileC* %25)
  %26 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %2, i32 0, i32 1
  %27 = bitcast [24 x i8]* %8 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %27)
  %28 = bitcast %TSS* %26 to i8*
  call void @swift_beginAccess(i8* %28, [24 x i8]* %8, i64 32, i8* null) #6
  %29 = getelementptr inbounds %TSS, %TSS* %26, i32 0, i32 0
  %30 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %29, i32 0, i32 0
  %31 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %30, i32 0, i32 0
  %32 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %31, i32 0, i32 0
  %33 = load i64, i64* %32, align 8
  %34 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %30, i32 0, i32 1
  %35 = load %swift.bridge*, %swift.bridge** %34, align 8
  %36 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %35) #6
  call void @swift_endAccess([24 x i8]* %8) #6
  %37 = bitcast [24 x i8]* %8 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %37)
  %38 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %35) #6
  %39 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %25, i32 0, i32 1
  %40 = bitcast [24 x i8]* %9 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %40)
  %41 = bitcast %TSS* %39 to i8*
  call void @swift_beginAccess(i8* %41, [24 x i8]* %9, i64 33, i8* null) #6
  %42 = getelementptr inbounds %TSS, %TSS* %39, i32 0, i32 0
  %43 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %42, i32 0, i32 0
  %44 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %43, i32 0, i32 0
  %45 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %44, i32 0, i32 0
  %46 = load i64, i64* %45, align 8
  %47 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %43, i32 0, i32 1
  %48 = load %swift.bridge*, %swift.bridge** %47, align 8
  %49 = getelementptr inbounds %TSS, %TSS* %39, i32 0, i32 0
  %50 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %49, i32 0, i32 0
  %51 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %50, i32 0, i32 0
  %52 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %51, i32 0, i32 0
  store i64 %33, i64* %52, align 8
  %53 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %50, i32 0, i32 1
  store %swift.bridge* %35, %swift.bridge** %53, align 8
  call void @swift_bridgeObjectRelease(%swift.bridge* %48) #6
  call void @swift_endAccess([24 x i8]* %9) #6
  %54 = bitcast [24 x i8]* %9 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %54)
  call void @swift_bridgeObjectRelease(%swift.bridge* %35) #6
  %55 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %2, i32 0, i32 2
  %56 = bitcast [24 x i8]* %10 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %56)
  %57 = bitcast %TSS* %55 to i8*
  call void @swift_beginAccess(i8* %57, [24 x i8]* %10, i64 32, i8* null) #6
  %58 = getelementptr inbounds %TSS, %TSS* %55, i32 0, i32 0
  %59 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %58, i32 0, i32 0
  %60 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %59, i32 0, i32 0
  %61 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %60, i32 0, i32 0
  %62 = load i64, i64* %61, align 8
  %63 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %59, i32 0, i32 1
  %64 = load %swift.bridge*, %swift.bridge** %63, align 8
  %65 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %64) #6
  call void @swift_endAccess([24 x i8]* %10) #6
  %66 = bitcast [24 x i8]* %10 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %66)
  %67 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %64) #6
  %68 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %25, i32 0, i32 2
  %69 = bitcast [24 x i8]* %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %69)
  %70 = bitcast %TSS* %68 to i8*
  call void @swift_beginAccess(i8* %70, [24 x i8]* %11, i64 33, i8* null) #6
  %71 = getelementptr inbounds %TSS, %TSS* %68, i32 0, i32 0
  %72 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %71, i32 0, i32 0
  %73 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %72, i32 0, i32 0
  %74 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %73, i32 0, i32 0
  %75 = load i64, i64* %74, align 8
  %76 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %72, i32 0, i32 1
  %77 = load %swift.bridge*, %swift.bridge** %76, align 8
  %78 = getelementptr inbounds %TSS, %TSS* %68, i32 0, i32 0
  %79 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %78, i32 0, i32 0
  %80 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %79, i32 0, i32 0
  %81 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %80, i32 0, i32 0
  store i64 %62, i64* %81, align 8
  %82 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %79, i32 0, i32 1
  store %swift.bridge* %64, %swift.bridge** %82, align 8
  call void @swift_bridgeObjectRelease(%swift.bridge* %77) #6
  call void @swift_endAccess([24 x i8]* %11) #6
  %83 = bitcast [24 x i8]* %11 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %83)
  call void @swift_bridgeObjectRelease(%swift.bridge* %64) #6
  %84 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %2, i32 0, i32 3
  %85 = bitcast [24 x i8]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %85)
  %86 = bitcast %TSS* %84 to i8*
  call void @swift_beginAccess(i8* %86, [24 x i8]* %12, i64 32, i8* null) #6
  %87 = getelementptr inbounds %TSS, %TSS* %84, i32 0, i32 0
  %88 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %87, i32 0, i32 0
  %89 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %88, i32 0, i32 0
  %90 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %89, i32 0, i32 0
  %91 = load i64, i64* %90, align 8
  %92 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %88, i32 0, i32 1
  %93 = load %swift.bridge*, %swift.bridge** %92, align 8
  %94 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %93) #6
  call void @swift_endAccess([24 x i8]* %12) #6
  %95 = bitcast [24 x i8]* %12 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %95)
  %96 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %93) #6
  %97 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %25, i32 0, i32 3
  %98 = bitcast [24 x i8]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %98)
  %99 = bitcast %TSS* %97 to i8*
  call void @swift_beginAccess(i8* %99, [24 x i8]* %13, i64 33, i8* null) #6
  %100 = getelementptr inbounds %TSS, %TSS* %97, i32 0, i32 0
  %101 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %100, i32 0, i32 0
  %102 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %101, i32 0, i32 0
  %103 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %102, i32 0, i32 0
  %104 = load i64, i64* %103, align 8
  %105 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %101, i32 0, i32 1
  %106 = load %swift.bridge*, %swift.bridge** %105, align 8
  %107 = getelementptr inbounds %TSS, %TSS* %97, i32 0, i32 0
  %108 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %107, i32 0, i32 0
  %109 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %108, i32 0, i32 0
  %110 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %109, i32 0, i32 0
  store i64 %91, i64* %110, align 8
  %111 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %108, i32 0, i32 1
  store %swift.bridge* %93, %swift.bridge** %111, align 8
  call void @swift_bridgeObjectRelease(%swift.bridge* %106) #6
  call void @swift_endAccess([24 x i8]* %13) #6
  %112 = bitcast [24 x i8]* %13 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %112)
  call void @swift_bridgeObjectRelease(%swift.bridge* %93) #6
  %113 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %2, i32 0, i32 4
  %114 = bitcast [24 x i8]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %114)
  %115 = bitcast %TSb* %113 to i8*
  call void @swift_beginAccess(i8* %115, [24 x i8]* %14, i64 32, i8* null) #6
  %116 = getelementptr inbounds %TSb, %TSb* %113, i32 0, i32 0
  %117 = load i1, i1* %116, align 8
  call void @swift_endAccess([24 x i8]* %14) #6
  %118 = bitcast [24 x i8]* %14 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %118)
  %119 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %25, i32 0, i32 4
  %120 = bitcast [24 x i8]* %15 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %120)
  %121 = bitcast %TSb* %119 to i8*
  call void @swift_beginAccess(i8* %121, [24 x i8]* %15, i64 33, i8* null) #6
  %122 = getelementptr inbounds %TSb, %TSb* %119, i32 0, i32 0
  store i1 %117, i1* %122, align 8
  call void @swift_endAccess([24 x i8]* %15) #6
  %123 = bitcast [24 x i8]* %15 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %123)
  %124 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %2, i32 0, i32 6
  %125 = bitcast [24 x i8]* %16 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %125)
  %126 = bitcast %Ts6UInt32V* %124 to i8*
  call void @swift_beginAccess(i8* %126, [24 x i8]* %16, i64 32, i8* null) #6
  %127 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %124, i32 0, i32 0
  %128 = load i32, i32* %127, align 4
  call void @swift_endAccess([24 x i8]* %16) #6
  %129 = bitcast [24 x i8]* %16 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %129)
  %130 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %25, i32 0, i32 6
  %131 = bitcast [24 x i8]* %17 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %131)
  %132 = bitcast %Ts6UInt32V* %130 to i8*
  call void @swift_beginAccess(i8* %132, [24 x i8]* %17, i64 33, i8* null) #6
  %133 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %130, i32 0, i32 0
  store i32 %128, i32* %133, align 4
  call void @swift_endAccess([24 x i8]* %17) #6
  %134 = bitcast [24 x i8]* %17 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %134)
  %135 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %2, i32 0, i32 7
  %136 = bitcast [24 x i8]* %18 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %136)
  %137 = bitcast %Ts6UInt32V* %135 to i8*
  call void @swift_beginAccess(i8* %137, [24 x i8]* %18, i64 32, i8* null) #6
  %138 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %135, i32 0, i32 0
  %139 = load i32, i32* %138, align 8
  call void @swift_endAccess([24 x i8]* %18) #6
  %140 = bitcast [24 x i8]* %18 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %140)
  %141 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %25, i32 0, i32 7
  %142 = bitcast [24 x i8]* %19 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %142)
  %143 = bitcast %Ts6UInt32V* %141 to i8*
  call void @swift_beginAccess(i8* %143, [24 x i8]* %19, i64 33, i8* null) #6
  %144 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %141, i32 0, i32 0
  store i32 %139, i32* %144, align 8
  call void @swift_endAccess([24 x i8]* %19) #6
  %145 = bitcast [24 x i8]* %19 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %145)
  %146 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %2, i32 0, i32 8
  %147 = bitcast [24 x i8]* %20 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %147)
  %148 = bitcast %Ts6UInt32V* %146 to i8*
  call void @swift_beginAccess(i8* %148, [24 x i8]* %20, i64 32, i8* null) #6
  %149 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %146, i32 0, i32 0
  %150 = load i32, i32* %149, align 4
  call void @swift_endAccess([24 x i8]* %20) #6
  %151 = bitcast [24 x i8]* %20 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %151)
  %152 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %25, i32 0, i32 8
  %153 = bitcast [24 x i8]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %153)
  %154 = bitcast %Ts6UInt32V* %152 to i8*
  call void @swift_beginAccess(i8* %154, [24 x i8]* %21, i64 33, i8* null) #6
  %155 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %152, i32 0, i32 0
  store i32 %150, i32* %155, align 4
  call void @swift_endAccess([24 x i8]* %21) #6
  %156 = bitcast [24 x i8]* %21 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %156)
  %157 = bitcast %T4file13KcptunProfileC* %25 to %swift.refcounted*
  %158 = call %swift.refcounted* @swift_retain(%swift.refcounted* returned %157) #6
  %159 = getelementptr inbounds %Any, %Any* %0, i32 0, i32 1
  store %swift.type* %24, %swift.type** %159, align 8
  %160 = getelementptr inbounds %Any, %Any* %0, i32 0, i32 0
  %161 = getelementptr inbounds %Any, %Any* %0, i32 0, i32 0
  %162 = bitcast [24 x i8]* %161 to %T4file13KcptunProfileC**
  store %T4file13KcptunProfileC* %25, %T4file13KcptunProfileC** %162, align 8
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T4file13KcptunProfileC*)*)(%T4file13KcptunProfileC* %25) #6
  ret void
}

define dso_local hidden swiftcc %T4file13KcptunProfileC* @"$s4file13KcptunProfileCACycfC"(%swift.type* swiftself) #0 {
  %2 = call swiftcc %swift.metadata_response @"$s4file13KcptunProfileCMa"(i64 0) #8
  %3 = extractvalue %swift.metadata_response %2, 0
  %4 = call noalias %swift.refcounted* @swift_allocObject(%swift.type* %3, i64 96, i64 7) #6
  %5 = bitcast %swift.refcounted* %4 to %T4file13KcptunProfileC*
  %6 = call swiftcc %T4file13KcptunProfileC* @"$s4file13KcptunProfileCACycfc"(%T4file13KcptunProfileC* swiftself %5)
  ret %T4file13KcptunProfileC* %6
}

define dso_local hidden swiftcc %swift.bridge* @"$s4file13KcptunProfileC12toDictionarySDySSyXlGyF"(%T4file13KcptunProfileC* swiftself) #0 {
  %2 = alloca %T4file13KcptunProfileC*, align 8
  %3 = bitcast %T4file13KcptunProfileC** %2 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %3, i8 0, i64 8, i1 false)
  %4 = alloca [24 x i8], align 8
  %5 = alloca [24 x i8], align 8
  %6 = alloca [24 x i8], align 8
  %7 = alloca [24 x i8], align 8
  %8 = alloca [24 x i8], align 8
  %9 = alloca [24 x i8], align 8
  %10 = alloca [24 x i8], align 8
  %11 = alloca [24 x i8], align 8
  store %T4file13KcptunProfileC* %0, %T4file13KcptunProfileC** %2, align 8
  %12 = call swiftcc %swift.metadata_response @"$sSS_yXltMa"(i64 0) #8
  %13 = extractvalue %swift.metadata_response %12, 0
  %14 = call swiftcc { %Ts28__ContiguousArrayStorageBaseC*, i8* } @"$ss27_allocateUninitializedArrayySayxG_BptBwlF"(i64 8, %swift.type* %13)
  %15 = extractvalue { %Ts28__ContiguousArrayStorageBaseC*, i8* } %14, 0
  %16 = extractvalue { %Ts28__ContiguousArrayStorageBaseC*, i8* } %14, 1
  %17 = bitcast i8* %16 to <{ %TSS, %AnyObject }>*
  %18 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %17, i32 0, i32 0
  %19 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %17, i32 0, i32 1
  %20 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @14, i64 0, i64 0), i64 4, i1 true)
  %21 = extractvalue { i64, %swift.bridge* } %20, 0
  %22 = extractvalue { i64, %swift.bridge* } %20, 1
  %23 = getelementptr inbounds %TSS, %TSS* %18, i32 0, i32 0
  %24 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %23, i32 0, i32 0
  %25 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %24, i32 0, i32 0
  %26 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %25, i32 0, i32 0
  store i64 %21, i64* %26, align 8
  %27 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %24, i32 0, i32 1
  store %swift.bridge* %22, %swift.bridge** %27, align 8
  %28 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 1
  %29 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %29)
  %30 = bitcast %TSS* %28 to i8*
  call void @swift_beginAccess(i8* %30, [24 x i8]* %4, i64 32, i8* null) #6
  %31 = getelementptr inbounds %TSS, %TSS* %28, i32 0, i32 0
  %32 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %31, i32 0, i32 0
  %33 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %32, i32 0, i32 0
  %34 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %33, i32 0, i32 0
  %35 = load i64, i64* %34, align 8
  %36 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %32, i32 0, i32 1
  %37 = load %swift.bridge*, %swift.bridge** %36, align 8
  %38 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %37) #6
  call void @swift_endAccess([24 x i8]* %4) #6
  %39 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %39)
  %40 = call swiftcc %T10Foundation8NSStringC* @"$sSS10FoundationE19_bridgeToObjectiveCAA8NSStringCyF"(i64 %35, %swift.bridge* %37)
  %41 = bitcast %T10Foundation8NSStringC* %40 to %swift.refcounted*
  %42 = getelementptr inbounds %AnyObject, %AnyObject* %19, i32 0, i32 0
  store %swift.refcounted* %41, %swift.refcounted** %42, align 8
  %43 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %17, i64 1
  %44 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %43, i32 0, i32 0
  %45 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %43, i32 0, i32 1
  %46 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @15, i64 0, i64 0), i64 3, i1 true)
  %47 = extractvalue { i64, %swift.bridge* } %46, 0
  %48 = extractvalue { i64, %swift.bridge* } %46, 1
  %49 = getelementptr inbounds %TSS, %TSS* %44, i32 0, i32 0
  %50 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %49, i32 0, i32 0
  %51 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %50, i32 0, i32 0
  %52 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %51, i32 0, i32 0
  store i64 %47, i64* %52, align 8
  %53 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %50, i32 0, i32 1
  store %swift.bridge* %48, %swift.bridge** %53, align 8
  %54 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 2
  %55 = bitcast [24 x i8]* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %55)
  %56 = bitcast %TSS* %54 to i8*
  call void @swift_beginAccess(i8* %56, [24 x i8]* %5, i64 32, i8* null) #6
  %57 = getelementptr inbounds %TSS, %TSS* %54, i32 0, i32 0
  %58 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %57, i32 0, i32 0
  %59 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %58, i32 0, i32 0
  %60 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %59, i32 0, i32 0
  %61 = load i64, i64* %60, align 8
  %62 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %58, i32 0, i32 1
  %63 = load %swift.bridge*, %swift.bridge** %62, align 8
  %64 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %63) #6
  call void @swift_endAccess([24 x i8]* %5) #6
  %65 = bitcast [24 x i8]* %5 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %65)
  %66 = call swiftcc %T10Foundation8NSStringC* @"$sSS10FoundationE19_bridgeToObjectiveCAA8NSStringCyF"(i64 %61, %swift.bridge* %63)
  %67 = bitcast %T10Foundation8NSStringC* %66 to %swift.refcounted*
  %68 = getelementptr inbounds %AnyObject, %AnyObject* %45, i32 0, i32 0
  store %swift.refcounted* %67, %swift.refcounted** %68, align 8
  %69 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %17, i64 2
  %70 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %69, i32 0, i32 0
  %71 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %69, i32 0, i32 1
  %72 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @16, i64 0, i64 0), i64 5, i1 true)
  %73 = extractvalue { i64, %swift.bridge* } %72, 0
  %74 = extractvalue { i64, %swift.bridge* } %72, 1
  %75 = getelementptr inbounds %TSS, %TSS* %70, i32 0, i32 0
  %76 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %75, i32 0, i32 0
  %77 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %76, i32 0, i32 0
  %78 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %77, i32 0, i32 0
  store i64 %73, i64* %78, align 8
  %79 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %76, i32 0, i32 1
  store %swift.bridge* %74, %swift.bridge** %79, align 8
  %80 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 3
  %81 = bitcast [24 x i8]* %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %81)
  %82 = bitcast %TSS* %80 to i8*
  call void @swift_beginAccess(i8* %82, [24 x i8]* %6, i64 32, i8* null) #6
  %83 = getelementptr inbounds %TSS, %TSS* %80, i32 0, i32 0
  %84 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %83, i32 0, i32 0
  %85 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %84, i32 0, i32 0
  %86 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %85, i32 0, i32 0
  %87 = load i64, i64* %86, align 8
  %88 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %84, i32 0, i32 1
  %89 = load %swift.bridge*, %swift.bridge** %88, align 8
  %90 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %89) #6
  call void @swift_endAccess([24 x i8]* %6) #6
  %91 = bitcast [24 x i8]* %6 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %91)
  %92 = call swiftcc %T10Foundation8NSStringC* @"$sSS10FoundationE19_bridgeToObjectiveCAA8NSStringCyF"(i64 %87, %swift.bridge* %89)
  %93 = bitcast %T10Foundation8NSStringC* %92 to %swift.refcounted*
  %94 = getelementptr inbounds %AnyObject, %AnyObject* %71, i32 0, i32 0
  store %swift.refcounted* %93, %swift.refcounted** %94, align 8
  %95 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %17, i64 3
  %96 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %95, i32 0, i32 0
  %97 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %95, i32 0, i32 1
  %98 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @19, i64 0, i64 0), i64 6, i1 true)
  %99 = extractvalue { i64, %swift.bridge* } %98, 0
  %100 = extractvalue { i64, %swift.bridge* } %98, 1
  %101 = getelementptr inbounds %TSS, %TSS* %96, i32 0, i32 0
  %102 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %101, i32 0, i32 0
  %103 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %102, i32 0, i32 0
  %104 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %103, i32 0, i32 0
  store i64 %99, i64* %104, align 8
  %105 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %102, i32 0, i32 1
  store %swift.bridge* %100, %swift.bridge** %105, align 8
  %106 = call swiftcc %swift.metadata_response @"$s10Foundation8NSNumberCMa"(i64 0) #8
  %107 = extractvalue %swift.metadata_response %106, 0
  %108 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 4
  %109 = bitcast [24 x i8]* %7 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %109)
  %110 = bitcast %TSb* %108 to i8*
  call void @swift_beginAccess(i8* %110, [24 x i8]* %7, i64 32, i8* null) #6
  %111 = getelementptr inbounds %TSb, %TSb* %108, i32 0, i32 0
  %112 = load i1, i1* %111, align 8
  call void @swift_endAccess([24 x i8]* %7) #6
  %113 = bitcast [24 x i8]* %7 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %113)
  %114 = call swiftcc %T10Foundation8NSNumberC* @"$s10Foundation8NSNumberC5valueACSb_tcfC"(i1 %112, %swift.type* swiftself %107)
  %115 = bitcast %T10Foundation8NSNumberC* %114 to %swift.refcounted*
  %116 = getelementptr inbounds %AnyObject, %AnyObject* %97, i32 0, i32 0
  store %swift.refcounted* %115, %swift.refcounted** %116, align 8
  %117 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %17, i64 4
  %118 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %117, i32 0, i32 0
  %119 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %117, i32 0, i32 1
  %120 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @17, i64 0, i64 0), i64 9, i1 true)
  %121 = extractvalue { i64, %swift.bridge* } %120, 0
  %122 = extractvalue { i64, %swift.bridge* } %120, 1
  %123 = getelementptr inbounds %TSS, %TSS* %118, i32 0, i32 0
  %124 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %123, i32 0, i32 0
  %125 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %124, i32 0, i32 0
  %126 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %125, i32 0, i32 0
  store i64 %121, i64* %126, align 8
  %127 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %124, i32 0, i32 1
  store %swift.bridge* %122, %swift.bridge** %127, align 8
  %128 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 6
  %129 = bitcast [24 x i8]* %8 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %129)
  %130 = bitcast %Ts6UInt32V* %128 to i8*
  call void @swift_beginAccess(i8* %130, [24 x i8]* %8, i64 32, i8* null) #6
  %131 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %128, i32 0, i32 0
  %132 = load i32, i32* %131, align 4
  call void @swift_endAccess([24 x i8]* %8) #6
  %133 = bitcast [24 x i8]* %8 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %133)
  %134 = call swiftcc %T10Foundation8NSNumberC* @"$s10Foundation8NSNumberC5valueACs6UInt32V_tcfC"(i32 %132, %swift.type* swiftself %107)
  %135 = bitcast %T10Foundation8NSNumberC* %134 to %swift.refcounted*
  %136 = getelementptr inbounds %AnyObject, %AnyObject* %119, i32 0, i32 0
  store %swift.refcounted* %135, %swift.refcounted** %136, align 8
  %137 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %17, i64 5
  %138 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %137, i32 0, i32 0
  %139 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %137, i32 0, i32 1
  %140 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @18, i64 0, i64 0), i64 11, i1 true)
  %141 = extractvalue { i64, %swift.bridge* } %140, 0
  %142 = extractvalue { i64, %swift.bridge* } %140, 1
  %143 = getelementptr inbounds %TSS, %TSS* %138, i32 0, i32 0
  %144 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %143, i32 0, i32 0
  %145 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %144, i32 0, i32 0
  %146 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %145, i32 0, i32 0
  store i64 %141, i64* %146, align 8
  %147 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %144, i32 0, i32 1
  store %swift.bridge* %142, %swift.bridge** %147, align 8
  %148 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 7
  %149 = bitcast [24 x i8]* %9 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %149)
  %150 = bitcast %Ts6UInt32V* %148 to i8*
  call void @swift_beginAccess(i8* %150, [24 x i8]* %9, i64 32, i8* null) #6
  %151 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %148, i32 0, i32 0
  %152 = load i32, i32* %151, align 8
  call void @swift_endAccess([24 x i8]* %9) #6
  %153 = bitcast [24 x i8]* %9 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %153)
  %154 = call swiftcc %T10Foundation8NSNumberC* @"$s10Foundation8NSNumberC5valueACs6UInt32V_tcfC"(i32 %152, %swift.type* swiftself %107)
  %155 = bitcast %T10Foundation8NSNumberC* %154 to %swift.refcounted*
  %156 = getelementptr inbounds %AnyObject, %AnyObject* %139, i32 0, i32 0
  store %swift.refcounted* %155, %swift.refcounted** %156, align 8
  %157 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %17, i64 6
  %158 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %157, i32 0, i32 0
  %159 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %157, i32 0, i32 1
  %160 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @20, i64 0, i64 0), i64 3, i1 true)
  %161 = extractvalue { i64, %swift.bridge* } %160, 0
  %162 = extractvalue { i64, %swift.bridge* } %160, 1
  %163 = getelementptr inbounds %TSS, %TSS* %158, i32 0, i32 0
  %164 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %163, i32 0, i32 0
  %165 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %164, i32 0, i32 0
  %166 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %165, i32 0, i32 0
  store i64 %161, i64* %166, align 8
  %167 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %164, i32 0, i32 1
  store %swift.bridge* %162, %swift.bridge** %167, align 8
  %168 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 8
  %169 = bitcast [24 x i8]* %10 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %169)
  %170 = bitcast %Ts6UInt32V* %168 to i8*
  call void @swift_beginAccess(i8* %170, [24 x i8]* %10, i64 32, i8* null) #6
  %171 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %168, i32 0, i32 0
  %172 = load i32, i32* %171, align 4
  call void @swift_endAccess([24 x i8]* %10) #6
  %173 = bitcast [24 x i8]* %10 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %173)
  %174 = call swiftcc %T10Foundation8NSNumberC* @"$s10Foundation8NSNumberC5valueACs6UInt32V_tcfC"(i32 %172, %swift.type* swiftself %107)
  %175 = bitcast %T10Foundation8NSNumberC* %174 to %swift.refcounted*
  %176 = getelementptr inbounds %AnyObject, %AnyObject* %159, i32 0, i32 0
  store %swift.refcounted* %175, %swift.refcounted** %176, align 8
  %177 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %17, i64 7
  %178 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %177, i32 0, i32 0
  %179 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %177, i32 0, i32 1
  %180 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @21, i64 0, i64 0), i64 9, i1 true)
  %181 = extractvalue { i64, %swift.bridge* } %180, 0
  %182 = extractvalue { i64, %swift.bridge* } %180, 1
  %183 = getelementptr inbounds %TSS, %TSS* %178, i32 0, i32 0
  %184 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %183, i32 0, i32 0
  %185 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %184, i32 0, i32 0
  %186 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %185, i32 0, i32 0
  store i64 %181, i64* %186, align 8
  %187 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %184, i32 0, i32 1
  store %swift.bridge* %182, %swift.bridge** %187, align 8
  %188 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 9
  %189 = bitcast [24 x i8]* %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %189)
  %190 = bitcast %TSS* %188 to i8*
  call void @swift_beginAccess(i8* %190, [24 x i8]* %11, i64 32, i8* null) #6
  %191 = getelementptr inbounds %TSS, %TSS* %188, i32 0, i32 0
  %192 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %191, i32 0, i32 0
  %193 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %192, i32 0, i32 0
  %194 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %193, i32 0, i32 0
  %195 = load i64, i64* %194, align 8
  %196 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %192, i32 0, i32 1
  %197 = load %swift.bridge*, %swift.bridge** %196, align 8
  %198 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %197) #6
  call void @swift_endAccess([24 x i8]* %11) #6
  %199 = bitcast [24 x i8]* %11 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %199)
  %200 = call swiftcc %T10Foundation8NSStringC* @"$sSS10FoundationE19_bridgeToObjectiveCAA8NSStringCyF"(i64 %195, %swift.bridge* %197)
  %201 = bitcast %T10Foundation8NSStringC* %200 to %swift.refcounted*
  %202 = getelementptr inbounds %AnyObject, %AnyObject* %179, i32 0, i32 0
  store %swift.refcounted* %201, %swift.refcounted** %202, align 8
  %203 = call swiftcc %swift.bridge* @"$sSD17dictionaryLiteralSDyxq_Gx_q_td_tcfC"(%Ts28__ContiguousArrayStorageBaseC* %15, %swift.type* @"$sSSN", %swift.type* getelementptr inbounds (%swift.full_type, %swift.full_type* @"$syXlN", i32 0, i32 1), i8** @"$sSSSHsWP")
  call void @swift_bridgeObjectRelease(%swift.bridge* %197) #6
  call void @swift_bridgeObjectRelease(%swift.bridge* %89) #6
  call void @swift_bridgeObjectRelease(%swift.bridge* %63) #6
  call void @swift_bridgeObjectRelease(%swift.bridge* %37) #6
  call void asm sideeffect "", "r"(%swift.bridge* %203)
  ret %swift.bridge* %203
}

define dso_local hidden swiftcc %swift.bridge* @"$s4file13KcptunProfileC12toJsonConfigSDySSyXlGyF"(%T4file13KcptunProfileC* swiftself) #0 {
  %2 = alloca %T4file13KcptunProfileC*, align 8
  %3 = bitcast %T4file13KcptunProfileC** %2 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %3, i8 0, i64 8, i1 false)
  %4 = alloca %TSS, align 8
  %5 = bitcast %TSS* %4 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %5, i8 0, i64 16, i1 false)
  %6 = alloca %Ts26DefaultStringInterpolationV, align 8
  %7 = bitcast %Ts26DefaultStringInterpolationV* %6 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %7, i8 0, i64 16, i1 false)
  %8 = alloca %TSS, align 8
  %9 = alloca %TSi, align 8
  %10 = alloca [24 x i8], align 8
  %11 = alloca [24 x i8], align 8
  %12 = alloca [24 x i8], align 8
  %13 = alloca [24 x i8], align 8
  %14 = alloca [24 x i8], align 8
  %15 = alloca [24 x i8], align 8
  %16 = alloca [24 x i8], align 8
  store %T4file13KcptunProfileC* %0, %T4file13KcptunProfileC** %2, align 8
  %17 = call swiftcc %swift.metadata_response @"$s10Foundation12UserDefaultsCMa"(i64 0) #8
  %18 = extractvalue %swift.metadata_response %17, 0
  %19 = bitcast %swift.type* %18 to %T10Foundation12UserDefaultsC* (%swift.type*)**
  %20 = getelementptr inbounds %T10Foundation12UserDefaultsC* (%swift.type*)*, %T10Foundation12UserDefaultsC* (%swift.type*)** %19, i64 34
  %21 = load %T10Foundation12UserDefaultsC* (%swift.type*)*, %T10Foundation12UserDefaultsC* (%swift.type*)** %20, align 8, !invariant.load !25
  %22 = call swiftcc %T10Foundation12UserDefaultsC* %21(%swift.type* swiftself %18)
  call void asm sideeffect "", "r"(%T10Foundation12UserDefaultsC* %22)
  %23 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @22, i64 0, i64 0), i64 16, i1 true)
  %24 = extractvalue { i64, %swift.bridge* } %23, 0
  %25 = extractvalue { i64, %swift.bridge* } %23, 1
  %26 = bitcast %T10Foundation12UserDefaultsC* %22 to %swift.type**
  %27 = load %swift.type*, %swift.type** %26, align 8
  %28 = bitcast %swift.type* %27 to { i64, i64 } (i64, %swift.bridge*, %T10Foundation12UserDefaultsC*)**
  %29 = getelementptr inbounds { i64, i64 } (i64, %swift.bridge*, %T10Foundation12UserDefaultsC*)*, { i64, i64 } (i64, %swift.bridge*, %T10Foundation12UserDefaultsC*)** %28, i64 40
  %30 = load { i64, i64 } (i64, %swift.bridge*, %T10Foundation12UserDefaultsC*)*, { i64, i64 } (i64, %swift.bridge*, %T10Foundation12UserDefaultsC*)** %29, align 8, !invariant.load !25
  %31 = call swiftcc { i64, i64 } %30(i64 %24, %swift.bridge* %25, %T10Foundation12UserDefaultsC* swiftself %22)
  %32 = extractvalue { i64, i64 } %31, 0
  %33 = extractvalue { i64, i64 } %31, 1
  call void @swift_bridgeObjectRelease(%swift.bridge* %25) #6
  %34 = icmp eq i64 %33, 0
  br i1 %34, label %37, label %35

; <label>:35:                                     ; preds = %1
  %36 = inttoptr i64 %33 to %swift.bridge*
  br label %46

; <label>:37:                                     ; preds = %1
  br label %38

; <label>:38:                                     ; preds = %37
  br label %39

; <label>:39:                                     ; preds = %38
  br label %40

; <label>:40:                                     ; preds = %39
  br label %41

; <label>:41:                                     ; preds = %40
  %42 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([58 x i8], [58 x i8]* @25, i64 0, i64 0), i64 57, i1 true)
  %43 = extractvalue { i64, %swift.bridge* } %42, 0
  %44 = extractvalue { i64, %swift.bridge* } %42, 1
  br label %45

; <label>:45:                                     ; preds = %41
  call swiftcc void @"$ss17_assertionFailure__4file4line5flagss5NeverOs12StaticStringV_SSAHSus6UInt32VtF"(i64 ptrtoint ([12 x i8]* @24 to i64), i64 11, i8 2, i64 %43, %swift.bridge* %44, i64 ptrtoint ([11 x i8]* @23 to i64), i64 10, i8 2, i64 71, i32 1)
  unreachable

; <label>:46:                                     ; preds = %35
  %47 = phi i64 [ %32, %35 ]
  %48 = phi %swift.bridge* [ %36, %35 ]
  %49 = bitcast %TSS* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %49)
  %50 = getelementptr inbounds %TSS, %TSS* %4, i32 0, i32 0
  %51 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %50, i32 0, i32 0
  %52 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %51, i32 0, i32 0
  %53 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %52, i32 0, i32 0
  store i64 %47, i64* %53, align 8
  %54 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %51, i32 0, i32 1
  store %swift.bridge* %48, %swift.bridge** %54, align 8
  %55 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @26, i64 0, i64 0), i64 16, i1 true)
  %56 = extractvalue { i64, %swift.bridge* } %55, 0
  %57 = extractvalue { i64, %swift.bridge* } %55, 1
  %58 = bitcast %T10Foundation12UserDefaultsC* %22 to %swift.type**
  %59 = load %swift.type*, %swift.type** %58, align 8
  %60 = bitcast %swift.type* %59 to i64 (i64, %swift.bridge*, %T10Foundation12UserDefaultsC*)**
  %61 = getelementptr inbounds i64 (i64, %swift.bridge*, %T10Foundation12UserDefaultsC*)*, i64 (i64, %swift.bridge*, %T10Foundation12UserDefaultsC*)** %60, i64 45
  %62 = load i64 (i64, %swift.bridge*, %T10Foundation12UserDefaultsC*)*, i64 (i64, %swift.bridge*, %T10Foundation12UserDefaultsC*)** %61, align 8, !invariant.load !25
  %63 = call swiftcc i64 %62(i64 %56, %swift.bridge* %57, %T10Foundation12UserDefaultsC* swiftself %22)
  call void @swift_bridgeObjectRelease(%swift.bridge* %57) #6
  call void asm sideeffect "", "r"(i64 %63)
  %64 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @27, i64 0, i64 0), i64 11, i1 true)
  %65 = extractvalue { i64, %swift.bridge* } %64, 0
  %66 = extractvalue { i64, %swift.bridge* } %64, 1
  %67 = bitcast %T10Foundation12UserDefaultsC* %22 to %swift.type**
  %68 = load %swift.type*, %swift.type** %67, align 8
  %69 = bitcast %swift.type* %68 to i64 (i64, %swift.bridge*, %T10Foundation12UserDefaultsC*)**
  %70 = getelementptr inbounds i64 (i64, %swift.bridge*, %T10Foundation12UserDefaultsC*)*, i64 (i64, %swift.bridge*, %T10Foundation12UserDefaultsC*)** %69, i64 45
  %71 = load i64 (i64, %swift.bridge*, %T10Foundation12UserDefaultsC*)*, i64 (i64, %swift.bridge*, %T10Foundation12UserDefaultsC*)** %70, align 8, !invariant.load !25
  %72 = call swiftcc i64 %71(i64 %65, %swift.bridge* %66, %T10Foundation12UserDefaultsC* swiftself %22)
  call void @swift_bridgeObjectRelease(%swift.bridge* %66) #6
  call void asm sideeffect "", "r"(i64 %72)
  %73 = call swiftcc %swift.metadata_response @"$sSS_yXltMa"(i64 0) #8
  %74 = extractvalue %swift.metadata_response %73, 0
  %75 = call swiftcc { %Ts28__ContiguousArrayStorageBaseC*, i8* } @"$ss27_allocateUninitializedArrayySayxG_BptBwlF"(i64 9, %swift.type* %74)
  %76 = extractvalue { %Ts28__ContiguousArrayStorageBaseC*, i8* } %75, 0
  %77 = extractvalue { %Ts28__ContiguousArrayStorageBaseC*, i8* } %75, 1
  %78 = bitcast i8* %77 to <{ %TSS, %AnyObject }>*
  %79 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %78, i32 0, i32 0
  %80 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %78, i32 0, i32 1
  %81 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @28, i64 0, i64 0), i64 9, i1 true)
  %82 = extractvalue { i64, %swift.bridge* } %81, 0
  %83 = extractvalue { i64, %swift.bridge* } %81, 1
  %84 = getelementptr inbounds %TSS, %TSS* %79, i32 0, i32 0
  %85 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %84, i32 0, i32 0
  %86 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %85, i32 0, i32 0
  %87 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %86, i32 0, i32 0
  store i64 %82, i64* %87, align 8
  %88 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %85, i32 0, i32 1
  store %swift.bridge* %83, %swift.bridge** %88, align 8
  %89 = bitcast %Ts26DefaultStringInterpolationV* %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %89)
  %90 = call swiftcc { i64, %swift.bridge* } @"$ss26DefaultStringInterpolationV15literalCapacity18interpolationCountABSi_SitcfC"(i64 1, i64 2)
  %91 = extractvalue { i64, %swift.bridge* } %90, 0
  %92 = extractvalue { i64, %swift.bridge* } %90, 1
  %93 = getelementptr inbounds %Ts26DefaultStringInterpolationV, %Ts26DefaultStringInterpolationV* %6, i32 0, i32 0
  %94 = getelementptr inbounds %TSS, %TSS* %93, i32 0, i32 0
  %95 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %94, i32 0, i32 0
  %96 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %95, i32 0, i32 0
  %97 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %96, i32 0, i32 0
  store i64 %91, i64* %97, align 8
  %98 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %95, i32 0, i32 1
  store %swift.bridge* %92, %swift.bridge** %98, align 8
  %99 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @13, i64 0, i64 0), i64 0, i1 true)
  %100 = extractvalue { i64, %swift.bridge* } %99, 0
  %101 = extractvalue { i64, %swift.bridge* } %99, 1
  call swiftcc void @"$ss26DefaultStringInterpolationV13appendLiteralyySSF"(i64 %100, %swift.bridge* %101, %Ts26DefaultStringInterpolationV* nocapture swiftself dereferenceable(16) %6)
  call void @swift_bridgeObjectRelease(%swift.bridge* %101) #6
  %102 = bitcast %TSS* %8 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %102)
  %103 = getelementptr inbounds %TSS, %TSS* %8, i32 0, i32 0
  %104 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %103, i32 0, i32 0
  %105 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %104, i32 0, i32 0
  %106 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %105, i32 0, i32 0
  store i64 %47, i64* %106, align 8
  %107 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %104, i32 0, i32 1
  store %swift.bridge* %48, %swift.bridge** %107, align 8
  %108 = bitcast %TSS* %8 to %swift.opaque*
  call swiftcc void @"$ss26DefaultStringInterpolationV06appendC0yyxs06CustomB11ConvertibleRzs20TextOutputStreamableRzlF"(%swift.opaque* noalias nocapture %108, %swift.type* @"$sSSN", i8** @"$sSSs23CustomStringConvertiblesWP", i8** @"$sSSs20TextOutputStreamablesWP", %Ts26DefaultStringInterpolationV* nocapture swiftself dereferenceable(16) %6)
  %109 = bitcast %TSS* %8 to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %109)
  %110 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @29, i64 0, i64 0), i64 1, i1 true)
  %111 = extractvalue { i64, %swift.bridge* } %110, 0
  %112 = extractvalue { i64, %swift.bridge* } %110, 1
  call swiftcc void @"$ss26DefaultStringInterpolationV13appendLiteralyySSF"(i64 %111, %swift.bridge* %112, %Ts26DefaultStringInterpolationV* nocapture swiftself dereferenceable(16) %6)
  call void @swift_bridgeObjectRelease(%swift.bridge* %112) #6
  %113 = bitcast %TSi* %9 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %113)
  %114 = getelementptr inbounds %TSi, %TSi* %9, i32 0, i32 0
  store i64 %63, i64* %114, align 8
  %115 = bitcast %TSi* %9 to %swift.opaque*
  call swiftcc void @"$ss26DefaultStringInterpolationV06appendC0yyxs06CustomB11ConvertibleRzlF"(%swift.opaque* noalias nocapture %115, %swift.type* @"$sSiN", i8** @"$sSis23CustomStringConvertiblesWP", %Ts26DefaultStringInterpolationV* nocapture swiftself dereferenceable(16) %6)
  %116 = bitcast %TSi* %9 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %116)
  %117 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @13, i64 0, i64 0), i64 0, i1 true)
  %118 = extractvalue { i64, %swift.bridge* } %117, 0
  %119 = extractvalue { i64, %swift.bridge* } %117, 1
  call swiftcc void @"$ss26DefaultStringInterpolationV13appendLiteralyySSF"(i64 %118, %swift.bridge* %119, %Ts26DefaultStringInterpolationV* nocapture swiftself dereferenceable(16) %6)
  call void @swift_bridgeObjectRelease(%swift.bridge* %119) #6
  %120 = getelementptr inbounds %Ts26DefaultStringInterpolationV, %Ts26DefaultStringInterpolationV* %6, i32 0, i32 0
  %121 = getelementptr inbounds %TSS, %TSS* %120, i32 0, i32 0
  %122 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %121, i32 0, i32 0
  %123 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %122, i32 0, i32 0
  %124 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %123, i32 0, i32 0
  %125 = load i64, i64* %124, align 8
  %126 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %122, i32 0, i32 1
  %127 = load %swift.bridge*, %swift.bridge** %126, align 8
  %128 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %127) #6
  %129 = call %Ts26DefaultStringInterpolationV* @"$ss26DefaultStringInterpolationVWOh"(%Ts26DefaultStringInterpolationV* %6)
  %130 = bitcast %Ts26DefaultStringInterpolationV* %6 to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %130)
  %131 = call swiftcc { i64, %swift.bridge* } @"$sSS19stringInterpolationSSs013DefaultStringB0V_tcfC"(i64 %125, %swift.bridge* %127)
  %132 = extractvalue { i64, %swift.bridge* } %131, 0
  %133 = extractvalue { i64, %swift.bridge* } %131, 1
  %134 = call swiftcc %T10Foundation8NSStringC* @"$sSS10FoundationE19_bridgeToObjectiveCAA8NSStringCyF"(i64 %132, %swift.bridge* %133)
  %135 = bitcast %T10Foundation8NSStringC* %134 to %swift.refcounted*
  %136 = getelementptr inbounds %AnyObject, %AnyObject* %80, i32 0, i32 0
  store %swift.refcounted* %135, %swift.refcounted** %136, align 8
  %137 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %78, i64 1
  %138 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %137, i32 0, i32 0
  %139 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %137, i32 0, i32 1
  %140 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @14, i64 0, i64 0), i64 4, i1 true)
  %141 = extractvalue { i64, %swift.bridge* } %140, 0
  %142 = extractvalue { i64, %swift.bridge* } %140, 1
  %143 = getelementptr inbounds %TSS, %TSS* %138, i32 0, i32 0
  %144 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %143, i32 0, i32 0
  %145 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %144, i32 0, i32 0
  %146 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %145, i32 0, i32 0
  store i64 %141, i64* %146, align 8
  %147 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %144, i32 0, i32 1
  store %swift.bridge* %142, %swift.bridge** %147, align 8
  %148 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 1
  %149 = bitcast [24 x i8]* %10 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %149)
  %150 = bitcast %TSS* %148 to i8*
  call void @swift_beginAccess(i8* %150, [24 x i8]* %10, i64 32, i8* null) #6
  %151 = getelementptr inbounds %TSS, %TSS* %148, i32 0, i32 0
  %152 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %151, i32 0, i32 0
  %153 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %152, i32 0, i32 0
  %154 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %153, i32 0, i32 0
  %155 = load i64, i64* %154, align 8
  %156 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %152, i32 0, i32 1
  %157 = load %swift.bridge*, %swift.bridge** %156, align 8
  %158 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %157) #6
  call void @swift_endAccess([24 x i8]* %10) #6
  %159 = bitcast [24 x i8]* %10 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %159)
  %160 = call swiftcc %T10Foundation8NSStringC* @"$sSS10FoundationE19_bridgeToObjectiveCAA8NSStringCyF"(i64 %155, %swift.bridge* %157)
  %161 = bitcast %T10Foundation8NSStringC* %160 to %swift.refcounted*
  %162 = getelementptr inbounds %AnyObject, %AnyObject* %139, i32 0, i32 0
  store %swift.refcounted* %161, %swift.refcounted** %162, align 8
  %163 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %78, i64 2
  %164 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %163, i32 0, i32 0
  %165 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %163, i32 0, i32 1
  %166 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @15, i64 0, i64 0), i64 3, i1 true)
  %167 = extractvalue { i64, %swift.bridge* } %166, 0
  %168 = extractvalue { i64, %swift.bridge* } %166, 1
  %169 = getelementptr inbounds %TSS, %TSS* %164, i32 0, i32 0
  %170 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %169, i32 0, i32 0
  %171 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %170, i32 0, i32 0
  %172 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %171, i32 0, i32 0
  store i64 %167, i64* %172, align 8
  %173 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %170, i32 0, i32 1
  store %swift.bridge* %168, %swift.bridge** %173, align 8
  %174 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 2
  %175 = bitcast [24 x i8]* %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %175)
  %176 = bitcast %TSS* %174 to i8*
  call void @swift_beginAccess(i8* %176, [24 x i8]* %11, i64 32, i8* null) #6
  %177 = getelementptr inbounds %TSS, %TSS* %174, i32 0, i32 0
  %178 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %177, i32 0, i32 0
  %179 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %178, i32 0, i32 0
  %180 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %179, i32 0, i32 0
  %181 = load i64, i64* %180, align 8
  %182 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %178, i32 0, i32 1
  %183 = load %swift.bridge*, %swift.bridge** %182, align 8
  %184 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %183) #6
  call void @swift_endAccess([24 x i8]* %11) #6
  %185 = bitcast [24 x i8]* %11 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %185)
  %186 = call swiftcc %T10Foundation8NSStringC* @"$sSS10FoundationE19_bridgeToObjectiveCAA8NSStringCyF"(i64 %181, %swift.bridge* %183)
  %187 = bitcast %T10Foundation8NSStringC* %186 to %swift.refcounted*
  %188 = getelementptr inbounds %AnyObject, %AnyObject* %165, i32 0, i32 0
  store %swift.refcounted* %187, %swift.refcounted** %188, align 8
  %189 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %78, i64 3
  %190 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %189, i32 0, i32 0
  %191 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %189, i32 0, i32 1
  %192 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @16, i64 0, i64 0), i64 5, i1 true)
  %193 = extractvalue { i64, %swift.bridge* } %192, 0
  %194 = extractvalue { i64, %swift.bridge* } %192, 1
  %195 = getelementptr inbounds %TSS, %TSS* %190, i32 0, i32 0
  %196 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %195, i32 0, i32 0
  %197 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %196, i32 0, i32 0
  %198 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %197, i32 0, i32 0
  store i64 %193, i64* %198, align 8
  %199 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %196, i32 0, i32 1
  store %swift.bridge* %194, %swift.bridge** %199, align 8
  %200 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 3
  %201 = bitcast [24 x i8]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %201)
  %202 = bitcast %TSS* %200 to i8*
  call void @swift_beginAccess(i8* %202, [24 x i8]* %12, i64 32, i8* null) #6
  %203 = getelementptr inbounds %TSS, %TSS* %200, i32 0, i32 0
  %204 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %203, i32 0, i32 0
  %205 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %204, i32 0, i32 0
  %206 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %205, i32 0, i32 0
  %207 = load i64, i64* %206, align 8
  %208 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %204, i32 0, i32 1
  %209 = load %swift.bridge*, %swift.bridge** %208, align 8
  %210 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %209) #6
  call void @swift_endAccess([24 x i8]* %12) #6
  %211 = bitcast [24 x i8]* %12 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %211)
  %212 = call swiftcc %T10Foundation8NSStringC* @"$sSS10FoundationE19_bridgeToObjectiveCAA8NSStringCyF"(i64 %207, %swift.bridge* %209)
  %213 = bitcast %T10Foundation8NSStringC* %212 to %swift.refcounted*
  %214 = getelementptr inbounds %AnyObject, %AnyObject* %191, i32 0, i32 0
  store %swift.refcounted* %213, %swift.refcounted** %214, align 8
  %215 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %78, i64 4
  %216 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %215, i32 0, i32 0
  %217 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %215, i32 0, i32 1
  %218 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @19, i64 0, i64 0), i64 6, i1 true)
  %219 = extractvalue { i64, %swift.bridge* } %218, 0
  %220 = extractvalue { i64, %swift.bridge* } %218, 1
  %221 = getelementptr inbounds %TSS, %TSS* %216, i32 0, i32 0
  %222 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %221, i32 0, i32 0
  %223 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %222, i32 0, i32 0
  %224 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %223, i32 0, i32 0
  store i64 %219, i64* %224, align 8
  %225 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %222, i32 0, i32 1
  store %swift.bridge* %220, %swift.bridge** %225, align 8
  %226 = call swiftcc %swift.metadata_response @"$s10Foundation8NSNumberCMa"(i64 0) #8
  %227 = extractvalue %swift.metadata_response %226, 0
  %228 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 4
  %229 = bitcast [24 x i8]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %229)
  %230 = bitcast %TSb* %228 to i8*
  call void @swift_beginAccess(i8* %230, [24 x i8]* %13, i64 32, i8* null) #6
  %231 = getelementptr inbounds %TSb, %TSb* %228, i32 0, i32 0
  %232 = load i1, i1* %231, align 8
  call void @swift_endAccess([24 x i8]* %13) #6
  %233 = bitcast [24 x i8]* %13 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %233)
  %234 = call swiftcc %T10Foundation8NSNumberC* @"$s10Foundation8NSNumberC5valueACSb_tcfC"(i1 %232, %swift.type* swiftself %227)
  %235 = bitcast %T10Foundation8NSNumberC* %234 to %swift.refcounted*
  %236 = getelementptr inbounds %AnyObject, %AnyObject* %217, i32 0, i32 0
  store %swift.refcounted* %235, %swift.refcounted** %236, align 8
  %237 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %78, i64 5
  %238 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %237, i32 0, i32 0
  %239 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %237, i32 0, i32 1
  %240 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @17, i64 0, i64 0), i64 9, i1 true)
  %241 = extractvalue { i64, %swift.bridge* } %240, 0
  %242 = extractvalue { i64, %swift.bridge* } %240, 1
  %243 = getelementptr inbounds %TSS, %TSS* %238, i32 0, i32 0
  %244 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %243, i32 0, i32 0
  %245 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %244, i32 0, i32 0
  %246 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %245, i32 0, i32 0
  store i64 %241, i64* %246, align 8
  %247 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %244, i32 0, i32 1
  store %swift.bridge* %242, %swift.bridge** %247, align 8
  %248 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 6
  %249 = bitcast [24 x i8]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %249)
  %250 = bitcast %Ts6UInt32V* %248 to i8*
  call void @swift_beginAccess(i8* %250, [24 x i8]* %14, i64 32, i8* null) #6
  %251 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %248, i32 0, i32 0
  %252 = load i32, i32* %251, align 4
  call void @swift_endAccess([24 x i8]* %14) #6
  %253 = bitcast [24 x i8]* %14 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %253)
  %254 = call swiftcc %T10Foundation8NSNumberC* @"$s10Foundation8NSNumberC5valueACs6UInt32V_tcfC"(i32 %252, %swift.type* swiftself %227)
  %255 = bitcast %T10Foundation8NSNumberC* %254 to %swift.refcounted*
  %256 = getelementptr inbounds %AnyObject, %AnyObject* %239, i32 0, i32 0
  store %swift.refcounted* %255, %swift.refcounted** %256, align 8
  %257 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %78, i64 6
  %258 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %257, i32 0, i32 0
  %259 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %257, i32 0, i32 1
  %260 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @18, i64 0, i64 0), i64 11, i1 true)
  %261 = extractvalue { i64, %swift.bridge* } %260, 0
  %262 = extractvalue { i64, %swift.bridge* } %260, 1
  %263 = getelementptr inbounds %TSS, %TSS* %258, i32 0, i32 0
  %264 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %263, i32 0, i32 0
  %265 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %264, i32 0, i32 0
  %266 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %265, i32 0, i32 0
  store i64 %261, i64* %266, align 8
  %267 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %264, i32 0, i32 1
  store %swift.bridge* %262, %swift.bridge** %267, align 8
  %268 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 7
  %269 = bitcast [24 x i8]* %15 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %269)
  %270 = bitcast %Ts6UInt32V* %268 to i8*
  call void @swift_beginAccess(i8* %270, [24 x i8]* %15, i64 32, i8* null) #6
  %271 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %268, i32 0, i32 0
  %272 = load i32, i32* %271, align 8
  call void @swift_endAccess([24 x i8]* %15) #6
  %273 = bitcast [24 x i8]* %15 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %273)
  %274 = call swiftcc %T10Foundation8NSNumberC* @"$s10Foundation8NSNumberC5valueACs6UInt32V_tcfC"(i32 %272, %swift.type* swiftself %227)
  %275 = bitcast %T10Foundation8NSNumberC* %274 to %swift.refcounted*
  %276 = getelementptr inbounds %AnyObject, %AnyObject* %259, i32 0, i32 0
  store %swift.refcounted* %275, %swift.refcounted** %276, align 8
  %277 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %78, i64 7
  %278 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %277, i32 0, i32 0
  %279 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %277, i32 0, i32 1
  %280 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @20, i64 0, i64 0), i64 3, i1 true)
  %281 = extractvalue { i64, %swift.bridge* } %280, 0
  %282 = extractvalue { i64, %swift.bridge* } %280, 1
  %283 = getelementptr inbounds %TSS, %TSS* %278, i32 0, i32 0
  %284 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %283, i32 0, i32 0
  %285 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %284, i32 0, i32 0
  %286 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %285, i32 0, i32 0
  store i64 %281, i64* %286, align 8
  %287 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %284, i32 0, i32 1
  store %swift.bridge* %282, %swift.bridge** %287, align 8
  %288 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 8
  %289 = bitcast [24 x i8]* %16 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %289)
  %290 = bitcast %Ts6UInt32V* %288 to i8*
  call void @swift_beginAccess(i8* %290, [24 x i8]* %16, i64 32, i8* null) #6
  %291 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %288, i32 0, i32 0
  %292 = load i32, i32* %291, align 4
  call void @swift_endAccess([24 x i8]* %16) #6
  %293 = bitcast [24 x i8]* %16 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %293)
  %294 = call swiftcc %T10Foundation8NSNumberC* @"$s10Foundation8NSNumberC5valueACs6UInt32V_tcfC"(i32 %292, %swift.type* swiftself %227)
  %295 = bitcast %T10Foundation8NSNumberC* %294 to %swift.refcounted*
  %296 = getelementptr inbounds %AnyObject, %AnyObject* %279, i32 0, i32 0
  store %swift.refcounted* %295, %swift.refcounted** %296, align 8
  %297 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %78, i64 8
  %298 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %297, i32 0, i32 0
  %299 = getelementptr inbounds <{ %TSS, %AnyObject }>, <{ %TSS, %AnyObject }>* %297, i32 0, i32 1
  %300 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @30, i64 0, i64 0), i64 4, i1 true)
  %301 = extractvalue { i64, %swift.bridge* } %300, 0
  %302 = extractvalue { i64, %swift.bridge* } %300, 1
  %303 = getelementptr inbounds %TSS, %TSS* %298, i32 0, i32 0
  %304 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %303, i32 0, i32 0
  %305 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %304, i32 0, i32 0
  %306 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %305, i32 0, i32 0
  store i64 %301, i64* %306, align 8
  %307 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %304, i32 0, i32 1
  store %swift.bridge* %302, %swift.bridge** %307, align 8
  %308 = call swiftcc %T10Foundation8NSNumberC* @"$s10Foundation8NSNumberC5valueACSi_tcfC"(i64 %72, %swift.type* swiftself %227)
  %309 = bitcast %T10Foundation8NSNumberC* %308 to %swift.refcounted*
  %310 = getelementptr inbounds %AnyObject, %AnyObject* %299, i32 0, i32 0
  store %swift.refcounted* %309, %swift.refcounted** %310, align 8
  %311 = call swiftcc %swift.bridge* @"$sSD17dictionaryLiteralSDyxq_Gx_q_td_tcfC"(%Ts28__ContiguousArrayStorageBaseC* %76, %swift.type* @"$sSSN", %swift.type* getelementptr inbounds (%swift.full_type, %swift.full_type* @"$syXlN", i32 0, i32 1), i8** @"$sSSSHsWP")
  call void @swift_bridgeObjectRelease(%swift.bridge* %209) #6
  call void @swift_bridgeObjectRelease(%swift.bridge* %183) #6
  call void @swift_bridgeObjectRelease(%swift.bridge* %157) #6
  call void @swift_bridgeObjectRelease(%swift.bridge* %133) #6
  call void asm sideeffect "", "r"(%swift.bridge* %311)
  call void @swift_bridgeObjectRelease(%swift.bridge* %48) #6
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T10Foundation12UserDefaultsC*)*)(%T10Foundation12UserDefaultsC* %22) #6
  ret %swift.bridge* %311
}

define dso_local hidden swiftcc %Ts28__ContiguousArrayStorageBaseC* @"$s4file13KcptunProfileC13urlQueryItemsSay10Foundation12URLQueryItemVGyF"(%T4file13KcptunProfileC* swiftself) #0 {
  %2 = alloca %T4file13KcptunProfileC*, align 8
  %3 = bitcast %T4file13KcptunProfileC** %2 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %3, i8 0, i64 8, i1 false)
  %4 = alloca [24 x i8], align 8
  %5 = alloca [24 x i8], align 8
  %6 = alloca [24 x i8], align 8
  %7 = alloca %Ts26DefaultStringInterpolationV, align 8
  %8 = bitcast %Ts26DefaultStringInterpolationV* %7 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %8, i8 0, i64 16, i1 false)
  %9 = alloca [24 x i8], align 8
  %10 = alloca %Ts6UInt32V, align 4
  %11 = alloca %Ts26DefaultStringInterpolationV, align 8
  %12 = bitcast %Ts26DefaultStringInterpolationV* %11 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %12, i8 0, i64 16, i1 false)
  %13 = alloca [24 x i8], align 8
  %14 = alloca %Ts6UInt32V, align 4
  %15 = alloca [24 x i8], align 8
  %16 = alloca %Ts26DefaultStringInterpolationV, align 8
  %17 = bitcast %Ts26DefaultStringInterpolationV* %16 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %17, i8 0, i64 16, i1 false)
  %18 = alloca [24 x i8], align 8
  %19 = alloca %Ts6UInt32V, align 4
  %20 = alloca [24 x i8], align 8
  store %T4file13KcptunProfileC* %0, %T4file13KcptunProfileC** %2, align 8
  %21 = call swiftcc { %Ts28__ContiguousArrayStorageBaseC*, i8* } @"$ss27_allocateUninitializedArrayySayxG_BptBwlF"(i64 8, %swift.type* @"$s10Foundation12URLQueryItemVN")
  %22 = extractvalue { %Ts28__ContiguousArrayStorageBaseC*, i8* } %21, 0
  %23 = extractvalue { %Ts28__ContiguousArrayStorageBaseC*, i8* } %21, 1
  %24 = bitcast i8* %23 to %T10Foundation12URLQueryItemV*
  %25 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @14, i64 0, i64 0), i64 4, i1 true)
  %26 = extractvalue { i64, %swift.bridge* } %25, 0
  %27 = extractvalue { i64, %swift.bridge* } %25, 1
  %28 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 1
  %29 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %29)
  %30 = bitcast %TSS* %28 to i8*
  call void @swift_beginAccess(i8* %30, [24 x i8]* %4, i64 32, i8* null) #6
  %31 = getelementptr inbounds %TSS, %TSS* %28, i32 0, i32 0
  %32 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %31, i32 0, i32 0
  %33 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %32, i32 0, i32 0
  %34 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %33, i32 0, i32 0
  %35 = load i64, i64* %34, align 8
  %36 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %32, i32 0, i32 1
  %37 = load %swift.bridge*, %swift.bridge** %36, align 8
  %38 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %37) #6
  call void @swift_endAccess([24 x i8]* %4) #6
  %39 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %39)
  %40 = ptrtoint %swift.bridge* %37 to i64
  %41 = call swiftcc %T10Foundation14NSURLQueryItemC* @"$s10Foundation12URLQueryItemV4name5valueACSS_SSSgtcfC"(i64 %26, %swift.bridge* %27, i64 %35, i64 %40)
  %42 = getelementptr inbounds %T10Foundation12URLQueryItemV, %T10Foundation12URLQueryItemV* %24, i32 0, i32 0
  store %T10Foundation14NSURLQueryItemC* %41, %T10Foundation14NSURLQueryItemC** %42, align 8
  %43 = getelementptr inbounds %T10Foundation12URLQueryItemV, %T10Foundation12URLQueryItemV* %24, i64 1
  %44 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @15, i64 0, i64 0), i64 3, i1 true)
  %45 = extractvalue { i64, %swift.bridge* } %44, 0
  %46 = extractvalue { i64, %swift.bridge* } %44, 1
  %47 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 2
  %48 = bitcast [24 x i8]* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %48)
  %49 = bitcast %TSS* %47 to i8*
  call void @swift_beginAccess(i8* %49, [24 x i8]* %5, i64 32, i8* null) #6
  %50 = getelementptr inbounds %TSS, %TSS* %47, i32 0, i32 0
  %51 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %50, i32 0, i32 0
  %52 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %51, i32 0, i32 0
  %53 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %52, i32 0, i32 0
  %54 = load i64, i64* %53, align 8
  %55 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %51, i32 0, i32 1
  %56 = load %swift.bridge*, %swift.bridge** %55, align 8
  %57 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %56) #6
  call void @swift_endAccess([24 x i8]* %5) #6
  %58 = bitcast [24 x i8]* %5 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %58)
  %59 = ptrtoint %swift.bridge* %56 to i64
  %60 = call swiftcc %T10Foundation14NSURLQueryItemC* @"$s10Foundation12URLQueryItemV4name5valueACSS_SSSgtcfC"(i64 %45, %swift.bridge* %46, i64 %54, i64 %59)
  %61 = getelementptr inbounds %T10Foundation12URLQueryItemV, %T10Foundation12URLQueryItemV* %43, i32 0, i32 0
  store %T10Foundation14NSURLQueryItemC* %60, %T10Foundation14NSURLQueryItemC** %61, align 8
  %62 = getelementptr inbounds %T10Foundation12URLQueryItemV, %T10Foundation12URLQueryItemV* %24, i64 2
  %63 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @16, i64 0, i64 0), i64 5, i1 true)
  %64 = extractvalue { i64, %swift.bridge* } %63, 0
  %65 = extractvalue { i64, %swift.bridge* } %63, 1
  %66 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 3
  %67 = bitcast [24 x i8]* %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %67)
  %68 = bitcast %TSS* %66 to i8*
  call void @swift_beginAccess(i8* %68, [24 x i8]* %6, i64 32, i8* null) #6
  %69 = getelementptr inbounds %TSS, %TSS* %66, i32 0, i32 0
  %70 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %69, i32 0, i32 0
  %71 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %70, i32 0, i32 0
  %72 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %71, i32 0, i32 0
  %73 = load i64, i64* %72, align 8
  %74 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %70, i32 0, i32 1
  %75 = load %swift.bridge*, %swift.bridge** %74, align 8
  %76 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %75) #6
  call void @swift_endAccess([24 x i8]* %6) #6
  %77 = bitcast [24 x i8]* %6 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %77)
  %78 = ptrtoint %swift.bridge* %75 to i64
  %79 = call swiftcc %T10Foundation14NSURLQueryItemC* @"$s10Foundation12URLQueryItemV4name5valueACSS_SSSgtcfC"(i64 %64, %swift.bridge* %65, i64 %73, i64 %78)
  %80 = getelementptr inbounds %T10Foundation12URLQueryItemV, %T10Foundation12URLQueryItemV* %62, i32 0, i32 0
  store %T10Foundation14NSURLQueryItemC* %79, %T10Foundation14NSURLQueryItemC** %80, align 8
  %81 = getelementptr inbounds %T10Foundation12URLQueryItemV, %T10Foundation12URLQueryItemV* %24, i64 3
  %82 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @17, i64 0, i64 0), i64 9, i1 true)
  %83 = extractvalue { i64, %swift.bridge* } %82, 0
  %84 = extractvalue { i64, %swift.bridge* } %82, 1
  %85 = bitcast %Ts26DefaultStringInterpolationV* %7 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %85)
  %86 = call swiftcc { i64, %swift.bridge* } @"$ss26DefaultStringInterpolationV15literalCapacity18interpolationCountABSi_SitcfC"(i64 0, i64 1)
  %87 = extractvalue { i64, %swift.bridge* } %86, 0
  %88 = extractvalue { i64, %swift.bridge* } %86, 1
  %89 = getelementptr inbounds %Ts26DefaultStringInterpolationV, %Ts26DefaultStringInterpolationV* %7, i32 0, i32 0
  %90 = getelementptr inbounds %TSS, %TSS* %89, i32 0, i32 0
  %91 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %90, i32 0, i32 0
  %92 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %91, i32 0, i32 0
  %93 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %92, i32 0, i32 0
  store i64 %87, i64* %93, align 8
  %94 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %91, i32 0, i32 1
  store %swift.bridge* %88, %swift.bridge** %94, align 8
  %95 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @13, i64 0, i64 0), i64 0, i1 true)
  %96 = extractvalue { i64, %swift.bridge* } %95, 0
  %97 = extractvalue { i64, %swift.bridge* } %95, 1
  call swiftcc void @"$ss26DefaultStringInterpolationV13appendLiteralyySSF"(i64 %96, %swift.bridge* %97, %Ts26DefaultStringInterpolationV* nocapture swiftself dereferenceable(16) %7)
  call void @swift_bridgeObjectRelease(%swift.bridge* %97) #6
  %98 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 6
  %99 = bitcast [24 x i8]* %9 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %99)
  %100 = bitcast %Ts6UInt32V* %98 to i8*
  call void @swift_beginAccess(i8* %100, [24 x i8]* %9, i64 32, i8* null) #6
  %101 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %98, i32 0, i32 0
  %102 = load i32, i32* %101, align 4
  call void @swift_endAccess([24 x i8]* %9) #6
  %103 = bitcast [24 x i8]* %9 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %103)
  %104 = bitcast %Ts6UInt32V* %10 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %104)
  %105 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %10, i32 0, i32 0
  store i32 %102, i32* %105, align 4
  %106 = bitcast %Ts6UInt32V* %10 to %swift.opaque*
  call swiftcc void @"$ss26DefaultStringInterpolationV06appendC0yyxs06CustomB11ConvertibleRzlF"(%swift.opaque* noalias nocapture %106, %swift.type* @"$ss6UInt32VN", i8** @"$ss6UInt32Vs23CustomStringConvertiblesWP", %Ts26DefaultStringInterpolationV* nocapture swiftself dereferenceable(16) %7)
  %107 = bitcast %Ts6UInt32V* %10 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %107)
  %108 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @13, i64 0, i64 0), i64 0, i1 true)
  %109 = extractvalue { i64, %swift.bridge* } %108, 0
  %110 = extractvalue { i64, %swift.bridge* } %108, 1
  call swiftcc void @"$ss26DefaultStringInterpolationV13appendLiteralyySSF"(i64 %109, %swift.bridge* %110, %Ts26DefaultStringInterpolationV* nocapture swiftself dereferenceable(16) %7)
  call void @swift_bridgeObjectRelease(%swift.bridge* %110) #6
  %111 = getelementptr inbounds %Ts26DefaultStringInterpolationV, %Ts26DefaultStringInterpolationV* %7, i32 0, i32 0
  %112 = getelementptr inbounds %TSS, %TSS* %111, i32 0, i32 0
  %113 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %112, i32 0, i32 0
  %114 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %113, i32 0, i32 0
  %115 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %114, i32 0, i32 0
  %116 = load i64, i64* %115, align 8
  %117 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %113, i32 0, i32 1
  %118 = load %swift.bridge*, %swift.bridge** %117, align 8
  %119 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %118) #6
  %120 = call %Ts26DefaultStringInterpolationV* @"$ss26DefaultStringInterpolationVWOh"(%Ts26DefaultStringInterpolationV* %7)
  %121 = bitcast %Ts26DefaultStringInterpolationV* %7 to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %121)
  %122 = call swiftcc { i64, %swift.bridge* } @"$sSS19stringInterpolationSSs013DefaultStringB0V_tcfC"(i64 %116, %swift.bridge* %118)
  %123 = extractvalue { i64, %swift.bridge* } %122, 0
  %124 = extractvalue { i64, %swift.bridge* } %122, 1
  %125 = ptrtoint %swift.bridge* %124 to i64
  %126 = call swiftcc %T10Foundation14NSURLQueryItemC* @"$s10Foundation12URLQueryItemV4name5valueACSS_SSSgtcfC"(i64 %83, %swift.bridge* %84, i64 %123, i64 %125)
  %127 = getelementptr inbounds %T10Foundation12URLQueryItemV, %T10Foundation12URLQueryItemV* %81, i32 0, i32 0
  store %T10Foundation14NSURLQueryItemC* %126, %T10Foundation14NSURLQueryItemC** %127, align 8
  %128 = getelementptr inbounds %T10Foundation12URLQueryItemV, %T10Foundation12URLQueryItemV* %24, i64 4
  %129 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @18, i64 0, i64 0), i64 11, i1 true)
  %130 = extractvalue { i64, %swift.bridge* } %129, 0
  %131 = extractvalue { i64, %swift.bridge* } %129, 1
  %132 = bitcast %Ts26DefaultStringInterpolationV* %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %132)
  %133 = call swiftcc { i64, %swift.bridge* } @"$ss26DefaultStringInterpolationV15literalCapacity18interpolationCountABSi_SitcfC"(i64 0, i64 1)
  %134 = extractvalue { i64, %swift.bridge* } %133, 0
  %135 = extractvalue { i64, %swift.bridge* } %133, 1
  %136 = getelementptr inbounds %Ts26DefaultStringInterpolationV, %Ts26DefaultStringInterpolationV* %11, i32 0, i32 0
  %137 = getelementptr inbounds %TSS, %TSS* %136, i32 0, i32 0
  %138 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %137, i32 0, i32 0
  %139 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %138, i32 0, i32 0
  %140 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %139, i32 0, i32 0
  store i64 %134, i64* %140, align 8
  %141 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %138, i32 0, i32 1
  store %swift.bridge* %135, %swift.bridge** %141, align 8
  %142 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @13, i64 0, i64 0), i64 0, i1 true)
  %143 = extractvalue { i64, %swift.bridge* } %142, 0
  %144 = extractvalue { i64, %swift.bridge* } %142, 1
  call swiftcc void @"$ss26DefaultStringInterpolationV13appendLiteralyySSF"(i64 %143, %swift.bridge* %144, %Ts26DefaultStringInterpolationV* nocapture swiftself dereferenceable(16) %11)
  call void @swift_bridgeObjectRelease(%swift.bridge* %144) #6
  %145 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 7
  %146 = bitcast [24 x i8]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %146)
  %147 = bitcast %Ts6UInt32V* %145 to i8*
  call void @swift_beginAccess(i8* %147, [24 x i8]* %13, i64 32, i8* null) #6
  %148 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %145, i32 0, i32 0
  %149 = load i32, i32* %148, align 8
  call void @swift_endAccess([24 x i8]* %13) #6
  %150 = bitcast [24 x i8]* %13 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %150)
  %151 = bitcast %Ts6UInt32V* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %151)
  %152 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %14, i32 0, i32 0
  store i32 %149, i32* %152, align 4
  %153 = bitcast %Ts6UInt32V* %14 to %swift.opaque*
  call swiftcc void @"$ss26DefaultStringInterpolationV06appendC0yyxs06CustomB11ConvertibleRzlF"(%swift.opaque* noalias nocapture %153, %swift.type* @"$ss6UInt32VN", i8** @"$ss6UInt32Vs23CustomStringConvertiblesWP", %Ts26DefaultStringInterpolationV* nocapture swiftself dereferenceable(16) %11)
  %154 = bitcast %Ts6UInt32V* %14 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %154)
  %155 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @13, i64 0, i64 0), i64 0, i1 true)
  %156 = extractvalue { i64, %swift.bridge* } %155, 0
  %157 = extractvalue { i64, %swift.bridge* } %155, 1
  call swiftcc void @"$ss26DefaultStringInterpolationV13appendLiteralyySSF"(i64 %156, %swift.bridge* %157, %Ts26DefaultStringInterpolationV* nocapture swiftself dereferenceable(16) %11)
  call void @swift_bridgeObjectRelease(%swift.bridge* %157) #6
  %158 = getelementptr inbounds %Ts26DefaultStringInterpolationV, %Ts26DefaultStringInterpolationV* %11, i32 0, i32 0
  %159 = getelementptr inbounds %TSS, %TSS* %158, i32 0, i32 0
  %160 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %159, i32 0, i32 0
  %161 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %160, i32 0, i32 0
  %162 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %161, i32 0, i32 0
  %163 = load i64, i64* %162, align 8
  %164 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %160, i32 0, i32 1
  %165 = load %swift.bridge*, %swift.bridge** %164, align 8
  %166 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %165) #6
  %167 = call %Ts26DefaultStringInterpolationV* @"$ss26DefaultStringInterpolationVWOh"(%Ts26DefaultStringInterpolationV* %11)
  %168 = bitcast %Ts26DefaultStringInterpolationV* %11 to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %168)
  %169 = call swiftcc { i64, %swift.bridge* } @"$sSS19stringInterpolationSSs013DefaultStringB0V_tcfC"(i64 %163, %swift.bridge* %165)
  %170 = extractvalue { i64, %swift.bridge* } %169, 0
  %171 = extractvalue { i64, %swift.bridge* } %169, 1
  %172 = ptrtoint %swift.bridge* %171 to i64
  %173 = call swiftcc %T10Foundation14NSURLQueryItemC* @"$s10Foundation12URLQueryItemV4name5valueACSS_SSSgtcfC"(i64 %130, %swift.bridge* %131, i64 %170, i64 %172)
  %174 = getelementptr inbounds %T10Foundation12URLQueryItemV, %T10Foundation12URLQueryItemV* %128, i32 0, i32 0
  store %T10Foundation14NSURLQueryItemC* %173, %T10Foundation14NSURLQueryItemC** %174, align 8
  %175 = getelementptr inbounds %T10Foundation12URLQueryItemV, %T10Foundation12URLQueryItemV* %24, i64 5
  %176 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @19, i64 0, i64 0), i64 6, i1 true)
  %177 = extractvalue { i64, %swift.bridge* } %176, 0
  %178 = extractvalue { i64, %swift.bridge* } %176, 1
  %179 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 4
  %180 = bitcast [24 x i8]* %15 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %180)
  %181 = bitcast %TSb* %179 to i8*
  call void @swift_beginAccess(i8* %181, [24 x i8]* %15, i64 32, i8* null) #6
  %182 = getelementptr inbounds %TSb, %TSb* %179, i32 0, i32 0
  %183 = load i1, i1* %182, align 8
  call void @swift_endAccess([24 x i8]* %15) #6
  %184 = bitcast [24 x i8]* %15 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %184)
  %185 = call swiftcc { i64, %swift.bridge* } @"$sSb11descriptionSSvg"(i1 %183)
  %186 = extractvalue { i64, %swift.bridge* } %185, 0
  %187 = extractvalue { i64, %swift.bridge* } %185, 1
  %188 = ptrtoint %swift.bridge* %187 to i64
  %189 = call swiftcc %T10Foundation14NSURLQueryItemC* @"$s10Foundation12URLQueryItemV4name5valueACSS_SSSgtcfC"(i64 %177, %swift.bridge* %178, i64 %186, i64 %188)
  %190 = getelementptr inbounds %T10Foundation12URLQueryItemV, %T10Foundation12URLQueryItemV* %175, i32 0, i32 0
  store %T10Foundation14NSURLQueryItemC* %189, %T10Foundation14NSURLQueryItemC** %190, align 8
  %191 = getelementptr inbounds %T10Foundation12URLQueryItemV, %T10Foundation12URLQueryItemV* %24, i64 6
  %192 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @20, i64 0, i64 0), i64 3, i1 true)
  %193 = extractvalue { i64, %swift.bridge* } %192, 0
  %194 = extractvalue { i64, %swift.bridge* } %192, 1
  %195 = bitcast %Ts26DefaultStringInterpolationV* %16 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %195)
  %196 = call swiftcc { i64, %swift.bridge* } @"$ss26DefaultStringInterpolationV15literalCapacity18interpolationCountABSi_SitcfC"(i64 0, i64 1)
  %197 = extractvalue { i64, %swift.bridge* } %196, 0
  %198 = extractvalue { i64, %swift.bridge* } %196, 1
  %199 = getelementptr inbounds %Ts26DefaultStringInterpolationV, %Ts26DefaultStringInterpolationV* %16, i32 0, i32 0
  %200 = getelementptr inbounds %TSS, %TSS* %199, i32 0, i32 0
  %201 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %200, i32 0, i32 0
  %202 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %201, i32 0, i32 0
  %203 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %202, i32 0, i32 0
  store i64 %197, i64* %203, align 8
  %204 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %201, i32 0, i32 1
  store %swift.bridge* %198, %swift.bridge** %204, align 8
  %205 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @13, i64 0, i64 0), i64 0, i1 true)
  %206 = extractvalue { i64, %swift.bridge* } %205, 0
  %207 = extractvalue { i64, %swift.bridge* } %205, 1
  call swiftcc void @"$ss26DefaultStringInterpolationV13appendLiteralyySSF"(i64 %206, %swift.bridge* %207, %Ts26DefaultStringInterpolationV* nocapture swiftself dereferenceable(16) %16)
  call void @swift_bridgeObjectRelease(%swift.bridge* %207) #6
  %208 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 8
  %209 = bitcast [24 x i8]* %18 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %209)
  %210 = bitcast %Ts6UInt32V* %208 to i8*
  call void @swift_beginAccess(i8* %210, [24 x i8]* %18, i64 32, i8* null) #6
  %211 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %208, i32 0, i32 0
  %212 = load i32, i32* %211, align 4
  call void @swift_endAccess([24 x i8]* %18) #6
  %213 = bitcast [24 x i8]* %18 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %213)
  %214 = bitcast %Ts6UInt32V* %19 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %214)
  %215 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %19, i32 0, i32 0
  store i32 %212, i32* %215, align 4
  %216 = bitcast %Ts6UInt32V* %19 to %swift.opaque*
  call swiftcc void @"$ss26DefaultStringInterpolationV06appendC0yyxs06CustomB11ConvertibleRzlF"(%swift.opaque* noalias nocapture %216, %swift.type* @"$ss6UInt32VN", i8** @"$ss6UInt32Vs23CustomStringConvertiblesWP", %Ts26DefaultStringInterpolationV* nocapture swiftself dereferenceable(16) %16)
  %217 = bitcast %Ts6UInt32V* %19 to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %217)
  %218 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @13, i64 0, i64 0), i64 0, i1 true)
  %219 = extractvalue { i64, %swift.bridge* } %218, 0
  %220 = extractvalue { i64, %swift.bridge* } %218, 1
  call swiftcc void @"$ss26DefaultStringInterpolationV13appendLiteralyySSF"(i64 %219, %swift.bridge* %220, %Ts26DefaultStringInterpolationV* nocapture swiftself dereferenceable(16) %16)
  call void @swift_bridgeObjectRelease(%swift.bridge* %220) #6
  %221 = getelementptr inbounds %Ts26DefaultStringInterpolationV, %Ts26DefaultStringInterpolationV* %16, i32 0, i32 0
  %222 = getelementptr inbounds %TSS, %TSS* %221, i32 0, i32 0
  %223 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %222, i32 0, i32 0
  %224 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %223, i32 0, i32 0
  %225 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %224, i32 0, i32 0
  %226 = load i64, i64* %225, align 8
  %227 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %223, i32 0, i32 1
  %228 = load %swift.bridge*, %swift.bridge** %227, align 8
  %229 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %228) #6
  %230 = call %Ts26DefaultStringInterpolationV* @"$ss26DefaultStringInterpolationVWOh"(%Ts26DefaultStringInterpolationV* %16)
  %231 = bitcast %Ts26DefaultStringInterpolationV* %16 to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %231)
  %232 = call swiftcc { i64, %swift.bridge* } @"$sSS19stringInterpolationSSs013DefaultStringB0V_tcfC"(i64 %226, %swift.bridge* %228)
  %233 = extractvalue { i64, %swift.bridge* } %232, 0
  %234 = extractvalue { i64, %swift.bridge* } %232, 1
  %235 = ptrtoint %swift.bridge* %234 to i64
  %236 = call swiftcc %T10Foundation14NSURLQueryItemC* @"$s10Foundation12URLQueryItemV4name5valueACSS_SSSgtcfC"(i64 %193, %swift.bridge* %194, i64 %233, i64 %235)
  %237 = getelementptr inbounds %T10Foundation12URLQueryItemV, %T10Foundation12URLQueryItemV* %191, i32 0, i32 0
  store %T10Foundation14NSURLQueryItemC* %236, %T10Foundation14NSURLQueryItemC** %237, align 8
  %238 = getelementptr inbounds %T10Foundation12URLQueryItemV, %T10Foundation12URLQueryItemV* %24, i64 7
  %239 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @21, i64 0, i64 0), i64 9, i1 true)
  %240 = extractvalue { i64, %swift.bridge* } %239, 0
  %241 = extractvalue { i64, %swift.bridge* } %239, 1
  %242 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 9
  %243 = bitcast [24 x i8]* %20 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %243)
  %244 = bitcast %TSS* %242 to i8*
  call void @swift_beginAccess(i8* %244, [24 x i8]* %20, i64 32, i8* null) #6
  %245 = getelementptr inbounds %TSS, %TSS* %242, i32 0, i32 0
  %246 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %245, i32 0, i32 0
  %247 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %246, i32 0, i32 0
  %248 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %247, i32 0, i32 0
  %249 = load i64, i64* %248, align 8
  %250 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %246, i32 0, i32 1
  %251 = load %swift.bridge*, %swift.bridge** %250, align 8
  %252 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %251) #6
  call void @swift_endAccess([24 x i8]* %20) #6
  %253 = bitcast [24 x i8]* %20 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %253)
  %254 = ptrtoint %swift.bridge* %251 to i64
  %255 = call swiftcc %T10Foundation14NSURLQueryItemC* @"$s10Foundation12URLQueryItemV4name5valueACSS_SSSgtcfC"(i64 %240, %swift.bridge* %241, i64 %249, i64 %254)
  %256 = getelementptr inbounds %T10Foundation12URLQueryItemV, %T10Foundation12URLQueryItemV* %238, i32 0, i32 0
  store %T10Foundation14NSURLQueryItemC* %255, %T10Foundation14NSURLQueryItemC** %256, align 8
  %257 = call swiftcc %Ts28__ContiguousArrayStorageBaseC* @"$sSa12arrayLiteralSayxGxd_tcfC"(%Ts28__ContiguousArrayStorageBaseC* %22, %swift.type* @"$s10Foundation12URLQueryItemVN")
  ret %Ts28__ContiguousArrayStorageBaseC* %257
}

define dso_local hidden swiftcc void @"$s4file13KcptunProfileC17loadUrlQueryItems5itemsySay10Foundation12URLQueryItemVG_tF"(%Ts28__ContiguousArrayStorageBaseC*, %T4file13KcptunProfileC* swiftself) #0 {
  %3 = alloca %Ts28__ContiguousArrayStorageBaseC*, align 8
  %4 = bitcast %Ts28__ContiguousArrayStorageBaseC** %3 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %4, i8 0, i64 8, i1 false)
  %5 = alloca %T4file13KcptunProfileC*, align 8
  %6 = bitcast %T4file13KcptunProfileC** %5 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %6, i8 0, i64 8, i1 false)
  %7 = alloca %Ts16IndexingIteratorVySay10Foundation12URLQueryItemVGG, align 8
  %8 = bitcast %Ts16IndexingIteratorVySay10Foundation12URLQueryItemVGG* %7 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %8, i8 0, i64 16, i1 false)
  %9 = alloca %TSa, align 8
  %10 = alloca %T10Foundation12URLQueryItemVSg, align 8
  %11 = alloca %T10Foundation14NSURLQueryItemC*, align 8
  %12 = bitcast %T10Foundation14NSURLQueryItemC** %11 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %12, i8 0, i64 8, i1 false)
  %13 = alloca %TSS, align 8
  %14 = bitcast %TSS* %13 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %14, i8 0, i64 16, i1 false)
  %15 = alloca [24 x i8], align 8
  %16 = alloca %TSS, align 8
  %17 = bitcast %TSS* %16 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %17, i8 0, i64 16, i1 false)
  %18 = alloca %Ts6UInt32VSg, align 4
  %19 = alloca i32, align 8
  %20 = bitcast i32* %19 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %20, i8 0, i64 4, i1 false)
  %21 = alloca [24 x i8], align 8
  %22 = alloca %TSS, align 8
  %23 = bitcast %TSS* %22 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %23, i8 0, i64 16, i1 false)
  %24 = alloca i1, align 8
  %25 = bitcast i1* %24 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %25, i8 0, i64 1, i1 false)
  %26 = alloca [24 x i8], align 8
  %27 = alloca %TSS, align 8
  %28 = bitcast %TSS* %27 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %28, i8 0, i64 16, i1 false)
  %29 = alloca %Ts6UInt32VSg, align 4
  %30 = alloca i32, align 8
  %31 = bitcast i32* %30 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %31, i8 0, i64 4, i1 false)
  %32 = alloca [24 x i8], align 8
  %33 = alloca %TSS, align 8
  %34 = bitcast %TSS* %33 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %34, i8 0, i64 16, i1 false)
  %35 = alloca %Ts6UInt32VSg, align 4
  %36 = alloca i32, align 8
  %37 = bitcast i32* %36 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %37, i8 0, i64 4, i1 false)
  %38 = alloca [24 x i8], align 8
  %39 = alloca %TSS, align 8
  %40 = bitcast %TSS* %39 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %40, i8 0, i64 16, i1 false)
  %41 = alloca [24 x i8], align 8
  %42 = alloca %TSS, align 8
  %43 = bitcast %TSS* %42 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %43, i8 0, i64 16, i1 false)
  %44 = alloca [24 x i8], align 8
  %45 = alloca %TSS, align 8
  %46 = bitcast %TSS* %45 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %46, i8 0, i64 16, i1 false)
  %47 = alloca [24 x i8], align 8
  store %Ts28__ContiguousArrayStorageBaseC* %0, %Ts28__ContiguousArrayStorageBaseC** %3, align 8
  store %T4file13KcptunProfileC* %1, %T4file13KcptunProfileC** %5, align 8
  %48 = bitcast %Ts16IndexingIteratorVySay10Foundation12URLQueryItemVGG* %7 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %48)
  %49 = bitcast %Ts28__ContiguousArrayStorageBaseC* %0 to %swift.refcounted*
  %50 = call %swift.refcounted* @swift_retain(%swift.refcounted* returned %49) #6
  %51 = bitcast %TSa* %9 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %51)
  %52 = getelementptr inbounds %TSa, %TSa* %9, i32 0, i32 0
  %53 = getelementptr inbounds %Ts22_ContiguousArrayBufferV, %Ts22_ContiguousArrayBufferV* %52, i32 0, i32 0
  store %Ts28__ContiguousArrayStorageBaseC* %0, %Ts28__ContiguousArrayStorageBaseC** %53, align 8
  %54 = bitcast %Ts16IndexingIteratorVySay10Foundation12URLQueryItemVGG* %7 to %Ts16IndexingIteratorV*
  %55 = call swiftcc %swift.metadata_response @"$sSay10Foundation12URLQueryItemVGMa"(i64 0) #8
  %56 = extractvalue %swift.metadata_response %55, 0
  %57 = call i8** @"$sSay10Foundation12URLQueryItemVGSayxGSlsWl"() #8
  %58 = bitcast %TSa* %9 to %swift.opaque*
  call swiftcc void @"$sSlss16IndexingIteratorVyxG0B0RtzrlE04makeB0ACyF"(%Ts16IndexingIteratorV* noalias nocapture sret %54, %swift.type* %56, i8** %57, %swift.opaque* noalias nocapture swiftself %58)
  %59 = bitcast %TSa* %9 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %59)
  br label %60

; <label>:60:                                     ; preds = %418, %417, %2
  %61 = bitcast %T10Foundation12URLQueryItemVSg* %10 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %61)
  %62 = bitcast %T10Foundation12URLQueryItemVSg* %10 to %TSq*
  %63 = call swiftcc %swift.metadata_response @"$ss16IndexingIteratorVySay10Foundation12URLQueryItemVGGMa"(i64 0) #8
  %64 = extractvalue %swift.metadata_response %63, 0
  %65 = bitcast %Ts16IndexingIteratorVySay10Foundation12URLQueryItemVGG* %7 to %Ts16IndexingIteratorV.0*
  call swiftcc void @"$ss16IndexingIteratorV4next7ElementQzSgyF"(%TSq* noalias nocapture sret %62, %swift.type* %64, %Ts16IndexingIteratorV.0* nocapture swiftself %65)
  %66 = bitcast %T10Foundation12URLQueryItemVSg* %10 to i64*
  %67 = load i64, i64* %66, align 8
  %68 = bitcast %T10Foundation12URLQueryItemVSg* %10 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %68)
  %69 = icmp eq i64 %67, 0
  br i1 %69, label %419, label %70

; <label>:70:                                     ; preds = %60
  %71 = inttoptr i64 %67 to %T10Foundation14NSURLQueryItemC*
  br label %72

; <label>:72:                                     ; preds = %70
  %73 = phi %T10Foundation14NSURLQueryItemC* [ %71, %70 ]
  store %T10Foundation14NSURLQueryItemC* %73, %T10Foundation14NSURLQueryItemC** %11, align 8
  %74 = call swiftcc { i64, %swift.bridge* } @"$s10Foundation12URLQueryItemV4nameSSvg"(%T10Foundation14NSURLQueryItemC* %73)
  %75 = extractvalue { i64, %swift.bridge* } %74, 0
  %76 = extractvalue { i64, %swift.bridge* } %74, 1
  %77 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %76) #6
  %78 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @14, i64 0, i64 0), i64 4, i1 true)
  %79 = extractvalue { i64, %swift.bridge* } %78, 0
  %80 = extractvalue { i64, %swift.bridge* } %78, 1
  %81 = call swiftcc i1 @"$sSS2eeoiySbSS_SStFZ"(i64 %79, %swift.bridge* %80, i64 %75, %swift.bridge* %76)
  call void @swift_bridgeObjectRelease(%swift.bridge* %80) #6
  br i1 %81, label %82, label %118

; <label>:82:                                     ; preds = %72
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  %83 = call swiftcc { i64, i64 } @"$s10Foundation12URLQueryItemV5valueSSSgvg"(%T10Foundation14NSURLQueryItemC* %73)
  %84 = extractvalue { i64, i64 } %83, 0
  %85 = extractvalue { i64, i64 } %83, 1
  %86 = icmp eq i64 %85, 0
  br i1 %86, label %89, label %87

; <label>:87:                                     ; preds = %82
  %88 = inttoptr i64 %85 to %swift.bridge*
  br label %90

; <label>:89:                                     ; preds = %82
  br label %117

; <label>:90:                                     ; preds = %87
  %91 = phi i64 [ %84, %87 ]
  %92 = phi %swift.bridge* [ %88, %87 ]
  %93 = bitcast %TSS* %45 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %93)
  %94 = getelementptr inbounds %TSS, %TSS* %45, i32 0, i32 0
  %95 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %94, i32 0, i32 0
  %96 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %95, i32 0, i32 0
  %97 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %96, i32 0, i32 0
  store i64 %91, i64* %97, align 8
  %98 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %95, i32 0, i32 1
  store %swift.bridge* %92, %swift.bridge** %98, align 8
  %99 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %92) #6
  %100 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %92) #6
  %101 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %1, i32 0, i32 1
  %102 = bitcast [24 x i8]* %47 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %102)
  %103 = bitcast %TSS* %101 to i8*
  call void @swift_beginAccess(i8* %103, [24 x i8]* %47, i64 33, i8* null) #6
  %104 = getelementptr inbounds %TSS, %TSS* %101, i32 0, i32 0
  %105 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %104, i32 0, i32 0
  %106 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %105, i32 0, i32 0
  %107 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %106, i32 0, i32 0
  %108 = load i64, i64* %107, align 8
  %109 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %105, i32 0, i32 1
  %110 = load %swift.bridge*, %swift.bridge** %109, align 8
  %111 = getelementptr inbounds %TSS, %TSS* %101, i32 0, i32 0
  %112 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %111, i32 0, i32 0
  %113 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %112, i32 0, i32 0
  %114 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %113, i32 0, i32 0
  store i64 %91, i64* %114, align 8
  %115 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %112, i32 0, i32 1
  store %swift.bridge* %92, %swift.bridge** %115, align 8
  call void @swift_bridgeObjectRelease(%swift.bridge* %110) #6
  call void @swift_endAccess([24 x i8]* %47) #6
  %116 = bitcast [24 x i8]* %47 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %116)
  call void @swift_bridgeObjectRelease(%swift.bridge* %92) #6
  call void @swift_bridgeObjectRelease(%swift.bridge* %92) #6
  br label %117

; <label>:117:                                    ; preds = %90, %89
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  br label %418

; <label>:118:                                    ; preds = %72
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  %119 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %76) #6
  %120 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @15, i64 0, i64 0), i64 3, i1 true)
  %121 = extractvalue { i64, %swift.bridge* } %120, 0
  %122 = extractvalue { i64, %swift.bridge* } %120, 1
  %123 = call swiftcc i1 @"$sSS2eeoiySbSS_SStFZ"(i64 %121, %swift.bridge* %122, i64 %75, %swift.bridge* %76)
  call void @swift_bridgeObjectRelease(%swift.bridge* %122) #6
  br i1 %123, label %124, label %160

; <label>:124:                                    ; preds = %118
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  %125 = call swiftcc { i64, i64 } @"$s10Foundation12URLQueryItemV5valueSSSgvg"(%T10Foundation14NSURLQueryItemC* %73)
  %126 = extractvalue { i64, i64 } %125, 0
  %127 = extractvalue { i64, i64 } %125, 1
  %128 = icmp eq i64 %127, 0
  br i1 %128, label %131, label %129

; <label>:129:                                    ; preds = %124
  %130 = inttoptr i64 %127 to %swift.bridge*
  br label %132

; <label>:131:                                    ; preds = %124
  br label %159

; <label>:132:                                    ; preds = %129
  %133 = phi i64 [ %126, %129 ]
  %134 = phi %swift.bridge* [ %130, %129 ]
  %135 = bitcast %TSS* %42 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %135)
  %136 = getelementptr inbounds %TSS, %TSS* %42, i32 0, i32 0
  %137 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %136, i32 0, i32 0
  %138 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %137, i32 0, i32 0
  %139 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %138, i32 0, i32 0
  store i64 %133, i64* %139, align 8
  %140 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %137, i32 0, i32 1
  store %swift.bridge* %134, %swift.bridge** %140, align 8
  %141 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %134) #6
  %142 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %134) #6
  %143 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %1, i32 0, i32 2
  %144 = bitcast [24 x i8]* %44 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %144)
  %145 = bitcast %TSS* %143 to i8*
  call void @swift_beginAccess(i8* %145, [24 x i8]* %44, i64 33, i8* null) #6
  %146 = getelementptr inbounds %TSS, %TSS* %143, i32 0, i32 0
  %147 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %146, i32 0, i32 0
  %148 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %147, i32 0, i32 0
  %149 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %148, i32 0, i32 0
  %150 = load i64, i64* %149, align 8
  %151 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %147, i32 0, i32 1
  %152 = load %swift.bridge*, %swift.bridge** %151, align 8
  %153 = getelementptr inbounds %TSS, %TSS* %143, i32 0, i32 0
  %154 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %153, i32 0, i32 0
  %155 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %154, i32 0, i32 0
  %156 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %155, i32 0, i32 0
  store i64 %133, i64* %156, align 8
  %157 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %154, i32 0, i32 1
  store %swift.bridge* %134, %swift.bridge** %157, align 8
  call void @swift_bridgeObjectRelease(%swift.bridge* %152) #6
  call void @swift_endAccess([24 x i8]* %44) #6
  %158 = bitcast [24 x i8]* %44 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %158)
  call void @swift_bridgeObjectRelease(%swift.bridge* %134) #6
  call void @swift_bridgeObjectRelease(%swift.bridge* %134) #6
  br label %159

; <label>:159:                                    ; preds = %132, %131
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  br label %418

; <label>:160:                                    ; preds = %118
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  %161 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %76) #6
  %162 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @16, i64 0, i64 0), i64 5, i1 true)
  %163 = extractvalue { i64, %swift.bridge* } %162, 0
  %164 = extractvalue { i64, %swift.bridge* } %162, 1
  %165 = call swiftcc i1 @"$sSS2eeoiySbSS_SStFZ"(i64 %163, %swift.bridge* %164, i64 %75, %swift.bridge* %76)
  call void @swift_bridgeObjectRelease(%swift.bridge* %164) #6
  br i1 %165, label %166, label %202

; <label>:166:                                    ; preds = %160
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  %167 = call swiftcc { i64, i64 } @"$s10Foundation12URLQueryItemV5valueSSSgvg"(%T10Foundation14NSURLQueryItemC* %73)
  %168 = extractvalue { i64, i64 } %167, 0
  %169 = extractvalue { i64, i64 } %167, 1
  %170 = icmp eq i64 %169, 0
  br i1 %170, label %173, label %171

; <label>:171:                                    ; preds = %166
  %172 = inttoptr i64 %169 to %swift.bridge*
  br label %174

; <label>:173:                                    ; preds = %166
  br label %201

; <label>:174:                                    ; preds = %171
  %175 = phi i64 [ %168, %171 ]
  %176 = phi %swift.bridge* [ %172, %171 ]
  %177 = bitcast %TSS* %39 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %177)
  %178 = getelementptr inbounds %TSS, %TSS* %39, i32 0, i32 0
  %179 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %178, i32 0, i32 0
  %180 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %179, i32 0, i32 0
  %181 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %180, i32 0, i32 0
  store i64 %175, i64* %181, align 8
  %182 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %179, i32 0, i32 1
  store %swift.bridge* %176, %swift.bridge** %182, align 8
  %183 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %176) #6
  %184 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %176) #6
  %185 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %1, i32 0, i32 3
  %186 = bitcast [24 x i8]* %41 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %186)
  %187 = bitcast %TSS* %185 to i8*
  call void @swift_beginAccess(i8* %187, [24 x i8]* %41, i64 33, i8* null) #6
  %188 = getelementptr inbounds %TSS, %TSS* %185, i32 0, i32 0
  %189 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %188, i32 0, i32 0
  %190 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %189, i32 0, i32 0
  %191 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %190, i32 0, i32 0
  %192 = load i64, i64* %191, align 8
  %193 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %189, i32 0, i32 1
  %194 = load %swift.bridge*, %swift.bridge** %193, align 8
  %195 = getelementptr inbounds %TSS, %TSS* %185, i32 0, i32 0
  %196 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %195, i32 0, i32 0
  %197 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %196, i32 0, i32 0
  %198 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %197, i32 0, i32 0
  store i64 %175, i64* %198, align 8
  %199 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %196, i32 0, i32 1
  store %swift.bridge* %176, %swift.bridge** %199, align 8
  call void @swift_bridgeObjectRelease(%swift.bridge* %194) #6
  call void @swift_endAccess([24 x i8]* %41) #6
  %200 = bitcast [24 x i8]* %41 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %200)
  call void @swift_bridgeObjectRelease(%swift.bridge* %176) #6
  call void @swift_bridgeObjectRelease(%swift.bridge* %176) #6
  br label %201

; <label>:201:                                    ; preds = %174, %173
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  br label %418

; <label>:202:                                    ; preds = %160
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  %203 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %76) #6
  %204 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @17, i64 0, i64 0), i64 9, i1 true)
  %205 = extractvalue { i64, %swift.bridge* } %204, 0
  %206 = extractvalue { i64, %swift.bridge* } %204, 1
  %207 = call swiftcc i1 @"$sSS2eeoiySbSS_SStFZ"(i64 %205, %swift.bridge* %206, i64 %75, %swift.bridge* %76)
  call void @swift_bridgeObjectRelease(%swift.bridge* %206) #6
  br i1 %207, label %208, label %247

; <label>:208:                                    ; preds = %202
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  %209 = call swiftcc { i64, i64 } @"$s10Foundation12URLQueryItemV5valueSSSgvg"(%T10Foundation14NSURLQueryItemC* %73)
  %210 = extractvalue { i64, i64 } %209, 0
  %211 = extractvalue { i64, i64 } %209, 1
  %212 = icmp eq i64 %211, 0
  br i1 %212, label %215, label %213

; <label>:213:                                    ; preds = %208
  %214 = inttoptr i64 %211 to %swift.bridge*
  br label %216

; <label>:215:                                    ; preds = %208
  br label %246

; <label>:216:                                    ; preds = %213
  %217 = phi i64 [ %210, %213 ]
  %218 = phi %swift.bridge* [ %214, %213 ]
  %219 = bitcast %TSS* %33 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %219)
  %220 = getelementptr inbounds %TSS, %TSS* %33, i32 0, i32 0
  %221 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %220, i32 0, i32 0
  %222 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %221, i32 0, i32 0
  %223 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %222, i32 0, i32 0
  store i64 %217, i64* %223, align 8
  %224 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %221, i32 0, i32 1
  store %swift.bridge* %218, %swift.bridge** %224, align 8
  %225 = bitcast %Ts6UInt32VSg* %35 to i8*
  call void @llvm.lifetime.start.p0i8(i64 5, i8* %225)
  %226 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %218) #6
  %227 = bitcast %Ts6UInt32VSg* %35 to %TSq.1*
  %228 = call i8** @"$ss6UInt32VABs17FixedWidthIntegersWl"() #8
  call swiftcc void @"$ss17FixedWidthIntegerPsEyxSgSScfC"(%TSq.1* noalias nocapture sret %227, i64 %217, %swift.bridge* %218, %swift.type* @"$ss6UInt32VN", i8** %228, %swift.type* swiftself @"$ss6UInt32VN")
  %229 = bitcast %Ts6UInt32VSg* %35 to i32*
  %230 = load i32, i32* %229, align 4
  %231 = getelementptr inbounds %Ts6UInt32VSg, %Ts6UInt32VSg* %35, i32 0, i32 1
  %232 = bitcast [1 x i8]* %231 to i1*
  %233 = load i1, i1* %232, align 4
  br i1 %233, label %235, label %234

; <label>:234:                                    ; preds = %216
  br label %237

; <label>:235:                                    ; preds = %216
  %236 = bitcast %Ts6UInt32VSg* %35 to i8*
  call void @llvm.lifetime.end.p0i8(i64 5, i8* %236)
  br label %245

; <label>:237:                                    ; preds = %234
  %238 = phi i32 [ %230, %234 ]
  store i32 %238, i32* %36, align 8
  %239 = bitcast %Ts6UInt32VSg* %35 to i8*
  call void @llvm.lifetime.end.p0i8(i64 5, i8* %239)
  %240 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %1, i32 0, i32 6
  %241 = bitcast [24 x i8]* %38 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %241)
  %242 = bitcast %Ts6UInt32V* %240 to i8*
  call void @swift_beginAccess(i8* %242, [24 x i8]* %38, i64 33, i8* null) #6
  %243 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %240, i32 0, i32 0
  store i32 %238, i32* %243, align 4
  call void @swift_endAccess([24 x i8]* %38) #6
  %244 = bitcast [24 x i8]* %38 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %244)
  br label %245

; <label>:245:                                    ; preds = %237, %235
  call void @swift_bridgeObjectRelease(%swift.bridge* %218) #6
  br label %246

; <label>:246:                                    ; preds = %245, %215
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  br label %418

; <label>:247:                                    ; preds = %202
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  %248 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %76) #6
  %249 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([12 x i8], [12 x i8]* @18, i64 0, i64 0), i64 11, i1 true)
  %250 = extractvalue { i64, %swift.bridge* } %249, 0
  %251 = extractvalue { i64, %swift.bridge* } %249, 1
  %252 = call swiftcc i1 @"$sSS2eeoiySbSS_SStFZ"(i64 %250, %swift.bridge* %251, i64 %75, %swift.bridge* %76)
  call void @swift_bridgeObjectRelease(%swift.bridge* %251) #6
  br i1 %252, label %253, label %292

; <label>:253:                                    ; preds = %247
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  %254 = call swiftcc { i64, i64 } @"$s10Foundation12URLQueryItemV5valueSSSgvg"(%T10Foundation14NSURLQueryItemC* %73)
  %255 = extractvalue { i64, i64 } %254, 0
  %256 = extractvalue { i64, i64 } %254, 1
  %257 = icmp eq i64 %256, 0
  br i1 %257, label %260, label %258

; <label>:258:                                    ; preds = %253
  %259 = inttoptr i64 %256 to %swift.bridge*
  br label %261

; <label>:260:                                    ; preds = %253
  br label %291

; <label>:261:                                    ; preds = %258
  %262 = phi i64 [ %255, %258 ]
  %263 = phi %swift.bridge* [ %259, %258 ]
  %264 = bitcast %TSS* %27 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %264)
  %265 = getelementptr inbounds %TSS, %TSS* %27, i32 0, i32 0
  %266 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %265, i32 0, i32 0
  %267 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %266, i32 0, i32 0
  %268 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %267, i32 0, i32 0
  store i64 %262, i64* %268, align 8
  %269 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %266, i32 0, i32 1
  store %swift.bridge* %263, %swift.bridge** %269, align 8
  %270 = bitcast %Ts6UInt32VSg* %29 to i8*
  call void @llvm.lifetime.start.p0i8(i64 5, i8* %270)
  %271 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %263) #6
  %272 = bitcast %Ts6UInt32VSg* %29 to %TSq.1*
  %273 = call i8** @"$ss6UInt32VABs17FixedWidthIntegersWl"() #8
  call swiftcc void @"$ss17FixedWidthIntegerPsEyxSgSScfC"(%TSq.1* noalias nocapture sret %272, i64 %262, %swift.bridge* %263, %swift.type* @"$ss6UInt32VN", i8** %273, %swift.type* swiftself @"$ss6UInt32VN")
  %274 = bitcast %Ts6UInt32VSg* %29 to i32*
  %275 = load i32, i32* %274, align 4
  %276 = getelementptr inbounds %Ts6UInt32VSg, %Ts6UInt32VSg* %29, i32 0, i32 1
  %277 = bitcast [1 x i8]* %276 to i1*
  %278 = load i1, i1* %277, align 4
  br i1 %278, label %280, label %279

; <label>:279:                                    ; preds = %261
  br label %282

; <label>:280:                                    ; preds = %261
  %281 = bitcast %Ts6UInt32VSg* %29 to i8*
  call void @llvm.lifetime.end.p0i8(i64 5, i8* %281)
  br label %290

; <label>:282:                                    ; preds = %279
  %283 = phi i32 [ %275, %279 ]
  store i32 %283, i32* %30, align 8
  %284 = bitcast %Ts6UInt32VSg* %29 to i8*
  call void @llvm.lifetime.end.p0i8(i64 5, i8* %284)
  %285 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %1, i32 0, i32 7
  %286 = bitcast [24 x i8]* %32 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %286)
  %287 = bitcast %Ts6UInt32V* %285 to i8*
  call void @swift_beginAccess(i8* %287, [24 x i8]* %32, i64 33, i8* null) #6
  %288 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %285, i32 0, i32 0
  store i32 %283, i32* %288, align 8
  call void @swift_endAccess([24 x i8]* %32) #6
  %289 = bitcast [24 x i8]* %32 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %289)
  br label %290

; <label>:290:                                    ; preds = %282, %280
  call void @swift_bridgeObjectRelease(%swift.bridge* %263) #6
  br label %291

; <label>:291:                                    ; preds = %290, %260
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  br label %418

; <label>:292:                                    ; preds = %247
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  %293 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %76) #6
  %294 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @19, i64 0, i64 0), i64 6, i1 true)
  %295 = extractvalue { i64, %swift.bridge* } %294, 0
  %296 = extractvalue { i64, %swift.bridge* } %294, 1
  %297 = call swiftcc i1 @"$sSS2eeoiySbSS_SStFZ"(i64 %295, %swift.bridge* %296, i64 %75, %swift.bridge* %76)
  call void @swift_bridgeObjectRelease(%swift.bridge* %296) #6
  br i1 %297, label %298, label %330

; <label>:298:                                    ; preds = %292
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  %299 = call swiftcc { i64, i64 } @"$s10Foundation12URLQueryItemV5valueSSSgvg"(%T10Foundation14NSURLQueryItemC* %73)
  %300 = extractvalue { i64, i64 } %299, 0
  %301 = extractvalue { i64, i64 } %299, 1
  %302 = icmp eq i64 %301, 0
  br i1 %302, label %305, label %303

; <label>:303:                                    ; preds = %298
  %304 = inttoptr i64 %301 to %swift.bridge*
  br label %306

; <label>:305:                                    ; preds = %298
  br label %329

; <label>:306:                                    ; preds = %303
  %307 = phi i64 [ %300, %303 ]
  %308 = phi %swift.bridge* [ %304, %303 ]
  %309 = bitcast %TSS* %22 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %309)
  %310 = getelementptr inbounds %TSS, %TSS* %22, i32 0, i32 0
  %311 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %310, i32 0, i32 0
  %312 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %311, i32 0, i32 0
  %313 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %312, i32 0, i32 0
  store i64 %307, i64* %313, align 8
  %314 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %311, i32 0, i32 1
  store %swift.bridge* %308, %swift.bridge** %314, align 8
  %315 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %308) #6
  %316 = call swiftcc i8 @"$sSbySbSgSScfC"(i64 %307, %swift.bridge* %308)
  %317 = icmp eq i8 %316, 2
  br i1 %317, label %320, label %318

; <label>:318:                                    ; preds = %306
  %319 = trunc i8 %316 to i1
  br label %321

; <label>:320:                                    ; preds = %306
  br label %328

; <label>:321:                                    ; preds = %318
  %322 = phi i1 [ %319, %318 ]
  store i1 %322, i1* %24, align 8
  %323 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %1, i32 0, i32 4
  %324 = bitcast [24 x i8]* %26 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %324)
  %325 = bitcast %TSb* %323 to i8*
  call void @swift_beginAccess(i8* %325, [24 x i8]* %26, i64 33, i8* null) #6
  %326 = getelementptr inbounds %TSb, %TSb* %323, i32 0, i32 0
  store i1 %322, i1* %326, align 8
  call void @swift_endAccess([24 x i8]* %26) #6
  %327 = bitcast [24 x i8]* %26 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %327)
  br label %328

; <label>:328:                                    ; preds = %321, %320
  call void @swift_bridgeObjectRelease(%swift.bridge* %308) #6
  br label %329

; <label>:329:                                    ; preds = %328, %305
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  br label %418

; <label>:330:                                    ; preds = %292
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  %331 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %76) #6
  %332 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @20, i64 0, i64 0), i64 3, i1 true)
  %333 = extractvalue { i64, %swift.bridge* } %332, 0
  %334 = extractvalue { i64, %swift.bridge* } %332, 1
  %335 = call swiftcc i1 @"$sSS2eeoiySbSS_SStFZ"(i64 %333, %swift.bridge* %334, i64 %75, %swift.bridge* %76)
  call void @swift_bridgeObjectRelease(%swift.bridge* %334) #6
  br i1 %335, label %336, label %375

; <label>:336:                                    ; preds = %330
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  %337 = call swiftcc { i64, i64 } @"$s10Foundation12URLQueryItemV5valueSSSgvg"(%T10Foundation14NSURLQueryItemC* %73)
  %338 = extractvalue { i64, i64 } %337, 0
  %339 = extractvalue { i64, i64 } %337, 1
  %340 = icmp eq i64 %339, 0
  br i1 %340, label %343, label %341

; <label>:341:                                    ; preds = %336
  %342 = inttoptr i64 %339 to %swift.bridge*
  br label %344

; <label>:343:                                    ; preds = %336
  br label %374

; <label>:344:                                    ; preds = %341
  %345 = phi i64 [ %338, %341 ]
  %346 = phi %swift.bridge* [ %342, %341 ]
  %347 = bitcast %TSS* %16 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %347)
  %348 = getelementptr inbounds %TSS, %TSS* %16, i32 0, i32 0
  %349 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %348, i32 0, i32 0
  %350 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %349, i32 0, i32 0
  %351 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %350, i32 0, i32 0
  store i64 %345, i64* %351, align 8
  %352 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %349, i32 0, i32 1
  store %swift.bridge* %346, %swift.bridge** %352, align 8
  %353 = bitcast %Ts6UInt32VSg* %18 to i8*
  call void @llvm.lifetime.start.p0i8(i64 5, i8* %353)
  %354 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %346) #6
  %355 = bitcast %Ts6UInt32VSg* %18 to %TSq.1*
  %356 = call i8** @"$ss6UInt32VABs17FixedWidthIntegersWl"() #8
  call swiftcc void @"$ss17FixedWidthIntegerPsEyxSgSScfC"(%TSq.1* noalias nocapture sret %355, i64 %345, %swift.bridge* %346, %swift.type* @"$ss6UInt32VN", i8** %356, %swift.type* swiftself @"$ss6UInt32VN")
  %357 = bitcast %Ts6UInt32VSg* %18 to i32*
  %358 = load i32, i32* %357, align 4
  %359 = getelementptr inbounds %Ts6UInt32VSg, %Ts6UInt32VSg* %18, i32 0, i32 1
  %360 = bitcast [1 x i8]* %359 to i1*
  %361 = load i1, i1* %360, align 4
  br i1 %361, label %363, label %362

; <label>:362:                                    ; preds = %344
  br label %365

; <label>:363:                                    ; preds = %344
  %364 = bitcast %Ts6UInt32VSg* %18 to i8*
  call void @llvm.lifetime.end.p0i8(i64 5, i8* %364)
  br label %373

; <label>:365:                                    ; preds = %362
  %366 = phi i32 [ %358, %362 ]
  store i32 %366, i32* %19, align 8
  %367 = bitcast %Ts6UInt32VSg* %18 to i8*
  call void @llvm.lifetime.end.p0i8(i64 5, i8* %367)
  %368 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %1, i32 0, i32 8
  %369 = bitcast [24 x i8]* %21 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %369)
  %370 = bitcast %Ts6UInt32V* %368 to i8*
  call void @swift_beginAccess(i8* %370, [24 x i8]* %21, i64 33, i8* null) #6
  %371 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %368, i32 0, i32 0
  store i32 %366, i32* %371, align 4
  call void @swift_endAccess([24 x i8]* %21) #6
  %372 = bitcast [24 x i8]* %21 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %372)
  br label %373

; <label>:373:                                    ; preds = %365, %363
  call void @swift_bridgeObjectRelease(%swift.bridge* %346) #6
  br label %374

; <label>:374:                                    ; preds = %373, %343
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  br label %418

; <label>:375:                                    ; preds = %330
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  %376 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %76) #6
  %377 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @21, i64 0, i64 0), i64 9, i1 true)
  %378 = extractvalue { i64, %swift.bridge* } %377, 0
  %379 = extractvalue { i64, %swift.bridge* } %377, 1
  %380 = call swiftcc i1 @"$sSS2eeoiySbSS_SStFZ"(i64 %378, %swift.bridge* %379, i64 %75, %swift.bridge* %76)
  call void @swift_bridgeObjectRelease(%swift.bridge* %379) #6
  br i1 %380, label %381, label %417

; <label>:381:                                    ; preds = %375
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  %382 = call swiftcc { i64, i64 } @"$s10Foundation12URLQueryItemV5valueSSSgvg"(%T10Foundation14NSURLQueryItemC* %73)
  %383 = extractvalue { i64, i64 } %382, 0
  %384 = extractvalue { i64, i64 } %382, 1
  %385 = icmp eq i64 %384, 0
  br i1 %385, label %388, label %386

; <label>:386:                                    ; preds = %381
  %387 = inttoptr i64 %384 to %swift.bridge*
  br label %389

; <label>:388:                                    ; preds = %381
  br label %416

; <label>:389:                                    ; preds = %386
  %390 = phi i64 [ %383, %386 ]
  %391 = phi %swift.bridge* [ %387, %386 ]
  %392 = bitcast %TSS* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* %392)
  %393 = getelementptr inbounds %TSS, %TSS* %13, i32 0, i32 0
  %394 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %393, i32 0, i32 0
  %395 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %394, i32 0, i32 0
  %396 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %395, i32 0, i32 0
  store i64 %390, i64* %396, align 8
  %397 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %394, i32 0, i32 1
  store %swift.bridge* %391, %swift.bridge** %397, align 8
  %398 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %391) #6
  %399 = call %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned %391) #6
  %400 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %1, i32 0, i32 9
  %401 = bitcast [24 x i8]* %15 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %401)
  %402 = bitcast %TSS* %400 to i8*
  call void @swift_beginAccess(i8* %402, [24 x i8]* %15, i64 33, i8* null) #6
  %403 = getelementptr inbounds %TSS, %TSS* %400, i32 0, i32 0
  %404 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %403, i32 0, i32 0
  %405 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %404, i32 0, i32 0
  %406 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %405, i32 0, i32 0
  %407 = load i64, i64* %406, align 8
  %408 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %404, i32 0, i32 1
  %409 = load %swift.bridge*, %swift.bridge** %408, align 8
  %410 = getelementptr inbounds %TSS, %TSS* %400, i32 0, i32 0
  %411 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %410, i32 0, i32 0
  %412 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %411, i32 0, i32 0
  %413 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %412, i32 0, i32 0
  store i64 %390, i64* %413, align 8
  %414 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %411, i32 0, i32 1
  store %swift.bridge* %391, %swift.bridge** %414, align 8
  call void @swift_bridgeObjectRelease(%swift.bridge* %409) #6
  call void @swift_endAccess([24 x i8]* %15) #6
  %415 = bitcast [24 x i8]* %15 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %415)
  call void @swift_bridgeObjectRelease(%swift.bridge* %391) #6
  call void @swift_bridgeObjectRelease(%swift.bridge* %391) #6
  br label %416

; <label>:416:                                    ; preds = %389, %388
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  br label %418

; <label>:417:                                    ; preds = %375
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  call void @swift_bridgeObjectRelease(%swift.bridge* %76) #6
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T10Foundation14NSURLQueryItemC*)*)(%T10Foundation14NSURLQueryItemC* %73) #6
  br label %60

; <label>:418:                                    ; preds = %416, %374, %329, %291, %246, %201, %159, %117
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T10Foundation14NSURLQueryItemC*)*)(%T10Foundation14NSURLQueryItemC* %73) #6
  br label %60

; <label>:419:                                    ; preds = %60
  %420 = call %Ts16IndexingIteratorVySay10Foundation12URLQueryItemVGG* @"$ss16IndexingIteratorVySay10Foundation12URLQueryItemVGGWOh"(%Ts16IndexingIteratorVySay10Foundation12URLQueryItemVGG* %7)
  %421 = bitcast %Ts16IndexingIteratorVySay10Foundation12URLQueryItemVGG* %7 to i8*
  call void @llvm.lifetime.end.p0i8(i64 16, i8* %421)
  ret void
}

define dso_local hidden swiftcc %T4file13KcptunProfileC* @"$s4file13KcptunProfileCACycfc"(%T4file13KcptunProfileC* swiftself) #0 {
  %2 = alloca %T4file13KcptunProfileC*, align 8
  %3 = bitcast %T4file13KcptunProfileC** %2 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %3, i8 0, i64 8, i1 false)
  %4 = alloca [24 x i8], align 8
  %5 = alloca [24 x i8], align 8
  %6 = alloca [24 x i8], align 8
  %7 = alloca [24 x i8], align 8
  %8 = alloca [24 x i8], align 8
  %9 = alloca [24 x i8], align 8
  %10 = alloca [24 x i8], align 8
  %11 = alloca [24 x i8], align 8
  %12 = bitcast %T4file13KcptunProfileC** %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %12)
  store %T4file13KcptunProfileC* %0, %T4file13KcptunProfileC** %2, align 8
  %13 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @10, i64 0, i64 0), i64 4, i1 true)
  %14 = extractvalue { i64, %swift.bridge* } %13, 0
  %15 = extractvalue { i64, %swift.bridge* } %13, 1
  %16 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 1
  %17 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %17)
  %18 = bitcast %TSS* %16 to i8*
  call void @swift_beginAccess(i8* %18, [24 x i8]* %4, i64 33, i8* null) #6
  %19 = getelementptr inbounds %TSS, %TSS* %16, i32 0, i32 0
  %20 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %19, i32 0, i32 0
  %21 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %20, i32 0, i32 0
  %22 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %21, i32 0, i32 0
  store i64 %14, i64* %22, align 8
  %23 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %20, i32 0, i32 1
  store %swift.bridge* %15, %swift.bridge** %23, align 8
  call void @swift_endAccess([24 x i8]* %4) #6
  %24 = bitcast [24 x i8]* %4 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %24)
  %25 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @11, i64 0, i64 0), i64 14, i1 true)
  %26 = extractvalue { i64, %swift.bridge* } %25, 0
  %27 = extractvalue { i64, %swift.bridge* } %25, 1
  %28 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 2
  %29 = bitcast [24 x i8]* %5 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %29)
  %30 = bitcast %TSS* %28 to i8*
  call void @swift_beginAccess(i8* %30, [24 x i8]* %5, i64 33, i8* null) #6
  %31 = getelementptr inbounds %TSS, %TSS* %28, i32 0, i32 0
  %32 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %31, i32 0, i32 0
  %33 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %32, i32 0, i32 0
  %34 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %33, i32 0, i32 0
  store i64 %26, i64* %34, align 8
  %35 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %32, i32 0, i32 1
  store %swift.bridge* %27, %swift.bridge** %35, align 8
  call void @swift_endAccess([24 x i8]* %5) #6
  %36 = bitcast [24 x i8]* %5 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %36)
  %37 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @12, i64 0, i64 0), i64 3, i1 true)
  %38 = extractvalue { i64, %swift.bridge* } %37, 0
  %39 = extractvalue { i64, %swift.bridge* } %37, 1
  %40 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 3
  %41 = bitcast [24 x i8]* %6 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %41)
  %42 = bitcast %TSS* %40 to i8*
  call void @swift_beginAccess(i8* %42, [24 x i8]* %6, i64 33, i8* null) #6
  %43 = getelementptr inbounds %TSS, %TSS* %40, i32 0, i32 0
  %44 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %43, i32 0, i32 0
  %45 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %44, i32 0, i32 0
  %46 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %45, i32 0, i32 0
  store i64 %38, i64* %46, align 8
  %47 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %44, i32 0, i32 1
  store %swift.bridge* %39, %swift.bridge** %47, align 8
  call void @swift_endAccess([24 x i8]* %6) #6
  %48 = bitcast [24 x i8]* %6 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %48)
  %49 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 4
  %50 = bitcast [24 x i8]* %7 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %50)
  %51 = bitcast %TSb* %49 to i8*
  call void @swift_beginAccess(i8* %51, [24 x i8]* %7, i64 33, i8* null) #6
  %52 = getelementptr inbounds %TSb, %TSb* %49, i32 0, i32 0
  store i1 false, i1* %52, align 8
  call void @swift_endAccess([24 x i8]* %7) #6
  %53 = bitcast [24 x i8]* %7 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %53)
  %54 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 6
  %55 = bitcast [24 x i8]* %8 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %55)
  %56 = bitcast %Ts6UInt32V* %54 to i8*
  call void @swift_beginAccess(i8* %56, [24 x i8]* %8, i64 33, i8* null) #6
  %57 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %54, i32 0, i32 0
  store i32 10, i32* %57, align 4
  call void @swift_endAccess([24 x i8]* %8) #6
  %58 = bitcast [24 x i8]* %8 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %58)
  %59 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 7
  %60 = bitcast [24 x i8]* %9 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %60)
  %61 = bitcast %Ts6UInt32V* %59 to i8*
  call void @swift_beginAccess(i8* %61, [24 x i8]* %9, i64 33, i8* null) #6
  %62 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %59, i32 0, i32 0
  store i32 3, i32* %62, align 8
  call void @swift_endAccess([24 x i8]* %9) #6
  %63 = bitcast [24 x i8]* %9 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %63)
  %64 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 8
  %65 = bitcast [24 x i8]* %10 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %65)
  %66 = bitcast %Ts6UInt32V* %64 to i8*
  call void @swift_beginAccess(i8* %66, [24 x i8]* %10, i64 33, i8* null) #6
  %67 = getelementptr inbounds %Ts6UInt32V, %Ts6UInt32V* %64, i32 0, i32 0
  store i32 1350, i32* %67, align 4
  call void @swift_endAccess([24 x i8]* %10) #6
  %68 = bitcast [24 x i8]* %10 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %68)
  %69 = call swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8* getelementptr inbounds ([1 x i8], [1 x i8]* @13, i64 0, i64 0), i64 0, i1 true)
  %70 = extractvalue { i64, %swift.bridge* } %69, 0
  %71 = extractvalue { i64, %swift.bridge* } %69, 1
  %72 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 9
  %73 = bitcast [24 x i8]* %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %73)
  %74 = bitcast %TSS* %72 to i8*
  call void @swift_beginAccess(i8* %74, [24 x i8]* %11, i64 33, i8* null) #6
  %75 = getelementptr inbounds %TSS, %TSS* %72, i32 0, i32 0
  %76 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %75, i32 0, i32 0
  %77 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %76, i32 0, i32 0
  %78 = getelementptr inbounds %Ts6UInt64V, %Ts6UInt64V* %77, i32 0, i32 0
  store i64 %70, i64* %78, align 8
  %79 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %76, i32 0, i32 1
  store %swift.bridge* %71, %swift.bridge** %79, align 8
  call void @swift_endAccess([24 x i8]* %11) #6
  %80 = bitcast [24 x i8]* %11 to i8*
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %80)
  %81 = bitcast %T4file13KcptunProfileC* %0 to %T10Foundation8NSObjectC*
  %82 = call swiftcc %T10Foundation8NSObjectC* @"$s10Foundation8NSObjectCACycfc"(%T10Foundation8NSObjectC* swiftself %81)
  %83 = bitcast %T10Foundation8NSObjectC* %82 to %T4file13KcptunProfileC*
  store %T4file13KcptunProfileC* %83, %T4file13KcptunProfileC** %2, align 8
  %84 = bitcast %T4file13KcptunProfileC* %83 to %swift.refcounted*
  %85 = call %swift.refcounted* @swift_retain(%swift.refcounted* returned %84) #6
  %86 = load %T4file13KcptunProfileC*, %T4file13KcptunProfileC** %2, align 8
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%T4file13KcptunProfileC*)*)(%T4file13KcptunProfileC* %86) #6
  %87 = bitcast %T4file13KcptunProfileC** %2 to i8*
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %87)
  ret %T4file13KcptunProfileC* %83
}

define dso_local hidden swiftcc void @"$s4file13KcptunProfileCfE"(%T4file13KcptunProfileC* swiftself) #0 {
  %2 = alloca %T4file13KcptunProfileC*, align 8
  %3 = bitcast %T4file13KcptunProfileC** %2 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %3, i8 0, i64 8, i1 false)
  store %T4file13KcptunProfileC* %0, %T4file13KcptunProfileC** %2, align 8
  %4 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 1
  %5 = call %TSS* @"$sSSWOh"(%TSS* %4)
  %6 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 2
  %7 = call %TSS* @"$sSSWOh"(%TSS* %6)
  %8 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 3
  %9 = call %TSS* @"$sSSWOh"(%TSS* %8)
  %10 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %0, i32 0, i32 9
  %11 = call %TSS* @"$sSSWOh"(%TSS* %10)
  ret void
}

define dso_local hidden swiftcc %swift.refcounted* @"$s4file13KcptunProfileCfd"(%T4file13KcptunProfileC* swiftself) #0 {
  %2 = alloca %T4file13KcptunProfileC*, align 8
  %3 = bitcast %T4file13KcptunProfileC** %2 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %3, i8 0, i64 8, i1 false)
  store %T4file13KcptunProfileC* %0, %T4file13KcptunProfileC** %2, align 8
  %4 = bitcast %T4file13KcptunProfileC* %0 to %T10Foundation8NSObjectC*
  %5 = call swiftcc %swift.refcounted* @"$s10Foundation8NSObjectCfd"(%T10Foundation8NSObjectC* swiftself %4)
  %6 = bitcast %swift.refcounted* %5 to %T4file13KcptunProfileC*
  %7 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %6, i32 0, i32 1
  %8 = call %TSS* @"$sSSWOh"(%TSS* %7)
  %9 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %6, i32 0, i32 2
  %10 = call %TSS* @"$sSSWOh"(%TSS* %9)
  %11 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %6, i32 0, i32 3
  %12 = call %TSS* @"$sSSWOh"(%TSS* %11)
  %13 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %6, i32 0, i32 9
  %14 = call %TSS* @"$sSSWOh"(%TSS* %13)
  ret %swift.refcounted* %5
}

define dso_local hidden swiftcc void @"$s4file13KcptunProfileCfD"(%T4file13KcptunProfileC* swiftself) #0 {
  %2 = alloca %T4file13KcptunProfileC*, align 8
  %3 = bitcast %T4file13KcptunProfileC** %2 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %3, i8 0, i64 8, i1 false)
  store %T4file13KcptunProfileC* %0, %T4file13KcptunProfileC** %2, align 8
  %4 = call swiftcc %swift.refcounted* @"$s4file13KcptunProfileCfd"(%T4file13KcptunProfileC* swiftself %0)
  %5 = bitcast %swift.refcounted* %4 to %T4file13KcptunProfileC*
  %6 = bitcast %T4file13KcptunProfileC* %5 to %swift.refcounted*
  call void @swift_deallocClassInstance(%swift.refcounted* %6, i64 96, i64 7)
  ret void
}

; Function Attrs: nounwind readnone
define dso_local hidden swiftcc %swift.metadata_response @"$s4file13KcptunProfileCMa"(i64) #2 {
  %2 = load %swift.type*, %swift.type** @"$s4file13KcptunProfileCML", align 8
  %3 = icmp eq %swift.type* %2, null
  br i1 %3, label %4, label %5

; <label>:4:                                      ; preds = %1
  store atomic %swift.type* bitcast (i64* getelementptr inbounds (<{ void (%T4file13KcptunProfileC*)*, i8**, i64, %swift.type*, %swift.opaque*, %swift.opaque*, i64, i32, i32, i32, i16, i16, i32, i32, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>*, void (%T4file13KcptunProfileC*)*, %T4file13KcptunProfileC* (%swift.type*)*, void (%Any*, %T10Foundation8NSObjectC*)*, void (%Any*, %T10Foundation8NSObjectC*)*, i1 (%TypSg*, %T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, %T10Foundation8NSObjectC* (%T10Foundation8NSObjectC*)*, i1 (%T10Foundation8NSObjectC*)*, { i64, %swift.bridge* } (%T10Foundation8NSObjectC*)*, { i64, %swift.bridge* } (%T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, %swift.type* (%T10Foundation8NSObjectC*)*, void (%TypSg*, %T10Foundation7NSCoderC*, %T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, void (%TypSg*, %T10Foundation15NSKeyedArchiverC*, %T10Foundation8NSObjectC*)*, %Ts28__ContiguousArrayStorageBaseC* (%swift.type*)*, %swift.type* (%swift.type*)*, i64 (%swift.type*)*, i64 (%swift.type*)*, i1 (%swift.type*, %swift.type*)*, i1 (%swift.type*, %T10Foundation8NSObjectC*)*, i1 (%swift.type*, %T10Foundation8NSObjectC*)*, i64, i64, i64, i64, i64, i64, i64, i64, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, i1 (%T4file13KcptunProfileC*)*, void (i1, %T4file13KcptunProfileC*)*, { i8*, %TSb* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, void (%Any*, i8, %T4file13KcptunProfileC*)*, %swift.bridge* (%T4file13KcptunProfileC*)*, %swift.bridge* (%T4file13KcptunProfileC*)*, %Ts28__ContiguousArrayStorageBaseC* (%T4file13KcptunProfileC*)*, void (%Ts28__ContiguousArrayStorageBaseC*, %T4file13KcptunProfileC*)* }>, <{ void (%T4file13KcptunProfileC*)*, i8**, i64, %swift.type*, %swift.opaque*, %swift.opaque*, i64, i32, i32, i32, i16, i16, i32, i32, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>*, void (%T4file13KcptunProfileC*)*, %T4file13KcptunProfileC* (%swift.type*)*, void (%Any*, %T10Foundation8NSObjectC*)*, void (%Any*, %T10Foundation8NSObjectC*)*, i1 (%TypSg*, %T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, %T10Foundation8NSObjectC* (%T10Foundation8NSObjectC*)*, i1 (%T10Foundation8NSObjectC*)*, { i64, %swift.bridge* } (%T10Foundation8NSObjectC*)*, { i64, %swift.bridge* } (%T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, %swift.type* (%T10Foundation8NSObjectC*)*, void (%TypSg*, %T10Foundation7NSCoderC*, %T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, void (%TypSg*, %T10Foundation15NSKeyedArchiverC*, %T10Foundation8NSObjectC*)*, %Ts28__ContiguousArrayStorageBaseC* (%swift.type*)*, %swift.type* (%swift.type*)*, i64 (%swift.type*)*, i64 (%swift.type*)*, i1 (%swift.type*, %swift.type*)*, i1 (%swift.type*, %T10Foundation8NSObjectC*)*, i1 (%swift.type*, %T10Foundation8NSObjectC*)*, i64, i64, i64, i64, i64, i64, i64, i64, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, i1 (%T4file13KcptunProfileC*)*, void (i1, %T4file13KcptunProfileC*)*, { i8*, %TSb* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, void (%Any*, i8, %T4file13KcptunProfileC*)*, %swift.bridge* (%T4file13KcptunProfileC*)*, %swift.bridge* (%T4file13KcptunProfileC*)*, %Ts28__ContiguousArrayStorageBaseC* (%T4file13KcptunProfileC*)*, void (%Ts28__ContiguousArrayStorageBaseC*, %T4file13KcptunProfileC*)* }>* @"$s4file13KcptunProfileCMf", i32 0, i32 2) to %swift.type*), %swift.type** @"$s4file13KcptunProfileCML" release, align 8
  br label %5

; <label>:5:                                      ; preds = %4, %1
  %6 = phi %swift.type* [ %2, %1 ], [ bitcast (i64* getelementptr inbounds (<{ void (%T4file13KcptunProfileC*)*, i8**, i64, %swift.type*, %swift.opaque*, %swift.opaque*, i64, i32, i32, i32, i16, i16, i32, i32, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>*, void (%T4file13KcptunProfileC*)*, %T4file13KcptunProfileC* (%swift.type*)*, void (%Any*, %T10Foundation8NSObjectC*)*, void (%Any*, %T10Foundation8NSObjectC*)*, i1 (%TypSg*, %T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, %T10Foundation8NSObjectC* (%T10Foundation8NSObjectC*)*, i1 (%T10Foundation8NSObjectC*)*, { i64, %swift.bridge* } (%T10Foundation8NSObjectC*)*, { i64, %swift.bridge* } (%T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, %swift.type* (%T10Foundation8NSObjectC*)*, void (%TypSg*, %T10Foundation7NSCoderC*, %T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, void (%TypSg*, %T10Foundation15NSKeyedArchiverC*, %T10Foundation8NSObjectC*)*, %Ts28__ContiguousArrayStorageBaseC* (%swift.type*)*, %swift.type* (%swift.type*)*, i64 (%swift.type*)*, i64 (%swift.type*)*, i1 (%swift.type*, %swift.type*)*, i1 (%swift.type*, %T10Foundation8NSObjectC*)*, i1 (%swift.type*, %T10Foundation8NSObjectC*)*, i64, i64, i64, i64, i64, i64, i64, i64, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, i1 (%T4file13KcptunProfileC*)*, void (i1, %T4file13KcptunProfileC*)*, { i8*, %TSb* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, void (%Any*, i8, %T4file13KcptunProfileC*)*, %swift.bridge* (%T4file13KcptunProfileC*)*, %swift.bridge* (%T4file13KcptunProfileC*)*, %Ts28__ContiguousArrayStorageBaseC* (%T4file13KcptunProfileC*)*, void (%Ts28__ContiguousArrayStorageBaseC*, %T4file13KcptunProfileC*)* }>, <{ void (%T4file13KcptunProfileC*)*, i8**, i64, %swift.type*, %swift.opaque*, %swift.opaque*, i64, i32, i32, i32, i16, i16, i32, i32, <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>*, void (%T4file13KcptunProfileC*)*, %T4file13KcptunProfileC* (%swift.type*)*, void (%Any*, %T10Foundation8NSObjectC*)*, void (%Any*, %T10Foundation8NSObjectC*)*, i1 (%TypSg*, %T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, %T10Foundation8NSObjectC* (%T10Foundation8NSObjectC*)*, i1 (%T10Foundation8NSObjectC*)*, { i64, %swift.bridge* } (%T10Foundation8NSObjectC*)*, { i64, %swift.bridge* } (%T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, %swift.type* (%T10Foundation8NSObjectC*)*, void (%TypSg*, %T10Foundation7NSCoderC*, %T10Foundation8NSObjectC*)*, i64 (%T10Foundation8NSObjectC*)*, void (%TypSg*, %T10Foundation15NSKeyedArchiverC*, %T10Foundation8NSObjectC*)*, %Ts28__ContiguousArrayStorageBaseC* (%swift.type*)*, %swift.type* (%swift.type*)*, i64 (%swift.type*)*, i64 (%swift.type*)*, i1 (%swift.type*, %swift.type*)*, i1 (%swift.type*, %T10Foundation8NSObjectC*)*, i1 (%swift.type*, %T10Foundation8NSObjectC*)*, i64, i64, i64, i64, i64, i64, i64, i64, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, i1 (%T4file13KcptunProfileC*)*, void (i1, %T4file13KcptunProfileC*)*, { i8*, %TSb* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, i32 (%T4file13KcptunProfileC*)*, void (i32, %T4file13KcptunProfileC*)*, { i8*, %Ts6UInt32V* } (i8*, %T4file13KcptunProfileC*)*, { i64, %swift.bridge* } (%T4file13KcptunProfileC*)*, void (i64, %swift.bridge*, %T4file13KcptunProfileC*)*, { i8*, %TSS* } (i8*, %T4file13KcptunProfileC*)*, void (%Any*, i8, %T4file13KcptunProfileC*)*, %swift.bridge* (%T4file13KcptunProfileC*)*, %swift.bridge* (%T4file13KcptunProfileC*)*, %Ts28__ContiguousArrayStorageBaseC* (%T4file13KcptunProfileC*)*, void (%Ts28__ContiguousArrayStorageBaseC*, %T4file13KcptunProfileC*)* }>* @"$s4file13KcptunProfileCMf", i32 0, i32 2) to %swift.type*), %4 ]
  %7 = insertvalue %swift.metadata_response undef, %swift.type* %6, 0
  %8 = insertvalue %swift.metadata_response %7, i64 0, 1
  ret %swift.metadata_response %8
}

declare swiftcc void @"$s10Foundation8NSObjectC4copyypyF"(%Any* noalias nocapture sret, %T10Foundation8NSObjectC* swiftself) #0

declare swiftcc void @"$s10Foundation8NSObjectC11mutableCopyypyF"(%Any* noalias nocapture sret, %T10Foundation8NSObjectC* swiftself) #0

declare swiftcc i1 @"$s10Foundation8NSObjectC7isEqualySbypSgF"(%TypSg* noalias nocapture dereferenceable(32), %T10Foundation8NSObjectC* swiftself) #0

declare swiftcc i64 @"$s10Foundation8NSObjectC4hashSivg"(%T10Foundation8NSObjectC* swiftself) #0

declare swiftcc %T10Foundation8NSObjectC* @"$s10Foundation8NSObjectC4selfACXDyF"(%T10Foundation8NSObjectC* swiftself) #0

declare swiftcc i1 @"$s10Foundation8NSObjectC7isProxySbyF"(%T10Foundation8NSObjectC* swiftself) #0

declare swiftcc { i64, %swift.bridge* } @"$s10Foundation8NSObjectC11descriptionSSvg"(%T10Foundation8NSObjectC* swiftself) #0

declare swiftcc { i64, %swift.bridge* } @"$s10Foundation8NSObjectC16debugDescriptionSSvg"(%T10Foundation8NSObjectC* swiftself) #0

declare swiftcc i64 @"$s10Foundation8NSObjectC9_cfTypeIDSuvg"(%T10Foundation8NSObjectC* swiftself) #0

declare swiftcc %swift.type* @"$s10Foundation8NSObjectC13classForCoderyXlXpvg"(%T10Foundation8NSObjectC* swiftself) #0

declare swiftcc void @"$s10Foundation8NSObjectC17replacementObject3forypSgAA7NSCoderC_tF"(%TypSg* noalias nocapture sret, %T10Foundation7NSCoderC*, %T10Foundation8NSObjectC* swiftself) #0

declare swiftcc i64 @"$s10Foundation8NSObjectC21classForKeyedArchiveryXlXpSgvg"(%T10Foundation8NSObjectC* swiftself) #0

declare swiftcc void @"$s10Foundation8NSObjectC17replacementObject3forypSgAA15NSKeyedArchiverC_tF"(%TypSg* noalias nocapture sret, %T10Foundation15NSKeyedArchiverC*, %T10Foundation8NSObjectC* swiftself) #0

declare swiftcc %Ts28__ContiguousArrayStorageBaseC* @"$s10Foundation8NSObjectC30classFallbacksForKeyedArchiverSaySSGyFZ"(%swift.type* swiftself) #0

declare swiftcc %swift.type* @"$s10Foundation8NSObjectC23classForKeyedUnarchiveryXlXpyFZ"(%swift.type* swiftself) #0

declare swiftcc i64 @"$s10Foundation8NSObjectC18nsObjectSuperclass33_6DA0945A07226B3278459E9368612FF4LLACmSgvgZ"(%swift.type* swiftself) #0

declare swiftcc i64 @"$s10Foundation8NSObjectC10superclassyXlXpSgvgZ"(%swift.type* swiftself) #0

declare swiftcc i1 @"$s10Foundation8NSObjectC10isSubclass2ofSbyXlXp_tFZ"(%swift.type*, %swift.type* swiftself) #0

declare swiftcc i1 @"$s10Foundation8NSObjectC8isMember2ofSbyXlXp_tF"(%swift.type*, %T10Foundation8NSObjectC* swiftself) #0

declare swiftcc i1 @"$s10Foundation8NSObjectC6isKind2ofSbyXlXp_tF"(%swift.type*, %T10Foundation8NSObjectC* swiftself) #0

declare extern_weak void @"_swift_FORCE_LOAD_$_swiftGlibc"()

define internal dso_local swiftcc void @"$s4file13KcptunProfileC10Foundation9NSCopyingAadEP4copy4withypAD6NSZoneVSg_tFTW"(%Any* noalias nocapture sret, i8, %T4file13KcptunProfileC** noalias nocapture swiftself dereferenceable(8), %swift.type*, i8**) #0 {
  %6 = trunc i8 %1 to i1
  %7 = load %T4file13KcptunProfileC*, %T4file13KcptunProfileC** %2, align 8
  %8 = getelementptr inbounds %T4file13KcptunProfileC, %T4file13KcptunProfileC* %7, i32 0, i32 0, i32 0
  %9 = load %swift.type*, %swift.type** %8, align 8
  %10 = bitcast %swift.type* %9 to void (%Any*, i8, %T4file13KcptunProfileC*)**
  %11 = getelementptr inbounds void (%Any*, i8, %T4file13KcptunProfileC*)*, void (%Any*, i8, %T4file13KcptunProfileC*)** %10, i64 63
  %12 = load void (%Any*, i8, %T4file13KcptunProfileC*)*, void (%Any*, i8, %T4file13KcptunProfileC*)** %11, align 8, !invariant.load !25
  %13 = zext i1 %6 to i8
  call swiftcc void %12(%Any* noalias nocapture sret %0, i8 %13, %T4file13KcptunProfileC* swiftself %7) #9
  ret void
}

; Function Attrs: noreturn nounwind
declare void @llvm.trap() #3

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #4

; Function Attrs: noinline nounwind
define linkonce_odr dso_local hidden %TSS* @"$sSSWOh"(%TSS*) #5 {
  %2 = getelementptr inbounds %TSS, %TSS* %0, i32 0, i32 0
  %3 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %2, i32 0, i32 0
  %4 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %3, i32 0, i32 1
  %5 = load %swift.bridge*, %swift.bridge** %4, align 8
  call void @swift_bridgeObjectRelease(%swift.bridge* %5) #6
  ret %TSS* %0
}

; Function Attrs: nounwind
declare void @swift_bridgeObjectRelease(%swift.bridge*) #6

; Function Attrs: nounwind
declare %swift.refcounted* @swift_allocObject(%swift.type*, i64, i64) #6

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #4

declare swiftcc { i64, %swift.bridge* } @"$sSS21_builtinStringLiteral17utf8CodeUnitCount7isASCIISSBp_BwBi1_tcfC"(i8*, i64, i1) #0

; Function Attrs: nounwind
declare void @swift_beginAccess(i8*, [24 x i8]*, i64, i8*) #6

; Function Attrs: nounwind
declare void @swift_endAccess([24 x i8]*) #6

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #4

declare swiftcc %T10Foundation8NSObjectC* @"$s10Foundation8NSObjectCACycfc"(%T10Foundation8NSObjectC* swiftself) #0

; Function Attrs: nounwind
declare %swift.refcounted* @swift_retain(%swift.refcounted* returned) #6

; Function Attrs: nounwind
declare void @swift_release(%swift.refcounted*) #6

declare swiftcc void @"$sSlss16IndexingIteratorVyxG0B0RtzrlE04makeB0ACyF"(%Ts16IndexingIteratorV* noalias nocapture sret, %swift.type*, i8**, %swift.opaque* noalias nocapture swiftself) #0

; Function Attrs: nounwind readnone
define linkonce_odr dso_local hidden swiftcc %swift.metadata_response @"$sSay10Foundation12URLQueryItemVGMa"(i64) #2 {
  %2 = load %swift.type*, %swift.type** @"$sSay10Foundation12URLQueryItemVGML", align 8
  %3 = icmp eq %swift.type* %2, null
  br i1 %3, label %4, label %10

; <label>:4:                                      ; preds = %1
  %5 = call swiftcc %swift.metadata_response @"$sSaMa"(i64 %0, %swift.type* @"$s10Foundation12URLQueryItemVN") #8
  %6 = extractvalue %swift.metadata_response %5, 0
  %7 = extractvalue %swift.metadata_response %5, 1
  %8 = icmp eq i64 %7, 0
  br i1 %8, label %9, label %10

; <label>:9:                                      ; preds = %4
  store atomic %swift.type* %6, %swift.type** @"$sSay10Foundation12URLQueryItemVGML" release, align 8
  br label %10

; <label>:10:                                     ; preds = %9, %4, %1
  %11 = phi %swift.type* [ %2, %1 ], [ %6, %9 ], [ %6, %4 ]
  %12 = phi i64 [ 0, %1 ], [ %7, %4 ], [ 0, %9 ]
  %13 = insertvalue %swift.metadata_response undef, %swift.type* %11, 0
  %14 = insertvalue %swift.metadata_response %13, i64 %12, 1
  ret %swift.metadata_response %14
}

declare swiftcc %swift.metadata_response @"$sSaMa"(i64, %swift.type*) #0

; Function Attrs: nounwind readnone
define linkonce_odr dso_local hidden i8** @"$sSay10Foundation12URLQueryItemVGSayxGSlsWl"() #2 {
  %1 = load i8**, i8*** @"$sSay10Foundation12URLQueryItemVGSayxGSlsWL", align 8
  %2 = icmp eq i8** %1, null
  br i1 %2, label %3, label %8

; <label>:3:                                      ; preds = %0
  %4 = call swiftcc %swift.metadata_response @"$sSay10Foundation12URLQueryItemVGMa"(i64 255) #8
  %5 = extractvalue %swift.metadata_response %4, 0
  %6 = extractvalue %swift.metadata_response %4, 1
  %7 = call i8** @swift_getWitnessTable(%swift.protocol_conformance_descriptor* @"$sSayxGSlsMc", %swift.type* %5, i8*** undef) #6
  store atomic i8** %7, i8*** @"$sSay10Foundation12URLQueryItemVGSayxGSlsWL" release, align 8
  br label %8

; <label>:8:                                      ; preds = %3, %0
  %9 = phi i8** [ %1, %0 ], [ %7, %3 ]
  ret i8** %9
}

; Function Attrs: nounwind readonly
declare i8** @swift_getWitnessTable(%swift.protocol_conformance_descriptor*, %swift.type*, i8***) #7

declare swiftcc void @"$ss16IndexingIteratorV4next7ElementQzSgyF"(%TSq* noalias nocapture sret, %swift.type*, %Ts16IndexingIteratorV.0* nocapture swiftself) #0

; Function Attrs: nounwind readnone
define linkonce_odr dso_local hidden swiftcc %swift.metadata_response @"$ss16IndexingIteratorVySay10Foundation12URLQueryItemVGGMa"(i64) #2 {
  %2 = load %swift.type*, %swift.type** @"$ss16IndexingIteratorVySay10Foundation12URLQueryItemVGGML", align 8
  %3 = icmp eq %swift.type* %2, null
  br i1 %3, label %4, label %14

; <label>:4:                                      ; preds = %1
  %5 = call swiftcc %swift.metadata_response @"$sSay10Foundation12URLQueryItemVGMa"(i64 255) #8
  %6 = extractvalue %swift.metadata_response %5, 0
  %7 = extractvalue %swift.metadata_response %5, 1
  %8 = call i8** @"$sSay10Foundation12URLQueryItemVGSayxGSlsWl"() #8
  %9 = call swiftcc %swift.metadata_response @"$ss16IndexingIteratorVMa"(i64 %0, %swift.type* %6, i8** %8) #8
  %10 = extractvalue %swift.metadata_response %9, 0
  %11 = extractvalue %swift.metadata_response %9, 1
  %12 = icmp eq i64 %11, 0
  br i1 %12, label %13, label %14

; <label>:13:                                     ; preds = %4
  store atomic %swift.type* %10, %swift.type** @"$ss16IndexingIteratorVySay10Foundation12URLQueryItemVGGML" release, align 8
  br label %14

; <label>:14:                                     ; preds = %13, %4, %1
  %15 = phi %swift.type* [ %2, %1 ], [ %10, %13 ], [ %10, %4 ]
  %16 = phi i64 [ 0, %1 ], [ %11, %4 ], [ 0, %13 ]
  %17 = insertvalue %swift.metadata_response undef, %swift.type* %15, 0
  %18 = insertvalue %swift.metadata_response %17, i64 %16, 1
  ret %swift.metadata_response %18
}

declare swiftcc %swift.metadata_response @"$ss16IndexingIteratorVMa"(i64, %swift.type*, i8**) #0

; Function Attrs: noinline nounwind
define linkonce_odr dso_local hidden %Ts16IndexingIteratorVySay10Foundation12URLQueryItemVGG* @"$ss16IndexingIteratorVySay10Foundation12URLQueryItemVGGWOh"(%Ts16IndexingIteratorVySay10Foundation12URLQueryItemVGG*) #5 {
  %2 = getelementptr inbounds %Ts16IndexingIteratorVySay10Foundation12URLQueryItemVGG, %Ts16IndexingIteratorVySay10Foundation12URLQueryItemVGG* %0, i32 0, i32 0
  %3 = getelementptr inbounds %TSa, %TSa* %2, i32 0, i32 0
  %4 = getelementptr inbounds %Ts22_ContiguousArrayBufferV, %Ts22_ContiguousArrayBufferV* %3, i32 0, i32 0
  %5 = load %Ts28__ContiguousArrayStorageBaseC*, %Ts28__ContiguousArrayStorageBaseC** %4, align 8
  call void bitcast (void (%swift.refcounted*)* @swift_release to void (%Ts28__ContiguousArrayStorageBaseC*)*)(%Ts28__ContiguousArrayStorageBaseC* %5) #6
  ret %Ts16IndexingIteratorVySay10Foundation12URLQueryItemVGG* %0
}

declare swiftcc { i64, %swift.bridge* } @"$s10Foundation12URLQueryItemV4nameSSvg"(%T10Foundation14NSURLQueryItemC*) #0

; Function Attrs: nounwind
declare %swift.bridge* @swift_bridgeObjectRetain(%swift.bridge* returned) #6

declare swiftcc i1 @"$sSS2eeoiySbSS_SStFZ"(i64, %swift.bridge*, i64, %swift.bridge*) #0

declare swiftcc { i64, i64 } @"$s10Foundation12URLQueryItemV5valueSSSgvg"(%T10Foundation14NSURLQueryItemC*) #0

declare swiftcc void @"$ss17FixedWidthIntegerPsEyxSgSScfC"(%TSq.1* noalias nocapture sret, i64, %swift.bridge*, %swift.type*, i8**, %swift.type* swiftself) #0

; Function Attrs: nounwind readnone
define linkonce_odr dso_local hidden i8** @"$ss6UInt32VABs17FixedWidthIntegersWl"() #2 {
  %1 = load i8**, i8*** @"$ss6UInt32VABs17FixedWidthIntegersWL", align 8
  %2 = icmp eq i8** %1, null
  br i1 %2, label %3, label %5

; <label>:3:                                      ; preds = %0
  %4 = call i8** @swift_getWitnessTable(%swift.protocol_conformance_descriptor* @"$ss6UInt32Vs17FixedWidthIntegersMc", %swift.type* @"$ss6UInt32VN", i8*** undef) #6
  store atomic i8** %4, i8*** @"$ss6UInt32VABs17FixedWidthIntegersWL" release, align 8
  br label %5

; <label>:5:                                      ; preds = %3, %0
  %6 = phi i8** [ %1, %0 ], [ %4, %3 ]
  ret i8** %6
}

declare swiftcc i8 @"$sSbySbSgSScfC"(i64, %swift.bridge*) #0

declare swiftcc { %Ts28__ContiguousArrayStorageBaseC*, i8* } @"$ss27_allocateUninitializedArrayySayxG_BptBwlF"(i64, %swift.type*) #0

declare swiftcc %T10Foundation14NSURLQueryItemC* @"$s10Foundation12URLQueryItemV4name5valueACSS_SSSgtcfC"(i64, %swift.bridge*, i64, i64) #0

declare swiftcc { i64, %swift.bridge* } @"$ss26DefaultStringInterpolationV15literalCapacity18interpolationCountABSi_SitcfC"(i64, i64) #0

declare swiftcc void @"$ss26DefaultStringInterpolationV13appendLiteralyySSF"(i64, %swift.bridge*, %Ts26DefaultStringInterpolationV* nocapture swiftself dereferenceable(16)) #0

declare swiftcc void @"$ss26DefaultStringInterpolationV06appendC0yyxs06CustomB11ConvertibleRzlF"(%swift.opaque* noalias nocapture, %swift.type*, i8**, %Ts26DefaultStringInterpolationV* nocapture swiftself dereferenceable(16)) #0

; Function Attrs: noinline nounwind
define linkonce_odr dso_local hidden %Ts26DefaultStringInterpolationV* @"$ss26DefaultStringInterpolationVWOh"(%Ts26DefaultStringInterpolationV*) #5 {
  %2 = getelementptr inbounds %Ts26DefaultStringInterpolationV, %Ts26DefaultStringInterpolationV* %0, i32 0, i32 0
  %3 = getelementptr inbounds %TSS, %TSS* %2, i32 0, i32 0
  %4 = getelementptr inbounds %Ts11_StringGutsV, %Ts11_StringGutsV* %3, i32 0, i32 0
  %5 = getelementptr inbounds %Ts13_StringObjectV, %Ts13_StringObjectV* %4, i32 0, i32 1
  %6 = load %swift.bridge*, %swift.bridge** %5, align 8
  call void @swift_bridgeObjectRelease(%swift.bridge* %6) #6
  ret %Ts26DefaultStringInterpolationV* %0
}

declare swiftcc { i64, %swift.bridge* } @"$sSS19stringInterpolationSSs013DefaultStringB0V_tcfC"(i64, %swift.bridge*) #0

declare swiftcc { i64, %swift.bridge* } @"$sSb11descriptionSSvg"(i1) #0

declare swiftcc %Ts28__ContiguousArrayStorageBaseC* @"$sSa12arrayLiteralSayxGxd_tcfC"(%Ts28__ContiguousArrayStorageBaseC*, %swift.type*) #0

declare swiftcc %swift.metadata_response @"$s10Foundation12UserDefaultsCMa"(i64) #0

; Function Attrs: noinline
declare swiftcc void @"$ss17_assertionFailure__4file4line5flagss5NeverOs12StaticStringV_SSAHSus6UInt32VtF"(i64, i64, i8, i64, %swift.bridge*, i64, i64, i8, i64, i32) #1

; Function Attrs: nounwind readnone
define linkonce_odr dso_local hidden swiftcc %swift.metadata_response @"$sSS_yXltMa"(i64) #2 {
  %2 = load %swift.type*, %swift.type** @"$sSS_yXltML", align 8
  %3 = icmp eq %swift.type* %2, null
  br i1 %3, label %4, label %10

; <label>:4:                                      ; preds = %1
  %5 = call swiftcc %swift.metadata_response @swift_getTupleTypeMetadata2(i64 %0, %swift.type* @"$sSSN", %swift.type* getelementptr inbounds (%swift.full_type, %swift.full_type* @"$syXlN", i32 0, i32 1), i8* null, i8** null) #6
  %6 = extractvalue %swift.metadata_response %5, 0
  %7 = extractvalue %swift.metadata_response %5, 1
  %8 = icmp eq i64 %7, 0
  br i1 %8, label %9, label %10

; <label>:9:                                      ; preds = %4
  store atomic %swift.type* %6, %swift.type** @"$sSS_yXltML" release, align 8
  br label %10

; <label>:10:                                     ; preds = %9, %4, %1
  %11 = phi %swift.type* [ %2, %1 ], [ %6, %9 ], [ %6, %4 ]
  %12 = phi i64 [ 0, %1 ], [ %7, %4 ], [ 0, %9 ]
  %13 = insertvalue %swift.metadata_response undef, %swift.type* %11, 0
  %14 = insertvalue %swift.metadata_response %13, i64 %12, 1
  ret %swift.metadata_response %14
}

; Function Attrs: nounwind readonly
declare swiftcc %swift.metadata_response @swift_getTupleTypeMetadata2(i64, %swift.type*, %swift.type*, i8*, i8**) #7

declare swiftcc void @"$ss26DefaultStringInterpolationV06appendC0yyxs06CustomB11ConvertibleRzs20TextOutputStreamableRzlF"(%swift.opaque* noalias nocapture, %swift.type*, i8**, i8**, %Ts26DefaultStringInterpolationV* nocapture swiftself dereferenceable(16)) #0

declare swiftcc %T10Foundation8NSStringC* @"$sSS10FoundationE19_bridgeToObjectiveCAA8NSStringCyF"(i64, %swift.bridge*) #0

declare swiftcc %swift.metadata_response @"$s10Foundation8NSNumberCMa"(i64) #0

declare swiftcc %T10Foundation8NSNumberC* @"$s10Foundation8NSNumberC5valueACSb_tcfC"(i1, %swift.type* swiftself) #0

declare swiftcc %T10Foundation8NSNumberC* @"$s10Foundation8NSNumberC5valueACs6UInt32V_tcfC"(i32, %swift.type* swiftself) #0

declare swiftcc %T10Foundation8NSNumberC* @"$s10Foundation8NSNumberC5valueACSi_tcfC"(i64, %swift.type* swiftself) #0

declare swiftcc %swift.bridge* @"$sSD17dictionaryLiteralSDyxq_Gx_q_td_tcfC"(%Ts28__ContiguousArrayStorageBaseC*, %swift.type*, %swift.type*, i8**) #0

declare swiftcc void @"$s4file13KcptunProfileCIetMg_TC"(i8* noalias dereferenceable(32), i1) #0

declare i8* @malloc(i64)

declare void @free(i8*)

; Function Attrs: nounwind
declare token @llvm.coro.id.retcon.once(i32, i32, i8*, i8*, i8*, i8*) #6

; Function Attrs: nounwind
declare i8* @llvm.coro.begin(token, i8* writeonly) #6

; Function Attrs: nounwind
declare i1 @llvm.coro.suspend.retcon.i1(...) #6

; Function Attrs: nounwind
declare i1 @llvm.coro.end(i8*, i1) #6

; Function Attrs: nounwind
declare void @swift_deallocClassInstance(%swift.refcounted*, i64, i64) #6

declare swiftcc %swift.refcounted* @"$s10Foundation8NSObjectCfd"(%T10Foundation8NSObjectC* swiftself) #0

attributes #0 = { "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #1 = { noinline "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #2 = { nounwind readnone "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #3 = { noreturn nounwind }
attributes #4 = { argmemonly nounwind }
attributes #5 = { noinline nounwind }
attributes #6 = { nounwind }
attributes #7 = { nounwind readonly }
attributes #8 = { nounwind readnone }
attributes #9 = { noinline }

!swift.module.flags = !{!0}
!llvm.asan.globals = !{!1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20}
!llvm.linker.options = !{}
!llvm.module.flags = !{!21, !22, !23, !24}

!0 = !{!"standard-library", i1 false}
!1 = !{<{ i32, i32, i32 }>* @"$s4fileMXM", null, null, i1 false, i1 true}
!2 = !{<{ [22 x i8], i8 }>* @"symbolic 10Foundation8NSObjectC", null, null, i1 false, i1 true}
!3 = !{<{ i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, %swift.method_descriptor, i32, %swift.method_override_descriptor }>* @"$s4file13KcptunProfileCMn", null, null, i1 false, i1 true}
!4 = !{<{ i8, i32, i8 }>* @"symbolic _____ 4file13KcptunProfileC", null, null, i1 false, i1 true}
!5 = !{<{ [2 x i8], i8 }>* @"symbolic SS", null, null, i1 false, i1 true}
!6 = !{[5 x i8]* @2, null, null, i1 false, i1 true}
!7 = !{[4 x i8]* @3, null, null, i1 false, i1 true}
!8 = !{[6 x i8]* @4, null, null, i1 false, i1 true}
!9 = !{<{ [2 x i8], i8 }>* @"symbolic Sb", null, null, i1 false, i1 true}
!10 = !{[7 x i8]* @5, null, null, i1 false, i1 true}
!11 = !{<{ [9 x i8], i8 }>* @"symbolic s6UInt32V", null, null, i1 false, i1 true}
!12 = !{[10 x i8]* @6, null, null, i1 false, i1 true}
!13 = !{[12 x i8]* @7, null, null, i1 false, i1 true}
!14 = !{[4 x i8]* @8, null, null, i1 false, i1 true}
!15 = !{[10 x i8]* @9, null, null, i1 false, i1 true}
!16 = !{{ i32, i32, i16, i16, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }* @"$s4file13KcptunProfileCMF", null, null, i1 false, i1 true}
!17 = !{%swift.protocol_conformance_descriptor* @"$s4file13KcptunProfileC10Foundation9NSCopyingAAMc", null, null, i1 false, i1 true}
!18 = !{[1 x i32]* @"\01l_protocol_conformances", null, null, i1 false, i1 true}
!19 = !{[1 x %swift.type_metadata_record]* @"\01l_type_metadata_table", null, null, i1 false, i1 true}
!20 = !{[6 x i8*]* @llvm.used, null, null, i1 false, i1 true}
!21 = !{i32 1, !"wchar_size", i32 4}
!22 = !{i32 7, !"PIC Level", i32 2}
!23 = !{i32 4, !"Objective-C Garbage Collection", i32 83953408}
!24 = !{i32 1, !"Swift Version", i32 7}
!25 = !{}
