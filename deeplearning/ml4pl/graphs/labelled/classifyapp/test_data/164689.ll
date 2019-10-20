; ModuleID = '/scratch/talbn/classifyapp_code/train//18/2000.txt.cpp'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::ios_base::Init" = type { i8 }
%"class.std::basic_istream" = type { i32 (...)**, i64, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", %"class.std::basic_ostream"*, i8, i8, %"class.std::basic_streambuf"*, %"class.std::ctype"*, %"class.std::num_put"*, %"class.std::num_get"* }
%"class.std::ios_base" = type { i32 (...)**, i64, i64, i32, i32, i32, %"struct.std::ios_base::_Callback_list"*, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, %"struct.std::ios_base::_Words"*, %"class.std::locale" }
%"struct.std::ios_base::_Callback_list" = type { %"struct.std::ios_base::_Callback_list"*, void (i32, %"class.std::ios_base"*, i32)*, i32, i32 }
%"struct.std::ios_base::_Words" = type { i8*, i64 }
%"class.std::locale" = type { %"class.std::locale::_Impl"* }
%"class.std::locale::_Impl" = type { i32, %"class.std::locale::facet"**, i64, %"class.std::locale::facet"**, i8** }
%"class.std::locale::facet" = type <{ i32 (...)**, i32, [4 x i8] }>
%"class.std::basic_ostream" = type { i32 (...)**, %"class.std::basic_ios" }
%"class.std::basic_streambuf" = type { i32 (...)**, i8*, i8*, i8*, i8*, i8*, i8*, %"class.std::locale" }
%"class.std::ctype" = type <{ %"class.std::locale::facet.base", [4 x i8], %struct.__locale_struct*, i8, [7 x i8], i32*, i32*, i16*, i8, [256 x i8], [256 x i8], i8, [6 x i8] }>
%"class.std::locale::facet.base" = type <{ i32 (...)**, i32 }>
%struct.__locale_struct = type { [13 x %struct.__locale_data*], i16*, i32*, i32*, [13 x i8*] }
%struct.__locale_data = type opaque
%"class.std::num_put" = type { %"class.std::locale::facet.base", [4 x i8] }
%"class.std::num_get" = type { %"class.std::locale::facet.base", [4 x i8] }

@_ZStL8__ioinit = internal global %"class.std::ios_base::Init" zeroinitializer, align 1
@__dso_handle = external global i8
@i = global i32 0, align 4
@j = global i32 0, align 4
@t = global i32 0, align 4
@n = global i32 0, align 4
@sum = global i32 0, align 4
@a = global [100 x [100 x i32]] zeroinitializer, align 16
@_ZSt3cin = external global %"class.std::basic_istream", align 8
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_2000.txt.cpp, i8* null }]

declare void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"*) #0

; Function Attrs: nounwind
declare void @_ZNSt8ios_base4InitD1Ev(%"class.std::ios_base::Init"*) #1

; Function Attrs: nounwind
declare i32 @__cxa_atexit(void (i8*)*, i8*, i8*) #2

; Function Attrs: noreturn nounwind uwtable
define i32 @_Z1fi(i32 %x) #3 {
entry:
  %cmp3.53 = icmp sgt i32 %x, 0
  %0 = add i32 %x, -2
  %1 = zext i32 %0 to i64
  %2 = add nuw nsw i64 %1, 1
  %3 = zext i32 %0 to i64
  %4 = add nuw nsw i64 %3, 1
  %5 = zext i32 %0 to i64
  %6 = add nuw nsw i64 %5, 1
  %7 = and i64 %6, 8589934584
  %8 = add nsw i64 %7, -8
  %9 = lshr exact i64 %8, 3
  %10 = add i32 %x, -2
  %11 = zext i32 %10 to i64
  %12 = add nuw nsw i64 %11, 1
  %13 = and i64 %12, 8589934584
  %14 = add nsw i64 %13, -8
  %15 = lshr exact i64 %14, 3
  %exitcond.66 = icmp eq i32 %x, 1
  %exitcond61.71 = icmp eq i32 %x, 1
  %end.idx = add nuw nsw i64 %1, 2
  %n.vec = and i64 %2, 8589934584
  %end.idx.rnd.down = or i64 %n.vec, 1
  %cmp.zero = icmp eq i64 %end.idx.rnd.down, 1
  %16 = and i64 %15, 1
  %lcmp.mod115 = icmp eq i64 %16, 0
  %17 = icmp eq i64 %15, 0
  %end.idx85 = add nuw nsw i64 %3, 2
  %n.vec87 = and i64 %4, 8589934584
  %end.idx.rnd.down88 = or i64 %n.vec87, 1
  %cmp.zero89 = icmp eq i64 %end.idx.rnd.down88, 1
  %18 = and i64 %9, 1
  %lcmp.mod = icmp eq i64 %18, 0
  %19 = icmp eq i64 %9, 0
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup.18, %entry
  %indvars.iv62 = phi i64 [ %indvars.iv.next63, %for.cond.cleanup.18 ], [ 0, %entry ]
  %arrayidx1 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv62, i64 0
  %20 = load i32, i32* %arrayidx1, align 16, !tbaa !1
  br i1 %cmp3.53, label %for.body.5.preheader, label %for.cond.cleanup.18

for.body.5.preheader:                             ; preds = %for.cond
  br i1 %exitcond.66, label %for.cond.16.preheader, label %overflow.checked84

overflow.checked84:                               ; preds = %for.body.5.preheader
  %minmax.ident.splatinsert = insertelement <4 x i32> undef, i32 %20, i32 0
  %minmax.ident.splat = shufflevector <4 x i32> %minmax.ident.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  br i1 %cmp.zero89, label %middle.block81, label %vector.body80.preheader

vector.body80.preheader:                          ; preds = %overflow.checked84
  br i1 %lcmp.mod, label %vector.body80.prol, label %vector.body80.preheader.split

vector.body80.prol:                               ; preds = %vector.body80.preheader
  %21 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv62, i64 1
  %22 = bitcast i32* %21 to <4 x i32>*
  %wide.load102.prol = load <4 x i32>, <4 x i32>* %22, align 4, !tbaa !1
  %23 = getelementptr [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv62, i64 5
  %24 = bitcast i32* %23 to <4 x i32>*
  %wide.load103.prol = load <4 x i32>, <4 x i32>* %24, align 4, !tbaa !1
  %25 = icmp slt <4 x i32> %wide.load102.prol, %minmax.ident.splat
  %26 = icmp slt <4 x i32> %wide.load103.prol, %minmax.ident.splat
  %27 = select <4 x i1> %25, <4 x i32> %wide.load102.prol, <4 x i32> %minmax.ident.splat
  %28 = select <4 x i1> %26, <4 x i32> %wide.load103.prol, <4 x i32> %minmax.ident.splat
  br label %vector.body80.preheader.split

vector.body80.preheader.split:                    ; preds = %vector.body80.prol, %vector.body80.preheader
  %.lcssa112.unr = phi <4 x i32> [ undef, %vector.body80.preheader ], [ %28, %vector.body80.prol ]
  %.lcssa.unr = phi <4 x i32> [ undef, %vector.body80.preheader ], [ %27, %vector.body80.prol ]
  %index83.unr = phi i64 [ 1, %vector.body80.preheader ], [ 9, %vector.body80.prol ]
  %vec.phi.unr = phi <4 x i32> [ %minmax.ident.splat, %vector.body80.preheader ], [ %27, %vector.body80.prol ]
  %vec.phi101.unr = phi <4 x i32> [ %minmax.ident.splat, %vector.body80.preheader ], [ %28, %vector.body80.prol ]
  br i1 %19, label %middle.block81.loopexit, label %vector.body80.preheader.split.split

vector.body80.preheader.split.split:              ; preds = %vector.body80.preheader.split
  br label %vector.body80

vector.body80:                                    ; preds = %vector.body80, %vector.body80.preheader.split.split
  %index83 = phi i64 [ %index83.unr, %vector.body80.preheader.split.split ], [ %index.next96.1, %vector.body80 ]
  %vec.phi = phi <4 x i32> [ %vec.phi.unr, %vector.body80.preheader.split.split ], [ %43, %vector.body80 ]
  %vec.phi101 = phi <4 x i32> [ %vec.phi101.unr, %vector.body80.preheader.split.split ], [ %44, %vector.body80 ]
  %29 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv62, i64 %index83
  %30 = bitcast i32* %29 to <4 x i32>*
  %wide.load102 = load <4 x i32>, <4 x i32>* %30, align 4, !tbaa !1
  %31 = getelementptr i32, i32* %29, i64 4
  %32 = bitcast i32* %31 to <4 x i32>*
  %wide.load103 = load <4 x i32>, <4 x i32>* %32, align 4, !tbaa !1
  %33 = icmp slt <4 x i32> %wide.load102, %vec.phi
  %34 = icmp slt <4 x i32> %wide.load103, %vec.phi101
  %35 = select <4 x i1> %33, <4 x i32> %wide.load102, <4 x i32> %vec.phi
  %36 = select <4 x i1> %34, <4 x i32> %wide.load103, <4 x i32> %vec.phi101
  %index.next96 = add i64 %index83, 8
  %37 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv62, i64 %index.next96
  %38 = bitcast i32* %37 to <4 x i32>*
  %wide.load102.1 = load <4 x i32>, <4 x i32>* %38, align 4, !tbaa !1
  %39 = getelementptr i32, i32* %37, i64 4
  %40 = bitcast i32* %39 to <4 x i32>*
  %wide.load103.1 = load <4 x i32>, <4 x i32>* %40, align 4, !tbaa !1
  %41 = icmp slt <4 x i32> %wide.load102.1, %35
  %42 = icmp slt <4 x i32> %wide.load103.1, %36
  %43 = select <4 x i1> %41, <4 x i32> %wide.load102.1, <4 x i32> %35
  %44 = select <4 x i1> %42, <4 x i32> %wide.load103.1, <4 x i32> %36
  %index.next96.1 = add i64 %index83, 16
  %45 = icmp eq i64 %index.next96.1, %end.idx.rnd.down88
  br i1 %45, label %middle.block81.loopexit.unr-lcssa, label %vector.body80, !llvm.loop !5

middle.block81.loopexit.unr-lcssa:                ; preds = %vector.body80
  %.lcssa117 = phi <4 x i32> [ %44, %vector.body80 ]
  %.lcssa116 = phi <4 x i32> [ %43, %vector.body80 ]
  br label %middle.block81.loopexit

middle.block81.loopexit:                          ; preds = %vector.body80.preheader.split, %middle.block81.loopexit.unr-lcssa
  %.lcssa112 = phi <4 x i32> [ %.lcssa112.unr, %vector.body80.preheader.split ], [ %.lcssa117, %middle.block81.loopexit.unr-lcssa ]
  %.lcssa = phi <4 x i32> [ %.lcssa.unr, %vector.body80.preheader.split ], [ %.lcssa116, %middle.block81.loopexit.unr-lcssa ]
  br label %middle.block81

middle.block81:                                   ; preds = %middle.block81.loopexit, %overflow.checked84
  %resume.val91 = phi i64 [ 1, %overflow.checked84 ], [ %end.idx.rnd.down88, %middle.block81.loopexit ]
  %rdx.vec.exit.phi = phi <4 x i32> [ %minmax.ident.splat, %overflow.checked84 ], [ %.lcssa, %middle.block81.loopexit ]
  %rdx.vec.exit.phi106 = phi <4 x i32> [ %minmax.ident.splat, %overflow.checked84 ], [ %.lcssa112, %middle.block81.loopexit ]
  %rdx.minmax.cmp = icmp slt <4 x i32> %rdx.vec.exit.phi, %rdx.vec.exit.phi106
  %rdx.minmax.select = select <4 x i1> %rdx.minmax.cmp, <4 x i32> %rdx.vec.exit.phi, <4 x i32> %rdx.vec.exit.phi106
  %rdx.shuf = shufflevector <4 x i32> %rdx.minmax.select, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %rdx.minmax.cmp107 = icmp slt <4 x i32> %rdx.minmax.select, %rdx.shuf
  %rdx.minmax.select108 = select <4 x i1> %rdx.minmax.cmp107, <4 x i32> %rdx.minmax.select, <4 x i32> %rdx.shuf
  %rdx.shuf109 = shufflevector <4 x i32> %rdx.minmax.select108, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %rdx.minmax.cmp110 = icmp slt <4 x i32> %rdx.minmax.select108, %rdx.shuf109
  %rdx.minmax.cmp110.elt = extractelement <4 x i1> %rdx.minmax.cmp110, i32 0
  %rdx.minmax.select108.elt = extractelement <4 x i32> %rdx.minmax.select108, i32 0
  %rdx.shuf109.elt = extractelement <4 x i32> %rdx.minmax.select108, i32 1
  %46 = select i1 %rdx.minmax.cmp110.elt, i32 %rdx.minmax.select108.elt, i32 %rdx.shuf109.elt
  %cmp.n95 = icmp eq i64 %end.idx85, %resume.val91
  br i1 %cmp.n95, label %for.cond.16.preheader, label %for.body.5.for.body.5_crit_edge.preheader

for.body.5.for.body.5_crit_edge.preheader:        ; preds = %middle.block81
  br label %for.body.5.for.body.5_crit_edge

for.cond.16.preheader.loopexit:                   ; preds = %for.body.5.for.body.5_crit_edge
  %.minn.0.lcssa118 = phi i32 [ %.minn.0, %for.body.5.for.body.5_crit_edge ]
  br label %for.cond.16.preheader

for.cond.16.preheader:                            ; preds = %for.cond.16.preheader.loopexit, %middle.block81, %for.body.5.preheader
  %.minn.0.lcssa = phi i32 [ %20, %for.body.5.preheader ], [ %46, %middle.block81 ], [ %.minn.0.lcssa118, %for.cond.16.preheader.loopexit ]
  br i1 %cmp3.53, label %for.body.19.preheader, label %for.cond.cleanup.18

for.body.19.preheader:                            ; preds = %for.cond.16.preheader
  %arrayidx23.69 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv62, i64 0
  %sub.70 = sub nsw i32 %20, %.minn.0.lcssa
  store i32 %sub.70, i32* %arrayidx23.69, align 16, !tbaa !1
  br i1 %exitcond61.71, label %for.cond.cleanup.18, label %overflow.checked

overflow.checked:                                 ; preds = %for.body.19.preheader
  br i1 %cmp.zero, label %middle.block, label %vector.ph

vector.ph:                                        ; preds = %overflow.checked
  %broadcast.splatinsert75 = insertelement <4 x i32> undef, i32 %.minn.0.lcssa, i32 0
  %broadcast.splat76 = shufflevector <4 x i32> %broadcast.splatinsert75, <4 x i32> undef, <4 x i32> zeroinitializer
  br i1 %lcmp.mod115, label %vector.body.prol, label %vector.ph.split

vector.body.prol:                                 ; preds = %vector.ph
  %47 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv62, i64 1
  %48 = bitcast i32* %47 to <4 x i32>*
  %wide.load.prol = load <4 x i32>, <4 x i32>* %48, align 4, !tbaa !1
  %49 = getelementptr [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv62, i64 5
  %50 = bitcast i32* %49 to <4 x i32>*
  %wide.load74.prol = load <4 x i32>, <4 x i32>* %50, align 4, !tbaa !1
  %51 = sub nsw <4 x i32> %wide.load.prol, %broadcast.splat76
  %52 = sub nsw <4 x i32> %wide.load74.prol, %broadcast.splat76
  %53 = bitcast i32* %47 to <4 x i32>*
  store <4 x i32> %51, <4 x i32>* %53, align 4, !tbaa !1
  %54 = bitcast i32* %49 to <4 x i32>*
  store <4 x i32> %52, <4 x i32>* %54, align 4, !tbaa !1
  br label %vector.ph.split

vector.ph.split:                                  ; preds = %vector.body.prol, %vector.ph
  %index.unr = phi i64 [ 1, %vector.ph ], [ 9, %vector.body.prol ]
  br i1 %17, label %middle.block.loopexit, label %vector.ph.split.split

vector.ph.split.split:                            ; preds = %vector.ph.split
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph.split.split
  %index = phi i64 [ %index.unr, %vector.ph.split.split ], [ %index.next.1, %vector.body ]
  %55 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv62, i64 %index
  %56 = bitcast i32* %55 to <4 x i32>*
  %wide.load = load <4 x i32>, <4 x i32>* %56, align 4, !tbaa !1
  %57 = getelementptr i32, i32* %55, i64 4
  %58 = bitcast i32* %57 to <4 x i32>*
  %wide.load74 = load <4 x i32>, <4 x i32>* %58, align 4, !tbaa !1
  %59 = sub nsw <4 x i32> %wide.load, %broadcast.splat76
  %60 = sub nsw <4 x i32> %wide.load74, %broadcast.splat76
  %61 = bitcast i32* %55 to <4 x i32>*
  store <4 x i32> %59, <4 x i32>* %61, align 4, !tbaa !1
  %62 = bitcast i32* %57 to <4 x i32>*
  store <4 x i32> %60, <4 x i32>* %62, align 4, !tbaa !1
  %index.next = add i64 %index, 8
  %63 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv62, i64 %index.next
  %64 = bitcast i32* %63 to <4 x i32>*
  %wide.load.1 = load <4 x i32>, <4 x i32>* %64, align 4, !tbaa !1
  %65 = getelementptr i32, i32* %63, i64 4
  %66 = bitcast i32* %65 to <4 x i32>*
  %wide.load74.1 = load <4 x i32>, <4 x i32>* %66, align 4, !tbaa !1
  %67 = sub nsw <4 x i32> %wide.load.1, %broadcast.splat76
  %68 = sub nsw <4 x i32> %wide.load74.1, %broadcast.splat76
  %69 = bitcast i32* %63 to <4 x i32>*
  store <4 x i32> %67, <4 x i32>* %69, align 4, !tbaa !1
  %70 = bitcast i32* %65 to <4 x i32>*
  store <4 x i32> %68, <4 x i32>* %70, align 4, !tbaa !1
  %index.next.1 = add i64 %index, 16
  %71 = icmp eq i64 %index.next.1, %end.idx.rnd.down
  br i1 %71, label %middle.block.loopexit.unr-lcssa, label %vector.body, !llvm.loop !8

middle.block.loopexit.unr-lcssa:                  ; preds = %vector.body
  br label %middle.block.loopexit

middle.block.loopexit:                            ; preds = %vector.ph.split, %middle.block.loopexit.unr-lcssa
  br label %middle.block

middle.block:                                     ; preds = %middle.block.loopexit, %overflow.checked
  %resume.val = phi i64 [ 1, %overflow.checked ], [ %end.idx.rnd.down, %middle.block.loopexit ]
  %cmp.n = icmp eq i64 %end.idx, %resume.val
  br i1 %cmp.n, label %for.cond.cleanup.18, label %for.body.19.for.body.19_crit_edge.preheader

for.body.19.for.body.19_crit_edge.preheader:      ; preds = %middle.block
  br label %for.body.19.for.body.19_crit_edge

for.body.5.for.body.5_crit_edge:                  ; preds = %for.body.5.for.body.5_crit_edge.preheader, %for.body.5.for.body.5_crit_edge
  %indvars.iv.next68 = phi i64 [ %indvars.iv.next, %for.body.5.for.body.5_crit_edge ], [ %resume.val91, %for.body.5.for.body.5_crit_edge.preheader ]
  %.minn.067 = phi i32 [ %.minn.0, %for.body.5.for.body.5_crit_edge ], [ %46, %for.body.5.for.body.5_crit_edge.preheader ]
  %arrayidx9.phi.trans.insert = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv62, i64 %indvars.iv.next68
  %.pre = load i32, i32* %arrayidx9.phi.trans.insert, align 4, !tbaa !1
  %cmp10 = icmp slt i32 %.pre, %.minn.067
  %.minn.0 = select i1 %cmp10, i32 %.pre, i32 %.minn.067
  %indvars.iv.next = add nuw nsw i64 %indvars.iv.next68, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %x
  br i1 %exitcond, label %for.cond.16.preheader.loopexit, label %for.body.5.for.body.5_crit_edge, !llvm.loop !9

for.cond.cleanup.18.loopexit:                     ; preds = %for.body.19.for.body.19_crit_edge
  br label %for.cond.cleanup.18

for.cond.cleanup.18:                              ; preds = %for.cond.cleanup.18.loopexit, %for.body.19.preheader, %middle.block, %for.cond, %for.cond.16.preheader
  %indvars.iv.next63 = add nuw nsw i64 %indvars.iv62, 1
  br label %for.cond

for.body.19.for.body.19_crit_edge:                ; preds = %for.body.19.for.body.19_crit_edge.preheader, %for.body.19.for.body.19_crit_edge
  %indvars.iv.next5972 = phi i64 [ %indvars.iv.next59, %for.body.19.for.body.19_crit_edge ], [ %resume.val, %for.body.19.for.body.19_crit_edge.preheader ]
  %arrayidx23.phi.trans.insert = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv62, i64 %indvars.iv.next5972
  %.pre64 = load i32, i32* %arrayidx23.phi.trans.insert, align 4, !tbaa !1
  %arrayidx23 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv62, i64 %indvars.iv.next5972
  %sub = sub nsw i32 %.pre64, %.minn.0.lcssa
  store i32 %sub, i32* %arrayidx23, align 4, !tbaa !1
  %indvars.iv.next59 = add nuw nsw i64 %indvars.iv.next5972, 1
  %lftr.wideiv60 = trunc i64 %indvars.iv.next59 to i32
  %exitcond61 = icmp eq i32 %lftr.wideiv60, %x
  br i1 %exitcond61, label %for.cond.cleanup.18.loopexit, label %for.body.19.for.body.19_crit_edge, !llvm.loop !11
}

; Function Attrs: noreturn nounwind uwtable
define i32 @_Z1gi(i32 %x) #3 {
entry:
  %cmp2.52 = icmp sgt i32 %x, 0
  %0 = add i32 %x, -1
  %1 = add i32 %x, -2
  %exitcond.65 = icmp eq i32 %x, 1
  %exitcond60.70 = icmp eq i32 %x, 1
  %xtraiter73 = and i32 %0, 3
  %lcmp.mod74 = icmp eq i32 %xtraiter73, 0
  %2 = icmp ult i32 %1, 3
  %xtraiter = and i32 %0, 3
  %lcmp.mod = icmp eq i32 %xtraiter, 0
  %3 = icmp ult i32 %1, 3
  br label %for.cond

for.cond:                                         ; preds = %for.cond.cleanup.17, %entry
  %indvars.iv61 = phi i64 [ %indvars.iv.next62, %for.cond.cleanup.17 ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 0, i64 %indvars.iv61
  %4 = load i32, i32* %arrayidx, align 4, !tbaa !1
  br i1 %cmp2.52, label %for.body.4.preheader, label %for.cond.cleanup.17

for.body.4.preheader:                             ; preds = %for.cond
  br i1 %exitcond.65, label %for.cond.15.preheader, label %for.body.4.for.body.4_crit_edge.preheader

for.body.4.for.body.4_crit_edge.preheader:        ; preds = %for.body.4.preheader
  br i1 %lcmp.mod, label %for.body.4.for.body.4_crit_edge.preheader.split, label %for.body.4.for.body.4_crit_edge.prol.preheader

for.body.4.for.body.4_crit_edge.prol.preheader:   ; preds = %for.body.4.for.body.4_crit_edge.preheader
  br label %for.body.4.for.body.4_crit_edge.prol

for.body.4.for.body.4_crit_edge.prol:             ; preds = %for.body.4.for.body.4_crit_edge.prol, %for.body.4.for.body.4_crit_edge.prol.preheader
  %indvars.iv.next67.prol = phi i64 [ %indvars.iv.next.prol, %for.body.4.for.body.4_crit_edge.prol ], [ 1, %for.body.4.for.body.4_crit_edge.prol.preheader ]
  %.minn.066.prol = phi i32 [ %.minn.0.prol, %for.body.4.for.body.4_crit_edge.prol ], [ %4, %for.body.4.for.body.4_crit_edge.prol.preheader ]
  %prol.iter = phi i32 [ %prol.iter.sub, %for.body.4.for.body.4_crit_edge.prol ], [ %xtraiter, %for.body.4.for.body.4_crit_edge.prol.preheader ]
  %arrayidx8.phi.trans.insert.prol = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next67.prol, i64 %indvars.iv61
  %.pre.prol = load i32, i32* %arrayidx8.phi.trans.insert.prol, align 4, !tbaa !1
  %cmp9.prol = icmp slt i32 %.pre.prol, %.minn.066.prol
  %.minn.0.prol = select i1 %cmp9.prol, i32 %.pre.prol, i32 %.minn.066.prol
  %indvars.iv.next.prol = add nuw nsw i64 %indvars.iv.next67.prol, 1
  %prol.iter.sub = add i32 %prol.iter, -1
  %prol.iter.cmp = icmp eq i32 %prol.iter.sub, 0
  br i1 %prol.iter.cmp, label %for.body.4.for.body.4_crit_edge.preheader.split.loopexit, label %for.body.4.for.body.4_crit_edge.prol, !llvm.loop !12

for.body.4.for.body.4_crit_edge.preheader.split.loopexit: ; preds = %for.body.4.for.body.4_crit_edge.prol
  %indvars.iv.next.prol.lcssa = phi i64 [ %indvars.iv.next.prol, %for.body.4.for.body.4_crit_edge.prol ]
  %.minn.0.prol.lcssa = phi i32 [ %.minn.0.prol, %for.body.4.for.body.4_crit_edge.prol ]
  br label %for.body.4.for.body.4_crit_edge.preheader.split

for.body.4.for.body.4_crit_edge.preheader.split:  ; preds = %for.body.4.for.body.4_crit_edge.preheader, %for.body.4.for.body.4_crit_edge.preheader.split.loopexit
  %.minn.0.lcssa72.unr = phi i32 [ undef, %for.body.4.for.body.4_crit_edge.preheader ], [ %.minn.0.prol.lcssa, %for.body.4.for.body.4_crit_edge.preheader.split.loopexit ]
  %indvars.iv.next67.unr = phi i64 [ 1, %for.body.4.for.body.4_crit_edge.preheader ], [ %indvars.iv.next.prol.lcssa, %for.body.4.for.body.4_crit_edge.preheader.split.loopexit ]
  %.minn.066.unr = phi i32 [ %4, %for.body.4.for.body.4_crit_edge.preheader ], [ %.minn.0.prol.lcssa, %for.body.4.for.body.4_crit_edge.preheader.split.loopexit ]
  br i1 %3, label %for.cond.15.preheader.loopexit, label %for.body.4.for.body.4_crit_edge.preheader.split.split

for.body.4.for.body.4_crit_edge.preheader.split.split: ; preds = %for.body.4.for.body.4_crit_edge.preheader.split
  br label %for.body.4.for.body.4_crit_edge

for.cond.15.preheader.loopexit.unr-lcssa:         ; preds = %for.body.4.for.body.4_crit_edge
  %.minn.0.3.lcssa = phi i32 [ %.minn.0.3, %for.body.4.for.body.4_crit_edge ]
  br label %for.cond.15.preheader.loopexit

for.cond.15.preheader.loopexit:                   ; preds = %for.body.4.for.body.4_crit_edge.preheader.split, %for.cond.15.preheader.loopexit.unr-lcssa
  %.minn.0.lcssa72 = phi i32 [ %.minn.0.lcssa72.unr, %for.body.4.for.body.4_crit_edge.preheader.split ], [ %.minn.0.3.lcssa, %for.cond.15.preheader.loopexit.unr-lcssa ]
  br label %for.cond.15.preheader

for.cond.15.preheader:                            ; preds = %for.cond.15.preheader.loopexit, %for.body.4.preheader
  %.minn.0.lcssa = phi i32 [ %4, %for.body.4.preheader ], [ %.minn.0.lcssa72, %for.cond.15.preheader.loopexit ]
  br i1 %cmp2.52, label %for.body.18.preheader, label %for.cond.cleanup.17

for.body.18.preheader:                            ; preds = %for.cond.15.preheader
  %arrayidx22.68 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 0, i64 %indvars.iv61
  %sub.69 = sub nsw i32 %4, %.minn.0.lcssa
  store i32 %sub.69, i32* %arrayidx22.68, align 4, !tbaa !1
  br i1 %exitcond60.70, label %for.cond.cleanup.17, label %for.body.18.for.body.18_crit_edge.preheader

for.body.18.for.body.18_crit_edge.preheader:      ; preds = %for.body.18.preheader
  br i1 %lcmp.mod74, label %for.body.18.for.body.18_crit_edge.preheader.split, label %for.body.18.for.body.18_crit_edge.prol.preheader

for.body.18.for.body.18_crit_edge.prol.preheader: ; preds = %for.body.18.for.body.18_crit_edge.preheader
  br label %for.body.18.for.body.18_crit_edge.prol

for.body.18.for.body.18_crit_edge.prol:           ; preds = %for.body.18.for.body.18_crit_edge.prol, %for.body.18.for.body.18_crit_edge.prol.preheader
  %indvars.iv.next5871.prol = phi i64 [ %indvars.iv.next58.prol, %for.body.18.for.body.18_crit_edge.prol ], [ 1, %for.body.18.for.body.18_crit_edge.prol.preheader ]
  %prol.iter75 = phi i32 [ %prol.iter75.sub, %for.body.18.for.body.18_crit_edge.prol ], [ %xtraiter73, %for.body.18.for.body.18_crit_edge.prol.preheader ]
  %arrayidx22.phi.trans.insert.prol = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next5871.prol, i64 %indvars.iv61
  %.pre63.prol = load i32, i32* %arrayidx22.phi.trans.insert.prol, align 4, !tbaa !1
  %arrayidx22.prol = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next5871.prol, i64 %indvars.iv61
  %sub.prol = sub nsw i32 %.pre63.prol, %.minn.0.lcssa
  store i32 %sub.prol, i32* %arrayidx22.prol, align 4, !tbaa !1
  %indvars.iv.next58.prol = add nuw nsw i64 %indvars.iv.next5871.prol, 1
  %prol.iter75.sub = add i32 %prol.iter75, -1
  %prol.iter75.cmp = icmp eq i32 %prol.iter75.sub, 0
  br i1 %prol.iter75.cmp, label %for.body.18.for.body.18_crit_edge.preheader.split.loopexit, label %for.body.18.for.body.18_crit_edge.prol, !llvm.loop !14

for.body.18.for.body.18_crit_edge.preheader.split.loopexit: ; preds = %for.body.18.for.body.18_crit_edge.prol
  %indvars.iv.next58.prol.lcssa = phi i64 [ %indvars.iv.next58.prol, %for.body.18.for.body.18_crit_edge.prol ]
  br label %for.body.18.for.body.18_crit_edge.preheader.split

for.body.18.for.body.18_crit_edge.preheader.split: ; preds = %for.body.18.for.body.18_crit_edge.preheader, %for.body.18.for.body.18_crit_edge.preheader.split.loopexit
  %indvars.iv.next5871.unr = phi i64 [ 1, %for.body.18.for.body.18_crit_edge.preheader ], [ %indvars.iv.next58.prol.lcssa, %for.body.18.for.body.18_crit_edge.preheader.split.loopexit ]
  br i1 %2, label %for.cond.cleanup.17.loopexit, label %for.body.18.for.body.18_crit_edge.preheader.split.split

for.body.18.for.body.18_crit_edge.preheader.split.split: ; preds = %for.body.18.for.body.18_crit_edge.preheader.split
  br label %for.body.18.for.body.18_crit_edge

for.body.4.for.body.4_crit_edge:                  ; preds = %for.body.4.for.body.4_crit_edge, %for.body.4.for.body.4_crit_edge.preheader.split.split
  %indvars.iv.next67 = phi i64 [ %indvars.iv.next67.unr, %for.body.4.for.body.4_crit_edge.preheader.split.split ], [ %indvars.iv.next.3, %for.body.4.for.body.4_crit_edge ]
  %.minn.066 = phi i32 [ %.minn.066.unr, %for.body.4.for.body.4_crit_edge.preheader.split.split ], [ %.minn.0.3, %for.body.4.for.body.4_crit_edge ]
  %arrayidx8.phi.trans.insert = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next67, i64 %indvars.iv61
  %.pre = load i32, i32* %arrayidx8.phi.trans.insert, align 4, !tbaa !1
  %cmp9 = icmp slt i32 %.pre, %.minn.066
  %.minn.0 = select i1 %cmp9, i32 %.pre, i32 %.minn.066
  %indvars.iv.next = add nuw nsw i64 %indvars.iv.next67, 1
  %arrayidx8.phi.trans.insert.1 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next, i64 %indvars.iv61
  %.pre.1 = load i32, i32* %arrayidx8.phi.trans.insert.1, align 4, !tbaa !1
  %cmp9.1 = icmp slt i32 %.pre.1, %.minn.0
  %.minn.0.1 = select i1 %cmp9.1, i32 %.pre.1, i32 %.minn.0
  %indvars.iv.next.1 = add nsw i64 %indvars.iv.next67, 2
  %arrayidx8.phi.trans.insert.2 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next.1, i64 %indvars.iv61
  %.pre.2 = load i32, i32* %arrayidx8.phi.trans.insert.2, align 4, !tbaa !1
  %cmp9.2 = icmp slt i32 %.pre.2, %.minn.0.1
  %.minn.0.2 = select i1 %cmp9.2, i32 %.pre.2, i32 %.minn.0.1
  %indvars.iv.next.2 = add nsw i64 %indvars.iv.next67, 3
  %arrayidx8.phi.trans.insert.3 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next.2, i64 %indvars.iv61
  %.pre.3 = load i32, i32* %arrayidx8.phi.trans.insert.3, align 4, !tbaa !1
  %cmp9.3 = icmp slt i32 %.pre.3, %.minn.0.2
  %.minn.0.3 = select i1 %cmp9.3, i32 %.pre.3, i32 %.minn.0.2
  %indvars.iv.next.3 = add nsw i64 %indvars.iv.next67, 4
  %lftr.wideiv.3 = trunc i64 %indvars.iv.next.3 to i32
  %exitcond.3 = icmp eq i32 %lftr.wideiv.3, %x
  br i1 %exitcond.3, label %for.cond.15.preheader.loopexit.unr-lcssa, label %for.body.4.for.body.4_crit_edge

for.cond.cleanup.17.loopexit.unr-lcssa:           ; preds = %for.body.18.for.body.18_crit_edge
  br label %for.cond.cleanup.17.loopexit

for.cond.cleanup.17.loopexit:                     ; preds = %for.body.18.for.body.18_crit_edge.preheader.split, %for.cond.cleanup.17.loopexit.unr-lcssa
  br label %for.cond.cleanup.17

for.cond.cleanup.17:                              ; preds = %for.cond.cleanup.17.loopexit, %for.body.18.preheader, %for.cond, %for.cond.15.preheader
  %indvars.iv.next62 = add nuw nsw i64 %indvars.iv61, 1
  br label %for.cond

for.body.18.for.body.18_crit_edge:                ; preds = %for.body.18.for.body.18_crit_edge, %for.body.18.for.body.18_crit_edge.preheader.split.split
  %indvars.iv.next5871 = phi i64 [ %indvars.iv.next5871.unr, %for.body.18.for.body.18_crit_edge.preheader.split.split ], [ %indvars.iv.next58.3, %for.body.18.for.body.18_crit_edge ]
  %arrayidx22.phi.trans.insert = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next5871, i64 %indvars.iv61
  %.pre63 = load i32, i32* %arrayidx22.phi.trans.insert, align 4, !tbaa !1
  %arrayidx22 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next5871, i64 %indvars.iv61
  %sub = sub nsw i32 %.pre63, %.minn.0.lcssa
  store i32 %sub, i32* %arrayidx22, align 4, !tbaa !1
  %indvars.iv.next58 = add nuw nsw i64 %indvars.iv.next5871, 1
  %arrayidx22.phi.trans.insert.1 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next58, i64 %indvars.iv61
  %.pre63.1 = load i32, i32* %arrayidx22.phi.trans.insert.1, align 4, !tbaa !1
  %arrayidx22.1 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next58, i64 %indvars.iv61
  %sub.1 = sub nsw i32 %.pre63.1, %.minn.0.lcssa
  store i32 %sub.1, i32* %arrayidx22.1, align 4, !tbaa !1
  %indvars.iv.next58.1 = add nsw i64 %indvars.iv.next5871, 2
  %arrayidx22.phi.trans.insert.2 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next58.1, i64 %indvars.iv61
  %.pre63.2 = load i32, i32* %arrayidx22.phi.trans.insert.2, align 4, !tbaa !1
  %arrayidx22.2 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next58.1, i64 %indvars.iv61
  %sub.2 = sub nsw i32 %.pre63.2, %.minn.0.lcssa
  store i32 %sub.2, i32* %arrayidx22.2, align 4, !tbaa !1
  %indvars.iv.next58.2 = add nsw i64 %indvars.iv.next5871, 3
  %arrayidx22.phi.trans.insert.3 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next58.2, i64 %indvars.iv61
  %.pre63.3 = load i32, i32* %arrayidx22.phi.trans.insert.3, align 4, !tbaa !1
  %arrayidx22.3 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next58.2, i64 %indvars.iv61
  %sub.3 = sub nsw i32 %.pre63.3, %.minn.0.lcssa
  store i32 %sub.3, i32* %arrayidx22.3, align 4, !tbaa !1
  %indvars.iv.next58.3 = add nsw i64 %indvars.iv.next5871, 4
  %lftr.wideiv59.3 = trunc i64 %indvars.iv.next58.3 to i32
  %exitcond60.3 = icmp eq i32 %lftr.wideiv59.3, %x
  br i1 %exitcond60.3, label %for.cond.cleanup.17.loopexit.unr-lcssa, label %for.body.18.for.body.18_crit_edge
}

; Function Attrs: noreturn nounwind uwtable
define i32 @_Z1hi(i32 %x) #3 {
entry:
  %0 = load i32, i32* @sum, align 4, !tbaa !1
  %1 = load i32, i32* getelementptr inbounds ([100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 1, i64 1), align 4, !tbaa !1
  %add = add nsw i32 %1, %0
  store i32 %add, i32* @sum, align 4, !tbaa !1
  %cmp.63 = icmp sgt i32 %x, 0
  br i1 %cmp.63, label %for.cond.1.preheader.lr.ph, label %for.cond.16.preheader

for.cond.1.preheader.lr.ph:                       ; preds = %entry
  %cmp2.61 = icmp sgt i32 %x, 1
  %2 = add i32 %x, -2
  %3 = zext i32 %2 to i64
  %4 = add nuw nsw i64 %3, 1
  %5 = zext i32 %2 to i64
  %6 = add nuw nsw i64 %5, 1
  %7 = and i64 %6, 8589934584
  %8 = add nsw i64 %7, -8
  %9 = lshr exact i64 %8, 3
  %end.idx = add nuw nsw i64 %3, 2
  %n.vec = and i64 %4, 8589934584
  %end.idx.rnd.down = or i64 %n.vec, 1
  %cmp.zero = icmp eq i64 %end.idx.rnd.down, 1
  %10 = and i64 %9, 1
  %lcmp.mod81 = icmp eq i64 %10, 0
  %11 = icmp eq i64 %9, 0
  br label %for.cond.1.preheader

for.cond.1.preheader:                             ; preds = %for.cond.cleanup.3, %for.cond.1.preheader.lr.ph
  %indvars.iv72 = phi i64 [ 0, %for.cond.1.preheader.lr.ph ], [ %indvars.iv.next73, %for.cond.cleanup.3 ]
  br i1 %cmp2.61, label %overflow.checked, label %for.cond.cleanup.3

overflow.checked:                                 ; preds = %for.cond.1.preheader
  br i1 %cmp.zero, label %middle.block, label %vector.body.preheader

vector.body.preheader:                            ; preds = %overflow.checked
  br i1 %lcmp.mod81, label %vector.body.prol, label %vector.body.preheader.split

vector.body.prol:                                 ; preds = %vector.body.preheader
  %12 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv72, i64 2
  %13 = bitcast i32* %12 to <4 x i32>*
  %wide.load.prol = load <4 x i32>, <4 x i32>* %13, align 8, !tbaa !1
  %14 = getelementptr [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv72, i64 6
  %15 = bitcast i32* %14 to <4 x i32>*
  %wide.load77.prol = load <4 x i32>, <4 x i32>* %15, align 8, !tbaa !1
  %16 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv72, i64 1
  %17 = bitcast i32* %16 to <4 x i32>*
  store <4 x i32> %wide.load.prol, <4 x i32>* %17, align 4, !tbaa !1
  %18 = getelementptr [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv72, i64 5
  %19 = bitcast i32* %18 to <4 x i32>*
  store <4 x i32> %wide.load77.prol, <4 x i32>* %19, align 4, !tbaa !1
  br label %vector.body.preheader.split

vector.body.preheader.split:                      ; preds = %vector.body.prol, %vector.body.preheader
  %index.unr = phi i64 [ 1, %vector.body.preheader ], [ 9, %vector.body.prol ]
  br i1 %11, label %middle.block.loopexit, label %vector.body.preheader.split.split

vector.body.preheader.split.split:                ; preds = %vector.body.preheader.split
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.body.preheader.split.split
  %index = phi i64 [ %index.unr, %vector.body.preheader.split.split ], [ %index.next.1, %vector.body ]
  %20 = add i64 %index, 1
  %21 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv72, i64 %20
  %22 = bitcast i32* %21 to <4 x i32>*
  %wide.load = load <4 x i32>, <4 x i32>* %22, align 4, !tbaa !1
  %23 = getelementptr i32, i32* %21, i64 4
  %24 = bitcast i32* %23 to <4 x i32>*
  %wide.load77 = load <4 x i32>, <4 x i32>* %24, align 4, !tbaa !1
  %25 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv72, i64 %index
  %26 = bitcast i32* %25 to <4 x i32>*
  store <4 x i32> %wide.load, <4 x i32>* %26, align 4, !tbaa !1
  %27 = getelementptr i32, i32* %25, i64 4
  %28 = bitcast i32* %27 to <4 x i32>*
  store <4 x i32> %wide.load77, <4 x i32>* %28, align 4, !tbaa !1
  %index.next = add i64 %index, 8
  %29 = add i64 %index, 9
  %30 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv72, i64 %29
  %31 = bitcast i32* %30 to <4 x i32>*
  %wide.load.1 = load <4 x i32>, <4 x i32>* %31, align 4, !tbaa !1
  %32 = getelementptr i32, i32* %30, i64 4
  %33 = bitcast i32* %32 to <4 x i32>*
  %wide.load77.1 = load <4 x i32>, <4 x i32>* %33, align 4, !tbaa !1
  %34 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv72, i64 %index.next
  %35 = bitcast i32* %34 to <4 x i32>*
  store <4 x i32> %wide.load.1, <4 x i32>* %35, align 4, !tbaa !1
  %36 = getelementptr i32, i32* %34, i64 4
  %37 = bitcast i32* %36 to <4 x i32>*
  store <4 x i32> %wide.load77.1, <4 x i32>* %37, align 4, !tbaa !1
  %index.next.1 = add i64 %index, 16
  %38 = icmp eq i64 %index.next.1, %end.idx.rnd.down
  br i1 %38, label %middle.block.loopexit.unr-lcssa, label %vector.body, !llvm.loop !15

middle.block.loopexit.unr-lcssa:                  ; preds = %vector.body
  br label %middle.block.loopexit

middle.block.loopexit:                            ; preds = %vector.body.preheader.split, %middle.block.loopexit.unr-lcssa
  br label %middle.block

middle.block:                                     ; preds = %middle.block.loopexit, %overflow.checked
  %resume.val = phi i64 [ 1, %overflow.checked ], [ %end.idx.rnd.down, %middle.block.loopexit ]
  %cmp.n = icmp eq i64 %end.idx, %resume.val
  br i1 %cmp.n, label %for.cond.cleanup.3, label %for.body.4.preheader

for.body.4.preheader:                             ; preds = %middle.block
  br label %for.body.4

for.cond.16.preheader.loopexit:                   ; preds = %for.cond.cleanup.3
  br label %for.cond.16.preheader

for.cond.16.preheader:                            ; preds = %for.cond.16.preheader.loopexit, %entry
  %cmp22.59 = icmp sgt i32 %x, 1
  %39 = add i32 %x, 3
  %40 = add i32 %x, -2
  %xtraiter = and i32 %39, 3
  %lcmp.mod = icmp eq i32 %xtraiter, 0
  %41 = icmp ult i32 %40, 3
  br label %for.cond.16

for.cond.cleanup.3.loopexit:                      ; preds = %for.body.4
  br label %for.cond.cleanup.3

for.cond.cleanup.3:                               ; preds = %for.cond.cleanup.3.loopexit, %middle.block, %for.cond.1.preheader
  %indvars.iv.next73 = add nuw nsw i64 %indvars.iv72, 1
  %lftr.wideiv74 = trunc i64 %indvars.iv.next73 to i32
  %exitcond75 = icmp eq i32 %lftr.wideiv74, %x
  br i1 %exitcond75, label %for.cond.16.preheader.loopexit, label %for.cond.1.preheader

for.body.4:                                       ; preds = %for.body.4.preheader, %for.body.4
  %indvars.iv67 = phi i64 [ %indvars.iv.next68, %for.body.4 ], [ %resume.val, %for.body.4.preheader ]
  %indvars.iv.next68 = add nuw nsw i64 %indvars.iv67, 1
  %arrayidx7 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv72, i64 %indvars.iv.next68
  %42 = load i32, i32* %arrayidx7, align 4, !tbaa !1
  %arrayidx11 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv72, i64 %indvars.iv67
  store i32 %42, i32* %arrayidx11, align 4, !tbaa !1
  %lftr.wideiv69 = trunc i64 %indvars.iv.next68 to i32
  %exitcond70 = icmp eq i32 %lftr.wideiv69, %x
  br i1 %exitcond70, label %for.cond.cleanup.3.loopexit, label %for.body.4, !llvm.loop !16

for.cond.16:                                      ; preds = %for.cond.16.preheader, %for.cond.cleanup.23
  %indvars.iv65 = phi i64 [ 0, %for.cond.16.preheader ], [ %indvars.iv.next66, %for.cond.cleanup.23 ]
  br i1 %cmp22.59, label %for.body.24.preheader, label %for.cond.cleanup.23

for.body.24.preheader:                            ; preds = %for.cond.16
  br i1 %lcmp.mod, label %for.body.24.preheader.split, label %for.body.24.prol.preheader

for.body.24.prol.preheader:                       ; preds = %for.body.24.preheader
  br label %for.body.24.prol

for.body.24.prol:                                 ; preds = %for.body.24.prol, %for.body.24.prol.preheader
  %indvars.iv.prol = phi i64 [ %indvars.iv.next.prol, %for.body.24.prol ], [ 1, %for.body.24.prol.preheader ]
  %prol.iter = phi i32 [ %prol.iter.sub, %for.body.24.prol ], [ %xtraiter, %for.body.24.prol.preheader ]
  %indvars.iv.next.prol = add nuw nsw i64 %indvars.iv.prol, 1
  %arrayidx29.prol = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next.prol, i64 %indvars.iv65
  %43 = load i32, i32* %arrayidx29.prol, align 4, !tbaa !1
  %arrayidx33.prol = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.prol, i64 %indvars.iv65
  store i32 %43, i32* %arrayidx33.prol, align 4, !tbaa !1
  %prol.iter.sub = add i32 %prol.iter, -1
  %prol.iter.cmp = icmp eq i32 %prol.iter.sub, 0
  br i1 %prol.iter.cmp, label %for.body.24.preheader.split.loopexit, label %for.body.24.prol, !llvm.loop !17

for.body.24.preheader.split.loopexit:             ; preds = %for.body.24.prol
  %indvars.iv.next.prol.lcssa = phi i64 [ %indvars.iv.next.prol, %for.body.24.prol ]
  br label %for.body.24.preheader.split

for.body.24.preheader.split:                      ; preds = %for.body.24.preheader, %for.body.24.preheader.split.loopexit
  %indvars.iv.unr = phi i64 [ 1, %for.body.24.preheader ], [ %indvars.iv.next.prol.lcssa, %for.body.24.preheader.split.loopexit ]
  br i1 %41, label %for.cond.cleanup.23.loopexit, label %for.body.24.preheader.split.split

for.body.24.preheader.split.split:                ; preds = %for.body.24.preheader.split
  br label %for.body.24

for.cond.cleanup.23.loopexit.unr-lcssa:           ; preds = %for.body.24
  br label %for.cond.cleanup.23.loopexit

for.cond.cleanup.23.loopexit:                     ; preds = %for.body.24.preheader.split, %for.cond.cleanup.23.loopexit.unr-lcssa
  br label %for.cond.cleanup.23

for.cond.cleanup.23:                              ; preds = %for.cond.cleanup.23.loopexit, %for.cond.16
  %indvars.iv.next66 = add nuw nsw i64 %indvars.iv65, 1
  br label %for.cond.16

for.body.24:                                      ; preds = %for.body.24, %for.body.24.preheader.split.split
  %indvars.iv = phi i64 [ %indvars.iv.unr, %for.body.24.preheader.split.split ], [ %indvars.iv.next.3, %for.body.24 ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx29 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next, i64 %indvars.iv65
  %44 = load i32, i32* %arrayidx29, align 4, !tbaa !1
  %arrayidx33 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv, i64 %indvars.iv65
  store i32 %44, i32* %arrayidx33, align 4, !tbaa !1
  %indvars.iv.next.1 = add nsw i64 %indvars.iv, 2
  %arrayidx29.1 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next.1, i64 %indvars.iv65
  %45 = load i32, i32* %arrayidx29.1, align 4, !tbaa !1
  %arrayidx33.1 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next, i64 %indvars.iv65
  store i32 %45, i32* %arrayidx33.1, align 4, !tbaa !1
  %indvars.iv.next.2 = add nsw i64 %indvars.iv, 3
  %arrayidx29.2 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next.2, i64 %indvars.iv65
  %46 = load i32, i32* %arrayidx29.2, align 4, !tbaa !1
  %arrayidx33.2 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next.1, i64 %indvars.iv65
  store i32 %46, i32* %arrayidx33.2, align 4, !tbaa !1
  %indvars.iv.next.3 = add nsw i64 %indvars.iv, 4
  %arrayidx29.3 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next.3, i64 %indvars.iv65
  %47 = load i32, i32* %arrayidx29.3, align 4, !tbaa !1
  %arrayidx33.3 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %indvars.iv.next.2, i64 %indvars.iv65
  store i32 %47, i32* %arrayidx33.3, align 4, !tbaa !1
  %lftr.wideiv.3 = trunc i64 %indvars.iv.next.3 to i32
  %exitcond.3 = icmp eq i32 %lftr.wideiv.3, %x
  br i1 %exitcond.3, label %for.cond.cleanup.23.loopexit.unr-lcssa, label %for.body.24
}

; Function Attrs: uwtable
define i32 @main() #4 {
entry:
  %call = tail call dereferenceable(280) %"class.std::basic_istream"* @_ZNSirsERi(%"class.std::basic_istream"* nonnull @_ZSt3cin, i32* nonnull dereferenceable(4) @n)
  %0 = load i32, i32* @n, align 4, !tbaa !1
  %cmp.41 = icmp sgt i32 %0, 0
  br i1 %cmp.41, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %_ZNKSt5ctypeIcE5widenEc.exit
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret i32 0

for.body:                                         ; preds = %for.body.preheader, %_ZNKSt5ctypeIcE5widenEc.exit
  %1 = phi i32 [ %20, %_ZNKSt5ctypeIcE5widenEc.exit ], [ %0, %for.body.preheader ]
  %qqq.042 = phi i32 [ %inc24, %_ZNKSt5ctypeIcE5widenEc.exit ], [ 0, %for.body.preheader ]
  store i32 0, i32* @sum, align 4, !tbaa !1
  store i32 0, i32* @i, align 4, !tbaa !1
  %cmp2.40 = icmp sgt i32 %1, 0
  br i1 %cmp2.40, label %for.cond.4.preheader.preheader, label %for.cond.13.thread

for.cond.4.preheader.preheader:                   ; preds = %for.body
  br label %for.cond.4.preheader

for.cond.13.thread:                               ; preds = %for.body
  store i32 %1, i32* @i, align 4, !tbaa !1
  br label %for.end.20

for.cond.4.preheader:                             ; preds = %for.cond.4.preheader.preheader, %for.inc.10
  %2 = phi i32 [ %inc11, %for.inc.10 ], [ 0, %for.cond.4.preheader.preheader ]
  %3 = phi i32 [ %8, %for.inc.10 ], [ %1, %for.cond.4.preheader.preheader ]
  store i32 0, i32* @j, align 4, !tbaa !1
  %cmp5.38 = icmp sgt i32 %3, 0
  br i1 %cmp5.38, label %for.body.6.preheader, label %for.inc.10

for.body.6.preheader:                             ; preds = %for.cond.4.preheader
  %idxprom7.50 = sext i32 %2 to i64
  %arrayidx8.51 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %idxprom7.50, i64 0
  %call9.52 = tail call dereferenceable(280) %"class.std::basic_istream"* @_ZNSirsERi(%"class.std::basic_istream"* nonnull @_ZSt3cin, i32* dereferenceable(4) %arrayidx8.51)
  %4 = load i32, i32* @j, align 4, !tbaa !1
  %inc.53 = add nsw i32 %4, 1
  store i32 %inc.53, i32* @j, align 4, !tbaa !1
  %5 = load i32, i32* @n, align 4, !tbaa !1
  %cmp5.54 = icmp slt i32 %inc.53, %5
  br i1 %cmp5.54, label %for.body.6.for.body.6_crit_edge.preheader, label %for.cond.4.for.inc.10_crit_edge

for.body.6.for.body.6_crit_edge.preheader:        ; preds = %for.body.6.preheader
  br label %for.body.6.for.body.6_crit_edge

for.body.6.for.body.6_crit_edge:                  ; preds = %for.body.6.for.body.6_crit_edge.preheader, %for.body.6.for.body.6_crit_edge
  %inc55 = phi i32 [ %inc, %for.body.6.for.body.6_crit_edge ], [ %inc.53, %for.body.6.for.body.6_crit_edge.preheader ]
  %.pre = load i32, i32* @i, align 4, !tbaa !1
  %idxprom = sext i32 %inc55 to i64
  %idxprom7 = sext i32 %.pre to i64
  %arrayidx8 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @a, i64 0, i64 %idxprom7, i64 %idxprom
  %call9 = tail call dereferenceable(280) %"class.std::basic_istream"* @_ZNSirsERi(%"class.std::basic_istream"* nonnull @_ZSt3cin, i32* dereferenceable(4) %arrayidx8)
  %6 = load i32, i32* @j, align 4, !tbaa !1
  %inc = add nsw i32 %6, 1
  store i32 %inc, i32* @j, align 4, !tbaa !1
  %7 = load i32, i32* @n, align 4, !tbaa !1
  %cmp5 = icmp slt i32 %inc, %7
  br i1 %cmp5, label %for.body.6.for.body.6_crit_edge, label %for.cond.4.for.inc.10_crit_edge.loopexit

for.cond.4.for.inc.10_crit_edge.loopexit:         ; preds = %for.body.6.for.body.6_crit_edge
  %.lcssa58 = phi i32 [ %7, %for.body.6.for.body.6_crit_edge ]
  br label %for.cond.4.for.inc.10_crit_edge

for.cond.4.for.inc.10_crit_edge:                  ; preds = %for.cond.4.for.inc.10_crit_edge.loopexit, %for.body.6.preheader
  %.lcssa = phi i32 [ %5, %for.body.6.preheader ], [ %.lcssa58, %for.cond.4.for.inc.10_crit_edge.loopexit ]
  %.pre46 = load i32, i32* @i, align 4, !tbaa !1
  br label %for.inc.10

for.inc.10:                                       ; preds = %for.cond.4.for.inc.10_crit_edge, %for.cond.4.preheader
  %8 = phi i32 [ %.lcssa, %for.cond.4.for.inc.10_crit_edge ], [ %3, %for.cond.4.preheader ]
  %9 = phi i32 [ %.pre46, %for.cond.4.for.inc.10_crit_edge ], [ %2, %for.cond.4.preheader ]
  %inc11 = add nsw i32 %9, 1
  store i32 %inc11, i32* @i, align 4, !tbaa !1
  %cmp2 = icmp slt i32 %inc11, %8
  br i1 %cmp2, label %for.cond.4.preheader, label %for.cond.13

for.cond.13:                                      ; preds = %for.inc.10
  %.lcssa59 = phi i32 [ %8, %for.inc.10 ]
  store i32 %.lcssa59, i32* @i, align 4, !tbaa !1
  %cmp14 = icmp sgt i32 %.lcssa59, 1
  br i1 %cmp14, label %for.body.15, label %for.end.20

for.body.15:                                      ; preds = %for.cond.13
  %.lcssa59.lcssa = phi i32 [ %.lcssa59, %for.cond.13 ]
  %call16 = tail call i32 @_Z1fi(i32 %.lcssa59.lcssa)
  unreachable

for.end.20:                                       ; preds = %for.cond.13.thread, %for.cond.13
  %10 = load i32, i32* @sum, align 4, !tbaa !1
  %call21 = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"* nonnull @_ZSt4cout, i32 %10)
  %11 = bitcast %"class.std::basic_ostream"* %call21 to i8**
  %vtable.i = load i8*, i8** %11, align 8, !tbaa !18
  %vbase.offset.ptr.i = getelementptr i8, i8* %vtable.i, i64 -24
  %12 = bitcast i8* %vbase.offset.ptr.i to i64*
  %vbase.offset.i = load i64, i64* %12, align 8
  %13 = bitcast %"class.std::basic_ostream"* %call21 to i8*
  %add.ptr.i = getelementptr inbounds i8, i8* %13, i64 %vbase.offset.i
  %_M_ctype.i = getelementptr inbounds i8, i8* %add.ptr.i, i64 240
  %14 = bitcast i8* %_M_ctype.i to %"class.std::ctype"**
  %15 = load %"class.std::ctype"*, %"class.std::ctype"** %14, align 8, !tbaa !20
  %tobool.i.34 = icmp eq %"class.std::ctype"* %15, null
  br i1 %tobool.i.34, label %if.then.i.35, label %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit

if.then.i.35:                                     ; preds = %for.end.20
  tail call void @_ZSt16__throw_bad_castv() #6
  unreachable

_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit:    ; preds = %for.end.20
  %_M_widen_ok.i = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %15, i64 0, i32 8
  %16 = load i8, i8* %_M_widen_ok.i, align 1, !tbaa !24
  %tobool.i = icmp eq i8 %16, 0
  br i1 %tobool.i, label %if.end.i, label %if.then.i

if.then.i:                                        ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit
  %arrayidx.i = getelementptr inbounds %"class.std::ctype", %"class.std::ctype"* %15, i64 0, i32 9, i64 10
  %17 = load i8, i8* %arrayidx.i, align 1, !tbaa !26
  br label %_ZNKSt5ctypeIcE5widenEc.exit

if.end.i:                                         ; preds = %_ZSt13__check_facetISt5ctypeIcEERKT_PS3_.exit
  tail call void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"* %15)
  %18 = bitcast %"class.std::ctype"* %15 to i8 (%"class.std::ctype"*, i8)***
  %vtable.i.32 = load i8 (%"class.std::ctype"*, i8)**, i8 (%"class.std::ctype"*, i8)*** %18, align 8, !tbaa !18
  %vfn.i = getelementptr inbounds i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vtable.i.32, i64 6
  %19 = load i8 (%"class.std::ctype"*, i8)*, i8 (%"class.std::ctype"*, i8)** %vfn.i, align 8
  %call.i.33 = tail call signext i8 %19(%"class.std::ctype"* %15, i8 signext 10)
  br label %_ZNKSt5ctypeIcE5widenEc.exit

_ZNKSt5ctypeIcE5widenEc.exit:                     ; preds = %if.then.i, %if.end.i
  %retval.0.i = phi i8 [ %17, %if.then.i ], [ %call.i.33, %if.end.i ]
  %call1.i = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"* nonnull %call21, i8 signext %retval.0.i)
  %call.i = tail call dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"* nonnull %call1.i)
  %inc24 = add nuw nsw i32 %qqq.042, 1
  %20 = load i32, i32* @n, align 4, !tbaa !1
  %cmp = icmp slt i32 %inc24, %20
  br i1 %cmp, label %for.body, label %for.cond.cleanup.loopexit
}

declare dereferenceable(280) %"class.std::basic_istream"* @_ZNSirsERi(%"class.std::basic_istream"*, i32* dereferenceable(4)) #0

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZNSolsEi(%"class.std::basic_ostream"*, i32) #0

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo3putEc(%"class.std::basic_ostream"*, i8 signext) #0

declare dereferenceable(272) %"class.std::basic_ostream"* @_ZNSo5flushEv(%"class.std::basic_ostream"*) #0

; Function Attrs: noreturn
declare void @_ZSt16__throw_bad_castv() #5

declare void @_ZNKSt5ctypeIcE13_M_widen_initEv(%"class.std::ctype"*) #0

define internal void @_GLOBAL__sub_I_2000.txt.cpp() #0 section ".text.startup" {
entry:
  tail call void @_ZNSt8ios_base4InitC1Ev(%"class.std::ios_base::Init"* nonnull @_ZStL8__ioinit)
  %0 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%"class.std::ios_base::Init"*)* @_ZNSt8ios_base4InitD1Ev to void (i8*)*), i8* getelementptr inbounds (%"class.std::ios_base::Init", %"class.std::ios_base::Init"* @_ZStL8__ioinit, i64 0, i32 0), i8* nonnull @__dso_handle) #2
  ret void
}

attributes #0 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { nounwind }
attributes #3 = { noreturn nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #4 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #5 = { noreturn "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #6 = { noreturn }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.1 (tags/RELEASE_371/final)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = distinct !{!5, !6, !7}
!6 = !{!"llvm.loop.vectorize.width", i32 1}
!7 = !{!"llvm.loop.interleave.count", i32 1}
!8 = distinct !{!8, !6, !7}
!9 = distinct !{!9, !10, !6, !7}
!10 = !{!"llvm.loop.unroll.runtime.disable"}
!11 = distinct !{!11, !10, !6, !7}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.unroll.disable"}
!14 = distinct !{!14, !13}
!15 = distinct !{!15, !6, !7}
!16 = distinct !{!16, !10, !6, !7}
!17 = distinct !{!17, !13}
!18 = !{!19, !19, i64 0}
!19 = !{!"vtable pointer", !4, i64 0}
!20 = !{!21, !22, i64 240}
!21 = !{!"_ZTSSt9basic_iosIcSt11char_traitsIcEE", !22, i64 216, !3, i64 224, !23, i64 225, !22, i64 232, !22, i64 240, !22, i64 248, !22, i64 256}
!22 = !{!"any pointer", !3, i64 0}
!23 = !{!"bool", !3, i64 0}
!24 = !{!25, !3, i64 56}
!25 = !{!"_ZTSSt5ctypeIcE", !22, i64 16, !23, i64 24, !22, i64 32, !22, i64 40, !22, i64 48, !3, i64 56, !3, i64 57, !3, i64 313, !3, i64 569}
!26 = !{!3, !3, i64 0}
