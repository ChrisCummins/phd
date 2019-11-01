; ModuleID = '/home/cec/.cache/bazel/_bazel_cec/d1665aef25bbeeb91c01df7ddc90dba7/execroot/phd/bazel-out/k8-opt/bin/experimental/compilers/reachability/cfg_datasets.runfiles/linux_srcs/arch/x86/pci/ce4100.c'
source_filename = "/home/cec/.cache/bazel/_bazel_cec/d1665aef25bbeeb91c01df7ddc90dba7/execroot/phd/bazel-out/k8-opt/bin/experimental/compilers/reachability/cfg_datasets.runfiles/linux_srcs/arch/x86/pci/ce4100.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.pci_raw_ops = type { i32 (i32, i32, i32, i32, i32, i32*)*, i32 (i32, i32, i32, i32, i32, i32)* }
%struct.sim_dev_reg = type { i32, i32, {}*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }
%struct.sim_reg = type { i32, i32 }
%struct.raw_spinlock = type { %struct.qspinlock }
%struct.qspinlock = type { %union.anon }
%union.anon = type { %struct.atomic_t }
%struct.atomic_t = type { i32 }

@pci_direct_conf1 = external constant %struct.pci_raw_ops, align 8
@ce4100_pci_conf = internal constant %struct.pci_raw_ops { i32 (i32, i32, i32, i32, i32, i32*)* @ce4100_conf_read, i32 (i32, i32, i32, i32, i32, i32)* @ce4100_conf_write }, align 8
@raw_pci_ops = external global %struct.pci_raw_ops*, align 8
@bus1_fixups = internal global <{ { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } }> <{ { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 16, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -16777216 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 16, i32 20, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -256 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 17, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -65536 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 24, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -65536 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 32, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -131072 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 33, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -131072 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 48, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -524288 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 49, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -524288 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 50, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -65536 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 64, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -1048576 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 65, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -65536 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 66, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -65536 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 72, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -1048576 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 72, i32 20, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -65536 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 80, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -256 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 80, i32 20, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -268435456 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 88, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -256 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 88, i32 20, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -256 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 89, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -256 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 90, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -256 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 90, i32 20, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -256 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 90, i32 24, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -256 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 91, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -256 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 91, i32 20, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -256 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 92, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -256 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 93, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -65536 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 94, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -256 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 95, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -65536 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 95, i32 60, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_noirq_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -256 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 96, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -131072 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 96, i32 20, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -256 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 97, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -1024 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 104, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @ehci_reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -32768 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 105, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @ehci_reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -32768 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 112, i32 8, void (%struct.sim_dev_reg*)* @sata_revid_init, void (%struct.sim_dev_reg*, i32*)* @sata_revid_read, void (%struct.sim_dev_reg*, i32)* null, %struct.sim_reg zeroinitializer }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 112, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg zeroinitializer }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 112, i32 20, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg zeroinitializer }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 112, i32 24, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg zeroinitializer }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 112, i32 28, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg zeroinitializer }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 112, i32 32, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg zeroinitializer }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 112, i32 36, void (%struct.sim_dev_reg*)* @sata_reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -512 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 120, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -65536 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 120, i32 20, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -65536 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 128, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -65536 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 128, i32 20, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -67108864 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 128, i32 24, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -67108864 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 128, i32 60, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_noirq_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -256 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 136, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -131072 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 144, i32 16, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -1024 } }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } { i32 144, i32 60, void (%struct.sim_dev_reg*)* @reg_init, void (%struct.sim_dev_reg*, i32*)* @reg_noirq_read, void (%struct.sim_dev_reg*, i32)* @reg_write, %struct.sim_reg { i32 0, i32 -256 } } }>, align 16
@.str = private unnamed_addr constant [192 x i8] c"/home/cec/.cache/bazel/_bazel_cec/d1665aef25bbeeb91c01df7ddc90dba7/execroot/phd/bazel-out/k8-opt/bin/experimental/compilers/reachability/cfg_datasets.runfiles/linux_srcs/arch/x86/pci/ce4100.c\00", align 1
@pci_config_lock = external global %struct.raw_spinlock, align 4

; Function Attrs: noinline nounwind optnone uwtable
define void @sata_revid_init(%struct.sim_dev_reg*) #0 {
  %2 = alloca %struct.sim_dev_reg*, align 8
  store %struct.sim_dev_reg* %0, %struct.sim_dev_reg** %2, align 8
  %3 = load %struct.sim_dev_reg*, %struct.sim_dev_reg** %2, align 8
  %4 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %3, i32 0, i32 5
  %5 = getelementptr inbounds %struct.sim_reg, %struct.sim_reg* %4, i32 0, i32 0
  store i32 17170688, i32* %5, align 8
  %6 = load %struct.sim_dev_reg*, %struct.sim_dev_reg** %2, align 8
  %7 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %6, i32 0, i32 5
  %8 = getelementptr inbounds %struct.sim_reg, %struct.sim_reg* %7, i32 0, i32 1
  store i32 0, i32* %8, align 4
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define i32 @bridge_read(i32, i32, i32, i32*) #0 {
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32*, align 8
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  store i32 %0, i32* %5, align 4
  store i32 %1, i32* %6, align 4
  store i32 %2, i32* %7, align 4
  store i32* %3, i32** %8, align 8
  store i32 0, i32* %11, align 4
  %12 = load i32, i32* %6, align 4
  switch i32 %12, label %64 [
    i32 16, label %13
    i32 17, label %13
    i32 18, label %13
    i32 19, label %13
    i32 24, label %15
    i32 26, label %21
    i32 32, label %23
    i32 34, label %23
    i32 36, label %56
    i32 38, label %58
    i32 28, label %60
    i32 29, label %62
  ]

; <label>:13:                                     ; preds = %4, %4, %4, %4
  %14 = load i32*, i32** %8, align 8
  store i32 0, i32* %14, align 4
  br label %65

; <label>:15:                                     ; preds = %4
  %16 = load i32, i32* %7, align 4
  %17 = icmp eq i32 %16, 4
  br i1 %17, label %18, label %20

; <label>:18:                                     ; preds = %15
  %19 = load i32*, i32** %8, align 8
  store i32 65792, i32* %19, align 4
  br label %20

; <label>:20:                                     ; preds = %18, %15
  br label %65

; <label>:21:                                     ; preds = %4
  %22 = load i32*, i32** %8, align 8
  store i32 1, i32* %22, align 4
  br label %65

; <label>:23:                                     ; preds = %4, %4
  %24 = load i32 (i32, i32, i32, i32, i32, i32*)*, i32 (i32, i32, i32, i32, i32, i32*)** getelementptr inbounds (%struct.pci_raw_ops, %struct.pci_raw_ops* @pci_direct_conf1, i32 0, i32 0), align 8
  %25 = load i32, i32* %5, align 4
  %26 = call i32 %24(i32 0, i32 0, i32 %25, i32 16, i32 4, i32* %9)
  %27 = load i32, i32* %9, align 4
  %28 = add i32 %27, 536870911
  store i32 %28, i32* %10, align 4
  %29 = load i32, i32* %10, align 4
  %30 = lshr i32 %29, 16
  store i32 %30, i32* %10, align 4
  %31 = load i32, i32* %10, align 4
  %32 = and i32 %31, 65520
  store i32 %32, i32* %10, align 4
  %33 = load i32, i32* %9, align 4
  %34 = lshr i32 %33, 16
  store i32 %34, i32* %9, align 4
  %35 = load i32, i32* %9, align 4
  %36 = and i32 %35, 65520
  store i32 %36, i32* %9, align 4
  %37 = load i32, i32* %6, align 4
  %38 = icmp eq i32 %37, 34
  br i1 %38, label %39, label %42

; <label>:39:                                     ; preds = %23
  %40 = load i32, i32* %10, align 4
  %41 = load i32*, i32** %8, align 8
  store i32 %40, i32* %41, align 4
  br label %55

; <label>:42:                                     ; preds = %23
  %43 = load i32, i32* %7, align 4
  %44 = icmp eq i32 %43, 2
  br i1 %44, label %45, label %48

; <label>:45:                                     ; preds = %42
  %46 = load i32, i32* %9, align 4
  %47 = load i32*, i32** %8, align 8
  store i32 %46, i32* %47, align 4
  br label %54

; <label>:48:                                     ; preds = %42
  %49 = load i32, i32* %10, align 4
  %50 = shl i32 %49, 16
  %51 = load i32, i32* %9, align 4
  %52 = or i32 %50, %51
  %53 = load i32*, i32** %8, align 8
  store i32 %52, i32* %53, align 4
  br label %54

; <label>:54:                                     ; preds = %48, %45
  br label %55

; <label>:55:                                     ; preds = %54, %39
  br label %65

; <label>:56:                                     ; preds = %4
  %57 = load i32*, i32** %8, align 8
  store i32 65520, i32* %57, align 4
  br label %65

; <label>:58:                                     ; preds = %4
  %59 = load i32*, i32** %8, align 8
  store i32 0, i32* %59, align 4
  br label %65

; <label>:60:                                     ; preds = %4
  %61 = load i32*, i32** %8, align 8
  store i32 240, i32* %61, align 4
  br label %65

; <label>:62:                                     ; preds = %4
  %63 = load i32*, i32** %8, align 8
  store i32 0, i32* %63, align 4
  br label %65

; <label>:64:                                     ; preds = %4
  store i32 1, i32* %11, align 4
  br label %65

; <label>:65:                                     ; preds = %64, %62, %60, %58, %56, %55, %21, %20, %13
  %66 = load i32, i32* %11, align 4
  ret i32 %66
}

; Function Attrs: cold noinline nounwind optnone uwtable
define i32 @ce4100_pci_init() #1 section ".init.text" {
  call void @init_sim_regs() #3
  store %struct.pci_raw_ops* @ce4100_pci_conf, %struct.pci_raw_ops** @raw_pci_ops, align 8
  ret i32 1
}

; Function Attrs: cold noinline nounwind optnone uwtable
define internal void @init_sim_regs() #1 section ".init.text" {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  br label %2

; <label>:2:                                      ; preds = %25, %0
  %3 = load i32, i32* %1, align 4
  %4 = sext i32 %3 to i64
  %5 = icmp ult i64 %4, 50
  br i1 %5, label %6, label %28

; <label>:6:                                      ; preds = %2
  %7 = load i32, i32* %1, align 4
  %8 = sext i32 %7 to i64
  %9 = getelementptr inbounds [50 x %struct.sim_dev_reg], [50 x %struct.sim_dev_reg]* bitcast (<{ { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } }>* @bus1_fixups to [50 x %struct.sim_dev_reg]*), i64 0, i64 %8
  %10 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %9, i32 0, i32 2
  %11 = bitcast {}** %10 to void (%struct.sim_dev_reg*)**
  %12 = load void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*)** %11, align 8
  %13 = icmp ne void (%struct.sim_dev_reg*)* %12, null
  br i1 %13, label %14, label %24

; <label>:14:                                     ; preds = %6
  %15 = load i32, i32* %1, align 4
  %16 = sext i32 %15 to i64
  %17 = getelementptr inbounds [50 x %struct.sim_dev_reg], [50 x %struct.sim_dev_reg]* bitcast (<{ { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } }>* @bus1_fixups to [50 x %struct.sim_dev_reg]*), i64 0, i64 %16
  %18 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %17, i32 0, i32 2
  %19 = bitcast {}** %18 to void (%struct.sim_dev_reg*)**
  %20 = load void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*)** %19, align 8
  %21 = load i32, i32* %1, align 4
  %22 = sext i32 %21 to i64
  %23 = getelementptr inbounds [50 x %struct.sim_dev_reg], [50 x %struct.sim_dev_reg]* bitcast (<{ { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } }>* @bus1_fixups to [50 x %struct.sim_dev_reg]*), i64 0, i64 %22
  call void %20(%struct.sim_dev_reg* %23)
  br label %24

; <label>:24:                                     ; preds = %14, %6
  br label %25

; <label>:25:                                     ; preds = %24
  %26 = load i32, i32* %1, align 4
  %27 = add nsw i32 %26, 1
  store i32 %27, i32* %1, align 4
  br label %2

; <label>:28:                                     ; preds = %2
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @reg_init(%struct.sim_dev_reg*) #0 {
  %2 = alloca %struct.sim_dev_reg*, align 8
  store %struct.sim_dev_reg* %0, %struct.sim_dev_reg** %2, align 8
  %3 = load i32 (i32, i32, i32, i32, i32, i32*)*, i32 (i32, i32, i32, i32, i32, i32*)** getelementptr inbounds (%struct.pci_raw_ops, %struct.pci_raw_ops* @pci_direct_conf1, i32 0, i32 0), align 8
  %4 = load %struct.sim_dev_reg*, %struct.sim_dev_reg** %2, align 8
  %5 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %4, i32 0, i32 0
  %6 = load i32, i32* %5, align 8
  %7 = load %struct.sim_dev_reg*, %struct.sim_dev_reg** %2, align 8
  %8 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %7, i32 0, i32 1
  %9 = load i32, i32* %8, align 4
  %10 = load %struct.sim_dev_reg*, %struct.sim_dev_reg** %2, align 8
  %11 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %10, i32 0, i32 5
  %12 = getelementptr inbounds %struct.sim_reg, %struct.sim_reg* %11, i32 0, i32 0
  %13 = call i32 %3(i32 0, i32 1, i32 %6, i32 %9, i32 4, i32* %12)
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @reg_read(%struct.sim_dev_reg*, i32*) #0 {
  %3 = alloca %struct.sim_dev_reg*, align 8
  %4 = alloca i32*, align 8
  store %struct.sim_dev_reg* %0, %struct.sim_dev_reg** %3, align 8
  store i32* %1, i32** %4, align 8
  %5 = load %struct.sim_dev_reg*, %struct.sim_dev_reg** %3, align 8
  %6 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %5, i32 0, i32 5
  %7 = getelementptr inbounds %struct.sim_reg, %struct.sim_reg* %6, i32 0, i32 0
  %8 = load i32, i32* %7, align 8
  %9 = load i32*, i32** %4, align 8
  store i32 %8, i32* %9, align 4
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @reg_write(%struct.sim_dev_reg*, i32) #0 {
  %3 = alloca %struct.sim_dev_reg*, align 8
  %4 = alloca i32, align 4
  store %struct.sim_dev_reg* %0, %struct.sim_dev_reg** %3, align 8
  store i32 %1, i32* %4, align 4
  %5 = load i32, i32* %4, align 4
  %6 = load %struct.sim_dev_reg*, %struct.sim_dev_reg** %3, align 8
  %7 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %6, i32 0, i32 5
  %8 = getelementptr inbounds %struct.sim_reg, %struct.sim_reg* %7, i32 0, i32 1
  %9 = load i32, i32* %8, align 4
  %10 = and i32 %5, %9
  %11 = load %struct.sim_dev_reg*, %struct.sim_dev_reg** %3, align 8
  %12 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %11, i32 0, i32 5
  %13 = getelementptr inbounds %struct.sim_reg, %struct.sim_reg* %12, i32 0, i32 0
  %14 = load i32, i32* %13, align 8
  %15 = load %struct.sim_dev_reg*, %struct.sim_dev_reg** %3, align 8
  %16 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %15, i32 0, i32 5
  %17 = getelementptr inbounds %struct.sim_reg, %struct.sim_reg* %16, i32 0, i32 1
  %18 = load i32, i32* %17, align 4
  %19 = xor i32 %18, -1
  %20 = and i32 %14, %19
  %21 = or i32 %10, %20
  %22 = load %struct.sim_dev_reg*, %struct.sim_dev_reg** %3, align 8
  %23 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %22, i32 0, i32 5
  %24 = getelementptr inbounds %struct.sim_reg, %struct.sim_reg* %23, i32 0, i32 0
  store i32 %21, i32* %24, align 8
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @reg_noirq_read(%struct.sim_dev_reg*, i32*) #0 {
  %3 = alloca %struct.sim_dev_reg*, align 8
  %4 = alloca i32*, align 8
  store %struct.sim_dev_reg* %0, %struct.sim_dev_reg** %3, align 8
  store i32* %1, i32** %4, align 8
  %5 = load %struct.sim_dev_reg*, %struct.sim_dev_reg** %3, align 8
  %6 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %5, i32 0, i32 5
  %7 = getelementptr inbounds %struct.sim_reg, %struct.sim_reg* %6, i32 0, i32 0
  %8 = load i32, i32* %7, align 8
  %9 = and i32 %8, 268370175
  %10 = load i32*, i32** %4, align 8
  store i32 %9, i32* %10, align 4
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @ehci_reg_read(%struct.sim_dev_reg*, i32*) #0 {
  %3 = alloca %struct.sim_dev_reg*, align 8
  %4 = alloca i32*, align 8
  store %struct.sim_dev_reg* %0, %struct.sim_dev_reg** %3, align 8
  store i32* %1, i32** %4, align 8
  %5 = load %struct.sim_dev_reg*, %struct.sim_dev_reg** %3, align 8
  %6 = load i32*, i32** %4, align 8
  call void @reg_read(%struct.sim_dev_reg* %5, i32* %6)
  %7 = load i32*, i32** %4, align 8
  %8 = load i32, i32* %7, align 4
  %9 = load %struct.sim_dev_reg*, %struct.sim_dev_reg** %3, align 8
  %10 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %9, i32 0, i32 5
  %11 = getelementptr inbounds %struct.sim_reg, %struct.sim_reg* %10, i32 0, i32 1
  %12 = load i32, i32* %11, align 4
  %13 = icmp ne i32 %8, %12
  br i1 %13, label %14, label %18

; <label>:14:                                     ; preds = %2
  %15 = load i32*, i32** %4, align 8
  %16 = load i32, i32* %15, align 4
  %17 = or i32 %16, 256
  store i32 %17, i32* %15, align 4
  br label %18

; <label>:18:                                     ; preds = %14, %2
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @sata_revid_read(%struct.sim_dev_reg*, i32*) #0 {
  %3 = alloca %struct.sim_dev_reg*, align 8
  %4 = alloca i32*, align 8
  store %struct.sim_dev_reg* %0, %struct.sim_dev_reg** %3, align 8
  store i32* %1, i32** %4, align 8
  %5 = load %struct.sim_dev_reg*, %struct.sim_dev_reg** %3, align 8
  %6 = load i32*, i32** %4, align 8
  call void @reg_read(%struct.sim_dev_reg* %5, i32* %6)
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @sata_reg_init(%struct.sim_dev_reg*) #0 {
  %2 = alloca %struct.sim_dev_reg*, align 8
  store %struct.sim_dev_reg* %0, %struct.sim_dev_reg** %2, align 8
  %3 = load i32 (i32, i32, i32, i32, i32, i32*)*, i32 (i32, i32, i32, i32, i32, i32*)** getelementptr inbounds (%struct.pci_raw_ops, %struct.pci_raw_ops* @pci_direct_conf1, i32 0, i32 0), align 8
  %4 = load %struct.sim_dev_reg*, %struct.sim_dev_reg** %2, align 8
  %5 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %4, i32 0, i32 5
  %6 = getelementptr inbounds %struct.sim_reg, %struct.sim_reg* %5, i32 0, i32 0
  %7 = call i32 %3(i32 0, i32 1, i32 112, i32 16, i32 4, i32* %6)
  %8 = load %struct.sim_dev_reg*, %struct.sim_dev_reg** %2, align 8
  %9 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %8, i32 0, i32 5
  %10 = getelementptr inbounds %struct.sim_reg, %struct.sim_reg* %9, i32 0, i32 0
  %11 = load i32, i32* %10, align 8
  %12 = add i32 %11, 1024
  store i32 %12, i32* %10, align 8
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @ce4100_conf_read(i32, i32, i32, i32, i32, i32*) #0 {
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca i32*, align 8
  %14 = alloca i32, align 4
  %15 = alloca i64, align 8
  store i32 %0, i32* %8, align 4
  store i32 %1, i32* %9, align 4
  store i32 %2, i32* %10, align 4
  store i32 %3, i32* %11, align 4
  store i32 %4, i32* %12, align 4
  store i32* %5, i32** %13, align 8
  %16 = load i32, i32* %8, align 4
  %17 = icmp ne i32 %16, 0
  %18 = xor i1 %17, true
  %19 = xor i1 %18, true
  %20 = zext i1 %19 to i32
  store i32 %20, i32* %14, align 4
  %21 = load i32, i32* %14, align 4
  %22 = icmp ne i32 %21, 0
  %23 = xor i1 %22, true
  %24 = xor i1 %23, true
  %25 = zext i1 %24 to i32
  %26 = sext i32 %25 to i64
  %27 = icmp ne i64 %26, 0
  br i1 %27, label %28, label %33

; <label>:28:                                     ; preds = %6
  br label %29

; <label>:29:                                     ; preds = %28
  br label %30

; <label>:30:                                     ; preds = %29
  call void asm sideeffect "1:\09.byte 0x0f, 0x0b\0A.pushsection __bug_table,\22aw\22\0A2:\09.long 1b - 2b\09# bug_entry::bug_addr\0A\09.long ${0:c} - 2b\09# bug_entry::file\0A\09.word ${1:c}\09# bug_entry::line\0A\09.word ${2:c}\09# bug_entry::flags\0A\09.org 2b+${3:c}\0A.popsection", "i,i,i,i,~{dirflag},~{fpsr},~{flags}"(i8* getelementptr inbounds ([192 x i8], [192 x i8]* @.str, i32 0, i32 0), i32 282, i32 2305, i64 12) #4, !srcloc !2
  br label %31

; <label>:31:                                     ; preds = %30
  call void asm sideeffect "${0:c}:\0A\09.pushsection .discard.reachable\0A\09.long ${0:c}b - .\0A\09.popsection\0A\09", "i,~{dirflag},~{fpsr},~{flags}"(i32 59) #4, !srcloc !3
  br label %32

; <label>:32:                                     ; preds = %31
  br label %33

; <label>:33:                                     ; preds = %32, %6
  %34 = load i32, i32* %14, align 4
  %35 = icmp ne i32 %34, 0
  %36 = xor i1 %35, true
  %37 = xor i1 %36, true
  %38 = zext i1 %37 to i32
  %39 = sext i32 %38 to i64
  store i64 %39, i64* %15, align 8
  %40 = load i64, i64* %15, align 8
  %41 = load i32, i32* %9, align 4
  %42 = icmp eq i32 %41, 1
  br i1 %42, label %43, label %51

; <label>:43:                                     ; preds = %33
  %44 = load i32, i32* %10, align 4
  %45 = load i32, i32* %11, align 4
  %46 = load i32, i32* %12, align 4
  %47 = load i32*, i32** %13, align 8
  %48 = call i32 @ce4100_bus1_read(i32 %44, i32 %45, i32 %46, i32* %47)
  %49 = icmp ne i32 %48, 0
  br i1 %49, label %51, label %50

; <label>:50:                                     ; preds = %43
  store i32 0, i32* %7, align 4
  br label %74

; <label>:51:                                     ; preds = %43, %33
  %52 = load i32, i32* %9, align 4
  %53 = icmp eq i32 %52, 0
  br i1 %53, label %54, label %65

; <label>:54:                                     ; preds = %51
  %55 = load i32, i32* %10, align 4
  %56 = icmp eq i32 8, %55
  br i1 %56, label %57, label %65

; <label>:57:                                     ; preds = %54
  %58 = load i32, i32* %10, align 4
  %59 = load i32, i32* %11, align 4
  %60 = load i32, i32* %12, align 4
  %61 = load i32*, i32** %13, align 8
  %62 = call i32 @bridge_read(i32 %58, i32 %59, i32 %60, i32* %61)
  %63 = icmp ne i32 %62, 0
  br i1 %63, label %65, label %64

; <label>:64:                                     ; preds = %57
  store i32 0, i32* %7, align 4
  br label %74

; <label>:65:                                     ; preds = %57, %54, %51
  %66 = load i32 (i32, i32, i32, i32, i32, i32*)*, i32 (i32, i32, i32, i32, i32, i32*)** getelementptr inbounds (%struct.pci_raw_ops, %struct.pci_raw_ops* @pci_direct_conf1, i32 0, i32 0), align 8
  %67 = load i32, i32* %8, align 4
  %68 = load i32, i32* %9, align 4
  %69 = load i32, i32* %10, align 4
  %70 = load i32, i32* %11, align 4
  %71 = load i32, i32* %12, align 4
  %72 = load i32*, i32** %13, align 8
  %73 = call i32 %66(i32 %67, i32 %68, i32 %69, i32 %70, i32 %71, i32* %72)
  store i32 %73, i32* %7, align 4
  br label %74

; <label>:74:                                     ; preds = %65, %64, %50
  %75 = load i32, i32* %7, align 4
  ret i32 %75
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @ce4100_conf_write(i32, i32, i32, i32, i32, i32) #0 {
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i32, align 4
  %11 = alloca i32, align 4
  %12 = alloca i32, align 4
  %13 = alloca i32, align 4
  %14 = alloca i32, align 4
  %15 = alloca i64, align 8
  store i32 %0, i32* %8, align 4
  store i32 %1, i32* %9, align 4
  store i32 %2, i32* %10, align 4
  store i32 %3, i32* %11, align 4
  store i32 %4, i32* %12, align 4
  store i32 %5, i32* %13, align 4
  %16 = load i32, i32* %8, align 4
  %17 = icmp ne i32 %16, 0
  %18 = xor i1 %17, true
  %19 = xor i1 %18, true
  %20 = zext i1 %19 to i32
  store i32 %20, i32* %14, align 4
  %21 = load i32, i32* %14, align 4
  %22 = icmp ne i32 %21, 0
  %23 = xor i1 %22, true
  %24 = xor i1 %23, true
  %25 = zext i1 %24 to i32
  %26 = sext i32 %25 to i64
  %27 = icmp ne i64 %26, 0
  br i1 %27, label %28, label %33

; <label>:28:                                     ; preds = %6
  br label %29

; <label>:29:                                     ; preds = %28
  br label %30

; <label>:30:                                     ; preds = %29
  call void asm sideeffect "1:\09.byte 0x0f, 0x0b\0A.pushsection __bug_table,\22aw\22\0A2:\09.long 1b - 2b\09# bug_entry::bug_addr\0A\09.long ${0:c} - 2b\09# bug_entry::file\0A\09.word ${1:c}\09# bug_entry::line\0A\09.word ${2:c}\09# bug_entry::flags\0A\09.org 2b+${3:c}\0A.popsection", "i,i,i,i,~{dirflag},~{fpsr},~{flags}"(i8* getelementptr inbounds ([192 x i8], [192 x i8]* @.str, i32 0, i32 0), i32 316, i32 2305, i64 12) #4, !srcloc !4
  br label %31

; <label>:31:                                     ; preds = %30
  call void asm sideeffect "${0:c}:\0A\09.pushsection .discard.reachable\0A\09.long ${0:c}b - .\0A\09.popsection\0A\09", "i,~{dirflag},~{fpsr},~{flags}"(i32 60) #4, !srcloc !5
  br label %32

; <label>:32:                                     ; preds = %31
  br label %33

; <label>:33:                                     ; preds = %32, %6
  %34 = load i32, i32* %14, align 4
  %35 = icmp ne i32 %34, 0
  %36 = xor i1 %35, true
  %37 = xor i1 %36, true
  %38 = zext i1 %37 to i32
  %39 = sext i32 %38 to i64
  store i64 %39, i64* %15, align 8
  %40 = load i64, i64* %15, align 8
  %41 = load i32, i32* %9, align 4
  %42 = icmp eq i32 %41, 1
  br i1 %42, label %43, label %51

; <label>:43:                                     ; preds = %33
  %44 = load i32, i32* %10, align 4
  %45 = load i32, i32* %11, align 4
  %46 = load i32, i32* %12, align 4
  %47 = load i32, i32* %13, align 4
  %48 = call i32 @ce4100_bus1_write(i32 %44, i32 %45, i32 %46, i32 %47)
  %49 = icmp ne i32 %48, 0
  br i1 %49, label %51, label %50

; <label>:50:                                     ; preds = %43
  store i32 0, i32* %7, align 4
  br label %71

; <label>:51:                                     ; preds = %43, %33
  %52 = load i32, i32* %9, align 4
  %53 = icmp eq i32 %52, 0
  br i1 %53, label %54, label %62

; <label>:54:                                     ; preds = %51
  %55 = load i32, i32* %10, align 4
  %56 = icmp eq i32 8, %55
  br i1 %56, label %57, label %62

; <label>:57:                                     ; preds = %54
  %58 = load i32, i32* %11, align 4
  %59 = and i32 %58, -4
  %60 = icmp eq i32 %59, 16
  br i1 %60, label %61, label %62

; <label>:61:                                     ; preds = %57
  store i32 0, i32* %7, align 4
  br label %71

; <label>:62:                                     ; preds = %57, %54, %51
  %63 = load i32 (i32, i32, i32, i32, i32, i32)*, i32 (i32, i32, i32, i32, i32, i32)** getelementptr inbounds (%struct.pci_raw_ops, %struct.pci_raw_ops* @pci_direct_conf1, i32 0, i32 1), align 8
  %64 = load i32, i32* %8, align 4
  %65 = load i32, i32* %9, align 4
  %66 = load i32, i32* %10, align 4
  %67 = load i32, i32* %11, align 4
  %68 = load i32, i32* %12, align 4
  %69 = load i32, i32* %13, align 4
  %70 = call i32 %63(i32 %64, i32 %65, i32 %66, i32 %67, i32 %68, i32 %69)
  store i32 %70, i32* %7, align 4
  br label %71

; <label>:71:                                     ; preds = %62, %61, %50
  %72 = load i32, i32* %7, align 4
  ret i32 %72
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @ce4100_bus1_read(i32, i32, i32, i32*) #0 {
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32*, align 8
  %10 = alloca i64, align 8
  %11 = alloca i32, align 4
  %12 = alloca i64, align 8
  %13 = alloca i64, align 8
  %14 = alloca i32, align 4
  %15 = alloca i64, align 8
  %16 = alloca i64, align 8
  %17 = alloca i32, align 4
  store i32 %0, i32* %6, align 4
  store i32 %1, i32* %7, align 4
  store i32 %2, i32* %8, align 4
  store i32* %3, i32** %9, align 8
  store i32 0, i32* %11, align 4
  br label %18

; <label>:18:                                     ; preds = %72, %4
  %19 = load i32, i32* %11, align 4
  %20 = sext i32 %19 to i64
  %21 = icmp ult i64 %20, 50
  br i1 %21, label %22, label %75

; <label>:22:                                     ; preds = %18
  %23 = load i32, i32* %11, align 4
  %24 = sext i32 %23 to i64
  %25 = getelementptr inbounds [50 x %struct.sim_dev_reg], [50 x %struct.sim_dev_reg]* bitcast (<{ { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } }>* @bus1_fixups to [50 x %struct.sim_dev_reg]*), i64 0, i64 %24
  %26 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %25, i32 0, i32 0
  %27 = load i32, i32* %26, align 8
  %28 = load i32, i32* %6, align 4
  %29 = icmp eq i32 %27, %28
  br i1 %29, label %30, label %71

; <label>:30:                                     ; preds = %22
  %31 = load i32, i32* %11, align 4
  %32 = sext i32 %31 to i64
  %33 = getelementptr inbounds [50 x %struct.sim_dev_reg], [50 x %struct.sim_dev_reg]* bitcast (<{ { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } }>* @bus1_fixups to [50 x %struct.sim_dev_reg]*), i64 0, i64 %32
  %34 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %33, i32 0, i32 1
  %35 = load i32, i32* %34, align 4
  %36 = load i32, i32* %7, align 4
  %37 = and i32 %36, -4
  %38 = icmp eq i32 %35, %37
  br i1 %38, label %39, label %71

; <label>:39:                                     ; preds = %30
  %40 = load i32, i32* %11, align 4
  %41 = sext i32 %40 to i64
  %42 = getelementptr inbounds [50 x %struct.sim_dev_reg], [50 x %struct.sim_dev_reg]* bitcast (<{ { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } }>* @bus1_fixups to [50 x %struct.sim_dev_reg]*), i64 0, i64 %41
  %43 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %42, i32 0, i32 3
  %44 = load void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32*)** %43, align 8
  %45 = icmp ne void (%struct.sim_dev_reg*, i32*)* %44, null
  br i1 %45, label %46, label %71

; <label>:46:                                     ; preds = %39
  br label %47

; <label>:47:                                     ; preds = %46
  %48 = icmp eq i64* %12, %13
  %49 = zext i1 %48 to i32
  store i32 1, i32* %14, align 4
  %50 = load i32, i32* %14, align 4
  %51 = call i64 @_raw_spin_lock_irqsave(%struct.raw_spinlock* @pci_config_lock)
  store i64 %51, i64* %10, align 8
  br label %52

; <label>:52:                                     ; preds = %47
  %53 = load i32, i32* %11, align 4
  %54 = sext i32 %53 to i64
  %55 = getelementptr inbounds [50 x %struct.sim_dev_reg], [50 x %struct.sim_dev_reg]* bitcast (<{ { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } }>* @bus1_fixups to [50 x %struct.sim_dev_reg]*), i64 0, i64 %54
  %56 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %55, i32 0, i32 3
  %57 = load void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32*)** %56, align 8
  %58 = load i32, i32* %11, align 4
  %59 = sext i32 %58 to i64
  %60 = getelementptr inbounds [50 x %struct.sim_dev_reg], [50 x %struct.sim_dev_reg]* bitcast (<{ { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } }>* @bus1_fixups to [50 x %struct.sim_dev_reg]*), i64 0, i64 %59
  %61 = load i32*, i32** %9, align 8
  call void %57(%struct.sim_dev_reg* %60, i32* %61)
  br label %62

; <label>:62:                                     ; preds = %52
  %63 = icmp eq i64* %15, %16
  %64 = zext i1 %63 to i32
  store i32 1, i32* %17, align 4
  %65 = load i32, i32* %17, align 4
  %66 = load i64, i64* %10, align 8
  call void @_raw_spin_unlock_irqrestore(%struct.raw_spinlock* @pci_config_lock, i64 %66)
  br label %67

; <label>:67:                                     ; preds = %62
  %68 = load i32*, i32** %9, align 8
  %69 = load i32, i32* %7, align 4
  %70 = load i32, i32* %8, align 4
  call void @extract_bytes(i32* %68, i32 %69, i32 %70)
  store i32 0, i32* %5, align 4
  br label %76

; <label>:71:                                     ; preds = %39, %30, %22
  br label %72

; <label>:72:                                     ; preds = %71
  %73 = load i32, i32* %11, align 4
  %74 = add nsw i32 %73, 1
  store i32 %74, i32* %11, align 4
  br label %18

; <label>:75:                                     ; preds = %18
  store i32 -1, i32* %5, align 4
  br label %76

; <label>:76:                                     ; preds = %75, %67
  %77 = load i32, i32* %5, align 4
  ret i32 %77
}

declare i64 @_raw_spin_lock_irqsave(%struct.raw_spinlock*) #2 section ".spinlock.text"

declare void @_raw_spin_unlock_irqrestore(%struct.raw_spinlock*, i64) #2 section ".spinlock.text"

; Function Attrs: noinline nounwind optnone uwtable
define internal void @extract_bytes(i32*, i32, i32) #0 {
  %4 = alloca i32*, align 8
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  store i32* %0, i32** %4, align 8
  store i32 %1, i32* %5, align 4
  store i32 %2, i32* %6, align 4
  %8 = load i32, i32* %5, align 4
  %9 = and i32 %8, 3
  %10 = mul nsw i32 %9, 8
  %11 = load i32*, i32** %4, align 8
  %12 = load i32, i32* %11, align 4
  %13 = lshr i32 %12, %10
  store i32 %13, i32* %11, align 4
  %14 = load i32, i32* %6, align 4
  %15 = sub nsw i32 4, %14
  %16 = mul nsw i32 %15, 8
  %17 = lshr i32 -1, %16
  store i32 %17, i32* %7, align 4
  %18 = load i32, i32* %7, align 4
  %19 = load i32*, i32** %4, align 8
  %20 = load i32, i32* %19, align 4
  %21 = and i32 %20, %18
  store i32 %21, i32* %19, align 4
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define internal i32 @ce4100_bus1_write(i32, i32, i32, i32) #0 {
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  %10 = alloca i64, align 8
  %11 = alloca i32, align 4
  %12 = alloca i64, align 8
  %13 = alloca i64, align 8
  %14 = alloca i32, align 4
  %15 = alloca i64, align 8
  %16 = alloca i64, align 8
  %17 = alloca i32, align 4
  store i32 %0, i32* %6, align 4
  store i32 %1, i32* %7, align 4
  store i32 %2, i32* %8, align 4
  store i32 %3, i32* %9, align 4
  store i32 0, i32* %11, align 4
  br label %18

; <label>:18:                                     ; preds = %69, %4
  %19 = load i32, i32* %11, align 4
  %20 = sext i32 %19 to i64
  %21 = icmp ult i64 %20, 50
  br i1 %21, label %22, label %72

; <label>:22:                                     ; preds = %18
  %23 = load i32, i32* %11, align 4
  %24 = sext i32 %23 to i64
  %25 = getelementptr inbounds [50 x %struct.sim_dev_reg], [50 x %struct.sim_dev_reg]* bitcast (<{ { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } }>* @bus1_fixups to [50 x %struct.sim_dev_reg]*), i64 0, i64 %24
  %26 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %25, i32 0, i32 0
  %27 = load i32, i32* %26, align 8
  %28 = load i32, i32* %6, align 4
  %29 = icmp eq i32 %27, %28
  br i1 %29, label %30, label %68

; <label>:30:                                     ; preds = %22
  %31 = load i32, i32* %11, align 4
  %32 = sext i32 %31 to i64
  %33 = getelementptr inbounds [50 x %struct.sim_dev_reg], [50 x %struct.sim_dev_reg]* bitcast (<{ { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } }>* @bus1_fixups to [50 x %struct.sim_dev_reg]*), i64 0, i64 %32
  %34 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %33, i32 0, i32 1
  %35 = load i32, i32* %34, align 4
  %36 = load i32, i32* %7, align 4
  %37 = and i32 %36, -4
  %38 = icmp eq i32 %35, %37
  br i1 %38, label %39, label %68

; <label>:39:                                     ; preds = %30
  %40 = load i32, i32* %11, align 4
  %41 = sext i32 %40 to i64
  %42 = getelementptr inbounds [50 x %struct.sim_dev_reg], [50 x %struct.sim_dev_reg]* bitcast (<{ { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } }>* @bus1_fixups to [50 x %struct.sim_dev_reg]*), i64 0, i64 %41
  %43 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %42, i32 0, i32 4
  %44 = load void (%struct.sim_dev_reg*, i32)*, void (%struct.sim_dev_reg*, i32)** %43, align 8
  %45 = icmp ne void (%struct.sim_dev_reg*, i32)* %44, null
  br i1 %45, label %46, label %68

; <label>:46:                                     ; preds = %39
  br label %47

; <label>:47:                                     ; preds = %46
  %48 = icmp eq i64* %12, %13
  %49 = zext i1 %48 to i32
  store i32 1, i32* %14, align 4
  %50 = load i32, i32* %14, align 4
  %51 = call i64 @_raw_spin_lock_irqsave(%struct.raw_spinlock* @pci_config_lock)
  store i64 %51, i64* %10, align 8
  br label %52

; <label>:52:                                     ; preds = %47
  %53 = load i32, i32* %11, align 4
  %54 = sext i32 %53 to i64
  %55 = getelementptr inbounds [50 x %struct.sim_dev_reg], [50 x %struct.sim_dev_reg]* bitcast (<{ { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } }>* @bus1_fixups to [50 x %struct.sim_dev_reg]*), i64 0, i64 %54
  %56 = getelementptr inbounds %struct.sim_dev_reg, %struct.sim_dev_reg* %55, i32 0, i32 4
  %57 = load void (%struct.sim_dev_reg*, i32)*, void (%struct.sim_dev_reg*, i32)** %56, align 8
  %58 = load i32, i32* %11, align 4
  %59 = sext i32 %58 to i64
  %60 = getelementptr inbounds [50 x %struct.sim_dev_reg], [50 x %struct.sim_dev_reg]* bitcast (<{ { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg }, { i32, i32, void (%struct.sim_dev_reg*)*, void (%struct.sim_dev_reg*, i32*)*, void (%struct.sim_dev_reg*, i32)*, %struct.sim_reg } }>* @bus1_fixups to [50 x %struct.sim_dev_reg]*), i64 0, i64 %59
  %61 = load i32, i32* %9, align 4
  call void %57(%struct.sim_dev_reg* %60, i32 %61)
  br label %62

; <label>:62:                                     ; preds = %52
  %63 = icmp eq i64* %15, %16
  %64 = zext i1 %63 to i32
  store i32 1, i32* %17, align 4
  %65 = load i32, i32* %17, align 4
  %66 = load i64, i64* %10, align 8
  call void @_raw_spin_unlock_irqrestore(%struct.raw_spinlock* @pci_config_lock, i64 %66)
  br label %67

; <label>:67:                                     ; preds = %62
  store i32 0, i32* %5, align 4
  br label %73

; <label>:68:                                     ; preds = %39, %30, %22
  br label %69

; <label>:69:                                     ; preds = %68
  %70 = load i32, i32* %11, align 4
  %71 = add nsw i32 %70, 1
  store i32 %71, i32* %11, align 4
  br label %18

; <label>:72:                                     ; preds = %18
  store i32 -1, i32* %5, align 4
  br label %73

; <label>:73:                                     ; preds = %72, %67
  %74 = load i32, i32* %5, align 4
  ret i32 %74
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { cold noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { cold }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.0 (tags/RELEASE_600/final)"}
!2 = !{i32 -2142617707, i32 -2142617677, i32 -2142617631, i32 -2142617573, i32 -2142617519, i32 -2142617465, i32 -2142617410, i32 -2142617379}
!3 = !{i32 -2142616810, i32 -2142616803, i32 -2142616751, i32 -2142616720, i32 -2142616690}
!4 = !{i32 -2142615291, i32 -2142615261, i32 -2142615215, i32 -2142615157, i32 -2142615103, i32 -2142615049, i32 -2142614994, i32 -2142614963}
!5 = !{i32 -2142614394, i32 -2142614387, i32 -2142614335, i32 -2142614304, i32 -2142614274}
