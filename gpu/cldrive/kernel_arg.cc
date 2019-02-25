#include "gpu/cldrive/kernel_arg.h"

#include "gpu/cldrive/array_kernel_arg_value.h"
#include "gpu/cldrive/scalar_kernel_arg_value.h"

#include "phd/status_macros.h"

namespace gpu {
namespace cldrive {

phd::StatusOr<OpenClArgType> OpenClArgTypeFromString(const string& type_name) {
  if (!type_name.compare("bool")) {
    return OpenClArgType::BOOL;
  } else if (!type_name.compare("char")) {
    return OpenClArgType::CHAR;
  } else if (!type_name.compare("unsigned char") ||
             !type_name.compare("uchar")) {
    return OpenClArgType::UCHAR;
  } else if (!type_name.compare("short")) {
    return OpenClArgType::SHORT;
  } else if (!type_name.compare("unsigned short") ||
             !type_name.compare("ushort")) {
    return OpenClArgType::USHORT;
  } else if (!type_name.compare("int")) {
    return OpenClArgType::INT;
  } else if (!type_name.compare("unsigned int") || !type_name.compare("uint")) {
    return OpenClArgType::UINT;
  } else if (!type_name.compare("long")) {
    return OpenClArgType::LONG;
  } else if (!type_name.compare("unsigned long") ||
             !type_name.compare("ulong")) {
    return OpenClArgType::ULONG;
  } else if (!type_name.compare("float")) {
    return OpenClArgType::FLOAT;
  } else if (!type_name.compare("double")) {
    return OpenClArgType::DOUBLE;
  } else if (!type_name.compare("half")) {
    return OpenClArgType::HALF;
  } else {
    return phd::Status::UNKNOWN;
  }
}

KernelArg::KernelArg(cl::Kernel* kernel, size_t arg_index)
    : kernel_(kernel),
      arg_index_(arg_index),
      address_(kernel->getArgInfo<CL_KERNEL_ARG_ADDRESS_QUALIFIER>(arg_index)),
      type_(OpenClArgType::DEFAULT_UNKNOWN) {
  CHECK(IsGlobal() || IsLocal() || IsConstant() || IsPrivate());
}

phd::Status KernelArg::Init() {
  // Address qualifier is one of:
  //   CL_KERNEL_ARG_ACCESS_READ_ONLY
  //   CL_KERNEL_ARG_ACCESS_WRITE_ONLY
  //   CL_KERNEL_ARG_ACCESS_READ_WRITE
  //   CL_KERNEL_ARG_ACCESS_NONE
  //
  // If argument is not an image type, CL_KERNEL_ARG_ACCESS_NONE is returned.
  // If argument is an image type, the access qualifier specified or the
  // default access qualifier is returned.
  auto access_qualifier =
      kernel_->getArgInfo<CL_KERNEL_ARG_ACCESS_QUALIFIER>(arg_index_);
  if (access_qualifier != CL_KERNEL_ARG_ACCESS_NONE) {
    LOG(ERROR) << "Argument " << arg_index_ << " is an image type";
    return phd::Status::UNKNOWN;
  }

  string full_type_name =
      kernel_->getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(arg_index_);
  CHECK(full_type_name.size());

  is_pointer_ = full_type_name[full_type_name.size() - 2] == '*';

  // Strip the trailing '*' on pointer types.
  string type_name = full_type_name;
  if (is_pointer_) {
    type_name = full_type_name.substr(0, full_type_name.size() - 2);
  }

  LOG(INFO) << "FULL TYPE NAME " << full_type_name;
  LOG(INFO) << "TYPE NAME " << type_name;
  ASSIGN_OR_RETURN(type_, OpenClArgTypeFromString(type_name));
  LOG(INFO) << "TYPE " << type_;

  return phd::Status::OK;
}

namespace {

template <typename T>
std::unique_ptr<ArrayKernelArgValueWithBuffer<T>> CreateArrayArgValue(
    size_t size, bool rand_values, const cl::Context& context) {
  auto arg_value =
      std::make_unique<ArrayKernelArgValueWithBuffer<T>>(context, size, 1);
  if (rand_values) {
    for (size_t i = 0; i < size; ++i) {
      arg_value->vector()[i] = rand();
    }
  }
  return arg_value;
}

std::unique_ptr<KernelArgValue> CreateArrayArgValue(
    const OpenClArgType& type, size_t size, bool rand_values,
    const cl::Context& context) {
  switch (type) {
    // TODO(cec): Fix compilation when bool enabled:
    // case OpenClArgType::BOOL: {
    //   return CreateArrayArgValue<bool>(size, rand_values, context);
    // }
    case OpenClArgType::CHAR: {
      return CreateArrayArgValue<cl_char>(size, rand_values, context);
    }
    case OpenClArgType::UCHAR: {
      return CreateArrayArgValue<cl_uchar>(size, rand_values, context);
    }
    case OpenClArgType::SHORT: {
      return CreateArrayArgValue<cl_short>(size, rand_values, context);
    }
    case OpenClArgType::USHORT: {
      return CreateArrayArgValue<cl_ushort>(size, rand_values, context);
    }
    case OpenClArgType::INT: {
      return CreateArrayArgValue<cl_int>(size, rand_values, context);
    }
    case OpenClArgType::UINT: {
      return CreateArrayArgValue<cl_uint>(size, rand_values, context);
    }
    case OpenClArgType::LONG: {
      return CreateArrayArgValue<cl_long>(size, rand_values, context);
    }
    case OpenClArgType::ULONG: {
      return CreateArrayArgValue<cl_ulong>(size, rand_values, context);
    }
    case OpenClArgType::FLOAT: {
      return CreateArrayArgValue<cl_float>(size, rand_values, context);
    }
    case OpenClArgType::DOUBLE: {
      return CreateArrayArgValue<cl_double>(size, rand_values, context);
    }
    case OpenClArgType::HALF: {
      return CreateArrayArgValue<cl_half>(size, rand_values, context);
    }
    default: {
      // This condition should never occur as KernelArg::Init() will return an
      // error status.
      LOG(FATAL) << "Unsupported OpenClArgType enum type: " << type;
      return std::unique_ptr<KernelArgValue>(nullptr);
    }
  }
}

template <typename T, typename Y>
std::unique_ptr<KernelArgValue> CreateScalarArgValue(const Y& value) {
  return std::make_unique<ScalarKernelArgValue<T>>(static_cast<T>(value));
}

std::unique_ptr<KernelArgValue> CreateScalarArgValue(const OpenClArgType& type,
                                                     const size_t value) {
  switch (type) {
    case OpenClArgType::BOOL: {
      return CreateScalarArgValue<bool>(value);
    }
    case OpenClArgType::CHAR: {
      return CreateScalarArgValue<cl_char>(value);
    }
    case OpenClArgType::UCHAR: {
      return CreateScalarArgValue<cl_uchar>(value);
    }
    case OpenClArgType::SHORT: {
      return CreateScalarArgValue<cl_short>(value);
    }
    case OpenClArgType::USHORT: {
      return CreateScalarArgValue<cl_ushort>(value);
    }
    case OpenClArgType::INT: {
      return CreateScalarArgValue<cl_int>(value);
    }
    case OpenClArgType::UINT: {
      return CreateScalarArgValue<cl_uint>(value);
    }
    case OpenClArgType::LONG: {
      return CreateScalarArgValue<cl_long>(value);
    }
    case OpenClArgType::ULONG: {
      return CreateScalarArgValue<cl_ulong>(value);
    }
    case OpenClArgType::FLOAT: {
      return CreateScalarArgValue<cl_float>(value);
    }
    case OpenClArgType::DOUBLE: {
      return CreateScalarArgValue<cl_double>(value);
    }
    case OpenClArgType::HALF: {
      return CreateScalarArgValue<cl_half>(value);
    }
    default: {
      // This condition should never occur as KernelArg::Init() will return an
      // error status.
      LOG(FATAL) << "Unsupported OpenClArgType enum type: " << type;
      return std::unique_ptr<KernelArgValue>(nullptr);
    }
  }
}

}  // anonymous namespace

std::unique_ptr<KernelArgValue> KernelArg::TryToCreateKernelArgValue(
    const cl::Context& context, const DynamicParams& dynamic_params,
    bool rand_values) {
  CHECK(type_ != OpenClArgType::DEFAULT_UNKNOWN) << "Init() not called";
  if (IsPointer() && IsGlobal()) {
    return CreateArrayArgValue(type_, dynamic_params.global_size_x(),
                               /*rand_values=*/true, context);
  } else if (!IsPointer()) {
    return CreateScalarArgValue(type_, dynamic_params.global_size_x());
  } else {
    return std::unique_ptr<KernelArgValue>(nullptr);
  }
}

std::unique_ptr<KernelArgValue> KernelArg::TryToCreateRandomValue(
    const cl::Context& context, const DynamicParams& dynamic_params) {
  return TryToCreateKernelArgValue(context, dynamic_params,
                                   /*rand_values=*/true);
}

std::unique_ptr<KernelArgValue> KernelArg::TryToCreateOnesValue(
    const cl::Context& context, const DynamicParams& dynamic_params) {
  return TryToCreateKernelArgValue(context, dynamic_params,
                                   /*rand_values=*/false);
}

bool KernelArg::IsGlobal() const {
  return address_ == CL_KERNEL_ARG_ADDRESS_GLOBAL;
}

bool KernelArg::IsLocal() const {
  return address_ == CL_KERNEL_ARG_ADDRESS_LOCAL;
}

bool KernelArg::IsConstant() const {
  return address_ == CL_KERNEL_ARG_ADDRESS_CONSTANT;
}

bool KernelArg::IsPrivate() const {
  return address_ == CL_KERNEL_ARG_ADDRESS_PRIVATE;
}

bool KernelArg::IsPointer() const { return is_pointer_; }

}  // namespace cldrive
}  // namespace gpu
