#include <torch/extension.h>
#include <cutlass/gemm/device/gemm.h>
#include <type_traits>

using RowMajor = cutlass::layout::RowMajor;

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
    using ArchTag = cutlass::arch::Sm80; 
#elif (__CUDA_ARCH__ >= 750)
    using ArchTag = cutlass::arch::Sm75;
#elif (__CUDA_ARCH__ >= 700)
    using ArchTag = cutlass::arch::Sm70;
#else
    using ArchTag = cutlass::arch::Sm50;
#endif

template <typename Element>
torch::Tensor fast_matmul(torch::Tensor A, torch::Tensor B) {
    auto m = static_cast<int>(A.size(0));
    auto k = static_cast<int>(A.size(1));
    auto n = static_cast<int>(B.size(1));

    auto torch_type = torch::kInt32;
    if (std::is_same<Element, int64_t>::value) {
        torch_type = torch::kInt64;
    } else if (std::is_same<Element, int16_t>::value) {
        torch_type = torch::kInt16;
    }
    
    auto C = torch::empty({m, n}, A.options().dtype(torch_type));

    using Gemm = cutlass::gemm::device::Gemm<
        Element, RowMajor,               
        Element, RowMajor,               
        Element, RowMajor,               
        Element,                         
        cutlass::arch::OpClassSimt,      
        ArchTag                          
    >;

    typename Gemm::Arguments arguments{
        {m, n, k}, 
        {static_cast<Element*>(A.data_ptr()), k},
        {static_cast<Element*>(B.data_ptr()), n},
        {static_cast<Element*>(C.data_ptr()), n},
        {static_cast<Element*>(C.data_ptr()), n},
        {1, 0} 
    };

    Gemm gemm_op;
    cutlass::Status status = gemm_op(arguments);

    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS Kernel failed on current architecture.");
    }

    return C;
}


torch::Tensor fast_matmul_int64(torch::Tensor A, torch::Tensor B) {
    return fast_matmul<int64_t>(A, B); 
}

torch::Tensor fast_matmul_int32(torch::Tensor A, torch::Tensor B) {
    return fast_matmul<int32_t>(A, B);
}

torch::Tensor fast_matmul_int16(torch::Tensor A, torch::Tensor B) {
    return fast_matmul<int16_t>(A, B);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_matmul_int64", &fast_matmul_int64, "High-perf INT64 MatMul");
    m.def("fast_matmul_int32", &fast_matmul_int32, "High-perf INT32 MatMul");
    m.def("fast_matmul_int16", &fast_matmul_int16, "High-perf INT16 MatMul");
}