# AI Infrastructure Engineering Roadmap
**18-Month Project-Driven Learning Path**

## Profile & Goals
- **Focus**: Library design & API engineering, compiler knowledge as tools
- **Career Target**: Framework/Library Engineer (PyTorch/JAX-style) + Rust ML tools
- **Algorithm Depth**: Basic grasp for good engineering decisions
- **Time Commitment**: 4 hours/day × 1.5 years ≈ 2000 hours
- **Learning Style**: Project-driven, hands-on building

---

## Timeline Overview

| Phase | Duration | Hours | Focus Area |
|-------|----------|-------|------------|
| **Phase 1:** Engineering Foundations | Weeks 1-8 | 180-200 | C++ APIs, CUDA basics, Build systems |
| **Phase 2:** Algorithm Essentials | Weeks 9-14 | 150-170 | GPU patterns (practical understanding) |
| **Phase 3:** MLIR Fundamentals | Weeks 15-24 | 260-300 | Dialects, passes, lowering pipelines |
| **Phase 4:** Rust Projects | Weeks 25-32 | 220-260 | Rust ML tools & compilers |
| **Phase 5:** Integration & Capstone | Weeks 33-42 | 250-300 | Multi-backend systems, distributed |
| **Ongoing:** Open Source | Weeks 10-42 | 50-100 | Community contributions |
| **TOTAL** | **42 weeks** | **~1200-1400 hrs** | **Complete ML infrastructure** |

---

## Phase 1: Engineering Foundations (Weeks 1-8)

**Goal:** Master C++ API design, CUDA basics, and build systems through building a tensor library.

### Week 1: CUDA API Wrapper Library
**Project:** Modern C++ RAII wrappers for CUDA runtime
```cpp
template<typename T> class device_ptr;  // Smart pointer for device memory
class stream;                            // Stream wrapper
class event;                             // Event wrapper
```
**Deliverable:** Header-only library, tests, examples, CMake  
**Skills:** CUDA API, RAII, C++ templates, CMake

---

### Week 2: cuBLAS Benchmark Suite
**Project:** Comprehensive benchmarking of cuBLAS
- Benchmark SGEMM, DGEMM, HGEMM
- Test different algorithms (ALGO0-23)
- Compare batched vs non-batched
- Performance graphs and analysis

**Deliverable:** Benchmark suite with visualizations  
**Skills:** cuBLAS API, benchmarking, performance analysis

---

### Week 3: Tensor Library - Basic API Design
**Project:** Simple tensor library with clean API
```cpp
class Tensor {
    // Construction, factory methods
    static Tensor zeros(shape, device);
    // Operations: +, -, *, matmul, transpose
    Tensor to(Device device);  // Device transfer
};
```
**Deliverable:** Working tensor class, CPU + CUDA support, tests  
**Skills:** API design, operator overloading, resource management

---

### Week 4: Multi-Backend Abstraction
**Project:** Backend abstraction layer
```cpp
class Backend {
    virtual void* allocate(size_t) = 0;
    virtual void matmul(...) = 0;
};
class CPUBackend : public Backend { /* Eigen */ };
class CUDABackend : public Backend { /* cuBLAS */ };
```
**Deliverable:** Refactored tensor library with pluggable backends  
**Skills:** Abstraction patterns, polymorphism, design patterns

---

### Week 5: Operator Registration System
**Project:** PyTorch-style operator registration
```cpp
OperatorRegistry::register_kernel("relu", Device::CUDA, relu_cuda);
ops::dispatch("relu", {x});  // Auto-selects right kernel
```
**Deliverable:** Registry system, 5 ops (CPU + CUDA), automatic dispatch  
**Skills:** Registration patterns, static initialization, dispatch

---

### Week 6: Build System & Packaging
**Project:** Professional CMake setup
- Proper install/export targets
- Separate: library, tests, benchmarks, examples
- CI/CD with GitHub Actions

**Deliverable:** Production-quality build system  
**Skills:** CMake, project structure, CI/CD

---

### Week 7: Python API (Optional)
**Project:** Python bindings with pybind11
```python
import mytensor as mt
x = mt.Tensor.randn([10, 10], device='cuda')
y = x @ x  # Uses your library
```
**Deliverable:** Python bindings, pip installable  
**Skills:** Python/C++ bindings (optional - skip if not interested)

---

### Week 8: Documentation & Polish
**Project:** Professional documentation
- API reference (Doxygen)
- User guide and tutorials
- Architecture documentation
- Published docs (GitHub Pages)

**Deliverable:** Complete, polished library ready for portfolio  
**Skills:** Technical writing, documentation tools

---

## Phase 2: Algorithm Essentials (Weeks 9-14)

**Goal:** Understand GPU algorithms well enough for engineering decisions. NOT to become a kernel expert.

### Week 9: Understanding GEMM
**Project:** 3 GEMM implementations to understand trade-offs
1. Naive (global memory)
2. Tiled (shared memory)
3. Optimized (vectorized loads, register blocking)

**Deliverable:** 3 implementations, benchmarks vs cuBLAS, memory analysis  
**Learning:** Memory hierarchy, tiling, shared memory  
**NOT aiming to beat cuBLAS - just understand concepts**

---

### Week 10: Reduction & Scan Patterns
**Project:** Parallel primitives
- Reduction (sum, max, min): naive → shared → warp shuffle
- Scan (prefix sum): Hillis-Steele, work-efficient

**Deliverable:** Working patterns, benchmark vs Thrust/CUB  
**Learning:** Parallel patterns, warp primitives

---

### Week 11: Fused Kernels
**Project:** Understand fusion benefits
```cuda
// Separate: matmul → bias_add → relu (3 kernels, 3 memory passes)
// Fused: matmul_bias_relu (1 kernel, 1 memory pass)
```
**Deliverable:** 3-4 fused kernels, bandwidth analysis, benchmarks  
**Learning:** Kernel fusion, memory bandwidth optimization

---

### Week 12: CUTLASS Study (Surface Level)
**Project:** Use CUTLASS as a library (not implement it)
- Integrate CUTLASS into your tensor library
- Support FP32, FP16, INT8
- Configuration guide: which config for which problem

**Deliverable:** CUTLASS integration, benchmarks, usage guide  
**Learning:** Template libraries, CUTLASS API (as power user)

---

### Week 13: Attention Mechanism
**Project:** Understand attention algorithms
- Naive attention (slow)
- Read Flash Attention paper
- Simplified tiled version

**Deliverable:** Working attention, Flash Attention understanding, memory analysis  
**Learning:** ML-specific algorithms, IO-aware optimization

---

### Week 14: Algorithm Knowledge Synthesis
**Project:** Design guide for library developers
- "GPU Algorithm Patterns for Library Developers"
- When to tile, fuse, optimize
- Decision trees for operator implementations

**Deliverable:** Technical document (20-30 pages), blog series  
**Learning:** Synthesis, engineering judgment

---

## Phase 3: MLIR Fundamentals (Weeks 15-24)

**Goal:** Learn MLIR through building custom dialects and compiler passes.

### Weeks 15-16: MLIR Toy Tutorial
**Project:** Complete all 7 chapters of official Toy tutorial
- Ch 1-3: Lexer, Parser, AST → MLIR, high-level optimization
- Ch 4-7: Interfaces, lowering, LLVM codegen, JIT

**Deliverable:** Working Toy compiler (end-to-end)  
**Learning:** MLIR basics, IR construction, lowering pipeline, JIT

---

### Week 17: Extend Toy Language
**Project:** Add 5 new operations to Toy
```tablegen
def TransposeOp, ReduceSumOp, ReshapeOp, BroadcastOp, ConcatOp
```
**Deliverable:** Extended Toy with new ops, optimization patterns, lowering  
**Learning:** Op definition, TableGen, pattern matching

---

### Week 18: Custom Dialect for Tensors
**Project:** Create new dialect for ML tensor operations
```tablegen
def MatMulOp, ConvOp, PoolOp, ReluOp, LayerNormOp, ...
// 10-15 ML operations
```
**Deliverable:** Custom dialect, shape verifiers, pretty printing, tests  
**Learning:** Dialect creation, ML domain modeling

---

### Week 19: Optimization Passes
**Project:** Transformation passes for your dialect
```cpp
// Pattern rewriting:
// - Fuse MatMul + Add → MatMulAdd
// - Fold constants
// - Eliminate redundant ops
// - Strength reduction
```
**Deliverable:** 5-7 optimization patterns, tests showing IR transformations  
**Learning:** Pattern matching, graph rewriting

---

### Week 20: Lowering to Linalg
**Project:** Convert your dialect to Linalg dialect
```cpp
struct MatMulOpLowering : public OpConversionPattern<MatMulOp> {
    // Your MatMul → linalg.matmul
};
```
**Deliverable:** Conversion patterns for all ops, working lowering pass  
**Learning:** Dialect conversion, Linalg dialect

---

### Week 21: Linalg to GPU
**Project:** GPU code generation path
- Tile Linalg operations
- Lower: Linalg → GPU → NVVM → PTX
- Generate CUDA kernels

**Deliverable:** Complete GPU lowering, generated kernels, performance comparison  
**Learning:** Linalg transformations, GPU dialect, code generation

---

### Week 22: Integration with Tensor Library
**Project:** Connect MLIR compiler to C++ tensor library
```cpp
class MLIRCompiler {
    void build_module(const ComputationGraph& graph);
    void optimize();
    void execute(inputs, outputs);  // JIT
};
```
**Deliverable:** MLIR compiler integrated, JIT execution, benchmarks  
**Learning:** Integration, JIT, end-to-end systems

---

### Week 23: Python Frontend (Optional)
**Project:** Python DSL that compiles via MLIR
```python
@mlir.compile
def my_model(x, y):
    return relu(x @ y + 1)
```
**Deliverable:** Python decorator, tracing, MLIR backend integration  
**Learning:** Frontend design, tracing, DSLs

---

### Week 24: MLIR Project Documentation
**Project:** Document your MLIR compiler
- Architecture, dialect reference, pass descriptions
- Lowering pipeline diagrams, user guide, examples

**Deliverable:** Complete documentation, blog post  
**Learning:** Technical writing

---

## Phase 4: Rust Projects (Weeks 25-32)

**Goal:** Leverage Rust for ML tools, diversify portfolio with systems programming.

### Week 25: Rust CUDA Wrapper
**Project:** Safe Rust API for CUDA
```rust
pub struct DeviceBuffer<T> {
    ptr: NonNull<T>,
    len: usize,
}
// RAII, type safety, Send/Sync traits
```
**Deliverable:** Safe CUDA wrapper, published crate  
**Learning:** Rust FFI, unsafe code, ownership-based resource management

---

### Week 26: Rust Tensor Library
**Project:** Tensor library in pure Rust
```rust
pub struct Tensor {
    shape: Vec<usize>,
    data: TensorData,  // CPU or CUDA
    device: Device,
}
```
**Deliverable:** Working tensor library, CPU + CUDA backends  
**Learning:** Rust traits, generics, advanced patterns

---

### Week 27: Computation Graph & Autograd
**Project:** Automatic differentiation in Rust
```rust
pub struct Variable {
    data: Tensor,
    grad: Option<Tensor>,
    backward_fn: Option<Rc<dyn BackwardFunction>>,
}
```
**Deliverable:** Autograd system, train a small network, gradient tests  
**Learning:** Computation graphs, autograd, Rc/RefCell patterns

---

### Week 28: LLVM Bindings in Rust
**Project:** Simple JIT compiler using Inkwell
```rust
pub struct SimpleJIT {
    // Expression → LLVM IR → JIT execution
}
```
**Deliverable:** Expression compiler, JIT execution, benchmarks  
**Learning:** LLVM in Rust, code generation

---

### Weeks 29-30: Rust ML Compiler (2 weeks)
**Project:** End-to-end compiler in Rust
```rust
// Pipeline:
// JSON graph → HIR → optimization passes → MIR → LLVM/CUDA
```
**Deliverable:** Complete ML compiler, 10-15 ops, CUDA codegen  
**Learning:** Compiler design in Rust, end-to-end system

---

### Week 31: Model Serving Backend
**Project:** HTTP server for model inference
```rust
// Actix-web server
async fn inference(req: InferenceRequest) -> InferenceResponse {
    // Load model, run inference on GPU, return results
}
```
**Deliverable:** Inference server, benchmarks, Docker deployment  
**Learning:** Web services in Rust, production systems

---

### Week 32: Rust Projects Documentation
**Project:** Polish and publish all Rust projects
- Publish crates to crates.io
- rustdoc documentation
- Examples and tutorials
- Blog posts

**Deliverable:** 3-4 published crates, complete docs, portfolio showcase  

---

## Phase 5: Integration & Capstone (Weeks 33-42)

**Goal:** Build production-scale systems integrating everything learned.

### Weeks 33-36: Multi-Backend Execution Engine
**Project:** ONNX Runtime-style framework
```cpp
class ExecutionProvider {
    virtual Status Execute(Graph, Inputs, Outputs) = 0;
};
// Providers: CPU, CUDA, MLIR compiler, TensorRT
class ExecutionEngine {
    // Graph partitioning, backend selection, optimization
};
```
**Deliverable:** Multi-backend engine, ONNX import, benchmarks on real models  
**Learning:** System integration, production quality  
**Time:** 100-120 hours (4 weeks)

---

### Weeks 37-40: Distributed Training Framework
**Project:** Simple distributed training system
```cpp
class DistributedTrainer {
    // Data parallelism with NCCL
    // Gradient synchronization
    // Simple model parallelism
};
```
**Deliverable:** Distributed training, NCCL integration, multi-GPU benchmarks  
**Learning:** Distributed systems, NCCL, multi-GPU programming  
**Time:** 100-120 hours (4 weeks)

---

### Weeks 41-42: Final Polish & Presentation
**Project:** Portfolio preparation
- Polish all projects
- Create portfolio website
- Write comprehensive blog series (10+ posts)
- Record demo videos
- Prepare technical presentations

**Deliverable:** Professional portfolio, blog series, demos  
**Time:** 50-60 hours (2 weeks)

---

## Ongoing: Open Source Contributions

**Starting Week 10:** Allocate 4-6 hours per week

### Weeks 10-15: Documentation
- Contribute docs to PyTorch, TVM, MLIR
- Write tutorials and examples

### Weeks 16-25: Bug Fixes
- Find "good first issue" tags
- Fix bugs, add small features
- Get familiar with contribution process

### Weeks 26-42: Substantial Contributions
- New features in PyTorch/TVM
- Operator implementations
- Performance improvements
- Integration with your projects

**Target:** 50-100 hours total (2-3 hrs/week for 30 weeks)

---

## Portfolio Outcomes

### By Week 24 (6 months):
✅ Tensor library with multi-backend  
✅ MLIR compiler with custom dialect  
✅ 5-10 blog posts  
✅ Open source contributions started  
**→ Ready to apply for internships**

### By Week 42 (10 months):
✅ Complete multi-backend execution engine  
✅ Distributed training framework  
✅ 3-4 Rust projects  
✅ 20+ blog posts  
✅ Substantial open source contributions  
**→ Ready for full-time ML systems roles**

---

## Weekly Schedule Template

### Weekdays (Mon-Fri): 4 hours/day
- **Hour 1-2:** Main project coding
- **Hour 3:** Testing / documentation
- **Hour 4:** Open source or reading

### Weekends: 4 hours/day
- **Saturday:** Finish weekly project, polish
- **Sunday:** Write blog post, plan next week

---

## Success Metrics

### Code Quality
- Modern C++17/20, idiomatic Rust
- Comprehensive tests (>80% coverage)
- CI/CD on all projects
- Professional documentation

### Portfolio
- 5-6 major C++ projects
- 3-4 Rust projects
- 20-30 blog posts
- Active GitHub profile

### Open Source
- 10-20 merged PRs (any size)
- Active in communities
- Known contributor in 2-3 projects

### Skills Acquired
- **API Design:** Multi-backend abstractions, clean C++ APIs
- **CUDA:** Practical understanding, can write optimized kernels when needed
- **MLIR:** Dialects, passes, lowering pipelines
- **Rust:** Systems programming, ML tools
- **Systems:** Multi-backend engines, distributed training
- **Engineering:** Build systems, testing, documentation, deployment

---

## Key Resources

### Documentation
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [MLIR Documentation](https://mlir.llvm.org/)
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)

### Tutorials
- [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) ⭐⭐⭐
- [LLVM Kaleidoscope](https://llvm.org/docs/tutorial/)
- [PyTorch Internals Blog](http://blog.ezyang.com/2019/05/pytorch-internals/)

### Books
- *API Design for C++* by Martin Reddy ⭐⭐⭐
- *Programming Massively Parallel Processors* (Kirk & Hwu) ⭐⭐⭐
- *Engineering a Compiler* (Cooper & Torczon) ⭐⭐

### Communities
- [LLVM Discourse](https://discourse.llvm.org/) - MLIR category
- [PyTorch Forums](https://discuss.pytorch.org/)
- [TVM Discuss](https://discuss.tvm.apache.org/)

---

## Target Companies & Roles

### NVIDIA
**Role:** GPU Libraries Engineer, CUDA Developer  
**Skills:** CUDA optimization, CUTLASS, library API design  
**Portfolio:** Tensor library, GEMM kernels, CUTLASS integration

### Microsoft
**Role:** ONNX Runtime Engineer, Azure ML Infrastructure  
**Skills:** Multi-backend systems, MLIR, API design  
**Portfolio:** Multi-backend execution engine, MLIR compiler

### Google/DeepMind
**Role:** ML Compiler Engineer, XLA/JAX  
**Skills:** MLIR, compiler optimization, functional APIs  
**Portfolio:** MLIR dialect, compiler passes, JAX-like system

### Meta
**Role:** PyTorch Core Engineer  
**Skills:** PyTorch internals, autograd, operator registration  
**Portfolio:** Tensor library, autograd, operator registry

### Modular, Startups
**Role:** Compiler Engineer, Rust ML Tools  
**Skills:** End-to-end compilers, Rust, novel approaches  
**Portfolio:** Rust compiler, MLIR work, research implementations
