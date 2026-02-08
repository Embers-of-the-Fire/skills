# AI Infrastructure Learning Roadmap: API Libraries & Compiler Focus (v2)

**Target Roles**: ML Compiler Engineer, CUDA Libraries Developer, GPU Systems Engineer  
**Companies**: NVIDIA, Microsoft, Google, Meta, AMD, Intel

---

## Phase 1: C++ API Mastery & Foundations (2-3 months)

### C++ API Design - Practical Focus

Since you understand C++ methodology but need API experience, focus on **reading and using** real production APIs:

#### Study Real CUDA APIs
- **CUDA Runtime API**:
  - [CUDA Runtime API Reference](https://docs.nvidia.com/cuda/cuda-runtime-api/)
  - Study: Memory management patterns, stream APIs, event APIs
  - Practice: Write programs using every major API category
  
- **cuBLAS API Deep Dive**:
  - [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
  - Study: Handle creation, descriptor pattern, algorithm selection
  - Note: Different APIs for different precisions (S/D/C/Z/H variants)
  - Practice: Write programs that call 10+ different cuBLAS functions
  
- **cuDNN API Patterns**:
  - [cuDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/)
  - Study: Workspace management, algorithm enumeration, persistence
  - Key pattern: Descriptor objects for configuration
  - Practice: Setup and execute convolution with different algorithms

#### C++ API Design Patterns in Practice
- **Books**:
  - *API Design for C++* by Martin Reddy ⭐⭐⭐ (Read chapters 1-6, 9)
  - *Large-Scale C++ Software Design* by Lakos (Focus on physical design)
  
- **Pattern Study** - Find these in real code:
  - **Handle/Descriptor pattern**: cuBLAS, cuDNN, NCCL
  - **Workspace pattern**: cuDNN convolution
  - **Builder pattern**: TensorRT network definition
  - **RAII wrappers**: Thrust, modern CUDA code
  - **Policy-based design**: Thrust backends
  - **Expression templates**: Eigen, Blaze

#### Modern C++ Templates - Practical Focus

Study template-heavy libraries by reading their code:

- **Thrust** - [NVIDIA/thrust](https://github.com/NVIDIA/thrust)
  - Read: `thrust/system/cuda/detail/*.h` 
  - Understand: Backend abstraction via policies
  - Practice: Write custom device_vector allocators
  
- **CUTLASS** - [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) ⭐⭐⭐
  - Read: `include/cutlass/gemm/` - template hierarchy
  - Understand: Tile iterators, thread block structure
  - Practice: Instantiate GEMM with different configs
  - Tutorial: [CUTLASS Documentation](https://github.com/NVIDIA/CUTLASS/blob/main/media/docs/quickstart.md)

- **Eigen** - [eigen.tuxfamily.org](https://eigen.tuxfamily.org)
  - Read: Core module source
  - Understand: Expression templates, lazy evaluation
  - Practice: Extend with custom operations

**Practical Exercises**:
1. Read CUTLASS GEMM example, modify tile sizes
2. Implement a simple RAII wrapper for CUDA streams
3. Create a policy-based allocator (like Thrust)

### Build Systems - Hands-On

You'll need these constantly:

#### CMake Modern Patterns
- **Resources**:
  - *Professional CMake* by Craig Scott (Chapters 1-15)
  - [CMake CUDA Documentation](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html#cuda)
- **Practice Projects**:
  - Build a library with CUDA + CPU code + tests
  - Use `find_package(CUDAToolkit)`
  - Export/install targets properly
  - Multi-configuration (Debug/Release)
- **Study**: [CUTLASS CMakeLists.txt](https://github.com/NVIDIA/cutlass/blob/main/CMakeLists.txt) - production example

#### LLVM/MLIR Build System
- **Practice**: Build LLVM and MLIR from source
- **Understand**: 
  - TableGen (LLVM's code generation)
  - `add_llvm_library` vs `add_library`
  - Out-of-tree builds
- **Reference**: [Building LLVM](https://llvm.org/docs/CMake.html)

### Computer Architecture for GPU/Compiler

- **Book**: *Computer Architecture: A Quantitative Approach* (Hennessy & Patterson)
  - Focus: Chapter 4 (Data-level parallelism), Chapter 5 (Thread-level)
- **GPU Architecture**:
  - [NVIDIA GPU Architecture Whitepapers](https://www.nvidia.com/en-us/data-center/resources/gpu-architecture/) - Volta, Ampere, Hopper
  - Understand: SM structure, warp scheduler, memory hierarchy
- **ISAs**:
  - PTX: [PTX ISA Guide](https://docs.nvidia.com/cuda/parallel-thread-execution/)
  - x86-64 SIMD: AVX-512 basics
  - RISC-V: Emerging in ML accelerators

### Compiler Theory Foundations

- **Books**:
  - *Engineering a Compiler* by Cooper & Torczon ⭐⭐ (Chapters 1-9)
  - *Compilers: Principles, Techniques, and Tools* (Dragon Book) - Reference
- **Focus Areas**:
  - SSA form (critical for LLVM/MLIR)
  - Data flow analysis
  - Loop optimizations
  - Instruction scheduling
- **Online**: [Cornell CS 6120: Advanced Compilers](https://www.cs.cornell.edu/courses/cs6120/) ⭐⭐⭐

---

## Phase 2: GPU Programming & Library Architecture (3-4 months)

### CUDA Programming - API-Centric Learning

#### Core CUDA APIs
- **Books**:
  - *Programming Massively Parallel Processors* (Kirk & Hwu) ⭐⭐⭐
  - *CUDA C Best Practices Guide* (NVIDIA docs)
  
#### API Categories to Master:
1. **Memory Management API**:
   - `cudaMalloc`, `cudaFree`, `cudaMemcpy`
   - `cudaMallocManaged` (Unified Memory)
   - `cudaMallocAsync` (stream-ordered)
   - Pitch allocations for 2D/3D
   
2. **Stream & Event API**:
   - Stream creation, destruction, synchronization
   - Event recording, timing
   - Stream priorities, callbacks
   
3. **Kernel Launch**:
   - Triple chevron syntax
   - `cudaLaunchKernel` API
   - Dynamic parallelism APIs
   
4. **Cooperative Groups**:
   - [Cooperative Groups Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups)
   - Grid-wide synchronization
   - Warp-level primitives

#### Advanced CUDA Features
- **Graphs API**: 
  - [CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
  - Capture and replay patterns
- **CUDA Toolkit Libraries**:
  - Study API design: cuFFT, cuSPARSE, cuSOLVER
  - Notice common patterns across libraries

### CUTLASS - Deep Study ⭐⭐⭐

This is the best example of modern C++ for GPU computing:

#### Study Plan:
1. **Week 1-2**: Read documentation, understand concepts
   - [CUTLASS Concepts](https://github.com/NVIDIA/CUTLASS/blob/main/media/docs/programming_guidelines.md)
   - Tile, Thread block, Warp, Thread hierarchy
   
2. **Week 3**: Trace through GEMM example
   - `examples/00_basic_gemm/basic_gemm.cu`
   - Follow template instantiation
   - Understand epilogue pattern
   
3. **Week 4**: Modify and experiment
   - Change tile sizes
   - Swap epilogue operations
   - Add custom element-wise operations

#### Key Files to Study:
- `include/cutlass/gemm/device/gemm.h` - Top-level API
- `include/cutlass/gemm/kernel/` - Kernel implementations
- `include/cutlass/epilogue/` - Post-processing patterns
- `include/cutlass/arch/` - Architecture-specific code

### Thrust - API Patterns

- **Repo**: [NVIDIA/thrust](https://github.com/NVIDIA/thrust)
- **Study**: How to design STL-like parallel APIs
  
#### Key Concepts:
- **Backends**: CUDA, OMP, TBB - study the abstraction
- **Execution policies**: How to select backend at runtime
- **Iterators**: Counting, transform, permutation iterators
- **Algorithms**: Parallel versions of STL algorithms

#### Practice:
- Use thrust for non-trivial algorithms
- Read source: `thrust/system/cuda/detail/` - implementation
- Understand: How policies dispatch to backends

### ROCm/HIP - Portability

- **Repo**: [ROCm/HIP](https://github.com/ROCm/HIP)
- **Study**: How to design portable GPU APIs
- **hipify tools**: CUDA → HIP translation
- **Practice**: Port a CUDA program to HIP

---

## Phase 3: Deep Learning Framework Internals (3-4 months)

**Note**: Focus on C++ core, not Python bindings

### PyTorch C++ Core (LibTorch)

#### Study Areas:
- **ATen Tensor Library**: [pytorch/pytorch/aten](https://github.com/pytorch/pytorch/tree/main/aten)
  - `c10/` - Core library (types, allocator, device)
  - `aten/src/ATen/` - Tensor implementation
  - `aten/src/ATen/native/` - Operator implementations
  
#### Key APIs to Understand:
1. **Tensor API**: 
   - Read: `aten/src/ATen/core/TensorBody.h`
   - Understand: Storage, strides, memory layout
   
2. **Dispatcher**:
   - [PyTorch Dispatcher](https://pytorch.org/tutorials/advanced/dispatcher.html) ⭐⭐
   - Operator registration: `TORCH_LIBRARY`, `TORCH_IMPL`
   - Multiple dispatch (CPU, CUDA, MPS, etc.)
   
3. **Autograd C++ API**:
   - Read: `torch/csrc/autograd/`
   - Understand: Function, Variable, Edge

#### Practical Learning:
1. **Build PyTorch from source**: 
   - [Building PyTorch](https://github.com/pytorch/pytorch#from-source)
   - Understand build system
   
2. **Write custom operators in C++**:
   - [Custom Ops Tutorial](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html)
   - Focus on C++ registration, not Python binding
   
3. **Study existing operators**:
   - `aten/src/ATen/native/cuda/`: Read 10+ CUDA kernels
   - Notice patterns: dispatch, kernel launch, error handling

#### Advanced: TorchScript C++ API
- Load and execute TorchScript models in C++
- [TorchScript C++ API](https://pytorch.org/cppdocs/)

### ONNX Runtime Architecture

- **Repo**: [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) ⭐⭐
- **Focus**: Clean separation of concerns

#### Study Areas:
1. **Execution Providers** (EPs):
   - Read: `onnxruntime/core/providers/`
   - Understand: How backends are abstracted
   - Study: CUDA EP, CPU EP, TensorRT EP
   
2. **Graph IR**:
   - Read: `onnxruntime/core/graph/`
   - Understand: Graph representation, optimization
   
3. **Kernel API**:
   - How operators are implemented per EP
   - Registration mechanism

#### Why Study This:
- **Excellent API design**: Clean, professional
- **Multi-backend**: Good abstraction patterns
- **Production quality**: Error handling, resource management

---

## Phase 4: Compiler Technologies - Core Focus (6-8 months) ⭐⭐⭐

This is your primary focus area.

### LLVM - Foundation

#### Learning Path:
1. **Books**:
   - *Getting Started with LLVM Core Libraries* by Lopes & Auler
   - Focus on IR, passes, backends
   
2. **Official Tutorial**:
   - [LLVM Kaleidoscope Tutorial](https://llvm.org/docs/tutorial/) ⭐
   - Complete all chapters
   - Understand: Lexer → Parser → AST → IR → JIT

3. **LLVM IR Deep Dive**:
   - [LLVM Language Reference](https://llvm.org/docs/LangRef.html)
   - Write IR by hand
   - Use `opt` tool to run passes
   - Use `lli` to execute IR

#### Key APIs to Master:

1. **IR Builder API**:
   - `llvm::IRBuilder<>` - construct IR programmatically
   - `llvm::Module`, `llvm::Function`, `llvm::BasicBlock`
   - Practice: Generate LLVM IR for simple programs
   
2. **Pass API**:
   - Read: [Writing an LLVM Pass](https://llvm.org/docs/WritingAnLLVMPass.html)
   - Understand: Analysis passes vs transformation passes
   - Practice: Write optimization passes
   
3. **Backend API** (Advanced):
   - Instruction selection
   - Register allocation
   - Instruction scheduling

#### Practical Projects:
1. Implement strength reduction pass
2. Write a simple language that compiles to LLVM IR
3. Profile LLVM optimization passes on real code

#### Study Real LLVM Passes:
- `llvm/lib/Transforms/Scalar/` - Scalar optimizations
- `llvm/lib/Transforms/Vectorize/` - Loop vectorizer
- `llvm/lib/Analysis/` - Analysis passes

### MLIR - Primary Focus ⭐⭐⭐

MLIR is THE technology for modern ML compilers. Deep expertise here is extremely valuable.

#### Why MLIR?
- Used by: TensorFlow (XLA), PyTorch (torch-mlir), AMD, Google TPU, Modular (Mojo)
- Extensible: Define custom dialects for your domain
- Modern: Better than traditional LLVM for high-level optimizations

#### Core Learning Path:

1. **Official Documentation**:
   - [MLIR Website](https://mlir.llvm.org/) ⭐⭐⭐
   - Read all: Concepts, Dialects, Passes, Interfaces
   
2. **Toy Tutorial** (CRITICAL):
   - [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) ⭐⭐⭐
   - **Complete ALL 7 chapters**
   - Implement every step yourself
   - This teaches: Dialect creation, lowering, optimization
   
3. **Video**: 
   - [MLIR: A Compiler Infrastructure](https://www.youtube.com/watch?v=qzljG6DKgic) by Chris Lattner
   - [MLIR Tutorial at LLVM Dev Meeting](https://www.youtube.com/watch?v=Y4SvqTtOIDk)

#### Key MLIR Concepts:

1. **Operations**:
   - Everything is an operation
   - Operations have: operands, results, attributes, regions
   - Define custom ops with ODS (Operation Definition Specification)
   
2. **Dialects**:
   - Collections of operations
   - Study built-in dialects:
     - `arith`: Arithmetic operations
     - `func`: Functions and calls
     - `scf`: Structured control flow
     - `tensor`/`linalg`: Tensor operations
     - `gpu`: GPU abstraction
     - `llvm`: LLVM IR dialect
   
3. **Types**:
   - MLIR has extensible type system
   - Study: Tensor types, memref, function types
   - Define custom types for your dialect
   
4. **Attributes**:
   - Compile-time constants
   - Used for: Sizes, strides, algorithm choices
   
5. **Regions & Blocks**:
   - Operations can contain regions (e.g., function body)
   - Regions contain blocks
   - Blocks contain operations

#### Pattern Rewriting - CRITICAL SKILL

- **Declarative Rewriting (DRR)**:
  - [Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)
  - Define rewrites in TableGen
  - Practice: Write 10+ rewrite patterns
  
- **C++ Rewriting**:
  - Implement `RewritePattern` class
  - More flexible than DRR
  - Study: `mlir/lib/Dialect/*/Transforms/`

#### Dialect Conversion & Lowering

- **Progressive Lowering**: High-level → Low-level
- **Study**: 
  - `mlir/lib/Conversion/` - built-in conversions
  - Example: `LinalgToLLVM`, `SCFToControlFlow`, `GPUToNVVM`
- **Practice**: Implement conversion for your custom dialect

#### MLIR APIs to Master:

1. **Operation APIs**:
   ```cpp
   mlir::Operation::create()
   mlir::OpBuilder
   mlir::RewriterBase
   ```
   
2. **Dialect Definition**:
   - TableGen (`.td` files)
   - `mlir-tblgen` tool
   - Study: `mlir/include/mlir/Dialect/Arith/IR/ArithOps.td`
   
3. **Pass Infrastructure**:
   ```cpp
   mlir::Pass
   mlir::PassManager
   mlir::createMyPass()
   ```
   
4. **Type/Attribute APIs**:
   - Define in TableGen
   - C++ API for manipulation

#### Key Files to Study:

- `mlir/include/mlir/IR/Operation.h` - Core operation
- `mlir/include/mlir/IR/Builders.h` - IR construction
- `mlir/include/mlir/Transforms/DialectConversion.h` - Lowering framework
- `mlir/lib/Dialect/Linalg/` - Excellent example dialect

#### Build and Experiment:

1. **Build MLIR**: 
   ```bash
   git clone https://github.com/llvm/llvm-project.git
   cd llvm-project
   mkdir build && cd build
   cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD="X86;NVPTX"
   ninja
   ```

2. **Tools**:
   - `mlir-opt`: Run optimization passes
   - `mlir-translate`: Translate MLIR to other formats
   - `mlir-tblgen`: Generate C++ from TableGen

3. **Create Standalone Dialect**:
   - [Standalone Example](https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone)
   - Build out-of-tree MLIR project

#### Advanced MLIR Topics:

1. **Linalg Dialect** ⭐:
   - Structured tensor operations
   - Transformation (tiling, fusion, vectorization)
   - Study: How convolution is represented and optimized
   
2. **GPU Dialect**:
   - Hardware-agnostic GPU operations
   - Lowering: `gpu` → `nvvm`/`rocdl` → PTX/GCN
   
3. **Transform Dialect**:
   - [Transform Dialect](https://mlir.llvm.org/docs/Dialects/Transform/)
   - Scripting transformations
   - Very new, very powerful
   
4. **Bufferization**:
   - Convert tensor ops to memref ops
   - Memory layout decisions

#### Resources:
- **Community**: [LLVM Discourse - MLIR](https://discourse.llvm.org/c/mlir/) ⭐
- **Office Hours**: MLIR community has open office hours
- **Papers**: "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation"

### Triton - Modern GPU Compiler ⭐⭐⭐

Triton is the most accessible way to learn modern GPU compiler design.

#### Why Triton?
- Python DSL → MLIR → PTX/LLVM
- Powers Flash Attention 2, many cutting-edge kernels
- Excellent learning resource for MLIR-based compilation

#### Learning Path:

1. **Documentation**: [Triton Lang](https://triton-lang.org/)
2. **Tutorials**: [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/) ⭐
   - Complete all tutorials
   - Implement additional kernels
   
3. **Repo**: [openai/triton](https://github.com/openai/triton)

#### Study Compiler Pipeline:

1. **Frontend** (`python/triton/language/`):
   - Python AST → Triton IR
   - Type inference
   
2. **Triton IR → MLIR**:
   - `lib/Dialect/Triton/` - Triton MLIR dialect
   - `lib/Dialect/TritonGPU/` - GPU-specific dialect
   
3. **Lowering to LLVM**:
   - `lib/Conversion/TritonGPUToLLVM/`
   - Study how block-level ops → PTX
   
4. **Backend**:
   - LLVM → PTX
   - PTX → SASS (by NVIDIA driver)

#### Key Concepts:

- **Block-level programming**: Abstracts thread details
- **Automatic parallelization**: Compiler decides thread mapping
- **Triton-GPU dialect**: Represents blocked memory layouts
- **MLIR transformations**: Optimize at multiple levels

#### Practical Projects:

1. Implement 5+ kernels in Triton (GEMM, softmax, attention, etc.)
2. Read generated MLIR: `MLIR_ENABLE_DUMP=1 python script.py`
3. Profile and optimize Triton kernels
4. Study compiler passes in source code

### TVM - Comprehensive ML Compiler

- **Repo**: [apache/tvm](https://github.com/apache/tvm) ⭐⭐
- **Documentation**: [TVM Docs](https://tvm.apache.org/docs/)

#### Architecture:

1. **Relay IR**: High-level graph representation
2. **TIR**: Tensor IR (like MLIR linalg)
3. **TE**: Tensor Expression (compute/schedule separation)
4. **AutoTVM**: Auto-tuning framework

#### What to Study:

- **Tensor Expression**: `python/tvm/te/`
- **Schedule Primitives**: Tiling, vectorization, parallelization
- **Auto-scheduling**: `python/tvm/auto_scheduler/`
- **BYOC**: Bring Your Own Codegen framework

#### Why Study TVM:

- Very active community
- Good for understanding auto-tuning
- Different approach than MLIR (complementary learning)

### XLA (Google)

- **Repo**: [tensorflow/tensorflow/compiler/xla](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla)
- **Documentation**: [XLA Reference](https://www.tensorflow.org/xla)

#### Focus Areas:

- **HLO IR**: XLA's intermediate representation
- **Fusion**: Aggressive operator fusion
- **Backends**: CPU, GPU, TPU
- **Service architecture**: XLA as a service

#### What to Study:

- `tensorflow/compiler/xla/service/` - Core compilation
- `tensorflow/compiler/xla/service/gpu/` - GPU backend
- Fusion heuristics and cost models

---

## Phase 5: Specialized Topics (4-6 months)

### GPU Code Generation - Low Level

#### PTX Assembly
- **Guide**: [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- **Practice**: Write PTX by hand
- **Tools**: `cuobjdump`, `nvdisasm`, `ptxas`
- **Study**: How CUDA → PTX → SASS

#### NVVM IR (LLVM for NVIDIA GPUs)
- LLVM IR with NVVM intrinsics
- Study: How LLVM backend generates PTX
- **Code**: `llvm/lib/Target/NVPTX/`

#### GPU Dialect in MLIR → PTX
- **Study**: 
  - `mlir/lib/Dialect/GPU/` - GPU dialect
  - `mlir/lib/Conversion/GPUToNVVM/` - NVVM lowering
  - `mlir/lib/Target/LLVM/NVVM/` - PTX emission

### Advanced MLIR Projects

#### 1. Custom Dialect for Specific Domain
- Example: Graph neural networks, sparse ops, attention
- Full lowering pipeline
- Integration with PyTorch via torch-mlir

#### 2. Contribute to torch-mlir
- **Repo**: [llvm/torch-mlir](https://github.com/llvm/torch-mlir)
- Add missing operator lowerings
- Optimization passes

#### 3. Study IREE
- **Repo**: [openxla/iree](https://github.com/openxla/iree) ⭐
- Comprehensive MLIR-based compiler
- Multi-backend: CPU, GPU, Vulkan
- Study: HAL design, ahead-of-time compilation

### Kernel Optimization - Deep Dive

#### Matrix Multiplication Evolution:
1. Naive CUDA
2. Shared memory tiling
3. Register blocking
4. Warp-level GEMM (WMMA)
5. Tensor cores (MMA)
6. Asynchronous copy (cp.async)
7. Warpgroup MMA (Hopper)

**Study**: How CUTLASS implements all of these

#### Memory Optimization Patterns:
- Coalescing
- Bank conflict avoidance
- Occupancy tuning
- Instruction-level parallelism

**Tools**:
- NVIDIA Nsight Compute
- Roofline model analysis

---

## Rust Integration Opportunities

Since you have Rust background:

### Rust + GPU

1. **rust-cuda** - [Rust-GPU/Rust-CUDA](https://github.com/Rust-GPU/Rust-CUDA)
   - Write CUDA kernels in Rust
   - Early stage but interesting
   
2. **wgpu** - [gfx-rs/wgpu](https://github.com/gfx-rs/wgpu)
   - WebGPU implementation in Rust
   - Cross-platform GPU compute

### Rust + Compilers

1. **Cranelift** - [bytecodealliance/wasmtime/cranelift](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift)
   - Code generator in Rust
   - Used by Wasmtime
   - Great for learning codegen in Rust
   
2. **Rust LLVM Bindings**:
   - [llvm-sys](https://crates.io/crates/llvm-sys)
   - [inkwell](https://crates.io/crates/inkwell)
   - Build LLVM-based tools in Rust

3. **MLIR Bindings**:
   - [mlir-rs](https://github.com/IBM/mlir-rs)
   - Experimental but growing

### Potential Projects:

- CUDA kernel launcher library in Rust (like cudarc)
- MLIR bindings for Rust
- Compiler for DSL → GPU code in Rust
- Auto-tuning framework in Rust

---

## Key GitHub Repositories Ranked

### Tier S - Must Study Deeply (⭐⭐⭐)
1. [llvm/llvm-project](https://github.com/llvm/llvm-project) - LLVM & MLIR
2. [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) - Modern C++ GPU templates
3. [openai/triton](https://github.com/openai/triton) - Modern GPU compiler
4. [pytorch/pytorch](https://github.com/pytorch/pytorch) - Framework internals
5. [apache/tvm](https://github.com/apache/tvm) - ML compiler

### Tier A - Study Thoroughly (⭐⭐)
6. [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) - Clean architecture
7. [openxla/iree](https://github.com/openxla/iree) - MLIR compiler
8. [llvm/torch-mlir](https://github.com/llvm/torch-mlir) - PyTorch to MLIR
9. [NVIDIA/thrust](https://github.com/NVIDIA/thrust) - Parallel algorithms
10. [google/jax](https://github.com/google/jax) - Functional API + XLA

### Tier B - Important References (⭐)
11. [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow) - XLA compiler
12. [ROCm/HIP](https://github.com/ROCm/HIP) - Portability layer
13. [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) - Official examples
14. [onnx/onnx](https://github.com/onnx/onnx) - IR format
15. [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) - Attention kernels

---

## Learning Resources

### Essential Books
1. *API Design for C++* by Martin Reddy ⭐⭐⭐
2. *Programming Massively Parallel Processors* (Kirk & Hwu) ⭐⭐⭐
3. *Engineering a Compiler* (Cooper & Torczon) ⭐⭐⭐
4. *Getting Started with LLVM Core Libraries* (Lopes & Auler) ⭐⭐

### Essential Courses
1. [Cornell CS 6120: Advanced Compilers](https://www.cs.cornell.edu/courses/cs6120/) ⭐⭐⭐
2. [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) ⭐⭐⭐
3. [LLVM Kaleidoscope](https://llvm.org/docs/tutorial/) ⭐⭐
4. [Stanford CS143: Compilers](https://web.stanford.edu/class/cs143/) ⭐⭐

### Essential Papers
1. "MLIR: Scaling Compiler Infrastructure" ⭐⭐⭐
2. "TVM: An Automated End-to-End Optimizing Compiler" ⭐⭐
3. "Triton: An Intermediate Language and Compiler" ⭐⭐
4. "FlashAttention: Fast and Memory-Efficient Exact Attention" ⭐

### Communities
- [LLVM Discourse](https://discourse.llvm.org/) - MLIR category especially
- [PyTorch Forums](https://discuss.pytorch.org/)
- [TVM Discuss](https://discuss.tvm.apache.org/)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

---

## Career Targets

### NVIDIA - CUDA Libraries
**Skills**: CUTLASS, cuBLAS/cuDNN architecture, kernel optimization, C++ templates
**Projects**: GEMM kernels, CUTLASS contributions, Triton work

### Microsoft - Azure ML, ONNX Runtime
**Skills**: ONNX Runtime architecture, multi-backend design, MLIR
**Projects**: ONNX Runtime contributions, execution provider work

### Google/DeepMind - XLA, JAX
**Skills**: XLA compiler, MLIR, functional programming, HLO
**Projects**: MLIR dialects, XLA contributions, JAX backend work

### Meta - PyTorch
**Skills**: PyTorch internals, TorchDynamo, torch-mlir
**Projects**: PyTorch contributions, compiler work, operator implementations

### Modular, Groq, Cerebras, Startups
**Skills**: End-to-end compiler, MLIR, novel architectures
**Projects**: Research implementations, MLIR dialects, performance work

---

This roadmap emphasizes C++ API mastery through real-world usage, deep MLIR expertise, and practical compiler construction. Expect 12-18 months for comprehensive coverage, with MLIR being your core differentiator.