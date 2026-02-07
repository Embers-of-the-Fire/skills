# AI Infrastructure Learning Roadmap: API Libraries & Compiler Focus

**Target Roles**: ML Compiler Engineer, CUDA Libraries Developer, Deep Learning Framework Engineer  
**Companies**: NVIDIA, Microsoft, Google, Meta, AMD, Intel

---

## Phase 1: Foundations (2-3 months)

### Programming Languages - Deep Dive

#### C++ Modern Features
- **C++17/20/23**: Templates, metaprogramming, concepts, coroutines
- **Books**:
  - *Effective Modern C++* by Scott Meyers
  - *C++ Templates: The Complete Guide* by Vandevoorde & Josuttis
  - *API Design for C++* by Martin Reddy ⭐
- **Practice**: 
  - [Modern C++ Tutorial](https://github.com/changkun/modern-cpp-tutorial)
  - [C++ Core Guidelines](https://github.com/isocpp/CppCoreGuidelines)

#### Python C Extensions & Bindings
- **pybind11**: Modern C++/Python bindings
  - [pybind11 documentation](https://pybind11.readthedocs.io/)
  - [pybind11 GitHub](https://github.com/pybind/pybind11)
- **Python C API**: Understanding CPython internals
- **cffi**: Alternative binding approach

#### Build Systems & Tooling
- **CMake**: Modern CMake (3.x+)
  - *Professional CMake: A Practical Guide* by Craig Scott
  - [CMake Examples](https://github.com/ttroy50/cmake-examples)
- **Bazel**: Used by TensorFlow, JAX
  - [Bazel Tutorial](https://bazel.build/start)
- **Ninja**: Fast build system
- **vcpkg/Conan**: C++ package management

### Computer Architecture for Compilers
- **Books**:
  - *Computer Architecture: A Quantitative Approach* (Hennessy & Patterson)
  - Focus on: Instruction pipelining, SIMD, memory hierarchy
- **ISAs**: x86-64, ARM, PTX (NVIDIA's virtual ISA)

### Compiler Theory Basics
- **Books**:
  - *Compilers: Principles, Techniques, and Tools* (Dragon Book) - Chapters 1-8
  - *Engineering a Compiler* by Cooper & Torczon ⭐
- **Online**:
  - [Stanford CS143: Compilers](https://web.stanford.edu/class/cs143/)
  - [Crafting Interpreters](https://craftinginterpreters.com/) - Free online book
- **Concepts**: Lexing, parsing, AST, IR, optimization passes, code generation

---

## Phase 2: GPU Programming & CUDA Libraries (3-4 months)

### CUDA Programming
- **Books**:
  - *Programming Massively Parallel Processors* (Kirk & Hwu) - Chapters 1-12 ⭐⭐⭐
  - *CUDA C Best Practices Guide* - NVIDIA official docs
- **Online Courses**:
  - [NVIDIA CUDA Training Series](https://www.nvidia.com/en-us/training/)
  - UIUC ECE408/CS483 (available on YouTube)

### CUDA Libraries Architecture
Study the design and implementation of:

#### cuBLAS (Linear Algebra)
- **Study**: How APIs are designed for different data types, memory layouts
- **Repo**: [CUDA Samples](https://github.com/NVIDIA/cuda-samples) - cuBLAS examples
- **Learn**: API versioning, backward compatibility, handle management

#### cuDNN (Deep Learning Primitives)
- **Documentation**: [cuDNN Developer Guide](https://docs.nvidia.com/deeplearning/cudnn/)
- **Focus**: Descriptor pattern, workspace management, algorithm selection
- **Study**: How convolution algorithms are exposed through API

#### Thrust (C++ Parallel Algorithms)
- **Repo**: [NVIDIA/thrust](https://github.com/NVIDIA/thrust) ⭐
- **Study**: STL-like API design, backend abstraction (CUDA/OpenMP/TBB)
- **Learn**: Template metaprogramming, iterator design patterns
- **Book**: *Thrust Quick Start Guide* (NVIDIA docs)

#### CUTLASS (CUDA Templates for Linear Algebra)
- **Repo**: [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) ⭐⭐⭐
- **Study**: Heavy template usage, tile iterators, epilogue patterns
- **Focus**: Modern C++ for high-performance kernels
- **Documentation**: Excellent Doxygen docs in repo

### ROCm/HIP (AMD Alternative)
- **Repo**: [ROCm/HIP](https://github.com/ROCm/HIP)
- **Study**: Portability layer design, how to support multiple backends
- **Documentation**: [ROCm Documentation](https://rocm.docs.amd.com/)

---

## Phase 3: Deep Learning Framework Internals (3-4 months)

### PyTorch Deep Dive

#### Core Architecture
- **Repo**: [pytorch/pytorch](https://github.com/pytorch/pytorch) ⭐⭐⭐
- **Study Areas**:
  - `aten/` - ATen tensor library (C++)
  - `torch/csrc/` - Python bindings
  - `c10/` - Core library (dispatcher, DeviceType)
  - `aten/src/ATen/native/cuda/` - CUDA kernels

#### Custom Extensions
- **Tutorial**: [PyTorch Custom C++/CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html) ⭐
- **Practice**: Write custom operators with both CPU and CUDA backends
- **Study**: `torch.utils.cpp_extension`, JIT compilation

#### Dispatcher & Registration
- **Learn**: PyTorch's dispatcher mechanism for multi-backend support
- **Read**: [PyTorch Dispatcher Documentation](https://pytorch.org/tutorials/advanced/dispatcher.html)
- **Study**: Operator registration macros, boxing/unboxing

#### Resources
- **Blog**: [PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/) by Edward Yang ⭐
- **Video**: [PyTorch Internals Talk](https://www.youtube.com/watch?v=W2WFrPhVEFY)

### TensorFlow/JAX (Optional but Recommended)
- **Repo**: [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)
- **Focus**: XLA compiler integration (covered in Phase 4)
- **Repo**: [google/jax](https://github.com/google/jax)
- **Study**: Functional API design, jit compilation with XLA

### ONNX Runtime
- **Repo**: [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) ⭐
- **Study**: Execution provider abstraction, graph optimization
- **Focus**: Clean separation between graph IR and execution

---

## Phase 4: Compiler Technologies (4-6 months) ⭐⭐⭐

This is your core focus area.

### LLVM Fundamentals

#### Theory & Practice
- **Books**:
  - *Getting Started with LLVM Core Libraries* by Lopes & Auler
  - *LLVM Essentials* by Sarda & Pandey
- **Online**:
  - [LLVM Tutorial](https://llvm.org/docs/tutorial/) - Build a language (Kaleidoscope)
  - [LLVM Language Reference](https://llvm.org/docs/LangRef.html)

#### Core Concepts
- **LLVM IR**: SSA form, basic blocks, instructions
- **Passes**: Analysis passes, transformation passes, pass manager
- **Targets**: Code generation, instruction selection, register allocation
- **Repo**: [llvm/llvm-project](https://github.com/llvm/llvm-project)

#### Practice Projects
- Write LLVM optimization passes
- Build a simple language that compiles to LLVM IR
- Study existing passes in `llvm/lib/Transforms/`

#### Resources
- **Course**: [UIUC CS 526: Advanced Compiler Construction](https://courses.engr.illinois.edu/cs526/)
- **Blog**: [LLVM Blog](https://blog.llvm.org/)

### MLIR (Multi-Level Intermediate Representation)

#### Why MLIR?
Essential for modern ML compilers - used by TensorFlow, PyTorch, AMD, Google

#### Learning Path
- **Documentation**: [MLIR Official Docs](https://mlir.llvm.org/) ⭐
- **Tutorial**: [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) - Build a language
- **Repo**: [llvm/llvm-project/mlir](https://github.com/llvm/llvm-project/tree/main/mlir)

#### Core Concepts
- **Dialects**: Extensible operation sets (tensor, linalg, scf, gpu, etc.)
- **Patterns**: Rewrite patterns, dialect conversion
- **Passes**: Multi-level lowering (high-level → low-level)
- **Traits & Interfaces**: Operation properties

#### Key Dialects to Study
- **Tensor/Linalg**: High-level operations
- **SCF**: Structured control flow
- **GPU**: GPU kernel abstraction
- **LLVM**: LLVM IR dialect
- **Arith/Math**: Arithmetic operations

#### Resources
- **Video**: [MLIR: A Compiler Infrastructure for the End of Moore's Law](https://www.youtube.com/watch?v=qzljG6DKgic) ⭐
- **Paper**: "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation"
- **Community**: [MLIR Discourse](https://discourse.llvm.org/c/mlir/)

### TVM (Tensor Virtual Machine)

#### Overview
Open-source ML compiler stack - very active, great for learning

#### Learning Path
- **Repo**: [apache/tvm](https://github.com/apache/tvm) ⭐⭐
- **Documentation**: [TVM Documentation](https://tvm.apache.org/docs/)
- **Tutorial**: [Get Started with TVM](https://tvm.apache.org/docs/tutorial/introduction.html)

#### Core Concepts
- **Relay IR**: High-level graph IR
- **TIR**: Tensor-level IR
- **TE (Tensor Expression)**: Compute/schedule separation
- **AutoTVM/AutoScheduler**: Auto-tuning
- **BYOC**: Bring Your Own Codegen

#### Projects
- Write custom optimization passes
- Implement operator scheduling for new hardware
- Contribute to TVM's BYOC framework

#### Resources
- **Paper**: "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning"
- **Discuss**: [TVM Forum](https://discuss.tvm.apache.org/)

### XLA (Accelerated Linear Algebra)

#### Overview
Google's domain-specific compiler for linear algebra - used by JAX and TensorFlow

#### Learning Path
- **Repo**: [tensorflow/tensorflow/compiler/xla](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla)
- **Documentation**: [XLA Documentation](https://www.tensorflow.org/xla) ⭐
- **Tutorial**: [XLA Custom Calls](https://www.tensorflow.org/xla/custom_call)

#### Core Concepts
- **HLO (High-Level Optimizer)**: XLA's IR
- **Fusion**: Operator fusion strategies
- **Backends**: CPU, GPU, TPU code generation
- **Custom Calls**: Interface to custom kernels

#### Resources
- **Video**: [XLA Overview](https://www.youtube.com/watch?v=kAOanJczHA0)
- **Blog**: TensorFlow blog posts about XLA

### Triton (OpenAI)

#### Overview
Python-like language for writing GPU kernels - modern alternative to CUDA

#### Learning Path
- **Repo**: [openai/triton](https://github.com/openai/triton) ⭐⭐⭐
- **Documentation**: [Triton Documentation](https://triton-lang.org/)
- **Tutorial**: [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)

#### Why Study Triton?
- Modern compiler design
- MLIR-based backend
- Clean Python API
- Powers many recent ML optimizations (Flash Attention 2, etc.)

#### Core Concepts
- **Block-level programming model**
- **Automatic parallelization**
- **MLIR-based compilation pipeline**
- **Triton-GPU dialect**

#### Projects
- Implement Flash Attention in Triton
- Write custom fused kernels
- Study the Triton → MLIR → PTX compilation flow

#### Resources
- **Paper**: "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"
- **GitHub Discussions**: Active community

---

## Phase 5: Advanced Topics & Specialization (4-6 months)

### Graph Compilers & Optimization

#### ONNX Ecosystem
- **ONNX IR**: [onnx/onnx](https://github.com/onnx/onnx)
- **ONNX-MLIR**: [onnx/onnx-mlir](https://github.com/onnx/onnx-mlir) - ONNX to MLIR compiler
- **Study**: Graph-level optimizations, constant folding, dead code elimination

#### TorchScript & TorchDynamo
- **TorchScript**: [PyTorch JIT](https://pytorch.org/docs/stable/jit.html)
- **TorchDynamo**: [pytorch/torchdynamo](https://github.com/pytorch/torchdynamo)
- **Torch-MLIR**: [llvm/torch-mlir](https://github.com/llvm/torch-mlir) ⭐

### Kernel Compilation & Code Generation

#### PTX (Parallel Thread Execution)
- **Guide**: [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- **Study**: How CUDA C++ compiles to PTX, then to SASS
- **Tool**: `cuobjdump`, `nvdisasm` for inspecting compiled kernels

#### NVVM IR
- **Documentation**: LLVM IR for NVIDIA GPUs
- **Study**: CUDA → NVVM IR → PTX pipeline
- **Repo**: Part of CUDA toolkit

#### GPU Code Generation in MLIR
- **Study**: `mlir/lib/Conversion/GPUTo*` conversions
- **Focus**: GPU dialect → NVVM/ROCDL → PTX/GCN

### Domain-Specific Compilers

#### IREE (Intermediate Representation Execution Environment)
- **Repo**: [openxla/iree](https://github.com/openxla/iree) ⭐
- **Focus**: MLIR-based, multi-platform, ahead-of-time compilation
- **Study**: HAL (Hardware Abstraction Layer), multi-architecture support

#### Mojo (Modular)
- **Website**: [Mojo Language](https://www.modular.com/mojo)
- **Focus**: Python-compatible language with MLIR backend
- **Study**: How modern syntax compiles to efficient MLIR

---

## Phase 6: Production & Optimization (Ongoing)

### Performance Engineering

#### Books
- *Performance Analysis and Tuning on Modern CPUs* by Denis Bakhvalov ⭐
- *Systems Performance* by Brendan Gregg

#### Tools
- **NVIDIA Nsight Compute**: Kernel profiling
- **NVIDIA Nsight Systems**: System-wide profiling
- **perf**: Linux profiling tool
- **Intel VTune**: CPU profiling

### Library API Design Patterns

#### Study These Libraries
- **Eigen**: [eigen.tuxfamily.org](http://eigen.tuxfamily.org)
  - Expression templates, lazy evaluation
- **Blaze**: [bitbucket.org/blaze-lib/blaze](https://bitbucket.org/blaze-lib/blaze)
  - Smart expression templates
- **PyBind11**: Clean C++/Python binding patterns

#### Principles
- **Books**:
  - *API Design for C++* by Martin Reddy ⭐⭐
  - *Large-Scale C++ Software Design* by Lakos
- **Patterns**: Handles, descriptors, workspaces, algorithm selection

### Testing & CI/CD

#### Testing Frameworks
- **GoogleTest**: C++ unit testing
- **Catch2**: Modern alternative
- **pytest**: Python testing with C++ extensions

#### CI/CD
- **GitHub Actions**: [actions/setup-cuda](https://github.com/Jimver/cuda-toolkit)
- **Docker**: Containerized build environments
- **CTest**: CMake testing integration

---

## Key GitHub Repositories to Study

### Must-Study (⭐⭐⭐)
1. [pytorch/pytorch](https://github.com/pytorch/pytorch) - Framework architecture
2. [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) - Modern C++ template kernels
3. [openai/triton](https://github.com/openai/triton) - Modern GPU compiler
4. [llvm/llvm-project](https://github.com/llvm/llvm-project) - Compiler infrastructure
5. [apache/tvm](https://github.com/apache/tvm) - ML compiler stack

### Highly Recommended (⭐⭐)
6. [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) - Execution providers
7. [openxla/iree](https://github.com/openxla/iree) - MLIR-based compiler
8. [NVIDIA/thrust](https://github.com/NVIDIA/thrust) - Parallel algorithms API
9. [google/jax](https://github.com/google/jax) - Functional API with XLA
10. [llvm/torch-mlir](https://github.com/llvm/torch-mlir) - PyTorch to MLIR

### Important (⭐)
11. [onnx/onnx](https://github.com/onnx/onnx) - IR standards
12. [ROCm/HIP](https://github.com/ROCm/HIP) - Portability layer
13. [google/gemmlowp](https://github.com/google/gemmlowp) - Low-precision GEMM
14. [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) - Attention kernels
15. [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) - Official examples

---

## Learning Resources Hub

### Courses
- [Stanford CS143: Compilers](https://web.stanford.edu/class/cs143/)
- [Cornell CS 6120: Advanced Compilers](https://www.cs.cornell.edu/courses/cs6120/) ⭐
- [MIT 6.172: Performance Engineering](https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/)
- [CMU 15-745: Optimizing Compilers](https://www.cs.cmu.edu/~15745/)

### Papers (Must-Read)
- "MLIR: Scaling Compiler Infrastructure" (Google)
- "TVM: An Automated End-to-End Optimizing Compiler" (UW)
- "Triton: An Intermediate Language and Compiler" (OpenAI)
- "FlashAttention: Fast and Memory-Efficient Exact Attention" (Stanford)
- "Halide: A Language and Compiler for Optimizing Parallelism" (MIT)

### Blogs & Websites
- [LLVM Blog](https://blog.llvm.org/)
- [PyTorch Blog](https://pytorch.org/blog/)
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog)
- [Triton-lang Blog](https://triton-lang.org/main/blog/)
- [Modular Blog](https://www.modular.com/blog)

### Communities
- [LLVM Discourse](https://discourse.llvm.org/)
- [PyTorch Forums](https://discuss.pytorch.org/)
- [TVM Discuss](https://discuss.tvm.apache.org/)
- [r/CUDA](https://reddit.com/r/CUDA)
- [r/Compilers](https://reddit.com/r/Compilers)

---

## Career-Relevant Skills Summary

### For NVIDIA
- CUDA kernel optimization
- CUTLASS template programming
- cuDNN/cuBLAS architecture
- Triton compiler contributions

### For Microsoft
- ONNX Runtime contributions
- DeepSpeed integration
- MLIR/compiler work
- Cross-platform API design

### For Google/Meta
- JAX/XLA compiler work
- MLIR dialect development
- PyTorch compiler (TorchDynamo)
- Distributed systems

---

## Monthly Study Plan Template

### Example Month (Adjust based on phase)
- **Week 1**: Theory (book chapters, papers)
- **Week 2**: Code reading (study existing implementations)
- **Week 3**: Toy project (learning exercise)
- **Week 4**: Contribution or portfolio project

### Daily Routine
- 1-2 hours: Reading (books/papers/docs)
- 1-2 hours: Code study (GitHub repos)
- 1-2 hours: Hands-on coding/projects
- 30 min: Community engagement (forums, issues)

---

This roadmap is self-paced but expect 12-18 months for comprehensive coverage. Focus on depth in MLIR and one ML compiler (recommend Triton or TVM) for maximum career impact.