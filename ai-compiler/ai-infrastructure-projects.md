# AI Infrastructure Projects: Learning & Portfolio (v2)
**Focus**: C++ APIs, Compilers, GPU Systems (No Python Bindings)

---

## Part 1: API Learning Projects (1-2 weeks each)

These teach you practical C++ API usage and design patterns.

### AL1: CUDA API Explorer

**Goal**: Master CUDA runtime APIs through practical use

**Tasks**:
1. Implement programs using every major CUDA API category:
   - Memory: malloc/free, memcpy (sync/async), managed, pitched
   - Streams: creation, priority, callbacks, synchronization
   - Events: recording, timing, inter-stream dependencies
   - Graphs: capture, instantiate, launch
   
2. Create a C++ wrapper library:
   - RAII wrappers for handles
   - Type-safe memory operations
   - Error handling utilities
   
3. **Deliverable**: Header-only CUDA utilities library

**Skills**: CUDA APIs, C++ RAII, error handling, modern C++ design

---

### AL2: cuBLAS Deep Dive

**Goal**: Understand high-performance library API design

**Tasks**:
1. Write programs calling 20+ cuBLAS functions
   - Different precisions (SGEMM, DGEMM, HGEMM)
   - Batched operations
   - Strided batched operations
   
2. Implement simple matrix ops using cuBLAS
   - Study handle management patterns
   - Understand workspace allocation
   
3. Benchmark and profile
   - Compare different algorithm choices
   - Measure API overhead

**Deliverable**: Benchmark suite + API pattern documentation

**Skills**: Library API design, BLAS, performance analysis

---

### AL3: CUTLASS Configuration Explorer

**Goal**: Navigate complex template-based APIs

**Tasks**:
1. Instantiate CUTLASS GEMM with 10+ different configurations:
   - Different tile sizes
   - Different thread block shapes
   - Different epilogue operations
   
2. Measure performance impact of each configuration
3. Understand template parameter meanings
4. Trace through template instantiation

**Deliverable**: Performance study with graphs

**Skills**: C++ templates, GPU optimization, CUTLASS APIs

---

### AL4: Thrust Algorithm Implementation

**Goal**: Design STL-like parallel APIs

**Tasks**:
1. Implement parallel algorithms using Thrust:
   - Transform, reduce, scan, sort
   - Custom functors and iterators
   
2. Study Thrust source code:
   - How backends are selected
   - Policy-based design patterns
   
3. Write custom iterator type

**Deliverable**: Parallel algorithms library with custom iterators

**Skills**: Generic programming, parallel algorithms, API design

---

### AL5: MLIR API Basics

**Goal**: Learn to construct IR programmatically

**Tasks**:
1. Use MLIR C++ APIs to build IR:
   - Create module, functions, basic blocks
   - Build operations programmatically
   - Set attributes and types
   
2. Implement simple programs in multiple dialects:
   - Arith dialect
   - SCF dialect
   - Func dialect
   
3. Print and verify IR

**Deliverable**: MLIR IR generator for simple programs

**Skills**: MLIR APIs, IR construction, dialect usage

---

## Part 2: Compiler Learning Projects (2-4 weeks each)

### CL1: Expression Compiler (LLVM)

**Goal**: End-to-end compiler with LLVM

**Components**:
- Lexer & Parser (hand-written or ANTLR)
- AST construction
- LLVM IR generation via IRBuilder
- JIT execution

**Language Features**:
- Variables (int, float)
- Arithmetic operations
- Functions
- Control flow (if, while)

**Deliverable**: Working compiler + documentation

**Skills**: Compiler pipeline, LLVM IR, IR Builder API

---

### CL2: LLVM Optimization Pass Collection

**Goal**: Master LLVM pass API

**Implement 5+ passes**:
1. Dead code elimination
2. Constant propagation
3. Strength reduction
4. Loop invariant code motion
5. Simple inlining

**Deliverable**: LLVM pass library with tests

**Skills**: LLVM Pass API, SSA form, optimization theory

---

### CL3: MLIR Toy Tutorial Extended

**Goal**: Deep MLIR understanding

**Tasks**:
1. Complete all 7 Toy tutorial chapters
2. Add 5 new operations:
   - Matrix transpose
   - Element-wise operations
   - Reduction operations
   
3. Implement optimizations:
   - Constant folding
   - Operation fusion patterns
   
4. Add new lowering: Toy → Linalg

**Deliverable**: Extended Toy language with documentation

**Skills**: MLIR dialects, pattern rewriting, lowering

---

### CL4: Mini Tensor Compiler (MLIR)

**Goal**: Build domain-specific compiler

**Description**:
- Compile tensor operations to optimized code
- Use existing MLIR dialects

**Pipeline**:
1. Parse tensor expressions
2. Generate MLIR (tensor/linalg dialect)
3. Apply transformations (tiling, vectorization)
4. Lower: Linalg → SCF → LLVM
5. Execute via JIT

**Deliverable**: Tensor DSL compiler

**Skills**: MLIR pipeline, Linalg dialect, transformations

---

### CL5: Triton Kernel Study

**Goal**: Understand modern GPU compiler

**Tasks**:
1. Implement 10+ kernels in Triton
2. Study generated MLIR (enable dumps)
3. Trace compilation: Triton IR → Triton-GPU → LLVM → PTX
4. Read Triton source code:
   - Frontend (AST → IR)
   - Transformations
   - Lowering passes

**Deliverable**: Kernel library + compiler analysis doc

**Skills**: Triton, MLIR-based compilation, GPU codegen

---

## Part 3: Portfolio Projects (4-8 weeks each)

These are substantial, interview-worthy projects.

### Portfolio Tier 1 (Foundational)

#### P1: Optimized GEMM Kernel Series

**Goal**: Demonstrate GPU optimization expertise

**Implementation Stages**:
1. Naive global memory (baseline)
2. Shared memory tiling
3. Register blocking
4. Vectorized memory access
5. Warp-level GEMM (WMMA)
6. Tensor Core version (MMA)
7. Async copy (Hopper if available)

**Additional**:
- Multiple data types (FP32, FP16, INT8, BF16)
- C++ template-based API for configuration
- Comprehensive benchmarks vs cuBLAS
- Roofline analysis

**Deliverable**:
- Clean, well-documented code
- Performance graphs
- Technical writeup explaining each optimization
- CMake build system

**Skills**: CUDA optimization, memory hierarchy, C++ templates, benchmarking

**Target Companies**: NVIDIA, AMD, any GPU-focused role

---

#### P2: CUTLASS-Inspired Template Library

**Goal**: Modern C++ template programming for GPUs

**Description**:
- Build a smaller, educational version of CUTLASS
- Focus on clarity over ultimate performance
- Heavy use of C++17/20 features

**Features**:
- Template-based tile operations
- Configurable thread block structure
- Compile-time optimizations
- Multiple backends (CPU for testing, CUDA)

**Structure**:
```
include/
  gemm/
    device/     # Device-level API
    kernel/     # Kernel templates
    thread/     # Thread-level tiles
  arch/         # Architecture-specific
  epilogue/     # Post-processing
```

**Deliverable**: Template library with examples and docs

**Skills**: C++ templates, metaprogramming, API design

---

#### P3: CUDA Library with Clean C++ API

**Goal**: Design production-quality GPU library API

**Domain**: Choose one:
- Image processing primitives
- Signal processing (FFT-based operations)
- Sparse matrix operations
- Graph algorithms

**API Requirements**:
- Handle-based resource management
- Clear error handling
- Workspace management pattern
- Algorithm selection APIs
- Batched operations support

**Implementation**:
- Multiple kernel implementations per operation
- Auto-tuning based on problem size
- Comprehensive unit tests (GoogleTest)
- Benchmarks (Google Benchmark)
- Doxygen documentation

**Deliverable**: Production-quality library

**Skills**: API design, CUDA, testing, documentation

**Target Companies**: NVIDIA (libraries team), AMD

---

### Portfolio Tier 2 (Advanced Compiler)

#### P4: ML Operator Compiler (MLIR-based)

**Goal**: Build end-to-end ML compiler

**Description**:
- Compile subset of neural network ops to GPU
- Focus on transformers or CNNs

**Architecture**:

1. **Frontend**:
   - Import from ONNX
   - Generate tensor/linalg dialect
   
2. **Optimization Pipeline**:
   - Operator fusion passes (5+)
   - Tiling transformations
   - Layout optimizations
   - Constant folding
   
3. **Backend**:
   - Lower to GPU dialect
   - Generate efficient kernels
   - JIT execution
   
4. **Runtime**:
   - Simple runtime for kernel launch
   - Workspace management

**Supported Ops** (minimum):
- MatMul, Conv2D
- Activations (ReLU, GELU)
- LayerNorm, Softmax
- Attention (bonus)

**Deliverable**:
- Working compiler
- Run inference on real model (ResNet/BERT)
- Performance comparison vs PyTorch
- Architecture documentation

**Skills**: MLIR, compiler design, optimization, GPU codegen

**Target Companies**: All (Google, Meta, NVIDIA, Modular)

---

#### P5: Custom MLIR Dialect + Full Lowering

**Goal**: Demonstrate MLIR mastery

**Choose Domain**:
- Sparse tensor operations
- Graph neural network ops
- Quantized operations
- Custom attention variants

**Requirements**:

1. **Dialect Definition**:
   - 15+ custom operations
   - Custom types
   - Traits and interfaces
   - TableGen definitions
   
2. **Optimizations**:
   - 10+ pattern rewrites
   - Dialect-specific transformations
   - Cost models for decisions
   
3. **Lowering Pipeline**:
   - High-level dialect → Linalg
   - Linalg → SCF + GPU
   - GPU → NVVM → PTX
   
4. **Integration**:
   - Usable from C++ API
   - Bonus: PyTorch integration via torch-mlir

**Deliverable**:
- Complete dialect implementation
- End-to-end examples
- Performance evaluation
- Contribution path (even if not accepted)

**Skills**: MLIR expertise, dialect design, optimization

**Target Companies**: Google, Modular, Meta, compiler-focused roles

---

#### P6: Triton-Like Compiler (Simplified)

**Goal**: Build block-level GPU language compiler

**Description**:
- Simpler than Triton but same concepts
- Educational focus

**Components**:

1. **Frontend DSL**:
   - C++-embedded DSL or simple language
   - Block-level programming model
   
2. **IR Design**:
   - MLIR-based
   - Custom dialect for block operations
   - Memory layout annotations
   
3. **Transformations**:
   - Automatic parallelization
   - Memory coalescing
   - Shared memory optimization
   
4. **Backend**:
   - Lower to GPU dialect
   - Generate PTX via LLVM

**Example Programs**:
- GEMM
- Softmax
- LayerNorm
- Flash Attention (simplified)

**Deliverable**: Working compiler with examples

**Skills**: Language design, MLIR, GPU codegen

---

#### P7: Auto-Tuning Framework

**Goal**: Automatic performance optimization

**Description**:
- Meta-programming system for kernel generation
- Search for optimal configurations

**Components**:

1. **Kernel Generator**:
   - Template-based or code generation
   - Generate 1000s of variants
   - Parameters: tile size, thread count, etc.
   
2. **Search Strategy**:
   - Random search (baseline)
   - Genetic algorithm
   - ML-guided (optional)
   
3. **Cost Model**:
   - Analytical model (roofline)
   - Or learned from profiling data
   
4. **Caching**:
   - Store tuned configs
   - Database of optimal parameters

**Focus on**: One kernel type (GEMM, Conv, Attention)

**Deliverable**:
- Auto-tuning framework
- Comparison vs hand-tuned and AutoTVM
- Documentation

**Skills**: Meta-programming, performance modeling, search algorithms

---

### Portfolio Tier 3 (Research-Level)

#### P8: Advanced Kernel Fusion System

**Goal**: Automatic fusion with novel techniques

**Approach**:
- Graph analysis for fusion opportunities
- Cost model for fusion decisions
- Support for multi-kernel fusion

**Novel Aspects** (choose 1-2):
- ML-based fusion decisions
- Polyhedral-based fusion
- Profile-guided fusion
- Memory-aware fusion

**Evaluation**:
- Test on large models (LLaMA, BERT, Stable Diffusion)
- Measure: Speedup, memory usage, compilation time
- Compare vs: Eager, XLA, TorchScript

**Deliverable**:
- Fusion framework
- Extensive evaluation
- Technical paper (MLSys format)

**Skills**: Research, advanced optimization, ML systems

---

#### P9: Domain-Specific Compiler for Attention

**Goal**: Specialize in critical ML primitive

**Support**:
- Multi-head attention
- Flash Attention variants
- Multi-query attention
- Sparse attention patterns

**Features**:
- High-level specification
- Automatic optimization (tiling, fusion)
- Multi-GPU support
- Memory-efficient implementations

**Technical Depth**:
- IO analysis (like Flash Attention paper)
- Optimal tiling strategies
- Kernel fusion
- Quantization support

**Evaluation**:
- Compare vs PyTorch native
- Compare vs Flash Attention
- Show memory/speed tradeoffs

**Deliverable**:
- Attention compiler
- Integration with transformers library
- Technical writeup

---

#### P10: Multi-Backend Execution Framework

**Goal**: Portable high-performance execution

**Description**:
- Single IR, multiple backends
- Inspired by ONNX Runtime

**Backends**:
- CUDA
- ROCm (via HIP)
- CPU (via LLVM)
- Vulkan (bonus)

**Architecture**:
1. Graph IR (use ONNX or custom)
2. Backend abstraction layer
3. Per-backend optimizations
4. Runtime dispatcher

**Advanced Features**:
- Heterogeneous execution (multi-device)
- Dynamic shapes
- Operator partial execution

**Deliverable**:
- Multi-backend framework
- Performance portability study

**Skills**: Systems design, multi-backend, abstraction

---

## Part 4: Rust-Based Projects

Since you have Rust background:

### R1: CUDA Kernel Launcher (Rust)

**Goal**: Safe Rust API for CUDA

**Features**:
- Type-safe kernel launches
- Memory management with Rust ownership
- Stream/event abstractions
- Error handling with Result types

**Study**: [cudarc](https://github.com/coreylowman/cudarc)

**Deliverable**: Rust CUDA library

---

### R2: LLVM Bindings Project (Rust)

**Goal**: LLVM-based compiler in Rust

**Use**: [inkwell](https://github.com/TheDan64/inkwell)

**Project**: Build a simple language compiler in Rust

**Benefits**: Rust safety + LLVM power

---

### R3: Cranelift Backend

**Goal**: Learn code generation in Rust

**Study**: [Cranelift](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift)

**Project**: Add optimizations or study existing backend

**Skills**: Code generation, Rust, compiler backend

---

## Part 5: Open Source Contribution Strategy

Contributing to major projects is highly valuable.

### Strategy 1: Documentation First

1. Read code deeply
2. Improve documentation
3. Add examples
4. Write tutorials

**Why**: Learn codebase, visible contributions, appreciated

### Strategy 2: Fix Beginner Issues

Search for labels:
- `good first issue`
- `documentation`
- `help wanted`

**Projects**:
- PyTorch
- TVM
- Triton
- ONNX Runtime

### Strategy 3: Implement Missing Features

1. Study roadmap/issues
2. Find unimplemented operators
3. Propose implementation
4. Submit PR

**Examples**:
- Missing CUDA kernels in PyTorch
- New operator schedules in TVM
- Dialect ops in MLIR

### Strategy 4: Performance Improvements

1. Profile existing code
2. Identify bottlenecks
3. Propose optimizations
4. Benchmark improvements

**Target**: CUTLASS, PyTorch ops, TVM schedules

### High-Value Targets:

#### PyTorch
- **Repo**: [pytorch/pytorch](https://github.com/pytorch/pytorch)
- **Focus**: `aten/src/ATen/native/cuda/` - CUDA kernels
- **Impact**: Used by millions

#### Apache TVM
- **Repo**: [apache/tvm](https://github.com/apache/tvm)
- **Focus**: Operator schedules, new hardware backends
- **Community**: Very welcoming

#### OpenAI Triton
- **Repo**: [openai/triton](https://github.com/openai/triton)
- **Focus**: Compiler passes, tutorials, examples
- **Impact**: Growing rapidly

#### MLIR
- **Repo**: [llvm/llvm-project/mlir](https://github.com/llvm/llvm-project/tree/main/mlir)
- **Focus**: Dialect operations, conversions, documentation
- **Prestige**: Very high

#### torch-mlir
- **Repo**: [llvm/torch-mlir](https://github.com/llvm/torch-mlir)
- **Focus**: Op coverage, optimization passes
- **Growing**: Active development

---

## Part 6: Project Selection Guide

### For Learning (First 3-6 months):
1. Do ALL API Learning projects (AL1-AL5)
2. Do ALL Compiler Learning projects (CL1-CL5)
3. Pick ONE Tier 1 portfolio project

### For Job Hunting (6-12 months):

#### Target: NVIDIA
- **Must**: P1 (GEMM), P2 (Templates)
- **Plus**: CUTLASS contributions
- **Bonus**: P3 (CUDA Library)

#### Target: Microsoft
- **Must**: P3 (Library API), P5 (MLIR Dialect)
- **Plus**: ONNX Runtime contributions
- **Bonus**: P10 (Multi-backend)

#### Target: Google
- **Must**: P4 (ML Compiler), P5 (MLIR Dialect)
- **Plus**: MLIR contributions
- **Bonus**: P9 (Attention)

#### Target: Meta
- **Must**: P4 (ML Compiler), PyTorch contributions
- **Plus**: torch-mlir work
- **Bonus**: P8 (Fusion)

#### Target: Modular, Startups
- **Must**: P4 (ML Compiler), P5 (MLIR Dialect)
- **Plus**: P8/P9 (Research)
- **Bonus**: Novel ideas, speed of implementation

### For Research/PhD:
- P8, P9, or P10
- Focus on novel contributions
- Write paper (MLSys, ASPLOS, CGO)

---

## Part 7: Implementation Best Practices

### Code Quality Standards

**C++ Style**:
- Follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Use clang-format (config from LLVM/PyTorch)
- Modern C++17/20 features

**Build System**:
- Modern CMake (3.18+)
- Proper target exports
- Find modules for dependencies

**Testing**:
- GoogleTest for C++
- Minimum 80% coverage
- Unit tests + integration tests

**Documentation**:
- Doxygen for APIs
- Markdown for architecture
- Examples/ directory

**CI/CD**:
- GitHub Actions
- Build + test on commit
- Run on CPU (always) + GPU (if possible)

### Repository Structure

```
project-name/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   └── tutorials/
├── include/
│   └── project/
│       ├── core/
│       ├── cuda/
│       └── utils/
├── src/
│   ├── core/
│   ├── cuda/
│   └── CMakeLists.txt
├── tests/
│   ├── unit/
│   ├── integration/
│   └── CMakeLists.txt
├── benchmarks/
│   └── gemm_bench.cpp
├── examples/
│   ├── basic/
│   └── advanced/
└── third_party/
```

### Performance Benchmarking

**Tools**:
- Google Benchmark (C++)
- Custom CUDA timing (events)
- Nsight Compute for profiling

**Metrics**:
- GFLOPS / TFLOPS
- Memory bandwidth utilization
- Occupancy
- Wall-clock time

**Visualization**:
- matplotlib / seaborn (Python)
- Generate graphs in README
- Compare vs baselines

### Documentation Requirements

**README Must Have**:
1. Project description (2-3 paragraphs)
2. Key features (bullet points)
3. Quick start (5-minute example)
4. Building from source
5. Performance benchmarks (with graphs!)
6. Architecture overview (diagram)
7. API examples
8. Comparison with alternatives
9. Future work
10. License

**API Documentation**:
- Doxygen comments on all public APIs
- Usage examples in comments
- Parameter descriptions
- Return value semantics

**Architecture Docs**:
- Design decisions explained
- Diagrams (draw.io, mermaid)
- Trade-offs discussed

---

## Part 8: Timeline

### Months 1-2: API Foundations
- Complete AL1-AL5 (API learning)
- Read CUTLASS, Thrust source code
- Practice using CUDA/cuBLAS/cuDNN APIs

### Months 3-4: Compiler Basics
- Complete CL1-CL5 (Compiler learning)
- Finish MLIR Toy tutorial
- Study LLVM passes

### Months 5-7: First Portfolio Project
- Choose P1, P2, or P3
- Build to production quality
- Write comprehensive docs
- Benchmark and visualize

### Months 8-12: Advanced Compiler Project
- Choose P4, P5, or P6
- Deep dive into MLIR
- Contribute to open source alongside

### Months 12-18: Research Project (Optional)
- P8, P9, or P10
- Novel contributions
- Write paper or extensive blog series

### Ongoing:
- Open source contributions (weekly)
- Read papers (2-3 per month)
- Engage with communities
- Blog about learnings

---

## Part 9: Showcasing Work

### GitHub Profile
- Pin 4-6 best projects
- Consistent activity (contributions graph)
- Star and watch relevant repos
- Detailed profile README

### Technical Blog
**Platform**: Medium, Dev.to, or personal site

**Post Ideas**:
- "Building a GEMM Kernel: 0 to 1 TFLOPS"
- "Understanding MLIR: A Practical Guide"
- "Diving into CUTLASS: Template Magic for GPUs"
- "My Journey Contributing to PyTorch"
- "Optimizing Attention: A Compiler Perspective"

**Frequency**: 1-2 posts per month

### Talks
- Local meetups (GPU computing, compilers)
- Conference lightning talks
- YouTube tutorials (screen recordings)

### Portfolio Site
Simple site with:
- Project showcase (with images/graphs)
- Technical blog aggregation
- About + CV
- Links to GitHub, LinkedIn

---

This updated version focuses on C++ API mastery, removes Python binding content, and includes Rust opportunities where relevant. The projects are more compiler and systems-focused, which aligns with your interests and background.