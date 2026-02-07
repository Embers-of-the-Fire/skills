# AI Infrastructure Projects: Learning Toys & Portfolio Ideas

This document contains projects organized by difficulty and purpose. **Learning toys** are smaller exercises to understand concepts. **Portfolio projects** are substantial enough to showcase on GitHub and discuss in interviews.

---

## Part 1: Learning Toys (Quick Exercises)

These are small, focused projects to understand specific concepts. Spend days to a week on each.

### Category: CUDA Basics

#### 1. Parallel Reduction Variants
**Goal**: Understand warp-level programming and optimization
- Implement sum reduction with different strategies:
  - Naive global memory
  - Shared memory with bank conflicts
  - Shared memory without bank conflicts
  - Warp shuffle intrinsics
  - Cooperative groups
- **Compare** performance at each level
- **Skills**: CUDA memory hierarchy, profiling

#### 2. Memory Coalescing Visualizer
**Goal**: Understand memory access patterns
- Write kernels with different access patterns (coalesced, strided, random)
- Use Nsight Compute to measure memory throughput
- Create a simple visualization/report of findings
- **Skills**: Memory optimization, profiling tools

#### 3. CUDA Stream Scheduler
**Goal**: Understand asynchronous execution
- Launch multiple kernels across different streams
- Implement data dependencies between streams
- Measure overlap between compute and memory transfers
- **Skills**: Streams, events, pipelining

### Category: C++ Templates & API Design

#### 4. Mini-Tensor Library
**Goal**: Template metaprogramming for compile-time optimization
- Create a simple tensor class with compile-time dimension checking
- Implement basic operations (add, multiply, transpose)
- Use expression templates for lazy evaluation (like Eigen)
- **Skills**: C++ templates, expression templates, API design

#### 5. Python Binding Explorer
**Goal**: Understand C++/Python interop
- Write a simple C++ math library
- Create bindings with:
  - pybind11
  - Python C API (manual)
  - Cython
- Compare performance and development experience
- **Skills**: Language bindings, performance profiling

#### 6. Plugin System
**Goal**: Dynamic loading and API versioning
- Create a simple plugin architecture for loading custom kernels
- Implement version checking and backward compatibility
- **Skills**: Shared libraries, API design, versioning

### Category: Compiler Basics

#### 7. Tiny Expression Compiler
**Goal**: End-to-end compiler understanding
- Write a compiler for simple math expressions (e.g., `x * 2 + y / 3`)
- Stages: Lexer → Parser → AST → LLVM IR → JIT execution
- Support variables and basic operations
- **Skills**: Compiler pipeline, LLVM basics

#### 8. LLVM Optimization Pass
**Goal**: Understand LLVM pass infrastructure
- Implement a simple optimization pass (e.g., strength reduction, constant folding)
- Use LLVM's pass manager
- Test on small programs
- **Skills**: LLVM IR, optimization theory

#### 9. MLIR Dialect Toy
**Goal**: MLIR dialect creation
- Follow the MLIR Toy tutorial completely
- Create one additional operation (e.g., matrix transpose)
- Implement its lowering through the pipeline
- **Skills**: MLIR dialects, pattern rewriting

### Category: ML Framework Internals

#### 10. Minimal Autograd Engine
**Goal**: Understand automatic differentiation
- Implement basic tensor with autograd (like micrograd)
- Support operations: add, mul, matmul, relu
- Compute gradients via backpropagation
- **Skills**: Autograd concepts, computational graphs

#### 11. PyTorch Custom Operator
**Goal**: Extend PyTorch with custom ops
- Implement a simple custom operation (e.g., fused LayerNorm)
- Write both CPU and CUDA kernels
- Register with PyTorch dispatcher
- Write Python tests with torch.autograd.gradcheck
- **Skills**: PyTorch internals, operator registration

#### 12. Kernel Fusion Study
**Goal**: Understand when/why to fuse operators
- Implement separate kernels: `x = relu(matmul(A, B) + bias)`
- Implement fused version in single kernel
- Measure memory bandwidth savings
- **Skills**: Kernel fusion, performance analysis

### Category: GPU Compiler

#### 13. Triton Kernel Portfolio
**Goal**: Learn Triton programming model
- Reimplement 5 common operations in Triton:
  - GEMM (matrix multiply)
  - Softmax
  - LayerNorm
  - Flash Attention
  - Fused GELU
- Compare against PyTorch built-ins
- **Skills**: Triton language, block-level programming

#### 14. PTX Inspector
**Goal**: Understand GPU assembly
- Write simple CUDA kernels
- Examine generated PTX and SASS
- Identify optimization opportunities (register usage, instruction mix)
- **Skills**: PTX, low-level GPU programming

---

## Part 2: Portfolio Projects (Substantial Work)

These are more involved projects that demonstrate real expertise. Each should take 2-6 weeks.

### Tier 1: Foundation Projects (Good First Portfolio Pieces)

#### P1. High-Performance GEMM Library
**Goal**: Demonstrate GPU optimization mastery

**Description**:
- Implement matrix multiplication with progressive optimizations
- Multiple versions: naive → tiled → shared memory → register blocking → tensor cores
- Support multiple data types (FP32, FP16, INT8)
- Performance comparison against cuBLAS
- Detailed README explaining each optimization

**Technical Details**:
- Use CUTLASS as reference (but implement yourself)
- Create Python bindings
- Benchmark suite with different matrix sizes
- Visualizations of performance vs cuBLAS

**Skills Demonstrated**: CUDA optimization, memory hierarchy, benchmarking, API design

**Bonus**: Implement in both CUDA and Triton, compare productivity vs performance

---

#### P2. PyTorch Operator Extension Pack
**Goal**: Show framework integration skills

**Description**:
- Collection of 5-10 custom PyTorch operators
- Each with CPU, CUDA, and optionally Triton implementations
- Full autograd support with gradient tests
- Python package with pip install

**Example Operators**:
- Fused Adam optimizer step
- Fused cross-entropy + softmax
- Flash Attention variant
- Fused dropout + residual connection
- Custom activation functions

**Technical Details**:
- Proper PyTorch dispatcher registration
- Comprehensive unit tests
- Benchmark comparisons vs native PyTorch
- CI/CD with GitHub Actions

**Skills Demonstrated**: PyTorch internals, operator fusion, production packaging

---

#### P3. TVM Custom Backend
**Goal**: Demonstrate compiler knowledge

**Description**:
- Implement a custom backend for TVM targeting specific hardware or simulation
- Alternative: Contribute scheduling templates for new operators
- Document the process thoroughly

**Options**:
- Custom CPU backend with specific intrinsics (AVX-512, etc.)
- WebGPU backend
- Custom accelerator simulation
- New operator schedules (Conv2D variants, Attention)

**Technical Details**:
- Integration with TVM's code generation
- AutoTVM/MetaSchedule support
- Benchmark improvements
- Contribution path to TVM (open PR)

**Skills Demonstrated**: Compiler backends, code generation, TVM architecture

---

### Tier 2: Advanced Projects (Strong Portfolio Pieces)

#### P4. Mini Deep Learning Compiler
**Goal**: End-to-end compiler implementation

**Description**:
- Build a compiler that takes a neural network and generates optimized code
- Focus on specific domain (e.g., transformers, CNNs)
- Multiple optimization passes

**Components**:
- **Frontend**: Import from PyTorch/ONNX
- **IR**: Define your own or use MLIR
- **Optimization**: Operator fusion, constant folding, layout optimization
- **Backend**: Generate CUDA or Triton code
- **Runtime**: Minimal runtime for execution

**Technical Details**:
- Use MLIR for IR (recommended) or design custom IR
- At least 3-4 optimization passes
- Support 10-15 operators
- End-to-end inference on a real model (ResNet, BERT)

**Skills Demonstrated**: Compiler architecture, graph optimization, code generation

**Examples to Study**: TVM, IREE, XLA

---

#### P5. Kernel Auto-Tuner Framework
**Goal**: Demonstrate autotuning and metaprogramming

**Description**:
- Framework for automatically tuning CUDA kernels
- Search over different tiling, threading, and memory strategies
- Machine learning-based search (optional)

**Features**:
- Template-based kernel generator
- Search space definition DSL
- Performance model (optional: learned or analytical)
- Auto-scheduler
- Caching of tuned kernels

**Technical Details**:
- Focus on one kernel class (GEMM, Conv, Attention)
- Generate 1000s of variants programmatically
- Profile and select best configurations
- Compare against AutoTVM approach

**Skills Demonstrated**: Meta-programming, performance modeling, automation

---

#### P6. MLIR Dialect for Domain-Specific Operations
**Goal**: Advanced MLIR expertise

**Description**:
- Design and implement a complete MLIR dialect
- Full lowering pipeline to GPU or CPU
- Demonstrate on real use case

**Example Domains**:
- Graph neural networks (GNN operations)
- Sparse tensor operations
- Quantized operations
- Attention variants

**Components**:
- Custom operations and types
- Dialect interfaces and traits
- Pattern rewriting for optimization
- Lowering to Linalg/GPU/LLVM dialects
- Integration with PyTorch via torch-mlir

**Technical Details**:
- At least 10 custom ops
- 5+ optimization patterns
- End-to-end example notebook
- Documentation and tests

**Skills Demonstrated**: MLIR mastery, dialect design, compiler infrastructure

---

#### P7. Distributed Training Micro-Framework
**Goal**: Show systems + libraries expertise

**Description**:
- Lightweight framework for distributed training
- Focus on clear API design and efficiency
- Support multiple parallelism strategies

**Features**:
- Data parallel with gradient synchronization (NCCL)
- Simple model parallel
- Pipeline parallel (optional)
- Python API wrapping C++ core
- Integration with PyTorch models

**Technical Details**:
- Use NCCL for communication
- Custom CUDA kernels for gradient operations
- Handle fault tolerance (checkpointing)
- Benchmarks vs PyTorch DDP

**Skills Demonstrated**: Distributed systems, NCCL, API design, PyTorch integration

---

### Tier 3: Research-Level Projects (Publication-Worthy)

#### P8. Novel Kernel Fusion Framework
**Goal**: Publishable contribution

**Description**:
- Automatic kernel fusion based on novel heuristics or learned models
- Apply to transformer or diffusion models
- Demonstrate speedups

**Approach**:
- Graph analysis to find fusion opportunities
- Cost model for fusion decisions (analytical or learned)
- Code generation for fused kernels
- Integration with PyTorch or JAX

**Validation**:
- Test on LLaMA, BERT, Stable Diffusion
- Measure end-to-end speedup and memory reduction
- Compare against eager execution and XLA/TorchScript

**Skills Demonstrated**: Research ability, compiler optimization, ML systems

**Publication Target**: MLSys, ASPLOS, CGO

---

#### P9. Attention Mechanism Compiler
**Goal**: Specialize in critical ML primitive

**Description**:
- Compiler specifically for attention variants
- Support Flash Attention, Multi-Query, Grouped-Query, etc.
- Automatic optimization based on hardware

**Features**:
- High-level DSL for attention patterns
- Automatic tiling and memory optimization
- Support for sparse attention
- Multi-GPU optimization

**Technical Details**:
- Use MLIR or custom IR
- Generate Triton or CUDA code
- Extensive benchmarks on A100/H100
- Integration with HuggingFace transformers

**Skills Demonstrated**: Domain expertise, advanced optimization, production quality

---

#### P10. Quantization-Aware Compiler
**Goal**: Address critical production need

**Description**:
- Compiler for mixed-precision and quantized models
- Automatic precision selection
- Custom kernels for quantized operations

**Features**:
- INT8, FP16, BF16, FP8 support
- Automatic mixed precision
- Calibration for quantization
- Quantized GEMM and convolution kernels
- Model accuracy vs performance tradeoff analysis

**Technical Details**:
- Integration with PyTorch/ONNX
- Custom CUDA kernels or CUTLASS-based
- Profile on real models (BERT, ResNet, etc.)
- Achieve <1% accuracy loss with 2-3x speedup

**Skills Demonstrated**: Quantization expertise, kernel optimization, production ML

---

## Part 3: Open Source Contribution Ideas

Contributing to major projects is highly valuable. Here are strategic areas:

### PyTorch
- **Repo**: [pytorch/pytorch](https://github.com/pytorch/pytorch)
- **Ideas**:
  - Implement missing CUDA kernels in `aten/src/ATen/native/cuda/`
  - Add CPU optimizations with AVX-512
  - Improve torch.compile coverage
  - Add custom ops to core library
- **Labels**: `module: cpu`, `module: cuda`, `good first issue`

### NVIDIA CUTLASS
- **Repo**: [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)
- **Ideas**:
  - New GEMM epilogue patterns
  - Support for new GPU architectures
  - Additional tensor operations
  - Documentation improvements

### Apache TVM
- **Repo**: [apache/tvm](https://github.com/apache/tvm)
- **Ideas**:
  - New operator schedules
  - Frontend improvements (PyTorch, ONNX)
  - AutoScheduler enhancements
  - Backend optimizations
- **Labels**: `good first issue`, `operator`

### OpenAI Triton
- **Repo**: [openai/triton](https://github.com/openai/triton)
- **Ideas**:
  - New tutorial implementations
  - Compiler optimization passes
  - Backend improvements
  - Documentation
- **Very active community, responsive maintainers**

### MLIR
- **Repo**: [llvm/llvm-project/mlir](https://github.com/llvm/llvm-project/tree/main/mlir)
- **Ideas**:
  - New dialect operations
  - Conversion patterns
  - Documentation
  - Examples
- **Join**: [LLVM Discourse MLIR category](https://discourse.llvm.org/c/mlir/)

### Microsoft ONNX Runtime
- **Repo**: [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)
- **Ideas**:
  - Execution provider improvements
  - Quantization enhancements
  - Graph optimizations
  - New operator support

### torch-mlir
- **Repo**: [llvm/torch-mlir](https://github.com/llvm/torch-mlir)
- **Ideas**:
  - Op coverage expansion
  - Optimization passes
  - Backend support
  - Testing infrastructure

---

## Part 4: Project Selection Strategy

### For Maximum Learning
1. Start with **Learning Toys** in each category
2. Build **P1 (GEMM)** or **P2 (PyTorch Extensions)** first
3. Deep dive into either **compiler** (P4, P6) or **libraries** (P7) based on interest
4. Contribute to open source alongside personal projects

### For Job Applications

#### Target: NVIDIA
**Priority**: P1 (GEMM) → P2 (PyTorch Ops) → CUTLASS contributions
**Skills**: CUDA, CUTLASS, performance optimization

#### Target: Microsoft
**Priority**: P2 (PyTorch Ops) → P7 (Distributed) → ONNX Runtime contributions
**Skills**: API design, PyTorch, ONNX, cross-platform

#### Target: Google/DeepMind
**Priority**: P4 (Compiler) → P6 (MLIR) → JAX/XLA contributions
**Skills**: Compilers, MLIR, functional programming

#### Target: Meta
**Priority**: P2 (PyTorch Ops) → P4 (Compiler) → PyTorch contributions
**Skills**: PyTorch internals, compilers, distributed training

#### Target: Startups (Modular, Together AI, etc.)
**Priority**: P4 (Compiler) → P6 (MLIR) → P8/P9 (Research)
**Skills**: End-to-end, research implementation, speed

---

## Part 5: Project Best Practices

### Code Quality
- **Style**: Follow Google C++ Style Guide
- **Documentation**: Doxygen for C++, docstrings for Python
- **Tests**: GoogleTest (C++), pytest (Python)
- **CI/CD**: GitHub Actions with CUDA support

### Repository Structure
```
project-name/
├── README.md              # Detailed, with examples
├── CMakeLists.txt         # Modern CMake
├── include/               # Public headers
│   └── project/
├── src/                   # Implementation
│   ├── cpu/              # CPU kernels
│   └── cuda/             # CUDA kernels
├── python/                # Python bindings
│   └── setup.py
├── tests/                 # Unit tests
├── benchmarks/            # Performance tests
├── examples/              # Usage examples
└── docs/                  # Documentation
```

### README Must-Haves
1. Clear project description
2. Installation instructions
3. Quick start example
4. Architecture overview
5. Performance benchmarks (with graphs!)
6. Comparison with alternatives
7. Future work / limitations

### Performance Benchmarks
- Use Google Benchmark (C++) or pytest-benchmark (Python)
- Test multiple problem sizes
- Compare against baselines (PyTorch, cuBLAS, etc.)
- Include graphs (matplotlib, seaborn)
- Report hardware used

### Documentation
- Architecture docs (explain design decisions)
- API reference (auto-generated)
- Tutorials (Jupyter notebooks)
- Performance optimization guide

---

## Part 6: Timeline & Milestones

### Month 1-2: Learning Toys
- Complete 3-4 toys from each category
- Focus on understanding fundamentals
- Build development environment

### Month 3-4: First Portfolio Project
- Choose P1 or P2
- Aim for production quality
- Public GitHub repo with good README

### Month 5-8: Advanced Project + Contributions
- Start P4, P5, or P6
- Begin open source contributions
- Engage with communities

### Month 9-12: Research Project (Optional)
- P8, P9, or P10 if aiming for research roles
- Or focus on substantial open source contributions
- Consider writing blog posts about your work

### Ongoing
- Maintain projects (respond to issues)
- Keep learning new techniques
- Expand portfolio based on job targets

---

## Part 7: Showcasing Your Work

### GitHub Profile
- Pin your best 4-6 projects
- Consistent commit history (shows dedication)
- Detailed READMEs with badges (build status, etc.)
- Star/watch relevant repos (shows engagement)

### Blog Posts
Platforms: Medium, Dev.to, personal blog
- "Optimizing GEMM: A Journey from 10 GFLOPS to 1 TFLOPS"
- "Building a Custom PyTorch Operator: A Complete Guide"
- "Understanding MLIR: Implementing a Custom Dialect"
- "How I Contributed to [Major Project]"

### Talks & Presentations
- Local meetups
- Conference lightning talks
- YouTube tutorials (screen recordings)

### Portfolio Website
Simple site with:
- Project showcase
- Technical blog
- About/CV
- Links to GitHub, LinkedIn

---

This document should keep you busy for 12-18 months! Start with the learning toys to build confidence, then tackle portfolio projects that align with your target companies. Remember: quality over quantity—2-3 really polished projects beat 10 mediocre ones.