# iFlow 项目上下文文件

## 项目概述

**ellalgo-cpp** 是一个现代 C++ 实现的椭球算法（Ellipsoid Algorithm）库。椭球算法是一种用于线性规划的多项式时间算法，由 L. G. Khachiyan 在 1979 年首次引入。该算法使用椭球体迭代缩小线性规划的可行域，直到找到最优解。

该项目提供了椭球算法的现代 C++ 实现，支持多种功能：
- 并行切割（Parallel cut）
- 离散优化
- 传统和稳定版本
- 使用现代 CMake 实践
- 集成测试套件
- 持续集成和代码覆盖率

## 项目架构

该项目遵循模块化设计，主要目录结构如下：

- `include/ellalgo/` - 库的头文件
- `source/` - 库的源文件
- `test/` - 测试代码
- `bench/` - 基准测试
- `standalone/` - 独立可执行文件示例
- `documentation/` - 文档生成配置
- `cmake/` - CMake 模块和配置

## 核心组件

### 椭球体类 (Ell)

`include/ellalgo/ell.hpp` 定义了 `Ell` 类，表示椭球搜索空间：

```cpp
ell = {x | (x - xc)' mq^-1 (x - xc) ≤ κ}
```

该类负责定义和操作多维空间中的椭球体，通过切割平面更新椭球体。

### 切割平面方法

`include/ellalgo/cutting_plane.hpp` 实现了切割平面方法，包括：
- `cutting_plane_feas` - 求解凸可行性问题
- `cutting_plane_optim` - 求解凸优化问题
- `cutting_plane_optim_q` - 求解离散凸优化问题

### 配置和状态

`include/ellalgo/ell_config.hpp` 定义了选项、切割状态和相关信息结构。

### Oracle（预言机）

`include/ellalgo/oracles/` 目录包含不同的 Oracle 实现，如 `lmi_oracle.hpp`，用于处理线性矩阵不等式（LMI）问题。

## 构建和运行

### 全部一次性构建

```
cmake -S all -B build
cmake --build build
```

### 构建和运行测试

```
cmake -S test -B build/test
cmake --build build/test
CTEST_OUTPUT_ON_FAILURE=1 cmake --build build/test --target test
```

### 构建和运行独立目标

```
cmake -S standalone -B build/standalone
cmake --build build/standalone
./build/standalone/EllAlgo --help
```

## 开发惯例

- 使用 C++14 标准
- 遵循现代 CMake 实践
- 代码格式化使用 clang-format 和 cmake-format
- 依赖管理使用 CPM.cmake
- 支持多种静态分析工具和 sanitizer
- 包含完整的测试套件和代码覆盖率

## 关键技术

- C++14
- CMake 3.14+
- 椭球算法
- 凸优化
- 切割平面方法
- 线性矩阵不等式