# DSA

## Sorting

### Quick Sort Operations

使用快速排序对序列a: 1,4,7,8,9,13,15,20,23和b: 9,4,7,13,1,8,23,20,15以第一个点为基点进行从小到大排序，序列b所需的操作更少。

正确。已经排序好的元素，每次都要把pivot去和后面所有的元素比较，1+2+...+8 = 36次比较。序列b反而更少（20）

# Python

### GIL

**Python 的全局解释器锁（GIL, Global Interpreter Lock）** 是 Python 解释器中的一个 **线程锁机制**，它限制了 **同一进程中的多个线程不能同时执行 Python 字节码**。这意味着：

- **Python 的多线程（threading）不能真正实现 CPU 并行计算**（即使有多个 CPU 核心）。
- **Python 的多进程（multiprocessing）可以绕过 GIL**，实现真正的并行计算。

Python 的 GIL 主要用于 **保证 CPython 解释器在多线程环境下的内存安全**：

- Python 采用 **引用计数（Reference Counting）** 来管理内存。
- 如果多个线程同时修改同一个对象的引用计数，可能会导致 **数据竞争（Race Condition）**，导致程序崩溃或内存泄漏。
- **GIL 通过加锁保证同一时刻只有一个线程修改对象引用计数**，从而避免数据竞争。

# Machine Learning

## Regularization

### Bias-Variance Tradeoff（偏差-方差权衡）

Bias-Variance Tradeoff 描述了机器学习模型误差的两个主要来源，即**偏差 (Bias)** 和 **方差 (Variance)**，它们之间存在一种平衡：

- **Bias（偏差）：**
   模型对真实关系（数据产生过程）的一种系统性的误差或偏离。
   通常偏差来源于模型过于简单，未能准确捕获数据中的潜在关系和规律（**underfitting**）。
   **特点：**
  - 模型越简单，偏差越高。
  - 无法捕获数据背后的复杂趋势。
- **Variance（方差）：**
   模型对训练数据的敏感程度，反映在不同训练集上模型估计值之间的波动大小。
   方差高通常意味着模型对训练数据的噪声过于敏感（**overfitting**）。
   **特点：**
  - 模型越复杂，方差越高。
  - 对数据变化敏感，导致泛化能力下降。

假设数据产生过程可描述为：

$Y=f(X)+ϵY = f(X) + \epsilon$

其中：

- Y 是我们要预测的变量。
- f(X)是实际的未知函数（数据真正产生的函数）。
- ϵ是噪声，具有均值为0，方差为 Var(ϵ)。

我们使用训练数据得到的模型 $\hat{f}(X) $来估计真实函数 $f(X)$。

我们的预测是：

$\hat{Y} = \hat{f}(X)$

我们关心预测误差的期望值：

$E\left[(Y - \hat{Y})^2\right]$

对这个表达式进行拆解：

首先将 Y 替换为真实函数加上噪声：

$Y = f(X) + \epsilon$

带入误差公式：

$E\left[(f(X) + \epsilon - \hat{f}(X))^2\right]$

由于$\epsilon$ 与估计函数$\hat{f}(X)$无关（噪声独立于估计），可进一步拆解为：

$E\left[(f(X)-\hat{f}(X))^2\right] + E[\epsilon^2] + 2E[\epsilon(f(X)-\hat{f}(X))]$

由于噪声$\epsilon $均值为0且与模型无关，第三项消失：

- $E[\epsilon] = 0$，因此$E[\epsilon(f(X)-\hat{f}(X))]=0$

则简化为：

$E\left[(Y - \hat{Y})^2\right] = E\left[(f(X)-\hat{f}(X))^2\right] + \text{Var}(\epsilon)$

其中 $\text{Var}(\epsilon) $称为**不可约误差（irreducible error）**，是我们无法控制的。

我们重点看可控制的部分：

$E\left[(f(X)-\hat{f}(X))^2\right]$

再进一步拆分：

- 记$\hat{f}(X)$的期望为$E[\hat{f}(X)]$。
- 有：

$E\left[(f(X)-\hat{f}(X))^2\right] = E\left[(\hat{f}(X)-E[\hat{f}(X)])^2\right] + (E[\hat{f}(X)] - f(X))^2$

因此我们得到完整的 **Bias-Variance 分解**公式：

$E\left[(Y-\hat{f}(X))^2\right] = \text{Var}(\hat{f}(X)) + [\text{Bias}(\hat{f}(X))]^2 + \text{Var}(\epsilon)$

总结该公式：

- $\text{Var}(\hat{f}(X))$： 模型的方差，描述模型预测随训练数据变化的波动程度。
- $\text{Bias}(\hat{f}(X))^2$： 模型的偏差，描述模型预测的期望与真实值的偏离程度。
- $\text{Var}(\epsilon)$：不可约误差，由数据本身的噪声产生，我们无法通过优化模型消除。

直观理解：

- 如果模型太简单（underfitting），会有较大的 **bias**，但低的方差。
- 如果模型太复杂（比如深层神经网络或高阶多项式回归），则方差很大，偏差小，但容易出现对数据的过拟合。
- 理想的模型是找到一个合适的复杂度，使得**偏差与方差之间取得平衡**，从而最小化整体误差。

**综上所述：**

- 模型设计过程本质上是在**bias与variance之间权衡**。
- 模型越复杂，bias降低但variance增高（可能过拟合）；越简单，variance降低，但bias增加。
- 选择适合的数据模型以实现误差最小化，是机器学习中最重要的过程之一。

## Tree-based Model

### Difference between Gradient Boosting DT and XGBoost

| **优化点**     | **XGBoost**           | **普通 GBM**           | **优势**           |
| -------------- | --------------------- | ---------------------- | ------------------ |
| **梯度计算**   | 二阶梯度（Hessian）   | 一阶梯度               | **更快收敛**       |
| **正则化**     | L1 + L2               | 无                     | **防止过拟合**     |
| **并行计算**   | 多线程 & GPU          | 串行                   | **训练速度提升**   |
| **缺失值处理** | 自动选择最优路径      | 需手动填充             | **提高数据鲁棒性** |
| **剪枝策略**   | 预剪枝（Pre-pruning） | 后剪枝（Post-pruning） | **减少无效分裂**   |

**计算优化**：XGBoost 采用 **二阶导数优化（Taylor Expansion）**，比普通 GBM 只用 **一阶梯度** 计算更快收敛。

**正则化**：XGBoost 添加了 **L1 & L2 正则项**，避免过拟合。

**并行计算**：XGBoost **支持多线程 & GPU 加速**，大大提升训练速度。

**缺失值处理**：XGBoost **自动处理缺失值**，无需手动填充。

**剪枝（Pruning）**：XGBoost 采用 **预剪枝（Pre-pruning）**，而普通 GBM 采用 **后剪枝（Post-pruning）**，XGBoost 计算更高效。

### Differences between RF and Bagging of trees

🌳 **Bagging (Bootstrap Aggregation)**

**Core Idea:**

- Build multiple decision trees by repeatedly sampling training data **with replacement** (bootstrap samples).
- Each tree is built independently using the entire feature set.
- Final prediction is usually the average (for regression) or majority vote (for classification) of all trees.

**Bagging Procedure:**

1. Randomly sample the training data **with replacement** to create many subsets.
2. Train a decision tree on each bootstrap sample.
3. Average or majority vote to produce the final prediction.

**Pros & Cons:**

- **Pros:** Reduces variance significantly; robust against overfitting.
- **Cons:** Trees may be correlated if there’s a strong feature dominating the splits.

🌲 **Random Forest**

**Core Idea (Improvement over Bagging):**

- Random forest adds an additional step on top of bagging:
  - At each split in each tree, only a **random subset of features** is considered for selecting the best split.
- Thus, randomness is introduced in two ways:
  - Bootstrap sampling of the training data.
  - Random selection of features at each split.

**Random Forest Procedure:**

1. Create bootstrap samples from the training set (same as bagging).
2. **When growing each tree, at every split**, randomly select only a subset (e.g., $\sqrt{p}$ for classification or $p/3$ for regression) of features, and choose the best split from these.
3. Average (regression) or majority vote (classification) to get the final prediction.

**Pros & Cons:**

- **Pros:** More effective at decorrelating trees, leading to even better variance reduction and improved prediction accuracy.
- **Cons:** Slightly more complexity due to the random feature selection step.

## Linear Model

### Difference And Similarity between Ridge and Lasso

Lasso（Least Absolute Shrinkage and Selection Operator）和 Ridge（岭回归）都是**线性回归的正则化方法**，主要用于防止模型过拟合并提高泛化能力。

**1. 相同点**

1. **都是正则化方法**：

   - 通过向损失函数中加入**惩罚项**来约束模型的参数，避免过拟合。

2. **都用于处理多重共线性问题**：

   - 当特征之间高度相关时，普通最小二乘回归（OLS）可能会导致不稳定的系数，Lasso 和 Ridge 能有效缓解这个问题。

3. **超参数控制正则化强度**：

   - 两者都引入一个超参数 

     α\alphaα

      来控制正则化的强度：

     - α\alphaα 较大 → 更强的正则化，模型更简单，偏差增大，方差减小。
     - α\alphaα 较小 → 更弱的正则化，模型更复杂，偏差减小，方差增大。

4. **损失函数结构类似**：

   - 都是**线性回归的损失函数 + 正则化项**。

**2. 不同点**

|                | **Lasso (L1 正则化)**                                    | **Ridge (L2 正则化)**                                    |
| -------------- | -------------------------------------------------------- | -------------------------------------------------------- |
| **正则化项**   | $\lambda \sum |w_i|$                                     | $\lambda \sum {w_i}^2$                                   |
| **作用**       | 产生**稀疏解**，可以用于**特征选择**                     | 不会让特征权重变成 0，适用于高维度但不需要特征选择的场景 |
| **系数收缩**   | 一些特征的权重会被强制降为 0（稀疏）                     | 所有权重都被缩小，但不会变成 0                           |
| **适用场景**   | 适用于特征冗余较多的情况（可以自动筛选特征）             | 适用于多重共线性但仍希望保留所有特征                     |
| **计算复杂度** | 由于 L1 范数的不可导性，需要用 **坐标下降法** 等优化算法 | L2 正则化是二次可导的，可直接用**梯度下降**              |

# Deep Learning

## Loss Function

### 多分类问题的损失函数

多分类问题的损失函数请给出L。假设y是batchsize\*dim, label是batchsize\*N.

分类问题主要分为以下几类，每种对应不同的损失函数：

| 分类类型               | 损失函数名称                     | 适用场景           | 公式                                                         |
| ---------------------- | -------------------------------- | ------------------ | ------------------------------------------------------------ |
| **二分类**             | Binary Cross Entropy             | 两个类别的分类问题 | $L = - \frac{1}{N} \sum_{i=1}^{N} \left( y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right)$ |
| **多分类**             | Categorical Cross Entropy        | 多类别分类问题     | $L = - \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(\hat{y}_{ij})$ |
| **多分类（标签独立）** | Sparse Categorical Cross Entropy | 用于稀疏标签       | 类似于 Categorical Cross Entropy，但使用索引形式的标签       |
| **多标签分类**         | Binary Cross Entropy             | 每个样本有多个标签 | 每个标签独立计算 BCE 进行优化                                |

那么就是y先加一层MLP，再softmax，最后套进公式。

## Double Descent

### 什么是双降现象？

**传统的偏差-方差权衡图（bias-variance tradeoff）** 中，模型复杂度提升时，测试误差先下降，后上升，最终导致 **过拟合（overfitting）**。

但在现代机器学习（尤其是 deep learning）中，出现了一个新的现象：

> **测试误差并不在过拟合点后持续上升，而是再次下降，形成“U”形后再下降 → 称为 Double Descent。**

第一个下降：模型足够复杂，可以拟合数据，误差降低；

上升：模型复杂度接近“**刚好能完全拟合训练集**”时（称为 **插值点/interpolation threshold**），测试误差反而升高；

**第二次下降**：模型复杂度继续增加，误差再次下降（deep learning 常常就处于这第二次下降之后）。

1. **Over-parameterization 有利于泛化**

现代深度网络中，参数数量远大于训练样本数量（典型 overparameterized）。但这并不一定导致坏结果。

- 训练误差 = 0（完全拟合）；
- 但模型学习到的是“低复杂度的函数” → 泛化能力强。

这与传统“越拟合越差”的观点相悖。

2. **Implicit Regularization（隐式正则化）**

> 即使没有显式加 L2 正则项，SGD、Adam 等优化器天然具有“倾向找简单解”的偏好。

例如：

- 在无数个完全拟合解中，SGD 会收敛到 “某种更平滑” 的函数。
- 这种平滑函数往往具有更强的泛化能力（Occam’s razor：简单即是好）。

3. **数据分布的高维结构 + 网络的归纳偏置（Inductive Bias）**

深度神经网络结构本身就带有“偏好某类函数”的倾向：

- 卷积层 → 偏好空间局部性（图像）；
- 自注意力 → 偏好序列建模（语言）；
- 残差结构 → 更易优化平滑函数。

这些归纳偏置 + 大量样本 + 合理初始化和训练策略，使得即使网络**过拟合训练集，也不容易过拟合噪声**。

### 双降现象出现的条件

Double Descent 只在高 Signal-to-Noise Ratio (SNR) 下才明显？是的。

**Double Descent 现象**在图像识别、语音识别这类 **高 SNR 的任务**中最常见：

- 数据噪声小（比如一张图大部分信息都是“有用的”）；
- 模型可以过拟合训练数据，但同时学到 general pattern；
- 在 interpolation 临界点之后，误差反而下降。

但在 **金融数据**里：

- **噪声远远大于信号**（SNR 极低）；
- 模型越复杂，越容易学到 market noise 而不是规律；
- 所以过拟合之后通常不会“再下降”，反而就崩了。

> 👉 结论：金融数据下不会明显看到 double descent，甚至会出现“单降+爆炸”，也就是模型复杂度增加，测试误差迅速上升。

### 为什么金融领域数据用 Regularized Model 比深度学习模型更好？

像 **Lasso、Ridge、ElasticNet** 或带正则项的线性/树模型：

- 有明确的结构假设（linear, additive, sparse）；
- 有很强的正则化；
- 在高噪声数据下 **抗过拟合能力强**；
- 能更稳定地提取信号。

> 在金融场景中，这类模型往往 **胜过 deep neural nets**，尤其在小样本、因子挖掘、跨资产建模中。

### 为什么RNN等模型在金融数据中不好？

📌 原因分析：

| 问题                                | 原因                                                         |
| ----------------------------------- | ------------------------------------------------------------ |
| **1. 序列没有长依赖性**             | 金融时间序列非常“短记忆”，今天的收益率和三天后的基本没关系，**不像语言或音频那样有长依赖结构**。RNN 的“记忆”在金融里没用武之地。 |
| **2. 样本极少**                     | RNN 的参数量远大，需要上万条样本。金融数据通常样本很少（尤其是股票个体维度），**很容易过拟合**。 |
| **3. 噪声大**                       | RNN 容易“记住”训练集的噪声模式而不是结构化信号，泛化性差。   |
| **4. 不平稳性（non-stationarity）** | 金融数据具有结构突变、跳变、回归失效等情况，RNN 不善于处理 regime switching（状态切换）问题。 |
| **5. 特征解释性差**                 | RNN 难以解释“为什么涨”，而金融更偏好**结构透明、可控性强**的模型。 |

✅ 所以用什么模型更合适？

| 场景              | 推荐模型                               |
| ----------------- | -------------------------------------- |
| 单因子/横截面建模 | 线性回归 / Lasso / GBDT                |
| 时间序列预测      | ARIMA / Prophet / XGBoost / LightGBM   |
| 高频交易          | 带滑窗的 ML / 强化学习（需要专门优化） |
| 跨资产预测        | Shallow NN + Feature selection         |
| 多因子模型        | PCA + Ridge / ElasticNet               |

## Regularization

### dropout层有什么用？其在训练和测试阶段有什么区别

Dropout 是一种**正则化**技术，旨在减少神经网络的过拟合（overfitting）。其核心思想是在**训练过程中**，以一定的概率$p$ 随机丢弃（置零）一部分神经元，使模型不会过度依赖某些特定的特征，而是学会更具鲁棒性的特征表示。

1. **训练阶段（Training Phase）**：
   - 对于每个神经元，以概率 $p$（一般设为 0.2~0.5）随机“丢弃”它，即将其输出置为 0。
   - 这样可以防止神经元之间的共适应（co-adaptation），迫使网络学习更通用的特征。
   - 剩下的激活神经元的输出会**按 1/(1-p) 进行缩放**，以保持输出的期望值不变。
2. **测试阶段（Inference Phase）**：
   - Dropout 层不会再丢弃任何神经元，而是让所有神经元都参与计算。
   - 由于训练时神经元的输出被缩放了 1/(1−p)，所以在测试阶段不需要额外缩放。

# LLM

## Scaling

### 模型怎么扩展

怎么把一个模型往上扩？比如7b的模型怎么扩成200b？

在实际中，扩展模型主要有以下几种方式：

| 扩展方式               | 描述                             | 优点             | 缺点                 | 适用场景                       |
| ---------------------- | -------------------------------- | ---------------- | -------------------- | ------------------------------ |
| **增加层数**           | 增加 Transformer Block 的数量    | 提升模型深度     | 训练变慢，梯度消失   | 需要更深层的抽象能力           |
| **增加宽度**           | 扩展隐藏层的维度和 FFN 层的大小  | 提升模型容量     | 计算量和显存需求暴增 | 大规模知识存储                 |
| **增加头数**           | 增加注意力头的数量               | 更好建模复杂关系 | 注意力计算量增加     | 更强的注意力建模               |
| **混合专家模型 (MoE)** | 使用多个专家网络，仅激活部分专家 | 大幅减少计算成本 | 通信和负载均衡困难   | 任务种类多的场景               |
| **多模态扩展**         | 添加图像、音频、视频等处理模块   | 支持跨模态任务   | 模型结构复杂化       | 多模态任务如视频推荐、图文检索 |

## PEFT

# DataBase

## MySQL

### Mysql有哪些索引类型

| **索引类型**                | **适用场景**                         | **存储引擎**           |
| --------------------------- | ------------------------------------ | ---------------------- |
| **PRIMARY KEY（主键索引）** | **唯一标识每一行数据**               | InnoDB, MyISAM         |
| **UNIQUE（唯一索引）**      | **防止重复值**                       | InnoDB, MyISAM         |
| **INDEX（普通索引）**       | **加速查询但不强制唯一**             | InnoDB, MyISAM         |
| **FULLTEXT（全文索引）**    | **搜索文本内容（LIKE 无法优化）**    | MyISAM, InnoDB（5.6+） |
| **SPATIAL（空间索引）**     | **地理信息（GIS 数据）查询**         | MyISAM, InnoDB         |
| **BTREE（B+树索引）**       | **默认索引结构，适用于范围查询**     | InnoDB, MyISAM         |
| **HASH（哈希索引）**        | **等值查询速度快，但不支持范围查询** | MEMORY, NDB            |

### Mysql实现原理

**MySQL 的实现原理可以从以下几个方面理解：**

1. **体系架构（MySQL 结构设计）**
2. **存储引擎（Storage Engine）**
3. **索引机制（Indexing）**
4. **查询执行流程（Query Execution）**
5. **事务管理（Transaction）**
6. **锁机制（Locking）**
7. **日志系统（Logging）**

MySQL 的架构可以分为三层：

1. 连接层（Connection Layer）
   - 负责处理用户连接、身份认证（Authentication）和权限管理（Authorization）。
   - 例如 `max_connections` 参数控制最大连接数。
2. 服务层（SQL Layer）
   - 负责 **SQL 解析（Parser）、优化（Optimizer）、执行（Execution）**。
   - 包含：
     - 查询优化器（Query Optimizer）
     - SQL 解析器（Parser）
     - 查询缓存（Query Cache）
     - 事务管理器（Transaction Manager）
3. 存储引擎层（Storage Engine Layer）
   - 负责数据存储和索引管理。
   - **不同存储引擎（如 InnoDB, MyISAM, Memory）提供不同的存储机制**。

**存储引擎**

**MySQL 存储引擎决定了数据存储的方式**。常见存储引擎：

| **存储引擎**           | **特点**                                             | **适用场景**                         |
| ---------------------- | ---------------------------------------------------- | ------------------------------------ |
| **InnoDB**             | **支持事务（ACID）**，默认存储引擎，行级锁，支持外键 | **OLTP（在线事务处理）**，高并发查询 |
| **MyISAM**             | **不支持事务**，只支持表级锁，查询速度快             | **OLAP（数据分析）**，读多写少的场景 |
| **Memory（内存存储）** | **数据存放在内存，访问极快**，但数据会丢失           | **缓存、临时表**                     |
| **NDB（集群存储）**    | **适用于分布式集群**，高可用                         | **高可用金融系统**                   |

**索引机制**

**MySQL 通过索引加速查询，常见的索引类型：**

| **索引类型**                | **特点**                                             |
| --------------------------- | ---------------------------------------------------- |
| **B+ 树索引（默认）**       | 适用于 **范围查询（>、<、BETWEEN）** 和 **等值查询** |
| **哈希索引（Memory 引擎）** | 适用于 **等值查询（=）**，但 **不支持范围查询**      |
| **全文索引（FULLTEXT）**    | 适用于 **全文搜索（MATCH ... AGAINST）**             |
| **空间索引（SPATIAL）**     | 适用于 **GIS（地理信息查询）**                       |

**查询执行流程**

当 MySQL 执行 `SELECT * FROM users WHERE age > 30;` 时，发生了什么？

**✅ 1. 查询解析（Parser）**

- SQL 语句会被解析为 **解析树（Parse Tree）**。
- 语法错误会在这里被检查。

**✅ 2. 查询优化（Optimizer）**

- 查询优化器（Query Optimizer）

   选择最优的执行计划：

  - 是否使用索引？
  - 是否需要全表扫描？
  - 是否可以优化 `JOIN` ？

```sql
EXPLAIN SELECT * FROM users WHERE age > 30;
```

🚀 **`EXPLAIN` 可用于查看 MySQL 查询执行计划**。

**✅ 3. 查询执行（Execution）**

- 根据优化后的查询计划，MySQL 访问存储引擎获取数据。
- **如果命中查询缓存（Query Cache），可以直接返回结果**。

**事务管理**

**事务（Transaction）** 是 MySQL **保证数据一致性的关键**，遵循 **ACID**：

| **属性**        | **描述**                             |
| --------------- | ------------------------------------ |
| **A（原子性）** | 事务不可分割，要么全部执行，要么回滚 |
| **C（一致性）** | 事务前后，数据库保持一致             |
| **I（隔离性）** | 事务之间相互独立                     |
| **D（持久性）** | 事务提交后，数据永久保存             |

**✅ InnoDB 事务示例**

```sql
START TRANSACTION;
UPDATE users SET balance = balance - 100 WHERE id = 1;
UPDATE users SET balance = balance + 100 WHERE id = 2;
COMMIT;
```

如果 `COMMIT` 之前发生错误，事务可以 `ROLLBACK`，保证一致性

**锁机制**

**MySQL 主要有两种锁：**

| **锁类型**           | **特点**                                    |
| -------------------- | ------------------------------------------- |
| **表级锁（MyISAM）** | 整张表被锁定，写性能高，适合 **读多写少**   |
| **行级锁（InnoDB）** | 只锁定特定行，提高并发，适合 **高并发事务** |

**✅ 行锁示例**

```sql
SELECT * FROM users WHERE id = 1 FOR UPDATE; -- 事务内锁住该行
```

🚀 **InnoDB 默认使用行锁，避免不必要的锁竞争**。

**日志系统**

MySQL 使用 **日志系统** 进行数据恢复：

| **日志类型**             | **作用**                     |
| ------------------------ | ---------------------------- |
| **Redo Log（重做日志）** | **崩溃恢复，保证事务持久化** |
| **Undo Log（回滚日志）** | **事务回滚，MVCC 支持**      |
| **Binary Log（Binlog）** | **主从复制，数据恢复**       |

**✅ Redo Log & Binlog**

- **Redo Log**：事务提交后写入磁盘，保证 **崩溃恢复**。
- **Binlog**：用于 **主从同步、数据恢复**。

```
SHOW BINLOG EVENTS;
```

### B-tree

**B-Tree 是一种广泛用于数据库和文件系统的自平衡树结构**，其核心特点是：

- **节点有多个子节点（不是二叉树的 2 个）**
- **数据存储在所有节点中（不像 Heap 只存根节点的最小/最大值）**
- **高度自平衡，减少磁盘 I/O（适用于数据库索引）**

