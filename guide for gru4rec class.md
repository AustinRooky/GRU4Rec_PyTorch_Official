下面按模块解释这段 GRU4Rec 代码里最容易迷惑的部分，尤其是 **IndexedAdagradM** 里 momentum 的数学作用和实现位置。  
整段文字都放在“无格式的 Markdown 代码块”里，方便直接复制到编辑器阅读。  


  ▸ IndexedAdagradM（自定义优化器）  
    参数  
      lr         全局学习率（clr）  
      momentum   动量系数，0 表示不用动量  
      eps        防止除零的微小常数  
    状态  
      acc   每个参数的累积平方梯度 ∑g² —— Adagrad 的核心  
      mom   （可选）一阶动量，存历史更新方向  
    step() 核心逻辑  
      1 读取当前梯度 grad  
      2 更新累积平方梯度  
         acc ← acc + grad²  
      3 计算自适应分母  
         denom = √(acc + eps)  
      4 计算带动量的 Δw  
         if momentum>0:  
           mom ← momentum·mom + grad / denom  
           Δw = -lr · mom  
         else:  
           Δw = -lr · grad / denom  
      5 稀疏参数（Embedding）用 index_add_ 只更新梯度出现的行，  
        稠密参数直接 add_ / addcdiv_。  
    直观效果  
      • Adagrad 部分让出现频率高的参数步长自动变小，适合稀疏类别特征。  
      • momentum>0 时进一步平滑方向，等价给 Δw 加惯性，  
        可加快收敛、减少震荡。  

  ▸ init_parameter_matrix()  
    Xavier-uniform 初始化的手写版：  
      σ = √(6 / (fan_in + fan_out))  
      权重均匀分布在 [-σ, σ] 之间。  
    dim0_scale / dim1_scale 只是为了像论文里那样按 gate 数量缩放 fan 值  
    （GRU 三个门，所以乘 3）。  

  ▸ GRUEmbedding  
    把第 0 层 GRU“拆进” Embedding，避免显式循环：  
      r, z, h̃ 按公式一次性算完，速度比 torch.GRUCell 慢不了多少  
      但可以稀疏更新 Wx。  

  ▸ GRU4RecModel 三种输入方案  
      constrained_embedding=True  
        用 Wy 做 Item Embedding，输出层权重即输入向量（论文里的 tied embedding）。  
      embedding>0  
        用单独的稀疏 nn.Embedding 当输入 E，Wy 只做输出。  
      以上都 False  
        用上面的 GRUEmbedding 把 one-hot→隐向量，完全不建 Embedding 表。  

  ▸ SampleCache + SessionDataIterator  
    • 为 BPR-Max 或负采样交叉熵提前在 GPU 上缓存 neg items  
      按流行度 ^α 采样，加速大量 mini-batch。  
    • 迭代器按 Session 分批，yield (输入 item_i, [正样本+负样本])。  

  ▸ loss_function 选项  
      cross-entropy       —— 带 softmax 的分类交叉熵  
      bpr-max (+elu,bpreg)—— GRU4Rec 论文里提出的 rank loss 变体  
                             softmax_neg() 先对非目标项归一化，再用 sigmoid 差值。  

  ▸ momentum 超参数到底“干啥”  
      - 当 momentum=0   → 纯 Indexed-Adagrad（历史 g² 调步长）  
      - 0<momentum<1   → 在第 4 步把上一次的 Δw 乘 momentum 再加进来  
                         数学上是指数滑动平均，避免梯度方向抖动  
      - 典型值 0.9 或 0.5，调得越大惯性越强，可能更快也可能过冲  

  ▸ 怎样验证动量是否生效  
      可以在训练时打印 state['mom'] 的范数；  
      momentum=0 时范数应长期保持 0，>0 时会逐步累积。  