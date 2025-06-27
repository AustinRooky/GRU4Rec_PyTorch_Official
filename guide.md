1. 项目总览  
    GRU4REC_PYTORCH_OFFICIAL/          ← 根目录  
        img/                           ← 论文里提到的训练速度对比图等静态资源  
        paramfiles/                    ← 已经调好的超参数配置（.py）  
        paramspaces/                   ← Optuna 搜参用的搜索空间定义（.json）  
        .gitignore                     ← Git 版本控制忽略项  
        evaluation.py                  ← 统一的离线评测入口（Recall/MRR 等）  
        gru4rec_pytorch.py             ← **核心模型实现**（GRU4Rec 类）  
        paropt.py                      ← 调参脚本，内部调用 Optuna  
        README.md                      ← 使用说明（你刚贴出的长文）  
        run.py                         ← **训练 / 推理 / 保存 / 加载** 的命令行封装  
        (其余零散脚本略)

2. 阅读顺序建议  
    · 先过一遍 README.md → 搞清楚作者在 PyTorch 版本里做的改动、命令行用法。  
    · 打开 run.py  
        - 看 `argparse` 部分知道有哪些 CLI 入口参数；  
        - 顺着 `main()` 看训练流程：数据加载 → 建模 → 训练 → 保存 / 评测。  
    · 再读 gru4rec_pytorch.py  
        - 关注 `GRU4Rec` 类：  
            · `__init__` 里完成网络层、采样缓存等组件构造；  
            · `forward()` 里是真正的前向计算（嵌入 → GRU → FC）；  
            · `loss_fn_*` 系列是两种官方损失（cross-entropy & BPR-max）的实现。  
    · evaluation.py  
        - `batch_eval()`/`evaluate_gpu()` 封装了 Recall/MRR 的批量打分逻辑；  
        - 搞清楚三种并列的 tiebreak 方式（standard / conservative / no_dedupe）。  
    · paropt.py  
        - 学习 Optuna Study 怎么包装进脚本；  
        - 搜参时固定与搜索的超参通过 `-fp` / `-opf` 分离。  
    · paramfiles / paramspaces  
        - paramfiles：直接 `python run.py -pf xxx.py` 即可复现作者的最佳设置；  
        - paramspaces：查看 JSON 理解搜索范围、步长、分布。  

3. 运行 & 调试套路  
    （1）最小化试跑  
        ```bash
        python run.py data/train.tsv -t data/test.tsv -ps layers=100,n_epochs=1 -d cuda:0
        ```
        先跑通流程，确认数据格式无误。  

    （2）正式训练  
        · 将数据按 README 里给的三列字段命名：`SessionId ItemId Time`  
        · GPU 训练示例：  
        ```bash
        python run.py data/train.tsv \
            -t data/test.tsv \
            -ps layers=224,batch_size=80,loss=bpr-max,n_epochs=10 \
            -d cuda:0 -s model.pt
        ```  
        用现有的参数

        ```python
            python run.py data/train.tsv \
            -pf paramfiles/yoochoose_xe_shared_best.py \
            -t data/test.tsv \
            -m 1 5 10 20 \
            -d cuda:0 \
            -s model_yoochoose.pt
        ```


    训练好之后评估

        ```python
            python run.py model_yoochoose.pt \
            -l \
            -t data/test.tsv \
            -m 1 5 10 20 \
            -e conservative \
            -d cuda:0 
        ```

    （3）调参  
        ```bash
        python paropt.py data/train_tr.tsv data/train_valid.tsv \
            -opf paramspaces/rr_xe.json \
            -fp n_sample=2048,loss=cross-entropy,constrained_embedding=True \
            -pm mrr -nt 200 -d cuda:0
        ```  

4. 源码阅读小贴士  
    · gru4rec_pytorch.py 里的“采样缓存”(SampleStore) 与“梯度累积”逻辑是论文提升效率的关键，可单步调试理解 negative sampling 流程。  
    · 两种损失函数共用同一 `forward()`，差异全部体现在 `loss_fn_*`；这里作者顺便把 final activation（softmax / ELU）也绑定到 loss 上，避免跑错组合。  
    · run.py 把“数据预处理→模型→评测”串起来，阅读时可以利用 VS Code 的“调用层级(Call Hierarchy)”定位函数跳转关系。  

5. 常见疑惑速查  
    · **embedding=0 / constrained_embedding=True** ➜ 表示共享输入 & 输出嵌入矩阵；  
    · **n_sample** ➜ 负采样一次取多少个 item；  
    · **logq** ➜ 交叉熵损失里控制 logits 是否对数变换；  
    · **bpreg** ➜ BPR-max 中对正样本 logits 加权的 λ；  
    · **momentum** ➜ 在作者自定义 Adagrad+Nesterov 中才起作用；  
    · **dropout_p_embed / dropout_p_hidden** ➜ 输入嵌入 & GRU 隐层 Dropout 概率。  

这样顺着 README → 脚本入口 → 核心模型 → 评测 → 调参的顺序看，就能迅速搞清整仓库的代码结构与可用功能。

| 命令         | 监控内容           | 主要用途                      |
|--------------|--------------------|-------------------------------|
| htop         | CPU / 内存 / 进程  | 查看 CPU 负载、系统运行状态  |
| nvidia-smi   | GPU 显卡状态       | 查看显卡利用率、谁在用 GPU   |