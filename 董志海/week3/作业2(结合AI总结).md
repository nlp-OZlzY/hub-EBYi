# 四个意图分类模型介绍

## 1. TF-IDF + 机器学习模型 

- **基本原理**：使用TF-IDF向量化文本特征，配合传统机器学习模型进行分类
- **实现方式**：通过[jieba](file://D:\AIStudy\project\test\01-intent-classify\model\tfidf_ml.py#L4-L4)分词，去除停用词后构建向量空间模型
- **加载方式**：从[TfidfVectorizer](file://D:\AIStudy\project\test\01-intent-classify\model\tfidf_ml.py#L8-L8)和训练好的分类器模型
- **处理流程**：文本预处理 → TF-IDF向量化 → 模型预测
- **适用场景**：对速度要求高、资源有限的实时分类任务
- **优点**：
  - 速度快：训练和预测都非常高效，适合实时响应场景
  - 资源消耗低：内存和CPU使用较少
  - 可解释性强：基于关键词匹配，结果易于理解和调试
  - 稳定性好：对输入格式变化不敏感
- **缺点**：
  - 语义理解差：无法理解上下文和语义关系
  - 泛化能力弱：对未见过的词汇组合处理效果不佳
  - 依赖预处理：效果受分词质量和停用词过滤影响大
  - 特征工程复杂：需要大量人工调整

## 2. 大语言模型 + Prompt工程

- **基本原理**：利用大语言模型的语义理解能力，通过精心设计的提示词完成意图识别
- **实现方式**：使用[openai](file://D:\AIStudy\project\test\01-intent-classify\model\prompt.py#L4-L4)客户端调用远程LLM API
- **特色功能**：采用动态示例检索机制，通过TF-IDF找到相似训练样本作为参考
- **Prompt模板**：结合待选类别、历史例子和待识别文本构建完整提示词
- **适用场景**：需要高语义理解能力、对准确性要求极高的场景
- **优点**：
  - 语义理解强：能够理解复杂的语言结构和上下文
  - 泛化能力强：对新样本有很好的适应性
  - 无需训练：直接利用预训练知识
  - 动态提示：结合相似样本提升准确性
- **缺点**：
  - 成本高昂：API调用费用较高，特别是大规模请求时
  - 延迟较高：网络请求导致响应时间不稳定
  - 可控性差：可能出现幻觉或不一致的结果
  - 依赖外部服务：受API限制和网络状况影响

## 3. BERT模型 

- **基本原理**：使用预训练的BERT模型进行序列分类任务
- **模型架构**：基于[AutoTokenizer](file://D:\AIStudy\project\test\01-intent-classify\model\bert.py#L8-L8)和[BertForSequenceClassification](file://D:\AIStudy\project\test\01-intent-classify\model\bert.py#L10-L10)实现
- **数据处理**：通过[NewsDataset](file://D:\AIStudy\test\project\test\01-intent-classify\model\bert.py#L16-L26)和[DataLoader](file://D:\AIStudy\project\test\01-intent-classify\model\bert.py#L34-L34)进行批处理
- **推理过程**：模型评估模式下的批量预测
- **适用场景**：平衡准确性和部署可控性的企业级应用
- **优点**：
  - 语义理解优秀：深层双向注意力机制
  - 准确率高：在多种NLP任务上表现优异
  - 端到端训练：无需复杂的特征工程
  - 上下文感知：能够理解词语在句子中的含义
- **缺点**：
  - 计算资源需求高：需要GPU加速，内存占用大
  - 推理速度慢：相比传统ML方法速度较慢
  - 模型体积大：占用存储空间较多
  - 微调复杂：需要专业知识进行模型调优

## 4. 正则表达式规则 

- **基本原理**：基于预定义的关键词模式进行规则匹配
- **实现方式**：使用[re](file://D:\AIStudy\project\test\01-intent-classify\model\regex_rule.py#L2-L2)模块编译规则，按优先级匹配
- **配置方式**：从[REGEX_RULE](file://D:\AIStudy\project\test\01-intent-classify\model\regex_rule.py#L6-L6)配置字典加载各类别的匹配模式
- **处理逻辑**：逐条文本进行规则匹配，返回第一个匹配成功的类别
- **适用场景**：特定领域、模式固定的分类任务，或作为其他模型的补充
- **优点**：
  - 精确匹配：对特定模式的识别非常准确
  - 执行速度极快：正则匹配效率很高
  - 完全可控：规则明确，行为可预测
  - 零学习成本：不需要训练过程
- **缺点**：
  - 覆盖范围有限：只能处理预定义的模式
  - 维护困难：需要不断更新规则库
  - 灵活性差：难以处理变化的语言表达
  - 扩展性弱：新增类别需要大量规则编写