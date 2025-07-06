# 2025年专硕机器学习课程项目说明

## 任务截止时间

- **提交截止：2025年7月15日中午12点前**
- 提交方式：上传报告PDF到下列任意一个链接（以链接内提交时间为准，逾期视为未交）
  - [提交入口1](https://send2me.cn/5oE8thju/R-eDQGDeGdzoIg)
  - [提交入口2](https://send2me.cn/jbzcZCXI/SZ60PrPqAR6oUQ)

## 团队与学术规范

- 允许1~2人组队，报告需注明各作者贡献及研究领域
- 参考其他团队或网络资料，务必在参考文献注明
- 抄袭/未注明引用每处扣33分

## 问题背景

随着智能家居和物联网的发展，家庭用电监控与预测对于节能降耗、智能调度具有现实意义。通过多变量时间序列建模，不仅帮助居民合理用电，还能支持电网公司进行负荷预测与优化。

- 典型任务：根据家庭历史电力消耗、天气等多源信息，对未来总有功功率进行短期（90天）和长期（365天）预测。

## 数据集与特征

- **主数据集**：[UCI Individual household electric power consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
  - 时间跨度：2006/12~2010/11
  - 采样粒度：每分钟
  - **建议处理**：按天聚合为汇总数据

- **可选天气数据**：[法国气象局月度数据](https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles)

- **主要变量说明**
  - `global_active_power`：全屋有功功率（kW）
  - `global_reactive_power`：全屋无功功率（kW）
  - `voltage`：平均电压（V）
  - `global_intensity`：平均电流强度（A）
  - `sub_metering_1/2/3`：各分区域能耗（Wh）
  - `sub_metering_remainder`：剩余能耗，可通过公式推算
  - RR/气象类：降水/雾等天气数据（选做）

- **日聚合建议**
  - `global_active_power`/`global_reactive_power`/`sub_metering_1/2/3`：按天累加
  - `voltage`/`global_intensity`：按天平均
  - 天气：任选一条当天数据即可

## 预测任务与评价

- **目标**：用最近90天多变量序列，预测未来90天（短期）和365天（长期）每日总有功功率的变化曲线
  - **短期与长期需分别建模，不得共用参数**

- **评价指标**：均方误差（MSE）、平均绝对误差（MAE），每个模型各跑5轮，报告均值和标准差

## 基础与创新建模要求

本项目分三部分，各占总分1/3：

1. **LSTM预测模型**
2. **Transformer预测模型**
3. **创新改进模型**（结构不限，可自行组合如CNN+Transformer等，鼓励创新，结构新颖性优先，性能为次）

## 数据处理流程

1. 数据缺失正常，不影响预测任务，可合理补全或剔除
2. 按日聚合生成训练样本，每个样本为（输入+输出）天数组成的滑动窗口
3. 训练/测试集建议用`train.csv`和`test.csv`分开
4. **滑窗处理、步长、窗口设计等详见下列博客参考：**
   - [博客1](https://blog.csdn.net/qq_47885795/article/details/143462299)
   - [博客2](https://blog.csdn.net/weixin_39653948/article/details/105431099)
   - [博客3](https://datac.blog.csdn.net/article/details/105928752?fromshare=blogdetail&sharetype=blogdetail&sharerId=105928752&sharerefer=PC&sharesource=weixin_44709585&sharefrom=from_link)

## 报告提交规范

- 实验报告分四部分
  1. 问题介绍
  2. 模型说明（含伪代码/结构图）
  3. 结果与分析（含多轮实验均值、方差、曲线可视化、表格和截图）
  4. 讨论（含模型创新思路、表现分析、局限性及展望）

- 附：完整代码（**务必提供Github链接**），实验可视化截图

- **允许使用ChatGPT、DeepSeek等工具辅助写作，但请注明；参考文献不可缺少**

---

## 附加说明

- 数据和模型实现如遇不明之处，可与往届同学交流或参考指定博客
- 有关“滑动窗口”、“输入输出样本”、“步长”等概念务必理解透彻
- 鼓励对创新部分结构进行充分原理说明与有效性分析

---

> **注**：请严格按此说明完成课程项目，否则将影响最终评分。

## 环境配置

项目代码基于 Python 3.12 开发，依赖包列在 `requirements.txt` 中，可通过以下命令安装：

```bash
pip install -r requirements.txt
```

可通过以下命令运行：

```bash
python -m py_compile data_utils.py models.py train.py
```

深度学习框架使用 [PyTorch](https://pytorch.org/)。若需 GPU 训练请根据硬件选择合适的 CUDA 版本。