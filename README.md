# DiffusionDrive 外部方向指令融合框架 v1

当前交互页面的冻结版本：在 DiffusionDrive 导出的 20 条候选轨迹上，将模型分数与外部方向意图分数融合，并比较接受指令、拒绝指令及自主回退的结果。

## 查看页面

下载或克隆整个仓库，在浏览器中打开 [`index.html`](index.html)。请保留文件之间的相对目录；页面交互依赖 `app.js`，图表与下载入口需要其他附带文件。GitHub 的文件预览不会直接运行 HTML 页面。

人工标注入口：[`annotation.html`](annotation.html)。标注完成后应主动导出 JSON，浏览器本地存储不等于保存到仓库。

## 版本内容

- 16 个开发场景、每场景 20 条冻结候选轨迹。
- 原生 Sigmoid 模型分基线与 Softmax 对照。
- 融合评分、统一拒绝基准、拒绝后的自主回退。
- 当前场景下的 α–ρ 敏感性对照。
- 静态障碍几何检查和动态障碍恒速预测检查；历史冻结障碍检查及离线指标保留作对照。
- 112 条待人工判断的场景指令，当前尚无完成的独立接受／拒绝标签。

默认 α=0.7、ρ=0.75、方向尺度 σ=15° 均为尚未标定的开发参数。原生 Sigmoid 使用 T=1；Softmax 对照默认 T=2。历史 10° 方向容差仅用于旧规则对照，不是默认统一拒绝规则。

这是离线候选回放与方法开发页面，不执行新的 DiffusionDrive 推理或车辆控制。模型分数不是安全概率，环境检查通过也不等于闭环安全保证。当前版本没有最优参数或独立测试结论。

## 运行与重建范围

融合回放仅需 Python 标准库：

```bash
python fusion.py --scene S08 --angle 0 --normalization sigmoid --rejection-threshold 0.75
```

使用随附数据重新生成两个 HTML 页面：

```bash
python build_page.py
python build_annotation_page.py
```

完整数据重建与部分审计脚本仍依赖原始工作区中的上游数据、来源清单和本机路径。特别是 `calibration.py` 需要父目录的 `scene-choice-inventory-v3/inventory.json`，此仓库未包含该上游数据；因此标注页面可以独立使用，标定 CLI 尚不能在本仓库独立完成。重新计算几何检查需要 Shapely，分类绘图需要 Matplotlib，浏览器验证需要 Playwright 和适用的浏览器路径。

## 文件说明

| 文件 | 用途 |
|---|---|
| `index.html`、`app.js` | 交互式融合框架 |
| `data.json`、`protocol.json` | 冻结场景数据及协议 |
| `fusion.py`、`safety.py`、`runtime_adapter.py` | 融合、环境检查及适配代码 |
| `annotation.html`、`annotation_template.json` | 人工标注准备 |
| `calibration.py`、`标定说明.txt` | 参数选择程序及使用边界 |
| `results/`、`work/` | 随当前版本保存的结果与验证记录 |
| `figures/` | 场景分类与诊断图 |
| `manifest.json` | 74 个原始交付文件的 SHA-256 与字节数 |

阅读方法细节：[框架说明](框架说明.txt)、[标定说明](标定说明.txt)、[场景分类说明](场景分类说明.txt)。

原始交付文件按现有交付包逐文件校验后保存，未修改评分公式、候选数据或页面行为。本仓库额外添加了 README、Git 忽略规则与换行保留配置；`manifest.json` 记录的是原始交付文件。

此仓库为研究开发快照；未附带模型权重或完整上游数据，也未新增对第三方材料的再许可。
