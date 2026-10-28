# 100 段共享场景渲染：已执行源码快照

此目录归档实际执行的渲染器与校验器；没有重新生成动作、修改 IK、改变训练数据或重渲染。六个渲染相关 Python 文件均与 PT 执行机文件逐字节 SHA256 匹配，原注释和历史入口保留；另附修正后的选片测量与排序复核源码。

## 入口与依赖边界

- `renderer/render_review.py`：本次百段入口，读取明确的 SOMA BVH、官方 G1 CSV 和生产优化 X2 CSV，按关节名字映射并核对输入及生产凭据。
- `renderer/legacy_scene.py`、`legacy_native.py`：复用的模型、SOMA 蒙皮、共享相机及绘制函数。其历史独立入口仍指向原来的 11 段及默认分支，**不是本次百段入口**。
- `renderer-executor/parallel_review.py`：续接控制器；核对已完成输出，按冻结的互斥任务清单执行剩余动作。
- `renderer-executor/render_review_threads4.py`：与串行百段渲染器相比，仅将 ffmpeg 线程参数由 8 改为 4；动作、相机、分道和画质代码相同。
- `renderer-executor/audit_render.py`：等待真实终止，核对精确选片覆盖、输入和视频哈希、ffprobe 参数、源帧索引，并检查四类代表视频的实际解码帧。

脚本保留执行时的 PT 绝对部署路径和环境依赖，不是可直接在任意目录运行的独立包。重放前必须隔离输出目录；不要直接运行历史入口或覆盖现有成品。依赖资产、输入动作、冻结任务清单、视频和数值凭据不随源码入库。

## 已验证结果与展示约定

本次 100 个视频全部通过媒体审计：1920×1080、30 FPS、H.264/yuv420p，共 23,035 个输出帧，对应 91,992 个原生源帧。一个真实视口、一块地面、原生 SOMA mesh、官方 G1 和生产 X2 同步显示；只使用整段固定水平分道，无高度补偿、逐帧根对齐或角色缩放。官方 G1 仅作固定世界 Z 轴 +90°根坐标显示转换，X2 不作额外基变换。

动作专属物体和地形几何没有随本批输入提供，视频中的统一平地不是真实楼梯或物体接触面。

- 冻结选片 SHA256：`0cd268a5436d7cedf538f7cf417d77571ec7fef5c7cbfe0776ca397bbadea778`
- PT 媒体审计：`/data/liuminghao/artifacts/soma-retargeter/agibot-x2ultra/challenge100-review/parallel-render/media-audit.json`
- 媒体审计 SHA256：`f5b296fe04e5411799ca5f21b2c836e2b1e515fd210762d16aab3b579fe4cad8`

## 执行源码 SHA256

| 文件 | SHA256 |
|---|---|
| `renderer/render_review.py` | `65293dbf06386bac589f7d4707912e0457fe8d0baec6abcb846b5b89d1de16dc` |
| `renderer/legacy_scene.py` | `47cb1a3e83b794a6690cbc67f8329c0a8d4dcf5088188c42314a2973f6759205` |
| `renderer/legacy_native.py` | `b7a857d3167b468e350553ecf4efd8d395fbdfa5ea147fbc68d25dd7471d5e0b` |
| `renderer-executor/parallel_review.py` | `fa03fc94c6d980d75c140518ab266888673934a46f722a33f9c9ae548f7cfb2c` |
| `renderer-executor/render_review_threads4.py` | `9c4c03fefb8e53e93d2efc20c99e8c9983ae597976edd32e28f9b14a68c7db5c` |
| `renderer-executor/audit_render.py` | `b02989c6f00c964f7e261f3aad90bcaa357a9190c135dbcd82a05a181b0e171b` |

归档检查保留了长编排函数、历史入口和少量未使用导入等非阻塞风格问题，没有为重构而改变已测试内容。编译检查不等于再次运行渲染；实际媒体结果以上述审计为准。

## 选片与指标版本

`selection/select_challenge100.py` 为修正后的测量／选片源码；`ranking-verification/verify_ranking.py` 为本次实际执行的候选复算及 selector AST 重放诊断。它们依赖 PT 已保存的元数据、候选缓存和脚本工件，不能把源码单独当作无依赖通用工具运行。

冻结清单保留原 11 段，共 100 个动作、97 个语义族；类型为杂技 7、手／肘支撑 8、地面起身 16、坐姿 8、地形／高差 18、物体交互 16、高速运动 10、舞蹈 8、罕见动作 9。按本次实际训练清单关联，56 保留、44 拒绝；这些标注不参与是否渲染的决定。

原手工 FK 曾把显式位置通道加到 OFFSET 上，而原生加载器使用替换语义。旧绝对足部基线因此错误，修正指标另存 `selection/corrected-kinematic-metrics.json`。对全部 506 个缓存候选重新计算、覆盖实际排序的 481 个候选后，发现额外平移在每段内为常量：五项实际排序输入保持不变。用实际 selector 重放，修正前后入选的 100 个成员完全相同，未舍入和 1e-8 舍入重放均无新增／移除。该证据支持本次名单不需更换，但不证明“全库唯一绝对最难 100”，也不提供接触真值。

- 排序复核：PT `selection/ranking-verification.json`，SHA256 `dfeaaac670ad6a80276392867ac0cdf85fda01dcc1016407f45161844910b608`。
- 审阅完成记录：PT `REVIEW_COMPLETE.json`，SHA256 `b58bdbfd071fe49050ee61de1c90f3cd8a675c7f7505d8038f09f945824eae2f`。
- 视频与离线网页包：PT `challenge100-videos.zip`，461,790,779 字节，100 个视频，SHA256 `0fee100bfea9f5427854009d215a542d08125dd1806437f606c249166e90fa40`。
- 审阅网站：`http://127.0.0.1:18767/`；清洗说明：`http://127.0.0.1:18767/cleaning.html`。

上述 PT 相对路径均基于 `/data/liuminghao/artifacts/soma-retargeter/agibot-x2ultra/challenge100-review/`。源码修复不重建当前训练数据；清洗说明明确区分执行版本、后续修复和未做的反事实重筛。
