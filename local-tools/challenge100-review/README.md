# 100 段共享场景渲染：已执行源码快照

此目录归档实际执行的渲染器与校验器；没有重新生成动作、修改 IK、改变训练数据或重渲染。六个 Python 文件均与 PT 执行机文件逐字节 SHA256 匹配，原注释和历史入口保留。

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
