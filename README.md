# Falcon Supplementary Website

《Trajectory Synthesis of a Three-Joint, Five-DOF Falcon-Inspired Flapping Mechanism by Sensitivity-Partitioned CMA-ES and SLSQP》的独立补充材料网站。

## 内容结构

- `app/`：网页内容、动画和统一视觉样式。
- `source_media/`：作者补充或替换的原始视频、图片与数据。
- `public/media/web/`：按照网站规范生成的视频与封面帧。
- `public/data/`：网页曲线直接读取的数据。
- `scripts/process-media.mjs`：统一转换视频。
- `scripts/build-og.ps1`：使用真实机构图和确定性排版生成分享封面。

素材投放位置和图像标准见 [source_media/README.md](source_media/README.md)。

## 常用操作

```text
npm run media   重新生成网站视频和封面帧
npm run og      重新生成分享封面
npm run dev     打开本地网站
npm run build   检查可发布版本
```

科学图像只来自项目真实材料；缺失内容不使用生成式图像代替。
