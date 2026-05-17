# Self-Supervised Vision Lab

一个可部署到 Streamlit Cloud 的图像自监督学习演示项目，包含两个轻量实验：

- **旋转预测**：把图像旋转为 0 / 90 / 180 / 270 度，训练 CNN 预测旋转角度。
- **遮挡重建**：简化 MAE 思路，对图像 patch 做遮挡，训练小型卷积自编码器重建被遮挡区域。

应用会可视化原图、变换或遮挡后的图像、训练前后模型输出，并展示 loss / accuracy 曲线。页面还包含三类简单对比：不同数据增强方式、不同遮挡比例、训练前后效果。数据由程序即时合成，不需要额外下载数据集。

`runtime.txt` 固定为 Python 3.11，方便 Streamlit Cloud 安装 PyTorch CPU 版本。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 上传到 GitHub

```bash
git init
git add .
git commit -m "Add self-supervised vision Streamlit demo"
git branch -M main
git remote add origin https://github.com/<your-name>/<your-repo>.git
git push -u origin main
```

把 `<your-name>` 和 `<your-repo>` 换成你的 GitHub 用户名和仓库名。

## 部署到 Streamlit Cloud

1. 打开 Streamlit Cloud，选择 **New app**。
2. 连接你的 GitHub 仓库。
3. Branch 选择 `main`。
4. Main file path 填 `app.py`。
5. 点击 **Deploy**。

## 课堂展示建议

- 旋转预测页：对比“基础”和“轻增强 / 强增强”的 accuracy 曲线。
- 旋转预测页：查看“训练前后效果对比”表，观察样本准确率和平均置信度变化。
- 遮挡重建页：对比低遮挡比例和高遮挡比例的 reconstruction loss，并查看训练前后 loss 下降。
- 用较小数据量和 4-8 个 epoch 做快速演示；需要更平滑曲线时再增加训练轮数。
