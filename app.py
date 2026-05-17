import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F


st.set_page_config(
    page_title="Self-Supervised Vision Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1220px;
    }
    [data-testid="stMetric"] {
        background: #f6f7f2;
        border: 1px solid #d9ded0;
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
    }
    div[data-testid="stCaptionContainer"] {
        color: #58616b;
    }
    .app-subtitle {
        color: #4a5562;
        font-size: 1.02rem;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }
    .status-pill {
        display: inline-block;
        border: 1px solid #ccd6dd;
        border-radius: 999px;
        padding: 0.15rem 0.55rem;
        margin-right: 0.35rem;
        background: #fbfcfd;
        color: #334155;
        font-size: 0.82rem;
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


DEVICE = torch.device("cpu")
ROTATION_LABELS = ["0°", "90°", "180°", "270°"]


@dataclass
class RotationResult:
    history: pd.DataFrame
    sample_original: torch.Tensor
    sample_transformed: torch.Tensor
    sample_logits_before: torch.Tensor
    sample_logits_after: torch.Tensor
    sample_labels: torch.Tensor


@dataclass
class MaskedResult:
    history: pd.DataFrame
    original: torch.Tensor
    masked: Dict[str, torch.Tensor]
    before: Dict[str, torch.Tensor]
    after: Dict[str, torch.Tensor]
    masks: Dict[str, torch.Tensor]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@st.cache_data(show_spinner=False)
def make_synthetic_dataset(count: int, image_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    images = []

    palettes = [
        ((24, 43, 58), (242, 174, 114), (77, 171, 145)),
        ((42, 50, 75), (116, 185, 255), (255, 204, 92)),
        ((34, 84, 61), (236, 112, 99), (247, 220, 111)),
        ((83, 65, 91), (103, 198, 227), (246, 151, 134)),
        ((55, 74, 97), (175, 207, 121), (248, 188, 84)),
    ]

    for idx in range(count):
        base, accent, secondary = palettes[idx % len(palettes)]
        img = Image.new("RGB", (image_size, image_size), base)
        pixels = np.asarray(img).astype(np.float32)

        yy = np.linspace(0.0, 1.0, image_size, dtype=np.float32)[:, None]
        xx = np.linspace(0.0, 1.0, image_size, dtype=np.float32)[None, :]
        glow = 28.0 * (0.65 * xx + 0.35 * yy)
        pixels = np.clip(pixels + glow[..., None], 0, 255)
        noise = rng.normal(0, 7.5, size=pixels.shape)
        pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(pixels)
        draw = ImageDraw.Draw(img, "RGBA")

        cx = int(rng.integers(image_size // 4, image_size * 3 // 4))
        cy = int(rng.integers(image_size // 4, image_size * 3 // 4))
        radius = int(rng.integers(image_size // 8, image_size // 4 + 1))

        shape_kind = idx % 4
        if shape_kind == 0:
            bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
            draw.ellipse(bbox, fill=accent + (230,), outline=(255, 255, 255, 120), width=2)
            draw.rectangle(
                [2, image_size - 8, image_size // 2, image_size - 4],
                fill=secondary + (210,),
            )
        elif shape_kind == 1:
            w = int(rng.integers(image_size // 4, image_size // 2))
            h = int(rng.integers(image_size // 5, image_size // 2))
            x0 = int(np.clip(cx - w // 2, 1, image_size - w - 1))
            y0 = int(np.clip(cy - h // 2, 1, image_size - h - 1))
            draw.rounded_rectangle(
                [x0, y0, x0 + w, y0 + h],
                radius=3,
                fill=accent + (225,),
                outline=secondary + (210,),
                width=2,
            )
            draw.line([x0, y0 + h + 3, x0 + w, y0 + h + 3], fill=(255, 255, 255, 120), width=2)
        elif shape_kind == 2:
            pts = [
                (cx, max(1, cy - radius - 2)),
                (max(1, cx - radius - 3), min(image_size - 2, cy + radius + 2)),
                (min(image_size - 2, cx + radius + 5), min(image_size - 2, cy + radius)),
            ]
            draw.polygon(pts, fill=accent + (230,), outline=(255, 255, 255, 120))
            draw.arc([3, 3, image_size - 5, image_size - 5], 200, 300, fill=secondary + (220,), width=2)
        else:
            for offset in range(-2, 3, 2):
                draw.line(
                    [4, cy + offset, image_size - 5, cy + offset + int(rng.integers(-5, 6))],
                    fill=accent + (190,),
                    width=2,
                )
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                outline=secondary + (235,),
                width=3,
            )

        marker_size = max(3, image_size // 8)
        corner = idx % 4
        marker_boxes = [
            [3, 3, 3 + marker_size, 3 + marker_size],
            [image_size - marker_size - 4, 3, image_size - 4, 3 + marker_size],
            [image_size - marker_size - 4, image_size - marker_size - 4, image_size - 4, image_size - 4],
            [3, image_size - marker_size - 4, 3 + marker_size, image_size - 4],
        ]
        draw.rectangle(marker_boxes[corner], fill=(245, 246, 238, 210))

        arr = np.asarray(img).astype(np.float32) / 255.0
        images.append(arr.transpose(2, 0, 1))

    return np.stack(images).astype(np.float32)


def to_tensor(images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(images).to(DEVICE)


def tensor_to_image(tensor: torch.Tensor, scale: int = 5) -> Image.Image:
    arr = tensor.detach().cpu().clamp(0, 1).numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    return img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)


def make_contact_sheet(
    rows: List[Tuple[str, torch.Tensor]],
    columns: int = 4,
    scale: int = 5,
    pad: int = 12,
) -> Image.Image:
    if not rows:
        return Image.new("RGB", (1, 1), "white")

    cells = [(label, tensor_to_image(img, scale)) for label, img in rows]
    cell_w, cell_h = cells[0][1].size
    label_h = 22
    total_rows = math.ceil(len(cells) / columns)
    sheet_w = columns * cell_w + (columns + 1) * pad
    sheet_h = total_rows * (cell_h + label_h) + (total_rows + 1) * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (250, 251, 249))
    draw = ImageDraw.Draw(sheet)

    for idx, (label, img) in enumerate(cells):
        row = idx // columns
        col = idx % columns
        x = pad + col * (cell_w + pad)
        y = pad + row * (cell_h + label_h + pad)
        sheet.paste(img, (x, y + label_h))
        draw.text((x, y), label, fill=(38, 48, 57))

    return sheet


def plot_curve(data: pd.DataFrame, y: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=150)
    for name, group in data.groupby("setting"):
        ax.plot(group["epoch"], group[y], marker="o", linewidth=2, label=name)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y.title())
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


class RotationNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 72, kernel_size=3, padding=1),
            nn.BatchNorm2d(72),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(72, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.classifier(x)


class MaskReconstructor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, masked_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([masked_image, mask], dim=1)
        return self.decoder(self.encoder(x))


def rotate_batch(images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    rotated = [torch.rot90(img, int(label.item()), dims=(1, 2)) for img, label in zip(images, labels)]
    return torch.stack(rotated, dim=0)


def augment_for_rotation(images: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "基础":
        return images

    batch = images.shape[0]
    out = images.clone()
    brightness = torch.empty(batch, 1, 1, 1, device=out.device).uniform_(0.86, 1.16)
    contrast = torch.empty(batch, 1, 1, 1, device=out.device).uniform_(0.88, 1.12)
    mean = out.mean(dim=(2, 3), keepdim=True)
    out = (out - mean) * contrast + mean
    out = out * brightness
    out = out + torch.randn_like(out) * 0.025

    if mode == "强增强":
        cut_size = max(4, images.shape[-1] // 5)
        for idx in range(batch):
            y = torch.randint(0, images.shape[-2] - cut_size + 1, (1,)).item()
            x = torch.randint(0, images.shape[-1] - cut_size + 1, (1,)).item()
            out[idx, :, y : y + cut_size, x : x + cut_size] = 0.45
        out = out + torch.randn_like(out) * 0.045

    return out.clamp(0, 1)


def train_rotation_setting(
    data: torch.Tensor,
    setting: str,
    epochs: int,
    batch_size: int,
    lr: float,
    sample_count: int,
) -> Tuple[pd.DataFrame, RotationNet, torch.Tensor]:
    model = RotationNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rows = []
    n = data.shape[0]

    for epoch in range(1, epochs + 1):
        order = torch.randperm(n)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        model.train()
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            batch = data[idx]
            labels = torch.randint(0, 4, (batch.shape[0],), device=DEVICE)
            rotated = rotate_batch(batch, labels)
            rotated = augment_for_rotation(rotated, setting)

            logits = model(rotated)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.shape[0]
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += batch.shape[0]

        rows.append(
            {
                "epoch": epoch,
                "loss": total_loss / total_seen,
                "accuracy": total_correct / total_seen,
                "setting": setting,
            }
        )

    sample_images = data[:sample_count]
    return pd.DataFrame(rows), model, sample_images


def run_rotation_experiment(
    data: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    compare_mode: str,
) -> RotationResult:
    set_seed(seed)
    settings = ["基础", compare_mode]
    histories = []
    sample_count = 8
    sample_labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=DEVICE)[:sample_count]
    sample_original = data[:sample_count]
    sample_transformed = rotate_batch(sample_original, sample_labels)

    before_model = RotationNet().to(DEVICE).eval()
    with torch.no_grad():
        logits_before = before_model(sample_transformed)

    trained_models = {}
    for setting in settings:
        history, model, _ = train_rotation_setting(
            data=data,
            setting=setting,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            sample_count=sample_count,
        )
        histories.append(history)
        trained_models[setting] = model.eval()

    best_setting = pd.concat(histories).sort_values(["epoch", "accuracy"]).groupby("setting").tail(1)
    selected_setting = best_setting.sort_values("accuracy", ascending=False)["setting"].iloc[0]
    with torch.no_grad():
        logits_after = trained_models[selected_setting](sample_transformed)

    return RotationResult(
        history=pd.concat(histories, ignore_index=True),
        sample_original=sample_original,
        sample_transformed=sample_transformed,
        sample_logits_before=logits_before,
        sample_logits_after=logits_after,
        sample_labels=sample_labels,
    )


def make_patch_mask(
    batch_size: int,
    image_size: int,
    patch_size: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    grid = image_size // patch_size
    patches = grid * grid
    masked_patches = max(1, min(patches - 1, int(round(patches * mask_ratio))))
    mask = torch.zeros(batch_size, 1, image_size, image_size, device=device)

    for idx in range(batch_size):
        ids = torch.randperm(patches, device=device)[:masked_patches]
        for patch_id in ids:
            patch_id_int = int(patch_id.item())
            row = patch_id_int // grid
            col = patch_id_int % grid
            y0 = row * patch_size
            x0 = col * patch_size
            mask[idx, :, y0 : y0 + patch_size, x0 : x0 + patch_size] = 1.0

    return mask


def apply_mask(images: torch.Tensor, mask: torch.Tensor, fill_value: float = 0.5) -> torch.Tensor:
    return images * (1.0 - mask) + mask * fill_value


def masked_mse(reconstruction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(target)
    return (((reconstruction - target) ** 2) * expanded).sum() / expanded.sum().clamp_min(1.0)


def train_mask_setting(
    data: torch.Tensor,
    ratio: float,
    label: str,
    epochs: int,
    batch_size: int,
    lr: float,
    patch_size: int,
) -> Tuple[pd.DataFrame, MaskReconstructor]:
    model = MaskReconstructor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rows = []
    n = data.shape[0]
    image_size = data.shape[-1]

    for epoch in range(1, epochs + 1):
        order = torch.randperm(n)
        total_loss = 0.0
        total_seen = 0

        model.train()
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            batch = data[idx]
            mask = make_patch_mask(batch.shape[0], image_size, patch_size, ratio, DEVICE)
            masked = apply_mask(batch, mask)
            pred = model(masked, mask)
            loss = masked_mse(pred, batch, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.shape[0]
            total_seen += batch.shape[0]

        rows.append(
            {
                "epoch": epoch,
                "loss": total_loss / total_seen,
                "setting": label,
                "mask_ratio": ratio,
            }
        )

    return pd.DataFrame(rows), model


def run_masked_experiment(
    data: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    low_ratio: float,
    high_ratio: float,
    patch_size: int,
) -> MaskedResult:
    set_seed(seed)
    ratio_labels = {
        f"遮挡 {int(low_ratio * 100)}%": low_ratio,
        f"遮挡 {int(high_ratio * 100)}%": high_ratio,
    }
    histories = []
    models = {}
    sample = data[:4]
    image_size = data.shape[-1]

    untrained = MaskReconstructor().to(DEVICE).eval()
    masked_samples = {}
    masks = {}
    before = {}
    after = {}

    for label, ratio in ratio_labels.items():
        mask = make_patch_mask(sample.shape[0], image_size, patch_size, ratio, DEVICE)
        masked = apply_mask(sample, mask)
        masked_samples[label] = masked
        masks[label] = mask
        with torch.no_grad():
            before[label] = untrained(masked, mask)

        history, model = train_mask_setting(
            data=data,
            ratio=ratio,
            label=label,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patch_size=patch_size,
        )
        histories.append(history)
        models[label] = model.eval()

        with torch.no_grad():
            after[label] = models[label](masked, mask)

    return MaskedResult(
        history=pd.concat(histories, ignore_index=True),
        original=sample,
        masked=masked_samples,
        before=before,
        after=after,
        masks=masks,
    )


def probability_table(logits: torch.Tensor, labels: torch.Tensor) -> pd.DataFrame:
    probs = logits.softmax(dim=1).detach().cpu().numpy()
    rows = []
    for idx, row in enumerate(probs):
        pred = int(row.argmax())
        rows.append(
            {
                "样本": idx + 1,
                "真实旋转": ROTATION_LABELS[int(labels[idx].item())],
                "预测旋转": ROTATION_LABELS[pred],
                "置信度": float(row[pred]),
            }
        )
    return pd.DataFrame(rows)


def render_rotation_tab(data: torch.Tensor, controls: Dict[str, float]) -> None:
    st.subheader("旋转预测")
    st.markdown(
        '<span class="status-pill">Transform SSL</span>'
        '<span class="status-pill">4-way classification</span>'
        '<span class="status-pill">augmentation comparison</span>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.66, 0.34], gap="large")
    with right:
        compare_mode = st.selectbox("对比设置", ["轻增强", "强增强"], index=0)
        run = st.button("训练旋转预测", type="primary", use_container_width=True)

    if run or "rotation_result" not in st.session_state:
        with st.spinner("正在训练旋转预测模型..."):
            st.session_state.rotation_result = run_rotation_experiment(
                data=data,
                epochs=int(controls["epochs"]),
                batch_size=int(controls["batch_size"]),
                lr=float(controls["lr"]),
                seed=int(controls["seed"]),
                compare_mode=compare_mode,
            )

    result: RotationResult = st.session_state.rotation_result

    final_rows = result.history.groupby("setting").tail(1).sort_values("accuracy", ascending=False)
    best = final_rows.iloc[0]

    with right:
        m1, m2 = st.columns(2)
        m1.metric("最佳准确率", f"{best['accuracy'] * 100:.1f}%")
        m2.metric("最终 loss", f"{best['loss']:.3f}")
        st.dataframe(
            final_rows[["setting", "loss", "accuracy"]].rename(
                columns={"setting": "设置", "loss": "Loss", "accuracy": "Accuracy"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    with left:
        rows = []
        for idx in range(result.sample_original.shape[0]):
            rows.append((f"Input {idx + 1}", result.sample_original[idx]))
            rows.append((f"Rot {ROTATION_LABELS[int(result.sample_labels[idx].item())]}", result.sample_transformed[idx]))
        st.image(make_contact_sheet(rows, columns=4, scale=4), use_container_width=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.pyplot(plot_curve(result.history, "loss", "Rotation loss"))
    with c2:
        st.pyplot(plot_curve(result.history, "accuracy", "Rotation accuracy"))

    b1, b2 = st.columns(2, gap="large")
    with b1:
        st.caption("训练前")
        st.dataframe(probability_table(result.sample_logits_before, result.sample_labels), use_container_width=True, hide_index=True)
    with b2:
        st.caption("训练后")
        st.dataframe(probability_table(result.sample_logits_after, result.sample_labels), use_container_width=True, hide_index=True)


def render_masked_tab(data: torch.Tensor, controls: Dict[str, float]) -> None:
    st.subheader("遮挡重建")
    st.markdown(
        '<span class="status-pill">MAE-style</span>'
        '<span class="status-pill">patch masking</span>'
        '<span class="status-pill">ratio comparison</span>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.66, 0.34], gap="large")
    with right:
        low_ratio = st.slider("低遮挡比例", 0.10, 0.70, 0.35, 0.05)
        high_ratio = st.slider("高遮挡比例", 0.20, 0.85, 0.65, 0.05)
        if high_ratio <= low_ratio:
            high_ratio = min(0.85, low_ratio + 0.10)
            st.info(f"高遮挡比例已调整为 {high_ratio:.2f}")
        run = st.button("训练遮挡重建", type="primary", use_container_width=True)

    if run or "masked_result" not in st.session_state:
        with st.spinner("正在训练遮挡重建模型..."):
            st.session_state.masked_result = run_masked_experiment(
                data=data,
                epochs=int(controls["epochs"]),
                batch_size=int(controls["batch_size"]),
                lr=float(controls["lr"]),
                seed=int(controls["seed"]) + 19,
                low_ratio=float(low_ratio),
                high_ratio=float(high_ratio),
                patch_size=int(controls["patch_size"]),
            )

    result: MaskedResult = st.session_state.masked_result
    final_rows = result.history.groupby("setting").tail(1).sort_values("loss")
    best = final_rows.iloc[0]

    with right:
        m1, m2 = st.columns(2)
        m1.metric("最低重建 loss", f"{best['loss']:.4f}")
        m2.metric("最佳设置", str(best["setting"]))
        st.dataframe(
            final_rows[["setting", "mask_ratio", "loss"]].rename(
                columns={"setting": "设置", "mask_ratio": "遮挡比例", "loss": "Loss"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    with left:
        chosen = st.radio("可视化设置", list(result.masked.keys()), horizontal=True)
        rows = []
        for idx in range(result.original.shape[0]):
            rows.extend(
                [
                    (f"Input {idx + 1}", result.original[idx]),
                    ("Masked", result.masked[chosen][idx]),
                    ("Before", result.before[chosen][idx]),
                    ("After", result.after[chosen][idx]),
                ]
            )
        st.image(make_contact_sheet(rows, columns=4, scale=4), use_container_width=True)

    st.pyplot(plot_curve(result.history, "loss", "Masked reconstruction loss"))


def render_project_tab() -> None:
    st.subheader("项目结构")
    st.code(
        """.
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
└── .streamlit/
    └── config.toml""",
        language="text",
    )
    st.markdown(
        """
这个应用包含两个自监督学习小实验：

- 旋转预测：模型根据旋转后的图像预测 0、90、180、270 度。
- 遮挡重建：模型只看到被遮挡的图像，并尝试重建被遮挡区域。

数据由程序即时生成，部署后不需要下载数据集。默认训练规模较小，适合课堂展示、GitHub README 截图和 Streamlit Cloud 演示。
"""
    )


def sidebar_controls() -> Dict[str, float]:
    st.sidebar.header("实验参数")
    seed = st.sidebar.number_input("随机种子", min_value=1, max_value=9999, value=7, step=1)
    dataset_size = st.sidebar.slider("合成图像数量", 64, 512, 192, 32)
    epochs = st.sidebar.slider("训练轮数", 2, 20, 6, 1)
    batch_size = st.sidebar.select_slider("Batch size", options=[16, 32, 64], value=32)
    lr = st.sidebar.select_slider("学习率", options=[0.0005, 0.001, 0.002, 0.003], value=0.001)
    patch_size = st.sidebar.select_slider("遮挡 patch", options=[4, 8, 16], value=8)

    return {
        "seed": int(seed),
        "dataset_size": int(dataset_size),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "patch_size": int(patch_size),
    }


def main() -> None:
    st.title("Self-Supervised Vision Lab")
    st.markdown('<div class="app-subtitle">旋转预测与 MAE 风格遮挡重建的轻量可视化示例</div>', unsafe_allow_html=True)

    controls = sidebar_controls()
    set_seed(int(controls["seed"]))
    data_np = make_synthetic_dataset(int(controls["dataset_size"]), image_size=32, seed=int(controls["seed"]))
    data = to_tensor(data_np)

    tabs = st.tabs(["旋转预测", "遮挡重建", "部署说明"])
    with tabs[0]:
        render_rotation_tab(data, controls)
    with tabs[1]:
        render_masked_tab(data, controls)
    with tabs[2]:
        render_project_tab()


if __name__ == "__main__":
    main()
