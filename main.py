import io
import numpy as np
import cv2
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import base64
from fastapi.middleware.cors import CORSMiddleware
from torch.nn import functional as F

# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# MODEL ARCHITECTURE (SAME AS TRAINING)
# =========================================================

class DepthwiseSepConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.dw  = nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False)
        self.pw  = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Linear(channels, channels // reduction)
        self.fc2  = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w.view(b, c, 1, 1)


class LightConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1   = DepthwiseSepConv(in_c, out_c)
        self.conv2   = DepthwiseSepConv(out_c, out_c)
        self.se      = SEBlock(out_c)
        self.skip    = nn.Conv2d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()
        self.bn_skip = nn.BatchNorm2d(out_c) if in_c != out_c else nn.Identity()

    def forward(self, x):
        residual = self.bn_skip(self.skip(x))
        out      = self.conv2(self.conv1(x))
        out      = self.se(out)
        return F.relu(out + residual, inplace=True)


class ASPPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1,  dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=2,  dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=4,  dilation=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x  = self.fuse(torch.cat([x1, x2, x3], dim=1))
        x  = self.se(x)
        return x


class MultiTaskNet(nn.Module):
    def __init__(self, num_classes=4, num_views=3):
        super().__init__()

        # Encoder
        self.e1 = LightConvBlock(1,   64)
        self.e2 = LightConvBlock(64,  128)
        self.e3 = LightConvBlock(128, 256)
        self.e4 = LightConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ASPPBlock(512, 512)

        # View embedding
        self.view_emb = nn.Embedding(num_views + 1, 32)

        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(512 + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d4  = LightConvBlock(256 + 512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d3  = LightConvBlock(128 + 256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d2  = LightConvBlock(64 + 128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d1  = LightConvBlock(32 + 64, 32)

        self.seg_out = nn.Conv2d(32, 1, 1)
        self.aux_seg = nn.Conv2d(64, 1, 1)

    def forward(self, x, view):
        # ── Encoder ──────────────────────────────────────────────────
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        # ── Bottleneck ───────────────────────────────────────────────
        b = self.bottleneck(self.pool(e4))

        # ── Classification ───────────────────────────────────────────
        ve       = self.view_emb(view)                      # (B, 32)
        gap_feat = self.gap(b).view(b.size(0), -1)          # (B, 512)
        cls_raw  = self.cls_head(torch.cat([gap_feat, ve], dim=1))

        # ── Decoder ──────────────────────────────────────────────────
        d4 = self.d4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.d3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))

        # ── Segmentation outputs ─────────────────────────────────────
        seg_out = self.seg_out(d1)                          # (B, 1, 256, 256)

        # ✅ aux_seg reads from d2 (correct), upsampled to input size
        aux_seg = F.interpolate(
            self.aux_seg(d2),
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )                                                   # (B, 1, 256, 256)

        return cls_raw, seg_out, aux_seg
# =========================================================
# LOAD MODEL
# =========================================================

model = MultiTaskNet()
model.load_state_dict(torch.load("full_model_10M_weights.pt", map_location=device))
model.to(device)
model.eval()

# =========================================================
# LABEL MAP
# =========================================================

label_map = {
    0: "No Tumor",
    1: "Meningioma",
    2: "Glioma",
    3: "Pituitary"
}

# =========================================================
# PREPROCESS IMAGE (SAME AS TRAINING)
# =========================================================

def preprocess_image(image):

    image = cv2.resize(image, (256, 256))

    image = image.astype(np.float32)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)

    return torch.tensor(image, dtype=torch.float32).to(device)


# =========================================================
# YOUR BOUNDING BOX FUNCTION
# =========================================================

def get_tumor_bbox(mask, image=None, draw_box=False, return_crop=False):

    if len(mask.shape) == 3:
        mask = mask.squeeze()

    mask = mask.astype(np.uint8)

    if np.sum(mask) == 0:
        return None

    num_labels, labels = cv2.connectedComponents(mask)

    if num_labels <= 1:
        return None

    areas = [np.sum(labels == i) for i in range(1, num_labels)]
    largest_label = 1 + np.argmax(areas)
    largest_component = (labels == largest_label)

    ys, xs = np.where(largest_component)

    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()

    pad = 5
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(mask.shape[1], x_max + pad)
    y_max = min(mask.shape[0], y_max + pad)

    bbox = (x_min, y_min, x_max, y_max)
    results = [bbox]

    boxed_mask = None
    boxed_image_color = None
    cropped = None

    if draw_box and image is not None:

        mask_img = (mask * 255).astype(np.uint8)

        cv2.rectangle(mask_img, (x_min, y_min), (x_max, y_max), 255, 2)
        boxed_mask = mask_img

        img = image.copy()

        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        red = (0,0,255)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), red, 2)

        boxed_image_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results.append(boxed_mask)
        results.append(boxed_image_color)

    if return_crop and image is not None:
        cropped = image[y_min:y_max, x_min:x_max]
        results.append(cropped)

    return tuple(results)


# =========================================================
# IMAGE ENCODING (RETURN AS BASE64)
# =========================================================

def encode_image(img):

    if len(img.shape) == 2:
        success, buffer = cv2.imencode(".png", img)
    else:
        success, buffer = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return base64.b64encode(buffer).decode()


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    view: int = Form(...)
):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("L")
    image = np.array(image)
    image = cv2.resize(image, (256, 256))

    input_tensor = preprocess_image(image)

    view_tensor = torch.tensor([view], dtype=torch.long).to(device)

    with torch.inference_mode():

        cls_out, seg_out,_ = model(input_tensor, view_tensor)

        pred_class = torch.argmax(cls_out, dim=1).item()

        seg_prob = torch.sigmoid(seg_out)
        seg_mask = (seg_prob > 0.5).float()

    mask = seg_mask.cpu().numpy().squeeze()

    result = get_tumor_bbox(
        mask,
        image=image,
        draw_box=True,
        return_crop=True
    )

    if result is None:

        boxed_mask = mask
        boxed_img = image
        crop_img = None

    else:

        bbox, boxed_mask, boxed_img, crop_img = result
    sum=boxed_mask.sum()
    if(sum==0):
        pred_class=0
        boxed_mask = mask
        boxed_img = image
        crop_img = None

    response = {

        "prediction": label_map[pred_class],

        "generated_mask": encode_image((mask*255).astype(np.uint8)),

        "predicted_mask_with_box": encode_image(boxed_mask),

        "image_with_box": encode_image(boxed_img),

        "cropped_tumor": encode_image(crop_img) if crop_img is not None else None
    }

    return JSONResponse(response)