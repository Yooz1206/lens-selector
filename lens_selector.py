#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工业相机 / 镜头选型助手 —— 机器视觉「镜头选型四要素」法
================================================================

四要素（光学系统最基本的四个参数，见 README 引用）：
    ① 视野 FOV      要看多大范围          -> 决定放大倍率、所需像素数
    ② 工作距离 WD   镜头前端到物体距离    -> 与焦距共同决定 FOV
    ③ 景深 DOF      允许的物距变化量      -> 决定光圈与焦距的取舍
    ④ 分辨率        最小可分辨特征        -> 决定相机像素数与像元尺寸

用法：
    python lens_selector.py                       # 交互式问答
    python lens_selector.py --fov 100x80 --wd 400 --depth 5 --defect 0.1
    python lens_selector.py --demo                # 内置演示场景
    python lens_selector.py ... --sweep           # 附视野-像素需求对照
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import math
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# 常量与假设（全部集中在这里，便于按实际项目修改）
# --------------------------------------------------------------------------
LAMBDA_UM = 0.55          # 照明波长，绿光/白光常用 0.55 µm
STOPS = [1.4, 1.8, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0]
VERSION = "1.0"


# --------------------------------------------------------------------------
# 器材库（通用规格，代表同级产品；实际采购时按此规格对标具体型号）
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Camera:
    mid: str
    res_w: int
    res_h: int
    pixel_um: float
    shutter: str          # 全局 / 卷帘
    interface: str        # GigE / USB3
    tier: int             # 1 入门 … 4 高端（用于成本打分）
    note: str = ""

    @property
    def mp(self) -> float:
        return self.res_w * self.res_h / 1e6

    @property
    def sensor_w(self) -> float:          # mm
        return self.res_w * self.pixel_um / 1000.0

    @property
    def sensor_h(self) -> float:
        return self.res_h * self.pixel_um / 1000.0

    @property
    def diag(self) -> float:
        return math.hypot(self.sensor_w, self.sensor_h)

    @property
    def fmt(self) -> str:
        d = {"1/2": 8.0, "1/2.8": 6.46, "1/1.8": 8.93, "2/3": 11.0, "1/1.7": 9.5, "1": 16.0}
        best = min(d.items(), key=lambda kv: abs(kv[1] - self.diag))
        return best[0] + '"'


@dataclass(frozen=True)
class Lens:
    lid: str
    focal: float          # mm
    circle: float         # 像圈直径 mm
    mod: float            # 最小物距 mm
    fno_max: float        # 最大光圈（最小 f 数）
    mtf_lpmm: int         # 30% MTF 对应的空间频率 lp/mm（厂商标称级别）
    tier: int


@dataclass(frozen=True)
class Telecentric:
    tid: str
    mag: float
    circle: float
    wd: float
    dof: float
    tier: int = 4


CAMERAS: List[Camera] = [
    Camera("CAM-1.3MP", 1280, 1024, 4.80, "全局", "GigE", 1, "1/2\" 入门"),
    Camera("CAM-2MP", 1920, 1080, 2.90, "卷帘", "USB3", 1, "1/2.8\" 小像元"),
    Camera("CAM-3MP", 2048, 1536, 3.45, "全局", "GigE", 2, "1/1.8\""),
    Camera("CAM-5MP-A", 2448, 2048, 3.45, "全局", "GigE", 2, "2/3\""),
    Camera("CAM-5MP-B", 2592, 1944, 2.40, "卷帘", "USB3", 2, "1/1.8\" 小像元"),
    Camera("CAM-12MP", 4024, 3036, 1.85, "卷帘", "USB3", 3, "1/1.7\" 高分辨率"),
    Camera("CAM-20MP", 5472, 3648, 2.40, "卷帘", "USB3", 4, "1\" 大靶面"),
]

LENSES: List[Lens] = [
    Lens("8mm f/1.8", 8.0, 11.0, 100, 1.8, 100, 1),
    Lens("12mm f/1.8", 12.0, 11.0, 100, 1.8, 120, 1),
    Lens("16mm f/1.8", 16.0, 11.0, 100, 1.8, 120, 2),
    Lens("20mm f/1.8", 20.0, 11.0, 150, 1.8, 120, 2),
    Lens("25mm f/1.8", 25.0, 11.0, 150, 1.8, 120, 2),
    Lens("35mm f/1.8", 35.0, 11.0, 200, 1.8, 110, 2),
    Lens("50mm f/2.0", 50.0, 11.0, 250, 2.0, 100, 3),
    Lens("8mm f/2.8 (1\")", 8.0, 16.0, 150, 2.8, 90, 3),
    Lens("12mm f/2.8 (1\")", 12.0, 16.0, 150, 2.8, 100, 3),
    Lens("16mm f/2.8 (1\")", 16.0, 16.0, 200, 2.8, 110, 3),
    Lens("25mm f/2.8 (1\")", 25.0, 16.0, 200, 2.8, 120, 3),
    Lens("35mm f/2.8 (1\")", 35.0, 16.0, 250, 2.8, 120, 4),
    Lens("50mm f/2.8 (1\")", 50.0, 16.0, 300, 2.8, 110, 4),
    Lens("16mm f/2.0 HR", 16.0, 11.0, 150, 2.0, 200, 4),
    Lens("20mm f/2.0 HR", 20.0, 11.0, 150, 2.0, 200, 4),
    Lens("25mm f/2.0 HR", 25.0, 11.0, 200, 2.0, 200, 4),
    Lens("35mm f/2.0 HR", 35.0, 11.0, 250, 2.0, 200, 4),
]

TELECENTRIC: List[Telecentric] = [
    Telecentric("TC-0.05x", 0.05, 16.0, 400, 12.0),
    Telecentric("TC-0.075x", 0.075, 16.0, 350, 8.0),
    Telecentric("TC-0.10x", 0.10, 11.0, 300, 5.0),
    Telecentric("TC-0.15x", 0.15, 11.0, 250, 3.5),
    Telecentric("TC-0.20x", 0.20, 11.0, 200, 2.5),
    Telecentric("TC-0.30x", 0.30, 11.0, 150, 1.6),
    Telecentric("TC-0.50x", 0.50, 11.0, 120, 0.8),
    Telecentric("TC-1.00x", 1.00, 11.0, 100, 0.4),
]


# --------------------------------------------------------------------------
# 需求
# --------------------------------------------------------------------------
@dataclass
class Requirement:
    fov_w: float
    fov_h: float
    wd: float
    depth: float
    defect: float
    px_per_defect: float = 3.0
    coc_px: float = 2.0
    margin: float = 0.0        # 视野余量（0.1 = +10%）

    @property
    def fov_req_w(self) -> float:
        return self.fov_w * (1 + self.margin)

    @property
    def fov_req_h(self) -> float:
        return self.fov_h * (1 + self.margin)

    @property
    def pixel_accuracy_max(self) -> float:
        return self.defect / self.px_per_defect

    @property
    def min_pixels_w(self) -> int:
        return math.ceil(self.fov_req_w / self.pixel_accuracy_max)

    @property
    def min_pixels_h(self) -> int:
        return math.ceil(self.fov_req_h / self.pixel_accuracy_max)

    @property
    def min_mp(self) -> float:
        return self.min_pixels_w * self.min_pixels_h / 1e6


@dataclass
class Combo:
    camera: Camera
    lens: Lens
    fno: float
    m: float
    fov_w: float
    fov_h: float
    pa: float                  # mm/px
    ppd: float                 # px per defect
    dof: float                 # mm
    airy_um: float             # 艾里斑直径（信心值）
    f_feature: float           # 缺陷对应的空间频率 lp/mm（像面）
    f_cutoff: float            # 衍射截止频率 lp/mm
    diff_ratio: float          # f_feature / f_cutoff
    lens_need: float           # 建议镜头解析力 lp/mm = 2×f_feature
    mtf_ratio: float           # 镜头标称 / 建议值
    score: float = 0.0
    rel_score: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.camera.mid} + {self.lens.lid} @ f/{self.fno:g}"


# --------------------------------------------------------------------------
# 计算
# --------------------------------------------------------------------------
def dof_mm(n: float, m: float, coc_px: float, pixel_um: float) -> float:
    """物方景深（mm）：DOF ≈ 2·N·c·(1+m)/m²，c 取 coc_px 个像元。"""
    c = coc_px * pixel_um / 1000.0
    return 2.0 * n * c * (1.0 + m) / (m * m)


def airy_um(n: float, m: float) -> float:
    """艾里斑直径（µm）= 2.44·λ·N_eff，N_eff = N·(1+m)。"""
    return 2.44 * LAMBDA_UM * n * (1.0 + m)


def feature_lpmm(defect_mm: float, m: float) -> float:
    """缺陷在像面上的空间频率（lp/mm）：一个亮/暗周期 = 2×缺陷宽度。"""
    return 1.0 / (2.0 * defect_mm * m)


def diffraction_cutoff_lpmm(n: float, m: float) -> float:
    """衍射截止频率（lp/mm）= 1/(λ·N_eff)。"""
    return 1.0 / (LAMBDA_UM * 1e-3 * n * (1.0 + m))


def evaluate(cam: Camera, lens: Lens, req: Requirement) -> Tuple[Optional[Combo], str]:
    if lens.circle + 1e-9 < cam.diag:
        return None, "像圈不覆盖靶面"
    if req.wd <= lens.focal * 1.02:
        return None, "工作距离过近"
    if req.wd < lens.mod:
        return None, f"低于 MOD({lens.mod:g}mm)"

    m = lens.focal / (req.wd - lens.focal)
    fov_w = cam.sensor_w / m
    fov_h = cam.sensor_h / m
    if fov_w < req.fov_req_w or fov_h < req.fov_req_h:
        return None, "视野不足"

    pa = cam.pixel_um / 1000.0 / m
    ppd = req.defect / pa
    if ppd < req.px_per_defect:
        return None, f"像素不足({ppd:.1f}px)"

    # 选光圈：在满足景深的前提下取最小 f 数（衍射最小）
    chosen = None
    for n in STOPS:
        if n < lens.fno_max - 1e-9:
            continue
        if dof_mm(n, m, req.coc_px, cam.pixel_um) >= req.depth:
            chosen = n
            break
    if chosen is None:
        return None, "景深不足"

    d = dof_mm(chosen, m, req.coc_px, cam.pixel_um)
    a_um = airy_um(chosen, m)
    f_feat = feature_lpmm(req.defect, m)
    f_cut = diffraction_cutoff_lpmm(chosen, m)
    diff_ratio = f_feat / f_cut
    lens_need = 2.0 * f_feat          # 经验：镜头 30%-MTF 频率 ≥ 2× 缺陷频率
    mtf_ratio = lens.mtf_lpmm / lens_need

    c = Combo(cam, lens, chosen, m, fov_w, fov_h, pa, ppd, d, a_um,
              f_feat, f_cut, diff_ratio, lens_need, mtf_ratio)
    if ppd < req.px_per_defect * 1.15:
        c.warnings.append("像素余量 <15%")
    if diff_ratio > 0.45:
        c.warnings.append(f"衍射受限（缺陷频率达截止频率的 {diff_ratio:.0%}）")
    if mtf_ratio < 1.0:
        c.warnings.append(f"镜头解析力偏低（{lens.mtf_lpmm} < 建议 {lens_need:.0f} lp/mm）")
    if fov_w > req.fov_req_w * 1.35:
        c.warnings.append(f"视野偏大（{fov_w / req.fov_req_w - 1:.0%} 余量）")
    if d < req.depth * 1.15:
        c.warnings.append("景深余量 <15%")
    if cam.shutter == "卷帘":
        c.warnings.append("卷帘快门，动态拍摄需注意")
    return c, ""


def score(c: Combo, req: Requirement) -> float:
    # 像素余量在 1.6× 要求处饱和：再高只是浪费靶面/成本
    res_m = _clamp((c.ppd / req.px_per_defect - 1.0) / 0.6, 0, 1)
    dof_m = _clamp((c.dof / req.depth - 1.0) / 1.0, 0, 1)
    dif_m = _clamp((0.60 - c.diff_ratio) / 0.60, 0, 1)
    mtf_m = _clamp(c.mtf_ratio / 1.5, 0, 1)
    cost = 0.6 * c.camera.tier + 0.4 * c.lens.tier
    cost_m = _clamp((4.0 - cost) / 3.0, 0, 1)
    waste = _clamp(((c.fov_w * c.fov_h) / (req.fov_req_w * req.fov_req_h) - 1.0) / 2.0, 0, 1)
    base = 100 * (0.32 * res_m + 0.16 * dof_m + 0.20 * dif_m + 0.12 * mtf_m + 0.20 * cost_m)
    shutter_bonus = 4.0 if c.camera.shutter == "全局" else 0.0
    return base - 8 * waste + shutter_bonus


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def screen_cameras(req: Requirement) -> List[Tuple[Camera, float, float, bool]]:
    """仅用 ①视野 + ④分辨率 初筛，与镜头无关：pa = FOV_w / res_w。"""
    rows = []
    for cam in CAMERAS:
        pa = req.fov_req_w / cam.res_w
        ppd = req.defect / pa
        rows.append((cam, pa, ppd, ppd >= req.px_per_defect))
    return rows


def enumerate_combos(req: Requirement) -> List[Combo]:
    out = []
    for cam, _, _, ok in screen_cameras(req):
        if not ok:
            continue
        for lens in LENSES:
            c, _ = evaluate(cam, lens, req)
            if c:
                c.score = score(c, req)
                out.append(c)
    out.sort(key=lambda c: c.score, reverse=True)
    if out:
        top = out[0].score if out[0].score > 1e-6 else 1.0
        for c in out:
            c.rel_score = max(0.0, c.score / top * 100.0)
    return out


def best_per_camera(combos: Sequence[Combo]) -> List[Combo]:
    seen = set()
    best = []
    for c in combos:
        if c.camera.mid in seen:
            continue
        seen.add(c.camera.mid)
        best.append(c)
    return best


def telecentric_options(req: Requirement) -> List[dict]:
    """覆盖视野的远心档位；ok=False 表示像素或景深不达标（供对比）。"""
    rows = []
    for cam, _, _, ok in screen_cameras(req):
        if not ok:
            continue
        for tc in TELECENTRIC:
            if tc.circle + 1e-9 < cam.diag:
                continue
            if abs(req.wd - tc.wd) > 25:
                continue
            fov_w = cam.sensor_w / tc.mag
            fov_h = cam.sensor_h / tc.mag
            if fov_w < req.fov_req_w or fov_h < req.fov_req_h:
                continue
            pa = cam.pixel_um / 1000.0 / tc.mag
            ppd = req.defect / pa
            why = []
            if ppd < req.px_per_defect:
                why.append(f"像素 {ppd:.1f}px")
            if tc.dof < req.depth:
                why.append(f"景深 {tc.dof:g}mm")
            rows.append(dict(cam=cam, tc=tc, fov_w=fov_w, fov_h=fov_h, pa=pa,
                             ppd=ppd, dof=tc.dof, ok=not why, why="、".join(why)))
    rows.sort(key=lambda r: (r["ok"], r["ppd"]), reverse=True)
    return rows


# --------------------------------------------------------------------------
# 文本排版
# --------------------------------------------------------------------------
def dw(s: object) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(s))


def pad(s: object, n: int, left: bool = False) -> str:
    s = str(s)
    gap = " " * max(0, n - dw(s))
    return gap + s if left else s + gap


def wrap_cjk(text: str, width: int = 74) -> List[str]:
    """按显示宽度换行，拉丁词/数字整体不拆开。"""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._/%+\-\"']*|\s+|.", text)
    lines, cur, curw = [], "", 0
    for tok in tokens:
        tw = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in tok)
        if curw + tw > width and cur:
            lines.append(cur.rstrip())
            cur, curw = "", 0
            if tok.isspace():
                continue
        cur += tok
        curw += tw
    if cur:
        lines.append(cur)
    return lines


def console_safe(text: str) -> str:
    """Map a few symbols onto GBK-safe equivalents (same display width)."""
    for k, v in {"⇒": "→", "✓": "√", "✗": "×", "µ": "u", "≈": "~"}.items():
        text = text.replace(k, v)
    return text


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(text.encode(enc, "replace").decode(enc))


def table(headers: Sequence[str], rows: Sequence[Sequence[object]], aligns: str = "") -> str:
    rows = [[str(x) for x in r] for r in rows]
    aligns = aligns or "l" * len(headers)
    aligns = (aligns + "l" * len(headers))[:len(headers)]
    widths = [max([dw(h)] + [dw(r[i]) for r in rows]) for i, h in enumerate(headers)]
    sep = "  "
    lines = [sep.join(pad(h, widths[i], aligns[i] == "r") for i, h in enumerate(headers))]
    lines.append(sep.join("-" * w for w in widths))
    for r in rows:
        lines.append(sep.join(pad(v, widths[i], aligns[i] == "r") for i, v in enumerate(r)))
    return "\n".join(lines)


def hr(ch: str = "=", n: int = 78) -> str:
    return ch * n


# --------------------------------------------------------------------------
# 控制台报告
# --------------------------------------------------------------------------
def conclusion_text(req: Requirement, combos: Sequence[Combo]) -> str:
    if not combos:
        return ("没有满足全部约束的组合：建议缩小视野、拉长工作距离、放宽景深，"
                "或改用线扫/多相机拼接。")
    b = best_per_camera(combos)[0]
    s = (f"{req.fov_w:g}×{req.fov_h:g} mm 视野、{req.wd:g} mm 工作距离下，"
         f"推荐 {b.camera.mid} + {b.lens.lid}（f/{b.fno:g}）："
         f"0.1 mm 划痕约 {b.ppd:.1f} px，像素精度 {b.pa * 1000:.1f} µm/px，"
         f"景深 {b.dof:.1f} mm。")
    if b.mtf_ratio < 1:
        s += (f" 该镜头解析力（{b.lens.mtf_lpmm} lp/mm）低于建议的 {b.lens_need:.0f} lp/mm，"
              "划痕边缘对比度会打折；建议换高解析镜头，或把视野缩小 20% 再来一遍。")
    alt = [c for c in best_per_camera(combos)[1:2]]
    if alt:
        a = alt[0]
        s += (f" 备选：{a.camera.mid} + {a.lens.lid}，"
              f"0.1 mm 约 {a.ppd:.1f} px、景深 {a.dof:.1f} mm。")
    return s


def console_report(req: Requirement, combos: List[Combo],
                   tele: List[dict], sweep: bool = False) -> str:
    L: List[str] = []
    A = L.append
    A(hr())
    A("  工业相机 / 镜头选型报告   ·   镜头选型四要素法")
    A(hr())
    A("")

    A("【1】需求解读")
    A(f"  最小可检特征（划痕宽度） : {req.defect:.3g} mm")
    A(f"  每个特征覆盖像素数       : {req.px_per_defect:.1f} px  "
      f"（2–3px 检出 / 3–5px 测量）")
    A(f"  ⇒ 像素精度上限           : {req.pixel_accuracy_max:.4f} mm/px")
    A(f"  ⇒ {req.fov_req_w:.0f}×{req.fov_req_h:.0f} mm 视野至少需要 "
      f"{req.min_pixels_w}×{req.min_pixels_h} = {req.min_mp:.1f} MP")
    A(f"  测量不确定度（±1px）     : ±{req.pixel_accuracy_max:.4f} mm")
    A("")

    A("【2】四要素落位")
    A(table(
        ["要素", "本方案输入", "决定什么 / 用的公式"],
        [
            ["① 视野 FOV", f"{req.fov_w:g}×{req.fov_h:g} mm",
             "所需像素 = FOV ÷ 像素精度；决定 m = 靶面/FOV"],
            ["② 工作距离 WD", f"{req.wd:g} mm",
             "FOV = 靶面 × (WD-f)/f；WD 越短，同焦距 FOV 越小"],
            ["③ 景深 DOF", f"≥ {req.depth:g} mm",
             "DOF ≈ 2Nc(1+m)/m²；由光圈 N 与放大倍率 m 共同决定"],
            ["④ 分辨率", f"{req.defect:g} mm @ {req.px_per_defect:g}px",
             f"像素精度 ≤ {req.pixel_accuracy_max:.4f} mm/px"],
        ]))
    A("")

    A("【3】相机初筛（只看 ①视野 + ④分辨率，与镜头无关）")
    rows = []
    for cam, pa, ppd, ok in screen_cameras(req):
        rows.append([cam.mid, f"{cam.res_w}×{cam.res_h}", f"{cam.mp:.1f}",
                     f"{cam.pixel_um:.2f}", f"{pa * 1000:.1f}",
                     f"{ppd:.2f}", "✓" if ok else "✗"])
    A(table(["相机", "分辨率", "MP", "像元(µm)", "精度(µm/px)", "px/缺陷", "判定"],
            rows, "lrrrrrc"))
    A("")

    if not combos:
        A("【4】没有满足全部约束的「相机+镜头」组合。")
        A("     建议：缩小视野、增加工作距离、放宽景深，或改用线扫/多相机拼接。")
        A("")
    else:
        A("【4】推荐组合（已按综合评分排序；同一相机取最优镜头）")
        rows = []
        for i, c in enumerate(best_per_camera(combos)[:4], 1):
            rows.append([
                i, c.camera.mid, c.lens.lid, f"f/{c.fno:g}",
                f"{c.fov_w:.1f}×{c.fov_h:.1f}", f"{c.pa * 1000:.2f}",
                f"{c.ppd:.2f}", f"{c.dof:.1f}", f"{c.diff_ratio:.0%}",
                f"{c.rel_score:.0f}",
            ])
        A(table(["#", "相机", "镜头", "光圈", "视野(mm)", "精度(µm/px)",
                 "px/缺陷", "景深(mm)", "衍射占比", "评分"], rows, "rllllrrrrr"))
        A("")

        for i, c in enumerate(best_per_camera(combos)[:3], 1):
            A(f"  ▌方案 {i}：{c.name}")
            A(f"     靶面 {c.camera.sensor_w:.2f}×{c.camera.sensor_h:.2f} mm"
              f"（{c.camera.fmt}），放大倍率 m={c.m:.4f}")
            A(f"     实际视野 {c.fov_w:.1f}×{c.fov_h:.1f} mm，"
              f"像素精度 {c.pa * 1000:.2f} µm/px，0.1mm≈{c.ppd:.1f}px")
            A(f"     景深 {c.dof:.1f} mm @ f/{c.fno:g}；艾里斑 {c.airy_um:.1f} µm")
            A(f"     缺陷频率 {c.f_feature:.0f} lp/mm vs 衍射截止 {c.f_cutoff:.0f} lp/mm"
              f"（占 {c.diff_ratio:.0%}）")
            A(f"     镜头解析力建议 ≥ {c.lens_need:.0f} lp/mm，"
              f"该镜头 {c.lens.mtf_lpmm} lp/mm（{c.mtf_ratio:.0%}）")
            if c.warnings:
                A("     注意：" + "；".join(c.warnings))
            A("")

        A("【结论】")
        for line in wrap_cjk(conclusion_text(req, combos), 74):
            A("  " + line)
        A("")

    if tele:
        A("【5】测量级替代：远心镜头（低畸变、无透视误差，WD 固定）")
        rows = []
        for r in tele[:3]:
            rows.append([r["cam"].mid, r["tc"].tid, f"{r['tc'].mag:g}×",
                         f"{r['fov_w']:.1f}×{r['fov_h']:.1f}",
                         f"{r['pa'] * 1000:.2f}", f"{r['ppd']:.2f}", f"{r['dof']:.1f}",
                         "可用" if r["ok"] else ("不满足：" + r["why"])])
        A(table(["相机", "远心镜头", "倍率", "视野(mm)", "精度(µm/px)", "px/缺陷",
                 "景深(mm)", "判定"], rows, "lllrrrrl"))
        A("")
    else:
        A("【5】远心镜头：当前 WD（±25 mm）内没有能覆盖该视野的标准档位，")
        A("     需按具体型号定制，或调整安装距离后另算。")
        A("")

    if sweep:
        A("【6】视野—像素需求对照（同一 0.1mm 精度要求）")
        rows = []
        max_mp = max(c.mp for c in CAMERAS)
        for w, h in [(50, 40), (100, 80), (200, 160), (300, 240)]:
            nw = math.ceil(w * (1 + req.margin) / req.pixel_accuracy_max)
            nh = math.ceil(h * (1 + req.margin) / req.pixel_accuracy_max)
            mp = nw * nh / 1e6
            verdict = "✓ 单相机" if mp <= max_mp else "✗ 需多相机/线扫"
            rows.append([f"{w}×{h}", f"{nw}×{nh}", f"{mp:.1f}", verdict])
        A(table(["视野(mm)", "所需像素", "MP", "结论"], rows, "lrrl"))
        A("")

    A(hr("-"))
    A("假设：薄透镜近似；WD 自镜头主平面起算；CoC = "
      f"{req.coc_px:g}×像元；λ={LAMBDA_UM}µm；镜头 MTF 按 30% 对比度标称。")
    A("以上为选型初算，最终需用实测 MTF/畸变与打光验证。")
    return "\n".join(L)


# --------------------------------------------------------------------------
# Markdown / HTML
# --------------------------------------------------------------------------
CSS = """
:root{--ink:#171a1f;--muted:#6b7280;--line:#e6e8ec;--bg:#f6f7f9;--card:#fff;--accent:#b45309;--ok:#0f7b3f;--bad:#b42318;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;}
.wrap{max-width:1080px;margin:0 auto;padding:40px 24px 72px;}
header{border-bottom:3px solid var(--accent);padding-bottom:18px;margin-bottom:28px;}
h1{font-size:26px;margin:0 0 6px;letter-spacing:.3px;}
.sub{color:var(--muted);font-size:13.5px;margin:0;}
h2{font-size:18px;margin:38px 0 12px;padding-left:10px;border-left:4px solid var(--accent);}
p{margin:8px 0;}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0 8px;}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;}
.card .k{font-size:12px;color:var(--muted);letter-spacing:.4px;}
.card .v{font-size:24px;font-weight:650;margin-top:4px;}
.card .u{font-size:12px;color:var(--muted);}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden;font-size:13.5px;margin:10px 0 4px;}
th{background:#f0f2f5;text-align:left;padding:9px 11px;font-weight:600;font-size:12.5px;color:#3d4450;white-space:nowrap;border-bottom:1px solid var(--line);}
td{padding:9px 11px;border-bottom:1px solid #eef0f3;}
tr:last-child td{border-bottom:none;}
.num{font-variant-numeric:tabular-nums;}
.ok{color:var(--ok);font-weight:600;}
.bad{color:var(--bad);font-weight:600;}
.bar{position:relative;height:8px;background:#eceef2;border-radius:4px;min-width:90px;}
.bar i{position:absolute;left:0;top:0;bottom:0;border-radius:4px;background:linear-gradient(90deg,#f59e0b,#b45309);}
.bar b{position:absolute;top:-3px;bottom:-3px;width:2px;background:#111;opacity:.55;}
.rec{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:12px 0;}
.rec h3{margin:0 0 4px;font-size:16.5px;}
.callout{background:#fdf8ef;border:1px solid #f0dcbd;border-left:4px solid var(--accent);border-radius:8px;padding:14px 16px;margin:14px 0 4px;font-size:14px;}
.rec .tag{display:inline-block;font-size:11.5px;padding:2px 8px;border-radius:20px;background:#fdf1e3;color:#92400e;margin-right:8px;vertical-align:2px;}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px 18px;margin:12px 0 6px;}
.grid div{font-size:13px;color:var(--muted);}
.grid b{display:block;color:var(--ink);font-size:15px;font-weight:600;}
ul.notes{margin:8px 0 0 18px;padding:0;font-size:13px;color:#4b5563;}
ul.notes li{margin:3px 0;}
.formula{background:#fff;border:1px solid var(--line);border-radius:8px;padding:12px 16px;font-size:13.5px;color:#374151;}
.formula code{background:#f3f4f6;padding:1px 5px;border-radius:4px;font-size:12.5px;}
footer{margin-top:44px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:14px;}
@media(max-width:820px){.cards{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:repeat(2,1fr)}}
"""


def e(s: object) -> str:
    return html.escape(str(s))


def html_table(headers: Sequence[str], rows: Sequence[Sequence[str]], cls: str = "") -> str:
    th = "".join(f"<th>{e(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>'


def html_report(req: Requirement, combos: List[Combo], tele: List[dict],
                stamp: str) -> str:
    screens = screen_cameras(req)
    best = best_per_camera(combos)[:4]
    max_ppd = max((r[2] for r in screens), default=1) or 1

    card_data = [
        ("视野 FOV", f"{req.fov_w:g}×{req.fov_h:g}", "mm"),
        ("工作距离 WD", f"{req.wd:g}", "mm"),
        ("景深要求", f"≥{req.depth:g}", "mm"),
        ("像素精度上限", f"{req.pixel_accuracy_max * 1000:.1f}", "µm/px"),
    ]
    cards = "".join(
        f'<div class="card"><div class="k">{e(k)}</div>'
        f'<div class="v">{e(v)}<span class="u"> {e(u)}</span></div></div>'
        for k, v, u in card_data)

    four = html_table(
        ["要素", "输入", "决定什么 / 公式"],
        [
            ["① 视野 FOV", f"{req.fov_w:g} × {req.fov_h:g} mm",
             f"至少需要 <b>{req.min_pixels_w}×{req.min_pixels_h}</b> 像素（{req.min_mp:.1f} MP）"],
            ["② 工作距离 WD", f"{req.wd:g} mm",
             "FOV = 靶面尺寸 × (WD−f) / f，决定可用的焦距区间"],
            ["③ 景深 DOF", f"≥ {req.depth:g} mm",
             "DOF ≈ 2Nc(1+m)/m²，光圈越大景深越浅，是分辨率与景深的取舍点"],
            ["④ 分辨率", f"{req.defect:g} mm @ {req.px_per_defect:g} px",
             f"像素精度 ≤ <b>{req.pixel_accuracy_max:.4f} mm/px</b>；测量不确定度 ±{req.pixel_accuracy_max:.4f} mm"],
        ])

    cam_rows = []
    for cam, pa, ppd, ok in screens:
        width = min(100, ppd / (max_ppd * 1.05) * 100)
        mark = 0.0  # threshold marker position
        if max_ppd > 0:
            mark = req.px_per_defect / (max_ppd * 1.05) * 100
        bar = (f'<div class="bar"><i style="width:{width:.1f}%"></i>'
               f'<b style="left:{min(99, mark):.1f}%"></b></div>')
        cam_rows.append([
            f"<b>{e(cam.mid)}</b><br><span style='color:#6b7280;font-size:12px'>{e(cam.note)}</span>",
            f"{cam.res_w}×{cam.res_h}", f"{cam.mp:.1f}", f"{cam.pixel_um:.2f}",
            f"{pa * 1000:.1f}", f"{ppd:.2f} {bar}",
            f'<span class="{"ok" if ok else "bad"}">{"通过" if ok else "不足"}</span>',
        ])
    cam_table = html_table(
        ["相机", "分辨率", "MP", "像元 µm", "精度 µm/px",
         f"px / {req.defect:g}mm（黑线 = 门槛 {req.px_per_defect:g}px）", "判定"],
        cam_rows)

    recs = ""
    for i, c in enumerate(best, 1):
        badge = "推荐" if i == 1 else "备选"
        notes = "".join(f"<li>{e(w)}</li>" for w in c.warnings) or "<li>无显著风险项</li>"
        recs += f"""
        <div class="rec">
          <div><span class="tag">{badge}</span><b>{e(c.name)}</b></div>
          <div class="grid">
            <div>实际视野<b>{c.fov_w:.1f} × {c.fov_h:.1f} mm</b></div>
            <div>像素精度<b>{c.pa*1000:.2f} µm/px</b></div>
            <div>0.1 mm 划痕<b>{c.ppd:.1f} px</b></div>
            <div>景深 @ f/{c.fno:g}<b>{c.dof:.1f} mm</b></div>
            <div>衍射占比（缺陷频率/截止）<b>{c.diff_ratio:.0%}</b></div>
            <div>镜头解析力<b>{c.lens.mtf_lpmm} / 建议 {c.lens_need:.0f} lp/mm</b></div>
          </div>
          <ul class="notes">{notes}</ul>
        </div>"""

    if not best:
        recs = ('<div class="rec"><b>没有满足全部约束的组合。</b><br>'
                '建议缩小视野、拉长工作距离、放宽景深，或采用线扫/多相机拼接。</div>')

    tele_html = ""
    if tele:
        rows = []
        for r in tele[:4]:
            badge = ('<span class="ok">可用</span>' if r["ok"]
                     else f'<span class="bad">不满足：{e(r["why"])}</span>')
            rows.append([r["cam"].mid, r["tc"].tid, f"{r['tc'].mag:g}×",
                         f"{r['fov_w']:.1f}×{r['fov_h']:.1f}",
                         f"{r['pa']*1000:.2f}", f"{r['ppd']:.2f}", f"{r['dof']:.1f}", badge])
        tele_html = html_table(["相机", "远心镜头", "倍率", "视野 mm", "精度 µm/px",
                                "px/缺陷", "景深 mm", "判定"], rows)
    else:
        tele_html = ('<p style="color:#6b7280">当前 WD（±25 mm）内没有能覆盖该视野的标准远心档位'
                     '（远心镜头工作距离固定，需要按型号定制或调整安装距离）。</p>')

    sweep_rows = []
    max_mp = max(cam.mp for cam in CAMERAS)
    for w, h in [(50, 40), (100, 80), (200, 160), (300, 240)]:
        nw = math.ceil(w * (1 + req.margin) / req.pixel_accuracy_max)
        nh = math.ceil(h * (1 + req.margin) / req.pixel_accuracy_max)
        mp = nw * nh / 1e6
        ok = mp <= max_mp
        sweep_rows.append([f"{w}×{h}", f"{nw}×{nh}", f"{mp:.1f}",
                           '<span class="ok">单相机</span>' if ok
                           else '<span class="bad">需多相机 / 线扫</span>'])
    sweep_table = html_table(["视野 mm", "所需像素", "MP", "结论"], sweep_rows)

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>工业相机/镜头选型报告 · {req.defect:g}mm 划痕检测</title>
<style>{CSS}</style></head>
<body><div class="wrap">
<header>
  <h1>工业相机 / 镜头选型报告</h1>
  <p class="sub">{req.defect:g} mm 划痕检测 · 镜头选型四要素法 · 生成于 {e(stamp)}</p>
</header>

<div class="cards">{cards}</div>

<h2>一、四要素如何落到参数上</h2>
{four}

<h2>二、相机初筛（只看视野与分辨率）</h2>
<p>在 {req.fov_req_w:.0f}×{req.fov_req_h:.0f} mm 视野下，每台相机的像素精度 = 视野 ÷ 水平像素数
（假设镜头刚好框满该视野，与镜头无关）。要稳定检出 {req.defect:g} mm 划痕，至少需要
{req.px_per_defect:g} px 覆盖划痕宽度。</p>
{cam_table}

<h2>三、相机 + 镜头组合</h2>
{recs}
<div class="callout"><b>结论　</b>{e(conclusion_text(req, combos))}</div>

<h2>四、测量级方案：远心镜头</h2>
<p>若 0.1 mm 是<b>公差</b>而非仅是检出阈值，建议远心镜头（畸变 &lt;0.1%、无透视误差），
并用标定板做像素当量标定。</p>
{tele_html}

<h2>五、视野对像素需求的影响</h2>
<p>若被测物尺寸尚未确定，可先用下表定位：同一 0.1 mm 精度要求下，视野越大，像素需求按面积增长。</p>
{sweep_table}

<h2>六、假设与公式</h2>
<div class="formula">
<code>m = f / (WD − f)</code>　放大倍率（薄透镜）<br>
<code>FOV = 靶面 / m</code>　实际视野<br>
<code>px_acc = 像元 / m</code>　像素精度 (mm/px)<br>
<code>DOF ≈ 2·N·c·(1+m)/m²</code>，c = {req.coc_px:g}×像元，N 为光圈 f 数<br>
<code>d_airy = 2.44·λ·N·(1+m)</code>，λ = {LAMBDA_UM} µm
</div>
<p style="color:#6b7280;font-size:13px">WD 自镜头主平面起算；实际镜头的 MOD、畸变、
镀膜与打光会影响最终效果，本报告用于初选，定型前需实测验证。</p>

<footer>lens_selector v{VERSION} · {e(stamp)} · 器材规格为同级通用值，采购时请以厂商 datasheet 复核。</footer>
</div></body></html>"""


def markdown_report(req: Requirement, combos: List[Combo], tele: List[dict], stamp: str) -> str:
    L = ["# 工业相机 / 镜头选型报告", "",
         f"*{req.defect:g} mm 划痕检测 · 四要素法 · {stamp}*", ""]
    L += ["## 需求", "",
          f"- 视野：{req.fov_w:g} × {req.fov_h:g} mm",
          f"- 工作距离：{req.wd:g} mm",
          f"- 景深要求：≥ {req.depth:g} mm",
          f"- 最小特征：{req.defect:g} mm，要求 {req.px_per_defect:g} px",
          f"- 像素精度上限：{req.pixel_accuracy_max:.4f} mm/px",
          f"- 最少像素：{req.min_pixels_w} × {req.min_pixels_h}（{req.min_mp:.1f} MP）", ""]

    L += ["## 相机初筛", "", "| 相机 | 分辨率 | 像元 µm | 精度 µm/px | px/0.1mm | 判定 |",
          "|---|---|---|---|---|---|"]
    for cam, pa, ppd, ok in screen_cameras(req):
        L.append(f"| {cam.mid} | {cam.res_w}×{cam.res_h} | {cam.pixel_um:.2f} | "
                 f"{pa*1000:.1f} | {ppd:.2f} | {'✓' if ok else '✗'} |")
    L.append("")

    L += ["## 推荐组合", ""]
    if combos:
        L += [f"| # | 相机 | 镜头 | 光圈 | 视野 mm | 精度 µm/px | px/{req.defect:g}mm | 景深 mm | 评分 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for i, c in enumerate(best_per_camera(combos)[:4], 1):
            L.append(f"| {i} | {c.camera.mid} | {c.lens.lid} | f/{c.fno:g} | "
                     f"{c.fov_w:.1f}×{c.fov_h:.1f} | {c.pa*1000:.2f} | {c.ppd:.2f} | "
                     f"{c.dof:.1f} | {c.rel_score:.0f} |")
    else:
        L.append("没有满足全部约束的组合。")
    L.append("")
    L += [f"> **结论**　{conclusion_text(req, combos)}", ""]

    if tele:
        L += ["## 远心镜头（测量级）", "", "| 相机 | 镜头 | 视野 mm | 精度 µm/px | px/缺陷 | 判定 |",
              "|---|---|---|---|---|---|"]
        for r in tele[:4]:
            L.append(f"| {r['cam'].mid} | {r['tc'].tid} | {r['fov_w']:.1f}×{r['fov_h']:.1f} | "
                     f"{r['pa']*1000:.2f} | {r['ppd']:.2f} | "
                     f"{'可用' if r['ok'] else '不满足：' + r['why']} |")
        L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_fov(text: str) -> Tuple[float, float]:
    for sep in ("x", "X", "*", "×", ","):
        if sep in text:
            a, b = text.split(sep, 1)
            return float(a), float(b)
    v = float(text)
    return v, v * 0.8


def interactive(req: Requirement) -> Requirement:
    print("镜头选型四要素问答（回车 = 用括号内默认值）\n")
    try:
        s = input(f"① 视野 宽x高 mm [{req.fov_w:g}x{req.fov_h:g}]: ").strip()
        if s:
            req.fov_w, req.fov_h = parse_fov(s)
        s = input(f"② 工作距离 WD mm [{req.wd:g}]: ").strip()
        if s:
            req.wd = float(s)
        s = input(f"③ 景深要求 mm（物体高度差 / 定位误差）[{req.depth:g}]: ").strip()
        if s:
            req.depth = float(s)
        s = input(f"④ 最小缺陷 mm [{req.defect:g}]: ").strip()
        if s:
            req.defect = float(s)
        s = input(f"   每个缺陷至少几个像素（2–3 检出，3–5 测量）[{req.px_per_defect:g}]: ").strip()
        if s:
            req.px_per_defect = float(s)
    except (EOFError, KeyboardInterrupt):
        print("\n（使用默认值）")
    print()
    return req


def selftest() -> int:
    """几组手算值，防止公式/单位回归。"""
    checks = []

    def chk(name, cond):
        checks.append((name, bool(cond)))

    req = Requirement(100, 80, 400, 5, 0.1, 3)
    chk("像素精度上限 = 0.1/3", abs(req.pixel_accuracy_max - 0.033333) < 1e-5)
    chk("100mm 视野需 3000px", req.min_pixels_w == 3000)
    chk("最少像素 7.2MP", abs(req.min_mp - 7.2) < 0.05)

    cam = next(c for c in CAMERAS if c.mid == "CAM-12MP")
    chk("12MP 靶面宽 7.44mm", abs(cam.sensor_w - 7.4444) < 1e-3)
    lens = next(l for l in LENSES if l.lid == "25mm f/1.8")
    m = lens.focal / (req.wd - lens.focal)
    chk("m = 25/(400-25)", abs(m - 0.066667) < 1e-5)
    chk("FOV = 111.7mm", abs(cam.sensor_w / m - 111.67) < 0.1)
    chk("DOF(f/4) = 7.1mm", abs(dof_mm(4, m, 2, cam.pixel_um) - 7.11) < 0.05)
    chk("衍射截止 426 lp/mm", abs(diffraction_cutoff_lpmm(4, m) - 426) < 2)

    combos = enumerate_combos(req)
    chk("有可行组合", len(combos) >= 1)
    chk("首选 12MP", combos and combos[0].camera.mid == "CAM-12MP")
    chk("5MP 被筛掉", not any(c.camera.mid.startswith("CAM-5MP") for c in combos))

    bad = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print(("  PASS  " if ok else "  FAIL  ") + n)
    print(f"\n{len(checks) - len(bad)}/{len(checks)} 项通过")
    return 1 if bad else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="工业相机/镜头选型助手（四要素法）")
    p.add_argument("--fov", default=None, help="视野 宽x高(mm)，如 100x80")
    p.add_argument("--wd", type=float, default=400.0, help="工作距离 mm")
    p.add_argument("--depth", type=float, default=5.0, help="景深要求 mm")
    p.add_argument("--defect", type=float, default=0.1, help="最小缺陷 mm")
    p.add_argument("--px-per-defect", type=float, default=3.0, help="每个缺陷的像素数")
    p.add_argument("--coc-px", type=float, default=2.0, help="弥散圆 = n 个像元")
    p.add_argument("--margin", type=float, default=0.0, help="视野余量，0.1=+10%%")
    p.add_argument("--out", default="reports", help="报告输出目录")
    p.add_argument("--demo", action="store_true", help="内置演示：100x80mm / WD400 / 0.1mm")
    p.add_argument("--interactive", action="store_true", help="强制进入四要素问答")
    p.add_argument("--sweep", action="store_true", help="附加视野-像素需求对照")
    p.add_argument("--json", action="store_true", help="同时输出 JSON")
    p.add_argument("--no-report", action="store_true", help="只打印，不写文件")
    p.add_argument("--selftest", action="store_true", help="运行公式自检")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.selftest:
        return selftest()
    req = Requirement(fov_w=100, fov_h=80, wd=400, depth=5, defect=0.1,
                      px_per_defect=args.px_per_defect, coc_px=args.coc_px,
                      margin=args.margin)
    if args.fov:
        req.fov_w, req.fov_h = parse_fov(args.fov)
    req.wd = args.wd
    req.depth = args.depth
    req.defect = args.defect

    if args.interactive or (not args.fov and not args.demo and sys.stdin.isatty()):
        req = interactive(req)

    combos = enumerate_combos(req)
    tele = telecentric_options(req)
    text = console_report(req, combos, tele, sweep=args.sweep)
    safe_print(console_safe(text))

    if args.no_report:
        return 0

    os.makedirs(args.out, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(args.out, "report.html"), "w", encoding="utf-8") as f:
        f.write(html_report(req, combos, tele, stamp))
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(markdown_report(req, combos, tele, stamp))
    if args.json:
        payload = {
            "requirement": asdict(req),
            "screening": [
                dict(camera=c.mid, res=[c.res_w, c.res_h], pixel_um=c.pixel_um,
                     pa_mm=round(pa, 6), px_per_defect=round(ppd, 3), pass_=ok)
                for c, pa, ppd, ok in screen_cameras(req)
            ],
            "combos": [
                dict(camera=c.camera.mid, lens=c.lens.lid, fno=c.fno,
                     fov=[round(c.fov_w, 2), round(c.fov_h, 2)],
                     pa_mm=round(c.pa, 6), px_per_defect=round(c.ppd, 3),
                     dof_mm=round(c.dof, 2), diff_ratio=round(c.diff_ratio, 3),
                     mtf_ratio=round(c.mtf_ratio, 2),
                     score=round(c.score, 1), warnings=c.warnings)
                for c in combos
            ],
            "telecentric": [
                dict(camera=r["cam"].mid, lens=r["tc"].tid, fov=[round(r["fov_w"], 2), round(r["fov_h"], 2)],
                     pa_mm=round(r["pa"], 6), px_per_defect=round(r["ppd"], 3))
                for r in tele
            ],
        }
        with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print()
    print(f"报告已写入：{os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
