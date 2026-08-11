"""三维 fourbar 机构模型、输入曲线定义与误差评价。

本文件是机构计算的唯一模型文件，主要执行以下工作：

1. 直接以 YZ 平行平面内的 B 点二维曲线作为机构输入，并求解全部机构节点；
2. 定义 B 点 Fourier 输入、周期杆长和目标共同位姿；
3. 计算 tip=U、wrist=L 与目标曲线之间的绝对位置 RMSE；
4. 为优化程序提供统一变量解码接口，并可独立复算保存的检查点。

坐标单位为 mm，角度单位为 rad。机构核心几何沿用原始 fourbar 模型。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
from scipy.io import loadmat


# =============================================================================
# 1. 机构参数和返回结果
# =============================================================================


class FourBarError(RuntimeError):
    """几何约束不成立或机构在某一帧不可装配时抛出的内部异常。"""

    pass


@dataclass
class FourBarParams:
    """原始 MATLAB fourbar 模型使用的杆长与固定几何参数。"""

    # User-editable fallback initial values.
    L1: float = 26.23
    L2: float = 69.0
    L31: float = 42.0
    # L32 仅保留为序列化兼容字段；正式模型始终使用 L3(t)-L31+2 逐帧派生。
    L32: float = 101.0
    L4: float = 41.0
    L41: float = 23.0
    L5: float = 165.0
    L51: float = 32.0
    L52: float = 32.0
    L6: float = 257.0
    L7: float = 255.0
    L8: float = 251.0
    L9: float = 32.0
    L10: float = 30.0
    L11: float = 15.0
    L12: float = 15.0
    L13: float = 15.0
    L14: float = 15.0
    L15: float = 10.0
    L17: float = 14.0
    H_finger: float = 3.0
    CenterX: float = 30.0
    CenterY: float = 10.0
    Radius: float = 50.0
    theta18_deg: float = 120.0
    CenterZ: float = 0.0
    L3: float = 141.0
    L61: float = 236.0
    Lf1: float = 412.0
    Lf2: float = 277.0
    L_CZ: float = 62.0
    My: float = 0.0
    L_down: float = 15.0
    LRod: float | None = 50.0

    @property
    def L16(self) -> float:
        """由 PO=L15、QP=L16 和 PO=QP 得到 L16=L15。"""

        return self.L15

    def to_base32(self) -> np.ndarray:
        """Return parameters in the legacy MATLAB base32 order."""
        # Legacy slot 18 is unused but retained as zero so later indices stay compatible.
        return np.array([
            self.L1, self.L2, self.L31, self.L4, self.L41,
            self.L5, self.L51, self.L52, self.L6, self.L7,
            self.L8, self.L9, self.L10, self.L11, self.L12,
            self.L14, self.L15, self.L17, 0.0, self.H_finger,
            self.CenterX, self.CenterY, self.Radius, self.theta18_deg, self.CenterZ,
            self.L3, self.L61, self.Lf1, self.Lf2, self.L_CZ,
            self.L_down, self.L12 if self.LRod is None else self.LRod,
        ], dtype=float)

    @staticmethod
    def from_base32(base32: np.ndarray) -> "FourBarParams":
        """按原始 base32 的固定顺序恢复参数，避免优化代码自行解释下标。"""
        b = np.asarray(base32, dtype=float).ravel()
        if b.size != 32:
            raise ValueError(f"base32 must have 32 values, got {b.size}.")
        return FourBarParams(
            L1=b[0], L2=b[1], L31=b[2], L4=b[3], L41=b[4],
            L5=b[5], L51=b[6], L52=b[7], L6=b[8], L7=b[9],
            L8=b[10], L9=b[11], L10=b[12], L11=b[13], L12=b[14],
            L14=b[15], L15=b[16], L17=b[17],
            H_finger=b[19], CenterX=b[20], CenterY=b[21],
            Radius=b[22], theta18_deg=b[23], CenterZ=b[24],
            L3=b[25], L61=b[26], Lf1=b[27], Lf2=b[28],
            L_CZ=b[29], My=0.0, L_down=b[30], LRod=b[31],
            # base32 没有独立 L32 槽位；该兼容字段直接按正式关系派生。
            L32=b[25] - b[2] + 2.0,
        )


@dataclass
class FourBarResult:
    """一次完整 76 帧运动计算的结果。"""

    valid: bool
    message: str
    tip: np.ndarray
    wrist: np.ndarray
    nodes: np.ndarray
    node_names: Tuple[str, ...]
    theta_wrist: np.ndarray
    theta20: np.ndarray
    theta21: np.ndarray
    thetaM: np.ndarray
    thetaN: np.ndarray
    angle_history: Dict[str, np.ndarray]
    # 旧字段名仅用于历史文件兼容；新模型中它与 b_curve 相同，不代表物理 Mot 点。
    mot_curve: np.ndarray
    b_curve: np.ndarray
    input_radius: np.ndarray
    mot_radius: np.ndarray
    theta01: np.ndarray
    theta02: np.ndarray
    l2_values: np.ndarray
    l6_values: np.ndarray
    l5_values: np.ndarray | None = None
    l31_values: np.ndarray | None = None
    l32_values: np.ndarray | None = None
    l3_values: np.ndarray | None = None
    l7_values: np.ndarray | None = None
    l8_values: np.ndarray | None = None
    zc_values: np.ndarray | None = None
    zc_start_index: int | None = None

    @property
    def bmot_length(self) -> np.ndarray:
        """历史兼容量；新模型已删除独立 Mot，因此恒为零。"""
        return np.zeros(self.b_curve.shape[0], dtype=float)


# 与 RMSE29.99 CMA-ES 基准一致，保留 A...Z 的稳定下标；Z 每帧严格等于 C。
NODE_NAMES = tuple(chr(ord("A") + i) for i in range(26))
IDX = {name: i for i, name in enumerate(NODE_NAMES)}

# Physical links used by the orthographic mechanism view.
# R and Y are intentionally absent because the current solver does not define them.
MECHANISM_LINKS = (
    ("A", "B"), ("B", "C"), ("A", "C"),
    ("A", "D"), ("C", "D"),
    ("D", "G"), ("G", "A"),
    ("E", "F"), ("F", "G"),
    ("F", "H"), ("H", "K"), ("K", "L"), ("L", "F"),
    ("H", "I"), ("I", "J"), ("J", "K"),
    ("K", "M"), ("M", "N"), ("N", "L"),
    ("N", "W"), ("M", "O"), ("O", "P"), ("P", "Q"),
)

# J-O 表示腕部下降量和折转角计算使用的构型关系，不把它误写成新的刚性闭环杆。
CONSTRUCTION_LINKS = (
    ("J", "O"),
)

EXTENSION_LINKS = (
    ("W", "S"), ("W", "T"), ("W", "U"),
)

WRIST_VIEW_NODES = ("J", "O", "W", "M", "K", "L", "Q", "P", "N")
# The wrist crop shows markers and labels only for WRIST_VIEW_NODES.  The
# remaining joints still participate in the true segmented link geometry.
WRIST_MECHANISM_LINKS = MECHANISM_LINKS
WRIST_EXTENSION_LINKS = (
    ("W", "O"), ("W", "P"), ("W", "Q"),
    ("O", "S"), ("P", "T"), ("Q", "U"),
    *EXTENSION_LINKS,
)


# =============================================================================
# 2. 基础几何运算
# =============================================================================
# 这些函数只实现三角形、四边形和空间延长线求解，不含优化逻辑。


def clip_acos(x: float) -> float:
    """抑制浮点舍入导致的 arccos 定义域越界。"""

    return float(np.arccos(np.clip(x, -1.0, 1.0)))


def t_angles(a: float, b: float, c: float) -> Tuple[float, float, float]:
    """由三边求三个内角，并同时检查三角形可装配性。"""

    vals = np.array([a, b, c], dtype=float)
    if np.any(~np.isfinite(vals)) or np.any(vals <= 0):
        raise FourBarError("TAngles invalid positive finite check.")
    if np.sum(vals) - np.max(vals) <= np.max(vals):
        raise FourBarError("TAngles triangle inequality failed.")
    angle_a = clip_acos((b * b + c * c - a * a) / (2 * b * c))
    angle_b = clip_acos((a * a + c * c - b * b) / (2 * a * c))
    angle_c = clip_acos((a * a + b * b - c * c) / (2 * a * b))
    if abs(angle_a + angle_b + angle_c - np.pi) > 1e-5:
        raise FourBarError("TAngles angle sum failed.")
    validate_closure_angle_values(
        (angle_a, angle_b, angle_c),
        context="TAngles",
    )
    return angle_a, angle_b, angle_c


def triangle_third_side_non_adjacent(a: float, b: float, angle_a: float) -> float:
    asin_arg = b * np.sin(angle_a) / a
    vals = np.array([a, b, angle_a, asin_arg], dtype=float)
    if np.any(~np.isfinite(vals)) or a <= 0 or b <= 0 or abs(asin_arg) > 1:
        raise FourBarError("triangleThirdSideNonAdjacent invalid input.")
    angle_b = np.arcsin(asin_arg)
    angle_c = np.pi - angle_a - angle_b
    c = np.sqrt(a * a + b * b - 2 * a * b * np.cos(angle_c))
    if angle_a < 0 or angle_b < 0 or angle_c < 0:
        raise FourBarError("triangleThirdSideNonAdjacent negative process angle.")
    if abs(angle_a + angle_b + angle_c - np.pi) > 1e-5 or not np.isfinite(c):
        raise FourBarError("triangleThirdSideNonAdjacent failed.")
    return float(c)


QA2_MIN_ABSOLUTE_MARGIN_MM = 2.0
QA2_MIN_NORMALIZED_MARGIN = 0.01
CLOSURE_ANGLE_MIN_RAD = float(np.deg2rad(5.0))
CLOSURE_ANGLE_MAX_RAD = float(np.deg2rad(175.0))


def validate_closure_angle_values(
    values: Tuple[float, ...] | list[float] | np.ndarray,
    context: str,
) -> None:
    """Require key closure angles to remain at least 5 deg away from toggle positions."""

    angles = np.asarray(values, dtype=float).reshape(-1)
    if np.any(~np.isfinite(angles)):
        raise FourBarError(f"{context} contains non-finite closure angles.")
    invalid = np.flatnonzero(
        (angles < CLOSURE_ANGLE_MIN_RAD)
        | (angles > CLOSURE_ANGLE_MAX_RAD)
    )
    if invalid.size:
        index = int(invalid[0])
        raise FourBarError(
            f"{context} closure angle {index}={np.rad2deg(angles[index]):.6f} deg "
            "is outside [5, 175] deg."
        )


def qa2_required_clearance_mm(BC: float, CD: float) -> float:
    """Return the engineering clearance required on both diagonal triangle limits."""

    return max(
        QA2_MIN_ABSOLUTE_MARGIN_MM,
        QA2_MIN_NORMALIZED_MARGIN * float(BC + CD),
    )


def qa2(
    AB: float,
    BC: float,
    CD: float,
    DA: float,
    angleA: float,
    context: str = "QA2",
) -> Tuple[float, float, float]:
    """已知四边及 A 角，通过对角线 BD 求其余三个内角。"""

    if AB <= 0 or BC <= 0 or CD <= 0 or DA <= 0:
        raise FourBarError("QA2 non-positive side.")
    validate_closure_angle_values((angleA,), context=f"{context} input")
    bd2 = AB * AB + DA * DA - 2 * AB * DA * np.cos(angleA)
    if bd2 <= 0:
        raise FourBarError("QA2 diagonal invalid.")
    BD = float(np.sqrt(bd2))
    if not np.isfinite(BD) or BD <= 0:
        raise FourBarError(f"{context} diagonal BD is invalid.")
    lower_clearance = BD - abs(BC - CD)
    upper_clearance = BC + CD - BD
    required_clearance = qa2_required_clearance_mm(BC, CD)
    if min(lower_clearance, upper_clearance) < required_clearance:
        raise FourBarError(
            f"{context} diagonal clearance is too small: "
            f"lower={lower_clearance:.6g} mm, upper={upper_clearance:.6g} mm, "
            f"required>={required_clearance:.6g} mm."
        )
    angleB_ABD = clip_acos((AB * AB + BD * BD - DA * DA) / (2 * AB * BD))
    angleD_ABD = clip_acos((BD * BD + DA * DA - AB * AB) / (2 * BD * DA))
    angleB_BDC = clip_acos((BC * BC + BD * BD - CD * CD) / (2 * BC * BD))
    angleC_BDC = clip_acos((BC * BC + CD * CD - BD * BD) / (2 * BC * CD))
    angleD_BDC = clip_acos((BD * BD + CD * CD - BC * BC) / (2 * BD * CD))
    angleB = angleB_ABD + angleB_BDC
    angleC = angleC_BDC
    angleD = angleD_ABD + angleD_BDC
    if angleB >= np.pi or angleC >= np.pi or angleD >= np.pi:
        raise FourBarError("QA2 angle >= pi.")
    if abs(angleA + angleB + angleC + angleD - 2 * np.pi) > 1e-5:
        raise FourBarError("QA2 angle sum failed.")
    validate_closure_angle_values(
        (angleB, angleC, angleD),
        context=f"{context} output",
    )
    return float(angleB), float(angleC), float(angleD)


def local_quadrilateral_angle_b(AB: float, BC: float, CD: float, DA: float, angleA: float) -> float:
    if AB <= 0 or BC <= 0 or CD <= 0 or DA <= 0 or angleA <= 0 or angleA >= np.pi:
        raise FourBarError("localQuadrilateralAngleB invalid input.")
    validate_closure_angle_values(
        (angleA,),
        context="localQuadrilateralAngleB input",
    )
    bd2 = AB * AB + DA * DA - 2 * AB * DA * np.cos(angleA)
    if bd2 <= 0:
        raise FourBarError("localQuadrilateralAngleB diagonal invalid.")
    BD = float(np.sqrt(bd2))
    if (not np.isfinite(BD)) or BD <= 0 or BD >= BC + CD or BD <= abs(BC - CD):
        raise FourBarError("localQuadrilateralAngleB triangle invalid.")
    angleB_ABD = clip_acos((AB * AB + BD * BD - DA * DA) / (2 * AB * BD))
    angleB_BDC = clip_acos((BC * BC + BD * BD - CD * CD) / (2 * BC * BD))
    angleB = angleB_ABD + angleB_BDC
    if (not np.isfinite(angleB)) or angleB >= np.pi:
        raise FourBarError("localQuadrilateralAngleB angle invalid.")
    validate_closure_angle_values(
        (angleB,),
        context="localQuadrilateralAngleB output",
    )
    return float(angleB)


def extend(A: np.ndarray, B: np.ndarray, L: float) -> np.ndarray:
    AB = B - A
    n = np.linalg.norm(AB)
    if (not np.isfinite(n)) or n < 1e-12:
        raise FourBarError("Exten invalid line.")
    return B + AB / n * L


def rexten(A: np.ndarray, B: np.ndarray, C: np.ndarray, theta: float, L: float) -> np.ndarray:
    """在 ABC 定义的空间平面内，从 B 点按 theta 旋转并延长 L。"""

    AB = B - A
    BC = C - B
    ab_norm = np.linalg.norm(AB)
    bc_norm = np.linalg.norm(BC)
    if ab_norm < 1e-9 or bc_norm < 1e-9:
        raise FourBarError("RExten AB too short.")
    ab_unit = AB / ab_norm
    normal = np.cross(AB, BC)
    normal_norm = np.linalg.norm(normal)
    # 腕部空间延长方向由 ABC 平面法向量确定。接近共线时，即使叉积仍非零，
    # 归一化也会放大舍入误差并使法向量跨帧翻转，造成 tip 瞬时跳跃。
    plane_sine = normal_norm / (ab_norm * bc_norm)
    if (not np.isfinite(plane_sine)) or plane_sine < 0.02:
        raise FourBarError(
            f"RExten plane is near-collinear (normalized sine={plane_sine:.6g})."
        )
    normal = normal / normal_norm
    v_unit = ab_unit * np.cos(theta) + normal * np.sin(theta)
    v_norm = np.linalg.norm(v_unit)
    if v_norm < 1e-12 or not np.isfinite(v_norm):
        raise FourBarError("RExten direction invalid.")
    return B + v_unit / v_norm * L


def extend_line_3d(P1: np.ndarray, P2: np.ndarray, L: float) -> np.ndarray:
    d = P2 - P1
    n = np.linalg.norm(d)
    if (not np.isfinite(n)) or n < 1e-12:
        raise FourBarError("extendLine3D invalid line.")
    return P2 + L * d / n


def bcurve_to_r_theta(b_curve: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把实际 B 轨迹转换为以 A 为原点的逐帧半径与两个方向角。"""

    b = np.asarray(b_curve, dtype=float)
    if b.ndim != 2 or b.shape[1] != 3:
        raise FourBarError("b_curve must have shape (n, 3).")
    if np.any(~np.isfinite(b)):
        raise FourBarError("B coordinates must be finite.")
    r = np.linalg.norm(b, axis=1)
    if float(np.min(r)) <= 1e-9:
        raise FourBarError("B must remain away from A.")
    radial = np.linalg.norm(b[:, [0, 2]], axis=1)
    theta01 = np.arctan2(radial, -b[:, 1])
    theta02 = np.unwrap(np.arctan2(-b[:, 2], b[:, 0]))
    return r, theta01, theta02


def bcurve_to_input_angles(
    b_curve: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """由真实 B(t) 直接得到可变 AB(t)、theta01 与 theta02。

    B 是机构节点本身，不再先通过独立 Mot 点求方向、再用固定 L1 重建。
    第四个返回量保留旧接口形状，等于 AB(t)。
    """
    input_radius, theta01, theta02 = bcurve_to_r_theta(b_curve)
    return input_radius, theta01, theta02, input_radius.copy()


def motcurve_to_input_angles(
    mot_curve: np.ndarray,
    ab_length: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """旧名称兼容层；输入数组在新模型中直接表示 B(t)，ab_length 被忽略。"""
    return bcurve_to_input_angles(mot_curve)


def input_angles_to_node_b(
    input_radius: np.ndarray,
    theta01: np.ndarray,
    theta02: np.ndarray,
) -> np.ndarray:
    """按核心求解器的旋转约定重建节点 B；仅用于结果输出和兼容检查。"""
    radius = np.asarray(input_radius, dtype=float).reshape(-1)
    angle1 = np.asarray(theta01, dtype=float).reshape(-1)
    angle2 = np.asarray(theta02, dtype=float).reshape(-1)
    planar_x = radius * np.sin(angle1)
    return np.column_stack([
        planar_x * np.cos(angle2),
        -radius * np.cos(angle1),
        -planar_x * np.sin(angle2),
    ])


def motcurve_to_b_input(
    mot_curve: np.ndarray,
    ab_length: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """旧名称兼容层；新模型返回输入 B(t) 本身。"""
    b_curve = np.asarray(mot_curve, dtype=float)
    input_radius, theta01, theta02, input_norm = bcurve_to_input_angles(b_curve)
    return input_radius, theta01, theta02, b_curve.copy(), input_norm


def resample_rows(arr: np.ndarray, n_rows: int) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.shape[0] == n_rows:
        return a.copy()
    x_old = np.linspace(0.0, 1.0, a.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_rows, endpoint=False)
    if a.ndim == 1:
        return np.interp(x_new, x_old, a, period=1.0)
    out = np.zeros((n_rows, a.shape[1]))
    for j in range(a.shape[1]):
        out[:, j] = np.interp(x_new, x_old, a[:, j], period=1.0)
    return out


def negative_angle_report(
    angle_history: Mapping[str, np.ndarray],
    tolerance_rad: float = 1e-10,
) -> list[Dict[str, Any]]:
    """Return every mechanism/process angle that becomes negative."""
    report: list[Dict[str, Any]] = []
    for name, values in angle_history.items():
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 0 or np.any(~np.isfinite(array)):
            raise FourBarError(f"{name} contains empty or non-finite angle values")
        negative_indices = np.flatnonzero(array < -abs(float(tolerance_rad)))
        if negative_indices.size:
            minimum_index = int(np.argmin(array))
            report.append({
                "name": str(name),
                "negative_count": int(negative_indices.size),
                "first_negative_frame": int(negative_indices[0]),
                "minimum_frame": minimum_index,
                "minimum_rad": float(array[minimum_index]),
                "minimum_deg": float(np.rad2deg(array[minimum_index])),
            })
    return report


def validate_nonnegative_angles(angle_history: Mapping[str, np.ndarray]) -> None:
    """Reject a mechanism when an angle required to be nonnegative is negative."""
    report = negative_angle_report(angle_history)
    if report:
        details = "; ".join(
            f"{item['name']} min={item['minimum_deg']:.3f} deg "
            f"at frame {item['minimum_frame']} ({item['negative_count']} negative frames)"
            for item in report
        )
        raise FourBarError(f"negative mechanism angle(s): {details}")


def validate_theta6_upper_bound(
    theta6: np.ndarray,
    maximum_rad: float = np.pi,
    tolerance_rad: float = 1e-10,
) -> None:
    """要求每一帧的 theta6 严格小于 180 度。"""

    angle = np.asarray(theta6, dtype=float).reshape(-1)
    if angle.size == 0 or np.any(~np.isfinite(angle)):
        raise FourBarError("theta6 contains empty or non-finite angle values")
    invalid = np.flatnonzero(angle >= float(maximum_rad) - abs(float(tolerance_rad)))
    if invalid.size:
        maximum_frame = int(np.argmax(angle))
        raise FourBarError(
            "theta6 must be strictly smaller than 180 deg: "
            f"max={np.rad2deg(angle[maximum_frame]):.6f} deg at frame "
            f"{maximum_frame} ({invalid.size} invalid frames)"
        )


def validate_angle_continuity(
    angle_history: Mapping[str, np.ndarray],
    max_step_rad: float = 0.45,
) -> None:
    """检查所有机构角在圆周意义下的逐帧与周期首尾连续性。

    角度差先映射到 ``[-pi, pi]``，因此正常跨越 ``pi/-pi`` 不会被误判；
    真正的装配分支跳转或不连续变化会因圆周步长过大而被拒绝。
    """

    for name, values in angle_history.items():
        angle = np.asarray(values, dtype=float).reshape(-1)
        if angle.size < 3 or np.any(~np.isfinite(angle)):
            raise FourBarError(f"{name} angle history is not finite or is too short.")
        raw_step = np.diff(np.concatenate([angle, angle[:1]]))
        circular_step = np.arctan2(np.sin(raw_step), np.cos(raw_step))
        maximum = float(np.max(np.abs(circular_step)))
        if maximum > max_step_rad:
            frame = int(np.argmax(np.abs(circular_step)))
            raise FourBarError(
                f"{name} angular discontinuity at frame {frame}: "
                f"circular step={maximum:.6g} rad > {max_step_rad:.6g} rad."
            )


STUTTER_STALL_RATIO_LIMIT = 0.002
STUTTER_ADJACENT_SPEED_RATIO_LIMIT = 10.0
STUTTER_JERK_PEAK_LIMIT = 6.0
STUTTER_PENALTY_WEIGHT_MM = 2.0


def curve_stutter_metrics(curve: np.ndarray) -> Dict[str, float]:
    """量化原始等时间帧轨迹中的停滞、速度骤变、加速度和 jerk。

    这里故意使用机构求解器输出的原始帧，而不是等弧长后的评分曲线。这样
    等弧长重采样只能规范几何对应，不能掩盖真实运动中的重复帧或卡顿。
    """

    points = np.asarray(curve, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 5:
        raise FourBarError(f"stutter curve must have shape (N,3), got {points.shape}")
    if np.any(~np.isfinite(points)):
        raise FourBarError("stutter curve contains non-finite values")
    velocity = np.roll(points, -1, axis=0) - points
    speed = np.linalg.norm(velocity, axis=1)
    positive = speed[speed > 1e-12]
    if positive.size == 0:
        raise FourBarError("stutter curve has no motion")
    median_speed = float(np.median(positive))
    mean_speed = float(np.mean(speed))
    if (not np.isfinite(median_speed)) or median_speed <= 1e-9:
        raise FourBarError("stutter curve has zero median speed")
    normalized_speed = speed / median_speed
    ratio_floor = STUTTER_STALL_RATIO_LIMIT
    safe_speed = np.maximum(normalized_speed, ratio_floor)
    adjacent_log_ratio = np.abs(np.log(safe_speed / np.roll(safe_speed, 1)))
    adjacent_speed_ratio = np.exp(adjacent_log_ratio)
    acceleration = velocity - np.roll(velocity, 1, axis=0)
    jerk = acceleration - np.roll(acceleration, 1, axis=0)
    acceleration_norm = np.linalg.norm(acceleration, axis=1) / median_speed
    jerk_norm = np.linalg.norm(jerk, axis=1) / median_speed
    stall_fraction = float(np.mean(normalized_speed < 0.05))
    adjacent_log_rms = float(np.sqrt(np.mean(adjacent_log_ratio ** 2)))
    jerk_rms = float(np.sqrt(np.mean(jerk_norm ** 2)))
    stutter_index = float(
        adjacent_log_rms + 0.25 * jerk_rms + 2.0 * stall_fraction
    )
    return {
        "median_step_mm": median_speed,
        "mean_step_mm": mean_speed,
        "minimum_speed_ratio": float(np.min(normalized_speed)),
        "speed_cv": float(np.std(speed) / max(mean_speed, 1e-12)),
        "stall_fraction": stall_fraction,
        "adjacent_speed_log_rms": adjacent_log_rms,
        "maximum_adjacent_speed_ratio": float(np.max(adjacent_speed_ratio)),
        "acceleration_rms_normalized": float(np.sqrt(np.mean(acceleration_norm ** 2))),
        "jerk_rms_normalized": jerk_rms,
        "jerk_peak_normalized": float(np.max(jerk_norm)),
        "stutter_index": stutter_index,
    }


def combined_stutter_metrics(tip: np.ndarray, wrist: np.ndarray) -> Dict[str, float]:
    """Return prefixed Tip/Wrist stutter diagnostics and a continuous penalty."""

    tip_metrics = curve_stutter_metrics(tip)
    wrist_metrics = curve_stutter_metrics(wrist)
    output: Dict[str, float] = {}
    output.update({f"tip_{key}": value for key, value in tip_metrics.items()})
    output.update({f"wrist_{key}": value for key, value in wrist_metrics.items()})
    maximum_index = max(
        float(tip_metrics["stutter_index"]),
        float(wrist_metrics["stutter_index"]),
    )
    output["stutter_index"] = maximum_index
    output["stutter_penalty_mm"] = STUTTER_PENALTY_WEIGHT_MM * maximum_index
    return output


def check_smoothness(tip: np.ndarray, wrist: np.ndarray) -> None:
    """排除无运动、离散跳变、重复帧和明显的停走式卡顿。"""

    for name, curve in (("Tip", tip), ("Wrist", wrist)):
        if curve.ndim != 2 or curve.shape[1] != 3 or curve.shape[0] < 5:
            raise FourBarError(f"{name}TooFewPoints")
        if not np.all(np.isfinite(curve)):
            raise FourBarError(f"{name}NonFinite")
        # 把末帧到首帧也纳入检查，防止周期边界存在隐藏跳跃。
        seg = np.diff(np.vstack([curve, curve[:1]]), axis=0)
        speed = np.linalg.norm(seg, axis=1)
        positive = speed[np.isfinite(speed) & (speed > 0)]
        if positive.size == 0:
            raise FourBarError(f"{name}NoMotion")
        med = float(np.median(positive))
        if (not np.isfinite(med)) or med < 1e-6:
            raise FourBarError(f"{name}NoMotion")
        if not np.all(np.isfinite(speed)):
            raise FourBarError(f"{name}VelocityNonFinite")
        span = float(np.linalg.norm(np.ptp(curve, axis=0)))
        maximum = float(np.max(speed))
        # 同时参考典型步长和完整轨迹尺度；这会排除法向翻转造成的离散跳跃，
        # 但保留真实机构在局部速度较高的连续运动。
        allowed = max(12.0 * med, 0.35 * max(span, 1e-9))
        if maximum > allowed:
            frame = int(np.argmax(speed))
            raise FourBarError(
                f"{name}VelocityTooLarge at frame {frame}: "
                f"step={maximum:.6g} mm, allowed={allowed:.6g} mm."
            )
        stutter = curve_stutter_metrics(curve)
        if stutter["minimum_speed_ratio"] < STUTTER_STALL_RATIO_LIMIT:
            frame = int(np.argmin(speed))
            raise FourBarError(
                f"{name}Stall at frame {frame}: speed/median="
                f"{stutter['minimum_speed_ratio']:.6g} < "
                f"{STUTTER_STALL_RATIO_LIMIT:.6g}."
            )
        if stutter["maximum_adjacent_speed_ratio"] > STUTTER_ADJACENT_SPEED_RATIO_LIMIT:
            raise FourBarError(
                f"{name}AbruptSpeedChange: adjacent ratio="
                f"{stutter['maximum_adjacent_speed_ratio']:.6g} > "
                f"{STUTTER_ADJACENT_SPEED_RATIO_LIMIT:.6g}."
            )
        if (
            stutter["jerk_peak_normalized"] > STUTTER_JERK_PEAK_LIMIT
            and stutter["maximum_adjacent_speed_ratio"] > 3.0
        ):
            raise FourBarError(
                f"{name}JerkSpike: normalized peak="
                f"{stutter['jerk_peak_normalized']:.6g}."
            )


# =============================================================================
# 3. 机构核心求解器
# =============================================================================


def fourbar_direct_b_l6(
    params: FourBarParams,
    b_curve: np.ndarray,
    l6_values: np.ndarray | None = None,
    wrist_node: str = "L",
    check_smooth: bool = True,
    time_varying_lengths: Mapping[str, np.ndarray] | None = None,
) -> FourBarResult:
    """由真实 B(t) 和可选的逐帧杆长计算机构运动。

    该函数是优化器调用的稳定入口。任何不可装配状态都会被转换为
    ``FourBarResult(valid=False)``，从而让优化器把该候选点视为无效解。

    ``l6_values`` 是历史 L6(t) 的兼容入口；新正式模式使用固定 L6。
    当前仅允许 L3(t)、L8(t) 作为周期长度输入；L5 必须在整个周期内固定。C 与 Z 是同一个
    物理节点，因而非零 ``L_CZ`` 输入会被明确拒绝。
    """

    try:
        return _fourbar_direct_b_l6(params, b_curve, l6_values, wrist_node, check_smooth, time_varying_lengths)
    except Exception as exc:
        empty = np.zeros((0, 3))
        return FourBarResult(
            valid=False,
            message=str(exc),
            tip=empty,
            wrist=empty,
            nodes=np.zeros((0, len(NODE_NAMES), 3)),
            node_names=NODE_NAMES,
            theta_wrist=np.zeros(0),
            theta20=np.zeros(0),
            theta21=np.zeros(0),
            thetaM=np.zeros(0),
            thetaN=np.zeros(0),
            angle_history={},
            mot_curve=np.asarray(b_curve),
            b_curve=empty,
            input_radius=np.zeros(0),
            mot_radius=np.zeros(0),
            theta01=np.zeros(0),
            theta02=np.zeros(0),
            l2_values=np.full(76, float(params.L2), dtype=float),
            l6_values=np.full(76, float(params.L6), dtype=float),
            l5_values=np.full(76, float(params.L5), dtype=float),
            l31_values=np.full(76, float(params.L31), dtype=float),
            l32_values=np.full(76, float(params.L3 - params.L31 + 2.0), dtype=float),
            l3_values=np.full(76, float(params.L3), dtype=float),
            zc_values=np.zeros(76, dtype=float),
            zc_start_index=None,
        )


def _fourbar_direct_b_l6(
    p: FourBarParams,
    supplied_b_curve: np.ndarray,
    supplied_l6_values: np.ndarray | None,
    wrist_node: str,
    check_smooth: bool,
    time_varying_lengths: Mapping[str, np.ndarray] | None = None,
) -> FourBarResult:
    """核心几何实现；外部应优先调用 fourbar_direct_b_l6。"""

    # 所有机构输入统一到 76 个等时间步。目标误差使用预测点到周期目标
    # 折线的最近距离，因此 76 帧只定义机构采样，不再绑定目标相位索引。
    b_curve = resample_rows(np.asarray(supplied_b_curve, dtype=float), 76)
    length_curves: dict[str, np.ndarray] = {}

    # 统一读取逐帧杆长；显式映射优先于兼容位置参数。
    if time_varying_lengths:
        for name, values in time_varying_lengths.items():
            if not hasattr(p, name):
                raise FourBarError(f"unknown time-varying length: {name}")
            curve = resample_rows(np.asarray(values, dtype=float).reshape(-1), 76).reshape(-1)
            if np.any(~np.isfinite(curve)):
                raise FourBarError(f"{name}(t) must be finite.")
            if str(name) == "L_CZ":
                if float(np.max(np.abs(curve))) > 1e-10:
                    raise FourBarError(
                        "CZ separation is disabled; legacy L_CZ(t) must be identically zero."
                    )
                continue
            if str(name) == "L32":
                raise FourBarError("L32(t) is derived from L3(t)-L31+2 and cannot be supplied independently.")
            if np.any(curve <= 0):
                raise FourBarError(f"{name}(t) must be positive finite.")
            if str(name) not in {"L3", "L8"} and float(np.ptp(curve)) > 1e-10:
                raise FourBarError(
                    f"{name} must be constant; only L3(t) and L8(t) may vary."
                )
            length_curves[str(name)] = curve
    if supplied_l6_values is not None and "L6" not in length_curves:
        curve = resample_rows(
            np.asarray(supplied_l6_values, dtype=float).reshape(-1), 76
        ).reshape(-1)
        if np.any(~np.isfinite(curve)) or np.any(curve <= 0):
            raise FourBarError("L6(t) must be positive finite.")
        length_curves["L6"] = curve

    # 在进入逐帧求解前先检查与机构装配直接相关的硬约束。
    derived_l32 = p.L3 - p.L31 + 2.0
    p = replace(p, L32=derived_l32)
    extra = np.array([p.L2, p.L3, p.L31, derived_l32, p.L61, p.Lf1, p.Lf2], dtype=float)
    if np.any(~np.isfinite(extra)) or np.any(extra <= 0):
        raise FourBarError("L2/L3/L31/L32/L61/Lf1/Lf2 must be positive finite.")
    if p.L51 > p.L5 / 3.0:
        raise FourBarError("require L51 <= L5/3.")
    if p.L52 > p.L5 / 3.0:
        raise FourBarError("require L52 <= L5/3.")
    if p.L4 < p.L41:
        raise FourBarError("require L4 >= L41.")

    wrist_rod_effective = p.LRod if p.LRod is not None and np.isfinite(p.LRod) else p.L12
    wrist_down_effective = p.L_down

    # B(t) 是真实机构输入；AB(t) 为其到固定点 A 的逐帧距离。
    input_radius, theta01, theta02, input_norm = bcurve_to_input_angles(b_curve)
    n = len(input_radius)
    l2_curve = length_curves.get("L2")
    if l2_curve is None:
        l2_curve = np.full(n, p.L2, dtype=float)
    l31_curve = length_curves.get("L31")
    if l31_curve is None:
        l31_curve = np.full(n, p.L31, dtype=float)
    l3_curve = length_curves.get("L3")
    if l3_curve is None:
        l3_curve = np.full(n, p.L3, dtype=float)
    l5_curve = length_curves.get("L5")
    if l5_curve is None:
        l5_curve = np.full(n, p.L5, dtype=float)
    if float(np.ptp(l5_curve)) > 1e-10:
        raise FourBarError("L5 must be one fixed optimization length over the full cycle.")
    # L32 不是独立自由度；随 L3(t) 逐帧派生，L31 在当前模式中为固定杆长。
    l32_curve = l3_curve - l31_curve + 2.0
    if np.any(~np.isfinite(l32_curve)) or float(np.min(l32_curve)) <= 0.0:
        raise FourBarError("require L32(t)=L3(t)-L31(t)+2 > 0 for every frame.")
    p = replace(p, L32=float(l32_curve[0]))
    triangle_sides = np.column_stack([l31_curve, l32_curve, l3_curve])
    longest_side = np.max(triangle_sides, axis=1)
    if np.any(np.sum(triangle_sides, axis=1) - longest_side <= longest_side):
        raise FourBarError(
            "L31, L32 and L3(t) must satisfy the triangle inequality for every frame."
        )
    if float(np.min(l5_curve)) <= 0.0:
        raise FourBarError("require L5(t) > 0 for every frame.")
    if p.L51 > float(np.min(l5_curve)) / 3.0:
        raise FourBarError("require L51 <= min(L5(t))/3.")
    if p.L52 > float(np.min(l5_curve)) / 3.0:
        raise FourBarError("require L52 <= min(L5(t))/3.")
    if float(np.max(l5_curve - p.L51 - p.L52 - l32_curve)) > 1e-8:
        raise FourBarError("require L5(t)-L51-L52 <= L32(t) for every frame.")
    l6_curve = length_curves.get("L6")
    if l6_curve is None:
        l6_curve = np.full(n, p.L6, dtype=float)
    l7_curve = length_curves.get("L7")
    if l7_curve is None:
        l7_curve = np.full(n, p.L7, dtype=float)
    l8_curve = length_curves.get("L8")
    if l8_curve is None:
        l8_curve = np.full(n, p.L8, dtype=float)
    if float(np.min(l6_curve)) <= p.L61:
        raise FourBarError("require L6(t) > L61 for every frame.")

    # 固定 BC=L2 仍会因 B(t) 改变而使 C 沿 A-B-C 三角形运动。
    # 当前拓扑中 C 与 Z 是同一个物理节点，不再计算分离相位或分离距离。
    Cy = np.array([
        triangle_third_side_non_adjacent(
            float(l2_curve[i]), float(input_radius[i]), float(theta01[i])
        )
        for i in range(n)
    ], dtype=float)
    # C、Z 完全合并：保留 Z 数组槽位只为兼容旧结果文件，任何下游几何都
    # 使用同一组坐标，不再存在分离相位、最低点基准或独立 ZC 长度。
    zc_curve = np.zeros(n, dtype=float)
    zc_start_index = None

    theta2 = np.zeros(n)
    theta3 = np.zeros(n)
    theta4 = np.zeros(n)
    theta5 = np.zeros(n)
    theta6 = np.zeros(n)
    theta7 = np.zeros(n)
    theta8 = np.zeros(n)
    theta9 = np.zeros(n)
    theta10 = np.zeros(n)
    theta11 = np.zeros(n)
    theta12 = np.zeros(n)
    theta13 = np.zeros(n)
    theta14 = np.zeros(n)
    theta15 = np.zeros(n)
    theta16 = np.zeros(n)
    theta17 = np.zeros(n)
    theta18 = np.full(n, np.deg2rad(p.theta18_deg), dtype=float)
    theta19 = np.zeros(n)
    theta20 = np.zeros(n)
    theta21 = np.zeros(n)
    theta_m = np.zeros(n)
    theta_n = np.zeros(n)
    alpha_values = np.zeros(n)
    theta_jmj_down = np.zeros(n)
    theta_o = np.zeros(n)
    theta_wrist = np.zeros(n)
    theta31 = np.zeros(n)
    theta32 = np.zeros(n)
    theta33 = np.zeros(n)
    AC = np.zeros(n)

    # 第一遍：逐帧求解闭环机构角度。角度求解成功后再构建节点坐标，便于定位错误。
    for i in range(n):
        L6i = float(l6_curve[i])
        L2i = float(l2_curve[i])
        L31i = float(l31_curve[i])
        L3i = float(l3_curve[i])
        L5i = float(l5_curve[i])
        L32i = float(l32_curve[i])
        L7i = float(l7_curve[i])
        L8i = float(l8_curve[i])
        theta31[i], theta32[i], theta33[i] = t_angles(L31i, L32i, L3i)
        # A-B-C 中 BC=L2 为固定长度；Z 与 C 重合，后续主闭环直接使用 A-C-D。
        Cyi = float(Cy[i])
        ACi = float(Cyi)
        theta4[i], theta2[i], theta3[i] = t_angles(ACi, p.L4, L31i)
        theta5[i] = 2 * np.pi - theta4[i] - theta33[i]
        theta6[i], theta7[i], theta8[i] = qa2(
            p.L41, L5i - p.L51 - p.L52, L6i - p.L61, L32i, theta5[i],
            context=f"EFGL frame {i}",
        )
        theta9[i] = np.pi - theta7[i]
        theta10[i], theta11[i], theta12[i] = qa2(
            p.L52, L7i, p.L10, L6i, theta9[i],
            context=f"FHKL frame {i}",
        )
        theta13[i] = np.pi - theta10[i]
        theta14[i], theta15[i], theta16[i] = qa2(
            p.L51, L8i, p.L9, L7i, theta13[i],
            context=f"HIJK frame {i}",
        )
        theta21[i] = 2 * np.pi - theta11[i] - theta16[i] - theta18[i]
        theta17[i], theta20[i], theta19[i] = qa2(
            p.L10, p.L11, p.L13, p.L12, theta21[i],
            context=f"KLMN frame {i}",
        )
        Cy[i] = Cyi
        AC[i] = ACi

    # tip、wrist 和全部 A...Z 节点的输出缓存。
    pos_u = np.zeros((n, 3))
    pos_w = np.zeros((n, 3))
    nodes = np.zeros((n, len(NODE_NAMES), 3))

    A = IDX["A"]; B = IDX["B"]; C = IDX["C"]; D = IDX["D"]; E = IDX["E"]
    F = IDX["F"]; G = IDX["G"]; H = IDX["H"]; I = IDX["I"]; J = IDX["J"]
    K = IDX["K"]; L = IDX["L"]; M = IDX["M"]; N = IDX["N"]; O = IDX["O"]
    Pidx = IDX["P"]; Q = IDX["Q"]; S = IDX["S"]; T = IDX["T"]; U = IDX["U"]
    W = IDX["W"]; Z = IDX["Z"]

    # 第二遍：先在局部二维机构平面内建立 A...N，再旋转到三维空间并构建末端节点。
    for i in range(n):
        Position = np.zeros((len(NODE_NAMES), 3), dtype=float)
        L6i = float(l6_curve[i])
        L2i = float(l2_curve[i])
        L31i = float(l31_curve[i])
        L3i = float(l3_curve[i])
        L5i = float(l5_curve[i])
        L7i = float(l7_curve[i])
        L8i = float(l8_curve[i])
        # 主闭环 A-B-C-D-E-F-G-H-I-J-K-L-M-N 的平面坐标。
        Position[A, :2] = [0.0, 0.0]
        Position[B, :2] = [input_radius[i] * np.sin(theta01[i]), -input_radius[i] * np.cos(theta01[i])]
        Position[C, :2] = [0.0, -Cy[i]]
        Position[Z, :] = Position[C, :]
        Position[D, :2] = [-L31i * np.sin(theta2[i]), -L31i * np.cos(theta2[i])]
        triangle_base = Position[C, :]
        Position[E, :2] = [
            triangle_base[0] - (p.L4 - p.L41) * np.sin(theta3[i]),
            triangle_base[1] + (p.L4 - p.L41) * np.cos(theta3[i]),
        ]
        alpha_values[i] = np.pi - theta6[i] - theta3[i]
        Position[F, :2] = [
            Position[E, 0] - np.sin(alpha_values[i]) * (L5i - p.L51 - p.L52),
            Position[E, 1] - np.cos(alpha_values[i]) * (L5i - p.L51 - p.L52),
        ]
        Position[G, :2] = [
            -L3i * np.sin(theta2[i] + theta32[i]),
            -L3i * np.cos(theta2[i] + theta32[i]),
        ]
        Position[H, :2] = [
            Position[E, 0] - np.sin(alpha_values[i]) * (L5i - p.L51),
            Position[E, 1] - np.cos(alpha_values[i]) * (L5i - p.L51),
        ]
        Position[I, :2] = [
            Position[E, 0] - np.sin(alpha_values[i]) * L5i,
            Position[E, 1] - np.cos(alpha_values[i]) * L5i,
        ]
        Position[J, :2] = [
            Position[I, 0] - np.sin(theta14[i] - alpha_values[i]) * L8i,
            Position[I, 1] + np.cos(theta14[i] - alpha_values[i]) * L8i,
        ]
        Position[L, :2] = [
            Position[F, 0] - np.sin(theta7[i] - alpha_values[i]) * L6i,
            Position[F, 1] + np.cos(theta7[i] - alpha_values[i]) * L6i,
        ]
        Position[K, :2] = [
            Position[H, 0] - np.sin(theta10[i] - alpha_values[i]) * L7i,
            Position[H, 1] + np.cos(theta10[i] - alpha_values[i]) * L7i,
        ]

        theta_m[i] = 2 * np.pi - theta18[i] - theta16[i] - (2 * np.pi - theta9[i] - theta6[i] - theta3[i])
        Position[M, :2] = [
            Position[K, 0] - np.sin(theta_m[i]) * p.L12,
            Position[K, 1] + np.cos(theta_m[i]) * p.L12,
        ]
        theta_n[i] = 2 * np.pi - theta17[i] - theta12[i] - (2 * np.pi - theta7[i] - theta6[i] - theta3[i])
        Position[N, :2] = [
            Position[L, 0] - np.sin(theta_n[i]) * p.L11,
            Position[L, 1] + np.cos(theta_n[i]) * p.L11,
        ]

        # theta_wrist 可先由局部二维 M-J-K 几何求得；theta02 是刚体旋转，
        # 不改变这里使用的长度和内积，因此先求角度与三维展开后再求完全等价。
        MK = Position[K, :] - Position[M, :]
        MJ = Position[J, :] - Position[M, :]
        denom = float(np.linalg.norm(MK))
        if denom <= 1e-12:
            raise FourBarError("invalid MK denominator.")
        d_MJ = float(np.dot(MJ, MK) / denom)
        MJ_down = float(np.sqrt(d_MJ * d_MJ + p.L_down * p.L_down))
        theta_jmj_down[i] = clip_acos(d_MJ / MJ_down)
        theta_o[i] = local_quadrilateral_angle_b(
            p.L14, MJ_down, wrist_rod_effective, wrist_down_effective, np.pi / 2
        )
        theta_wist = np.pi - theta_o[i] - theta_jmj_down[i]
        theta_wrist[i] = theta_wist

        if not np.isfinite(theta_wist):
            raise FourBarError("theta_wist not finite.")

        # 先根据 theta02 把 A...N（包括 M/N）从局部二维平面展开到三维。
        # 随后 theta_wrist 只作用于 M/N 之后构造的 O/P/Q/W/S/T/U。
        x_before = Position[:, 0].copy()
        Position[:, 2] = -x_before * np.sin(theta02[i])
        Position[:, 0] = x_before * np.cos(theta02[i])
        # 与 RMSE29.99 CMA-ES 模型一致，三维展开后再次强制 Z=C。
        Position[Z, :] = Position[C, :]

        # 在 A...N 完成三维展开后，才用腕部折转角构造后续节点；最终 tip=U。
        Position[W, :] = rexten(Position[L, :], Position[N, :], Position[K, :], -theta_wist, p.L17)
        Position[O, :] = rexten(Position[K, :], Position[M, :], Position[L, :], theta_wist, p.L14)
        Position[Pidx, :] = rexten(Position[K, :], Position[M, :], Position[L, :], theta_wist, p.L14 + p.L15)
        Position[Q, :] = rexten(Position[K, :], Position[M, :], Position[L, :], theta_wist, p.L14 + p.L15 + p.L16)
        Position[U, :] = extend(Position[W, :], Position[Q, :], p.Lf2)
        Position[T, :] = extend(Position[W, :], Position[Pidx, :], p.Lf2)
        Position[S, :] = extend(Position[W, :], Position[O, :], p.Lf2)

        # 当前研究口径固定 wrist=L；保留 N 仅用于旧结果回归检查。
        pos_w[i, :] = Position[L, :] if wrist_node.upper() == "L" else Position[N, :]
        pos_u[i, :] = Position[U, :]
        nodes[i, :, :] = Position

    angle_history = {
        "theta01": theta01,
        "theta2": theta2,
        "theta3": theta3,
        "theta4": theta4,
        "theta5": theta5,
        "theta6": theta6,
        "theta7": theta7,
        "theta8": theta8,
        "theta9": theta9,
        "theta10": theta10,
        "theta11": theta11,
        "theta12": theta12,
        "theta13": theta13,
        "theta14": theta14,
        "theta15": theta15,
        "theta16": theta16,
        "theta17": theta17,
        "theta18": theta18,
        "theta19": theta19,
        "theta20": theta20,
        "theta21": theta21,
        "theta31": theta31,
        "theta32": theta32,
        "theta33": theta33,
        "alpha": alpha_values,
        "thetaM": theta_m,
        "thetaN": theta_n,
        "theta_JMJ_down": theta_jmj_down,
        "theta_O": theta_o,
        "theta_wrist": theta_wrist,
    }
    # theta02 is a signed 3D orientation coordinate; its sign indicates rotation
    # direction rather than a negative physical joint angle.
    # theta_wrist 是腕部相对主机构平面的带符号折转角：正值表示向下折转，
    # 负值表示向上折转。它不属于必须非负的闭环内角，因此只从非负筛选中排除
    # theta_wrist；其余机构角仍执行原有硬约束。
    validate_nonnegative_angles({
        name: values for name, values in angle_history.items()
        if name != "theta_wrist"
    })
    validate_theta6_upper_bound(theta6)
    validate_angle_continuity({
        **angle_history,
        "theta02": theta02,
    })

    if check_smooth:
        check_smoothness(pos_u, pos_w)

    solved_b_curve = nodes[:, B, :].copy()
    if float(np.max(np.linalg.norm(solved_b_curve - b_curve, axis=1))) > 1e-7:
        raise FourBarError("solved node B is inconsistent with the prescribed B input curve")
    # 逐帧核验参数名与真实节点距离，防止 L2/L3 对应关系再次被交换。
    actual_ab = np.linalg.norm(nodes[:, B, :] - nodes[:, A, :], axis=1)
    actual_bc = np.linalg.norm(nodes[:, C, :] - nodes[:, B, :], axis=1)
    actual_ad = np.linalg.norm(nodes[:, D, :] - nodes[:, A, :], axis=1)
    actual_ag = np.linalg.norm(nodes[:, G, :] - nodes[:, A, :], axis=1)
    actual_dg = np.linalg.norm(nodes[:, G, :] - nodes[:, D, :], axis=1)
    actual_ac = np.linalg.norm(nodes[:, C, :] - nodes[:, A, :], axis=1)
    actual_cz = np.linalg.norm(nodes[:, Z, :] - nodes[:, C, :], axis=1)
    actual_ei = np.linalg.norm(nodes[:, I, :] - nodes[:, E, :], axis=1)
    actual_op = np.linalg.norm(nodes[:, Pidx, :] - nodes[:, O, :], axis=1)
    actual_pq = np.linalg.norm(nodes[:, Q, :] - nodes[:, Pidx, :], axis=1)
    expected_lengths = {
        "AB": (actual_ab, input_radius),
        "BC=L2": (actual_bc, l2_curve),
        "AD=L31": (actual_ad, l31_curve),
        "AG=L3": (actual_ag, l3_curve),
        "DG=L32": (actual_dg, l32_curve),
        "AC": (actual_ac, AC),
        "EI=L5": (actual_ei, l5_curve),
        "OP=L15": (actual_op, np.full(n, p.L15, dtype=float)),
        "PQ=L16=PO": (actual_pq, np.full(n, p.L15, dtype=float)),
    }
    if float(np.max(actual_cz)) > 1e-10:
        raise FourBarError("CZ separation is disabled; node Z must coincide with node C.")
    for label, (actual, expected) in expected_lengths.items():
        maximum_error = float(np.max(np.abs(actual - expected)))
        if maximum_error > 1e-6:
            raise FourBarError(
                f"link-length identity failed for {label}: max error={maximum_error:.6g} mm"
            )

    return FourBarResult(
        valid=True,
        message="ok",
        tip=pos_u,
        wrist=pos_w,
        nodes=nodes,
        node_names=NODE_NAMES,
        theta_wrist=theta_wrist,
        theta20=theta20,
        theta21=theta21,
        thetaM=theta_m,
        thetaN=theta_n,
        angle_history=angle_history,
        # mot_curve 仅作为旧文件字段保留；其数值与真实 B 输入完全相同。
        mot_curve=b_curve.copy(),
        b_curve=solved_b_curve,
        input_radius=input_radius,
        mot_radius=input_norm,
        theta01=theta01,
        theta02=theta02,
        l2_values=l2_curve,
        l6_values=l6_curve,
        l5_values=l5_curve,
        l31_values=l31_curve,
        l32_values=l32_curve,
        l3_values=l3_curve,
        l7_values=l7_curve,
        l8_values=l8_curve,
        zc_values=zc_curve,
        zc_start_index=zc_start_index,
    )


# =============================================================================
# 4. 固定初始化点的严格等弧长相位误差
# =============================================================================


STRICT_CORRESPONDENCE_MODE = "strict_initialized_equal_arc_index"


def resample_closed_curve_equal_arclength(
    curve: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    """从给定首点开始，沿闭合三维折线按等弧长重采样。

    首点定义相位零点，函数不会搜索锚点、循环滚动或反转方向。最后一个采样点
    位于闭合段终点之前，因此输出仍使用 ``[0, 1)`` 的周期相位约定。
    """

    points = np.asarray(curve, dtype=float)
    count = int(sample_count)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        raise ValueError(f"curve must have shape (N,3) with N>=3, got {points.shape}")
    if count < 3:
        raise ValueError("sample_count must be at least 3")
    if np.any(~np.isfinite(points)):
        raise ValueError("curve must contain only finite values")
    if np.linalg.norm(points[-1] - points[0]) <= 1e-12:
        points = points[:-1]
    segment_start = points
    segment_vector = np.roll(points, -1, axis=0) - points
    segment_length = np.linalg.norm(segment_vector, axis=1)
    positive = segment_length > 1e-12
    if np.count_nonzero(positive) < 3:
        raise ValueError("curve does not contain enough nondegenerate segments")
    segment_start = segment_start[positive]
    segment_vector = segment_vector[positive]
    segment_length = segment_length[positive]
    cumulative = np.concatenate([[0.0], np.cumsum(segment_length)])
    total_length = float(cumulative[-1])
    if (not np.isfinite(total_length)) or total_length <= 1e-9:
        raise ValueError("curve has zero or non-finite closed length")
    requested = total_length * np.arange(count, dtype=float) / float(count)
    segment_index = np.searchsorted(cumulative[1:], requested, side="right")
    segment_index = np.minimum(segment_index, segment_length.size - 1)
    local = (requested - cumulative[segment_index]) / segment_length[segment_index]
    sampled = (
        segment_start[segment_index]
        + local[:, None] * segment_vector[segment_index]
    )
    sampled[0] = points[0]
    return sampled


def resample_closed_tip_wrist_equal_arclength(
    tip: np.ndarray,
    wrist: np.ndarray,
    sample_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Use one coupled 6D arc-length phase while preserving Tip/Wrist simultaneity."""

    tip_curve = np.asarray(tip, dtype=float)
    wrist_curve = np.asarray(wrist, dtype=float)
    if (
        tip_curve.ndim != 2 or wrist_curve.ndim != 2
        or tip_curve.shape != wrist_curve.shape
        or tip_curve.shape[1] != 3 or tip_curve.shape[0] < 3
    ):
        raise ValueError("tip and wrist must have the same shape (N,3), with N>=3")
    if np.any(~np.isfinite(tip_curve)) or np.any(~np.isfinite(wrist_curve)):
        raise ValueError("tip and wrist must contain only finite values")
    paired = np.column_stack([tip_curve, wrist_curve])
    if np.linalg.norm(paired[-1] - paired[0]) <= 1e-12:
        paired = paired[:-1]
    segment_start = paired
    segment_vector = np.roll(paired, -1, axis=0) - paired
    segment_length = np.linalg.norm(segment_vector, axis=1)
    positive = segment_length > 1e-12
    if np.count_nonzero(positive) < 3:
        raise ValueError("paired Tip/Wrist curve has too few nondegenerate segments")
    segment_start = segment_start[positive]
    segment_vector = segment_vector[positive]
    segment_length = segment_length[positive]
    cumulative = np.concatenate([[0.0], np.cumsum(segment_length)])
    total_length = float(cumulative[-1])
    requested = total_length * np.arange(sample_count, dtype=float) / float(sample_count)
    segment_index = np.searchsorted(cumulative[1:], requested, side="right")
    segment_index = np.minimum(segment_index, segment_length.size - 1)
    local = (requested - cumulative[segment_index]) / segment_length[segment_index]
    sampled = (
        segment_start[segment_index]
        + local[:, None] * segment_vector[segment_index]
    )
    sampled[0] = paired[0]
    return sampled[:, :3], sampled[:, 3:]


def strict_initialized_paired_arclength_residuals(
    tip: np.ndarray,
    wrist: np.ndarray,
    target_tip: np.ndarray,
    target_wrist: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return synchronized same-index residuals under one coupled 6D arc phase."""

    target_tip_array = np.asarray(target_tip, dtype=float)
    target_wrist_array = np.asarray(target_wrist, dtype=float)
    if target_tip_array.shape != target_wrist_array.shape:
        raise ValueError("initialized Tip/Wrist targets must have identical shapes")
    sampled_tip, sampled_wrist = resample_closed_tip_wrist_equal_arclength(
        tip, wrist, target_tip_array.shape[0]
    )
    return (
        sampled_tip - target_tip_array,
        sampled_wrist - target_wrist_array,
        sampled_tip,
        sampled_wrist,
    )


def strict_initialized_arclength_residuals(
    generated: np.ndarray,
    initialized_equal_arc_target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按固定相位零点和同一等弧长索引计算三维残差。

    目标数组必须已经从 Excel 指定初始事件开始按等弧长划分。生成曲线保留
    机构第 0 帧作为初始点，再按相同点数等弧长划分。第 ``k`` 个生成点只能
    与第 ``k`` 个目标点比较；不允许最近点投影、循环移位、反向匹配或动态锚定。
    """

    target = np.asarray(initialized_equal_arc_target, dtype=float)
    if target.ndim != 2 or target.shape[1] != 3 or target.shape[0] < 3:
        raise ValueError(f"target must have shape (M,3) with M>=3, got {target.shape}")
    if np.any(~np.isfinite(target)):
        raise ValueError("target must contain only finite values")
    sampled_generated = resample_closed_curve_equal_arclength(
        generated, target.shape[0]
    )
    residual = sampled_generated - target
    distance = np.linalg.norm(residual, axis=1)
    return residual, distance, sampled_generated


def nearest_periodic_polyline_residuals(
    generated: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Legacy diagnostic: closest point on a periodic target polyline.

    This function remains only for reading or auditing older phase-independent runs.
    The initialized equal-arc optimizer never calls it when computing its objective.
    """

    points = np.asarray(generated, dtype=float)
    curve = np.asarray(target, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"generated must have shape (N,3), got {points.shape}")
    if curve.ndim != 2 or curve.shape[1] != 3 or curve.shape[0] < 2:
        raise ValueError(f"target must have shape (M,3) with M>=2, got {curve.shape}")
    if np.any(~np.isfinite(points)) or np.any(~np.isfinite(curve)):
        raise ValueError("generated and target curves must contain only finite values")

    segment_start = curve
    segment_vector = np.roll(curve, -1, axis=0) - curve
    segment_length_squared = np.einsum(
        "ij,ij->i", segment_vector, segment_vector
    )
    offset = points[:, None, :] - segment_start[None, :, :]
    projection = np.zeros((points.shape[0], curve.shape[0]), dtype=float)
    nondegenerate = segment_length_squared > 1e-18
    projection[:, nondegenerate] = (
        np.einsum(
            "nmi,mi->nm", offset[:, nondegenerate, :],
            segment_vector[nondegenerate, :]
        )
        / segment_length_squared[nondegenerate][None, :]
    )
    projection = np.clip(projection, 0.0, 1.0)
    projected = (
        segment_start[None, :, :]
        + projection[:, :, None] * segment_vector[None, :, :]
    )
    all_residuals = points[:, None, :] - projected
    squared_distance = np.einsum(
        "nmi,nmi->nm", all_residuals, all_residuals
    )
    segment_index = np.argmin(squared_distance, axis=1)
    row_index = np.arange(points.shape[0])
    residual = all_residuals[row_index, segment_index, :]
    distance = np.sqrt(squared_distance[row_index, segment_index])
    segment_fraction = projection[row_index, segment_index]
    return residual, distance, segment_index, segment_fraction


def pointwise_rmse(
    generated: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """Return strict initialized equal-arc index RMSE and peak."""

    _, distance, _ = strict_initialized_arclength_residuals(generated, target)
    return (
        float(np.sqrt(np.mean(distance ** 2))),
        float(np.max(distance)),
        distance,
    )


def combined_rmse(tip: np.ndarray, wrist: np.ndarray, target_tip: np.ndarray, target_wrist: np.ndarray) -> Dict[str, float]:
    """Combine synchronized strict initialized equal-arc Tip/Wrist distances."""

    tip_residual, wrist_residual, _, _ = strict_initialized_paired_arclength_residuals(
        tip, wrist, target_tip, target_wrist
    )
    tip_err = np.linalg.norm(tip_residual, axis=1)
    wrist_err = np.linalg.norm(wrist_residual, axis=1)
    tip_rmse = float(np.sqrt(np.mean(tip_err ** 2)))
    wrist_rmse = float(np.sqrt(np.mean(wrist_err ** 2)))
    tip_peak = float(np.max(tip_err))
    wrist_peak = float(np.max(wrist_err))
    combined = float(np.sqrt(np.mean(np.concatenate([tip_err ** 2, wrist_err ** 2]))))
    return {
        "combined": combined,
        "tip": tip_rmse,
        "wrist": wrist_rmse,
        "tip_peak": tip_peak,
        "wrist_peak": wrist_peak,
        "strict_correspondence": 1.0,
    }


# =============================================================================
# 完整问题定义
# =============================================================================
# 本节把原先散落在 fourbar_problem*.py 和 optimize_* 适配脚本中的模型
# 定义集中到当前文件。上面的机构几何求解没有改写；下面只负责输入、参数解码、
# 相位、目标曲线共同位姿和结果评价。

MODEL_DIR = Path(__file__).resolve().parent
# 模型文件既可能位于“当前文件/model”，也可能作为运行快照位于
# “当前文件/output/<run>”。逐级向上寻找最近的有效 input 目录，可保证
# 主代码、运行快照和独立复现都读取同一组目标曲线，而不依赖文件层级深度。
def _find_current_dir(model_dir: Path) -> Path:
    for candidate in (model_dir, *model_dir.parents):
        input_dir = candidate / "input"
        if (
            (input_dir / "rmse25_matlab_data.mat").is_file()
            and (input_dir / "Tip_fourier_function.txt").is_file()
            and (input_dir / "Wrist_fourier_function.txt").is_file()
        ):
            return candidate
    # 保留明确的回退位置，使缺少输入时的 FileNotFoundError 指向预期目录。
    return model_dir.parent


CURRENT_DIR = _find_current_dir(MODEL_DIR)
INPUT_DIR = CURRENT_DIR / "input"
DEFAULT_MATLAB_DATA = INPUT_DIR / "rmse25_matlab_data.mat"
# 正式目标来自 2026-07-24 新 Excel，并已按 76 个模型相位重采样为毫米坐标。
DEFAULT_TIP_TARGET = INPUT_DIR / "New_Tip_fourier_function_mm.txt"
DEFAULT_WRIST_TARGET = INPUT_DIR / "New_Wrist_fourier_function_mm.txt"
DEFAULT_INITIALIZED_TARGET_WORKBOOK = (
    INPUT_DIR / "Length_normalized_target_documentation_with_initialization.xlsx"
)
DEFAULT_INITIALIZED_TARGET_CSV = (
    INPUT_DIR / "Length_normalized_target_initialized_equal_arc_76_mm.csv"
)
DEFAULT_INITIALIZED_TARGET_METADATA = (
    INPUT_DIR / "Length_normalized_target_initialized_equal_arc_76_metadata.json"
)

PREVIOUS_STATIC_NAMES = (
    "L1", "L2", "L31", "L4", "L41", "L5", "L51", "L52", "L7", "L8", "L9",
    "L10", "L11", "L12", "LRod", "L14", "L15", "L17",
    "H_finger", "L3", "L61", "Lf1", "Lf2", "L_CZ", "L_down",
)
PRE_L32_STATIC_NAMES = (
    "L1", "L2", "L31", "L4", "L41", "L5", "L51", "L52", "L6", "L7", "L8", "L9",
    "L10", "L11", "L12", "L13", "LRod", "L14", "L15", "L17",
    "H_finger", "L3", "L61", "Lf1", "Lf2", "L_CZ", "L_down", "theta18_deg",
)
STATIC_NAMES = (
    "L1", "L2", "L31", "L32", "L4", "L41", "L5", "L51", "L52", "L6", "L7", "L8", "L9",
    "L10", "L11", "L12", "L13", "LRod", "L14", "L15", "L17",
    "H_finger", "L3", "L61", "Lf1", "Lf2", "L_CZ", "L_down", "theta18_deg",
)
PRE_L13_STATIC_NAMES = tuple(name for name in PRE_L32_STATIC_NAMES if name != "L13")
PRE_THETA18_STATIC_NAMES = tuple(
    name for name in PRE_L32_STATIC_NAMES if name not in {"L13", "theta18_deg"}
)
LEGACY_STATIC_NAMES = tuple(name for name in PREVIOUS_STATIC_NAMES if name != "L1")
# L1 只保留为旧 MATLAB base32 的兼容槽位；新模型 AB(t)=|B(t)|，因此 L1
# 不进入任何新优化空间。周期杆的常数项继续代替其静态标量。
L7_PERIODIC_FIXED_STATIC_NAMES = ("L1", "L7", "H_finger", "Lf1", "L_CZ")
L6_PERIODIC_FIXED_STATIC_NAMES = ("L1", "L6", "L_CZ")
L7_PERIODIC_ACTIVE_STATIC_NAMES = tuple(
    name for name in STATIC_NAMES if name not in L7_PERIODIC_FIXED_STATIC_NAMES
)
L6_PERIODIC_ACTIVE_STATIC_NAMES = tuple(
    name for name in STATIC_NAMES if name not in L6_PERIODIC_FIXED_STATIC_NAMES
)
L7_L8_PERIODIC_FIXED_STATIC_NAMES = ("L1", "L7", "L8", "L_CZ")
L7_L8_PERIODIC_ACTIVE_STATIC_NAMES = tuple(
    name for name in STATIC_NAMES if name not in L7_L8_PERIODIC_FIXED_STATIC_NAMES
)
# 共同位姿不足时的机构自由度替换方案：L7 恢复为固定静态杆，L31(t) 与
# L8(t) 使用三阶 Fourier。L31、L8 的静态槽位分别由各自 C0 代替。
L31_L8_PERIODIC_L7_FIXED_STATIC_NAMES = ("L1", "L31", "L8", "L_CZ")
L31_L8_PERIODIC_L7_FIXED_ACTIVE_STATIC_NAMES = tuple(
    name for name in STATIC_NAMES
    if name not in L31_L8_PERIODIC_L7_FIXED_STATIC_NAMES
)
# 当前替换拓扑：保留 L31(t) 三阶周期特征，把原 L8(t) 恢复为固定优化杆，
# 同时将 L6 提升为三阶周期杆。L7、L8 均在一个周期内保持固定。
L31_L6_PERIODIC_L7_L8_FIXED_STATIC_NAMES = ("L1", "L31", "L6", "L_CZ")
L31_L6_PERIODIC_L7_L8_FIXED_ACTIVE_STATIC_NAMES = tuple(
    name for name in STATIC_NAMES
    if name not in L31_L6_PERIODIC_L7_L8_FIXED_STATIC_NAMES
)
# 收敛性扩展：在上一模式上把 BC=L2 也提升为三阶周期长度。
# L2、L31 和 L6 的静态槽位均由对应 Fourier 常数项代替。
L2_L31_L6_PERIODIC_L7_L8_FIXED_STATIC_NAMES = (
    "L1", "L2", "L31", "L6", "L_CZ",
)
L2_L31_L6_PERIODIC_L7_L8_FIXED_ACTIVE_STATIC_NAMES = tuple(
    name for name in STATIC_NAMES
    if name not in L2_L31_L6_PERIODIC_L7_L8_FIXED_STATIC_NAMES
)
# 当前正式拓扑：BC=L2、AG=L3、L31、L6 和 L7 均为固定优化杆；
# L5(t)、L8(t) 使用三阶 Fourier；L_CZ 由单连通二阶分离曲线代替。
L5_L8_ZC_SPLIT_PERIODIC_FIXED_STATIC_NAMES = (
    "L1", "L5", "L8", "L_CZ",
)
L5_L8_ZC_SPLIT_PERIODIC_ACTIVE_STATIC_NAMES = tuple(
    name for name in STATIC_NAMES
    if name not in L5_L8_ZC_SPLIT_PERIODIC_FIXED_STATIC_NAMES
)
# 当前正式拓扑：L6、L31 为周期内固定优化变量，L32(t)=L3(t)-L31+2 为派生杆；
# 仅 L3(t)、L8(t) 使用三阶 Fourier；L5 为固定优化杆，C 与 Z 始终重合。
L3_L5_L8_PERIODIC3_L32_FIXED_STATIC_NAMES = (
    "L1", "L3", "L5", "L8", "L32", "L_CZ",
)
L3_L5_L8_PERIODIC3_L32_FIXED_ACTIVE_STATIC_NAMES = tuple(
    name for name in STATIC_NAMES
    if name not in L3_L5_L8_PERIODIC3_L32_FIXED_STATIC_NAMES
)
# 当前正式拓扑：L3(t) 与 L8(t) 为三阶 Fourier 周期杆，L5 为固定优化杆；
# L32(t)=L3(t)-L31+2 逐帧派生，因此 L3、L8、L32 不重复占用静态变量槽位。
L3_L8_PERIODIC3_L5_FIXED_STATIC_NAMES = (
    "L1", "L3", "L8", "L32", "H_finger", "Lf1", "L_CZ",
)
L3_L8_PERIODIC3_L5_FIXED_ACTIVE_STATIC_NAMES = tuple(
    name for name in STATIC_NAMES
    if name not in L3_L8_PERIODIC3_L5_FIXED_STATIC_NAMES
)
# 结构自由度验证模式：完整保留当前正式拓扑，只把 Wrist=L 上游的固定
# L6 替换为三阶 Fourier 周期长度。零谐波时该模式严格退化为正式模式，
# 因而可以把结果差异归因于新增的 L6(t) 自由度，而不是机构方程变化。
L5_L6_L8_ZC_SPLIT_PERIODIC_FIXED_STATIC_NAMES = (
    "L1", "L5", "L6", "L8", "L_CZ",
)
L5_L6_L8_ZC_SPLIT_PERIODIC_ACTIVE_STATIC_NAMES = tuple(
    name for name in STATIC_NAMES
    if name not in L5_L6_L8_ZC_SPLIT_PERIODIC_FIXED_STATIC_NAMES
)
# 本轮模式：L3(t) 和 L7(t) 为独立二阶傅里叶周期杆，L8 保持固定但参与优化；
# Z 点继续固定在 (0,-150,0)，因此 L_CZ 不进入设计向量。
L3_L7_PERIODIC_FIXED_STATIC_NAMES = ("L1", "L3", "L7", "L_CZ")
L3_L7_PERIODIC_ACTIVE_STATIC_NAMES = tuple(
    name for name in STATIC_NAMES if name not in L3_L7_PERIODIC_FIXED_STATIC_NAMES
)
# 本轮模式：L3 和 L8 是固定杆长自变量，仅 L7(t) 使用二阶傅里叶周期变化。
# L_CZ 固定为 150 mm，保持与上一轮机构空间位置一致。
L7_PERIODIC_L3_FIXED_FIXED_STATIC_NAMES = ("L1", "L7", "L_CZ")
L7_PERIODIC_L3_FIXED_ACTIVE_STATIC_NAMES = tuple(
    name for name in STATIC_NAMES
    if name not in L7_PERIODIC_L3_FIXED_FIXED_STATIC_NAMES
)
# 新拓扑：L3 为固定优化杆，L7(t) 与 C-Z 延展长度 ZC(t) 周期变化。
# L_CZ 的静态槽位由 ZC 傅里叶常数项代替，不重复进入设计向量。
ZC_L7_PERIODIC_FIXED_STATIC_NAMES = ("L1", "L7", "L_CZ")
ZC_L7_PERIODIC_ACTIVE_STATIC_NAMES = tuple(
    name for name in STATIC_NAMES
    if name not in ZC_L7_PERIODIC_FIXED_STATIC_NAMES
)
# 扩展空间：在 L7(t)+ZC(t) 拓扑上，把静态 L3 替换为独立二阶
# Fourier 周期杆。其余静态边界和可装配性检查保持不变。
ZC_L3_L7_PERIODIC_FIXED_STATIC_NAMES = ("L1", "L3", "L7", "L_CZ")
ZC_L3_L7_PERIODIC_ACTIVE_STATIC_NAMES = tuple(
    name for name in STATIC_NAMES
    if name not in ZC_L3_L7_PERIODIC_FIXED_STATIC_NAMES
)
# Evidence-driven expansion probe: retain the same solver and user-table bounds,
# but promote the currently static L6 to the already-supported second-order
# Fourier representation.  The constant-L6 subspace is embedded exactly by
# L6_C0=L6 and all four harmonic terms equal to zero.
ZC_L3_L6_L7_PERIODIC_FIXED_STATIC_NAMES = (
    "L1", "L3", "L6", "L7", "L_CZ",
)
ZC_L3_L6_L7_PERIODIC_ACTIVE_STATIC_NAMES = tuple(
    name for name in STATIC_NAMES
    if name not in ZC_L3_L6_L7_PERIODIC_FIXED_STATIC_NAMES
)
# Shape-expansion probe: replace the currently static L8 variable by the same
# constrained second-order Fourier representation used for the other periodic
# rods.  L8_C0=L8 and zero harmonics embed the selected space exactly.
ZC_L3_L7_L8_PERIODIC_FIXED_STATIC_NAMES = (
    "L1", "L3", "L7", "L8", "L_CZ",
)
ZC_L3_L7_L8_PERIODIC_ACTIVE_STATIC_NAMES = tuple(
    name for name in STATIC_NAMES
    if name not in ZC_L3_L7_L8_PERIODIC_FIXED_STATIC_NAMES
)
# 历史替换拓扑：L3 恢复为固定优化杆，L8 取代 L3 成为周期杆；
# L7(t)、L8(t) 和 ZC(t) 均使用四阶 Fourier。仅保留用于读取旧结果。
ZC_L7_L8_PERIODIC_L3_FIXED_STATIC_NAMES = ("L1", "L7", "L8", "L_CZ")
ZC_L7_L8_PERIODIC_L3_FIXED_ACTIVE_STATIC_NAMES = tuple(
    name for name in STATIC_NAMES
    if name not in ZC_L7_L8_PERIODIC_L3_FIXED_STATIC_NAMES
)
# 保留旧名称作为 L7 周期模式的兼容别名，历史检查点和区域分析不受影响。
FIXED_STATIC_NAMES = L7_PERIODIC_FIXED_STATIC_NAMES
ACTIVE_STATIC_NAMES = L7_PERIODIC_ACTIVE_STATIC_NAMES

B_FOURIER_NAMES = (
    "B_Y_C0", "B_Y_C1c", "B_Y_C1s", "B_Y_C2c", "B_Y_C2s",
    "B_Z_C1c", "B_Z_C1s", "B_Z_C2c", "B_Z_C2s", "B_CenterX",
)
B_FOURIER_X_NAMES = B_FOURIER_NAMES + (
    "B_X_C1c", "B_X_C1s", "B_X_C2c", "B_X_C2s",
)
B_FOURIER_Z3_NAMES = B_FOURIER_NAMES + ("B_Z_C3c", "B_Z_C3s")
# Preserve the 59.9 baseline Bx/By/Bz harmonics and add only the Bz center.
B_FOURIER_Z3_C0_NAMES = B_FOURIER_NAMES + (
    "B_Z_C0", "B_Z_C3c", "B_Z_C3s",
)
# 当前正式 B 曲线：By 二阶、Bz 三阶且中心 Z 可调、Bx 二阶。
# 前 10 个槽位保持不变，便于读取旧输入；新增槽位依次为 Bz 中心、
# Bz 三阶项和 Bx 的一/二阶项。
B_FOURIER_XYZ3_NAMES = B_FOURIER_NAMES + (
    "B_Z_C0", "B_Z_C3c", "B_Z_C3s",
    "B_X_C1c", "B_X_C1s", "B_X_C2c", "B_X_C2s",
)
B_X_MAX_EXCURSION_MM = 30.0
# 旧常量名保留为导入兼容层；当前内容是直接二维二阶 Fourier 参数。
MOT_POLAR_NAMES = B_FOURIER_NAMES
B_POLAR_NAMES = B_FOURIER_NAMES
# L6/L7 都由二阶傅里叶的常数项、一次谐波和二次谐波定义。
L6_FOURIER_NAMES = ("L6_C0", "L6_C1c", "L6_C1s", "L6_C2c", "L6_C2s")
L2_FOURIER_NAMES = ("L2_C0", "L2_C1c", "L2_C1s", "L2_C2c", "L2_C2s")
L31_FOURIER_NAMES = ("L31_C0", "L31_C1c", "L31_C1s", "L31_C2c", "L31_C2s")
L3_FOURIER_NAMES = ("L3_C0", "L3_C1c", "L3_C1s", "L3_C2c", "L3_C2s")
L5_FOURIER_NAMES = ("L5_C0", "L5_C1c", "L5_C1s", "L5_C2c", "L5_C2s")
L7_FOURIER_NAMES = ("L7_C0", "L7_C1c", "L7_C1s", "L7_C2c", "L7_C2s")
L8_FOURIER_NAMES = ("L8_C0", "L8_C1c", "L8_C1s", "L8_C2c", "L8_C2s")
ZC_FOURIER_NAMES = ("ZC_C0", "ZC_C1c", "ZC_C1s", "ZC_C2c", "ZC_C2s")
L3_FOURIER3_NAMES = L3_FOURIER_NAMES + ("L3_C3c", "L3_C3s")
L2_FOURIER3_NAMES = L2_FOURIER_NAMES + ("L2_C3c", "L2_C3s")
L31_FOURIER3_NAMES = L31_FOURIER_NAMES + ("L31_C3c", "L31_C3s")
L5_FOURIER3_NAMES = L5_FOURIER_NAMES + ("L5_C3c", "L5_C3s")
L6_FOURIER3_NAMES = L6_FOURIER_NAMES + ("L6_C3c", "L6_C3s")
L7_FOURIER3_NAMES = L7_FOURIER_NAMES + ("L7_C3c", "L7_C3s")
L8_FOURIER3_NAMES = L8_FOURIER_NAMES + ("L8_C3c", "L8_C3s")
ZC_FOURIER3_NAMES = ZC_FOURIER_NAMES + ("ZC_C3c", "ZC_C3s")
L3_FOURIER4_NAMES = L3_FOURIER3_NAMES + ("L3_C4c", "L3_C4s")
L5_FOURIER4_NAMES = L5_FOURIER3_NAMES + ("L5_C4c", "L5_C4s")
L7_FOURIER4_NAMES = L7_FOURIER3_NAMES + ("L7_C4c", "L7_C4s")
L8_FOURIER4_NAMES = L8_FOURIER3_NAMES + ("L8_C4c", "L8_C4s")
ZC_FOURIER4_NAMES = ZC_FOURIER3_NAMES + ("ZC_C4c", "ZC_C4s")
# 三个独立参数通过 zc_split_fourier_coefficients() 展开为五个标准二阶系数。
ZC_SPLIT_NAMES = ("ZC_Amplitude_mm", "ZC_ShapeCos", "ZC_ShapeSin")


CURRENT_L32_DERIVED_MODE = "l3_l8_periodic3_l5_fixed_l32_derived"
LEGACY_L32_DERIVED3_MODE = "l3_l5_l8_periodic3_l32_derived"
LEGACY_L32_DERIVED4_MODE = "l3_l5_l8_periodic4_l32_derived"
LEGACY_L32_FIXED_MODE_NAME = "l3_l5_l8_periodic3_l32_fixed"
L32_DERIVED_PERIODIC_MODES = {
    CURRENT_L32_DERIVED_MODE,
    LEGACY_L32_DERIVED3_MODE,
    LEGACY_L32_DERIVED4_MODE,
    LEGACY_L32_FIXED_MODE_NAME,
}


def periodic_coefficient_count(periodic_length_mode: str) -> int:
    """返回每根周期杆的 Fourier 系数个数，避免解码器散落阶次判断。"""

    if periodic_length_mode in {
        "zc_l3_l7_periodic4",
        "zc_l7_l8_periodic4_l3_fixed",
        LEGACY_L32_DERIVED4_MODE,
    }:
        return 9
    if periodic_length_mode in {
        "l7_l8_periodic3",
        "l31_l8_periodic3_l7_fixed",
        "l31_l6_periodic3_l7_l8_fixed",
        "l2_l31_l6_periodic3_l7_l8_fixed",
        "l5_l8_zc_split_periodic3",
        "l5_l6_l8_zc_split_periodic3",
        CURRENT_L32_DERIVED_MODE,
        LEGACY_L32_DERIVED3_MODE,
        LEGACY_L32_FIXED_MODE_NAME,
        "l3_l5_l8_zc2_periodic_l32_fixed",
        "zc_l3_l7_periodic3",
    }:
        return 7
    return 5
TARGET_POSE_NAMES = (
    "Target_Tx_mm", "Target_Ty_mm", "Target_Tz_mm",
    "Target_Rx_rad", "Target_Ry_rad", "Target_Rz_rad", "Target_Scale",
)
TARGET_POSE_LB = np.array([-150.0, -200.0, -150.0, -1.0, -1.0, -1.0, 0.8], dtype=float)
TARGET_POSE_UB = np.array([150.0, 200.0, 150.0, 1.0, 1.0, 1.0, 1.2], dtype=float)
TARGET_POSE_INITIAL = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
# 六自由度刚体位姿：放开三轴平移和三轴旋转，但不允许目标尺度变化。
RIGID6_TARGET_POSE_NAMES = TARGET_POSE_NAMES[:6]
RIGID6_TARGET_POSE_LB = TARGET_POSE_LB[:6].copy()
RIGID6_TARGET_POSE_UB = TARGET_POSE_UB[:6].copy()
RIGID6_TARGET_POSE_INITIAL = TARGET_POSE_INITIAL[:6].copy()
FULL_VARIABLE_NAMES = (
    ACTIVE_STATIC_NAMES + MOT_POLAR_NAMES + L7_FOURIER_NAMES + TARGET_POSE_NAMES
)
RESTRICTED_TARGET_POSE_NAMES = ("Target_Ty_mm", "Target_Ry_rad")
RESTRICTED_TARGET_POSE_LB = np.array([-50.0, -0.5], dtype=float)
RESTRICTED_TARGET_POSE_UB = np.array([50.0, 0.5], dtype=float)
RESTRICTED_TARGET_POSE_INITIAL = np.array([0.0, 0.0], dtype=float)
# 本轮受限目标位姿：公共 Y 平移限制在 [-200,-100] mm，绕 Y 轴扩大到
# +/-0.8 rad，绕 Z 轴继续为 +/-0.5 rad。
# Tx/Tz/Rx 固定为 0，目标尺度固定为 1，避免重新引入无关位姿自由度。
TY_RY_RZ_TARGET_POSE_NAMES = (
    "Target_Ty_mm", "Target_Ry_rad", "Target_Rz_rad"
)
TY_RY_RZ_TARGET_POSE_LB = np.array([-200.0, -0.8, -0.5], dtype=float)
TY_RY_RZ_TARGET_POSE_UB = np.array([-100.0, 0.8, 0.5], dtype=float)
TY_RY_RZ_TARGET_POSE_INITIAL = np.array([-100.0, 0.0, 0.0], dtype=float)
# 当前受限公共目标位姿：增加 Tx/Tz 的小范围平移和目标尺度，保持 Rx=0。
CONSTRAINED6_TARGET_POSE_NAMES = (
    "Target_Tx_mm", "Target_Ty_mm", "Target_Tz_mm",
    "Target_Ry_rad", "Target_Rz_rad", "Target_Scale",
)
CONSTRAINED6_TARGET_POSE_LB = np.array(
    [-50.0, -200.0, -50.0, -0.8, -0.5, 0.8], dtype=float
)
CONSTRAINED6_TARGET_POSE_UB = np.array(
    [50.0, -100.0, 50.0, 0.8, 0.5, 1.2], dtype=float
)
CONSTRAINED6_TARGET_POSE_INITIAL = np.array(
    [0.0, -100.0, 0.0, 0.0, 0.0, 1.0], dtype=float
)
# Tip 与 Wrist 分别拥有独立的 Tx/Ty/Tz/Ry/Rz，边界沿用 constrained6；
# Rx 对两条曲线都固定为 0，尺度仍由一个公共变量控制。
DECOUPLED_CONSTRAINED_TARGET_POSE_NAMES = (
    "Target_Tip_Tx_mm", "Target_Tip_Ty_mm", "Target_Tip_Tz_mm",
    "Target_Tip_Ry_rad", "Target_Tip_Rz_rad",
    "Target_Wrist_Tx_mm", "Target_Wrist_Ty_mm", "Target_Wrist_Tz_mm",
    "Target_Wrist_Ry_rad", "Target_Wrist_Rz_rad",
    "Target_Scale",
)
DECOUPLED_CONSTRAINED_TARGET_POSE_LB = np.array(
    [-50.0, -200.0, -50.0, -0.8, -0.5,
     -50.0, -200.0, -50.0, -0.8, -0.5, 0.8],
    dtype=float,
)
DECOUPLED_CONSTRAINED_TARGET_POSE_UB = np.array(
    [50.0, -100.0, 50.0, 0.8, 0.5,
     50.0, -100.0, 50.0, 0.8, 0.5, 1.2],
    dtype=float,
)
DECOUPLED_CONSTRAINED_TARGET_POSE_INITIAL = np.array(
    [0.0, -100.0, 0.0, 0.0, 0.0,
     0.0, -100.0, 0.0, 0.0, 0.0, 1.0],
    dtype=float,
)

# 当前目标约束仅保留两条目标曲线各自的 Tx/Ty/Tz/Ry。
# Rx、Rz 均固定为 0，Tip/Wrist 尺度均固定为 1，不进入优化向量。
DECOUPLED_FIXED_RZ_SCALE_TARGET_POSE_NAMES = (
    "Target_Tip_Tx_mm", "Target_Tip_Ty_mm", "Target_Tip_Tz_mm",
    "Target_Tip_Ry_rad",
    "Target_Wrist_Tx_mm", "Target_Wrist_Ty_mm", "Target_Wrist_Tz_mm",
    "Target_Wrist_Ry_rad",
)
DECOUPLED_FIXED_RZ_SCALE_TARGET_POSE_LB = np.array(
    [-200.0, -200.0, -200.0, -0.5,
     -200.0, -100.0, -100.0, -0.5],
    dtype=float,
)
DECOUPLED_FIXED_RZ_SCALE_TARGET_POSE_UB = np.array(
    [0.0, -100.0, 0.0, 0.8,
     0.0, 100.0, 0.0, 0.8],
    dtype=float,
)
DECOUPLED_FIXED_RZ_SCALE_TARGET_POSE_INITIAL = np.array(
    [-50.0, -150.0, -100.0, 0.0,
     -50.0, 0.0, -50.0, 0.0],
    dtype=float,
)

# 当前快速搜索目标位姿共 8 个变量：Tip 独立优化 Tx/Ty/Tz/Ry，
# Wrist 仅优化 Tx/Ty/Tz，公共尺度参与优化。两条曲线的 Rx/Rz 以及
# Wrist_Ry 均固定为 0；保留上一组常量以便旧检查点继续复算。
DECOUPLED_TIP_RY_SCALE_TARGET_POSE_NAMES = (
    "Target_Tip_Tx_mm", "Target_Tip_Ty_mm", "Target_Tip_Tz_mm",
    "Target_Tip_Ry_rad",
    "Target_Wrist_Tx_mm", "Target_Wrist_Ty_mm", "Target_Wrist_Tz_mm",
    "Target_Scale",
)
DECOUPLED_TIP_RY_SCALE_TARGET_POSE_LB = np.array(
    [-200.0, -200.0, -200.0, -0.5,
     -200.0, -100.0, -100.0, 0.8],
    dtype=float,
)
DECOUPLED_TIP_RY_SCALE_TARGET_POSE_UB = np.array(
    [200.0, 200.0, 0.0, 0.8,
     0.0, 100.0, 0.0, 1.2],
    dtype=float,
)
DECOUPLED_TIP_RY_SCALE_TARGET_POSE_INITIAL = np.array(
    [-50.0, -150.0, -100.0, 0.0,
     -50.0, 0.0, -50.0, 1.0],
    dtype=float,
)

# 保留上一轮的全部平移和公共尺度边界，只为 Wrist 增加与 Tip 对称的 Ry。
# 这是针对 Wrist RMSE 瓶颈的最小位姿扩展，不放宽尺度或位移范围。
DECOUPLED_BOTH_RY_SCALE_TARGET_POSE_NAMES = (
    "Target_Tip_Tx_mm", "Target_Tip_Ty_mm", "Target_Tip_Tz_mm",
    "Target_Tip_Ry_rad",
    "Target_Wrist_Tx_mm", "Target_Wrist_Ty_mm", "Target_Wrist_Tz_mm",
    "Target_Wrist_Ry_rad", "Target_Scale",
)
DECOUPLED_BOTH_RY_SCALE_TARGET_POSE_LB = np.array(
    [-200.0, -200.0, -200.0, -0.5,
     -200.0, -100.0, -100.0, -0.5, 0.8],
    dtype=float,
)
DECOUPLED_BOTH_RY_SCALE_TARGET_POSE_UB = np.array(
    [200.0, 200.0, 0.0, 0.8,
     0.0, 100.0, 0.0, 1.0, 1.2],
    dtype=float,
)
DECOUPLED_BOTH_RY_SCALE_TARGET_POSE_INITIAL = np.array(
    [-50.0, -150.0, -100.0, 0.0,
     -50.0, 0.0, -50.0, 0.0, 1.0],
    dtype=float,
)

# Alignment-only expansion probe: preserve every current translation, Ry and
# common-scale bound, while promoting the two previously fixed Rz components
# to the model's existing constrained +/-0.5 rad range.  Rz=0 embeds the
# selected 58-D design space exactly.
DECOUPLED_BOTH_RY_RZ_SCALE_TARGET_POSE_NAMES = (
    "Target_Tip_Tx_mm", "Target_Tip_Ty_mm", "Target_Tip_Tz_mm",
    "Target_Tip_Ry_rad", "Target_Tip_Rz_rad",
    "Target_Wrist_Tx_mm", "Target_Wrist_Ty_mm", "Target_Wrist_Tz_mm",
    "Target_Wrist_Ry_rad", "Target_Wrist_Rz_rad", "Target_Scale",
)
DECOUPLED_BOTH_RY_RZ_SCALE_TARGET_POSE_LB = np.array(
    [-200.0, -200.0, -200.0, -0.5, -0.5,
     -200.0, -100.0, -100.0, -0.5, -0.5, 0.8],
    dtype=float,
)
DECOUPLED_BOTH_RY_RZ_SCALE_TARGET_POSE_UB = np.array(
    [200.0, 200.0, 0.0, 0.8, 0.5,
     0.0, 100.0, 0.0, 1.0, 0.5, 1.2],
    dtype=float,
)
DECOUPLED_BOTH_RY_RZ_SCALE_TARGET_POSE_INITIAL = np.array(
    [-50.0, -150.0, -100.0, 0.0, 0.0,
     -50.0, 0.0, -50.0, 0.0, 0.0, 1.0],
    dtype=float,
)

# Evidence-driven full-orientation expansion: retain every bound from the
# selected decoupled Ry/Rz space and add only the two previously fixed Rx
# components.  The +/-1 rad interval is the model's existing full-pose Rx
# constraint; setting both new variables to zero exactly embeds the 60-D space.
DECOUPLED_FULL_ROTATION_SCALE_TARGET_POSE_NAMES = (
    "Target_Tip_Tx_mm", "Target_Tip_Ty_mm", "Target_Tip_Tz_mm",
    "Target_Tip_Rx_rad", "Target_Tip_Ry_rad", "Target_Tip_Rz_rad",
    "Target_Wrist_Tx_mm", "Target_Wrist_Ty_mm", "Target_Wrist_Tz_mm",
    "Target_Wrist_Rx_rad", "Target_Wrist_Ry_rad", "Target_Wrist_Rz_rad",
    "Target_Scale",
)
DECOUPLED_FULL_ROTATION_SCALE_TARGET_POSE_LB = np.array(
    [-400.0, -200.0, -200.0, -1.0, -0.5, -0.5,
     -200.0, -100.0, -100.0, -1.0, -0.5, -0.5, 0.8],
    dtype=float,
)
DECOUPLED_FULL_ROTATION_SCALE_TARGET_POSE_UB = np.array(
    [0.0, 200.0, 0.0, 1.0, 0.8, 0.5,
     0.0, 100.0, 0.0, 1.0, 1.0, 0.5, 1.2],
    dtype=float,
)
DECOUPLED_FULL_ROTATION_SCALE_TARGET_POSE_INITIAL = np.array(
    [-200.0, -150.0, -100.0, 0.0, 0.0, 0.0,
     -50.0, 0.0, -50.0, 0.0, 0.0, 0.0, 1.0],
    dtype=float,
)

# 固定 Wrist 姿态模式：保留完整模式中的六个独立平移、Tip 三轴旋转和
# 公共尺度；Wrist 三轴旋转不进入优化向量，始终使用下面的实验指定值。
FIXED_WRIST_ROTATION_XYZ_RAD = np.array(
    [-0.269241, -0.099714, -0.705977],
    dtype=float,
)

# 当前正式目标位姿：Tip 与 Wrist 共用图示给定的固定旋转，只允许三轴共同
# 平移进入优化。公共尺度固定为 1，旋转和尺度都不再占用优化维度。
FIXED_COMMON_ROTATION_XYZ_RAD = np.array(
    [-0.27, -0.10, -0.71],
    dtype=float,
)
SHARED_FIXED_ROTATION_TRANSLATION_TARGET_POSE_NAMES = (
    "Target_Tx_mm", "Target_Ty_mm", "Target_Tz_mm",
)
SHARED_FIXED_ROTATION_TRANSLATION_TARGET_POSE_LB = np.array(
    [-400.0, -400.0, -200.0],
    dtype=float,
)
SHARED_FIXED_ROTATION_TRANSLATION_TARGET_POSE_UB = np.array(
    [0.0, 0.0, 50.0],
    dtype=float,
)
SHARED_FIXED_ROTATION_TRANSLATION_TARGET_POSE_INITIAL = np.array(
    [-125.0, -75.0, -75.0],
    dtype=float,
)

# Tip 与 Wrist 的公共刚体位姿模式。两条曲线共享同一个平移、旋转和尺度，
# 不再允许通过独立位姿消除机构本身的相对位置误差。
# 第一种模式固定公共旋转为上一版 Wrist 的实验指定角，仅保留公共平移和尺度。
SHARED_FIXED_ROTATION_SCALE_TARGET_POSE_NAMES = (
    "Target_Tx_mm", "Target_Ty_mm", "Target_Tz_mm", "Target_Scale",
)
SHARED_FIXED_ROTATION_SCALE_TARGET_POSE_LB = np.array(
    [-400.0, -200.0, -200.0, 0.8],
    dtype=float,
)
SHARED_FIXED_ROTATION_SCALE_TARGET_POSE_UB = np.array(
    [0.0, 200.0, 50.0, 1.2],
    dtype=float,
)
SHARED_FIXED_ROTATION_SCALE_TARGET_POSE_INITIAL = np.array(
    [-125.0, -75.0, -75.0, 1.0],
    dtype=float,
)

# 第二种模式仍为唯一的公共刚体变换，但允许旋转角在实验指定角附近小幅调整。
# 每个旋转分量仅开放 +/-0.15 rad，避免重新把机构误差转移到目标位姿自由度。
SHARED_LIMITED_ROTATION_DELTA_RAD = 0.15
SHARED_LIMITED_ROTATION_SCALE_TARGET_POSE_NAMES = TARGET_POSE_NAMES
SHARED_LIMITED_ROTATION_SCALE_TARGET_POSE_LB = np.concatenate([
    SHARED_FIXED_ROTATION_SCALE_TARGET_POSE_LB[:3],
    FIXED_WRIST_ROTATION_XYZ_RAD - SHARED_LIMITED_ROTATION_DELTA_RAD,
    [0.8],
])
SHARED_LIMITED_ROTATION_SCALE_TARGET_POSE_UB = np.concatenate([
    SHARED_FIXED_ROTATION_SCALE_TARGET_POSE_UB[:3],
    FIXED_WRIST_ROTATION_XYZ_RAD + SHARED_LIMITED_ROTATION_DELTA_RAD,
    [1.2],
])
SHARED_LIMITED_ROTATION_SCALE_TARGET_POSE_INITIAL = np.concatenate([
    SHARED_FIXED_ROTATION_SCALE_TARGET_POSE_INITIAL[:3],
    FIXED_WRIST_ROTATION_XYZ_RAD,
    [1.0],
])

# 公共正向 Rz 模式：Tip 与 Wrist 仍严格共用同一组平移、旋转和尺度。
# Rx、Ry 只允许在历史共同角附近各调整 +/-0.15 rad；Rz 改为非负的小角度
# [0, 0.15] rad，防止通过较大的独立空间旋转掩盖机构相对轨迹误差。
SHARED_POSITIVE_RZ_SCALE_TARGET_POSE_NAMES = TARGET_POSE_NAMES
SHARED_POSITIVE_RZ_SCALE_TARGET_POSE_LB = np.concatenate([
    SHARED_FIXED_ROTATION_SCALE_TARGET_POSE_LB[:3],
    [
        FIXED_WRIST_ROTATION_XYZ_RAD[0] - SHARED_LIMITED_ROTATION_DELTA_RAD,
        FIXED_WRIST_ROTATION_XYZ_RAD[1] - SHARED_LIMITED_ROTATION_DELTA_RAD,
        0.0,
    ],
    [0.8],
])
SHARED_POSITIVE_RZ_SCALE_TARGET_POSE_UB = np.concatenate([
    SHARED_FIXED_ROTATION_SCALE_TARGET_POSE_UB[:3],
    [
        FIXED_WRIST_ROTATION_XYZ_RAD[0] + SHARED_LIMITED_ROTATION_DELTA_RAD,
        FIXED_WRIST_ROTATION_XYZ_RAD[1] + SHARED_LIMITED_ROTATION_DELTA_RAD,
        SHARED_LIMITED_ROTATION_DELTA_RAD,
    ],
    [1.2],
])
SHARED_POSITIVE_RZ_SCALE_TARGET_POSE_INITIAL = np.concatenate([
    SHARED_FIXED_ROTATION_SCALE_TARGET_POSE_INITIAL[:3],
    [
        FIXED_WRIST_ROTATION_XYZ_RAD[0],
        FIXED_WRIST_ROTATION_XYZ_RAD[1],
        0.0,
    ],
    [1.0],
])

# 扩展公共位姿模式：Tip 与 Wrist 严格共用同一平移、旋转和尺度。
# 未列出的范围沿用当前公共位姿设置；Ty、Rx、Ry、Rz 按本轮要求扩展。
SHARED_NEGATIVE_RZ_EXPANDED_SCALE_TARGET_POSE_NAMES = TARGET_POSE_NAMES
SHARED_NEGATIVE_RZ_EXPANDED_SCALE_TARGET_POSE_LB = np.array(
    [-400.0, -400.0, -200.0, -0.5, -0.5, -0.5, 0.8],
    dtype=float,
)
SHARED_NEGATIVE_RZ_EXPANDED_SCALE_TARGET_POSE_UB = np.array(
    [0.0, 0.0, 50.0, 0.5, 0.0, 0.0, 1.2],
    dtype=float,
)
SHARED_NEGATIVE_RZ_EXPANDED_SCALE_TARGET_POSE_INITIAL = np.array(
    [-125.0, -75.0, -75.0, -0.269241, -0.099714, 0.0, 1.0],
    dtype=float,
)

# 本轮公共刚体位姿：Tip/Wrist 共用三轴平移与三轴旋转，尺度固定为 1。
# 旋转范围沿用上一轮公共旋转约束；只把 Ty 下界扩展到 -600 mm。
SHARED_ROTATION_TRANSLATION_TARGET_POSE_NAMES = TARGET_POSE_NAMES[:6]
SHARED_ROTATION_TRANSLATION_TARGET_POSE_LB = np.array(
    [-400.0, -600.0, -200.0, -0.5, -0.8, -0.8],
    dtype=float,
)
SHARED_ROTATION_TRANSLATION_TARGET_POSE_UB = np.array(
    [0.0, 0.0, 50.0, 0.5, 0.0, 0.0],
    dtype=float,
)
# 无锚点任务从目标曲线原始位姿出发，不读取任何历史最优解。
SHARED_ROTATION_TRANSLATION_TARGET_POSE_INITIAL = np.zeros(6, dtype=float)

DECOUPLED_FIXED_WRIST_ROTATION_SCALE_TARGET_POSE_NAMES = (
    "Target_Tip_Tx_mm", "Target_Tip_Ty_mm", "Target_Tip_Tz_mm",
    "Target_Tip_Rx_rad", "Target_Tip_Ry_rad", "Target_Tip_Rz_rad",
    "Target_Wrist_Tx_mm", "Target_Wrist_Ty_mm", "Target_Wrist_Tz_mm",
    "Target_Scale",
)
DECOUPLED_FIXED_WRIST_ROTATION_SCALE_TARGET_POSE_LB = np.array(
    [-400.0, -200.0, -200.0, -1.0, -0.5, -0.5,
     -200.0, -100.0, -100.0, 0.8],
    dtype=float,
)
DECOUPLED_FIXED_WRIST_ROTATION_SCALE_TARGET_POSE_UB = np.array(
    [0.0, 200.0, 0.0, 1.0, 0.8, 0.5,
     0.0, 100.0, 0.0, 1.2],
    dtype=float,
)
DECOUPLED_FIXED_WRIST_ROTATION_SCALE_TARGET_POSE_INITIAL = np.array(
    [-200.0, -150.0, -100.0, 0.0, 0.0, 0.0,
     -50.0, 0.0, -50.0, 1.0],
    dtype=float,
)

FOURIER_TERM_RE = re.compile(
    r"(?P<sign>[+-]?)\s*(?P<coef>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"(?:\s*\*\s*(?P<trig>cos|sin)\(2\*pi\*(?P<harm>\d+)\*phi\))?"
)


@dataclass
class ProblemData:
    phase: np.ndarray
    base32: np.ndarray
    b0: np.ndarray
    b_yz_knots0: np.ndarray
    l6_coeff0: np.ndarray
    l6_values0: np.ndarray
    target_tip: np.ndarray
    target_wrist: np.ndarray
    best_y29: np.ndarray
    lb29: np.ndarray
    ub29: np.ndarray
    best_b_curve: np.ndarray
    best_l6_values: np.ndarray
    best_tip: np.ndarray
    best_wrist: np.ndarray
    best_nodes: np.ndarray
    variable_names: Tuple[str, ...]
    best_metrics: Dict[str, float]
    target_source: str = ""
    target_tip_txt_path: str = ""
    target_wrist_txt_path: str = ""
    target_initialized_csv_path: str = ""
    target_initialization_metadata_path: str = ""
    target_initialization: Dict[str, Any] | None = None


@dataclass
class DesignSpace:
    """完整优化所用的物理量向量及上下界。"""

    x0: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    names: Tuple[str, ...]
    seed_name: str
    periodic_length_mode: str = CURRENT_L32_DERIVED_MODE
    # 当前正式 B 点恢复 RMSE29.99 基准：Bx 为常数，By 二阶、Bz 三阶，C0z=0。
    b_curve_mode: str = "fourier_z3"
    target_pose_mode: str = "shared_fixed_rotation_translation3"


@dataclass
class DesignState:
    """已经解码的机构状态，可来自优化向量或历史检查点。"""

    static: np.ndarray
    mot_curve: np.ndarray
    b_curve: np.ndarray
    l6_values: np.ndarray
    l7_values: np.ndarray
    target_tip: np.ndarray
    target_wrist: np.ndarray
    target_pose: np.ndarray
    b_fourier_coeff: np.ndarray
    l6_fourier_coeff: np.ndarray
    l7_fourier_coeff: np.ndarray
    target_tip_pose: np.ndarray | None = None
    target_wrist_pose: np.ndarray | None = None
    variable_names: Tuple[str, ...] = ()
    x: np.ndarray | None = None
    lb: np.ndarray | None = None
    ub: np.ndarray | None = None
    l2_values: np.ndarray | None = None
    l2_fourier_coeff: np.ndarray | None = None
    l5_values: np.ndarray | None = None
    l5_fourier_coeff: np.ndarray | None = None
    l8_values: np.ndarray | None = None
    l8_fourier_coeff: np.ndarray | None = None
    l31_values: np.ndarray | None = None
    l31_fourier_coeff: np.ndarray | None = None
    l32_values: np.ndarray | None = None
    l3_values: np.ndarray | None = None
    l3_fourier_coeff: np.ndarray | None = None
    zc_values: np.ndarray | None = None
    zc_fourier_coeff: np.ndarray | None = None
    zc_split_parameters: np.ndarray | None = None

    @property
    def mot_fourier_coeff(self) -> np.ndarray:
        """旧属性名兼容层；当前字段表示 B 点傅里叶参数。"""
        return self.b_fourier_coeff


def _mat_struct_to_dict(obj: Any) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name in getattr(obj, "_fieldnames", []):
        value = getattr(obj, name)
        if np.isscalar(value):
            out[name] = float(value)
    return out


def _parse_fourier_axis(text: str, axis: str) -> list[tuple[str, float, int]]:
    match = re.search(rf"{axis}\(phi\)\s*=\s*(.+)", text)
    if not match:
        raise ValueError(f"Missing {axis}(phi) equation in Fourier target file.")
    terms: list[tuple[str, float, int]] = []
    for term in FOURIER_TERM_RE.finditer(match.group(1).strip()):
        sign = -1.0 if term.group("sign") == "-" else 1.0
        terms.append((
            term.group("trig") or "const",
            sign * float(term.group("coef")),
            int(term.group("harm") or 0),
        ))
    if not terms:
        raise ValueError(f"No parseable terms for {axis}(phi).")
    return terms


def curve_from_fourier_txt(path: Path | str, phase01: np.ndarray) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    phase = np.asarray(phase01, dtype=float).reshape(-1)
    curve = np.zeros((phase.size, 3), dtype=float)
    for column, axis in enumerate(("x", "y", "z")):
        values = np.zeros_like(phase)
        for trig, coefficient, harmonic in _parse_fourier_axis(text, axis):
            if trig == "const":
                values += coefficient
            elif trig == "cos":
                values += coefficient * np.cos(2.0 * np.pi * harmonic * phase)
            else:
                values += coefficient * np.sin(2.0 * np.pi * harmonic * phase)
        curve[:, column] = values
    return curve


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_initialized_equal_arc_target(
    csv_path: Path | str,
    metadata_path: Path | str,
    expected_count: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load and fail-closed validate the Excel-derived initialized target."""

    target_path = Path(csv_path)
    meta_path = Path(metadata_path)
    if not target_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            "initialized target CSV and metadata must both exist: "
            f"{target_path}, {meta_path}"
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("correspondence_mode") != STRICT_CORRESPONDENCE_MODE:
        raise ValueError("initialized target metadata has the wrong correspondence mode")
    if int(metadata.get("sample_count", -1)) != int(expected_count):
        raise ValueError(
            "initialized target sample count does not match model phase count: "
            f"{metadata.get('sample_count')} != {expected_count}"
        )
    observed_csv_hash = _sha256_path(target_path)
    expected_csv_hash = str(metadata.get("target_csv_sha256", "")).upper()
    if observed_csv_hash != expected_csv_hash:
        raise ValueError(
            "initialized target CSV SHA256 mismatch: "
            f"{observed_csv_hash} != {expected_csv_hash}"
        )
    workbook = Path(str(metadata.get("source_workbook_path", "")))
    if not workbook.is_absolute():
        workbook = INPUT_DIR / workbook.name
    if not workbook.is_file():
        raise FileNotFoundError(f"source target workbook is missing: {workbook}")
    observed_workbook_hash = _sha256_path(workbook)
    expected_workbook_hash = str(metadata.get("source_workbook_sha256", "")).upper()
    if observed_workbook_hash != expected_workbook_hash:
        raise ValueError(
            "source target workbook SHA256 mismatch: "
            f"{observed_workbook_hash} != {expected_workbook_hash}"
        )

    required = (
        "phase_index", "phase_equal_arc",
        "wrist_x_mm", "wrist_y_mm", "wrist_z_mm",
        "tip_x_mm", "tip_y_mm", "tip_z_mm",
    )
    rows: list[dict[str, str]] = []
    with target_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in required):
            raise ValueError(f"initialized target CSV must contain columns {required}")
        rows.extend(reader)
    if len(rows) != int(expected_count):
        raise ValueError(
            f"initialized target CSV contains {len(rows)} rows; expected {expected_count}"
        )
    indices = np.asarray([int(row["phase_index"]) for row in rows], dtype=int)
    phase = np.asarray([float(row["phase_equal_arc"]) for row in rows], dtype=float)
    if not np.array_equal(indices, np.arange(expected_count, dtype=int)):
        raise ValueError("initialized target phase_index must be exactly 0..N-1")
    expected_phase = np.arange(expected_count, dtype=float) / float(expected_count)
    if not np.allclose(phase, expected_phase, rtol=0.0, atol=1e-12):
        raise ValueError("initialized target phase_equal_arc must be exactly i/N")
    wrist = np.asarray([
        [float(row["wrist_x_mm"]), float(row["wrist_y_mm"]), float(row["wrist_z_mm"])]
        for row in rows
    ], dtype=float)
    tip = np.asarray([
        [float(row["tip_x_mm"]), float(row["tip_y_mm"]), float(row["tip_z_mm"])]
        for row in rows
    ], dtype=float)
    if np.any(~np.isfinite(wrist)) or np.any(~np.isfinite(tip)):
        raise ValueError("initialized target contains non-finite coordinates")
    initialization = metadata.get("initialization", {})
    if float(initialization.get("model_phase", np.nan)) != 0.0:
        raise ValueError("initialized target model_phase must be exactly zero")
    if float(initialization.get("wrist_dz_dphase_mm_per_cycle", 0.0)) <= 0.0:
        raise ValueError("initialized target must use the positive Wrist z crossing")
    expected_wrist = np.asarray(initialization.get("wrist_mm", []), dtype=float)
    expected_tip = np.asarray(initialization.get("tip_mm", []), dtype=float)
    if expected_wrist.shape != (3,) or not np.allclose(
        wrist[0], expected_wrist, rtol=0.0, atol=1e-9
    ):
        raise ValueError("initialized target Wrist phase-zero coordinate mismatch")
    if expected_tip.shape != (3,) or not np.allclose(
        tip[0], expected_tip, rtol=0.0, atol=1e-9
    ):
        raise ValueError("initialized target Tip phase-zero coordinate mismatch")
    return tip, wrist, metadata


def load_problem_data(
    path: Path | str = DEFAULT_MATLAB_DATA,
    initial_params: FourBarParams | None = None,
    use_matlab_initial: bool = False,
    target_tip_txt_path: Path | str | None = None,
    target_wrist_txt_path: Path | str | None = None,
    target_initialized_csv_path: Path | str | None = None,
    target_initialization_metadata_path: Path | str | None = None,
) -> ProblemData:
    """加载机构与目标数据；新优化必须显式选择初始化等弧长目标。"""
    mat = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    phase = np.asarray(mat["phase"], dtype=float).reshape(-1)
    if initial_params is not None and use_matlab_initial:
        raise ValueError("initial_params and use_matlab_initial=True are mutually exclusive")
    if use_matlab_initial:
        initial_base32 = np.asarray(mat["base32"], dtype=float).reshape(-1)
        initial_l6_coeff = np.asarray(mat["l6Coeff0"], dtype=float).reshape(-1)
        initial_l6_values = np.asarray(mat["l6Values0"], dtype=float).reshape(-1)
    else:
        fallback = initial_params or FourBarParams()
        initial_base32 = fallback.to_base32()
        initial_l6_coeff = np.array([fallback.L6, 0.0, 0.0, 0.0, 0.0], dtype=float)
        initial_l6_values = np.full(phase.shape, fallback.L6, dtype=float)
    target_tip = np.asarray(mat["targetTip"], dtype=float)
    target_wrist = np.asarray(mat["targetWrist"], dtype=float)
    target_source = "rmse25_matlab_data.mat:targetTip/targetWrist"
    target_initialization: Dict[str, Any] | None = None
    initialized_selected = target_initialized_csv_path is not None
    if initialized_selected != (target_initialization_metadata_path is not None):
        raise ValueError(
            "target_initialized_csv_path and target_initialization_metadata_path "
            "must be supplied together"
        )
    if initialized_selected and (
        target_tip_txt_path is not None or target_wrist_txt_path is not None
    ):
        raise ValueError("initialized CSV target and legacy Fourier TXT targets are mutually exclusive")
    if (target_tip_txt_path is None) != (target_wrist_txt_path is None):
        raise ValueError(
            "target_tip_txt_path and target_wrist_txt_path must be supplied together"
        )
    explicit_targets = target_tip_txt_path is not None
    tip_target_path = (
        Path(target_tip_txt_path) if explicit_targets else DEFAULT_TIP_TARGET
    )
    wrist_target_path = (
        Path(target_wrist_txt_path) if explicit_targets else DEFAULT_WRIST_TARGET
    )
    if explicit_targets and (
        not tip_target_path.is_file() or not wrist_target_path.is_file()
    ):
        raise FileNotFoundError(
            "Both explicit Fourier target files must exist: "
            f"{tip_target_path}, {wrist_target_path}"
        )
    if tip_target_path.exists() and wrist_target_path.exists():
        target_tip = curve_from_fourier_txt(tip_target_path, phase)
        target_wrist = curve_from_fourier_txt(wrist_target_path, phase)
        target_source = (
            f"{tip_target_path.resolve()} + {wrist_target_path.resolve()}"
        )
    initialized_csv_path = Path(target_initialized_csv_path) if initialized_selected else None
    initialized_metadata_path = (
        Path(target_initialization_metadata_path) if initialized_selected else None
    )
    if initialized_selected:
        target_tip, target_wrist, target_initialization = load_initialized_equal_arc_target(
            initialized_csv_path,
            initialized_metadata_path,
            expected_count=phase.size,
        )
        target_source = (
            f"{initialized_csv_path.resolve()} + {initialized_metadata_path.resolve()}"
        )
    return ProblemData(
        phase=phase,
        base32=initial_base32,
        b0=np.asarray(mat["b0"], dtype=float),
        b_yz_knots0=np.asarray(mat["bYzKnots0"], dtype=float),
        l6_coeff0=initial_l6_coeff,
        l6_values0=initial_l6_values,
        target_tip=target_tip,
        target_wrist=target_wrist,
        best_y29=np.asarray(mat["bestY29"], dtype=float).reshape(-1),
        lb29=np.asarray(mat["lb29"], dtype=float).reshape(-1),
        ub29=np.asarray(mat["ub29"], dtype=float).reshape(-1),
        best_b_curve=np.asarray(mat["bestBCurve"], dtype=float),
        best_l6_values=np.asarray(mat["bestL6Values"], dtype=float).reshape(-1),
        best_tip=np.asarray(mat["bestTip"], dtype=float),
        best_wrist=np.asarray(mat["bestWrist"], dtype=float),
        best_nodes=np.asarray(mat["bestNodes"], dtype=float),
        variable_names=tuple(str(v) for v in np.asarray(mat["variableNames"]).reshape(-1)),
        best_metrics=_mat_struct_to_dict(mat["bestMetrics"]),
        target_source=target_source,
        target_tip_txt_path=(
            str(tip_target_path.resolve())
            if (not initialized_selected and tip_target_path.exists()) else ""
        ),
        target_wrist_txt_path=(
            str(wrist_target_path.resolve())
            if (not initialized_selected and wrist_target_path.exists()) else ""
        ),
        target_initialized_csv_path=(
            str(initialized_csv_path.resolve()) if initialized_csv_path is not None else ""
        ),
        target_initialization_metadata_path=(
            str(initialized_metadata_path.resolve()) if initialized_metadata_path is not None else ""
        ),
        target_initialization=target_initialization,
    )


def basis2(phase01: np.ndarray) -> np.ndarray:
    phi = 2.0 * np.pi * np.asarray(phase01, dtype=float).reshape(-1)
    return np.column_stack([
        np.ones_like(phi), np.cos(phi), np.sin(phi),
        np.cos(2.0 * phi), np.sin(2.0 * phi),
    ])


def b_x_max_excursion(b_coeff: np.ndarray) -> float:
    """返回连续周期内 ``max|Bx(t)-B_CenterX|`` 的数值精确值。

    Bx 只有一、二阶谐波，因此极值只会出现在导数零点。这里先用固定网格
    包围全部变号根，再二分求根；与仅检查 76/512 个时间点相比，不会漏掉
    采样点之间的峰值。
    """
    coefficient = np.asarray(b_coeff, dtype=float).reshape(-1)
    if coefficient.size != len(B_FOURIER_XYZ3_NAMES):
        return 0.0
    c1, s1, c2, s2 = coefficient[13:17]

    def value(theta: np.ndarray | float) -> np.ndarray:
        angle = np.asarray(theta, dtype=float)
        return (
            c1 * np.cos(angle) + s1 * np.sin(angle)
            + c2 * np.cos(2.0 * angle) + s2 * np.sin(2.0 * angle)
        )

    def derivative(theta: np.ndarray | float) -> np.ndarray:
        angle = np.asarray(theta, dtype=float)
        return (
            -c1 * np.sin(angle) + s1 * np.cos(angle)
            - 2.0 * c2 * np.sin(2.0 * angle)
            + 2.0 * s2 * np.cos(2.0 * angle)
        )

    grid = np.linspace(0.0, 2.0 * np.pi, 257)
    derivative_grid = derivative(grid)
    roots: list[float] = [0.0]
    for index in range(grid.size - 1):
        left = float(grid[index])
        right = float(grid[index + 1])
        left_value = float(derivative_grid[index])
        right_value = float(derivative_grid[index + 1])
        if abs(left_value) <= 1e-13:
            roots.append(left)
        if left_value * right_value < 0.0:
            for _ in range(48):
                middle = 0.5 * (left + right)
                middle_value = float(derivative(middle))
                if left_value * middle_value <= 0.0:
                    right = middle
                else:
                    left = middle
                    left_value = middle_value
            roots.append(0.5 * (left + right))
    return float(np.max(np.abs(value(np.asarray(roots, dtype=float)))))


def basis3(phase01: np.ndarray) -> np.ndarray:
    phi = 2.0 * np.pi * np.asarray(phase01, dtype=float).reshape(-1)
    return np.column_stack([
        np.ones_like(phi), np.cos(phi), np.sin(phi),
        np.cos(2.0 * phi), np.sin(2.0 * phi),
        np.cos(3.0 * phi), np.sin(3.0 * phi),
    ])


def periodic_basis(phase01: np.ndarray, coefficient_count: int) -> np.ndarray:
    if coefficient_count == 5:
        return basis2(phase01)
    if coefficient_count == 7:
        return basis3(phase01)
    if coefficient_count == 9:
        phi = 2.0 * np.pi * np.asarray(phase01, dtype=float).reshape(-1)
        return np.column_stack([
            basis3(phase01), np.cos(4.0 * phi), np.sin(4.0 * phi)
        ])
    raise ValueError(f"Unsupported periodic coefficient count: {coefficient_count}")


def zc_split_fourier_coefficients(
    amplitude_mm: float,
    shape_cos: float,
    shape_sin: float,
) -> np.ndarray:
    """把单连通 C-Z 分离参数展开成标准二阶 Fourier 系数。

    分离相位 ``tau`` 从 C 到 A 距离最大的时间步开始，距离定义为

    ``d(tau)=A*sin(pi*tau)^2*(1+b*cos(2*pi*tau)+c*sin(2*pi*tau))``。

    当 ``sqrt(b^2+c^2)<1`` 时，括号项始终为正，因此距离只在周期首尾
    为零，中间保持非负且连续。展开后最高谐波为二阶。
    """

    amplitude = float(amplitude_mm)
    b = float(shape_cos)
    c = float(shape_sin)
    if not np.isfinite(amplitude) or amplitude < 0.0:
        raise FourBarError("ZC split amplitude must be nonnegative finite.")
    if not np.isfinite(b) or not np.isfinite(c) or np.hypot(b, c) > 0.98 + 1e-12:
        raise FourBarError("require hypot(ZC_ShapeCos,ZC_ShapeSin) <= 0.98.")
    return np.array([
        amplitude * (2.0 - b) / 4.0,
        amplitude * (b - 1.0) / 2.0,
        amplitude * c / 2.0,
        -amplitude * b / 4.0,
        -amplitude * c / 4.0,
    ], dtype=float)


def zc_split_profile(
    frame_count: int,
    amplitude_mm: float,
    shape_cos: float,
    shape_sin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成从分离到重合的二阶 Fourier 距离和对应标准系数。"""

    if frame_count < 3:
        raise ValueError("ZC split profile requires at least three frames.")
    coefficient = zc_split_fourier_coefficients(
        amplitude_mm, shape_cos, shape_sin
    )
    # ZC 单独使用包含首尾的相位，使第一帧和最后一帧都严格为零。
    separation_phase = np.linspace(0.0, 1.0, int(frame_count), endpoint=True)
    values = periodic_basis(separation_phase, 5) @ coefficient
    values[np.abs(values) < 1e-10] = 0.0
    if float(np.min(values)) < -1e-8:
        raise FourBarError("ZC split profile became negative.")
    values = np.maximum(values, 0.0)
    return values, coefficient


def fit_mot_polar_fourier7(phase01: np.ndarray, mot_curve: np.ndarray) -> np.ndarray:
    """旧名称兼容层：拟合 B 的 CenterX/CenterY 与二阶径向傅里叶规律。"""
    basis = basis2(phase01)
    curve = np.asarray(mot_curve, dtype=float)
    if curve.ndim != 2 or curve.shape[1] != 3 or curve.shape[0] != basis.shape[0]:
        raise ValueError("mot_curve must have shape (len(phase01), 3).")
    center_x = float(np.mean(curve[:, 0]))
    center_y = float(np.mean(curve[:, 1]))
    radius = np.sqrt(
        (curve[:, 1] - center_y) ** 2 + curve[:, 2] ** 2
    )
    coeff, *_ = np.linalg.lstsq(basis, radius, rcond=None)
    fitted = np.concatenate([[center_y], coeff.astype(float), [center_x]])

    # 高阶拟合可能产生局部凹陷。逐步收缩谐波、保留平均半径，得到离原拟合
    # 最近的严格凸初始轨迹；后续优化仍可独立改变全部谐波系数。
    for harmonic_scale in np.linspace(1.0, 0.0, 101):
        candidate = fitted.copy()
        candidate[2:6] *= harmonic_scale
        try:
            validate_mot_polar_coeff(candidate)
            return candidate
        except FourBarError:
            continue
    raise FourBarError("Unable to construct a positive convex Mot polar seed.")


def fit_mot_polar_fourier6(phase01: np.ndarray, mot_curve: np.ndarray) -> np.ndarray:
    """历史函数名兼容层；当前返回 7 个 Mot 参数。"""
    return fit_mot_polar_fourier7(phase01, mot_curve)


def fit_mot_polar_fourier10(phase01: np.ndarray, mot_curve: np.ndarray) -> np.ndarray:
    """历史函数名兼容层；当前 Mot 表示为 7 个参数。"""
    return fit_mot_polar_fourier7(phase01, mot_curve)


def fit_mot_polar_fourier11(phase01: np.ndarray, mot_curve: np.ndarray) -> np.ndarray:
    """历史函数名兼容层；当前不再拟合三、四阶项。"""
    return fit_mot_polar_fourier7(phase01, mot_curve)


def fit_b_polar_fourier10(phase01: np.ndarray, b_curve: np.ndarray) -> np.ndarray:
    """兼容旧调用；输入曲线现在表示 Mot。"""
    return fit_mot_polar_fourier10(phase01, b_curve)


def fit_b_fourier9(phase01: np.ndarray, b_curve: np.ndarray) -> np.ndarray:
    """拟合直接二维二阶 Fourier B 曲线；Bz 常数项固定为0。"""
    basis = basis2(phase01)
    curve = np.asarray(b_curve, dtype=float)
    y_coeff, *_ = np.linalg.lstsq(basis, curve[:, 1], rcond=None)
    z_coeff, *_ = np.linalg.lstsq(basis[:, 1:], curve[:, 2], rcond=None)
    coefficient = np.concatenate([y_coeff, z_coeff, [float(np.mean(curve[:, 0]))]])
    return project_mot_polar_coeff(coefficient)


def fit_fourier5(phase01: np.ndarray, values: np.ndarray) -> np.ndarray:
    coefficient, *_ = np.linalg.lstsq(
        basis2(phase01), np.asarray(values, dtype=float).reshape(-1), rcond=None
    )
    return coefficient.astype(float)


def mot_radius_kinematics(
    phase01: np.ndarray | float,
    b_coeff: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 R、dR/dtheta 和 d2R/dtheta2。"""
    shape = np.shape(phase01)
    phase = np.asarray(phase01, dtype=float).reshape(-1)
    coeff = np.asarray(b_coeff, dtype=float).reshape(-1)
    if coeff.size not in (6, 7):
        raise ValueError(
            f"Expected 6 historical or {len(MOT_POLAR_NAMES)} current Mot parameters, "
            f"got {coeff.size}."
        )

    theta = 2.0 * np.pi * phase
    radius = np.full_like(theta, coeff[1])
    radius_d1 = np.zeros_like(theta)
    radius_d2 = np.zeros_like(theta)
    for harmonic in range(1, 3):
        cosine_coeff = coeff[2 * harmonic]
        sine_coeff = coeff[2 * harmonic + 1]
        angle = harmonic * theta
        radius += cosine_coeff * np.cos(angle) + sine_coeff * np.sin(angle)
        radius_d1 += harmonic * (-cosine_coeff * np.sin(angle) + sine_coeff * np.cos(angle))
        radius_d2 -= harmonic ** 2 * (
            cosine_coeff * np.cos(angle) + sine_coeff * np.sin(angle)
        )
    return radius.reshape(shape), radius_d1.reshape(shape), radius_d2.reshape(shape)


def mot_radius_value(phase01: np.ndarray | float, mot_coeff: np.ndarray) -> np.ndarray:
    radius, _radius_d1, _radius_d2 = mot_radius_kinematics(phase01, mot_coeff)
    return radius


def _validate_single_connected_projection(
    first: np.ndarray,
    second: np.ndarray,
    axis_names: str,
) -> None:
    """验证一个正交投影为正则、严格凸且只围成一个连通区域。"""
    du, dv = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    speed2 = du[0] ** 2 + dv[0] ** 2
    curvature_cross = du[0] * dv[1] - dv[0] * du[1]
    speed_scale = max(float(np.max(speed2)), 1.0)
    curvature_scale = max(float(np.max(np.abs(curvature_cross))), 1.0)
    if np.any(~np.isfinite(speed2)) or float(np.min(speed2)) <= 1e-8 * speed_scale:
        raise FourBarError(f"B {axis_names} projection must be a regular closed curve")
    margin = 1e-8 * curvature_scale
    if not (
        float(np.min(curvature_cross)) > margin
        or float(np.max(curvature_cross)) < -margin
    ):
        raise FourBarError(
            f"B {axis_names} projection must be strictly convex and single-connected"
        )
    tangent = np.arctan2(dv[0], du[0])
    increments = np.angle(np.exp(1j * (np.roll(tangent, -1) - tangent)))
    total_turn = float(np.sum(increments))
    if not 1.9 * np.pi <= abs(total_turn) <= 2.1 * np.pi:
        raise FourBarError(
            f"B {axis_names} projection must have exactly one winding"
        )


def _b_xyz3_derivatives(
    coefficient: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 B 三轴 Fourier 曲线对相位角的一阶、二阶导数。"""
    c = np.asarray(coefficient, dtype=float).reshape(-1)
    first = np.empty((theta.size, 3), dtype=float)
    second = np.empty((theta.size, 3), dtype=float)

    axis_terms = (
        ((c[13], c[14]), (c[15], c[16]), (0.0, 0.0)),
        ((c[1], c[2]), (c[3], c[4]), (0.0, 0.0)),
        ((c[5], c[6]), (c[7], c[8]), (c[11], c[12])),
    )
    for axis, harmonics in enumerate(axis_terms):
        d1 = np.zeros_like(theta)
        d2 = np.zeros_like(theta)
        for harmonic, (cosine, sine) in enumerate(harmonics, start=1):
            angle = harmonic * theta
            d1 += harmonic * (-cosine * np.sin(angle) + sine * np.cos(angle))
            d2 -= harmonic ** 2 * (cosine * np.cos(angle) + sine * np.sin(angle))
        first[:, axis] = d1
        second[:, axis] = d2
    return first, second


def validate_mot_polar_coeff(mot_coeff: np.ndarray, samples: int = 512) -> None:
    """验证 B 曲线；10 参数模式要求直接二维曲线严格凸且单连通。"""
    coefficient = np.asarray(mot_coeff, dtype=float).reshape(-1)
    if coefficient.size == len(B_FOURIER_XYZ3_NAMES):
        if np.any(~np.isfinite(coefficient)):
            raise FourBarError("B Fourier coefficients must be finite")
        phase = np.linspace(0.0, 1.0, max(128, int(samples)), endpoint=False)
        if b_x_max_excursion(coefficient) > B_X_MAX_EXCURSION_MM + 1e-8:
            raise FourBarError(
                f"require max|Bx(t)-B_CenterX| <= {B_X_MAX_EXCURSION_MM:g} mm"
            )
        theta = 2.0 * np.pi * phase
        first, second = _b_xyz3_derivatives(coefficient, theta)
        for axis_u, axis_v, label in ((0, 1, "XY"), (1, 2, "YZ"), (0, 2, "XZ")):
            _validate_single_connected_projection(
                (first[:, axis_u], second[:, axis_u]),
                (first[:, axis_v], second[:, axis_v]),
                label,
            )
        return
    if coefficient.size in (len(B_FOURIER_Z3_NAMES), len(B_FOURIER_Z3_C0_NAMES)):
        phase = np.linspace(0.0, 1.0, max(64, int(samples)), endpoint=False)
        theta = 2.0 * np.pi * phase
        yc1, ys1, yc2, ys2 = coefficient[1:5]
        zc1, zs1, zc2, zs2 = coefficient[5:9]
        zc3, zs3 = (
            coefficient[11:13]
            if coefficient.size == len(B_FOURIER_Z3_C0_NAMES)
            else coefficient[10:12]
        )
        dy = -yc1*np.sin(theta)+ys1*np.cos(theta)-2*yc2*np.sin(2*theta)+2*ys2*np.cos(2*theta)
        dz = (-zc1*np.sin(theta)+zs1*np.cos(theta)-2*zc2*np.sin(2*theta)
              +2*zs2*np.cos(2*theta)-3*zc3*np.sin(3*theta)+3*zs3*np.cos(3*theta))
        ddy = -yc1*np.cos(theta)-ys1*np.sin(theta)-4*yc2*np.cos(2*theta)-4*ys2*np.sin(2*theta)
        ddz = (-zc1*np.cos(theta)-zs1*np.sin(theta)-4*zc2*np.cos(2*theta)
               -4*zs2*np.sin(2*theta)-9*zc3*np.cos(3*theta)-9*zs3*np.sin(3*theta))
        speed2 = dy*dy + dz*dz
        cross = dy*ddz - dz*ddy
        if np.any(~np.isfinite(speed2)) or float(np.min(speed2)) <= 1e-6:
            raise FourBarError("require a regular closed B Fourier curve")
        if not (float(np.min(cross)) > 1e-5 or float(np.max(cross)) < -1e-5):
            raise FourBarError("require a strictly convex, single-connected B Fourier curve")
        return
    if coefficient.size in (len(B_FOURIER_NAMES), len(B_FOURIER_X_NAMES)):
        phase = np.linspace(0.0, 1.0, max(64, int(samples)), endpoint=False)
        theta = 2.0 * np.pi * phase
        yc1, ys1, yc2, ys2 = coefficient[1:5]
        zc1, zs1, zc2, zs2 = coefficient[5:9]
        dy = -yc1*np.sin(theta)+ys1*np.cos(theta)-2*yc2*np.sin(2*theta)+2*ys2*np.cos(2*theta)
        dz = -zc1*np.sin(theta)+zs1*np.cos(theta)-2*zc2*np.sin(2*theta)+2*zs2*np.cos(2*theta)
        ddy = -yc1*np.cos(theta)-ys1*np.sin(theta)-4*yc2*np.cos(2*theta)-4*ys2*np.sin(2*theta)
        ddz = -zc1*np.cos(theta)-zs1*np.sin(theta)-4*zc2*np.cos(2*theta)-4*zs2*np.sin(2*theta)
        speed2 = dy*dy + dz*dz
        cross = dy*ddz - dz*ddy
        if np.any(~np.isfinite(speed2)) or float(np.min(speed2)) <= 1e-6:
            raise FourBarError("require a regular closed B Fourier curve")
        if not (float(np.min(cross)) > 1e-5 or float(np.max(cross)) < -1e-5):
            raise FourBarError("require a strictly convex, single-connected B Fourier curve")
        return
    phase = np.linspace(0.0, 1.0, max(64, int(samples)), endpoint=False)
    radius, radius_d1, radius_d2 = mot_radius_kinematics(phase, coefficient)
    if np.any(~np.isfinite(radius)) or float(np.min(radius)) < 30.0 - 1e-8:
        raise FourBarError("require B polar radius R(theta) >= 30 mm")
    curvature_numerator = radius ** 2 + 2.0 * radius_d1 ** 2 - radius * radius_d2
    if np.any(~np.isfinite(curvature_numerator)) or float(np.min(curvature_numerator)) <= 1e-6:
        raise FourBarError("require a strictly convex B polar curve")


def project_mot_polar_coeff(mot_coeff: np.ndarray, samples: int = 512) -> np.ndarray:
    """把任意 B 谐波组合投影到最近的严格凸径向曲线。

    Mot 中心和平均半径 C0 保持不变，只统一收缩四个径向谐波系数。这样既保留
    CMA-ES 选择的形状方向，又避免绝大多数随机候选因局部负曲率直接失效。
    """
    coefficient = np.asarray(mot_coeff, dtype=float).reshape(-1).copy()
    if coefficient.size == len(B_FOURIER_XYZ3_NAMES):
        if np.any(~np.isfinite(coefficient)):
            raise FourBarError("B Fourier coefficients must be finite")
        x_indices = np.array([13, 14, 15, 16], dtype=int)
        maximum = b_x_max_excursion(coefficient)
        if maximum > B_X_MAX_EXCURSION_MM:
            coefficient[x_indices] *= B_X_MAX_EXCURSION_MM / maximum
        try:
            validate_mot_polar_coeff(coefficient, samples=samples)
            return coefficient
        except FourBarError:
            pass
        # 保留一阶主椭圆，只收缩高阶项，直至三个正交投影均为单连通域。
        high_indices = np.array([3, 4, 7, 8, 11, 12, 15, 16], dtype=int)
        higher = coefficient[high_indices].copy()
        lower_scale, upper_scale = 0.0, 1.0
        for _ in range(36):
            scale = 0.5 * (lower_scale + upper_scale)
            trial = coefficient.copy()
            trial[high_indices] = scale * higher
            try:
                validate_mot_polar_coeff(trial, samples=samples)
                lower_scale = scale
            except FourBarError:
                upper_scale = scale
        coefficient[high_indices] = 0.995 * lower_scale * higher
        validate_mot_polar_coeff(coefficient, samples=samples)
        return coefficient
    if coefficient.size in (len(B_FOURIER_Z3_NAMES), len(B_FOURIER_Z3_C0_NAMES)):
        first = np.array([[coefficient[1], coefficient[2]], [coefficient[5], coefficient[6]]])
        u, singular, vt = np.linalg.svd(first)
        singular = np.clip(singular, 30.0, 150.0)
        first = u @ np.diag(singular) @ vt
        coefficient[1], coefficient[2] = first[0]
        coefficient[5], coefficient[6] = first[1]
        try:
            validate_mot_polar_coeff(coefficient, samples=samples)
            return coefficient
        except FourBarError:
            pass
        high_indices = np.array(
            [3, 4, 7, 8, 11, 12]
            if coefficient.size == len(B_FOURIER_Z3_C0_NAMES)
            else [3, 4, 7, 8, 10, 11],
            dtype=int,
        )
        higher = coefficient[high_indices].copy()
        lower_scale, upper_scale = 0.0, 1.0
        for _ in range(36):
            scale = 0.5 * (lower_scale + upper_scale)
            trial = coefficient.copy()
            trial[high_indices] = scale * higher
            try:
                validate_mot_polar_coeff(trial, samples=samples)
                lower_scale = scale
            except FourBarError:
                upper_scale = scale
        coefficient[high_indices] = 0.995 * lower_scale * higher
        validate_mot_polar_coeff(coefficient, samples=samples)
        return coefficient
    if coefficient.size in (len(B_FOURIER_NAMES), len(B_FOURIER_X_NAMES)):
        # 一阶谐波是单位圆的仿射像。把两个奇异值限制在30-150 mm，
        # 再只收缩二阶谐波，得到严格凸、非自交且内部单连通的闭曲线。
        first = np.array([[coefficient[1], coefficient[2]], [coefficient[5], coefficient[6]]])
        u, singular, vt = np.linalg.svd(first)
        singular = np.clip(singular, 30.0, 150.0)
        first = u @ np.diag(singular) @ vt
        coefficient[1], coefficient[2] = first[0]
        coefficient[5], coefficient[6] = first[1]
        try:
            validate_mot_polar_coeff(coefficient, samples=samples)
            return coefficient
        except FourBarError:
            pass
        second = coefficient[[3, 4, 7, 8]].copy()
        lower_scale, upper_scale = 0.0, 1.0
        for _ in range(36):
            scale = 0.5 * (lower_scale + upper_scale)
            trial = coefficient.copy()
            trial[[3, 4, 7, 8]] = scale * second
            try:
                validate_mot_polar_coeff(trial, samples=samples)
                lower_scale = scale
            except FourBarError:
                upper_scale = scale
        coefficient[[3, 4, 7, 8]] = 0.995 * lower_scale * second
        validate_mot_polar_coeff(coefficient, samples=samples)
        return coefficient
    if coefficient.size != 7:
        raise ValueError(
            f"Expected {len(MOT_POLAR_NAMES)} Mot polar parameters, got {coefficient.size}."
        )
    coefficient[1] = max(float(coefficient[1]), 30.0)
    try:
        validate_mot_polar_coeff(coefficient, samples=samples)
        return coefficient
    except FourBarError:
        pass

    harmonics = coefficient[2:6].copy()
    lower_scale, upper_scale = 0.0, 1.0
    for _ in range(36):
        middle_scale = 0.5 * (lower_scale + upper_scale)
        trial = coefficient.copy()
        trial[2:6] = middle_scale * harmonics
        try:
            validate_mot_polar_coeff(trial, samples=samples)
            lower_scale = middle_scale
        except FourBarError:
            upper_scale = middle_scale
    coefficient[2:6] = 0.999 * lower_scale * harmonics
    validate_mot_polar_coeff(coefficient, samples=samples)
    return coefficient


# 旧函数名作为兼容层；新代码应使用 Mot 名称。
def b_radius_kinematics(
    phase01: np.ndarray | float,
    b_coeff: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return mot_radius_kinematics(phase01, b_coeff)


def b_radius_value(phase01: np.ndarray | float, b_coeff: np.ndarray) -> np.ndarray:
    return mot_radius_value(phase01, b_coeff)


def validate_b_polar_coeff(b_coeff: np.ndarray, samples: int = 512) -> None:
    validate_mot_polar_coeff(b_coeff, samples)


def bz_value(phase01: np.ndarray | float, b_coeff: np.ndarray) -> np.ndarray:
    phase = np.asarray(phase01, dtype=float)
    coefficient = np.asarray(b_coeff, dtype=float).reshape(-1)
    if coefficient.size in (
        len(B_FOURIER_NAMES), len(B_FOURIER_X_NAMES), len(B_FOURIER_Z3_NAMES),
        len(B_FOURIER_Z3_C0_NAMES),
        len(B_FOURIER_XYZ3_NAMES),
    ):
        theta = 2.0 * np.pi * phase
        zc1, zs1, zc2, zs2 = coefficient[5:9]
        value = (
            zc1 * np.cos(theta)
            + zs1 * np.sin(theta)
            + zc2 * np.cos(2.0 * theta)
            + zs2 * np.sin(2.0 * theta)
        )
        if coefficient.size == len(B_FOURIER_Z3_NAMES):
            value = value + coefficient[10] * np.cos(3.0 * theta) + coefficient[11] * np.sin(3.0 * theta)
        elif coefficient.size == len(B_FOURIER_Z3_C0_NAMES):
            value = (
                value + coefficient[10]
                + coefficient[11] * np.cos(3.0 * theta)
                + coefficient[12] * np.sin(3.0 * theta)
            )
        elif coefficient.size == len(B_FOURIER_XYZ3_NAMES):
            value = (
                value + coefficient[10]
                + coefficient[11] * np.cos(3.0 * theta)
                + coefficient[12] * np.sin(3.0 * theta)
            )
        return value
    theta = 2.0 * np.pi * phase
    return -b_radius_value(phase01, b_coeff) * np.sin(theta)


def phase_origin_from_b_coeff(b_coeff: np.ndarray) -> float:
    """theta=0 对应 B_z=0，随后 B_z 沿负方向运动。"""
    return 0.0


def direct_b_phase_origin(b_coeff: np.ndarray) -> float:
    """令第一帧位于 Bz=0 且随后沿负 z 方向运动。"""
    coefficient = np.asarray(b_coeff, dtype=float).reshape(-1)
    if coefficient.size not in (
        len(B_FOURIER_NAMES), len(B_FOURIER_X_NAMES), len(B_FOURIER_Z3_NAMES),
        len(B_FOURIER_Z3_C0_NAMES),
        len(B_FOURIER_XYZ3_NAMES),
    ):
        return phase_origin_from_b_coeff(coefficient)

    zc1, zs1, zc2, zs2 = coefficient[5:9]
    if coefficient.size == len(B_FOURIER_XYZ3_NAMES):
        zc3, zs3 = coefficient[11:13]
    elif coefficient.size == len(B_FOURIER_Z3_C0_NAMES):
        zc3, zs3 = coefficient[11:13]
    elif coefficient.size == len(B_FOURIER_Z3_NAMES):
        zc3, zs3 = coefficient[10:12]
    else:
        zc3, zs3 = 0.0, 0.0

    def downward_derivative(root: float) -> float:
        theta = 2.0 * np.pi * root
        return float(
            -zc1 * np.sin(theta)
            + zs1 * np.cos(theta)
            - 2.0 * zc2 * np.sin(2.0 * theta)
            + 2.0 * zs2 * np.cos(2.0 * theta)
            - 3.0 * zc3 * np.sin(3.0 * theta)
            + 3.0 * zs3 * np.cos(3.0 * theta)
        )

    def bracketed_roots(sample_count: int) -> list[float]:
        sample_phase = np.linspace(0.0, 1.0, sample_count)
        sample_z = np.asarray(bz_value(sample_phase, coefficient), dtype=float)
        roots: list[float] = []
        exact = np.flatnonzero(np.abs(sample_z[:-1]) <= 1e-12)
        roots.extend(float(sample_phase[index]) for index in exact)
        crossing = np.flatnonzero(sample_z[:-1] * sample_z[1:] < 0.0)
        for index in crossing:
            left_phase = float(sample_phase[index])
            right_phase = float(sample_phase[index + 1])
            left_value = float(sample_z[index])
            # 40 次二分在 1/128 的初始区间内已远小于双精度轨迹误差。
            for _ in range(40):
                middle_phase = 0.5 * (left_phase + right_phase)
                middle_value = float(bz_value(middle_phase, coefficient))
                if left_value * middle_value <= 0.0:
                    right_phase = middle_phase
                else:
                    left_phase = middle_phase
                    left_value = middle_value
            roots.append(0.5 * (left_phase + right_phase))
        unique: list[float] = []
        for root in sorted(float(value % 1.0) for value in roots):
            if not unique or abs(root - unique[-1]) > 1e-10:
                unique.append(root)
        return unique

    # 三阶严格凸正则曲线先使用 128 个区间；极端参数化情况下仍回退到
    # 原来的 1024 个区间，保持模型判定的稳健性。
    for sample_count in (129, 1025):
        roots = bracketed_roots(sample_count)
        for root in roots:
            if downward_derivative(root) < -1e-9:
                return float(root % 1.0)
    raise FourBarError("B Fourier curve has no downward Bz=0 phase crossing")


def canonicalize_direct_b_phase(
    b_coeff: np.ndarray,
    phase_origin: float | None = None,
) -> np.ndarray:
    """把直接 B 曲线系数转换为第一帧向下穿越 Bz=0 的唯一相位表示。"""

    coefficient = np.asarray(b_coeff, dtype=float).reshape(-1).copy()
    if coefficient.size not in (
        len(B_FOURIER_NAMES),
        len(B_FOURIER_X_NAMES),
        len(B_FOURIER_Z3_NAMES),
        len(B_FOURIER_Z3_C0_NAMES),
        len(B_FOURIER_XYZ3_NAMES),
    ):
        return coefficient
    origin = (
        direct_b_phase_origin(coefficient)
        if phase_origin is None
        else float(phase_origin)
    )

    def rotate_pair(cosine_index: int, sine_index: int, harmonic: int) -> None:
        cosine = float(coefficient[cosine_index])
        sine = float(coefficient[sine_index])
        angle = 2.0 * np.pi * harmonic * origin
        coefficient[cosine_index] = (
            cosine * np.cos(angle) + sine * np.sin(angle)
        )
        coefficient[sine_index] = (
            -cosine * np.sin(angle) + sine * np.cos(angle)
        )

    for cosine_index, sine_index, harmonic in (
        (1, 2, 1),
        (3, 4, 2),
        (5, 6, 1),
        (7, 8, 2),
    ):
        rotate_pair(cosine_index, sine_index, harmonic)
    if coefficient.size == len(B_FOURIER_Z3_NAMES):
        rotate_pair(10, 11, 3)
    elif coefficient.size == len(B_FOURIER_Z3_C0_NAMES):
        rotate_pair(11, 12, 3)
    elif coefficient.size == len(B_FOURIER_XYZ3_NAMES):
        rotate_pair(11, 12, 3)
        rotate_pair(13, 14, 1)
        rotate_pair(15, 16, 2)
    elif coefficient.size == len(B_FOURIER_X_NAMES):
        rotate_pair(10, 11, 1)
        rotate_pair(12, 13, 2)
    return coefficient


def decode_mot_curve(phase01: np.ndarray, mot_coeff: np.ndarray) -> np.ndarray:
    """旧名称兼容层；当前正式模式可同时解码 B 的 x/y/z 周期运动。"""
    return decode_mot_curve_in_yz_plane(phase01, mot_coeff)


def decode_mot_curve_in_yz_plane(
    phase01: np.ndarray,
    mot_coeff: np.ndarray,
    center_y: float | None = None,
    center_z: float = 0.0,
) -> np.ndarray:
    """解码 B 曲线；函数名保留用于旧调用，当前模式可同时改变 x/y/z。"""
    phase = np.asarray(phase01, dtype=float).reshape(-1)
    theta = 2.0 * np.pi * phase
    coeff = np.asarray(mot_coeff, dtype=float).reshape(-1)
    if coeff.size in (
        len(B_FOURIER_NAMES), len(B_FOURIER_X_NAMES), len(B_FOURIER_Z3_NAMES),
        len(B_FOURIER_Z3_C0_NAMES),
        len(B_FOURIER_XYZ3_NAMES),
    ):
        basis = basis2(phase)
        y = basis @ coeff[:5]
        z = basis[:, 1:] @ coeff[5:9]
        x_values = np.full_like(y, coeff[9])
        if coeff.size == len(B_FOURIER_Z3_NAMES):
            z = z + coeff[10] * np.cos(3.0 * theta) + coeff[11] * np.sin(3.0 * theta)
        if coeff.size == len(B_FOURIER_Z3_C0_NAMES):
            z = z + coeff[10] + coeff[11] * np.cos(3.0 * theta) + coeff[12] * np.sin(3.0 * theta)
        if coeff.size == len(B_FOURIER_XYZ3_NAMES):
            z = z + coeff[10] + coeff[11] * np.cos(3.0 * theta) + coeff[12] * np.sin(3.0 * theta)
            x_values = x_values + basis[:, 1:] @ coeff[13:17]
        if coeff.size == len(B_FOURIER_X_NAMES):
            x_values = x_values + basis[:, 1:] @ coeff[10:14]
        return np.column_stack([x_values, y, z])
    if coeff.size not in (6, 7):
        raise ValueError(
            f"Expected 6 historical or {len(MOT_POLAR_NAMES)} current Mot parameters, "
            f"got {coeff.size}."
        )
    curve_center_x = float(coeff[6]) if coeff.size == 7 else 0.0
    curve_center_y = float(coeff[0]) if center_y is None else float(center_y)
    radius = mot_radius_value(phase, mot_coeff)
    return np.column_stack([
        np.full_like(radius, curve_center_x),
        curve_center_y - radius * np.cos(theta),
        -radius * np.sin(theta),
    ])


def decode_b_curve(phase01: np.ndarray, b_coeff: np.ndarray) -> np.ndarray:
    """解码真实 B 输入轨迹。"""
    return decode_mot_curve(phase01, b_coeff)


def base_static_values(data: ProblemData) -> np.ndarray:
    params = FourBarParams.from_base32(data.base32)
    return np.array([getattr(params, name) for name in STATIC_NAMES], dtype=float)


# RMSE25 主基线（Combined RMSE=25.150 mm）的同名静态变量边界。该历史
# 模型使用 direct-B 输入，因此这里只桥接物理含义相同的静态几何参数；
# B 逐帧变量不能直接当作当前 Mot 傅里叶变量边界。
RMSE25_SHARED_STATIC_BOUNDS: Dict[str, Tuple[float, float]] = {
    "L2": (109.57419804791778, 244.43474949150888),
    "L31": (66.53299548568815, 148.41975916038123),
    "L32": (10.0, 300.0),
    "L4": (52.61173711126404, 117.36464432512747),
    "L41": (0.00005, 0.5),
    "L5": (143.65696602084003, 320.4655395849508),
    "L51": (0.00000005, 0.5),
    "L52": (12.770218193189274, 28.487409815576072),
    "L7": (197.17144338742511, 439.8439890950253),
    "L8": (193.7623730780765, 432.23913994340137),
    "L9": (43.168763465986416, 96.29954927027738),
    "L10": (31.622034914929944, 70.54146250253602),
    "L11": (99.44285717541298, 221.83406600669048),
    # 旧代码 L_rod 对应当前 L12。
    "L12": (1.9294893369830677, 12.005711430116865),
    "LRod": (32.212796741873355, 200.435179727212),
    "L14": (6.630134532971385, 41.2541704273775),
    "L15": (43.724184644572546, 272.06159334400695),
    "L17": (33.295318625545605, 207.17087144783932),
    "H_finger": (1.271477368310462, 15.893467103880774),
    "L3": (198.56607095996085, 442.9550813722203),
    "L61": (95.50848666227898, 183.67016665822882),
    "Lf1": (261.9813394835053, 436.63556580584213),
    "Lf2": (167.76381947334832, 279.6063657889139),
    "L_CZ": (130.61297204853435, 291.36739918519197),
    "L_down": (17.314008460413454, 89.04347208212634),
}

# 用户于 2026-07-20 明确给出的本轮权威范围。未在原表列出的 H_finger 和
# Lf1 由当前模型补充；L_CZ 按用户要求固定为原始值 62 mm。
USER_TABLE_STATIC_BOUNDS: Dict[str, Tuple[float, float]] = {
    "L1": (10.0, 50.0),
    "L2": (10.0, 250.0),
    "L31": (10.0, 150.0),
    # 新增独立固定杆；上界覆盖原始 101 mm，并与相邻主杆尺度一致。
    "L32": (10.0, 300.0),
    "L4": (10.0, 100.0),
    "L41": (5.0, 50.0),
    "L5": (150.0, 300.0),
    "L51": (10.0, 50.0),
    "L52": (10.0, 50.0),
    "L6": (200.0, 300.0),
    "L7": (200.0, 300.0),
    "L8": (200.0, 300.0),
    "L9": (10.0, 100.0),
    "L10": (10.0, 50.0),
    "L11": (10.0, 100.0),
    "L12": (5.0, 50.0),
    "L13": (10.0, 50.0),
    "LRod": (10.0, 150.0),
    "L14": (10.0, 50.0),
    "L15": (10.0, 150.0),
    "L17": (10.0, 100.0),
    "H_finger": (0.5, 15.0),
    "L3": (100.0, 350.0),
    "L61": (100.0, 250.0),
    "Lf1": (250.0, 520.0),
    "Lf2": (200.0, 300.0),
    "L_CZ": (62.0, 62.0),
    "L_down": (10.0, 50.0),
    "theta18_deg": (30.0, 150.0),
}

# 2026-07-27 边界审计扩展配置。该配置不改变变量数量、初始值或机构拓扑，
# 只扩展在无锚点长时搜索中反复贴边的方向。旧的 user_table 配置保持不变，
# 因而旧实验仍可按原边界严格复现。
ADAPTIVE_EXPANDED_BOUND_MODE = "adaptive_expanded_20260727"
ADAPTIVE_EXPANDED_V2_BOUND_MODE = "adaptive_expanded_v2_20260727"
ADAPTIVE_EXPANDED_V3_BOUND_MODE = "adaptive_expanded_v3_20260727"
ADAPTIVE_EXPANDED_V4_BOUND_MODE = "adaptive_expanded_v4_20260728"
ADAPTIVE_EXPANDED_V5_BOUND_MODE = "adaptive_expanded_v5_20260728"
ADAPTIVE_EXPANDED_V6_BOUND_MODE = "adaptive_expanded_v6_20260728"
BROAD_ALL54_BOUND_MODE = "broad_all54_20260802"
ADAPTIVE_BOUND_MODES = {
    ADAPTIVE_EXPANDED_BOUND_MODE,
    ADAPTIVE_EXPANDED_V2_BOUND_MODE,
    ADAPTIVE_EXPANDED_V3_BOUND_MODE,
    ADAPTIVE_EXPANDED_V4_BOUND_MODE,
    ADAPTIVE_EXPANDED_V5_BOUND_MODE,
    ADAPTIVE_EXPANDED_V6_BOUND_MODE,
    BROAD_ALL54_BOUND_MODE,
}
USER_TABLE_BOUND_MODES = {"user_table", *ADAPTIVE_BOUND_MODES}
ADAPTIVE_EXPANDED_STATIC_BOUNDS: Dict[str, Tuple[float, float]] = {
    # L5 最优值距原下界不足 2%，向下扩展 30 mm。
    "L5": (120.0, 300.0),
    # L3 位于原范围下侧 10% 内，保守向下扩展 20 mm。
    "L3": (80.0, 350.0),
    # Lf2 位于原范围上侧 12% 内，向上扩展 30 mm。
    "Lf2": (200.0, 330.0),
}
ADAPTIVE_EXPANDED_V4_STATIC_BOUNDS: Dict[str, Tuple[float, float]] = {
    **ADAPTIVE_EXPANDED_STATIC_BOUNDS,
    # 2026-07-28 分块精修候选将 L_down 推到 50 mm 上界，仅扩展受阻方向。
    "L_down": (10.0, 100.0),
}
ADAPTIVE_EXPANDED_V5_STATIC_BOUNDS: Dict[str, Tuple[float, float]] = {
    **ADAPTIVE_EXPANDED_V4_STATIC_BOUNDS,
    # V4 正式候选分别达到原上界的 97.6% 与 97.6%。
    "L13": (10.0, 80.0),
    "L17": (10.0, 150.0),
}
ADAPTIVE_EXPANDED_V6_STATIC_BOUNDS: Dict[str, Tuple[float, float]] = {
    **ADAPTIVE_EXPANDED_V5_STATIC_BOUNDS,
    # V5 核验解的 L2 位于上界侧 9.6%；仅沿该受阻方向扩展。
    "L2": (10.0, 300.0),
}
ADAPTIVE_EXPANDED_ALL54_STATIC_BOUNDS: Dict[str, Tuple[float, float]] = {
    **ADAPTIVE_EXPANDED_V6_STATIC_BOUNDS,
    # Broad no-anchor coverage. These are engineering search limits, not an
    # embedding of any previous solution.
    "L2": (10.0, 350.0),
    "L3": (60.0, 420.0),
    "L5": (100.0, 340.0),
    "L13": (8.0, 100.0),
    "L17": (8.0, 180.0),
    "Lf2": (180.0, 380.0),
    "L_down": (8.0, 130.0),
}


def static_bounds(
    values: np.ndarray,
    minimum_mm: float = 10.0,
    bound_mode: str = "current",
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    lb = 0.65 * values
    ub = 1.45 * values
    for index, name in enumerate(STATIC_NAMES):
        value = float(values[index])
        if name == "L13":
            lb[index], ub[index] = 5.0, 25.0
            continue
        if name == "theta18_deg":
            lb[index], ub[index] = 1.0, 179.0
            continue
        if name in {"Lf1", "Lf2"}:
            lb[index], ub[index] = 0.75 * value, 1.25 * value
        elif name == "H_finger":
            lb[index], ub[index] = 0.20 * value, max(8.0, 2.50 * value)
        elif name == "L_down":
            lb[index], ub[index] = 0.35 * value, max(45.0, 1.80 * value)
        elif name == "L61":
            lb[index], ub[index] = 0.65 * value, max(value + 20.0, 1.25 * value)
        elif name in {"L12", "LRod", "L14", "L15", "L17"}:
            lb[index], ub[index] = 0.45 * value, max(value + 5.0, 2.80 * value)
        parameter_minimum = 0.1 if name == "H_finger" else float(minimum_mm)
        lb[index] = max(parameter_minimum, float(lb[index]))
        ub[index] = max(float(lb[index]) + 1e-4, float(ub[index]))
    if bound_mode in USER_TABLE_BOUND_MODES:
        for index, name in enumerate(STATIC_NAMES):
            if name in USER_TABLE_STATIC_BOUNDS:
                lb[index], ub[index] = USER_TABLE_STATIC_BOUNDS[name]
        if bound_mode in {
            ADAPTIVE_EXPANDED_BOUND_MODE,
            ADAPTIVE_EXPANDED_V3_BOUND_MODE,
            ADAPTIVE_EXPANDED_V4_BOUND_MODE,
            ADAPTIVE_EXPANDED_V5_BOUND_MODE,
            ADAPTIVE_EXPANDED_V6_BOUND_MODE,
            BROAD_ALL54_BOUND_MODE,
        }:
            expanded = (
                ADAPTIVE_EXPANDED_ALL54_STATIC_BOUNDS
                if bound_mode == BROAD_ALL54_BOUND_MODE
                else ADAPTIVE_EXPANDED_V6_STATIC_BOUNDS
                if bound_mode == ADAPTIVE_EXPANDED_V6_BOUND_MODE
                else ADAPTIVE_EXPANDED_V5_STATIC_BOUNDS
                if bound_mode == ADAPTIVE_EXPANDED_V5_BOUND_MODE
                else ADAPTIVE_EXPANDED_V4_STATIC_BOUNDS
                if bound_mode == ADAPTIVE_EXPANDED_V4_BOUND_MODE
                else ADAPTIVE_EXPANDED_STATIC_BOUNDS
            )
            for index, name in enumerate(STATIC_NAMES):
                if name in expanded:
                    lb[index], ub[index] = expanded[name]
        elif bound_mode == ADAPTIVE_EXPANDED_V2_BOUND_MODE:
            expanded_v2 = {
                **ADAPTIVE_EXPANDED_STATIC_BOUNDS,
                # 一级扩展的最好候选将 L2 推到原上界的 99.6%。
                "L2": (10.0, 320.0),
            }
            for index, name in enumerate(STATIC_NAMES):
                if name in expanded_v2:
                    lb[index], ub[index] = expanded_v2[name]
    elif bound_mode == "rmse25_union":
        for index, name in enumerate(STATIC_NAMES):
            if name not in RMSE25_SHARED_STATIC_BOUNDS:
                continue
            reference_lower, reference_upper = RMSE25_SHARED_STATIC_BOUNDS[name]
            # 工程下限仍优先：普通杆长不得低于 minimum_mm；H_finger 是几何
            # 高度而非杆件，沿用其原有 0.1 mm 下限。
            parameter_minimum = 0.1 if name == "H_finger" else float(minimum_mm)
            lb[index] = max(
                parameter_minimum,
                min(float(lb[index]), float(reference_lower)),
            )
            ub[index] = max(float(ub[index]), float(reference_upper))
    elif bound_mode != "current":
        raise ValueError(f"Unsupported static bound mode: {bound_mode}")
    return lb, ub


def params_from_static(static_values: np.ndarray, data: ProblemData) -> FourBarParams:
    static_values = np.asarray(static_values, dtype=float).reshape(-1)
    if static_values.size != len(STATIC_NAMES):
        raise ValueError(f"Expected {len(STATIC_NAMES)} static parameters, got {static_values.size}.")
    params = FourBarParams.from_base32(data.base32)
    updates = {name: float(value) for name, value in zip(STATIC_NAMES, static_values)}
    return replace(params, **updates)


def rotation_xyz_matrix(rx_rad: float, ry_rad: float, rz_rad: float) -> np.ndarray:
    cx, sx = float(np.cos(rx_rad)), float(np.sin(rx_rad))
    cy, sy = float(np.cos(ry_rad)), float(np.sin(ry_rad))
    cz, sz = float(np.cos(rz_rad)), float(np.sin(rz_rad))
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def target_pose_center(data: ProblemData) -> np.ndarray:
    return np.mean(np.vstack([data.target_tip, data.target_wrist]), axis=0)


def apply_target_pose(points: np.ndarray, target_pose: np.ndarray, center: np.ndarray) -> np.ndarray:
    pose = np.asarray(target_pose, dtype=float).reshape(-1)
    translation = pose[:3]
    scale = float(pose[6])
    centered = np.asarray(points, dtype=float) - np.asarray(center, dtype=float).reshape(1, 3)
    rotated = centered @ rotation_xyz_matrix(*pose[3:6]).T
    return scale * rotated + np.asarray(center).reshape(1, 3) + translation.reshape(1, 3)


def build_design_space(
    data: ProblemData,
    minimum_static_mm: float = 10.0,
    target_pose_mode: str = "shared_fixed_rotation_translation3",
    periodic_length_mode: str = CURRENT_L32_DERIVED_MODE,
    static_bound_mode: str = "current",
    b_curve_mode: str = "fourier_z3",
) -> DesignSpace:
    """从原始参数构造完整优化空间，不使用历史最优锚点。

    当前目标位姿默认使用 ``shared_fixed_rotation_translation3``：
    Tip/Wrist 共用图示指定的固定三轴旋转，只优化共同三轴平移，尺度固定为 1。

    ``periodic_length_mode`` 可选择历史单周期杆模式，也可选择双周期模式：
    ``l7_l8_periodic`` 使用二阶 L7/L8，``l7_l8_periodic3`` 使用三阶
    L7/L8；二者都满足 C=Z 且不含 ZC 变量。``l3_l7_periodic`` 使用
    L3/L7 且 L8 固定；
    ``l3_l8_periodic3_l5_fixed_l32_derived`` 是当前正式模式：L5、L6、L31
    为周期内固定优化变量，L32(t)=L3(t)-L31+2，仅 L3(t)、L8(t) 使用三阶
    Fourier，且 C=Z。旧的 L3/L5/L8 三周期杆模式只用于读取历史结果。
    ``l7_periodic_l3_fixed`` 仅使用 L7(t)。所有包含 ``ZC`` 的历史模式
    均已禁用；当前模型只允许 C 与 Z 合并的拓扑。
    """
    if "zc" in periodic_length_mode.lower():
        raise ValueError(
            "C and Z are merged in the current model; ZC separation modes are disabled."
        )
    static0 = base_static_values(data)
    static_lb, static_ub = static_bounds(
        static0,
        minimum_static_mm,
        bound_mode=static_bound_mode,
    )
    static_by_name = dict(zip(STATIC_NAMES, static0))
    lb_by_name = dict(zip(STATIC_NAMES, static_lb))
    ub_by_name = dict(zip(STATIC_NAMES, static_ub))

    if periodic_length_mode == "l6_periodic":
        active_static_names = L6_PERIODIC_ACTIVE_STATIC_NAMES
        periodic_blocks = (("L6", L6_FOURIER_NAMES),)
    elif periodic_length_mode == "l7_periodic":
        active_static_names = L7_PERIODIC_ACTIVE_STATIC_NAMES
        periodic_blocks = (("L7", L7_FOURIER_NAMES),)
    elif periodic_length_mode == "l7_l8_periodic":
        active_static_names = L7_L8_PERIODIC_ACTIVE_STATIC_NAMES
        periodic_blocks = (
            ("L7", L7_FOURIER_NAMES),
            ("L8", L8_FOURIER_NAMES),
        )
    elif periodic_length_mode == "l7_l8_periodic3":
        active_static_names = L7_L8_PERIODIC_ACTIVE_STATIC_NAMES
        periodic_blocks = (
            ("L7", L7_FOURIER3_NAMES),
            ("L8", L8_FOURIER3_NAMES),
        )
    elif periodic_length_mode == "l31_l8_periodic3_l7_fixed":
        active_static_names = L31_L8_PERIODIC_L7_FIXED_ACTIVE_STATIC_NAMES
        periodic_blocks = (
            ("L31", L31_FOURIER3_NAMES),
            ("L8", L8_FOURIER3_NAMES),
        )
    elif periodic_length_mode == "l31_l6_periodic3_l7_l8_fixed":
        active_static_names = L31_L6_PERIODIC_L7_L8_FIXED_ACTIVE_STATIC_NAMES
        periodic_blocks = (
            ("L31", L31_FOURIER3_NAMES),
            ("L6", L6_FOURIER3_NAMES),
        )
    elif periodic_length_mode == "l2_l31_l6_periodic3_l7_l8_fixed":
        active_static_names = L2_L31_L6_PERIODIC_L7_L8_FIXED_ACTIVE_STATIC_NAMES
        periodic_blocks = (
            ("L2", L2_FOURIER3_NAMES),
            ("L31", L31_FOURIER3_NAMES),
            ("L6", L6_FOURIER3_NAMES),
        )
    elif periodic_length_mode == "l5_l8_zc_split_periodic3":
        active_static_names = L5_L8_ZC_SPLIT_PERIODIC_ACTIVE_STATIC_NAMES
        periodic_blocks = (
            ("L5", L5_FOURIER3_NAMES),
            ("L8", L8_FOURIER3_NAMES),
        )
    elif periodic_length_mode == CURRENT_L32_DERIVED_MODE:
        active_static_names = L3_L8_PERIODIC3_L5_FIXED_ACTIVE_STATIC_NAMES
        periodic_blocks = (
            ("L3", L3_FOURIER3_NAMES),
            ("L8", L8_FOURIER3_NAMES),
        )
    elif periodic_length_mode == LEGACY_L32_DERIVED3_MODE:
        active_static_names = L3_L5_L8_PERIODIC3_L32_FIXED_ACTIVE_STATIC_NAMES
        periodic_blocks = (
            ("L3", L3_FOURIER3_NAMES),
            ("L5", L5_FOURIER3_NAMES),
            ("L8", L8_FOURIER3_NAMES),
        )
    elif periodic_length_mode == LEGACY_L32_DERIVED4_MODE:
        active_static_names = L3_L5_L8_PERIODIC3_L32_FIXED_ACTIVE_STATIC_NAMES
        periodic_blocks = (
            ("L3", L3_FOURIER4_NAMES),
            ("L5", L5_FOURIER4_NAMES),
            ("L8", L8_FOURIER4_NAMES),
        )
    elif periodic_length_mode == LEGACY_L32_FIXED_MODE_NAME:
        active_static_names = L3_L5_L8_PERIODIC3_L32_FIXED_ACTIVE_STATIC_NAMES
        periodic_blocks = (
            ("L3", L3_FOURIER3_NAMES),
            ("L5", L5_FOURIER3_NAMES),
            ("L8", L8_FOURIER3_NAMES),
        )
    elif periodic_length_mode == "l3_l5_l8_zc2_periodic_l32_fixed":
        active_static_names = L3_L5_L8_PERIODIC3_L32_FIXED_ACTIVE_STATIC_NAMES
        periodic_blocks = (
            ("L3", L3_FOURIER3_NAMES),
            ("L5", L5_FOURIER3_NAMES),
            ("L8", L8_FOURIER3_NAMES),
            ("L_CZ", ZC_FOURIER_NAMES),
        )
    elif periodic_length_mode == "l5_l6_l8_zc_split_periodic3":
        active_static_names = L5_L6_L8_ZC_SPLIT_PERIODIC_ACTIVE_STATIC_NAMES
        periodic_blocks = (
            ("L5", L5_FOURIER3_NAMES),
            ("L6", L6_FOURIER3_NAMES),
            ("L8", L8_FOURIER3_NAMES),
        )
    elif periodic_length_mode == "l3_l7_periodic":
        active_static_names = L3_L7_PERIODIC_ACTIVE_STATIC_NAMES
        periodic_blocks = (
            ("L3", L3_FOURIER_NAMES),
            ("L7", L7_FOURIER_NAMES),
        )
    elif periodic_length_mode == "l7_periodic_l3_fixed":
        active_static_names = L7_PERIODIC_L3_FIXED_ACTIVE_STATIC_NAMES
        periodic_blocks = (("L7", L7_FOURIER_NAMES),)
    elif periodic_length_mode == "zc_l7_periodic_l3_fixed":
        active_static_names = ZC_L7_PERIODIC_ACTIVE_STATIC_NAMES
        periodic_blocks = (
            ("L7", L7_FOURIER_NAMES),
            ("L_CZ", ZC_FOURIER_NAMES),
        )
    elif periodic_length_mode == "zc_l3_l7_periodic":
        active_static_names = ZC_L3_L7_PERIODIC_ACTIVE_STATIC_NAMES
        periodic_blocks = (
            ("L3", L3_FOURIER_NAMES),
            ("L7", L7_FOURIER_NAMES),
            ("L_CZ", ZC_FOURIER_NAMES),
        )
    elif periodic_length_mode == "zc_l3_l7_periodic3":
        active_static_names = ZC_L3_L7_PERIODIC_ACTIVE_STATIC_NAMES
        periodic_blocks = (
            ("L3", L3_FOURIER3_NAMES),
            ("L7", L7_FOURIER3_NAMES),
            ("L_CZ", ZC_FOURIER3_NAMES),
        )
    elif periodic_length_mode == "zc_l3_l7_periodic4":
        active_static_names = ZC_L3_L7_PERIODIC_ACTIVE_STATIC_NAMES
        periodic_blocks = (
            ("L3", L3_FOURIER4_NAMES),
            ("L7", L7_FOURIER4_NAMES),
            ("L_CZ", ZC_FOURIER4_NAMES),
        )
    elif periodic_length_mode == "zc_l7_l8_periodic4_l3_fixed":
        active_static_names = ZC_L7_L8_PERIODIC_L3_FIXED_ACTIVE_STATIC_NAMES
        periodic_blocks = (
            ("L7", L7_FOURIER4_NAMES),
            ("L8", L8_FOURIER4_NAMES),
            ("L_CZ", ZC_FOURIER4_NAMES),
        )
    elif periodic_length_mode == "zc_l3_l6_l7_periodic":
        active_static_names = ZC_L3_L6_L7_PERIODIC_ACTIVE_STATIC_NAMES
        periodic_blocks = (
            ("L3", L3_FOURIER_NAMES),
            ("L6", L6_FOURIER_NAMES),
            ("L7", L7_FOURIER_NAMES),
            ("L_CZ", ZC_FOURIER_NAMES),
        )
    elif periodic_length_mode == "zc_l3_l7_l8_periodic":
        active_static_names = ZC_L3_L7_L8_PERIODIC_ACTIVE_STATIC_NAMES
        periodic_blocks = (
            ("L3", L3_FOURIER_NAMES),
            ("L7", L7_FOURIER_NAMES),
            ("L8", L8_FOURIER_NAMES),
            ("L_CZ", ZC_FOURIER_NAMES),
        )
    else:
        raise ValueError(f"Unsupported periodic_length_mode: {periodic_length_mode}")

    x0 = [float(static_by_name[name]) for name in active_static_names]
    lb = [float(lb_by_name[name]) for name in active_static_names]
    ub = [float(ub_by_name[name]) for name in active_static_names]

    initial_params = FourBarParams.from_base32(data.base32)
    # B_CenterX 是可优化常数；当前正式模式恢复 RMSE29.99 基准：
    # Bx(t)=B_CenterX，By 使用二阶 Fourier，Bz 使用无常数项的三阶 Fourier。
    if b_curve_mode == "constant_x":
        mot_names = MOT_POLAR_NAMES
    elif b_curve_mode == "fourier_x":
        mot_names = B_FOURIER_X_NAMES
    elif b_curve_mode == "fourier_z3":
        mot_names = B_FOURIER_Z3_NAMES
    elif b_curve_mode == "fourier_z3_c0":
        mot_names = B_FOURIER_Z3_C0_NAMES
    elif b_curve_mode == "fourier_xyz3":
        mot_names = B_FOURIER_XYZ3_NAMES
    else:
        raise ValueError(f"Unsupported B-curve mode: {b_curve_mode}")
    mot0 = np.zeros(len(mot_names), dtype=float)
    mot0[0] = initial_params.CenterY
    mot0[1] = -max(30.0, initial_params.Radius)
    mot0[6] = -max(30.0, initial_params.Radius)
    mot0[9] = initial_params.CenterX
    if b_curve_mode == "fourier_xyz3":
        # 令初始一阶空间椭圆在 XY/YZ/XZ 三个投影中均非退化。
        mot0[13] = 15.0
        mot0[14] = 15.0
    mot_lb = np.array([-50.0, -150.0, -150.0, -60.0, -60.0,
                       -150.0, -150.0, -60.0, -60.0, -80.0])
    mot_ub = np.array([50.0, 150.0, 150.0, 60.0, 60.0,
                       150.0, 150.0, 60.0, 60.0, 80.0])
    if static_bound_mode in ADAPTIVE_BOUND_MODES:
        # B_CenterX 在原搜索中精确贴住 +80 mm，仅沿受阻方向扩展到 +120 mm。
        mot_ub[9] = 120.0
        if static_bound_mode in {
            ADAPTIVE_EXPANDED_V4_BOUND_MODE,
            ADAPTIVE_EXPANDED_V5_BOUND_MODE,
            ADAPTIVE_EXPANDED_V6_BOUND_MODE,
            BROAD_ALL54_BOUND_MODE,
        }:
            # V4 候选再次精确贴住 +120 mm，继续沿同一受阻方向扩展。
            mot_ub[9] = 160.0
            if static_bound_mode == ADAPTIVE_EXPANDED_V6_BOUND_MODE:
                # V5 解的 By 中心与 Bx 中心同时接近边界，仅扩展受阻侧。
                mot_lb[0] = -100.0
                mot_ub[9] = 240.0
            elif static_bound_mode == BROAD_ALL54_BOUND_MODE:
                mot_lb[:10] = np.array(
                    [-130.0, -180.0, -180.0, -80.0, -80.0,
                     -180.0, -180.0, -80.0, -80.0, -120.0],
                    dtype=float,
                )
                mot_ub[:10] = np.array(
                    [80.0, 180.0, 180.0, 80.0, 80.0,
                     180.0, 180.0, 80.0, 80.0, 300.0],
                    dtype=float,
                )
    if b_curve_mode == "fourier_x":
        mot_lb = np.concatenate([mot_lb, [-60.0, -60.0, -30.0, -30.0]])
        mot_ub = np.concatenate([mot_ub, [60.0, 60.0, 30.0, 30.0]])
    elif b_curve_mode == "fourier_z3":
        mot_lb = np.concatenate([mot_lb, [-30.0, -30.0]])
        mot_ub = np.concatenate([mot_ub, [30.0, 30.0]])
    elif b_curve_mode == "fourier_z3_c0":
        if static_bound_mode == BROAD_ALL54_BOUND_MODE:
            mot_lb = np.concatenate([mot_lb, [-60.0, -45.0, -45.0]])
            mot_ub = np.concatenate([mot_ub, [40.0, 45.0, 45.0]])
        else:
            mot_lb = np.concatenate([mot_lb, [-20.0, -30.0, -30.0]])
            mot_ub = np.concatenate([mot_ub, [20.0, 30.0, 30.0]])
    elif b_curve_mode == "fourier_xyz3":
        mot_lb = np.concatenate([
            mot_lb,
            [-20.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0],
        ])
        mot_ub = np.concatenate([
            mot_ub,
            [20.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
        ])
    x0.extend(mot0.tolist())
    lb.extend(mot_lb.tolist())
    ub.extend(mot_ub.tolist())

    periodic_names: Tuple[str, ...] = ()
    for periodic_rod, block_names in periodic_blocks:
        periodic_c0 = float(static_by_name[periodic_rod])
        if static_bound_mode == BROAD_ALL54_BOUND_MODE:
            first_span, second_span = 55.0, 35.0
        elif static_bound_mode in USER_TABLE_BOUND_MODES:
            first_span, second_span = 35.0, 20.0
        else:
            first_span = min(80.0, max(35.0, 0.28 * abs(periodic_c0)))
            second_span = min(65.0, max(25.0, 0.22 * abs(periodic_c0)))
        third_order = len(block_names) >= 7
        fourth_order = len(block_names) >= 9
        periodic0 = np.zeros(len(block_names), dtype=float)
        periodic0[0] = periodic_c0
        c0_lower = float(lb_by_name[periodic_rod])
        c0_upper = float(ub_by_name[periodic_rod])
        if periodic_rod == "L_CZ":
            # ZC(t) 是从 C 沿全局 y 负方向的周期延展长度。短时试算采用
            # 10-150 mm 工程范围，避免固定 150 mm 使 A-Z-D 三角形天然不可装配。
            c0_lower = 10.0
            c0_upper = 150.0
        if periodic_rod == "L6":
            c0_upper = min(420.0, c0_upper)
        harmonic_spans = [first_span, first_span, second_span, second_span]
        if third_order:
            third_span = (
                20.0
                if static_bound_mode == BROAD_ALL54_BOUND_MODE
                else 12.0
                if static_bound_mode in USER_TABLE_BOUND_MODES
                else min(40.0, 0.14 * abs(periodic_c0))
            )
            harmonic_spans.extend([third_span, third_span])
        if fourth_order:
            fourth_span = 8.0 if static_bound_mode in USER_TABLE_BOUND_MODES else min(28.0, 0.10 * abs(periodic_c0))
            harmonic_spans.extend([fourth_span, fourth_span])
        periodic_lb = np.array([c0_lower] + [-span for span in harmonic_spans])
        periodic_ub = np.array([c0_upper] + harmonic_spans)
        periodic_names += block_names
        x0.extend(periodic0.tolist())
        lb.extend(periodic_lb.tolist())
        ub.extend(periodic_ub.tolist())
    if periodic_length_mode in {
        "l5_l8_zc_split_periodic3",
        "l5_l6_l8_zc_split_periodic3",
    }:
        # A=20 mm 给出可见但温和的初始分离；b=c=0 对应标准单峰 raised-cosine。
        periodic_names += ZC_SPLIT_NAMES
        x0.extend([20.0, 0.0, 0.0])
        lb.extend([0.0, -0.95, -0.95])
        ub.extend([150.0, 0.95, 0.95])

    if target_pose_mode == "full":
        target_names = TARGET_POSE_NAMES
        target_initial = TARGET_POSE_INITIAL
        target_lb = TARGET_POSE_LB
        target_ub = TARGET_POSE_UB
    elif target_pose_mode == "rigid6":
        target_names = RIGID6_TARGET_POSE_NAMES
        target_initial = RIGID6_TARGET_POSE_INITIAL
        target_lb = RIGID6_TARGET_POSE_LB
        target_ub = RIGID6_TARGET_POSE_UB
    elif target_pose_mode == "ty_ry":
        # 受限目标位姿只保留 Y 平移和绕 Y 轴旋转；其余分量在解码时固定为零，尺度固定为 1。
        target_names = RESTRICTED_TARGET_POSE_NAMES
        target_initial = RESTRICTED_TARGET_POSE_INITIAL
        target_lb = RESTRICTED_TARGET_POSE_LB
        target_ub = RESTRICTED_TARGET_POSE_UB
    elif target_pose_mode == "ty_ry_rz":
        # 受限位姿只保留 Ty、Ry 和 Rz；Ty 位于 [-200,-100] mm，
        # Ry 限制在 +/-0.8 rad，Rz 限制在 +/-0.5 rad。
        target_names = TY_RY_RZ_TARGET_POSE_NAMES
        target_initial = TY_RY_RZ_TARGET_POSE_INITIAL
        target_lb = TY_RY_RZ_TARGET_POSE_LB
        target_ub = TY_RY_RZ_TARGET_POSE_UB
    elif target_pose_mode == "constrained6":
        # 小范围三轴平移 + Ry/Rz + 尺度；Rx 固定为 0。
        target_names = CONSTRAINED6_TARGET_POSE_NAMES
        target_initial = CONSTRAINED6_TARGET_POSE_INITIAL
        target_lb = CONSTRAINED6_TARGET_POSE_LB
        target_ub = CONSTRAINED6_TARGET_POSE_UB
    elif target_pose_mode == "decoupled_constrained11":
        # 两条目标曲线分别优化 Tx/Ty/Tz/Ry/Rz；Rx 固定为 0，尺度保持公共。
        target_names = DECOUPLED_CONSTRAINED_TARGET_POSE_NAMES
        target_initial = DECOUPLED_CONSTRAINED_TARGET_POSE_INITIAL
        target_lb = DECOUPLED_CONSTRAINED_TARGET_POSE_LB
        target_ub = DECOUPLED_CONSTRAINED_TARGET_POSE_UB
    elif target_pose_mode == "decoupled_fixed_rz_scale8":
        target_names = DECOUPLED_FIXED_RZ_SCALE_TARGET_POSE_NAMES
        target_initial = DECOUPLED_FIXED_RZ_SCALE_TARGET_POSE_INITIAL
        target_lb = DECOUPLED_FIXED_RZ_SCALE_TARGET_POSE_LB
        target_ub = DECOUPLED_FIXED_RZ_SCALE_TARGET_POSE_UB
    elif target_pose_mode == "decoupled_tip_ry_scale8":
        target_names = DECOUPLED_TIP_RY_SCALE_TARGET_POSE_NAMES
        target_initial = DECOUPLED_TIP_RY_SCALE_TARGET_POSE_INITIAL
        target_lb = DECOUPLED_TIP_RY_SCALE_TARGET_POSE_LB
        target_ub = DECOUPLED_TIP_RY_SCALE_TARGET_POSE_UB
    elif target_pose_mode == "decoupled_both_ry_scale9":
        target_names = DECOUPLED_BOTH_RY_SCALE_TARGET_POSE_NAMES
        target_initial = DECOUPLED_BOTH_RY_SCALE_TARGET_POSE_INITIAL
        target_lb = DECOUPLED_BOTH_RY_SCALE_TARGET_POSE_LB
        target_ub = DECOUPLED_BOTH_RY_SCALE_TARGET_POSE_UB
    elif target_pose_mode == "decoupled_both_ry_rz_scale11":
        target_names = DECOUPLED_BOTH_RY_RZ_SCALE_TARGET_POSE_NAMES
        target_initial = DECOUPLED_BOTH_RY_RZ_SCALE_TARGET_POSE_INITIAL
        target_lb = DECOUPLED_BOTH_RY_RZ_SCALE_TARGET_POSE_LB
        target_ub = DECOUPLED_BOTH_RY_RZ_SCALE_TARGET_POSE_UB
    elif target_pose_mode == "decoupled_full_rotation_scale13":
        target_names = DECOUPLED_FULL_ROTATION_SCALE_TARGET_POSE_NAMES
        target_initial = DECOUPLED_FULL_ROTATION_SCALE_TARGET_POSE_INITIAL
        target_lb = DECOUPLED_FULL_ROTATION_SCALE_TARGET_POSE_LB
        target_ub = DECOUPLED_FULL_ROTATION_SCALE_TARGET_POSE_UB
    elif target_pose_mode == "decoupled_fixed_wrist_rotation_scale10":
        target_names = DECOUPLED_FIXED_WRIST_ROTATION_SCALE_TARGET_POSE_NAMES
        target_initial = DECOUPLED_FIXED_WRIST_ROTATION_SCALE_TARGET_POSE_INITIAL
        target_lb = DECOUPLED_FIXED_WRIST_ROTATION_SCALE_TARGET_POSE_LB
        target_ub = DECOUPLED_FIXED_WRIST_ROTATION_SCALE_TARGET_POSE_UB
    elif target_pose_mode == "shared_fixed_rotation_translation3":
        target_names = SHARED_FIXED_ROTATION_TRANSLATION_TARGET_POSE_NAMES
        target_initial = SHARED_FIXED_ROTATION_TRANSLATION_TARGET_POSE_INITIAL
        target_lb = SHARED_FIXED_ROTATION_TRANSLATION_TARGET_POSE_LB
        target_ub = SHARED_FIXED_ROTATION_TRANSLATION_TARGET_POSE_UB
    elif target_pose_mode == "shared_fixed_rotation_scale4":
        target_names = SHARED_FIXED_ROTATION_SCALE_TARGET_POSE_NAMES
        target_initial = SHARED_FIXED_ROTATION_SCALE_TARGET_POSE_INITIAL
        target_lb = SHARED_FIXED_ROTATION_SCALE_TARGET_POSE_LB
        target_ub = SHARED_FIXED_ROTATION_SCALE_TARGET_POSE_UB
    elif target_pose_mode == "shared_limited_rotation_scale7":
        target_names = SHARED_LIMITED_ROTATION_SCALE_TARGET_POSE_NAMES
        target_initial = SHARED_LIMITED_ROTATION_SCALE_TARGET_POSE_INITIAL
        target_lb = SHARED_LIMITED_ROTATION_SCALE_TARGET_POSE_LB
        target_ub = SHARED_LIMITED_ROTATION_SCALE_TARGET_POSE_UB
    elif target_pose_mode == "shared_positive_rz_scale7":
        target_names = SHARED_POSITIVE_RZ_SCALE_TARGET_POSE_NAMES
        target_initial = SHARED_POSITIVE_RZ_SCALE_TARGET_POSE_INITIAL
        target_lb = SHARED_POSITIVE_RZ_SCALE_TARGET_POSE_LB
        target_ub = SHARED_POSITIVE_RZ_SCALE_TARGET_POSE_UB
    elif target_pose_mode == "shared_negative_rz_expanded_scale7":
        target_names = SHARED_NEGATIVE_RZ_EXPANDED_SCALE_TARGET_POSE_NAMES
        target_initial = SHARED_NEGATIVE_RZ_EXPANDED_SCALE_TARGET_POSE_INITIAL
        target_lb = SHARED_NEGATIVE_RZ_EXPANDED_SCALE_TARGET_POSE_LB
        target_ub = SHARED_NEGATIVE_RZ_EXPANDED_SCALE_TARGET_POSE_UB
    elif target_pose_mode == "shared_rotation_translation6":
        target_names = SHARED_ROTATION_TRANSLATION_TARGET_POSE_NAMES
        target_initial = SHARED_ROTATION_TRANSLATION_TARGET_POSE_INITIAL
        target_lb = SHARED_ROTATION_TRANSLATION_TARGET_POSE_LB
        target_ub = SHARED_ROTATION_TRANSLATION_TARGET_POSE_UB
    else:
        raise ValueError(f"Unsupported target_pose_mode: {target_pose_mode}")
    target_initial = np.asarray(target_initial, dtype=float).copy()
    target_lb = np.asarray(target_lb, dtype=float).copy()
    target_ub = np.asarray(target_ub, dtype=float).copy()
    if (
        static_bound_mode == BROAD_ALL54_BOUND_MODE
        and target_pose_mode == "shared_rotation_translation6"
    ):
        target_lb = np.array(
            [-600.0, -100.0, -350.0, -0.25, -0.25, -0.25],
            dtype=float,
        )
        target_ub = np.array(
            [200.0, 200.0, 150.0, 0.25, 0.25, 0.25],
            dtype=float,
        )
    if (
        static_bound_mode in ADAPTIVE_BOUND_MODES
        and target_pose_mode == "decoupled_fixed_wrist_rotation_scale10"
    ):
        target_index = {name: i for i, name in enumerate(target_names)}
        # Tip 两个旋转分量分别精确或近似贴住原下界；只沿受阻方向扩展。
        # 公共尺度按用户复核保持原约束 [0.8, 1.2]，不得向下扩展。
        target_lb[target_index["Target_Tip_Rx_rad"]] = -1.30
        target_lb[target_index["Target_Tip_Ry_rad"]] = -0.85
        target_lb[target_index["Target_Scale"]] = 0.80
        if static_bound_mode == ADAPTIVE_EXPANDED_V3_BOUND_MODE:
            # 完成的四小时基线中 Tip Ty 精确贴住 -200 mm，Wrist Ty 距
            # -100 mm 下界仅 3.8%；尺度保持不变，只扩展这两个受阻平移。
            target_lb[target_index["Target_Tip_Ty_mm"]] = -300.0
            target_lb[target_index["Target_Wrist_Ty_mm"]] = -150.0
        elif static_bound_mode == ADAPTIVE_EXPANDED_V4_BOUND_MODE:
            # 仅扩展本轮仍贴边的 Wrist Ty；Tip Ty 保持原约束。
            target_lb[target_index["Target_Wrist_Ty_mm"]] = -150.0
        elif static_bound_mode in {
            ADAPTIVE_EXPANDED_V5_BOUND_MODE,
            ADAPTIVE_EXPANDED_V6_BOUND_MODE,
        }:
            # 延续 V4 的 Wrist Ty，并沿 V4 新候选受阻方向扩展 Tip Tz 上界。
            target_lb[target_index["Target_Wrist_Ty_mm"]] = -150.0
            target_ub[target_index["Target_Tip_Tz_mm"]] = 50.0
    variable_names = active_static_names + mot_names + periodic_names + target_names
    x0.extend(target_initial.tolist())
    lb.extend(target_lb.tolist())
    ub.extend(target_ub.tolist())
    x0_array = np.asarray(x0, dtype=float)
    lb_array = np.asarray(lb, dtype=float)
    ub_array = np.asarray(ub, dtype=float)
    if x0_array.size != len(variable_names):
        raise RuntimeError(
            f"Design-space definition has {x0_array.size} values but "
            f"{len(variable_names)} variable names."
        )
    return DesignSpace(
        x0=np.clip(x0_array, lb_array, ub_array),
        lb=lb_array,
        ub=ub_array,
        names=variable_names,
        seed_name=(
            f"original_static_parameters_with_direct_B_cam_"
            f"{periodic_length_mode}_{target_pose_mode}_{static_bound_mode}"
        ),
        periodic_length_mode=periodic_length_mode,
        b_curve_mode=b_curve_mode,
        target_pose_mode=target_pose_mode,
    )


def repair_design_vector(
    x: np.ndarray,
    data: ProblemData,
    space: DesignSpace,
) -> np.ndarray:
    """把候选投影回显式可行域，不使用任何历史最优锚点。

    这里只处理可由代数直接确定的约束：B 输入严格凸性、若干静态装配不等式、
    L6(t) 与 L61 的间隙，以及周期杆的长度范围。完整闭环和角度约束仍由
    fourbar 求解器逐帧检查，不能通过修复掩盖。
    """
    vector = np.clip(np.asarray(x, dtype=float).reshape(-1), space.lb, space.ub)
    if vector.size != len(space.names):
        raise ValueError(f"Expected {len(space.names)} variables, got {vector.size}.")
    vector = vector.copy()
    index = {name: position for position, name in enumerate(space.names)}

    def value(name: str) -> float:
        return float(vector[index[name]])

    def assign(name: str, new_value: float) -> None:
        position = index[name]
        vector[position] = np.clip(new_value, space.lb[position], space.ub[position])

    def rod_reference_name(name: str) -> str:
        coefficient_name = f"{name}_C0"
        return coefficient_name if coefficient_name in index else name

    periodic_count = periodic_coefficient_count(space.periodic_length_mode)
    phase = np.asarray(data.phase, dtype=float)
    uses_derived_l32 = (
        space.periodic_length_mode in L32_DERIVED_PERIODIC_MODES
    )

    def candidate_rod_curve(name: str) -> np.ndarray:
        coefficient_name = f"{name}_C0"
        if coefficient_name in index:
            start = index[coefficient_name]
            return (
                periodic_basis(phase, periodic_count)
                @ vector[start:start + periodic_count]
            )
        return np.full(phase.shape, value(name), dtype=float)

    def l32_curve() -> np.ndarray:
        if uses_derived_l32:
            return candidate_rod_curve("L3") - candidate_rod_curve("L31") + 2.0
        return np.full(phase.shape, value("L32"), dtype=float)

    def l32_value() -> float:
        return float(np.min(l32_curve()))

    # L4>=L41：先增大主杆 L4，仍不足时再减小分段长度 L41。
    if value("L4") < value("L41") + 0.01:
        assign("L4", value("L41") + 0.01)
        if value("L4") < value("L41") + 0.01:
            assign("L41", value("L4") - 0.01)

    # L51、L52 均不得超过 L5/3；优先增大 L5，再收缩分段长度。
    l5_reference_name = rod_reference_name("L5")
    required_l5 = 3.0 * max(value("L51"), value("L52")) + 0.03
    if value(l5_reference_name) < required_l5:
        assign(l5_reference_name, required_l5)
    assign(
        "L51",
        min(value("L51"), (value(l5_reference_name) - 0.03) / 3.0),
    )
    assign(
        "L52",
        min(value("L52"), (value(l5_reference_name) - 0.03) / 3.0),
    )

    # L5 的闭环段不得长于 L32(t)。正式模式依次调整 L3 的常数项、L31
    # 和 L5 的常数项；历史兼容模式仍可直接调整独立 L32。
    # 历史兼容模式仍可直接调整独立 L32。最后才降低 L5 常数项。
    proposed_l5 = candidate_rod_curve("L5")
    deficiency = float(
        np.max(proposed_l5 - value("L51") - value("L52") - l32_curve() + 0.01)
    )
    if deficiency > 0.0 and uses_derived_l32:
        l3_reference_name = rod_reference_name("L3")
        old_l3 = value(l3_reference_name)
        assign(l3_reference_name, old_l3 + deficiency)
        deficiency = float(
            np.max(candidate_rod_curve("L5") - value("L51") - value("L52") - l32_curve() + 0.01)
        )
        if deficiency > 0.0:
            old_l31 = value("L31")
            assign("L31", old_l31 - deficiency)
            deficiency = float(
                np.max(candidate_rod_curve("L5") - value("L51") - value("L52") - l32_curve() + 0.01)
            )
    elif deficiency > 0.0:
        old_l32 = value("L32")
        assign("L32", old_l32 + deficiency)
        deficiency -= value("L32") - old_l32
    if deficiency > 0.0:
        old_l5 = value(l5_reference_name)
        assign(l5_reference_name, old_l5 - deficiency)

    # 旧模式中的固定 L6 必须比 L61 长；新模式的逐帧约束在傅里叶投影中处理。
    if "L6" in index:
        required_l6 = value("L61") + 0.06
        if value("L6") < required_l6:
            assign("L6", required_l6)
            if value("L6") < value("L61") + 0.06:
                assign("L61", value("L6") - 0.06)

    mot_start = index[B_FOURIER_NAMES[0]]
    mot_count = (
        len(B_FOURIER_X_NAMES) if space.b_curve_mode == "fourier_x"
        else len(B_FOURIER_Z3_NAMES) if space.b_curve_mode == "fourier_z3"
        else len(B_FOURIER_Z3_C0_NAMES) if space.b_curve_mode == "fourier_z3_c0"
        else len(B_FOURIER_XYZ3_NAMES) if space.b_curve_mode == "fourier_xyz3"
        else len(MOT_POLAR_NAMES)
    )
    mot_end = mot_start + mot_count
    vector[mot_start:mot_end] = project_mot_polar_coeff(vector[mot_start:mot_end])

    # 每个周期杆独立调整 C0 并等比例收缩谐波，保证整个周期满足长度约束。
    periodic_starts = [
        index[name]
        for name in (
            "L2_C0", "L31_C0", "L3_C0", "L6_C0",
            "L5_C0", "L7_C0", "L8_C0", "ZC_C0",
        )
        if name in index
    ]
    for periodic_start in periodic_starts:
        periodic_name = space.names[periodic_start]
        block_count = (
            len(ZC_FOURIER_NAMES)
            if periodic_name == "ZC_C0"
            and space.periodic_length_mode == "l3_l5_l8_zc2_periodic_l32_fixed"
            else periodic_count
        )
        periodic_end = periodic_start + block_count
        periodic_coeff = vector[periodic_start:periodic_end].copy()
        periodic_is_l6 = periodic_name == "L6_C0"
        if periodic_is_l6 and value("L61") + 0.06 > space.ub[periodic_start]:
            assign("L61", space.ub[periodic_start] - 0.06)
        if periodic_is_l6:
            lower_curve = max(float(space.lb[periodic_start]), value("L61") + 0.06)
            upper_curve = float(space.ub[periodic_start])
        elif periodic_name == "L5_C0":
            lower_curve = max(
                float(space.lb[periodic_start]),
                3.0 * max(value("L51"), value("L52")) + 0.03,
            )
            upper_curve = min(
                float(space.ub[periodic_start]),
                float(np.min(l32_curve())) + value("L51") + value("L52") - 0.01,
            )
        elif periodic_name == "L3_C0":
            if "L31_C0" in index:
                l31_start = index["L31_C0"]
                l31_curve_for_triangle = (
                    periodic_basis(phase, periodic_count)
                    @ vector[l31_start:l31_start + periodic_count]
                )
            else:
                l31_curve_for_triangle = np.full(
                    phase.shape, value("L31"), dtype=float
                )
            lower_curve = max(
                float(space.lb[periodic_start]),
                float(np.max(l31_curve_for_triangle)) - 0.99,
            )
            upper_curve = float(space.ub[periodic_start])
        elif (
            "L8_C0" in index
            or "L3_C0" in index
            or "ZC_C0" in index
            or space.periodic_length_mode == "l7_periodic_l3_fixed"
        ):
            # 新模式采用用户给出的静态边界作为周期杆的全周期包络。
            lower_curve = float(space.lb[periodic_start])
            upper_curve = float(space.ub[periodic_start])
        elif periodic_name == "ZC_C0":
            lower_curve = 0.0
            upper_curve = 150.0
        else:
            lower_curve = 10.01
            upper_curve = np.inf
        feasible_c0_lower = max(float(space.lb[periodic_start]), lower_curve)
        feasible_c0_upper = min(float(space.ub[periodic_start]), upper_curve)
        if feasible_c0_lower > feasible_c0_upper + 1e-10:
            raise FourBarError("No feasible periodic-length C0 interval remains.")
        periodic_coeff[0] = np.clip(
            periodic_coeff[0], feasible_c0_lower, feasible_c0_upper
        )
        harmonic_values = (
            periodic_basis(phase, block_count)[:, 1:] @ periodic_coeff[1:]
        )
        harmonic_scale = 1.0
        minimum_harmonic = float(np.min(harmonic_values))
        maximum_harmonic = float(np.max(harmonic_values))
        if minimum_harmonic < 0.0:
            harmonic_scale = min(
                harmonic_scale,
                max(0.0, (periodic_coeff[0] - lower_curve) / (-minimum_harmonic)),
            )
        if np.isfinite(upper_curve) and maximum_harmonic > 0.0:
            harmonic_scale = min(
                harmonic_scale,
                max(0.0, (upper_curve - periodic_coeff[0]) / maximum_harmonic),
            )
        periodic_coeff[1:] *= harmonic_scale
        vector[periodic_start:periodic_end] = np.clip(
            periodic_coeff,
            space.lb[periodic_start:periodic_end],
            space.ub[periodic_start:periodic_end],
        )

    # 新的 A-Z-D 闭环必须逐周期可装配。这里用当前 B/L2 计算 C 的最低
    # Y 基准，再只通过当前候选自身的 L4、L31 和 ZC 系数完成可行化；
    # 不读取任何历史最优点。
    if "ZC_C0" in index:
        b_coeff = vector[mot_start:mot_end]
        candidate_b = decode_mot_curve_in_yz_plane(data.phase, b_coeff)
        b_radius, b_theta01, _b_theta02, _b_norm = bcurve_to_input_angles(
            candidate_b
        )
        if "L2_C0" in index:
            l2_block_count = periodic_count
            l2_start = index["L2_C0"]
            candidate_l2 = (
                periodic_basis(phase, l2_block_count)
                @ vector[l2_start:l2_start + l2_block_count]
            )
        else:
            candidate_l2 = np.full(phase.shape, value("L2"), dtype=float)
        candidate_cy = np.array([
            triangle_third_side_non_adjacent(
                float(candidate_l2[i]),
                float(b_radius[i]),
                float(b_theta01[i]),
            )
            for i in range(phase.size)
        ])
        zc_start = index["ZC_C0"]
        zc_count = (
            len(ZC_FOURIER_NAMES)
            if space.periodic_length_mode
            == "l3_l5_l8_zc2_periodic_l32_fixed"
            else periodic_count
        )
        candidate_zc = (
            periodic_basis(phase, zc_count)
            @ vector[zc_start:zc_start + zc_count]
        )
        candidate_az = float(np.max(candidate_cy)) + candidate_zc
        required_sum = float(np.max(candidate_az)) + 0.10
        for rod_name in ("L4", "L31"):
            if value("L4") + value("L31") >= required_sum:
                break
            deficit = required_sum - value("L4") - value("L31")
            assign(rod_name, value(rod_name) + deficit)
        minimum_az = float(np.min(candidate_az))
        if abs(value("L4") - value("L31")) >= minimum_az - 0.10:
            common = 0.5 * (value("L4") + value("L31"))
            assign("L4", common)
            assign("L31", common)

    if all(name in index for name in ZC_SPLIT_NAMES):
        shape_cos_index = index["ZC_ShapeCos"]
        shape_sin_index = index["ZC_ShapeSin"]
        shape = vector[[shape_cos_index, shape_sin_index]]
        norm = float(np.linalg.norm(shape))
        if norm > 0.98:
            shape *= 0.98 / norm
            vector[shape_cos_index] = shape[0]
            vector[shape_sin_index] = shape[1]

    return np.clip(vector, space.lb, space.ub)


def decode_design_vector(
    x: np.ndarray,
    data: ProblemData,
    space: DesignSpace,
) -> DesignState:
    vector = repair_design_vector(x, data, space)
    if vector.size != len(space.names):
        raise ValueError(f"Expected {len(space.names)} variables, got {vector.size}.")
    cursor = 0
    mot_start = space.names.index(B_FOURIER_NAMES[0])
    active_static_names = space.names[:mot_start]
    active_static = vector[cursor:mot_start]
    cursor = mot_start
    mot_count = (
        len(B_FOURIER_X_NAMES) if space.b_curve_mode == "fourier_x"
        else len(B_FOURIER_Z3_NAMES) if space.b_curve_mode == "fourier_z3"
        else len(B_FOURIER_Z3_C0_NAMES) if space.b_curve_mode == "fourier_z3_c0"
        else len(B_FOURIER_XYZ3_NAMES) if space.b_curve_mode == "fourier_xyz3"
        else len(MOT_POLAR_NAMES)
    )
    mot_coeff = vector[cursor:cursor + mot_count]
    cursor += mot_count
    periodic_coeff_by_rod: Dict[str, np.ndarray] = {}
    periodic_c0_to_rod = {
        "L2_C0": "L2", "L31_C0": "L31", "L3_C0": "L3",
        "L5_C0": "L5", "L6_C0": "L6",
        "L7_C0": "L7", "L8_C0": "L8",
        "ZC_C0": "L_CZ",
    }
    periodic_count = periodic_coefficient_count(space.periodic_length_mode)
    while cursor < len(space.names) and space.names[cursor] in periodic_c0_to_rod:
        periodic_rod = periodic_c0_to_rod[space.names[cursor]]
        block_count = (
            len(ZC_FOURIER_NAMES)
            if periodic_rod == "L_CZ"
            and space.periodic_length_mode == "l3_l5_l8_zc2_periodic_l32_fixed"
            else periodic_count
        )
        periodic_coeff_by_rod[periodic_rod] = vector[
            cursor:cursor + block_count
        ].copy()
        cursor += block_count
    zc_split_parameters = None
    if space.periodic_length_mode in {
        "l5_l8_zc_split_periodic3",
        "l5_l6_l8_zc_split_periodic3",
    }:
        if tuple(space.names[cursor:cursor + len(ZC_SPLIT_NAMES)]) != ZC_SPLIT_NAMES:
            raise ValueError("ZC split parameter block is missing or out of order.")
        zc_split_parameters = vector[
            cursor:cursor + len(ZC_SPLIT_NAMES)
        ].copy()
        cursor += len(ZC_SPLIT_NAMES)
    target_pose = TARGET_POSE_INITIAL.copy()
    target_tip_pose = TARGET_POSE_INITIAL.copy()
    target_wrist_pose = TARGET_POSE_INITIAL.copy()
    if space.target_pose_mode == "decoupled_fixed_wrist_rotation_scale10":
        target_wrist_pose[3:6] = FIXED_WRIST_ROTATION_XYZ_RAD
    elif space.target_pose_mode == "shared_fixed_rotation_translation3":
        target_pose[3:6] = FIXED_COMMON_ROTATION_XYZ_RAD
        target_pose[6] = 1.0
    elif space.target_pose_mode == "shared_fixed_rotation_scale4":
        target_pose[3:6] = FIXED_WRIST_ROTATION_XYZ_RAD
    target_name_to_pose_index = {
        name: index for index, name in enumerate(TARGET_POSE_NAMES)
    }
    decoupled_name_to_pose = {
        "Target_Tip_Tx_mm": ("tip", 0),
        "Target_Tip_Ty_mm": ("tip", 1),
        "Target_Tip_Tz_mm": ("tip", 2),
        "Target_Tip_Rx_rad": ("tip", 3),
        "Target_Tip_Ry_rad": ("tip", 4),
        "Target_Tip_Rz_rad": ("tip", 5),
        "Target_Wrist_Tx_mm": ("wrist", 0),
        "Target_Wrist_Ty_mm": ("wrist", 1),
        "Target_Wrist_Tz_mm": ("wrist", 2),
        "Target_Wrist_Rx_rad": ("wrist", 3),
        "Target_Wrist_Ry_rad": ("wrist", 4),
        "Target_Wrist_Rz_rad": ("wrist", 5),
    }
    decoupled_target_pose = any(
        name in decoupled_name_to_pose for name in space.names[cursor:]
    )
    for name, value in zip(space.names[cursor:], vector[cursor:]):
        if name in target_name_to_pose_index:
            pose_index = target_name_to_pose_index[name]
            target_pose[pose_index] = float(value)
            if name == "Target_Scale":
                target_tip_pose[6] = float(value)
                target_wrist_pose[6] = float(value)
        elif name in decoupled_name_to_pose:
            target_label, pose_index = decoupled_name_to_pose[name]
            selected_pose = target_tip_pose if target_label == "tip" else target_wrist_pose
            selected_pose[pose_index] = float(value)
        else:
            raise ValueError(f"Unknown target-pose variable in design space: {name}")
    if not decoupled_target_pose:
        target_tip_pose = target_pose.copy()
        target_wrist_pose = target_pose.copy()
    else:
        # target_pose 保留为兼容字段；解耦模式下以 Tip 位姿代表公共尺度信息。
        target_pose = target_tip_pose.copy()

    active_map = dict(zip(active_static_names, active_static))
    fixed_map = dict(zip(STATIC_NAMES, base_static_values(data)))
    if space.periodic_length_mode in L32_DERIVED_PERIODIC_MODES:
        l31_value = float(active_map.get("L31", fixed_map["L31"]))
        l3_reference = float(periodic_coeff_by_rod["L3"][0])
        fixed_map["L32"] = l3_reference - l31_value + 2.0
    if space.periodic_length_mode in {
        "l7_l8_periodic", "l7_l8_periodic3",
        "l31_l8_periodic3_l7_fixed",
        "l31_l6_periodic3_l7_l8_fixed",
        "l2_l31_l6_periodic3_l7_l8_fixed",
        "l3_l7_periodic", "l7_periodic_l3_fixed",
        CURRENT_L32_DERIVED_MODE,
        LEGACY_L32_FIXED_MODE_NAME,
    }:
        # L_CZ 仅保留为 base32 兼容槽位；当前几何中 C=Z，不读取该值。
        fixed_map["L_CZ"] = 0.0
    elif space.periodic_length_mode in {
        "l5_l8_zc_split_periodic3",
        "l5_l6_l8_zc_split_periodic3",
    }:
        # L_CZ 仅保留为 base32 兼容槽位；当前模式由三个 ZC 分离参数生成时变距离。
        fixed_map["L_CZ"] = 0.0
    full_static = np.array([
        active_map[name]
        if name in active_map
        else float(periodic_coeff_by_rod[name][0])
        if name in periodic_coeff_by_rod
        else fixed_map[name]
        for name in STATIC_NAMES
    ], dtype=float)
    params = params_from_static(full_static, data)
    origin = direct_b_phase_origin(mot_coeff)
    canonical_mot_coeff = canonicalize_direct_b_phase(mot_coeff, origin)
    b_curve = decode_mot_curve_in_yz_plane(
        data.phase,
        canonical_mot_coeff,
    )
    # B 的内部相位平移只用于把第一帧对齐到向下穿越 Bz=0 的位置。
    # L7/L8/ZC 的 Fourier 系数以该第一帧为共同时间原点，因此必须按固定
    # 时间步 data.phase 计算，不能再次叠加 B 系数自身的内部相位偏移。
    periodic_phase = np.asarray(data.phase, dtype=float)
    def constant_periodic(value: float) -> np.ndarray:
        coefficient = np.zeros(periodic_count, dtype=float)
        coefficient[0] = value
        return coefficient

    coefficient_defaults = {
        "L2": constant_periodic(params.L2),
        "L31": constant_periodic(params.L31),
        "L3": constant_periodic(params.L3),
        "L5": constant_periodic(params.L5),
        "L6": constant_periodic(params.L6),
        "L7": constant_periodic(params.L7),
        "L8": constant_periodic(params.L8),
        "L_CZ": constant_periodic(0.0),
    }
    l2_coeff = periodic_coeff_by_rod.get("L2", coefficient_defaults["L2"]).copy()
    l31_coeff = periodic_coeff_by_rod.get("L31", coefficient_defaults["L31"]).copy()
    l3_coeff = periodic_coeff_by_rod.get("L3", coefficient_defaults["L3"]).copy()
    l5_coeff = periodic_coeff_by_rod.get("L5", coefficient_defaults["L5"]).copy()
    l6_coeff = periodic_coeff_by_rod.get("L6", coefficient_defaults["L6"]).copy()
    l7_coeff = periodic_coeff_by_rod.get("L7", coefficient_defaults["L7"]).copy()
    l8_coeff = periodic_coeff_by_rod.get("L8", coefficient_defaults["L8"]).copy()
    zc_coeff = periodic_coeff_by_rod.get("L_CZ", coefficient_defaults["L_CZ"]).copy()
    l2_values = periodic_basis(periodic_phase, l2_coeff.size) @ l2_coeff
    l31_values = periodic_basis(periodic_phase, l31_coeff.size) @ l31_coeff
    l3_values = periodic_basis(periodic_phase, l3_coeff.size) @ l3_coeff
    l32_values = l3_values - l31_values + 2.0
    params = replace(params, L32=float(l32_values[0]))
    l5_values = periodic_basis(periodic_phase, l5_coeff.size) @ l5_coeff
    l6_values = periodic_basis(periodic_phase, l6_coeff.size) @ l6_coeff
    l7_values = periodic_basis(periodic_phase, l7_coeff.size) @ l7_coeff
    l8_values = periodic_basis(periodic_phase, l8_coeff.size) @ l8_coeff
    if zc_split_parameters is not None:
        zc_values, zc_coeff = zc_split_profile(
            periodic_phase.size,
            float(zc_split_parameters[0]),
            float(zc_split_parameters[1]),
            float(zc_split_parameters[2]),
        )
    elif "L_CZ" in periodic_coeff_by_rod:
        zc_values = periodic_basis(periodic_phase, zc_coeff.size) @ zc_coeff
    else:
        zc_values = np.zeros_like(periodic_phase, dtype=float)
    if decoupled_target_pose:
        tip_center = np.mean(data.target_tip, axis=0)
        wrist_center = np.mean(data.target_wrist, axis=0)
    else:
        tip_center = target_pose_center(data)
        wrist_center = tip_center
    canonical_vector = vector.copy()
    canonical_vector[mot_start:mot_start + mot_count] = canonical_mot_coeff
    return DesignState(
        static=full_static,
        # mot_curve 是旧检查点字段名；新模型中与 b_curve 相同且不代表物理 Mot。
        mot_curve=b_curve.copy(),
        b_curve=b_curve,
        l6_values=l6_values,
        l7_values=l7_values,
        target_tip=apply_target_pose(data.target_tip, target_tip_pose, tip_center),
        target_wrist=apply_target_pose(data.target_wrist, target_wrist_pose, wrist_center),
        target_pose=target_pose.copy(),
        b_fourier_coeff=canonical_mot_coeff.copy(),
        l6_fourier_coeff=l6_coeff,
        l7_fourier_coeff=l7_coeff.copy(),
        target_tip_pose=target_tip_pose.copy(),
        target_wrist_pose=target_wrist_pose.copy(),
        variable_names=space.names,
        x=canonical_vector,
        lb=space.lb.copy(),
        ub=space.ub.copy(),
        l2_values=l2_values,
        l2_fourier_coeff=l2_coeff.copy(),
        l5_values=l5_values,
        l5_fourier_coeff=l5_coeff.copy(),
        l8_values=l8_values,
        l8_fourier_coeff=l8_coeff.copy(),
        l31_values=l31_values,
        l31_fourier_coeff=l31_coeff.copy(),
        l32_values=l32_values,
        l3_values=l3_values,
        l3_fourier_coeff=l3_coeff.copy(),
        zc_values=zc_values,
        zc_fourier_coeff=zc_coeff.copy(),
        zc_split_parameters=(
            None if zc_split_parameters is None else zc_split_parameters.copy()
        ),
    )


def load_checkpoint_state(path: Path | str) -> DesignState:
    """读取历史检查点；该接口用于模型复现、区域分析和回归验证。"""
    with np.load(Path(path), allow_pickle=True) as saved:
        names = tuple(str(value) for value in np.asarray(saved.get("variable_names", []), dtype=object).reshape(-1))
        target_pose = np.asarray(saved.get("target_pose", TARGET_POSE_INITIAL), dtype=float).reshape(-1)
        target_tip_pose = np.asarray(saved.get("target_tip_pose", target_pose), dtype=float).reshape(-1)
        target_wrist_pose = np.asarray(saved.get("target_wrist_pose", target_pose), dtype=float).reshape(-1)
        stored_b_curve = np.asarray(saved["b_curve"], dtype=float)
        stored_mot_curve = np.asarray(saved.get("mot_curve", stored_b_curve), dtype=float)
        raw_static = np.asarray(saved["static"], dtype=float).reshape(-1)
        stored_l6_values = np.asarray(saved.get("l6_values", []), dtype=float).reshape(-1)
        if stored_l6_values.size:
            fixed_l6 = float(np.mean(stored_l6_values))
        else:
            legacy_l6_coeff = np.asarray(saved.get("l6_fourier_coeff", []), dtype=float).reshape(-1)
            fixed_l6 = float(legacy_l6_coeff[0]) if legacy_l6_coeff.size else FourBarParams().L6

        if raw_static.size == len(LEGACY_STATIC_NAMES):
            if "input_radius" in saved:
                legacy_l1 = float(np.asarray(saved["input_radius"], dtype=float).reshape(-1)[0])
            else:
                legacy_l1 = float(np.linalg.norm(stored_b_curve[0]))
            static_map = dict(zip(LEGACY_STATIC_NAMES, raw_static))
            static_map["L1"] = legacy_l1
            static_map["L6"] = fixed_l6
            static_map["L13"] = float(np.clip(1.2 * static_map["L11"], 5.0, 25.0))
            static_map["theta18_deg"] = FourBarParams().theta18_deg
            static_map["L32"] = static_map["L3"] - static_map["L31"] + 2.0
            static = np.array([static_map[name] for name in STATIC_NAMES], dtype=float)
        elif raw_static.size == len(PREVIOUS_STATIC_NAMES):
            static_map = dict(zip(PREVIOUS_STATIC_NAMES, raw_static))
            static_map["L6"] = fixed_l6
            static_map["L13"] = float(np.clip(1.2 * static_map["L11"], 5.0, 25.0))
            static_map["theta18_deg"] = FourBarParams().theta18_deg
            static_map["L32"] = static_map["L3"] - static_map["L31"] + 2.0
            static = np.array([static_map[name] for name in STATIC_NAMES], dtype=float)
        elif raw_static.size == len(PRE_THETA18_STATIC_NAMES):
            static_map = dict(zip(PRE_THETA18_STATIC_NAMES, raw_static))
            fixed_l6 = float(static_map["L6"])
            static_map["L13"] = float(np.clip(1.2 * static_map["L11"], 5.0, 25.0))
            static_map["theta18_deg"] = FourBarParams().theta18_deg
            static_map["L32"] = static_map["L3"] - static_map["L31"] + 2.0
            static = np.array([static_map[name] for name in STATIC_NAMES], dtype=float)
        elif raw_static.size == len(PRE_L13_STATIC_NAMES):
            static_map = dict(zip(PRE_L13_STATIC_NAMES, raw_static))
            fixed_l6 = float(static_map["L6"])
            static_map["L13"] = float(np.clip(1.2 * static_map["L11"], 5.0, 25.0))
            static_map["L32"] = static_map["L3"] - static_map["L31"] + 2.0
            static = np.array([static_map[name] for name in STATIC_NAMES], dtype=float)
        elif raw_static.size == len(PRE_L32_STATIC_NAMES):
            static_map = dict(zip(PRE_L32_STATIC_NAMES, raw_static))
            fixed_l6 = float(static_map["L6"])
            # 仅在读取旧检查点时构造迁移初值；当前模型不再使用该关系。
            static_map["L32"] = static_map["L3"] - static_map["L31"] + 2.0
            static = np.array([static_map[name] for name in STATIC_NAMES], dtype=float)
        elif raw_static.size == len(STATIC_NAMES):
            static = raw_static.copy()
            fixed_l6 = float(static[STATIC_NAMES.index("L6")])
        else:
            raise ValueError(
                f"Checkpoint has {raw_static.size} static parameters; expected "
                f"{len(LEGACY_STATIC_NAMES)}, {len(PREVIOUS_STATIC_NAMES)}, "
                f"{len(PRE_THETA18_STATIC_NAMES)}, {len(PRE_L13_STATIC_NAMES)}, "
                f"{len(PRE_L32_STATIC_NAMES)}, "
                f"or {len(STATIC_NAMES)}."
            )
        l2_default = float(static[STATIC_NAMES.index("L2")])
        l3_default = float(static[STATIC_NAMES.index("L3")])
        l5_default = float(static[STATIC_NAMES.index("L5")])
        l31_default = float(static[STATIC_NAMES.index("L31")])
        l32_default = float(static[STATIC_NAMES.index("L32")])
        l7_default = float(static[STATIC_NAMES.index("L7")])
        l8_default = float(static[STATIC_NAMES.index("L8")])
        zc_default = float(static[STATIC_NAMES.index("L_CZ")])
        phase_count = int(stored_mot_curve.shape[0])
        _input_radius, _theta01, _theta02, current_b_curve, _mot_radius = (
            motcurve_to_b_input(
                stored_mot_curve,
                float(static[STATIC_NAMES.index("L1")]),
            )
        )
        mot_coeff = np.asarray(
            saved.get("mot_fourier_coeff", saved.get("b_fourier_coeff", np.zeros(len(MOT_POLAR_NAMES)))),
            dtype=float,
        ).reshape(-1)
        saved_x = np.asarray(saved.get("x", []), dtype=float).reshape(-1)
        saved_lb = np.asarray(saved.get("variable_lb", []), dtype=float).reshape(-1)
        saved_ub = np.asarray(saved.get("variable_ub", []), dtype=float).reshape(-1)
        if names and saved_x.size == len(names) and all(name in names for name in FULL_VARIABLE_NAMES):
            index_by_name = {name: index for index, name in enumerate(names)}
            keep = np.array([index_by_name[name] for name in FULL_VARIABLE_NAMES], dtype=int)
            saved_x = saved_x[keep]
            if saved_lb.size == len(names):
                saved_lb = saved_lb[keep]
            if saved_ub.size == len(names):
                saved_ub = saved_ub[keep]
            names = FULL_VARIABLE_NAMES

        stored_l6_coeff = np.asarray(
            saved.get(
                "l6_fourier_coeff",
                fit_fourier5(
                    np.arange(phase_count) / phase_count,
                    stored_l6_values
                    if stored_l6_values.size == phase_count
                    else np.full(phase_count, fixed_l6),
                ),
            ),
            dtype=float,
        ).reshape(-1)
        # 新版检查点中的 L6(t) 是优化变量，读取时必须保留其逐时间步数值和
        # 傅里叶系数；旧版固定 L6 检查点仍自然退化为常数曲线。
        restored_l6_values = (
            stored_l6_values.copy()
            if stored_l6_values.size == phase_count
            else np.full(phase_count, fixed_l6, dtype=float)
        )
        if stored_l6_coeff.size != 5:
            stored_l6_coeff = fit_fourier5(
                np.arange(phase_count) / phase_count,
                restored_l6_values,
            )

        return DesignState(
            static=static,
            mot_curve=stored_mot_curve,
            b_curve=current_b_curve,
            l6_values=restored_l6_values,
            l7_values=np.asarray(saved.get("l7_values", np.full(phase_count, l7_default)), dtype=float).reshape(-1),
            target_tip=np.asarray(saved["target_tip"], dtype=float),
            target_wrist=np.asarray(saved["target_wrist"], dtype=float),
            target_pose=target_pose,
            b_fourier_coeff=mot_coeff,
            l6_fourier_coeff=stored_l6_coeff,
            l7_fourier_coeff=np.asarray(saved.get("l7_fourier_coeff", fit_fourier5(np.arange(phase_count) / phase_count, saved.get("l7_values", np.full(phase_count, l7_default)))), dtype=float).reshape(-1),
            target_tip_pose=target_tip_pose,
            target_wrist_pose=target_wrist_pose,
            variable_names=names,
            x=saved_x,
            lb=saved_lb,
            ub=saved_ub,
            l2_values=np.asarray(
                saved.get("l2_values", np.full(phase_count, l2_default)), dtype=float
            ).reshape(-1),
            l2_fourier_coeff=np.asarray(
                saved.get(
                    "l2_fourier_coeff",
                    fit_fourier5(
                        np.arange(phase_count) / phase_count,
                        saved.get("l2_values", np.full(phase_count, l2_default)),
                    ),
                ),
                dtype=float,
            ).reshape(-1),
            l5_values=np.asarray(
                saved.get("l5_values", np.full(phase_count, l5_default)), dtype=float
            ).reshape(-1),
            l5_fourier_coeff=np.asarray(
                saved.get(
                    "l5_fourier_coeff",
                    fit_fourier5(
                        np.arange(phase_count) / phase_count,
                        saved.get("l5_values", np.full(phase_count, l5_default)),
                    ),
                ),
                dtype=float,
            ).reshape(-1),
            l8_values=np.asarray(
                saved.get("l8_values", np.full(phase_count, l8_default)), dtype=float
            ).reshape(-1),
            l8_fourier_coeff=np.asarray(
                saved.get(
                    "l8_fourier_coeff",
                    fit_fourier5(
                        np.arange(phase_count) / phase_count,
                        saved.get("l8_values", np.full(phase_count, l8_default)),
                    ),
                ),
                dtype=float,
            ).reshape(-1),
            l31_values=np.asarray(
                saved.get("l31_values", np.full(phase_count, l31_default)), dtype=float
            ).reshape(-1),
            l31_fourier_coeff=np.asarray(
                saved.get(
                    "l31_fourier_coeff",
                    fit_fourier5(
                        np.arange(phase_count) / phase_count,
                        saved.get("l31_values", np.full(phase_count, l31_default)),
                    ),
                ),
                dtype=float,
            ).reshape(-1),
            l32_values=np.asarray(
                saved.get("l32_values", np.full(phase_count, l32_default)), dtype=float
            ).reshape(-1),
            l3_values=np.asarray(
                saved.get("l3_values", np.full(phase_count, l3_default)), dtype=float
            ).reshape(-1),
            l3_fourier_coeff=np.asarray(
                saved.get(
                    "l3_fourier_coeff",
                    fit_fourier5(
                        np.arange(phase_count) / phase_count,
                        saved.get("l3_values", np.full(phase_count, l3_default)),
                    ),
                ),
                dtype=float,
            ).reshape(-1),
            zc_values=np.asarray(
                saved.get("zc_values", np.full(phase_count, zc_default)), dtype=float
            ).reshape(-1),
            zc_fourier_coeff=np.asarray(
                saved.get(
                    "zc_fourier_coeff",
                    fit_fourier5(
                        np.arange(phase_count) / phase_count,
                        saved.get("zc_values", np.full(phase_count, zc_default)),
                    ),
                ),
                dtype=float,
            ).reshape(-1),
            zc_split_parameters=np.asarray(
                saved.get("zc_split_parameters", []), dtype=float
            ).reshape(-1),
        )


def evaluate_design_state(
    state: DesignState,
    data: ProblemData,
    check_smooth: bool = False,
    fixed_moving_lengths: bool = False,
    raise_errors: bool = False,
) -> Tuple[FourBarResult | None, Dict[str, float]]:
    """计算机构。

    当前正式状态使用固定 L2/L5/L6/L7/L31、逐帧派生 L32(t)=L3(t)-L31+2、
    三阶周期 L3/L8；保留 Z 节点但强制 Z=C，且不存在 ZC 分离自由度。
    ``fixed_moving_lengths=True`` 仅用于灵敏度/窄区域分析。
    """
    try:
        static = np.asarray(state.static, dtype=float).copy()
        l2_values = (
            np.asarray(state.l2_values, dtype=float).reshape(-1)
            if state.l2_values is not None else np.zeros(0, dtype=float)
        )
        l31_values = (
            np.asarray(state.l31_values, dtype=float).reshape(-1)
            if state.l31_values is not None else np.zeros(0, dtype=float)
        )
        l32_values = (
            np.asarray(state.l32_values, dtype=float).reshape(-1)
            if state.l32_values is not None else np.zeros(0, dtype=float)
        )
        l3_values = (
            np.asarray(state.l3_values, dtype=float).reshape(-1)
            if state.l3_values is not None else np.zeros(0, dtype=float)
        )
        l5_values = (
            np.asarray(state.l5_values, dtype=float).reshape(-1)
            if state.l5_values is not None else np.zeros(0, dtype=float)
        )
        l6_values = np.asarray(state.l6_values, dtype=float).reshape(-1)
        l7_values = np.asarray(state.l7_values, dtype=float).reshape(-1)
        l8_values = (
            np.asarray(state.l8_values, dtype=float).reshape(-1)
            if state.l8_values is not None else np.zeros(0, dtype=float)
        )
        zc_values = (
            np.asarray(state.zc_values, dtype=float).reshape(-1)
            if state.zc_values is not None else np.zeros(0, dtype=float)
        )
        l32_derived_periodic_mode = (
            state.variable_names
            and "L3_C0" in state.variable_names
            and "L8_C0" in state.variable_names
            and "L32" not in state.variable_names
            and "ZC_Amplitude_mm" not in state.variable_names
        )
        if l32_derived_periodic_mode:
            if l31_values.size == 0:
                l31_values = np.full(
                    np.asarray(data.phase).shape,
                    float(static[STATIC_NAMES.index("L31")]),
                    dtype=float,
                )
            if l3_values.size == 0:
                l3_values = np.full(
                    np.asarray(data.phase).shape,
                    float(static[STATIC_NAMES.index("L3")]),
                    dtype=float,
                )
            l32_values = l3_values - l31_values + 2.0
            static[STATIC_NAMES.index("L32")] = float(l32_values[0])
        if l32_derived_periodic_mode:
            fixed_curves = [
                ("L2", l2_values),
                ("L31", l31_values),
                ("L6", l6_values),
                ("L7", l7_values),
            ]
            if "L5_C0" not in state.variable_names:
                fixed_curves.append(("L5", l5_values))
            for fixed_name, fixed_curve in fixed_curves:
                if fixed_curve.size and float(np.ptp(fixed_curve)) > 1e-8:
                    raise FourBarError(
                        f"{fixed_name} must be constant in the selected L5/L8 split mode."
                    )
        # B 曲线约束由其参数化模式决定；当前空间模式允许自交，但限制 X 行程。
        if np.asarray(state.b_fourier_coeff).size in (
            len(MOT_POLAR_NAMES), len(B_FOURIER_X_NAMES), len(B_FOURIER_Z3_NAMES),
            len(B_FOURIER_Z3_C0_NAMES),
            len(B_FOURIER_XYZ3_NAMES),
        ):
            validate_mot_polar_coeff(state.b_fourier_coeff)
        if fixed_moving_lengths:
            if l2_values.size:
                l2_values = np.full(76, float(np.mean(l2_values)))
            if l31_values.size:
                l31_values = np.full(76, float(np.mean(l31_values)))
            if l3_values.size:
                l3_values = np.full(76, float(np.mean(l3_values)))
            if l5_values.size:
                l5_values = np.full(76, float(np.mean(l5_values)))
            l6_values = np.full(76, float(np.mean(l6_values)))
            l7_values = np.full(76, float(np.mean(l7_values)))
            if l8_values.size:
                l8_values = np.full(76, float(np.mean(l8_values)))
            if zc_values.size:
                # 冻结分离执行器时恢复 C=Z；正的常量距离不满足“分离后重合”。
                zc_values = np.full(76, float(np.mean(zc_values)))
            if l31_values.size:
                static[STATIC_NAMES.index("L31")] = float(l31_values[0])
            if l2_values.size:
                static[STATIC_NAMES.index("L2")] = float(l2_values[0])
            if l3_values.size:
                static[STATIC_NAMES.index("L3")] = float(l3_values[0])
            if l5_values.size:
                static[STATIC_NAMES.index("L5")] = float(l5_values[0])
            static[STATIC_NAMES.index("L6")] = float(l6_values[0])
            static[STATIC_NAMES.index("L7")] = float(l7_values[0])
            if l8_values.size:
                static[STATIC_NAMES.index("L8")] = float(l8_values[0])
            if zc_values.size:
                static[STATIC_NAMES.index("L_CZ")] = float(zc_values[0])
        if l32_derived_periodic_mode:
            l32_values = l3_values - l31_values + 2.0
            static[STATIC_NAMES.index("L32")] = float(l32_values[0])
        params = params_from_static(static, data)
        if l2_values.size == 0:
            l2_values = np.full(76, params.L2, dtype=float)
        if l31_values.size == 0:
            l31_values = np.full(76, params.L31, dtype=float)
        if l3_values.size == 0:
            l3_values = np.full(76, params.L3, dtype=float)
        if l32_derived_periodic_mode:
            l32_values = l3_values - l31_values + 2.0
            params = replace(params, L32=float(l32_values[0]))
        elif l32_values.size == 0:
            l32_values = np.full(76, params.L32, dtype=float)
        if l5_values.size == 0:
            l5_values = np.full(76, params.L5, dtype=float)
        if l6_values.size == 0:
            l6_values = np.full(76, params.L6, dtype=float)
        if l7_values.size == 0:
            l7_values = np.full(76, params.L7, dtype=float)
        if l8_values.size == 0:
            l8_values = np.full(76, params.L8, dtype=float)
        if zc_values.size and float(np.max(np.abs(zc_values))) > 1e-10:
            raise FourBarError("CZ separation is disabled; decoded legacy ZC(t) must be zero")
        zc_values = np.zeros(76, dtype=float)
        l6_lower_bound = -np.inf
        l6_upper_bound = 420.0
        if state.variable_names and "L6_C0" in state.variable_names:
            l6_index = state.variable_names.index("L6_C0")
            if state.lb is not None and len(state.lb) > l6_index:
                l6_lower_bound = float(state.lb[l6_index])
            if state.ub is not None and len(state.ub) > l6_index:
                l6_upper_bound = float(state.ub[l6_index])
        if float(np.min(l6_values)) < l6_lower_bound - 1e-8:
            raise FourBarError(f"require L6(t) >= {l6_lower_bound:g} mm")
        if float(np.min(l6_values)) <= params.L61 + 0.05:
            raise FourBarError("require L6(t) > L61 + 0.05 mm")
        if float(np.max(l6_values)) > l6_upper_bound + 1e-8:
            raise FourBarError(f"require L6(t) <= {l6_upper_bound:g} mm")
        # 周期杆的常数项边界同时作为完整周期包络；固定杆只检查正长度。
        for rod_name, curve in (
            ("L2", l2_values),
            ("L31", l31_values),
            ("L3", l3_values),
            ("L5", l5_values),
            ("L7", l7_values),
            ("L8", l8_values),
        ):
            coeff_name = f"{rod_name}_C0"
            if state.variable_names and coeff_name in state.variable_names:
                coeff_index = state.variable_names.index(coeff_name)
                lower = float(state.lb[coeff_index]) if state.lb is not None else 10.0
                upper = float(state.ub[coeff_index]) if state.ub is not None else np.inf
                if float(np.min(curve)) < lower - 1e-8:
                    raise FourBarError(f"require {rod_name}(t) >= {lower:g} mm")
                if float(np.max(curve)) > upper + 1e-8:
                    raise FourBarError(f"require {rod_name}(t) <= {upper:g} mm")
        if float(np.min(l3_values)) < 10.0:
            raise FourBarError("require L3 >= 10 mm")
        if float(np.min(l7_values)) < 10.0:
            raise FourBarError("require L7 >= 10 mm")
        if float(np.min(l8_values)) < 10.0:
            raise FourBarError("require L8 >= 10 mm")
        time_varying_lengths = {
            "L2": l2_values,
            "L31": l31_values,
            "L3": l3_values,
            "L5": l5_values,
            "L6": l6_values,
            "L7": l7_values,
            "L8": l8_values,
        }
        result = fourbar_direct_b_l6(
            params,
            state.b_curve,
            l6_values,
            wrist_node="L",
            check_smooth=check_smooth,
            time_varying_lengths=time_varying_lengths,
        )
        if not result.valid:
            raise FourBarError(result.message)
        metrics = combined_rmse(
            result.tip, result.wrist, state.target_tip, state.target_wrist
        )
        metrics.update(combined_stutter_metrics(result.tip, result.wrist))
        return result, metrics
    except Exception:
        if raise_errors:
            raise
        return None, {
            "combined": 1e6, "tip": 1e6, "wrist": 1e6,
            "tip_peak": 1e6, "wrist_peak": 1e6,
            "strict_correspondence": 1.0,
            "stutter_index": 1e6, "stutter_penalty_mm": 1e6,
        }


def evaluate_design_vector(
    x: np.ndarray,
    data: ProblemData,
    space: DesignSpace,
    check_smooth: bool = False,
) -> Tuple[FourBarResult | None, Dict[str, float], DesignState]:
    state = decode_design_vector(x, data, space)
    result, metrics = evaluate_design_state(state, data, check_smooth=check_smooth)
    return result, metrics, state


def objective_from_metrics(metrics: Mapping[str, float]) -> float:
    """严格等弧长 RMSE、峰值误差和原始时间轨迹卡顿惩罚。"""
    return float(
        metrics["combined"]
        + 0.015 * max(metrics["tip_peak"], metrics["wrist_peak"])
        + float(metrics.get("stutter_penalty_mm", 0.0))
    )


def coordinate_error_metrics(
    tip: np.ndarray,
    wrist: np.ndarray,
    target_tip: np.ndarray,
    target_wrist: np.ndarray,
) -> Dict[str, float]:
    """区域分析使用的固定初始化等弧长同索引三坐标误差指标。"""
    output = combined_rmse(tip, wrist, target_tip, target_wrist)
    output.update(combined_stutter_metrics(tip, wrist))
    tip_residual, wrist_residual, _, _ = strict_initialized_paired_arclength_residuals(
        tip, wrist, target_tip, target_wrist
    )
    for label, residual in (
        ("tip", tip_residual), ("wrist", wrist_residual)
    ):
        error = np.abs(residual)
        for axis, column in zip(("x", "y", "z"), range(3)):
            output[f"{label}_{axis}_rmse"] = float(np.sqrt(np.mean(error[:, column] ** 2)))
            output[f"{label}_{axis}_peak"] = float(np.max(error[:, column]))
    output["objective"] = objective_from_metrics(output)
    return output


def checkpoint_metrics(path: Path | str, data: ProblemData | None = None) -> Dict[str, Any]:
    """从一个检查点重新计算轨迹和 RMSE，便于做模型回归验证。"""
    problem = data or load_problem_data()
    state = load_checkpoint_state(path)
    result, metrics = evaluate_design_state(state, problem, check_smooth=False)
    if result is None:
        return {"valid": False, **metrics}
    return {
        "valid": True,
        **metrics,
        "generated_tip": result.tip,
        "generated_wrist": result.wrist,
        "nodes": result.nodes,
    }


def plot_initial_three_views(
    result: FourBarResult,
    frame: int = 0,
    output_dir: Path | str = CURRENT_DIR / "output",
    prefix: str = "fourbar_initial_three_view",
    show: bool = True,
    view_nodes: Tuple[str, ...] | None = None,
    mechanism_links: Tuple[Tuple[str, str], ...] | None = None,
    construction_links: Tuple[Tuple[str, str], ...] | None = None,
    extension_links: Tuple[Tuple[str, str], ...] | None = None,
    include_input: bool = False,
    figure_title: str | None = None,
) -> Dict[str, str]:
    """Export front, top, and side orthographic views of one mechanism frame."""
    import matplotlib as mpl
    if not show:
        mpl.use("Agg")
    import matplotlib.pyplot as plt

    if not result.valid or result.nodes.ndim != 3 or result.nodes.shape[0] == 0:
        raise FourBarError("a valid FourBarResult is required for the three-view plot")
    if not 0 <= frame < result.nodes.shape[0]:
        raise IndexError(f"frame must be in [0, {result.nodes.shape[0] - 1}], got {frame}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    png_path = destination / f"{prefix}.png"
    svg_path = destination / f"{prefix}.svg"
    pdf_path = destination / f"{prefix}.pdf"
    csv_path = destination / f"{prefix}_nodes.csv"

    nodes = np.asarray(result.nodes[frame], dtype=float)
    active_names = (
        tuple(name for name in result.node_names if name not in {"R", "V", "X", "Y"})
        if view_nodes is None else tuple(view_nodes)
    )
    if not active_names or any(name not in IDX for name in active_names):
        raise ValueError("view_nodes must contain known mechanism node names")
    active_mechanism_links = MECHANISM_LINKS if mechanism_links is None else mechanism_links
    active_construction_links = (
        CONSTRUCTION_LINKS if construction_links is None else construction_links
    )
    active_extension_links = EXTENSION_LINKS if extension_links is None else extension_links
    active_indices = np.array([IDX[name] for name in active_names], dtype=int)
    display_points = nodes[active_indices]
    center = 0.5 * (np.min(display_points, axis=0) + np.max(display_points, axis=0))
    common_span = max(float(np.max(np.ptp(display_points, axis=0))) * 1.12, 1.0)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.5,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    projections = (
        (0, 1, "Front view", "x (mm)", "y (mm)"),
        (0, 2, "Top view", "x (mm)", "z (mm)"),
        (1, 2, "Side view", "y (mm)", "z (mm)"),
    )
    label_groups_by_panel: list[dict[tuple[float, float], list[str]]] = [
        {}, {}, {},
    ]
    active_points = nodes[active_indices]
    for local_index, name in enumerate(active_names):
        panel_scores = []
        for horizontal, vertical, *_ in projections:
            projected = active_points[:, [horizontal, vertical]]
            distances = np.linalg.norm(projected - projected[local_index], axis=1)
            positive = np.sort(distances[distances > 1e-9])
            if positive.size == 0:
                panel_scores.append(0.0)
            else:
                nearest = float(positive[0])
                neighborhood = float(np.mean(positive[:min(3, positive.size)]))
                panel_scores.append(nearest + 0.20 * neighborhood)
        selected_panel = int(np.argmax(panel_scores))
        horizontal, vertical, *_ = projections[selected_panel]
        point = nodes[IDX[name]]
        key = (round(float(point[horizontal]), 6), round(float(point[vertical]), 6))
        label_groups_by_panel[selected_panel].setdefault(key, []).append(name)
    colors = {
        "mechanism": "#3F4852",
        "construction": "#CC79A7",
        "extension": "#26718C",
        "input": "#C46A32",
        "node": "#20252A",
        "wrist": "#2F7D52",
        "tip": "#C4473A",
    }

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.25), constrained_layout=True)
    label_jobs = []
    for panel_index, (axis, projection) in enumerate(zip(axes, projections)):
        horizontal, vertical, title, xlabel, ylabel = projection

        for start, end in active_mechanism_links:
            indices = [IDX[start], IDX[end]]
            axis.plot(
                nodes[indices, horizontal], nodes[indices, vertical],
                color=colors["mechanism"], linewidth=1.35, solid_capstyle="round",
                zorder=1,
            )
        for start, end in active_construction_links:
            indices = [IDX[start], IDX[end]]
            axis.plot(
                nodes[indices, horizontal], nodes[indices, vertical],
                color=colors["construction"], linewidth=1.20,
                linestyle=(0, (3, 2)), solid_capstyle="round", zorder=2,
            )
        for start, end in active_extension_links:
            indices = [IDX[start], IDX[end]]
            axis.plot(
                nodes[indices, horizontal], nodes[indices, vertical],
                color=colors["extension"], linewidth=1.55, solid_capstyle="round",
                zorder=2,
            )

        axis.scatter(
            nodes[active_indices, horizontal], nodes[active_indices, vertical],
            s=15, facecolor="white", edgecolor=colors["node"], linewidth=0.65,
            zorder=3,
        )
        if "A" in active_names:
            a_point = nodes[IDX["A"]]
            axis.scatter(
                a_point[horizontal], a_point[vertical], marker="s", s=26,
                color=colors["node"], zorder=4,
            )
        if "L" in active_names:
            axis.scatter(
                nodes[IDX["L"], horizontal], nodes[IDX["L"], vertical],
                marker="o", s=28, color=colors["wrist"], zorder=4,
            )
        if "U" in active_names:
            axis.scatter(
                nodes[IDX["U"], horizontal], nodes[IDX["U"], vertical],
                marker="*", s=58, color=colors["tip"], zorder=4,
            )
        if include_input:
            # 当前输入曲线直接定义机构节点 B；这里只高亮 B，不再绘制历史 Mot 伪节点。
            b_point = nodes[IDX["B"]]
            axis.scatter(
                b_point[horizontal], b_point[vertical], marker="D", s=26,
                color=colors["input"], zorder=4,
            )

        label_jobs.append(
            (axis, horizontal, vertical, label_groups_by_panel[panel_index], panel_index)
        )

        axis.set_xlim(center[horizontal] - common_span / 2, center[horizontal] + common_span / 2)
        axis.set_ylim(center[vertical] - common_span / 2, center[vertical] + common_span / 2)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=9.5, pad=7)
        axis.grid(True, color="#D9DEE2", linewidth=0.45, linestyle=":")
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            0.015, 0.985, chr(ord("a") + panel_index), transform=axis.transAxes,
            fontsize=10, fontweight="bold", ha="left", va="top",
        )

    normalized_time = frame / result.nodes.shape[0]
    figure.suptitle(
        figure_title or f"Four-bar mechanism configuration at t/T = {normalized_time:.3f} (frame {frame})",
        fontsize=10.5,
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    candidate_offsets = (
        (6, 6), (-6, 6), (6, -6), (-6, -6),
        (12, 9), (-12, 9), (12, -9), (-12, -9),
        (18, 12), (-18, 12), (18, -12), (-18, -12),
        (24, 0), (-24, 0), (0, 20), (0, -20),
        (28, 18), (-28, 18), (28, -18), (-28, -18),
        (36, 0), (-36, 0), (0, 30), (0, -30),
    )
    label_priority = {
        name: rank for rank, name in enumerate(
            ("A", "B", "L", "U", "W", "K", "M", "N", "C", "D", "G", "J")
        )
    }
    for axis, horizontal, vertical, projected_groups, panel_index in label_jobs:
        occupied = []
        line_samples = []
        for line in axis.lines:
            vertices = np.asarray(line.get_xydata(), dtype=float)
            if vertices.ndim != 2 or vertices.shape[0] < 2:
                continue
            display_vertices = axis.transData.transform(vertices)
            for start, end in zip(display_vertices[:-1], display_vertices[1:]):
                length_px = float(np.linalg.norm(end - start))
                sample_count = max(2, int(np.ceil(length_px / 3.0)) + 1)
                weights = np.linspace(0.0, 1.0, sample_count)[:, None]
                line_samples.append(start + weights * (end - start))
        sampled_curves = np.vstack(line_samples) if line_samples else np.zeros((0, 2))

        groups = sorted(
            projected_groups.items(),
            key=lambda item: min(label_priority.get(name, 100 + IDX[name]) for name in item[1]),
        )
        for (x_value, y_value), names in groups:
            label = "/".join(names)
            selected_annotation = None
            for offset_x, offset_y in candidate_offsets:
                annotation = axis.annotate(
                    label, (x_value, y_value), xytext=(offset_x, offset_y),
                    textcoords="offset points", fontsize=6.4,
                    ha="left" if offset_x >= 0 else "right",
                    va="bottom" if offset_y >= 0 else "top",
                    color=colors["node"], zorder=5, clip_on=True,
                )
                label_box = annotation.get_window_extent(renderer).expanded(1.04, 1.12)
                curve_box = label_box.expanded(1.10, 1.24)
                inside_axis = (
                    axis.bbox.contains(label_box.x0, label_box.y0)
                    and axis.bbox.contains(label_box.x1, label_box.y1)
                )
                overlaps_curve = bool(
                    sampled_curves.size
                    and np.any(
                        (sampled_curves[:, 0] >= curve_box.x0)
                        & (sampled_curves[:, 0] <= curve_box.x1)
                        & (sampled_curves[:, 1] >= curve_box.y0)
                        & (sampled_curves[:, 1] <= curve_box.y1)
                    )
                )
                if (
                    inside_axis
                    and not overlaps_curve
                    and not any(label_box.overlaps(other) for other in occupied)
                ):
                    selected_annotation = annotation
                    occupied.append(label_box)
                    break
                annotation.remove()
            if selected_annotation is None:
                offset_x, offset_y = candidate_offsets[-1]
                selected_annotation = axis.annotate(
                    label, (x_value, y_value), xytext=(offset_x, offset_y),
                    textcoords="offset points", fontsize=6.4,
                    ha="right", va="top", color=colors["node"], zorder=5, clip_on=True,
                )
                occupied.append(
                    selected_annotation.get_window_extent(renderer).expanded(1.04, 1.12)
                )
            elif max(abs(offset_x), abs(offset_y)) >= 18:
                selected_annotation.remove()
                selected_annotation = axis.annotate(
                    label, (x_value, y_value), xytext=(offset_x, offset_y),
                    textcoords="offset points", fontsize=6.4,
                    ha="left" if offset_x >= 0 else "right",
                    va="bottom" if offset_y >= 0 else "top",
                    color=colors["node"], zorder=5, clip_on=True,
                    arrowprops={
                        "arrowstyle": "-", "color": "#7C858C", "linewidth": 0.45,
                        "shrinkA": 1.0, "shrinkB": 2.0,
                    },
                )

    figure.savefig(png_path, dpi=450, bbox_inches="tight", facecolor="white")
    figure.savefig(svg_path, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    with csv_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(("node", "x_mm", "y_mm", "z_mm", "frame"))
        for name in active_names:
            x_value, y_value, z_value = nodes[IDX[name]]
            writer.writerow((name, f"{x_value:.8f}", f"{y_value:.8f}", f"{z_value:.8f}", frame))

    if show:
        plt.show(block=True)
    else:
        plt.close(figure)
    return {
        "three_view_png": str(png_path.resolve()),
        "three_view_svg": str(svg_path.resolve()),
        "three_view_pdf": str(pdf_path.resolve()),
        "frame_nodes_csv": str(csv_path.resolve()),
    }


def plot_wrist_three_views(
    result: FourBarResult,
    frame: int = 0,
    output_dir: Path | str = CURRENT_DIR / "output",
    prefix: str = "fourbar_initial_wrist_three_view",
    show: bool = True,
) -> Dict[str, str]:
    """Export wrist views with complete links and selected visible joints."""
    normalized_time = frame / result.nodes.shape[0]
    return plot_initial_three_views(
        result,
        frame=frame,
        output_dir=output_dir,
        prefix=prefix,
        show=show,
        view_nodes=WRIST_VIEW_NODES,
        mechanism_links=WRIST_MECHANISM_LINKS,
        construction_links=CONSTRUCTION_LINKS,
        extension_links=WRIST_EXTENSION_LINKS,
        include_input=False,
        figure_title=f"Wrist mechanism detail at t/T = {normalized_time:.3f} (frame {frame})",
    )


def plot_jo_drop_geometry(
    result: FourBarResult,
    frame: int = 0,
    output_dir: Path | str = CURRENT_DIR / "output",
    prefix: str = "wrist_jo_drop_geometry",
    show: bool = True,
) -> Dict[str, str]:
    """用实际节点坐标显示 J/O 的全局下降量、J-O 构型线和 PO:QP 比例。"""
    import matplotlib as mpl
    if not show:
        mpl.use("Agg")
    import matplotlib.pyplot as plt

    if not result.valid or result.nodes.ndim != 3 or result.nodes.shape[0] == 0:
        raise FourBarError("a valid FourBarResult is required for the J/O geometry plot")
    if not 0 <= frame < result.nodes.shape[0]:
        raise IndexError(f"frame must be in [0, {result.nodes.shape[0] - 1}], got {frame}")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    png_path = destination / f"{prefix}.png"
    svg_path = destination / f"{prefix}.svg"
    pdf_path = destination / f"{prefix}.pdf"
    csv_path = destination / f"{prefix}_measurements.csv"

    nodes = np.asarray(result.nodes[frame], dtype=float)
    visible_names = ("I", "J", "K", "M", "O", "P", "Q")
    visible_indices = np.array([IDX[name] for name in visible_names], dtype=int)
    physical_links = (
        ("I", "J"), ("J", "K"), ("K", "M"),
        ("M", "O"), ("O", "P"), ("P", "Q"),
    )
    po = float(np.linalg.norm(nodes[IDX["P"]] - nodes[IDX["O"]]))
    qp = float(np.linalg.norm(nodes[IDX["Q"]] - nodes[IDX["P"]]))
    ratio = po / max(qp, 1e-12)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.0,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    colors = {
        "physical": "#3F4852",
        "construction": "#CC79A7",
        "drop": "#D55E00",
        "reference": "#9AA4AA",
        "node": "#20252A",
        "j": "#0072B2",
        "o": "#009E73",
    }
    projections = (
        (0, 1, "Front: global downward offsets", "x (mm)", "y (mm)"),
        (1, 2, "Side: J-O spatial relation", "y (mm)", "z (mm)"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), constrained_layout=True)
    for panel, (axis, projection) in enumerate(zip(axes, projections)):
        horizontal, vertical, title, xlabel, ylabel = projection
        for start, end in physical_links:
            axis.plot(
                nodes[[IDX[start], IDX[end]], horizontal],
                nodes[[IDX[start], IDX[end]], vertical],
                color=colors["physical"], linewidth=1.55,
                solid_capstyle="round", zorder=1,
            )
        axis.plot(
            nodes[[IDX["J"], IDX["O"]], horizontal],
            nodes[[IDX["J"], IDX["O"]], vertical],
            color=colors["construction"], linewidth=1.45,
            linestyle=(0, (4, 2)), zorder=2,
            label="J-O construction",
        )
        axis.scatter(
            nodes[visible_indices, horizontal], nodes[visible_indices, vertical],
            s=22, facecolor="white", edgecolor=colors["node"],
            linewidth=0.75, zorder=3,
        )
        axis.scatter(
            nodes[IDX["J"], horizontal], nodes[IDX["J"], vertical],
            s=34, color=colors["j"], zorder=4,
        )
        axis.scatter(
            nodes[IDX["O"], horizontal], nodes[IDX["O"], vertical],
            s=34, color=colors["o"], zorder=4,
        )
        for name in visible_names:
            point = nodes[IDX[name]]
            axis.annotate(
                name, (point[horizontal], point[vertical]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=7.0, color=colors["node"], zorder=5,
            )

        if panel == 0:
            axis.axhline(
                0.0, color=colors["reference"], linewidth=0.75,
                linestyle=(0, (2, 2)), zorder=0,
            )
            for name, color in (("J", colors["j"]), ("O", colors["o"])):
                point = nodes[IDX[name]]
                axis.annotate(
                    "", xy=(point[0], point[1]), xytext=(point[0], 0.0),
                    arrowprops={
                        "arrowstyle": "-|>", "color": color,
                        "linewidth": 1.1, "shrinkA": 0.0, "shrinkB": 2.0,
                    },
                )
            # 数值集中放入独立说明框，避免 J/O 距离较近时标注互相遮挡。
            axis.text(
                0.985, 0.965,
                (
                    f"J global y = {nodes[IDX['J'], 1]:.2f} mm\n"
                    f"O global y = {nodes[IDX['O'], 1]:.2f} mm"
                ),
                transform=axis.transAxes, ha="right", va="top", fontsize=7.2,
                bbox={
                    "facecolor": "white", "edgecolor": "#C9D0D5",
                    "linewidth": 0.6, "boxstyle": "square,pad=0.35",
                },
            )
        else:
            axis.text(
                0.02, 0.98,
                f"PO={po:.2f} mm\nQP={qp:.2f} mm\nPO/QP={ratio:.3f}",
                transform=axis.transAxes, ha="left", va="top", fontsize=7.4,
                bbox={
                    "facecolor": "white", "edgecolor": "#C9D0D5",
                    "linewidth": 0.6, "boxstyle": "square,pad=0.35",
                },
            )

        displayed = nodes[visible_indices][:, [horizontal, vertical]]
        margin = max(float(np.max(np.ptp(displayed, axis=0))) * 0.12, 8.0)
        axis.set_xlim(float(np.min(displayed[:, 0])) - margin, float(np.max(displayed[:, 0])) + margin)
        vertical_min = min(float(np.min(displayed[:, 1])) - margin, -margin if panel == 0 else np.inf)
        vertical_max = max(float(np.max(displayed[:, 1])) + margin, margin if panel == 0 else -np.inf)
        axis.set_ylim(vertical_min, vertical_max)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=9.0, pad=7)
        axis.grid(True, color="#D9DEE2", linewidth=0.45, linestyle=":")
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            0.015, 0.985, chr(ord("a") + panel),
            transform=axis.transAxes, ha="left", va="top",
            fontsize=10, fontweight="bold",
        )

    figure.suptitle(
        f"Wrist construction at t/T={frame / result.nodes.shape[0]:.3f}: J/O drop and PO=QP",
        fontsize=10.0,
    )
    figure.savefig(png_path, dpi=450, bbox_inches="tight", facecolor="white")
    figure.savefig(svg_path, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame", "J_y_mm", "O_y_mm", "PO_mm", "QP_mm", "PO_over_QP"))
        writer.writerow((frame, nodes[IDX["J"], 1], nodes[IDX["O"], 1], po, qp, ratio))
    if show:
        plt.show(block=True)
    else:
        plt.close(figure)
    return {
        "jo_geometry_png": str(png_path.resolve()),
        "jo_geometry_svg": str(svg_path.resolve()),
        "jo_geometry_pdf": str(pdf_path.resolve()),
        "jo_geometry_csv": str(csv_path.resolve()),
    }


def animate_three_views(
    result: FourBarResult,
    output_dir: Path | str = CURRENT_DIR / "output",
    prefix: str = "fourbar_initial_three_view_motion",
    fps: int = 15,
    dpi: int = 150,
    target_tip: np.ndarray | None = None,
    target_wrist: np.ndarray | None = None,
) -> Dict[str, str]:
    """Export synchronized views, optionally including index-matched targets."""
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import imageio_ffmpeg

    if not result.valid or result.nodes.ndim != 3 or result.nodes.shape[0] == 0:
        raise FourBarError("a valid FourBarResult is required for animation")
    if fps <= 0:
        raise ValueError("fps must be positive")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    mp4_path = destination / f"{prefix}.mp4"
    gif_path = destination / f"{prefix}.gif"

    nodes = np.asarray(result.nodes, dtype=float)
    b_curve = np.asarray(result.b_curve, dtype=float)
    target_tip_curve = (
        resample_rows(np.asarray(target_tip, dtype=float), nodes.shape[0])
        if target_tip is not None else None
    )
    target_wrist_curve = (
        resample_rows(np.asarray(target_wrist, dtype=float), nodes.shape[0])
        if target_wrist is not None else None
    )
    active_names = tuple(
        name for name in result.node_names if name not in {"R", "V", "X", "Y"}
    )
    active_indices = np.array([IDX[name] for name in active_names], dtype=int)
    display_blocks = [nodes[:, active_indices, :].reshape(-1, 3)]
    if target_tip_curve is not None:
        display_blocks.append(target_tip_curve)
    if target_wrist_curve is not None:
        display_blocks.append(target_wrist_curve)
    display_points = np.vstack(display_blocks)
    center = 0.5 * (np.min(display_points, axis=0) + np.max(display_points, axis=0))
    common_span = max(float(np.max(np.ptp(display_points, axis=0))) * 1.10, 1.0)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.5,
        "axes.linewidth": 0.8,
    })
    projections = (
        (0, 1, "Front view", "x (mm)", "y (mm)"),
        (0, 2, "Top view", "x (mm)", "z (mm)"),
        (1, 2, "Side view", "y (mm)", "z (mm)"),
    )
    colors = {
        "mechanism": "#3F4852",
        "extension": "#26718C",
        "input": "#C46A32",
        "node": "#20252A",
        "wrist": "#2F7D52",
        "tip": "#C4473A",
        "target_wrist": "#72A98F",
        "target_tip": "#E39A7A",
    }

    frame0 = nodes[0, active_indices]
    labels_by_panel: list[list[str]] = [[], [], []]
    for local_index, name in enumerate(active_names):
        panel_scores = []
        for horizontal, vertical, *_ in projections:
            projected = frame0[:, [horizontal, vertical]]
            distances = np.linalg.norm(projected - projected[local_index], axis=1)
            positive = np.sort(distances[distances > 1e-9])
            if positive.size == 0:
                panel_scores.append(0.0)
            else:
                nearest = float(positive[0])
                neighborhood = float(np.mean(positive[:min(3, positive.size)]))
                panel_scores.append(nearest + 0.20 * neighborhood)
        labels_by_panel[int(np.argmax(panel_scores))].append(name)

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.25), constrained_layout=True)
    primary_artists = []
    construction_artists = []
    extension_artists = []
    node_artists = []
    fixed_artists = []
    wrist_artists = []
    tip_artists = []
    target_wrist_artists = []
    target_tip_artists = []
    label_artists: list[tuple[Any, str, int, int]] = []
    label_offsets = (
        (7, 7), (-7, 7), (7, -7), (-7, -7),
        (13, 9), (-13, 9), (13, -9), (-13, -9),
    )

    for panel_index, (axis, projection) in enumerate(zip(axes, projections)):
        horizontal, vertical, title, xlabel, ylabel = projection
        panel_primary = []
        for start, end in MECHANISM_LINKS:
            line, = axis.plot(
                [], [], color=colors["mechanism"], linewidth=1.35,
                solid_capstyle="round", zorder=2,
            )
            panel_primary.append((line, IDX[start], IDX[end]))
        primary_artists.append(panel_primary)

        panel_construction = []
        for start, end in CONSTRUCTION_LINKS:
            line, = axis.plot(
                [], [], color=colors.get("construction", "#CC79A7"),
                linewidth=1.20, linestyle=(0, (3, 2)),
                solid_capstyle="round", zorder=3,
            )
            panel_construction.append((line, IDX[start], IDX[end]))
        construction_artists.append(panel_construction)

        panel_extensions = []
        for start, end in EXTENSION_LINKS:
            line, = axis.plot(
                [], [], color=colors["extension"], linewidth=1.55,
                solid_capstyle="round", zorder=3,
            )
            panel_extensions.append((line, IDX[start], IDX[end]))
        extension_artists.append(panel_extensions)

        node_artists.append(axis.scatter(
            [], [], s=15, facecolor="white", edgecolor=colors["node"],
            linewidth=0.65, zorder=4,
        ))
        fixed_artists.append(axis.scatter([], [], marker="s", s=26, color=colors["node"], zorder=5))
        wrist_artists.append(axis.scatter([], [], marker="o", s=28, color=colors["wrist"], zorder=5))
        tip_artists.append(axis.scatter([], [], marker="*", s=58, color=colors["tip"], zorder=5))
        target_wrist_artists.append(axis.scatter(
            [], [], marker="o", s=31, facecolor="white",
            edgecolor=colors["target_wrist"], linewidth=1.0, zorder=5,
        ))
        target_tip_artists.append(axis.scatter(
            [], [], marker="*", s=68, facecolor="white",
            edgecolor=colors["target_tip"], linewidth=1.0, zorder=5,
        ))

        axis.plot(
            nodes[:, IDX["U"], horizontal], nodes[:, IDX["U"], vertical],
            color=colors["tip"], linewidth=0.85, alpha=0.22, zorder=0,
        )
        axis.plot(
            nodes[:, IDX["L"], horizontal], nodes[:, IDX["L"], vertical],
            color=colors["wrist"], linewidth=0.85, alpha=0.22, zorder=0,
        )
        axis.plot(
            b_curve[:, horizontal], b_curve[:, vertical],
            color=colors["input"], linewidth=1.0, linestyle=(0, (3, 2)),
            alpha=0.42, zorder=1,
        )
        if target_tip_curve is not None:
            axis.plot(
                target_tip_curve[:, horizontal], target_tip_curve[:, vertical],
                color=colors["target_tip"], linewidth=1.0,
                linestyle=(0, (5, 2)), alpha=0.72, zorder=1,
            )
        if target_wrist_curve is not None:
            axis.plot(
                target_wrist_curve[:, horizontal], target_wrist_curve[:, vertical],
                color=colors["target_wrist"], linewidth=1.0,
                linestyle=(0, (5, 2)), alpha=0.72, zorder=1,
            )
        for label_index, name in enumerate(labels_by_panel[panel_index]):
            offset_x, offset_y = label_offsets[label_index % len(label_offsets)]
            annotation = axis.annotate(
                name, (0.0, 0.0), xytext=(offset_x, offset_y),
                textcoords="offset points", fontsize=6.4,
                ha="left" if offset_x >= 0 else "right",
                va="bottom" if offset_y >= 0 else "top",
                color=colors["node"], zorder=6, clip_on=True,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.15},
            )
            label_artists.append((annotation, name, horizontal, vertical))
        axis.set_xlim(center[horizontal] - common_span / 2, center[horizontal] + common_span / 2)
        axis.set_ylim(center[vertical] - common_span / 2, center[vertical] + common_span / 2)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=9.5, pad=7)
        axis.grid(True, color="#D9DEE2", linewidth=0.45, linestyle=":")
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            0.015, 0.985, chr(ord("a") + panel_index), transform=axis.transAxes,
            fontsize=10, fontweight="bold", ha="left", va="top",
        )

    title_artist = figure.suptitle(
        "Four-bar mechanism motion over one cycle | t/T = 0.000", fontsize=10.5
    )

    def update(frame: int) -> list[Any]:
        frame_nodes = nodes[frame]
        updated: list[Any] = [title_artist]
        for panel_index, (horizontal, vertical, *_rest) in enumerate(projections):
            for line, start, end in primary_artists[panel_index]:
                line.set_data(
                    [frame_nodes[start, horizontal], frame_nodes[end, horizontal]],
                    [frame_nodes[start, vertical], frame_nodes[end, vertical]],
                )
                updated.append(line)
            for line, start, end in extension_artists[panel_index]:
                line.set_data(
                    [frame_nodes[start, horizontal], frame_nodes[end, horizontal]],
                    [frame_nodes[start, vertical], frame_nodes[end, vertical]],
                )
                updated.append(line)
            for line, start, end in construction_artists[panel_index]:
                line.set_data(
                    [frame_nodes[start, horizontal], frame_nodes[end, horizontal]],
                    [frame_nodes[start, vertical], frame_nodes[end, vertical]],
                )
                updated.append(line)
            node_artists[panel_index].set_offsets(frame_nodes[active_indices][:, [horizontal, vertical]])
            fixed_artists[panel_index].set_offsets([frame_nodes[IDX["A"], [horizontal, vertical]]])
            wrist_artists[panel_index].set_offsets([frame_nodes[IDX["L"], [horizontal, vertical]]])
            tip_artists[panel_index].set_offsets([frame_nodes[IDX["U"], [horizontal, vertical]]])
            if target_wrist_curve is not None:
                target_wrist_artists[panel_index].set_offsets([
                    target_wrist_curve[frame, [horizontal, vertical]]
                ])
            if target_tip_curve is not None:
                target_tip_artists[panel_index].set_offsets([
                    target_tip_curve[frame, [horizontal, vertical]]
                ])
            updated.extend([
                node_artists[panel_index], fixed_artists[panel_index],
                wrist_artists[panel_index], tip_artists[panel_index],
                target_wrist_artists[panel_index], target_tip_artists[panel_index],
            ])
        for annotation, name, horizontal, vertical in label_artists:
            point = frame_nodes[IDX[name]]
            annotation.xy = (float(point[horizontal]), float(point[vertical]))
            updated.append(annotation)
        title_artist.set_text(
            f"Four-bar mechanism motion over one cycle | t/T = {frame / nodes.shape[0]:.3f}"
        )
        return updated

    frame_sequence = list(range(nodes.shape[0])) + [0]
    motion = animation.FuncAnimation(
        figure, update, frames=frame_sequence,
        interval=1000.0 / fps, blit=False, repeat=True, cache_frame_data=False,
    )
    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    mp4_writer = animation.FFMpegWriter(
        fps=fps, codec="h264", bitrate=3200,
        extra_args=["-pix_fmt", "yuv420p"],
        metadata={"title": "Four-bar mechanism three-view motion"},
    )
    motion.save(mp4_path, writer=mp4_writer, dpi=dpi)
    motion.save(gif_path, writer=animation.PillowWriter(fps=fps), dpi=100)
    plt.close(figure)
    return {
        "three_view_motion_mp4": str(mp4_path.resolve()),
        "three_view_motion_gif": str(gif_path.resolve()),
    }


def export_interactive_threejs(
    result: FourBarResult,
    output_dir: Path | str = CURRENT_DIR / "output",
    filename: str = "fourbar_interactive_3d.html",
) -> Dict[str, str]:
    """Export a rotatable Three.js mechanism view with a motion-phase slider."""
    if not result.valid or result.nodes.ndim != 3 or result.nodes.shape[0] == 0:
        raise FourBarError("a valid FourBarResult is required for interactive 3D export")

    active_names = tuple(
        name for name in result.node_names if name not in {"R", "V", "X", "Y"}
    )
    active_indices = np.array([IDX[name] for name in active_names], dtype=int)
    index_by_name = {name: index for index, name in enumerate(active_names)}
    links = []
    for kind, source in (
        ("mechanism", MECHANISM_LINKS),
        ("construction", CONSTRUCTION_LINKS),
        ("extension", EXTENSION_LINKS),
    ):
        for start, end in source:
            if start in index_by_name and end in index_by_name:
                links.append({
                    "start": index_by_name[start],
                    "end": index_by_name[end],
                    "kind": kind,
                })

    frames = np.asarray(result.nodes[:, active_indices, :], dtype=float)
    display_points = frames.reshape(-1, 3)
    bounds_min = np.min(display_points, axis=0)
    bounds_max = np.max(display_points, axis=0)
    center = 0.5 * (bounds_min + bounds_max)
    span = max(float(np.max(bounds_max - bounds_min)), 1.0)
    payload = {
        "nodeNames": active_names,
        "frames": np.round(frames, 8).tolist(),
        "links": links,
        "phase": (np.arange(frames.shape[0], dtype=float) / frames.shape[0]).tolist(),
        "bounds": {
            "min": bounds_min.tolist(),
            "max": bounds_max.tolist(),
            "center": center.tolist(),
            "span": span,
        },
    }

    html = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Four-bar mechanism interactive 3D</title>
  <style>
    :root { color-scheme: light; font-family: Arial, Helvetica, sans-serif; }
    * { box-sizing: border-box; }
    html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; background: #f7f9fa; color: #20252a; }
    #viewport { position: absolute; inset: 0; }
    .heading { position: absolute; top: 16px; left: 18px; z-index: 2; pointer-events: none; }
    .heading h1 { margin: 0; font-size: 18px; line-height: 1.2; font-weight: 650; letter-spacing: 0; }
    .heading p { margin: 5px 0 0; font-size: 12px; color: #58636d; }
    .legend { display: flex; gap: 14px; margin-top: 9px; font-size: 11px; color: #46515a; }
    .legend span { display: inline-flex; align-items: center; gap: 5px; }
    .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
    .toolbar { position: absolute; top: 16px; right: 18px; z-index: 3; display: flex; overflow: hidden; border: 1px solid #c9d0d5; border-radius: 6px; background: rgba(255,255,255,.94); box-shadow: 0 2px 8px rgba(31,40,48,.09); }
    .toolbar button { height: 34px; min-width: 50px; padding: 0 11px; border: 0; border-right: 1px solid #d8dde1; background: transparent; color: #28323a; font-size: 11px; cursor: pointer; }
    .toolbar button:last-child { border-right: 0; }
    .toolbar button:hover, .toolbar button:focus-visible { background: #edf2f4; outline: none; }
    .toolbar button.active { background: #dfeaec; color: #174e5d; font-weight: 650; }
    .phasebar { position: absolute; left: 18px; right: 18px; bottom: 16px; z-index: 3; display: grid; grid-template-columns: 92px minmax(160px, 1fr) 62px; align-items: center; gap: 12px; height: 42px; padding: 0 13px; border: 1px solid #c9d0d5; border-radius: 6px; background: rgba(255,255,255,.94); box-shadow: 0 2px 8px rgba(31,40,48,.09); }
    .phasebar label, .phasebar output { font-size: 11px; color: #3f4a52; }
    .phasebar output { text-align: right; font-variant-numeric: tabular-nums; }
    .phasebar input { width: 100%; accent-color: #26718c; }
    .error { position: absolute; inset: 0; display: none; place-items: center; padding: 24px; color: #9d3127; font-size: 14px; text-align: center; }
    @media (max-width: 760px) {
      .heading h1 { font-size: 15px; }
      .legend { display: none; }
      .toolbar { top: 72px; left: 18px; right: auto; max-width: calc(100% - 36px); }
      .toolbar button { min-width: 44px; padding: 0 8px; }
      .phasebar { grid-template-columns: 58px minmax(90px, 1fr) 50px; }
    }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="viewport"></div>
  <div class="heading">
    <h1>Four-bar mechanism</h1>
    <p>Spatial configuration in millimetres</p>
    <div class="legend">
      <span><i class="dot" style="background:#20252a"></i>A fixed</span>
      <span><i class="dot" style="background:#2f7d52"></i>L wrist</span>
      <span><i class="dot" style="background:#c4473a"></i>U tip</span>
      <span><i class="dot" style="background:#c46a32"></i>B input path</span>
      <span><i class="dot" style="background:#cc79a7"></i>J-O construction</span>
    </div>
  </div>
  <div class="toolbar" role="group" aria-label="Camera views">
    <button type="button" data-view="iso" class="active" title="Isometric view">Iso</button>
    <button type="button" data-view="front" title="Front view">Front</button>
    <button type="button" data-view="top" title="Top view">Top</button>
    <button type="button" data-view="side" title="Side view">Side</button>
    <button type="button" id="labels" class="active" title="Show or hide joint labels">Labels</button>
  </div>
  <div class="phasebar">
    <label for="phase">Motion phase</label>
    <input id="phase" type="range" min="0" max="75" step="1" value="0">
    <output id="phaseValue" for="phase">t/T 0.000</output>
  </div>
  <div id="error" class="error"></div>
  <script>
    const MODEL = __MODEL_DATA__;
    const host = document.getElementById('viewport');
    const errorBox = document.getElementById('error');
    if (!window.THREE || !THREE.OrbitControls) {
      errorBox.style.display = 'grid';
      errorBox.textContent = 'The 3D rendering library could not be loaded.';
      throw new Error(errorBox.textContent);
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf7f9fa);
    const camera = new THREE.PerspectiveCamera(36, window.innerWidth / window.innerHeight, 0.1, 10000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputEncoding = THREE.sRGBEncoding;
    host.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.screenSpacePanning = true;
    controls.minDistance = MODEL.bounds.span * 0.18;
    controls.maxDistance = MODEL.bounds.span * 8;

    scene.add(new THREE.HemisphereLight(0xffffff, 0xaeb8bf, 1.15));
    const keyLight = new THREE.DirectionalLight(0xffffff, 0.85);
    keyLight.position.set(1.5, 2.2, 1.8);
    scene.add(keyLight);

    const center = new THREE.Vector3(...MODEL.bounds.center);
    const span = MODEL.bounds.span;
    const grid = new THREE.GridHelper(span * 1.55, 20, 0x9ba7ae, 0xd8dfe3);
    grid.position.set(center.x, MODEL.bounds.min[1] - span * 0.05, center.z);
    scene.add(grid);
    const axes = new THREE.AxesHelper(span * 0.18);
    scene.add(axes);

    function labelSprite(text, color) {
      const canvas = document.createElement('canvas');
      canvas.width = 160; canvas.height = 72;
      const context = canvas.getContext('2d');
      context.font = '600 32px Arial';
      context.textAlign = 'center'; context.textBaseline = 'middle';
      context.fillStyle = 'rgba(255,255,255,.86)';
      context.fillRect(35, 16, 90, 40);
      context.fillStyle = color;
      context.fillText(text, 80, 37);
      const texture = new THREE.CanvasTexture(canvas);
      texture.minFilter = THREE.LinearFilter;
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }));
      sprite.scale.set(span * 0.07, span * 0.0315, 1);
      sprite.userData.baseScale = [span * 0.07, span * 0.0315];
      sprite.renderOrder = 20;
      return sprite;
    }

    const nodeRadius = Math.max(span * 0.0065, 1.35);
    const nodeGeometry = new THREE.SphereGeometry(nodeRadius, 22, 14);
    const nodeMeshes = [];
    const nodeLabels = [];
    const keyColors = { A: 0x20252a, L: 0x2f7d52, U: 0xc4473a };
    MODEL.nodeNames.forEach((name, index) => {
      const material = new THREE.MeshPhongMaterial({
        color: keyColors[name] || 0xf9fbfc,
        emissive: keyColors[name] || 0x000000,
        emissiveIntensity: keyColors[name] ? 0.09 : 0,
        specular: 0x60717a,
        shininess: 45,
      });
      const mesh = new THREE.Mesh(nodeGeometry, material);
      mesh.userData.nodeIndex = index;
      scene.add(mesh); nodeMeshes.push(mesh);
      const label = labelSprite(name, keyColors[name] ? '#233038' : '#46515a');
      scene.add(label); nodeLabels.push(label);
    });

    function lineObject(kind) {
      const positions = new Float32Array(6);
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      const material = new THREE.LineBasicMaterial({
        color: kind === 'extension' ? 0x26718c : kind === 'construction' ? 0xcc79a7 : kind === 'input' ? 0xc46a32 : 0x46515a,
        transparent: true,
        opacity: kind === 'mechanism' ? 0.96 : 0.92,
      });
      return new THREE.Line(geometry, material);
    }

    const linkObjects = MODEL.links.map(link => {
      const line = lineObject(link.kind);
      line.userData.link = link;
      scene.add(line);
      return line;
    });
    function setLine(line, start, end) {
      const values = line.geometry.attributes.position.array;
      values[0] = start[0]; values[1] = start[1]; values[2] = start[2];
      values[3] = end[0]; values[4] = end[1]; values[5] = end[2];
      line.geometry.attributes.position.needsUpdate = true;
      line.geometry.computeBoundingSphere();
    }

    const phaseInput = document.getElementById('phase');
    const phaseValue = document.getElementById('phaseValue');
    phaseInput.max = String(MODEL.frames.length - 1);
    function updateFrame(frameIndex) {
      const frame = MODEL.frames[frameIndex];
      frame.forEach((point, index) => {
        nodeMeshes[index].position.set(...point);
      });
      MODEL.links.forEach((link, index) => setLine(linkObjects[index], frame[link.start], frame[link.end]));
      phaseValue.value = `t/T ${MODEL.phase[frameIndex].toFixed(3)}`;
    }
    phaseInput.addEventListener('input', event => updateFrame(Number(event.target.value)));

    const distance = span / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.28;
    function updateLabelPresentation() {
      const cameraDistance = camera.position.distanceTo(controls.target);
      const scaleFactor = THREE.MathUtils.clamp(cameraDistance / distance, 0.05, 4.0);
      const offset = nodeRadius * 3.1 * scaleFactor;
      nodeLabels.forEach((label, index) => {
        const [baseWidth, baseHeight] = label.userData.baseScale;
        label.scale.set(baseWidth * scaleFactor, baseHeight * scaleFactor, 1);
        label.position.copy(nodeMeshes[index].position);
        label.position.y += offset;
      });
    }
    function placeCamera(direction, up) {
      camera.up.set(...up);
      const vector = new THREE.Vector3(...direction).normalize().multiplyScalar(distance);
      camera.position.copy(center).add(vector);
      camera.near = Math.max(distance / 500, 0.1);
      camera.far = distance * 20;
      camera.updateProjectionMatrix();
      controls.target.copy(center);
      controls.update();
    }
    const views = {
      iso: [[1.15, 0.82, 1.2], [0, 1, 0]],
      front: [[0, 0, 1], [0, 1, 0]],
      top: [[0, 1, 0], [0, 0, 1]],
      side: [[1, 0, 0], [0, 0, 1]],
    };
    document.querySelectorAll('[data-view]').forEach(button => {
      button.addEventListener('click', () => {
        document.querySelectorAll('[data-view]').forEach(item => item.classList.remove('active'));
        button.classList.add('active');
        const [direction, up] = views[button.dataset.view];
        placeCamera(direction, up);
      });
    });
    let labelsVisible = true;
    document.getElementById('labels').addEventListener('click', event => {
      labelsVisible = !labelsVisible;
      nodeLabels.forEach(label => { label.visible = labelsVisible; });
      event.currentTarget.classList.toggle('active', labelsVisible);
    });

    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });
    updateFrame(0);
    placeCamera(...views.iso);
    function render() {
      requestAnimationFrame(render);
      controls.update();
      updateLabelPresentation();
      renderer.render(scene, camera);
    }
    render();
  </script>
</body>
</html>
"""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    html_path = destination / filename
    html_path.write_text(
        html.replace("__MODEL_DATA__", json.dumps(payload, separators=(",", ":"))),
        encoding="utf-8",
    )
    return {"interactive_threejs_html": str(html_path.resolve())}


def plot_internal_angle_curves(
    result: FourBarResult,
    phase01: np.ndarray | None = None,
    output_dir: Path | str = CURRENT_DIR / "output",
    prefix: str = "fourbar_internal_angles",
    show: bool = True,
) -> Dict[str, Any]:
    """Plot theta_wrist, theta20, and theta21 over one motion cycle."""
    import matplotlib as mpl
    if not show:
        mpl.use("Agg")
    import matplotlib.pyplot as plt

    if not result.valid:
        raise FourBarError("a valid FourBarResult is required for angle curves")
    angles_rad = np.column_stack([
        np.asarray(result.theta_wrist, dtype=float),
        np.asarray(result.theta20, dtype=float),
        np.asarray(result.theta21, dtype=float),
    ])
    if angles_rad.ndim != 2 or angles_rad.shape[0] == 0 or np.any(~np.isfinite(angles_rad)):
        raise FourBarError("internal angle arrays must be non-empty and finite")
    if phase01 is None:
        phase = np.arange(angles_rad.shape[0], dtype=float) / angles_rad.shape[0]
    else:
        phase = np.asarray(phase01, dtype=float).reshape(-1)
        if phase.size != angles_rad.shape[0]:
            raise ValueError("phase01 and internal angle arrays must have the same length")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    png_path = destination / f"{prefix}.png"
    svg_path = destination / f"{prefix}.svg"
    pdf_path = destination / f"{prefix}.pdf"
    csv_path = destination / f"{prefix}.csv"
    angles_deg = np.rad2deg(angles_rad)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.5,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(
        3, 1, figsize=(7.2, 6.2), sharex=True, constrained_layout=True
    )
    series = (
        (r"$\theta_{wrist}$", "#26718C"),
        (r"$\theta_{20}$", "#C46A32"),
        (r"$\theta_{21}$", "#3F7D5A"),
    )
    for panel_index, (axis, (label, color)) in enumerate(zip(axes, series)):
        axis.plot(phase, angles_deg[:, panel_index], color=color, linewidth=1.8)
        axis.set_ylabel(f"{label} (deg)")
        axis.grid(True, color="#D9DEE2", linewidth=0.55, linestyle=":")
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(
            0.012, 0.94, chr(ord("a") + panel_index), transform=axis.transAxes,
            fontsize=9.5, fontweight="bold", ha="left", va="top",
        )
    axes[0].set_title("Internal mechanism angles over one motion cycle", fontsize=10)
    axes[-1].set_xlabel("Normalized time, t/T")
    axes[-1].set_xlim(float(np.min(phase)), float(np.max(phase)))
    figure.savefig(png_path, dpi=300, facecolor="white")
    figure.savefig(svg_path, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(figure)

    table = np.column_stack([phase, angles_rad, angles_deg])
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header=(
            "t_over_T,theta_wrist_rad,theta20_rad,theta21_rad,"
            "theta_wrist_deg,theta20_deg,theta21_deg"
        ),
        comments="",
        fmt="%.10g",
    )
    return {
        "angle_curves_png": str(png_path.resolve()),
        "angle_curves_svg": str(svg_path.resolve()),
        "angle_curves_pdf": str(pdf_path.resolve()),
        "angle_curves_csv": str(csv_path.resolve()),
        "angle_ranges_deg": {
            "theta_wrist": [float(np.min(angles_deg[:, 0])), float(np.max(angles_deg[:, 0]))],
            "theta20": [float(np.min(angles_deg[:, 1])), float(np.max(angles_deg[:, 1]))],
            "theta21": [float(np.min(angles_deg[:, 2])), float(np.max(angles_deg[:, 2]))],
        },
    }


def plot_theta_m_curve(
    result: FourBarResult,
    phase01: np.ndarray | None = None,
    output_dir: Path | str = CURRENT_DIR / "output",
    prefix: str = "fourbar_thetaM",
    show: bool = True,
) -> Dict[str, Any]:
    """Plot thetaM over one motion cycle and export radians/degrees to CSV."""
    import matplotlib as mpl
    if not show:
        mpl.use("Agg")
    import matplotlib.pyplot as plt

    theta_m_rad = np.asarray(result.thetaM, dtype=float).reshape(-1)
    if not result.valid or theta_m_rad.size == 0 or np.any(~np.isfinite(theta_m_rad)):
        raise FourBarError("a valid finite thetaM curve is required")
    if phase01 is None:
        phase = np.arange(theta_m_rad.size, dtype=float) / theta_m_rad.size
    else:
        phase = np.asarray(phase01, dtype=float).reshape(-1)
        if phase.size != theta_m_rad.size:
            raise ValueError("phase01 and thetaM must have the same length")
    theta_m_deg = np.rad2deg(theta_m_rad)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    png_path = destination / f"{prefix}.png"
    svg_path = destination / f"{prefix}.svg"
    pdf_path = destination / f"{prefix}.pdf"
    csv_path = destination / f"{prefix}.csv"

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 9.0,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    axis.plot(phase, theta_m_deg, color="#6A4C93", linewidth=1.9)
    axis.set_xlabel("Normalized time, t/T")
    axis.set_ylabel(r"$\theta_M$ (deg)")
    axis.set_title(r"$\theta_M$ over one motion cycle", fontsize=10)
    axis.set_xlim(float(np.min(phase)), float(np.max(phase)))
    axis.grid(True, color="#D9DEE2", linewidth=0.55, linestyle=":")
    axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(png_path, dpi=300, facecolor="white")
    figure.savefig(svg_path, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(figure)

    np.savetxt(
        csv_path,
        np.column_stack([phase, theta_m_rad, theta_m_deg]),
        delimiter=",",
        header="t_over_T,thetaM_rad,thetaM_deg",
        comments="",
        fmt="%.10g",
    )
    return {
        "thetaM_png": str(png_path.resolve()),
        "thetaM_svg": str(svg_path.resolve()),
        "thetaM_pdf": str(pdf_path.resolve()),
        "thetaM_csv": str(csv_path.resolve()),
        "thetaM_range_deg": [float(np.min(theta_m_deg)), float(np.max(theta_m_deg))],
    }


def plot_mot_input_diagnostics(
    mot_curve: np.ndarray,
    ab_length: float,
    phase01: np.ndarray | None = None,
    output_dir: Path | str = CURRENT_DIR / "output",
    prefix: str = "mot_input",
    show: bool = True,
) -> Dict[str, str]:
    """输出直接 B 输入曲线、AB(t) 及两个空间方向角。"""
    import matplotlib as mpl
    if not show:
        mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9.0,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    b_curve = np.asarray(mot_curve, dtype=float)
    if b_curve.ndim != 2 or b_curve.shape[1] != 3:
        raise ValueError("b_curve must have shape (n, 3).")
    if phase01 is None:
        normalized_time = np.arange(b_curve.shape[0], dtype=float) / b_curve.shape[0]
    else:
        normalized_time = np.asarray(phase01, dtype=float).reshape(-1)
        if normalized_time.size != b_curve.shape[0]:
            raise ValueError("phase01 and b_curve must contain the same number of samples.")

    input_radius, theta01, theta02, _input_norm = bcurve_to_input_angles(b_curve)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    spatial_path = destination / f"{prefix}_spatial_curve.png"
    input_path = destination / f"{prefix}_parameters_vs_time.png"
    csv_path = destination / f"{prefix}_parameters_vs_time.csv"
    curve_path = destination / f"{prefix}_radius_theta_curves.png"
    curve_svg_path = destination / f"{prefix}_radius_theta_curves.svg"
    curve_pdf_path = destination / f"{prefix}_radius_theta_curves.pdf"
    curve_csv_path = destination / f"{prefix}_radius_theta_curves.csv"
    theta_path = destination / f"{prefix}_theta01_theta02_curves.png"
    theta_svg_path = destination / f"{prefix}_theta01_theta02_curves.svg"
    theta_pdf_path = destination / f"{prefix}_theta01_theta02_curves.pdf"
    theta_csv_path = destination / f"{prefix}_theta01_theta02_curves.csv"

    closed_b = np.vstack([b_curve, b_curve[0]])
    # B 始终位于 x=B_CenterX 的 YZ 平行平面，因此直接用 By-Bz 二维图表达；
    # 不再用三轴随时间曲线或倾斜三维视图掩盖真实的平面几何形状。
    spatial_figure, spatial_axis = plt.subplots(
        1, 1, figsize=(7.2, 6.4), constrained_layout=True
    )
    spatial_axis.plot(
        closed_b[:, 1], closed_b[:, 2],
        color="#26718C", linewidth=2.4, label="Direct B input trajectory",
    )
    ray_indices = np.linspace(0, b_curve.shape[0] - 1, min(8, b_curve.shape[0]), dtype=int)
    for index in ray_indices:
        ray = np.vstack([np.zeros(2), b_curve[index, 1:3]])
        spatial_axis.plot(
            ray[:, 0], ray[:, 1],
            color="#A7A7A7", linewidth=0.7, alpha=0.65,
        )
    spatial_axis.scatter(
        0.0, 0.0,
        color="#202020", s=42, label="A projection",
    )
    spatial_axis.scatter(
        b_curve[0, 1], b_curve[0, 2],
        color="#26718C", s=38, label="B at t/T = 0",
    )
    spatial_axis.set_aspect("equal", adjustable="box")
    spatial_axis.set_xlabel("$B_y$ (mm)")
    spatial_axis.set_ylabel("$B_z$ (mm)")
    spatial_axis.set_title(
        f"Direct periodic B input in the YZ plane (Bx={np.mean(b_curve[:, 0]):.2f} mm)"
    )
    spatial_axis.grid(True, color="#D8D8D8", linewidth=0.6)
    spatial_axis.spines["top"].set_visible(False)
    spatial_axis.spines["right"].set_visible(False)
    spatial_axis.legend(loc="upper right")
    spatial_figure.savefig(spatial_path, dpi=300, facecolor="white")

    input_figure, axes = plt.subplots(3, 1, figsize=(9.0, 7.2), sharex=True, constrained_layout=True)
    axes[0].plot(normalized_time, input_radius, color="#26718C", linewidth=1.8, label="|AB(t)|")
    axes[0].set_ylabel("radius (mm)")
    axes[0].legend(loc="best")
    series = (
        (axes[1], theta01, "theta01 (rad)", "#B05A1F"),
        (axes[2], theta02, "theta02 (rad)", "#6A4C93"),
    )
    for axis, values, ylabel, color in series:
        axis.plot(normalized_time, values, color=color, linewidth=1.8)
        axis.set_ylabel(ylabel)
    for axis in axes:
        axis.grid(True, color="#D8D8D8", linewidth=0.6)
    axes[0].set_title("Direct-B fourbar input over one motion cycle")
    axes[-1].set_xlabel("Normalized time, t/T")
    input_figure.savefig(input_path, dpi=300, facecolor="white")

    curve_figure, curve_axes = plt.subplots(
        3, 1, figsize=(7.2, 6.2), sharex=True, constrained_layout=True
    )
    curve_series = (
        (input_radius, r"$r_{in}$ (mm)", "#C46A32"),
        (theta01, r"$\theta_{01}$ (rad)", "#26718C"),
        (theta02, r"$\theta_{02}$ (rad)", "#6A4C93"),
    )
    for panel_index, (axis, (values, ylabel, color)) in enumerate(
        zip(curve_axes, curve_series)
    ):
        axis.plot(normalized_time, values, color=color, linewidth=1.8)
        axis.set_ylabel(ylabel)
        axis.grid(True, color="#D9D9D9", linewidth=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            0.01, 0.91, f"({chr(97 + panel_index)})",
            transform=axis.transAxes, fontsize=9.5, fontweight="bold",
            ha="left", va="top",
        )
        value_span = float(np.ptp(values))
        if value_span <= 1e-12:
            center = float(np.mean(values))
            padding = max(0.5, 0.03 * abs(center))
            axis.set_ylim(center - padding, center + padding)
        else:
            padding = 0.08 * value_span
            axis.set_ylim(float(np.min(values)) - padding, float(np.max(values)) + padding)
    curve_axes[0].set_title("Four-bar input parameters over one motion cycle")
    curve_axes[-1].set_xlabel("Normalized time, $t/T$")
    curve_figure.savefig(curve_path, dpi=450, bbox_inches="tight", facecolor="white")
    curve_figure.savefig(curve_svg_path, bbox_inches="tight", facecolor="white")
    curve_figure.savefig(curve_pdf_path, bbox_inches="tight", facecolor="white")

    curve_table = np.column_stack([normalized_time, input_radius, theta01, theta02])
    np.savetxt(
        curve_csv_path,
        curve_table,
        delimiter=",",
        header="normalized_time,input_radius_mm,theta01_rad,theta02_rad",
        comments="",
    )

    theta_figure, theta_axes = plt.subplots(
        2, 1, figsize=(7.2, 4.6), sharex=True, constrained_layout=True
    )
    theta_series = (
        (theta01, r"$\theta_{01}$ (rad)", "#26718C"),
        (theta02, r"$\theta_{02}$ (rad)", "#6A4C93"),
    )
    for panel_index, (axis, (values, ylabel, color)) in enumerate(
        zip(theta_axes, theta_series)
    ):
        axis.plot(normalized_time, values, color=color, linewidth=1.8)
        axis.set_ylabel(ylabel)
        axis.grid(True, color="#D9D9D9", linewidth=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            0.01, 0.91, f"({chr(97 + panel_index)})",
            transform=axis.transAxes, fontsize=9.5, fontweight="bold",
            ha="left", va="top",
        )
        value_span = float(np.ptp(values))
        padding = max(0.02, 0.08 * value_span)
        axis.set_ylim(float(np.min(values)) - padding, float(np.max(values)) + padding)
    theta_axes[0].set_title("Four-bar input orientation angles")
    theta_axes[-1].set_xlabel("Normalized time, $t/T$")
    theta_figure.savefig(theta_path, dpi=450, bbox_inches="tight", facecolor="white")
    theta_figure.savefig(theta_svg_path, bbox_inches="tight", facecolor="white")
    theta_figure.savefig(theta_pdf_path, bbox_inches="tight", facecolor="white")
    np.savetxt(
        theta_csv_path,
        np.column_stack([normalized_time, theta01, theta02]),
        delimiter=",",
        header="normalized_time,theta01_rad,theta02_rad",
        comments="",
    )

    table = np.column_stack([normalized_time, b_curve, input_radius, theta01, theta02])
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header=(
            "normalized_time,Bx_mm,By_mm,Bz_mm,AB_mm,theta01_rad,theta02_rad"
        ),
        comments="",
    )
    if show:
        plt.show(block=True)
    else:
        plt.close(spatial_figure)
        plt.close(input_figure)
        plt.close(curve_figure)
        plt.close(theta_figure)
    return {
        "B_spatial_curve_png": str(spatial_path.resolve()),
        "input_parameters_png": str(input_path.resolve()),
        "input_parameters_csv": str(csv_path.resolve()),
        "input_radius_theta_curves_png": str(curve_path.resolve()),
        "input_radius_theta_curves_svg": str(curve_svg_path.resolve()),
        "input_radius_theta_curves_pdf": str(curve_pdf_path.resolve()),
        "input_radius_theta_curves_csv": str(curve_csv_path.resolve()),
        "theta01_theta02_curves_png": str(theta_path.resolve()),
        "theta01_theta02_curves_svg": str(theta_svg_path.resolve()),
        "theta01_theta02_curves_pdf": str(theta_pdf_path.resolve()),
        "theta01_theta02_curves_csv": str(theta_csv_path.resolve()),
    }


def plot_b_input_diagnostics(
    b_curve: np.ndarray,
    ab_length: float,
    phase01: np.ndarray | None = None,
    output_dir: Path | str = CURRENT_DIR / "output",
    prefix: str = "mot_input",
    show: bool = True,
) -> Dict[str, str]:
    """直接 B 输入诊断入口；ab_length 仅保留调用兼容性。"""
    return plot_mot_input_diagnostics(
        b_curve,
        ab_length,
        phase01=phase01,
        output_dir=output_dir,
        prefix=prefix,
        show=show,
    )


# =============================================================================
# 5. 模型文件独立运行入口
# =============================================================================


def _compact_run_summary(
    mode: str,
    result: FourBarResult | None,
    metrics: Mapping[str, float],
    source: str,
) -> Dict[str, Any]:
    """只打印关键数值，避免在终端输出 76 帧完整数组。"""
    summary: Dict[str, Any] = {
        "mode": mode,
        "source": source,
        "valid": result is not None and bool(result.valid),
        "metrics_mm": {name: float(value) for name, value in metrics.items()},
    }
    if result is not None and result.valid:
        summary.update({
            "frames": int(result.tip.shape[0]),
            "wrist_node": "L",
            "tip_node": "U",
            "tip_frame0_mm": result.tip[0].tolist(),
            "wrist_frame0_mm": result.wrist[0].tolist(),
            "B_frame0_mm": result.b_curve[0].tolist(),
            "AB_range_mm": [float(np.min(result.input_radius)), float(np.max(result.input_radius))],
            "B_x_range_mm": [
                float(np.min(result.b_curve[:, 0])),
                float(np.max(result.b_curve[:, 0])),
            ],
            "B_y_range_mm": [
                float(np.min(result.b_curve[:, 1])),
                float(np.max(result.b_curve[:, 1])),
            ],
            "B_z_range_mm": [
                float(np.min(result.b_curve[:, 2])),
                float(np.max(result.b_curve[:, 2])),
            ],
        })
    return summary


def _main() -> None:
    """允许直接运行模型，也允许由优化文件 import。"""
    parser = argparse.ArgumentParser(description="Run or verify the consolidated fourbar model")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="NPZ checkpoint to recompute; default uses the current audited checkpoint.",
    )
    parser.add_argument(
        "--initial",
        action="store_true",
        help="Evaluate the original fourbar initial design vector instead of a checkpoint.",
    )
    parser.add_argument(
        "--plot-input",
        action="store_true",
        help="Plot direct B trajectory and AB/theta01/theta02 over time.",
    )
    parser.add_argument(
        "--plot-three-view",
        action="store_true",
        help="Plot front, top, and side orthographic views of one mechanism frame.",
    )
    parser.add_argument(
        "--plot-wrist-three-view",
        action="store_true",
        help="Plot local front, top, and side views of wrist nodes WMKLQPN.",
    )
    parser.add_argument(
        "--animate-three-view",
        action="store_true",
        help="Export synchronized front, top, and side motion as MP4 and GIF.",
    )
    parser.add_argument(
        "--plot-angles",
        action="store_true",
        help="Plot theta_wrist, theta20, and theta21 over normalized time.",
    )
    parser.add_argument(
        "--plot-theta-m",
        action="store_true",
        help="Plot thetaM over normalized time.",
    )
    parser.add_argument(
        "--interactive-3d",
        action="store_true",
        help="Export a rotatable Three.js mechanism view with a phase slider.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second for --animate-three-view; default is 15.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Zero-based frame index for the orthographic views; default is 0.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save input plots without opening interactive windows.",
    )
    args = parser.parse_args()
    data = load_problem_data()

    if args.initial:
        # 原始起点不读取历史结果；若它不满足当前机构约束，valid 会明确显示为 false。
        space = build_design_space(data)
        result, metrics, state = evaluate_design_vector(space.x0, data, space, check_smooth=False)
        summary = _compact_run_summary("original_initial", result, metrics, space.seed_name)
    else:
        default_checkpoint = CURRENT_DIR / "output" / "strict_feasible_from_initial_checkpoint.npz"
        checkpoint = Path(args.checkpoint) if args.checkpoint else default_checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}. Use --initial to evaluate the original parameters."
            )
        state = load_checkpoint_state(checkpoint)
        result, metrics = evaluate_design_state(state, data, check_smooth=False)
        summary = _compact_run_summary("checkpoint", result, metrics, str(checkpoint.resolve()))

    if args.plot_input:
        summary["B_input_outputs"] = plot_b_input_diagnostics(
            state.b_curve,
            0.0,
            phase01=data.phase,
            output_dir=CURRENT_DIR / "output",
            prefix="b_input",
            show=not args.no_show,
        )

    if args.plot_three_view:
        if result is None:
            raise FourBarError("cannot plot three views because the selected design is invalid")
        summary["three_view_outputs"] = plot_initial_three_views(
            result,
            frame=args.frame,
            output_dir=CURRENT_DIR / "output",
            prefix="fourbar_initial_three_view" if args.frame == 0 else f"fourbar_frame_{args.frame:03d}_three_view",
            show=not args.no_show,
        )

    if args.plot_wrist_three_view:
        if result is None:
            raise FourBarError("cannot plot wrist views because the selected design is invalid")
        summary["wrist_three_view_outputs"] = plot_wrist_three_views(
            result,
            frame=args.frame,
            output_dir=CURRENT_DIR / "output",
            prefix=(
                "fourbar_initial_wrist_three_view"
                if args.frame == 0 else f"fourbar_frame_{args.frame:03d}_wrist_three_view"
            ),
            show=not args.no_show,
        )

    if args.animate_three_view:
        if result is None:
            raise FourBarError("cannot animate three views because the selected design is invalid")
        summary["three_view_motion_outputs"] = animate_three_views(
            result,
            output_dir=CURRENT_DIR / "output",
            prefix="fourbar_initial_three_view_motion" if args.initial else "fourbar_three_view_motion",
            fps=args.fps,
        )

    if args.plot_angles:
        if result is None:
            raise FourBarError("cannot plot angles because the selected design is invalid")
        summary["internal_angle_outputs"] = plot_internal_angle_curves(
            result,
            phase01=data.phase,
            output_dir=CURRENT_DIR / "output",
            show=not args.no_show,
        )

    if args.plot_theta_m:
        if result is None:
            raise FourBarError("cannot plot thetaM because the selected design is invalid")
        summary["thetaM_outputs"] = plot_theta_m_curve(
            result,
            phase01=data.phase,
            output_dir=CURRENT_DIR / "output",
            show=not args.no_show,
        )

    if args.interactive_3d:
        if result is None:
            raise FourBarError("cannot export interactive 3D because the selected design is invalid")
        summary["interactive_3d_outputs"] = export_interactive_threejs(
            result,
            output_dir=CURRENT_DIR / "output",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    _main()
