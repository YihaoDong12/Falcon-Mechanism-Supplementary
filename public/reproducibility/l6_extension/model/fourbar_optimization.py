from __future__ import annotations

"""Fourbar 优化与区域分析的统一入口。

对外只使用 run_fourbar_optimization()：

1. task="optimize"：从原始 fourbar 初始参数出发，执行灵敏度分区、区域
   CMA-ES、全独立变量直接搜索和区域内 SLSQP 精修。当前正式模型使用固定
   L2/L5/L6/L7/L31、逐帧派生 L32(t)=L3(t)-L31+2、三阶 L3(t)/L8(t)，并令 C 与 Z 始终重合。
2. task="region_analysis"：冻结 Mot 输入和目标旋转；L6 使用静态杆长，L7
   折算为当前周期均值，再分析静态几何、固定 L7、目标平移和目标尺度。
3. 每次完整优化建立独立版本归档，保存全部候选、分区、CMA-ES
   分布/选择、SLSQP 精修和最优轨迹快照；可用
   load_optimization_history() 直接读取作图，不重跑机构或优化器。

该文件不依赖其他 optimize_*.py。机构计算全部来自 fourbar3d_python.py。
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from fourbar3d_python import (
    ACTIVE_STATIC_NAMES,
    CURRENT_L32_DERIVED_MODE,
    L12_PERIODIC_L32_DERIVED_MODE,
    CURRENT_DIR,
    DesignSpace,
    DesignState,
    DECOUPLED_CONSTRAINED_TARGET_POSE_NAMES,
    DECOUPLED_FIXED_RZ_SCALE_TARGET_POSE_NAMES,
    DECOUPLED_TIP_RY_SCALE_TARGET_POSE_NAMES,
    FourBarError,
    B_FOURIER_Z3_C0_NAMES,
    B_FOURIER_XYZ3_NAMES,
    CLOSURE_ANGLE_MAX_RAD,
    CLOSURE_ANGLE_MIN_RAD,
    L31_FOURIER_NAMES,
    L3_FOURIER_NAMES,
    L5_FOURIER_NAMES,
    L6_FOURIER_NAMES,
    L6_FOURIER3_NAMES,
    L7_FOURIER_NAMES,
    L8_FOURIER_NAMES,
    ZC_FOURIER_NAMES,
    ZC_SPLIT_NAMES,
    MOT_POLAR_NAMES,
    STATIC_NAMES,
    TARGET_POSE_NAMES,
    ProblemData,
    apply_target_pose,
    build_design_space,
    combined_rmse,
    coordinate_error_metrics,
    decode_design_vector,
    evaluate_design_state,
    evaluate_design_vector,
    load_checkpoint_state,
    load_problem_data,
    motcurve_to_b_input,
    motcurve_to_input_angles,
    objective_from_metrics,
    periodic_coefficient_count,
    params_from_static,
    qa2,
    qa2_required_clearance_mm,
    static_bounds,
    t_angles,
    target_pose_center,
    triangle_third_side_non_adjacent,
)

__all__ = ["run_fourbar_optimization", "load_optimization_history"]

_L5_L8_ZC_SPLIT_MODES = {
    "l5_l8_zc_split_periodic3",
    "l5_l6_l8_zc_split_periodic3",
}
_CURRENT_PERIODIC_MODE = L12_PERIODIC_L32_DERIVED_MODE


# =============================================================================
# 1. 求解器配置和内部状态
# =============================================================================
# _OptimizeConfig 控制当前完整优化；_RegionConfig 只控制固定杆区域分析。


@dataclass
class _OptimizeConfig:
    """完整优化的停止条件、种群、分区和局部精修参数。"""

    # 正式优化默认不设置墙钟时限；仅在调用方显式传入正数时启用临时限时试验。
    minutes: float | None = None
    seed: int = 20260839
    max_rounds: int = 10_000
    max_evaluations: int = 100_000_000
    workers: int = 8
    # 每个 CMA-ES 种群使用全部逻辑处理器并行评估；候选顺序由 map 保持稳定。
    parallel_backend: str = "process"
    popsize: int = 192
    generations_per_region: int = 10
    regions_per_round: int = 12
    # 沿用昨晚正式无锚点任务的广域初始尺度。
    sigma0: float = 0.040
    max_depth: int = 16
    split_every_rounds: int = 1
    splits_per_event: int = 12
    sqp_maxiter: int = 600
    sqp_ftol: float = 1e-7
    # SQP 不再使用 RMSE 门槛，而是按阶段定期精修局部灵敏度最高的变量。
    # 按既定规则，粗差超过 100 mm 时不调用局部 SQP。
    sqp_trigger_rmse_mm: float = 140.0
    sqp_interval_rounds: int = 1
    sqp_active_dimensions: int = 54
    sqp_fd_step: float = 5.0e-4
    sqp_trust_radius: float = 0.035
    # 局部精修可在不改机构模型的前提下切换；本轮正式只使用 SLSQP。
    refinement_method: str = "slsqp"
    pso_particles: int = 48
    pso_iterations: int = 24
    pso_interval_rounds: int = 4
    pso_active_dimensions: int = 24
    pso_fd_step: float = 2e-3
    pso_trust_radius: float = 0.06
    pso_inertia_start: float = 0.78
    pso_inertia_end: float = 0.38
    pso_cognitive: float = 1.55
    pso_social: float = 1.75
    pso_velocity_fraction: float = 0.20
    pso_stall_iterations: int = 7
    pso_restart_fraction: float = 0.15
    exploration: float = 0.55
    minimum_static_mm: float = 10.0
    prefix: str = "fourbar_baseline_matched_coldstart"
    record_full_history: bool = True
    record_candidate_events: bool = False
    event_flush_interval: int = 256
    # 本轮将 Tip/Wrist 同时小于阈值作为真正成功条件。
    objective_mode: str = "initialized_equal_arc_bottleneck_p64_with_stutter"
    # 与基准一致，使用高阶 p 范数逼近 Tip/Wrist 两项 RMSE 的最大值。
    stage3_objective_mode: str = "initialized_equal_arc_bottleneck_p64_with_stutter"
    target_rmse_mm: float = 25.0
    # Screening runs continue to the wall-clock deadline after first hitting the target.
    continue_after_target: bool = False
    # Tip/Wrist 共用三轴平移和三轴旋转，目标尺度固定为 1。
    target_pose_mode: str = "shared_rotation_translation6"
    # 当前拓扑：固定 L2/L5/L6/L7/L31，逐帧派生 L32(t)=L3(t)-L31+2，
    # 仅 L3/L8 使用三阶 Fourier，且 Z=C。
    periodic_length_mode: str = _CURRENT_PERIODIC_MODE
    # 恢复 RMSE29.99 基准 B 点：Bx 为常数，By 二阶、Bz 三阶且 C0z=0。
    b_curve_mode: str = "fourier_z3_c0"
    # 全 54 维正式边界；其中 Target_Ty_mm 严格保持 [-100,200] mm。
    static_bound_mode: str = "broad_all54_20260802"
    # 每次正式优化先校验这份机读合同及其冻结源文件，防止模型、目标和报告定义分叉。
    input_contract_path: str = str(
        CURRENT_DIR / "input" / "fourbar_initialized_equal_arc_optimization_input_v2.json"
    )
    # 显式锁定新增 Excel 的初始化事件和 76 点等弧长目标；旧 TXT 仅保留兼容入口。
    target_initialized_csv_path: str = str(
        CURRENT_DIR / "input" / "Length_normalized_target_initialized_equal_arc_76_mm.csv"
    )
    target_initialization_metadata_path: str = str(
        CURRENT_DIR / "input" / "Length_normalized_target_initialized_equal_arc_76_metadata.json"
    )
    target_tip_txt_path: str = ""
    target_wrist_txt_path: str = ""
    # 仅用于同一正式优化任务的累计续跑。不得指向历史 PSO/CMA 结果；
    # 空字符串表示严格从原始 Fourbar 初值重新进行无锚点搜索。
    start_checkpoint_path: str = ""
    # 当同一任务把固定杆提升为 Fourier 周期杆时，记录严格的嵌套空间映射。
    # 该字段由续跑加载器填写，不接受命令行输入，也不会改变原始边界。
    continuation_space_mapping: dict[str, Any] = field(default_factory=dict)
    # 三阶段仅按 RMSE 质量门槛切换；以下两个 fraction 字段只为旧配置兼容，
    # 不再参与阶段判定。
    three_stage_schedule: bool = False
    stage1_end_fraction: float = 0.30
    stage2_end_fraction: float = 0.72
    # 第一阶段只调整能够改变 Wrist=L 的上游变量；达到该 Wrist 门槛后，
    # 第二阶段再打开只影响 Tip 的下游变量。
    stage1_advance_rmse_mm: float = 105.0
    stage2_advance_rmse_mm: float = 110.0
    stage1_sigma0: float = 0.22
    stage2_sigma0: float = 0.12
    stage1_objective_mode: str = "wrist_curriculum_direct"
    stage1_search_scope: str = "wrist_upstream"
    stage2_objective_mode: str = ""
    stage2_search_scope: str = "tip_downstream"
    reseed_split_children: bool = False
    # 联合阶段可按“上游、下游、全变量”循环搜索，避免 62 维同步扰动稀释有效样本。
    stage3_block_cycle: bool = False
    # 广域联合模式让调用方设置的种群、区域数和树深真正进入关键第三阶段。
    stage3_broad_search: bool = False
    # 双曲线进入深层盆地后，可把当前最优点放回一个全设计域叶区，
    # 清空旧叶边界/协方差并集中全部 CMA-ES 与 SQP 预算继续精修。
    stage3_deep_consolidation: bool = False
    stage3_deep_threshold_mm: float = 35.0
    stage3_deep_sigma0: float = 0.008
    search_scope: str = "all"
    wrist_phase_guidance: bool = False
    # 区域对角 CMA-ES 的内部超参数。显式化后可进行可复现的超参数筛选。
    cma_elite_fraction: float = 0.35
    cma_covariance_learning_rate: float = 0.18
    cma_variance_floor: float = 1e-8
    cma_normalize_covariance_by_sigma: bool = True
    # 完整协方差允许静态几何、B 输入和周期杆之间学习协同可行方向。
    cma_full_covariance: bool = True
    # 完整协方差模式使用标准 CMA-ES 的两条进化路径。
    cma_use_evolution_paths: bool = True
    cma_sigma_min: float = 0.00025
    cma_sigma_max: float = 0.18
    cma_sigma_no_valid_factor: float = 0.65
    cma_sigma_expand_factor: float = 1.12
    # 中等可行率或短暂停滞时保持尺度，只有低可行率才收缩；避免高维搜索
    # 每代固定缩小 4% 而过早塌缩到单一局部盆地。
    cma_sigma_default_factor: float = 1.00
    cma_low_valid_fraction: float = 0.12
    cma_high_valid_fraction: float = 0.45
    cma_stagnation_generations: int = 12
    cma_restart_sigma_factor: float = 2.0
    # RMSE 29.99 基准按外层轮次判断停滞：连续 3 轮无改进后扩大重启尺度。
    # 该模式关闭叶区内部按代数触发的重启，避免同一次停滞被重复计算。
    cma_restart_mode: str = "outer_round_baseline"
    cma_outer_stagnation_rounds: int = 3
    cma_outer_min_improvement_mm: float = 0.15
    cma_outer_min_improvement_fraction: float = 0.0015
    cma_boundary_trigger_fraction: float = 0.03
    cma_boundary_stagnation_rounds: int = 2
    cma_boundary_restart_center_blend: float = 0.40
    cma_restart_growth_factor: float = 1.35
    cma_restart_factor_max: float = 4.0
    # 每代保留少量按物理参数块生成的远距候选和边界候选。若候选不可行，
    # 仍沿当前区域的已知可行点逐级回退，不绕过 fourbar 几何检验。
    cma_local_block_fraction: float = 0.18
    cma_global_injection_fraction: float = 0.15
    cma_boundary_injection_fraction: float = 0.08
    # 先以原始 fourbar 起点构造分散的可行池，再允许区域树切分。
    initial_pool_size: int = 1024
    initial_pool_max_proposals: int = 24000
    initial_pool_min_distance: float = 0.006
    minimum_samples_before_split: int = 240
    minimum_leaf_evaluations: int = 160
    # 灵敏度分区的最小区域宽度、探针步长和切分保护比例。
    sensitivity_min_region_width: float = 0.018
    sensitivity_probe_step: float = 0.020
    sensitivity_probe_width_fraction: float = 0.35
    split_guard_fraction: float = 0.06


@dataclass
class _RegionConfig:
    """固定杆灵敏度步长和窄区间判据。"""

    delta_fraction: float = 0.005
    narrow_tolerance_mm: float = 5.0
    narrow_bisection_steps: int = 16
    prefix: str = "fixed_link_region_analysis"


@dataclass
class _CampaignConfig:
    """可恢复长时搜索的总预算和独立阶段设置。"""

    hours: float = 8.0
    reserve_minutes: float = 30.0
    phase_minutes: float = 27.0
    max_phases: int = 20
    seed: int = 20260719
    target_rmse_mm: float = 25.0
    prefix: str = "partition_cmaes_sqp_rmse25_campaign"


def _reported_optimizer_config(config: _OptimizeConfig) -> dict[str, Any]:
    """只输出本轮实际启用的求解器超参数，避免未启用算法混入正式报告。"""

    payload = asdict(config)
    if config.refinement_method != "pso":
        for name in tuple(payload):
            if name.startswith("pso_"):
                payload.pop(name)
    if config.refinement_method != "slsqp":
        for name in tuple(payload):
            if name.startswith("sqp_"):
                payload.pop(name)
    return payload


def _three_stage_config(
    base: _OptimizeConfig,
    elapsed_fraction: float,
    maximum_curve_rmse_mm: float = math.inf,
    tip_rmse_mm: float | None = None,
    wrist_rmse_mm: float | None = None,
    minimum_stage_rank: int = 1,
) -> tuple[str, _OptimizeConfig]:
    """按当前双曲线 RMSE 质量门槛切换阶段，不用墙钟比例强制降档。"""

    if not base.three_stage_schedule:
        return "single_stage", base
    maximum_rmse = float(maximum_curve_rmse_mm)
    tip_rmse = maximum_rmse if tip_rmse_mm is None else float(tip_rmse_mm)
    wrist_rmse = maximum_rmse if wrist_rmse_mm is None else float(wrist_rmse_mm)
    local_block_base = float(np.clip(base.cma_local_block_fraction, 0.0, 0.70))
    # 第一阶段聚焦 Wrist 上游，第二阶段聚焦只影响 Tip 的下游，
    # 最后才恢复全部变量联合收敛。
    proposed_rank = (
        1
        if wrist_rmse > float(base.stage1_advance_rmse_mm)
        else 2
        if tip_rmse > float(base.stage2_advance_rmse_mm)
        else 3
    )
    stage_rank = max(int(np.clip(minimum_stage_rank, 1, 3)), proposed_rank)
    if stage_rank == 1:
        return "stage1_wrist_upstream", replace(
            base,
            popsize=64,
            generations_per_region=10,
            regions_per_round=4,
            sigma0=float(base.stage1_sigma0),
            exploration=0.46,
            max_depth=6,
            split_every_rounds=6,
            splits_per_event=2,
            cma_local_block_fraction=min(0.70, 2.0 * local_block_base),
            sqp_maxiter=40,
            sqp_interval_rounds=8,
            sqp_active_dimensions=28,
            sqp_trust_radius=0.050,
            sqp_ftol=1e-6,
            objective_mode=(
                "wrist_curriculum"
                if base.wrist_phase_guidance
                else base.stage1_objective_mode
            ),
            search_scope=base.stage1_search_scope,
        )
    if stage_rank == 2:
        return "stage2_tip_downstream", replace(
            base,
            popsize=48,
            generations_per_region=10,
            # 叶区形成后每轮只推进当前优先区域；全域/边界注入仍负责跨区探索。
            # 这避免在已证实较差的叶区上重复消耗一半以上的模型评估。
            regions_per_round=1,
            sigma0=float(base.stage2_sigma0),
            exploration=0.28,
            max_depth=7,
            split_every_rounds=5,
            splits_per_event=2,
            cma_local_block_fraction=min(0.70, (5.0 / 3.0) * local_block_base),
            sqp_maxiter=100,
            sqp_interval_rounds=3,
            sqp_active_dimensions=44,
            sqp_trust_radius=0.035,
            sqp_ftol=1e-8,
            objective_mode=(
                base.stage2_objective_mode
                or f"tip_curriculum_w{float(base.stage1_advance_rmse_mm):g}"
            ),
            search_scope=base.stage2_search_scope,
        )
    if base.stage3_broad_search:
        # 广域阶段先覆盖多个叶区；一旦双曲线瓶颈进入 50 mm，再把主要预算
        # 集中到高质量叶区。这样仍保留 UCB 探索项和周期性分裂，但不会在已经
        # 找到强盆地后继续用同样比例反复访问明显较差的区域。
        focused = maximum_rmse <= 50.0
        deeply_focused = maximum_rmse <= float(base.stage3_deep_threshold_mm)
        regions_per_round = (
            1
            if deeply_focused and base.stage3_deep_consolidation
            else 2
            if deeply_focused
            else 3
            if focused
            else max(5, int(base.regions_per_round))
        )
        exploration = (
            0.02
            if deeply_focused and base.stage3_deep_consolidation
            else 0.04
            if deeply_focused
            else 0.08
            if focused
            else float(np.clip(base.exploration, 0.18, 0.35))
        )
        split_every_rounds = 10 if deeply_focused else 8 if focused else 2
        splits_per_event = (
            1
            if focused
            else max(2, min(4, int(base.splits_per_event)))
        )
        return "stage3_joint_refinement", replace(
            base,
            popsize=max(64, int(base.popsize)),
            generations_per_region=max(12, int(base.generations_per_region)),
            regions_per_round=regions_per_round,
            sigma0=(
                float(
                    np.clip(
                        base.stage3_deep_sigma0,
                        base.cma_sigma_min,
                        0.12,
                    )
                )
                if deeply_focused and base.stage3_deep_consolidation
                else float(np.clip(base.sigma0, 0.01, 0.12))
            ),
            exploration=exploration,
            max_depth=(
                0
                if deeply_focused and base.stage3_deep_consolidation
                else max(9, int(base.max_depth))
            ),
            split_every_rounds=(
                1_000_000
                if deeply_focused and base.stage3_deep_consolidation
                else split_every_rounds
            ),
            splits_per_event=splits_per_event,
            cma_local_block_fraction=min(0.70, (4.0 / 3.0) * local_block_base),
            sqp_maxiter=max(180, int(base.sqp_maxiter)),
            sqp_interval_rounds=max(2, int(base.sqp_interval_rounds)),
            sqp_active_dimensions=max(56, int(base.sqp_active_dimensions)),
            sqp_trust_radius=max(0.025, float(base.sqp_trust_radius)),
            sqp_ftol=1e-9,
            objective_mode=base.stage3_objective_mode,
            search_scope="all",
        )
    return "stage3_joint_refinement", replace(
        base,
        popsize=40,
        generations_per_region=12,
        regions_per_round=3,
        sigma0=0.05,
        exploration=0.10,
        max_depth=7,
        split_every_rounds=8,
        splits_per_event=1,
        cma_local_block_fraction=min(0.70, (4.0 / 3.0) * local_block_base),
        sqp_maxiter=180,
        sqp_interval_rounds=1,
        sqp_active_dimensions=56,
        sqp_trust_radius=0.020,
        sqp_ftol=1e-9,
        objective_mode=base.stage3_objective_mode,
        search_scope="all",
    )


def _stage3_block_config(
    base: _OptimizeConfig,
    cycle_index: int,
) -> tuple[str, _OptimizeConfig]:
    """把联合阶段拆成结构上可解释的三轮循环，同时保留周期性的全维耦合搜索。"""

    phase = int(cycle_index) % 3
    if phase == 0:
        return "wrist_upstream", replace(
            base,
            search_scope="wrist_upstream",
            objective_mode="rmse_bottleneck_p32",
            popsize=max(48, int(base.popsize)),
            generations_per_region=10,
            regions_per_round=1,
            sigma0=max(0.07, float(base.sigma0)),
            sqp_active_dimensions=32,
            sqp_trust_radius=max(0.035, float(base.sqp_trust_radius)),
        )
    if phase == 1:
        return "tip_downstream", replace(
            base,
            search_scope="tip_downstream",
            objective_mode="rmse_bottleneck_p32",
            popsize=max(48, int(base.popsize)),
            generations_per_region=10,
            regions_per_round=1,
            sigma0=max(0.06, float(base.sigma0)),
            sqp_active_dimensions=34,
            sqp_trust_radius=max(0.035, float(base.sqp_trust_radius)),
        )
    return "joint", replace(
        base,
        search_scope="all",
        objective_mode="rmse_bottleneck_p32",
        popsize=max(56, int(base.popsize)),
        generations_per_region=12,
        regions_per_round=1,
        sigma0=max(0.05, float(base.sigma0)),
        sqp_active_dimensions=max(56, int(base.sqp_active_dimensions)),
        sqp_trust_radius=max(0.025, float(base.sqp_trust_radius)),
    )


@dataclass
class _Candidate:
    """一个已计算的候选机构，包括物理量、归一化量和误差指标。"""

    y: np.ndarray
    x: np.ndarray
    score: float
    metrics: dict[str, float]
    valid: bool
    region_id: int
    stage: str
    requested_y: np.ndarray
    evaluation_id: int = -1
    snapshot: dict[str, Any] | None = field(default=None, repr=False)


@dataclass
class _Region:
    """LA-MCTS 风格叶区域；边界均位于归一化空间 [0,1]^n。"""

    region_id: int
    lo: np.ndarray
    hi: np.ndarray
    depth: int = 0
    visits: int = 0
    evaluations_since_creation: int = 0
    best_score: float = math.inf
    prior_score: float = math.inf
    best_y: np.ndarray | None = None
    samples: list[_Candidate] = field(default_factory=list)
    # 保存区域内 CMA-ES 状态，避免每次重新访问区域时丢失已学习的搜索尺度。
    cma_mean: np.ndarray | None = None
    cma_diagonal: np.ndarray | None = None
    cma_covariance: np.ndarray | None = None
    cma_sigma: float | None = None
    cma_path_c: np.ndarray | None = None
    cma_path_sigma: np.ndarray | None = None
    cma_generation_count: int = 0
    cma_stale_generations: int = 0

    def width(self) -> np.ndarray:
        return np.maximum(self.hi - self.lo, 1e-12)

    def center(self) -> np.ndarray:
        return 0.5 * (self.lo + self.hi)


@dataclass
class _AnalysisContext:
    """区域分析所需的冻结参数、29 个活动变量及其绝对上下界。"""

    data: ProblemData
    source: DesignState
    x0: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    names: tuple[str, ...]
    fixed_target_rotation: np.ndarray


# =============================================================================
# 2. 通用输入输出和变量坐标转换
# =============================================================================


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist JSON without terminating a run on a transient sync-file lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    last_error: OSError | None = None
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        for attempt in range(12):
            try:
                os.replace(temporary, path)
                return
            except OSError as exc:
                if not isinstance(exc, PermissionError) and getattr(
                    exc, "winerror", None
                ) not in {5, 32}:
                    raise
                last_error = exc
                time.sleep(min(0.05 * (2**attempt), 0.8))

        deferred = path.with_name(
            f"{path.stem}.deferred.{time.time_ns()}{path.suffix}"
        )
        try:
            os.replace(temporary, deferred)
        except OSError as exc:
            if not isinstance(exc, PermissionError) and getattr(
                exc, "winerror", None
            ) not in {5, 32}:
                raise
            last_error = exc
        if temporary.exists():
            # Keep the complete unique temporary file when even the fallback
            # rename is briefly blocked. A later canonical write can succeed.
            return
    finally:
        if temporary.exists() and last_error is None:
            try:
                temporary.unlink()
            except OSError:
                pass


_METRIC_NAMES = (
    "combined",
    "tip",
    "wrist",
    "tip_peak",
    "wrist_peak",
    "tip_high_harmonic_rms",
    "wrist_high_harmonic_rms",
    "wrist_plane_mismatch_deg",
    "stutter_index",
    "stutter_penalty_mm",
    "tip_minimum_speed_ratio",
    "wrist_minimum_speed_ratio",
    "tip_maximum_adjacent_speed_ratio",
    "wrist_maximum_adjacent_speed_ratio",
)


def _high_harmonic_rms(curve: np.ndarray, retained_harmonic: int = 2) -> float:
    """返回不能由二阶目标曲线表示的高阶轨迹分量 RMS。"""

    values = np.asarray(curve, dtype=float)
    spectrum = np.fft.rfft(values - np.mean(values, axis=0), axis=0)
    spectrum[: retained_harmonic + 1, :] = 0.0
    residual = np.fft.irfft(spectrum, n=values.shape[0], axis=0)
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def _plane_normal_mismatch_deg(first: np.ndarray, second: np.ndarray) -> float:
    """返回两条三维闭合曲线 PCA 最小方差法向量之间的无向夹角。"""

    normals: list[np.ndarray] = []
    for curve in (first, second):
        centered = np.asarray(curve, dtype=float) - np.mean(curve, axis=0)
        covariance = centered.T @ centered / max(1, centered.shape[0])
        _, eigenvectors = np.linalg.eigh(covariance)
        normal = np.asarray(eigenvectors[:, 0], dtype=float)
        normals.append(normal / max(float(np.linalg.norm(normal)), 1e-12))
    cosine = float(np.clip(abs(np.dot(normals[0], normals[1])), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _best_forward_cyclic_alignment(
    generated: np.ndarray,
    target: np.ndarray,
) -> tuple[float, int]:
    """返回仅允许正向时间的最佳循环对齐误差及有符号偏移步数。

    该量只保留用于旧课程模式的诊断。正式 RMSE 与成功判据使用每个
    机构点到周期目标折线的单向最近距离，不绑定目标相位索引。
    """

    generated_centered = np.asarray(generated, dtype=float) - np.mean(
        generated, axis=0
    )
    target_centered = np.asarray(target, dtype=float) - np.mean(target, axis=0)
    count = generated_centered.shape[0]
    best_rmse = math.inf
    best_shift = 0
    for shift in range(count):
        shifted = np.roll(target_centered, -shift, axis=0)
        residual = generated_centered - shifted
        rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        if rmse < best_rmse:
            best_rmse = rmse
            best_shift = shift
    signed_shift = best_shift if best_shift <= count // 2 else best_shift - count
    return best_rmse, int(signed_shift)


def _optimization_objective(
    metrics: Mapping[str, float],
    mode: str = "balanced_rmse_target",
    target_rmse_mm: float = 30.0,
) -> float:
    """把用户的双 RMSE 硬目标转换为可连续搜索的标量目标。

    ``balanced_rmse_target`` 先最小化 Tip/Wrist 超出阈值的二维距离，
    再用较小权重继续压低较差的一条曲线、combined RMSE 和峰值。
    这样不会用 Wrist 的改善抵消 Tip 的大误差。
    """

    if mode.startswith("initialized_equal_arc_bottleneck_p"):
        suffix = mode.removeprefix("initialized_equal_arc_bottleneck_p")
        power_text = suffix.split("_", 1)[0]
        power = int(power_text)
        tip = float(metrics["tip"])
        wrist = float(metrics["wrist"])
        maximum = max(tip, wrist, 1e-12)
        bottleneck = maximum * (
            0.5 * ((tip / maximum) ** power + (wrist / maximum) ** power)
        ) ** (1.0 / power)
        return float(bottleneck + float(metrics.get("stutter_penalty_mm", 0.0)))
    if mode == "combined_peak":
        return objective_from_metrics(metrics)
    if mode == "equal_weight_rmse":
        # combined = sqrt((RMSE_tip^2 + RMSE_wrist^2) / 2)，
        # 两条曲线严格等权，不附加峰值、频谱或单侧引导项。
        return float(metrics["combined"])
    if mode == "wrist_curriculum":
        tip = float(metrics["tip"])
        wrist = float(metrics["wrist"])
        aligned_wrist = float(
            metrics.get("wrist_cyclic_aligned_rmse", wrist)
        )
        phase_offset = abs(float(metrics.get("wrist_phase_offset_steps", 0.0)))
        wrist_peak = float(metrics["wrist_peak"])
        wrist_harmonic = float(metrics.get("wrist_high_harmonic_rms", 0.0))
        plane_mismatch = float(metrics.get("wrist_plane_mismatch_deg", 0.0))
        return float(
            0.70 * wrist
            + 0.30 * aligned_wrist
            + 2.0 * phase_offset
            + 0.03 * tip
            + 0.001 * wrist_peak
            + 0.02 * wrist_harmonic
            + 0.03 * plane_mismatch
        )
    if mode == "wrist_curriculum_direct":
        tip = float(metrics["tip"])
        wrist = float(metrics["wrist"])
        wrist_peak = float(metrics["wrist_peak"])
        wrist_harmonic = float(metrics.get("wrist_high_harmonic_rms", 0.0))
        plane_mismatch = float(metrics.get("wrist_plane_mismatch_deg", 0.0))
        return float(
            wrist
            + 0.03 * tip
            + 0.001 * wrist_peak
            + 0.02 * wrist_harmonic
            + 0.03 * plane_mismatch
        )
    if mode == "wrist_rmse_only":
        return float(metrics["wrist"])
    if mode == "tip_rmse_only":
        return float(metrics["tip"])
    if mode == "wrist_tip_compatibility" or mode.startswith(
        "wrist_tip_compatibility_w"
    ):
        wrist = float(metrics["wrist"])
        tip_weight = 0.40
        if mode.startswith("wrist_tip_compatibility_w"):
            try:
                tip_weight = float(mode.rsplit("_w", 1)[1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid Wrist/Tip compatibility weight in objective mode: {mode}"
                ) from exc
        scaled_tip = tip_weight * float(metrics["tip"])
        maximum = max(wrist, scaled_tip, 1e-12)
        return float(
            maximum
            * (
                0.5
                * (
                    (wrist / maximum) ** 8
                    + (scaled_tip / maximum) ** 8
                )
            )
            ** (1.0 / 8.0)
        )
    if mode == "tip_curriculum" or mode.startswith("tip_curriculum_w"):
        tip = float(metrics["tip"])
        wrist = float(metrics["wrist"])
        wrist_limit = 105.0
        if mode.startswith("tip_curriculum_w"):
            try:
                wrist_limit = float(mode.rsplit("_w", 1)[1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid Tip-curriculum Wrist limit in objective mode: {mode}"
                ) from exc
        peak = max(float(metrics["tip_peak"]), float(metrics["wrist_peak"]))
        high_harmonic = max(
            float(metrics.get("tip_high_harmonic_rms", 0.0)),
            float(metrics.get("wrist_high_harmonic_rms", 0.0)),
        )
        # 第二阶段的下游变量不改变 Wrist 几何，因此将 Wrist=105 mm 作为
        # 软保护栏，只利用这些变量压低 Tip，避免共享尺度把已获得的 Wrist
        # 解重新推离上游阶段的质量门槛。
        return float(
            tip
            + 25.0 * max(0.0, wrist - wrist_limit)
            + 0.10 * wrist
            + 0.001 * peak
            + 0.02 * high_harmonic
        )
    if mode.startswith("smooth_bottleneck_p"):
        power = int(mode.rsplit("p", 1)[1])
        tip = float(metrics["tip"])
        wrist = float(metrics["wrist"])
        maximum = max(tip, wrist, 1e-12)
        smooth_maximum = maximum * (
            0.5 * ((tip / maximum) ** power + (wrist / maximum) ** power)
        ) ** (1.0 / power)
        peak = max(float(metrics["tip_peak"]), float(metrics["wrist_peak"]))
        high_harmonic = max(
            float(metrics.get("tip_high_harmonic_rms", 0.0)),
            float(metrics.get("wrist_high_harmonic_rms", 0.0)),
        )
        plane_mismatch = float(metrics.get("wrist_plane_mismatch_deg", 0.0))
        spectral_weight = 0.10 if power <= 8 else 0.02
        plane_weight = 0.15 if power <= 8 else 0.03
        return float(
            smooth_maximum
            + 0.05 * float(metrics["combined"])
            + 0.001 * peak
            + spectral_weight * high_harmonic
            + plane_weight * plane_mismatch
        )
    if mode.startswith("rmse_bottleneck_p"):
        power = int(mode.rsplit("p", 1)[1])
        tip = float(metrics["tip"])
        wrist = float(metrics["wrist"])
        maximum = max(tip, wrist, 1e-12)
        # 只平滑逼近 max(RMSE_tip, RMSE_wrist)，不混入峰值、频谱或平面角度。
        return float(
            maximum
            * (
                0.5
                * (
                    (tip / maximum) ** power
                    + (wrist / maximum) ** power
                )
            )
            ** (1.0 / power)
        )
    if mode != "balanced_rmse_target":
        raise ValueError(f"Unsupported objective_mode: {mode}")
    tip = float(metrics["tip"])
    wrist = float(metrics["wrist"])
    target = float(target_rmse_mm)
    exceedance = math.hypot(max(0.0, tip - target), max(0.0, wrist - target))
    peak = max(float(metrics["tip_peak"]), float(metrics["wrist_peak"]))
    high_harmonic = max(
        float(metrics.get("tip_high_harmonic_rms", 0.0)),
        float(metrics.get("wrist_high_harmonic_rms", 0.0)),
    )
    plane_mismatch = float(metrics.get("wrist_plane_mismatch_deg", 0.0))
    return float(
        exceedance
        + 0.10 * max(tip, wrist)
        + 0.05 * float(metrics["combined"])
        + 0.001 * peak
        + 0.02 * high_harmonic
        + 0.02 * plane_mismatch
    )


def _goal_reached(metrics: Mapping[str, float], target_rmse_mm: float) -> bool:
    return bool(
        float(metrics["tip"]) < float(target_rmse_mm)
        and float(metrics["wrist"]) < float(target_rmse_mm)
    )


def _sha256_file(path: Path) -> str:
    """计算源文件指纹，用于判断机构模型或优化逻辑是否已经更新。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


@dataclass
class _OptimizationRecorder:
    """把一次完整求解保存成可直接绘图的版本化数据集。

    ``events.jsonl`` 在运行过程中持续刷新，即使求解意外中断也保留已经完成的
    评价。结束时再压缩为 NPZ，便于后续快速加载全部变量历史。
    """

    run_dir: Path
    manifest_path: Path
    latest_path: Path
    event_stream: TextIO
    space: DesignSpace
    data: ProblemData
    manifest: dict[str, Any]
    candidate_rows: list[dict[str, Any]] = field(default_factory=list)
    cma_rows: list[dict[str, Any]] = field(default_factory=list)
    region_rows: list[dict[str, Any]] = field(default_factory=list)
    best_snapshots: list[dict[str, Any]] = field(default_factory=list)
    selected_evaluations: set[int] = field(default_factory=set)
    best_improvement_evaluations: set[int] = field(default_factory=set)
    best_score: float = math.inf
    valid_evaluation_count: int = 0
    invalid_evaluation_count: int = 0
    record_candidate_events: bool = False
    event_flush_interval: int = 256
    events_since_flush: int = 0

    @classmethod
    def create(
        cls,
        output_dir: Path,
        data: ProblemData,
        space: DesignSpace,
        config: _OptimizeConfig,
    ) -> "_OptimizationRecorder":
        model_path = Path(__file__).with_name("fourbar3d_python.py")
        optimizer_path = Path(__file__)
        model_hash = _sha256_file(model_path)
        optimizer_hash = _sha256_file(optimizer_path)
        target_hash = _sha256_arrays(data.target_tip, data.target_wrist)
        design_hash = _sha256_arrays(space.x0, space.lb, space.ub)
        problem_payload = {
            "model_sha256": model_hash,
            "target_sha256": target_hash,
            "design_space_sha256": design_hash,
            "variable_names": list(space.names),
            "wrist_node": "L",
            "tip_node": "U",
            "frames": int(data.phase.size),
            "rmse_rule": (
                "Excel-fixed phase zero; Tip and Wrist are concatenated into one "
                "synchronized closed 6D polyline and divided by its coupled arc length; "
                "generated and target point k are compared only at the same equal-arc "
                "phase index; no nearest-point or cyclic alignment"
            ),
            "correspondence_mode": "strict_initialized_equal_arc_index",
            "target_source": data.target_source,
            "target_initialization": data.target_initialization,
            "stutter_rule": (
                "hard checks use original equal-time mechanism frames; continuous "
                "stutter penalty is added after strict RMSE"
            ),
        }
        problem_fingerprint = hashlib.sha256(
            json.dumps(problem_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id_base = f"{timestamp}_{problem_fingerprint[:10]}"
        archive_root = output_dir / "optimization_data"
        archive_root.mkdir(parents=True, exist_ok=True)
        run_dir = archive_root / run_id_base
        suffix = 1
        while run_dir.exists():
            run_dir = archive_root / f"{run_id_base}_{suffix:02d}"
            suffix += 1
        run_dir.mkdir(parents=True)
        run_id = run_dir.name
        created_at = datetime.now(timezone.utc).isoformat()
        manifest = {
            "schema_version": "1.0",
            "status": "running",
            "run_id": run_id,
            "created_at_utc": created_at,
            "problem_fingerprint": problem_fingerprint,
            "problem_definition": problem_payload,
            "optimizer_sha256": optimizer_hash,
            "optimizer_config": _reported_optimizer_config(config),
            "variable_count": len(space.names),
            "variable_names": list(space.names),
            "metric_names": list(_METRIC_NAMES),
            "data_coverage": {
                "every_fourbar_evaluation": True,
                "requested_and_repaired_variables": True,
                "cma_distribution_and_selection": True,
                "region_boundaries_and_split_sensitivity": True,
                "local_refinement_method": "SLSQP",
                "slsqp_trials": True,
                "best_trajectory_snapshots": True,
                "final_full_mechanism": True,
            },
        }
        manifest_path = run_dir / "manifest.json"
        latest_path = archive_root / "latest.json"
        _write_json(manifest_path, manifest)
        _write_json(latest_path, {
            "run_id": run_id,
            "status": "running",
            "problem_fingerprint": problem_fingerprint,
            "manifest": str(manifest_path.resolve()),
        })
        np.savez_compressed(
            run_dir / "problem_snapshot.npz",
            phase=data.phase,
            base32=data.base32,
            raw_target_tip=data.target_tip,
            raw_target_wrist=data.target_wrist,
            x0=space.x0,
            lower_bounds=space.lb,
            upper_bounds=space.ub,
            variable_names=np.asarray(space.names, dtype="U64"),
        )
        with (run_dir / "variable_schema.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["index", "name", "initial", "lower", "upper"])
            for index, (name, initial, lower, upper) in enumerate(
                zip(space.names, space.x0, space.lb, space.ub), start=1
            ):
                writer.writerow([index, name, initial, lower, upper])
        event_stream = (run_dir / "events.jsonl").open(
            "a", encoding="utf-8", buffering=64 * 1024
        )
        recorder = cls(
            run_dir=run_dir,
            manifest_path=manifest_path,
            latest_path=latest_path,
            event_stream=event_stream,
            space=space,
            data=data,
            manifest=manifest,
            record_candidate_events=bool(config.record_candidate_events),
            event_flush_interval=max(1, int(config.event_flush_interval)),
        )
        recorder._append_event({
            "event": "run_start",
            "run_id": run_id,
            "created_at_utc": created_at,
            "problem_fingerprint": problem_fingerprint,
            "config": _reported_optimizer_config(config),
        })
        return recorder

    def _append_event(
        self,
        payload: Mapping[str, Any],
        *,
        force_flush: bool = False,
    ) -> None:
        self.event_stream.write(
            json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n"
        )
        self.events_since_flush += 1
        if force_flush or self.events_since_flush >= self.event_flush_interval:
            self.event_stream.flush()
            self.events_since_flush = 0

    def _write_live_checkpoint(
        self,
        candidate: _Candidate,
        evaluation: int,
        round_index: int,
    ) -> dict[str, Any]:
        """Persist the current best without letting a transient sync lock stop search."""

        checkpoint = self.run_dir / "best_so_far_checkpoint.npz"
        temporary = self.run_dir / (
            f".best_so_far_checkpoint.{os.getpid()}.{threading.get_ident()}."
            f"{int(evaluation)}.tmp"
        )
        payload = {
            "x": candidate.x,
            "variable_names": np.asarray(self.space.names, dtype="U64"),
            "variable_lb": self.space.lb,
            "variable_ub": self.space.ub,
            "metrics": json.dumps(candidate.metrics, ensure_ascii=False),
            "score": float(candidate.score),
            "evaluation": int(evaluation),
            "round": int(round_index),
        }
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **payload)
                stream.flush()
                os.fsync(stream.fileno())

            # Reject a truncated archive before it can replace the last good checkpoint.
            with np.load(temporary, allow_pickle=False) as saved:
                if int(saved["evaluation"]) != int(evaluation):
                    raise ValueError("Live checkpoint evaluation failed validation.")
                if np.asarray(saved["x"]).shape != np.asarray(candidate.x).shape:
                    raise ValueError("Live checkpoint design vector failed validation.")

            last_error: OSError | None = None
            for attempt in range(12):
                try:
                    os.replace(temporary, checkpoint)
                    return {
                        "path": str(checkpoint),
                        "mode": "atomic_replace",
                        "attempts": attempt + 1,
                    }
                except OSError as exc:
                    # OneDrive and antivirus scanners can briefly hold the old target open.
                    if not isinstance(exc, PermissionError) and getattr(
                        exc, "winerror", None
                    ) not in {5, 32}:
                        raise
                    last_error = exc
                    time.sleep(min(0.05 * (2**attempt), 0.8))

            fallback = self.run_dir / (
                f"best_so_far_checkpoint_e{int(evaluation):09d}.npz"
            )
            if fallback.exists():
                fallback = self.run_dir / (
                    f"best_so_far_checkpoint_e{int(evaluation):09d}_{time.time_ns()}.npz"
                )
            os.replace(temporary, fallback)
            return {
                "path": str(fallback),
                "mode": "versioned_fallback",
                "attempts": 12,
                "error": repr(last_error),
            }
        except Exception as exc:
            # A persistence failure must not discard the in-memory optimization state.
            return {
                "path": str(temporary) if temporary.exists() else None,
                "mode": "write_deferred",
                "attempts": 0,
                "error": repr(exc),
            }

    def record_candidate(
        self,
        candidate: _Candidate,
        evaluation: int,
        *,
        round_index: int,
        generation: int = -1,
        population_index: int = -1,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        candidate.evaluation_id = int(evaluation)
        if candidate.valid:
            self.valid_evaluation_count += 1
        else:
            self.invalid_evaluation_count += 1
        repair_l2 = float(np.linalg.norm(candidate.y - candidate.requested_y))
        is_improvement = bool(candidate.valid and candidate.score < self.best_score)
        if is_improvement:
            self.best_score = float(candidate.score)
            self.best_improvement_evaluations.add(int(evaluation))
            _write_json(
                self.run_dir / "progress.json",
                {
                    "status": "running",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "evaluation": int(evaluation),
                    "allowed_combinations": int(self.valid_evaluation_count),
                    "rejected_combinations": int(self.invalid_evaluation_count),
                    "round": int(round_index),
                    "stage": candidate.stage,
                    "region_id": int(candidate.region_id),
                    "best_score": float(candidate.score),
                    "metrics": {
                        key: float(value)
                        for key, value in candidate.metrics.items()
                    },
                },
            )
            if candidate.snapshot is not None:
                snapshot = {
                    key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
                    for key, value in candidate.snapshot.items()
                }
                snapshot.update({
                    "evaluation": int(evaluation),
                    "score": float(candidate.score),
                    "x": candidate.x.copy(),
                })
                self.best_snapshots.append(snapshot)
            # 每次刷新最优解时同步写出最小可续跑检查点。它只属于本轮任务，
            # 不含历史种群、协方差或分区树，进程中断后仍可严格恢复当前最优向量。
            checkpoint_result = self._write_live_checkpoint(
                candidate,
                evaluation,
                round_index,
            )
            if checkpoint_result["mode"] != "atomic_replace":
                self._append_event(
                    {
                        "event": "checkpoint_write_deferred",
                        "evaluation": int(evaluation),
                        "round": int(round_index),
                        **checkpoint_result,
                    },
                    force_flush=True,
                )
        row = {
            "evaluation": int(evaluation),
            "stage": candidate.stage,
            "round": int(round_index),
            "region_id": int(candidate.region_id),
            "generation": int(generation),
            "population_index": int(population_index),
            "valid": bool(candidate.valid),
            "score": float(candidate.score),
            "metrics": dict(candidate.metrics),
            "requested_y": candidate.requested_y.copy(),
            "actual_y": candidate.y.copy(),
            "actual_x": candidate.x.copy(),
            "repair_l2_normalized": repair_l2,
            "best_improvement": is_improvement,
            "metadata": dict(metadata or {}),
        }
        self.candidate_rows.append(row)
        if (self.valid_evaluation_count + self.invalid_evaluation_count) % 128 == 0:
            _write_json(
                self.run_dir / "evaluation_counts.json",
                {
                    "status": "running",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "evaluation": int(evaluation),
                    "allowed_combinations": int(self.valid_evaluation_count),
                    "rejected_combinations": int(self.invalid_evaluation_count),
                    "best_score": float(self.best_score),
                },
            )
        # 全部候选始终保存在 candidate_history.npz；JSONL 仅保留改进点和
        # 求解器结构事件，避免逐候选 JSON 序列化成为主要运行开销。
        if self.record_candidate_events or is_improvement:
            self._append_event(
                {"event": "candidate", **row},
                force_flush=is_improvement,
            )
        candidate.snapshot = None

    def record_cma_generation(self, payload: Mapping[str, Any]) -> None:
        row = dict(payload)
        selected = [int(value) for value in row.get("selected_evaluations", [])]
        self.selected_evaluations.update(selected)
        self.cma_rows.append(row)
        self._append_event({"event": "cma_generation", **row})

    def record_region_event(self, event: str, payload: Mapping[str, Any]) -> None:
        row = {"event": event, **dict(payload)}
        self.region_rows.append(row)
        self._append_event(row)

    def _save_candidate_arrays(self) -> None:
        rows = self.candidate_rows
        n_variables = len(self.space.names)
        if not rows:
            return
        evaluations = np.asarray([row["evaluation"] for row in rows], dtype=np.int64)
        metrics = np.asarray(
            [[row["metrics"].get(name, 1e6) for name in _METRIC_NAMES] for row in rows],
            dtype=float,
        )
        np.savez_compressed(
            self.run_dir / "candidate_history.npz",
            evaluation=evaluations,
            stage=np.asarray([row["stage"] for row in rows], dtype="U24"),
            round=np.asarray([row["round"] for row in rows], dtype=np.int32),
            region_id=np.asarray([row["region_id"] for row in rows], dtype=np.int32),
            generation=np.asarray([row["generation"] for row in rows], dtype=np.int32),
            population_index=np.asarray(
                [row["population_index"] for row in rows], dtype=np.int32
            ),
            valid=np.asarray([row["valid"] for row in rows], dtype=bool),
            selected=np.asarray(
                [row["evaluation"] in self.selected_evaluations for row in rows],
                dtype=bool,
            ),
            best_improvement=np.asarray(
                [row["best_improvement"] for row in rows], dtype=bool
            ),
            score=np.asarray([row["score"] for row in rows], dtype=float),
            metrics=metrics,
            metric_names=np.asarray(_METRIC_NAMES, dtype="U32"),
            requested_y=np.stack([row["requested_y"] for row in rows]).reshape(-1, n_variables),
            actual_y=np.stack([row["actual_y"] for row in rows]).reshape(-1, n_variables),
            actual_x=np.stack([row["actual_x"] for row in rows]).reshape(-1, n_variables),
            repair_l2_normalized=np.asarray(
                [row["repair_l2_normalized"] for row in rows], dtype=float
            ),
            variable_names=np.asarray(self.space.names, dtype="U64"),
            lower_bounds=self.space.lb,
            upper_bounds=self.space.ub,
        )
        with (self.run_dir / "candidate_summary.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "evaluation", "stage", "round", "region_id", "generation",
                "population_index", "valid", "selected", "best_improvement",
                "score", *_METRIC_NAMES, "repair_l2_normalized",
            ])
            for row in rows:
                writer.writerow([
                    row["evaluation"], row["stage"], row["round"], row["region_id"],
                    row["generation"], row["population_index"], row["valid"],
                    row["evaluation"] in self.selected_evaluations,
                    row["best_improvement"], row["score"],
                    *[row["metrics"].get(name, 1e6) for name in _METRIC_NAMES],
                    row["repair_l2_normalized"],
                ])

    def _save_cma_arrays(self) -> None:
        if not self.cma_rows:
            return
        maximum_selected = max(
            1, max(len(row.get("selected_evaluations", [])) for row in self.cma_rows)
        )
        selected = np.full((len(self.cma_rows), maximum_selected), -1, dtype=np.int64)
        for index, row in enumerate(self.cma_rows):
            values = np.asarray(row.get("selected_evaluations", []), dtype=np.int64)
            selected[index, :values.size] = values
        np.savez_compressed(
            self.run_dir / "cma_generation_history.npz",
            round=np.asarray([row["round"] for row in self.cma_rows], dtype=np.int32),
            region_id=np.asarray([row["region_id"] for row in self.cma_rows], dtype=np.int32),
            generation=np.asarray([row["generation"] for row in self.cma_rows], dtype=np.int32),
            evaluation_end=np.asarray(
                [row["evaluation_end"] for row in self.cma_rows], dtype=np.int64
            ),
            sigma_before=np.asarray(
                [row["sigma_before"] for row in self.cma_rows], dtype=float
            ),
            sigma_after=np.asarray(
                [row["sigma_after"] for row in self.cma_rows], dtype=float
            ),
            mean_before=np.stack([row["mean_before"] for row in self.cma_rows]),
            mean_after=np.stack([row["mean_after"] for row in self.cma_rows]),
            diagonal_before=np.stack([row["diagonal_before"] for row in self.cma_rows]),
            diagonal_after=np.stack([row["diagonal_after"] for row in self.cma_rows]),
            proposal_min=np.stack([row["proposal_min"] for row in self.cma_rows]),
            proposal_max=np.stack([row["proposal_max"] for row in self.cma_rows]),
            tested_min=np.stack([row["tested_min"] for row in self.cma_rows]),
            tested_max=np.stack([row["tested_max"] for row in self.cma_rows]),
            selected_evaluations=selected,
            valid_count=np.asarray(
                [row["valid_count"] for row in self.cma_rows], dtype=np.int32
            ),
            population=np.asarray(
                [row["population"] for row in self.cma_rows], dtype=np.int32
            ),
            generation_best=np.asarray(
                [row["generation_best"] for row in self.cma_rows], dtype=float
            ),
            variable_names=np.asarray(self.space.names, dtype="U64"),
        )

    def _save_best_snapshots(self) -> None:
        if not self.best_snapshots:
            return
        snapshots = self.best_snapshots
        angle_names = tuple(snapshots[0]["angle_names"])
        np.savez_compressed(
            self.run_dir / "best_trajectory_snapshots.npz",
            evaluation=np.asarray([row["evaluation"] for row in snapshots], dtype=np.int64),
            score=np.asarray([row["score"] for row in snapshots], dtype=float),
            x=np.stack([row["x"] for row in snapshots]),
            tip=np.stack([row["tip"] for row in snapshots]),
            wrist=np.stack([row["wrist"] for row in snapshots]),
            target_tip=np.stack([row["target_tip"] for row in snapshots]),
            target_wrist=np.stack([row["target_wrist"] for row in snapshots]),
            b_curve=np.stack([row["b_curve"] for row in snapshots]),
            l2_values=np.stack([row["l2_values"] for row in snapshots]),
            l31_values=np.stack([row["l31_values"] for row in snapshots]),
            l32_values=np.stack([row["l32_values"] for row in snapshots]),
            l3_values=np.stack([row["l3_values"] for row in snapshots]),
            l6_values=np.stack([row["l6_values"] for row in snapshots]),
            l7_values=np.stack([row["l7_values"] for row in snapshots]),
            l8_values=np.stack([row["l8_values"] for row in snapshots]),
            l12_values=np.stack([row["l12_values"] for row in snapshots]),
            angle_history=np.stack([row["angle_history"] for row in snapshots]),
            angle_names=np.asarray(angle_names, dtype="U48"),
            variable_names=np.asarray(self.space.names, dtype="U64"),
        )

    def finalize(
        self,
        summary: Mapping[str, Any],
        checkpoint: Path,
    ) -> None:
        self._append_event({
            "event": "run_complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary": dict(summary),
        }, force_flush=True)
        self.event_stream.close()
        self._save_candidate_arrays()
        self._save_cma_arrays()
        self._save_best_snapshots()
        with (self.run_dir / "region_history.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for row in self.region_rows:
                stream.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
        shutil.copy2(checkpoint, self.run_dir / "final_checkpoint.npz")
        readme = (
            "# Optimization data archive\n\n"
            "- `manifest.json`: problem/version fingerprints and run configuration.\n"
            "- `problem_snapshot.npz`: targets, phase, original design and all bounds.\n"
            "- `candidate_history.npz`: every requested/repaired candidate and metric.\n"
            "- `candidate_summary.csv`: lightweight evaluation index.\n"
            "- `cma_generation_history.npz`: means, diagonal covariance, sigma and selection.\n"
            "- `region_history.jsonl`: leaf selection, bounds and split sensitivity.\n"
            "- `best_trajectory_snapshots.npz`: curves at every best-so-far update.\n"
            "- `best_so_far_checkpoint.npz`: crash-safe current best design vector.\n"
            "- `events.jsonl`: durable chronological event stream.\n"
            "- `final_checkpoint.npz`: complete final mechanism state.\n"
        )
        (self.run_dir / "README.md").write_text(readme, encoding="utf-8")
        self.manifest.update({
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_count": len(self.candidate_rows),
            "allowed_combinations": int(self.valid_evaluation_count),
            "rejected_combinations": int(self.invalid_evaluation_count),
            "cma_generation_count": len(self.cma_rows),
            "region_event_count": len(self.region_rows),
            "best_snapshot_count": len(self.best_snapshots),
            "best_score": float(self.best_score),
            "summary": dict(summary),
            "files": sorted(path.name for path in self.run_dir.iterdir()),
        })
        _write_json(
            self.run_dir / "progress.json",
            {
                "status": "complete",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_count": len(self.candidate_rows),
                "cma_generation_count": len(self.cma_rows),
                "best_score": float(self.best_score),
                "metrics": dict(summary.get("metrics", {})),
                "success": bool(summary.get("success", False)),
            },
        )
        _write_json(self.manifest_path, self.manifest)
        _write_json(self.latest_path, {
            "run_id": self.manifest["run_id"],
            "status": "complete",
            "problem_fingerprint": self.manifest["problem_fingerprint"],
            "manifest": str(self.manifest_path.resolve()),
            "run_dir": str(self.run_dir.resolve()),
        })

    def fail(self, error: BaseException) -> None:
        try:
            if not self.event_stream.closed:
                self._append_event({
                    "event": "run_failed",
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }, force_flush=True)
                self.event_stream.close()
        except Exception:
            pass
        finally:
            self.manifest.update({
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "candidate_count": len(self.candidate_rows),
            })
            _write_json(
                self.run_dir / "progress.json",
                {
                    "status": "failed",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "candidate_count": len(self.candidate_rows),
                    "best_score": float(self.best_score),
                },
            )
            _write_json(self.manifest_path, self.manifest)
            _write_json(self.latest_path, {
                "run_id": self.manifest["run_id"],
                "status": "failed",
                "problem_fingerprint": self.manifest["problem_fingerprint"],
                "manifest": str(self.manifest_path.resolve()),
                "run_dir": str(self.run_dir.resolve()),
            })


def load_optimization_history(
    run_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """读取已保存的完整优化历史，不再调用 fourbar 模型。

    ``run_dir`` 省略时自动读取 ``optimization_data/latest.json`` 指向的最新
    完整数据集。返回值可直接用于绘制参数包络、区域分割、CMA-ES 选择、
    SLSQP 收敛及历次最优轨迹。
    """

    if run_dir is None:
        archive_root = (
            Path(output_dir) if output_dir is not None else CURRENT_DIR / "output"
        ) / "optimization_data"
        latest_path = archive_root / "latest.json"
        if not latest_path.exists():
            raise FileNotFoundError(f"Optimization archive index not found: {latest_path}")
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        directory = Path(latest["run_dir"])
    else:
        directory = Path(run_dir)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Optimization manifest not found: {manifest_path}")

    def load_npz(name: str) -> dict[str, np.ndarray]:
        path = directory / name
        if not path.exists():
            return {}
        with np.load(path, allow_pickle=False) as saved:
            return {key: np.asarray(saved[key]).copy() for key in saved.files}

    region_events: list[dict[str, Any]] = []
    region_path = directory / "region_history.jsonl"
    if region_path.exists():
        region_events = [
            json.loads(line)
            for line in region_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return {
        "run_dir": directory.resolve(),
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "problem": load_npz("problem_snapshot.npz"),
        "candidates": load_npz("candidate_history.npz"),
        "cma_generations": load_npz("cma_generation_history.npz"),
        "best_snapshots": load_npz("best_trajectory_snapshots.npz"),
        "region_events": region_events,
        "final_checkpoint_path": (directory / "final_checkpoint.npz").resolve(),
    }


def _to_physical(y: np.ndarray, space: DesignSpace) -> np.ndarray:
    """把优化器使用的 [0,1] 变量映射为 mm/rad/无量纲的物理值。"""

    normalized = np.clip(np.asarray(y, dtype=float), 0.0, 1.0)
    return space.lb + normalized * (space.ub - space.lb)


def _to_normalized(x: np.ndarray, space: DesignSpace) -> np.ndarray:
    """把物理量映射到统一尺度，避免不同单位主导 CMA-ES 步长。"""

    return np.clip(
        (np.asarray(x, dtype=float) - space.lb) / np.maximum(space.ub - space.lb, 1e-12),
        0.0,
        1.0,
    )


_PROMOTABLE_PERIODIC_RODS = ("L2", "L3", "L5", "L6", "L7", "L8", "L12")


def _map_checkpoint_vector_to_space(
    checkpoint: Mapping[str, Any],
    space: DesignSpace,
) -> tuple[np.ndarray, dict[str, Any]]:
    """把同一任务的固定杆检查点严格嵌入对应的 Fourier 周期杆空间。

    允许的唯一维度变化是 ``Lk -> Lk_C0``，并把该杆新增的所有谐波系数
    初始化为零。映射后必须逐项位于当前物理边界内；其余变量缺失、旧变量
    无法解释或检查点没有变量名时均拒绝续跑。
    """

    if "x" not in checkpoint:
        raise ValueError("Continuation checkpoint has no design vector 'x'.")
    source_x = np.asarray(checkpoint["x"], dtype=float).reshape(-1)
    source_names_raw = checkpoint.get("variable_names")
    target_names = tuple(str(name) for name in space.names)

    if source_names_raw is None:
        if source_x.shape != space.x0.shape:
            raise ValueError(
                "Continuation checkpoint dimension mismatch and variable_names are absent: "
                f"{source_x.size} != {space.x0.size}"
            )
        return source_x.copy(), {
            "mode": "same_dimension_without_schema",
            "source_dimension": int(source_x.size),
            "target_dimension": int(space.x0.size),
            "promoted_rods": [],
        }

    source_names = tuple(
        str(name.decode("utf-8") if isinstance(name, bytes) else name)
        for name in np.asarray(source_names_raw).reshape(-1)
    )
    if len(source_names) != source_x.size:
        raise ValueError(
            "Continuation checkpoint variable schema length does not match x: "
            f"{len(source_names)} != {source_x.size}"
        )
    if len(set(source_names)) != len(source_names):
        raise ValueError("Continuation checkpoint contains duplicate variable names.")

    source_by_name = dict(zip(source_names, source_x))
    target_x = np.asarray(space.x0, dtype=float).copy()
    promoted_rods: list[str] = []
    missing: list[str] = []

    for index, target_name in enumerate(target_names):
        if target_name in source_by_name:
            target_x[index] = float(source_by_name[target_name])
            continue

        promoted = False
        for rod_name in _PROMOTABLE_PERIODIC_RODS:
            prefix = f"{rod_name}_C"
            if target_name == f"{rod_name}_C0" and rod_name in source_by_name:
                target_x[index] = float(source_by_name[rod_name])
                promoted_rods.append(rod_name)
                promoted = True
                break
            if target_name.startswith(prefix) and rod_name in source_by_name:
                target_x[index] = 0.0
                promoted = True
                break
        if not promoted:
            missing.append(target_name)

    promoted_rods = sorted(set(promoted_rods))
    allowed_unmapped = set(promoted_rods)
    unmapped_source = sorted(set(source_names) - set(target_names) - allowed_unmapped)
    if missing or unmapped_source:
        raise ValueError(
            "Continuation checkpoint schema is not a strict fixed-to-periodic embedding: "
            f"missing target variables={missing}, unmapped source variables={unmapped_source}"
        )

    below = np.flatnonzero(target_x < np.asarray(space.lb) - 1e-9)
    above = np.flatnonzero(target_x > np.asarray(space.ub) + 1e-9)
    if below.size or above.size:
        violations = [
            {
                "name": target_names[int(index)],
                "value": float(target_x[int(index)]),
                "lower": float(space.lb[int(index)]),
                "upper": float(space.ub[int(index)]),
            }
            for index in np.concatenate([below, above])
        ]
        raise ValueError(
            "Continuation checkpoint embedding violates current physical bounds: "
            f"{violations}"
        )

    return target_x, {
        "mode": (
            "same_schema"
            if source_names == target_names
            else "strict_fixed_to_periodic_embedding"
        ),
        "source_dimension": int(source_x.size),
        "target_dimension": int(target_x.size),
        "promoted_rods": promoted_rods,
        "constant_term_rule": {
            rod_name: f"{rod_name}_C0={rod_name}"
            for rod_name in promoted_rods
        },
        "new_harmonic_rule": "all newly introduced cosine/sine coefficients are zero",
        "exact_subspace_embedding": True,
    }


_INNER_SOLVED_VARIABLES = {
    "Target_Tx_mm",
    "Target_Ty_mm",
    "Target_Tz_mm",
    "Target_Tip_Tx_mm",
    "Target_Tip_Ty_mm",
    "Target_Tip_Tz_mm",
    "Target_Wrist_Tx_mm",
    "Target_Wrist_Ty_mm",
    "Target_Wrist_Tz_mm",
    "Target_Scale",
}

# 这两个量仍保留在 73 项完整设计向量中，用于原始 MATLAB 参数兼容和
# 结果审计，但当前输出拓扑对它们没有依赖：H_finger 只属于已停用的旧腕部
# 角度分支；Lf1 只连接已按要求删除的 V/X 支路。把它们交给 CMA-ES 会增加
# 两个严格零灵敏度方向，因此固定在原始值而不计入外层搜索。
_OUTPUT_INACTIVE_VARIABLES = {
    "H_finger",
    "Lf1",
}


_DECOUPLED_ROTATION_GROUPS = (
    {
        "Target_Tip_Rx_rad",
        "Target_Tip_Ry_rad",
        "Target_Tip_Rz_rad",
    },
    {
        "Target_Wrist_Rx_rad",
        "Target_Wrist_Ry_rad",
        "Target_Wrist_Rz_rad",
    },
)

_SHARED_ROTATION_GROUP = {
    "Target_Rx_rad",
    "Target_Ry_rad",
    "Target_Rz_rad",
}


def _inner_solved_variables(space: DesignSpace) -> set[str]:
    """Return variables intentionally overwritten by an inner optimizer.

    The all-variable task disables inner pose fitting. Every independent pose
    coordinate is therefore sampled, adapted and refined with the mechanism.
    """

    return set()


def _outer_active_indices(space: DesignSpace) -> np.ndarray:
    """Return every independent coordinate used by the global optimizer."""

    inner_solved = _inner_solved_variables(space)
    # B 曲线已被规范化到 Bz(0)=0，因此最高阶 z 余弦项由其余 z
    # 余弦项决定。保留该值用于完整输出，但不再把冗余相位坐标交给优化器。
    dependent_repaired = {
        "B_Z_C3c"
        if space.b_curve_mode in {"fourier_z3", "fourier_z3_c0", "fourier_xyz3"}
        else "B_Z_C2c"
    }
    return np.asarray(
        [
            index
            for index, name in enumerate(space.names)
            if (
                name not in inner_solved
                and name not in dependent_repaired
                and name not in _OUTPUT_INACTIVE_VARIABLES
            )
        ],
        dtype=np.int32,
    )


_WRIST_UPSTREAM_STATIC_NAMES = {
    "L2",
    "L31",
    "L4",
    "L41",
    "L5",
    "L51",
    "L52",
    "L6",
    "L3",
    "L61",
}

_WRIST_CORE_NAMES = {
    "L2",
    "L31",
    "L4",
    "L41",
    "L6",
    "L3",
    "L61",
    "B_Y_C1c",
    "B_Y_C1s",
    "B_Y_C2c",
    "B_Y_C2s",
    "B_Z_C1c",
    "B_Z_C1s",
    "B_Z_C0",
    "B_CenterX",
    "ZC_C1s",
}

# 灵敏度活动集仅控制阶段搜索范围；当前固定 L5 属于 Wrist 上游，
# L8(t) 属于 Wrist=L 之后的 Tip 形状调节。
# 这些集合只控制某个 CMA-ES 阶段允许扰动的坐标，不删除变量，也不改变其物理边界。
_SENSITIVITY_WRIST_NAMES = {
    "L2_C0",
    "L2_C1c",
    "L2_C1s",
    "L2_C2c",
    "L2_C2s",
    "L2_C3c",
    "L2_C3s",
    "L31",
    "L31_C0",
    "L31_C1c",
    "L31_C1s",
    "L31_C2c",
    "L31_C2s",
    "L31_C3c",
    "L31_C3s",
    "L4",
    "L41",
    "L5",
    "L5_C0",
    "L5_C1c",
    "L5_C1s",
    "L5_C2c",
    "L5_C2s",
    "L5_C3c",
    "L5_C3s",
    "L6",
    "L6_C0",
    "L6_C1c",
    "L6_C1s",
    "L6_C2c",
    "L6_C2s",
    "L6_C3c",
    "L6_C3s",
    "L3",
    "L3_C0",
    "L3_C1c",
    "L3_C1s",
    "L3_C2c",
    "L3_C2s",
    "L3_C3c",
    "L3_C3s",
    "L61",
    "B_Y_C1c",
    "B_Y_C1s",
    "B_Y_C2c",
    "B_Y_C2s",
    "B_Z_C1c",
    "B_Z_C1s",
    "B_Z_C0",
    "B_CenterX",
}

_SENSITIVITY_TIP_NAMES = {
    "L9",
    "Lf2",
    "L_down",
    "theta18_deg",
    "L7_C0",
    "L7_C1c",
    "L7_C1s",
    "L7_C2c",
    "L7_C2s",
    "L8_C0",
    "L8_C1c",
    "L8_C1s",
    "L8_C2c",
    "L8_C2s",
    "L8_C3c",
    "L8_C3s",
    "L12_C0",
    "L12_C1c",
    "L12_C1s",
    "L12_C2c",
    "L12_C2s",
    "L12_C3c",
    "L12_C3s",
}

_SENSITIVITY_JOINT_NAMES = (
    _SENSITIVITY_WRIST_NAMES
    | _SENSITIVITY_TIP_NAMES
    | {"L2"}
)


def _search_active_indices(
    space: DesignSpace,
    search_scope: str = "all",
) -> np.ndarray:
    """返回当前结构阶段允许改变的外层变量。

    Wrist=L 位于后级 L7/L8 和指端杆组之前，因此其轨迹只受前级静态几何、
    B 输入曲线、L3(t) 及固定 L5 影响。把这组变量与只影响 Tip 的下游变量分开，
    可避免高维 CMA-ES 把一半种群预算花在当前目标完全不敏感的坐标上。
    """

    outer = _outer_active_indices(space)
    if search_scope == "all":
        return outer
    upstream = np.asarray(
        [
            index
            for index in outer
            if (
                space.names[index] in _WRIST_UPSTREAM_STATIC_NAMES
                or space.names[index].startswith("B_")
                or space.names[index].startswith("L2_")
                or space.names[index].startswith("L31_")
                or space.names[index].startswith("L3_")
                or space.names[index].startswith("L5_")
                or space.names[index].startswith("L6_")
                or space.names[index].startswith("ZC_")
            )
        ],
        dtype=np.int32,
    )
    if search_scope == "wrist_upstream":
        return upstream
    if search_scope == "wrist_core":
        return np.asarray(
            [
                index
                for index in upstream
                if space.names[index] in _WRIST_CORE_NAMES
            ],
            dtype=np.int32,
        )
    if search_scope == "tip_downstream":
        upstream_set = set(int(index) for index in upstream)
        return np.asarray(
            [index for index in outer if int(index) not in upstream_set],
            dtype=np.int32,
        )
    sensitivity_scopes = {
        "sensitivity_wrist": _SENSITIVITY_WRIST_NAMES,
        "sensitivity_tip": _SENSITIVITY_TIP_NAMES,
        "sensitivity_joint": _SENSITIVITY_JOINT_NAMES,
    }
    if search_scope in sensitivity_scopes:
        selected_names = sensitivity_scopes[search_scope]
        return np.asarray(
            [
                index
                for index in outer
                if space.names[index] in selected_names
            ],
            dtype=np.int32,
        )
    raise ValueError(f"Unsupported search_scope: {search_scope}")


def _cma_blocks(
    space: DesignSpace,
    full_covariance: bool = False,
    active_indices: np.ndarray | None = None,
) -> list[np.ndarray]:
    """按物理类别构造协方差块，学习块内耦合而不混合无关参数。"""

    outer_active = (
        _outer_active_indices(space)
        if active_indices is None
        else np.asarray(active_indices, dtype=np.int32)
    )
    if full_covariance:
        return [outer_active.copy()]
    active = set(int(index) for index in outer_active)
    groups: dict[str, list[int]] = {
        "wrist_static": [],
        "tip_static": [],
        "b_curve": [],
        "l2_periodic": [],
        "l31_periodic": [],
        "l3_periodic": [],
        "l5_periodic": [],
        "l6_periodic": [],
        "l7_periodic": [],
        "l8_periodic": [],
        "l12_periodic": [],
        "zc_periodic": [],
        "target_rotation": [],
    }
    for index, name in enumerate(space.names):
        if index not in active:
            continue
        if name.startswith("B_"):
            groups["b_curve"].append(index)
        elif name.startswith("L2_"):
            groups["l2_periodic"].append(index)
        elif name.startswith("L31_"):
            groups["l31_periodic"].append(index)
        elif name.startswith("L3_"):
            groups["l3_periodic"].append(index)
        elif name.startswith("L5_"):
            groups["l5_periodic"].append(index)
        elif name.startswith("L6_"):
            groups["l6_periodic"].append(index)
        elif name.startswith("L7_"):
            groups["l7_periodic"].append(index)
        elif name.startswith("L8_"):
            groups["l8_periodic"].append(index)
        elif name.startswith("L12_"):
            groups["l12_periodic"].append(index)
        elif name.startswith("ZC_"):
            groups["zc_periodic"].append(index)
        elif name.startswith("Target_"):
            groups["target_rotation"].append(index)
        elif name in _WRIST_UPSTREAM_STATIC_NAMES:
            groups["wrist_static"].append(index)
        else:
            groups["tip_static"].append(index)
    return [
        np.asarray(indices, dtype=np.int32)
        for indices in groups.values()
        if indices
    ]


def _seed_cma_covariance(
    seed_y: np.ndarray,
    block: np.ndarray,
    learning_rate: float,
    variance_floor: float,
) -> np.ndarray:
    """由优良样本提取相关方向，但不重复缩小 CMA-ES 的搜索幅度。

    ``seed_y`` 已经位于归一化设计空间，而 CMA 采样还会显式乘以
    ``sigma * region_width``。因此这里必须使用单位方差的相关矩阵，
    不能把样本本身很小的绝对方差再次作为采样尺度。
    """

    values = np.asarray(seed_y, dtype=float)[:, np.asarray(block, dtype=int)]
    dimension = int(values.shape[1])
    identity = np.eye(dimension, dtype=float)
    if values.shape[0] < 3:
        return identity
    centered = values - np.mean(values, axis=0, keepdims=True)
    standard_deviation = np.std(centered, axis=0, ddof=1)
    standardized = centered / np.maximum(standard_deviation, 1e-8)
    correlation = (
        standardized.T @ standardized / max(1, values.shape[0] - 1)
    )
    correlation = 0.5 * (correlation + correlation.T)
    shrinkage = float(np.clip(learning_rate, 0.0, 1.0))
    covariance = (
        (1.0 - shrinkage) * identity + shrinkage * correlation
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, float(variance_floor))
    covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    # 数值截断后重新固定平均方差为 1，让 sigma 成为唯一的全局幅度参数。
    mean_variance = max(float(np.trace(covariance)) / dimension, 1e-12)
    return covariance / mean_variance


def _cma_restart_sigma(
    current_sigma: float,
    config: _OptimizeConfig,
) -> float:
    """按当前阶段尺度重启，避免硬编码 0.25 造成跨区过冲。"""

    target = float(config.cma_restart_sigma_factor) * float(config.sigma0)
    return float(
        np.clip(
            max(float(current_sigma), target),
            float(config.cma_sigma_min),
            float(config.cma_sigma_max),
        )
    )


def _point_in_region(
    y: np.ndarray,
    region: _Region,
    active_indices: np.ndarray,
    tolerance: float = 1e-10,
) -> bool:
    values = np.asarray(y, dtype=float)[active_indices]
    return bool(
        np.all(values >= region.lo[active_indices] - tolerance)
        and np.all(values <= region.hi[active_indices] + tolerance)
    )


def _locate_region(
    y: np.ndarray,
    regions: list[_Region],
    active_indices: np.ndarray,
) -> _Region | None:
    matches = [
        region
        for region in regions
        if _point_in_region(y, region, active_indices)
    ]
    if not matches:
        return None
    return max(matches, key=lambda region: region.depth)


def _project_shared_target_translation(
    state: DesignState,
    result: Any,
    metrics: Mapping[str, float],
    data: ProblemData,
    space: DesignSpace,
    objective_mode: str,
    target_rmse_mm: float,
) -> tuple[DesignState, dict[str, float]]:
    """在当前几何、旋转和缩放下优化两条目标曲线的公共平移。

    Tip 与 Wrist 各自 RMSE 最优平移分别等于其残差均值。双硬目标使用这两个点
    连线上的有界一维搜索，从而避免单纯最小化 combined RMSE 时牺牲其中一条
    曲线；候选平移仍严格截断到三轴物理边界内。
    """

    residual_tip = result.tip - state.target_tip
    residual_wrist = result.wrist - state.target_wrist
    mean_tip = np.mean(residual_tip, axis=0)
    mean_wrist = np.mean(residual_wrist, axis=0)
    base_x = np.asarray(state.x, dtype=float).copy()
    name_to_index = {name: index for index, name in enumerate(space.names)}
    best_state = state
    best_metrics = {key: float(value) for key, value in metrics.items()}
    best_score = _optimization_objective(
        best_metrics, objective_mode, target_rmse_mm
    )
    # 两组二次 RMSE 的无界最小最大平移位于两组残差均值的连线上。
    # 在连线上稠密扫描，并对三轴物理边界截断，成本远小于机构复算。
    for alpha in np.linspace(0.0, 1.0, 25):
        translation_delta = (1.0 - alpha) * mean_tip + alpha * mean_wrist
        trial_x = base_x.copy()
        for axis, name in enumerate(TARGET_POSE_NAMES[:3]):
            if name not in name_to_index:
                continue
            index = name_to_index[name]
            trial_x[index] = np.clip(
                trial_x[index] + translation_delta[axis],
                space.lb[index],
                space.ub[index],
            )
        trial_state = decode_design_vector(trial_x, data, space)
        trial_metrics = combined_rmse(
            result.tip,
            result.wrist,
            trial_state.target_tip,
            trial_state.target_wrist,
        )
        trial_score = _optimization_objective(
            trial_metrics, objective_mode, target_rmse_mm
        )
        if trial_score < best_score:
            best_state = trial_state
            best_metrics = trial_metrics
            best_score = trial_score
    return best_state, best_metrics


def _project_decoupled_target_translation(
    state: DesignState,
    result: Any,
    metrics: Mapping[str, float],
    data: ProblemData,
    space: DesignSpace,
    objective_mode: str,
    target_rmse_mm: float,
) -> tuple[DesignState, dict[str, float]]:
    """分别求解 Tip 与 Wrist 的有界最小二乘平移。

    在旋转和共享尺度固定时，每条曲线的平方位置误差关于平移是凸二次函数，
    无界最优增量就是逐帧残差均值。两条曲线互不共享平移，因此分别投影到
    各自三轴边界即可同时降低两项 RMSE，不需要公共平移中的权衡线搜索。
    """

    base_x = np.asarray(state.x, dtype=float).copy()
    name_to_index = {name: index for index, name in enumerate(space.names)}
    trial_x = base_x.copy()
    residual_by_target = {
        "Tip": np.mean(result.tip - state.target_tip, axis=0),
        "Wrist": np.mean(result.wrist - state.target_wrist, axis=0),
    }
    for target_label, residual_mean in residual_by_target.items():
        for axis, axis_name in enumerate(("Tx_mm", "Ty_mm", "Tz_mm")):
            name = f"Target_{target_label}_{axis_name}"
            index = name_to_index[name]
            trial_x[index] = np.clip(
                trial_x[index] + residual_mean[axis],
                space.lb[index],
                space.ub[index],
            )
    trial_state = decode_design_vector(trial_x, data, space)
    trial_metrics = combined_rmse(
        result.tip,
        result.wrist,
        trial_state.target_tip,
        trial_state.target_wrist,
    )
    if _optimization_objective(trial_metrics, objective_mode, target_rmse_mm) <= (
        _optimization_objective(metrics, objective_mode, target_rmse_mm)
    ):
        return trial_state, trial_metrics
    return state, {key: float(value) for key, value in metrics.items()}


def _project_decoupled_target_rotation_scale_and_translation(
    state: DesignState,
    result: Any,
    metrics: Mapping[str, float],
    data: ProblemData,
    space: DesignSpace,
    objective_mode: str,
    target_rmse_mm: float,
) -> tuple[DesignState, dict[str, float]]:
    """确定性求解两条目标曲线的有界旋转、独立平移和公共尺度。

    平移消去后，正公共尺度下的最优旋转等价于最大化生成曲线与旋转后目标
    曲线的中心化内积。Tip 与 Wrist 的旋转彼此独立；仅对设计空间中完整
    存在 Rx/Ry/Rz 的曲线求解旋转，缺失的旋转分量保持模型指定的固定值。
    随后复用公共尺度的一维精确候选集和六平移闭式投影。
    """

    name_to_index = {name: index for index, name in enumerate(space.names)}
    available_rotation_groups = [
        group for group in _DECOUPLED_ROTATION_GROUPS
        if group.issubset(name_to_index)
    ]
    if not available_rotation_groups:
        return _project_decoupled_target_scale_and_translation(
            state,
            result,
            metrics,
            data,
            space,
            objective_mode,
            target_rmse_mm,
        )

    trial_x = np.asarray(state.x, dtype=float).copy()
    target_definitions = (
        (
            "Tip",
            np.asarray(data.target_tip, dtype=float),
            np.asarray(result.tip, dtype=float),
        ),
        (
            "Wrist",
            np.asarray(data.target_wrist, dtype=float),
            np.asarray(result.wrist, dtype=float),
        ),
    )
    for label, raw_target, generated in target_definitions:
        rotation_names = tuple(
            f"Target_{label}_R{axis}_rad" for axis in ("x", "y", "z")
        )
        if not all(name in name_to_index for name in rotation_names):
            continue
        rotation_indices = np.asarray(
            [name_to_index[name] for name in rotation_names], dtype=np.int32
        )
        lower = space.lb[rotation_indices]
        upper = space.ub[rotation_indices]
        raw_centered = raw_target - np.mean(raw_target, axis=0)
        generated_centered = generated - np.mean(generated, axis=0)

        def alignment_loss_with_gradient(
            angles: np.ndarray,
        ) -> tuple[float, np.ndarray]:
            ax, ay, az = np.asarray(angles, dtype=float)
            sx, cx = math.sin(ax), math.cos(ax)
            sy, cy = math.sin(ay), math.cos(ay)
            sz, cz = math.sin(az), math.cos(az)
            rx = np.array(
                [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]
            )
            ry = np.array(
                [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
            )
            rz = np.array(
                [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]
            )
            drx = np.array(
                [[0.0, 0.0, 0.0], [0.0, -sx, -cx], [0.0, cx, -sx]]
            )
            dry = np.array(
                [[-sy, 0.0, cy], [0.0, 0.0, 0.0], [-cy, 0.0, -sy]]
            )
            drz = np.array(
                [[-sz, -cz, 0.0], [cz, -sz, 0.0], [0.0, 0.0, 0.0]]
            )
            rotation = rz @ ry @ rx
            derivatives = (
                rz @ ry @ drx,
                rz @ dry @ rx,
                drz @ ry @ rx,
            )
            rotated = raw_centered @ rotation.T
            loss = float(-np.sum(generated_centered * rotated))
            gradient = np.asarray(
                [
                    -np.sum(generated_centered * (raw_centered @ derivative.T))
                    for derivative in derivatives
                ],
                dtype=float,
            )
            return loss, gradient

        def alignment_loss(angles: np.ndarray) -> float:
            return alignment_loss_with_gradient(angles)[0]

        current = np.clip(trial_x[rotation_indices], lower, upper)
        midpoint = 0.5 * (lower + upper)
        starts = [current, midpoint, np.clip(np.zeros(3), lower, upper)]
        try:
            kabsch_rotation, _ = Rotation.align_vectors(
                generated_centered, raw_centered
            )
            kabsch = kabsch_rotation.as_euler("xyz")
            starts.append(np.clip(kabsch, lower, upper))
            # 同一旋转的另一组常用 xyz 欧拉角表示可落在不同的有界盆地。
            equivalent = np.array(
                [kabsch[0] + np.pi, np.pi - kabsch[1], kabsch[2] + np.pi]
            )
            equivalent = (equivalent + np.pi) % (2.0 * np.pi) - np.pi
            starts.append(np.clip(equivalent, lower, upper))
        except (ValueError, np.linalg.LinAlgError):
            pass

        best_angles = current.copy()
        best_loss = alignment_loss(best_angles)
        unique_starts: list[np.ndarray] = []
        for start in starts:
            bounded_start = np.clip(np.asarray(start, dtype=float), lower, upper)
            if not any(
                np.allclose(bounded_start, known, atol=1e-10, rtol=0.0)
                for known in unique_starts
            ):
                unique_starts.append(bounded_start)
        # 先比较解析/边界候选，只从其中最好的一点启动一次三维局部求解。
        # 这比对每个起点重复求解快得多，适合在数十万次 fourbar 评价中使用。
        local_start = current.copy()
        local_start_loss = best_loss
        for start in unique_starts:
            start_loss = alignment_loss(start)
            if start_loss < best_loss:
                best_loss = start_loss
                best_angles = start.copy()
            if start_loss < local_start_loss:
                local_start_loss = start_loss
                local_start = start.copy()
        try:
            optimized = minimize(
                alignment_loss_with_gradient,
                local_start,
                method="L-BFGS-B",
                jac=True,
                bounds=list(zip(lower, upper)),
                options={"maxiter": 25, "ftol": 1e-10, "gtol": 1e-7},
            )
            optimized_angles = np.clip(optimized.x, lower, upper)
            optimized_loss = alignment_loss(optimized_angles)
            if optimized_loss < best_loss:
                best_loss = optimized_loss
                best_angles = optimized_angles
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            pass
        trial_x[rotation_indices] = best_angles

    trial_state = decode_design_vector(trial_x, data, space)
    trial_metrics = combined_rmse(
        result.tip,
        result.wrist,
        trial_state.target_tip,
        trial_state.target_wrist,
    )
    return _project_decoupled_target_scale_and_translation(
        trial_state,
        result,
        trial_metrics,
        data,
        space,
        objective_mode,
        target_rmse_mm,
    )


def _project_decoupled_target_scale_and_translation(
    state: DesignState,
    result: Any,
    metrics: Mapping[str, float],
    data: ProblemData,
    space: DesignSpace,
    objective_mode: str,
    target_rmse_mm: float,
) -> tuple[DesignState, dict[str, float]]:
    """在不重算机构的条件下内层求公共尺度和两条曲线的独立平移。

    六个平移具有闭式最小二乘解；公共尺度只是一维有界变量。把这七个变量
    放到每次真实机构评价的内层求解，可减少全局 CMA/SQP 的有效维数，同时
    仍在最终结果中完整输出它们。
    """

    name_to_index = {name: index for index, name in enumerate(space.names)}
    scale_index = name_to_index.get("Target_Scale")
    if scale_index is None:
        return _project_decoupled_target_translation(
            state,
            result,
            metrics,
            data,
            space,
            objective_mode,
            target_rmse_mm,
        )

    base_x = np.asarray(state.x, dtype=float).copy()
    best_state, best_metrics = _project_decoupled_target_translation(
        state,
        result,
        metrics,
        data,
        space,
        objective_mode,
        target_rmse_mm,
    )
    best_score = _optimization_objective(
        best_metrics, objective_mode, target_rmse_mm
    )
    def evaluate_scale(scale: float) -> float:
        nonlocal best_state, best_metrics, best_score
        bounded = float(np.clip(scale, space.lb[scale_index], space.ub[scale_index]))
        trial_x = base_x.copy()
        trial_x[scale_index] = bounded
        trial_state = decode_design_vector(trial_x, data, space)
        trial_metrics = combined_rmse(
            result.tip,
            result.wrist,
            trial_state.target_tip,
            trial_state.target_wrist,
        )
        trial_state, trial_metrics = _project_decoupled_target_translation(
            trial_state,
            result,
            trial_metrics,
            data,
            space,
            objective_mode,
            target_rmse_mm,
        )
        score = _optimization_objective(
            trial_metrics, objective_mode, target_rmse_mm
        )
        if score < best_score:
            best_score = float(score)
            best_state = trial_state
            best_metrics = trial_metrics
        return float(score)

    # 平移消去后，每条曲线的平方误差关于尺度都是凸二次函数。最小最大解
    # 必定位于某条二次函数的极小点、两条函数交点或尺度边界。
    unit_x = base_x.copy()
    unit_x[scale_index] = 1.0
    unit_state = decode_design_vector(unit_x, data, space)
    quadratic_terms: list[tuple[float, float, float]] = []
    for generated, target in (
        (result.tip, unit_state.target_tip),
        (result.wrist, unit_state.target_wrist),
    ):
        generated_centered = generated - np.mean(generated, axis=0)
        target_centered = target - np.mean(target, axis=0)
        count = max(1, generated.shape[0])
        a = float(np.sum(target_centered * target_centered) / count)
        b = float(-2.0 * np.sum(generated_centered * target_centered) / count)
        c = float(np.sum(generated_centered * generated_centered) / count)
        quadratic_terms.append((a, b, c))
    lower = float(space.lb[scale_index])
    upper = float(space.ub[scale_index])
    candidates = {
        lower,
        upper,
        float(np.clip(base_x[scale_index], lower, upper)),
    }
    for a, b, _c in quadratic_terms:
        if a > 1e-12:
            candidates.add(float(np.clip(-b / (2.0 * a), lower, upper)))
    total_a = sum(term[0] for term in quadratic_terms)
    total_b = sum(term[1] for term in quadratic_terms)
    if total_a > 1e-12:
        candidates.add(float(np.clip(-total_b / (2.0 * total_a), lower, upper)))
    difference = np.asarray(quadratic_terms[0]) - np.asarray(quadratic_terms[1])
    roots = np.roots(difference) if abs(difference[0]) > 1e-12 else (
        np.asarray([-difference[2] / difference[1]])
        if abs(difference[1]) > 1e-12
        else np.asarray([])
    )
    for root in roots:
        if abs(float(np.imag(root))) <= 1e-10:
            real_root = float(np.real(root))
            if lower <= real_root <= upper:
                candidates.add(real_root)
    for scale in sorted(candidates):
        evaluate_scale(scale)
    return best_state, best_metrics


def _project_shared_target_scale_and_translation(
    state: DesignState,
    result: Any,
    metrics: Mapping[str, float],
    data: ProblemData,
    space: DesignSpace,
    objective_mode: str,
    target_rmse_mm: float,
) -> tuple[DesignState, dict[str, float]]:
    """在固定公共旋转下，解析求解公共尺度与公共平移。

    Tip 和 Wrist 始终使用同一个刚体变换。公共尺度是一维有界变量；对每个
    尺度候选，公共平移由两条 RMSE 的最小最大平衡线搜索确定。
    """

    name_to_index = {name: index for index, name in enumerate(space.names)}
    scale_index = name_to_index.get("Target_Scale")
    if scale_index is None:
        return _project_shared_target_translation(
            state,
            result,
            metrics,
            data,
            space,
            objective_mode,
            target_rmse_mm,
        )

    base_x = np.asarray(state.x, dtype=float).copy()
    best_state, best_metrics = _project_shared_target_translation(
        state,
        result,
        metrics,
        data,
        space,
        objective_mode,
        target_rmse_mm,
    )
    best_score = _optimization_objective(
        best_metrics, objective_mode, target_rmse_mm
    )

    def evaluate_scale(scale: float) -> float:
        nonlocal best_state, best_metrics, best_score
        trial_x = base_x.copy()
        trial_x[scale_index] = np.clip(
            scale,
            space.lb[scale_index],
            space.ub[scale_index],
        )
        trial_state = decode_design_vector(trial_x, data, space)
        trial_metrics = combined_rmse(
            result.tip,
            result.wrist,
            trial_state.target_tip,
            trial_state.target_wrist,
        )
        trial_state, trial_metrics = _project_shared_target_translation(
            trial_state,
            result,
            trial_metrics,
            data,
            space,
            objective_mode,
            target_rmse_mm,
        )
        score = _optimization_objective(
            trial_metrics, objective_mode, target_rmse_mm
        )
        if score < best_score:
            best_score = float(score)
            best_state = trial_state
            best_metrics = trial_metrics
        return float(score)

    # 去除公共平移后，每条曲线的均方误差关于尺度均为凸二次函数。候选集
    # 包含两条曲线各自极小点、总误差极小点、两条误差曲线交点和硬边界。
    unit_x = base_x.copy()
    unit_x[scale_index] = 1.0
    unit_state = decode_design_vector(unit_x, data, space)
    quadratic_terms: list[tuple[float, float, float]] = []
    for generated, target in (
        (result.tip, unit_state.target_tip),
        (result.wrist, unit_state.target_wrist),
    ):
        generated_centered = generated - np.mean(generated, axis=0)
        target_centered = target - np.mean(target, axis=0)
        count = max(1, generated.shape[0])
        quadratic_terms.append((
            float(np.sum(target_centered * target_centered) / count),
            float(-2.0 * np.sum(generated_centered * target_centered) / count),
            float(np.sum(generated_centered * generated_centered) / count),
        ))

    lower = float(space.lb[scale_index])
    upper = float(space.ub[scale_index])
    candidates = {
        lower,
        upper,
        float(np.clip(base_x[scale_index], lower, upper)),
    }
    for a, b, _c in quadratic_terms:
        if a > 1e-12:
            candidates.add(float(np.clip(-b / (2.0 * a), lower, upper)))
    total_a = sum(term[0] for term in quadratic_terms)
    total_b = sum(term[1] for term in quadratic_terms)
    if total_a > 1e-12:
        candidates.add(float(np.clip(-total_b / (2.0 * total_a), lower, upper)))
    difference = np.asarray(quadratic_terms[0]) - np.asarray(quadratic_terms[1])
    roots = np.roots(difference) if abs(difference[0]) > 1e-12 else (
        np.asarray([-difference[2] / difference[1]])
        if abs(difference[1]) > 1e-12
        else np.asarray([])
    )
    for root in roots:
        if abs(float(np.imag(root))) <= 1e-10:
            real_root = float(np.real(root))
            if lower <= real_root <= upper:
                candidates.add(real_root)
    for scale in sorted(candidates):
        evaluate_scale(scale)
    return best_state, best_metrics


def _project_shared_target_pose(
    state: DesignState,
    result: Any,
    metrics: Mapping[str, float],
    data: ProblemData,
    space: DesignSpace,
    objective_mode: str,
    target_rmse_mm: float,
) -> tuple[DesignState, dict[str, float]]:
    """用有界加权 Procrustes 解旋转/尺度，再平衡求解公共平移。

    旋转、尺度和平移仍是设计变量，只是改由每个机构候选内部的解析配准
    直接求解，避免外层 CMA-ES 在窄闭环可行域中同时猜测机构和目标位姿。
    使用多组 Tip/Wrist 权重产生候选位姿，再以完整双目标函数选择最终结果。
    """

    decoupled_translation_names = (
        "Target_Tip_Tx_mm", "Target_Tip_Ty_mm", "Target_Tip_Tz_mm",
        "Target_Wrist_Tx_mm", "Target_Wrist_Ty_mm", "Target_Wrist_Tz_mm",
    )
    if all(name in space.names for name in decoupled_translation_names):
        if any(group.issubset(space.names) for group in _DECOUPLED_ROTATION_GROUPS):
            return _project_decoupled_target_rotation_scale_and_translation(
                state,
                result,
                metrics,
                data,
                space,
                objective_mode,
                target_rmse_mm,
            )
        # 旋转自由度不完整的解耦模式继续由外层搜索允许的角度；六个平移和
        # 公共尺度仍在每个可行机构候选内直接求解。
        return _project_decoupled_target_scale_and_translation(
            state,
            result,
            metrics,
            data,
            space,
            objective_mode,
            target_rmse_mm,
        )

    best_state, best_metrics = _project_shared_target_scale_and_translation(
        state,
        result,
        metrics,
        data,
        space,
        objective_mode,
        target_rmse_mm,
    )
    best_score = _optimization_objective(
        best_metrics, objective_mode, target_rmse_mm
    )
    raw_target = np.vstack([data.target_tip, data.target_wrist])
    generated = np.vstack([result.tip, result.wrist])
    base_x = np.asarray(state.x, dtype=float).copy()
    name_to_index = {name: index for index, name in enumerate(space.names)}
    rigid_pose_names = TARGET_POSE_NAMES[:6]
    if not all(name in name_to_index for name in rigid_pose_names):
        # 受限 Ty/Ry 模式不允许 Procrustes 重新引入 Tx/Tz 或缺失的旋转轴。
        # Ry 由外层区域 CMA-ES/SLSQP 优化，Ty 仍由上面的有界一维平衡投影精确处理。
        return best_state, best_metrics
    rotation_indices = [name_to_index[name] for name in TARGET_POSE_NAMES[3:6]]
    scale_index = name_to_index.get(TARGET_POSE_NAMES[6])
    target_center = np.mean(raw_target, axis=0)
    frame_count = data.target_tip.shape[0]

    for tip_weight in (0.20, 0.35, 0.50, 0.65, 0.80):
        weights = np.concatenate([
            np.full(frame_count, tip_weight / frame_count),
            np.full(frame_count, (1.0 - tip_weight) / frame_count),
        ])
        raw_mean = np.sum(raw_target * weights[:, None], axis=0)
        generated_mean = np.sum(generated * weights[:, None], axis=0)
        raw_centered = raw_target - raw_mean
        generated_centered = generated - generated_mean
        covariance = raw_centered.T @ (weights[:, None] * generated_centered)
        left, _singular, right_t = np.linalg.svd(covariance, full_matrices=False)
        row_rotation = left @ right_t
        if np.linalg.det(row_rotation) < 0.0:
            left[:, -1] *= -1.0
            row_rotation = left @ right_t
        rotation_matrix = row_rotation.T
        euler_xyz = Rotation.from_matrix(rotation_matrix).as_euler("xyz")
        trial_x = base_x.copy()
        for pose_index, value in zip(rotation_indices, euler_xyz):
            trial_x[pose_index] = np.clip(
                value, space.lb[pose_index], space.ub[pose_index]
            )

        # 在有界欧拉角截断后重新计算最小二乘尺度，保证尺度与实际旋转一致。
        unit_pose = np.array([
            0.0,
            0.0,
            0.0,
            trial_x[rotation_indices[0]],
            trial_x[rotation_indices[1]],
            trial_x[rotation_indices[2]],
            1.0,
        ])
        rotated = np.vstack([
            apply_target_pose(data.target_tip, unit_pose, target_center),
            apply_target_pose(data.target_wrist, unit_pose, target_center),
        ])
        rotated_mean = np.sum(rotated * weights[:, None], axis=0)
        rotated_centered = rotated - rotated_mean
        denominator = float(np.sum(weights[:, None] * rotated_centered ** 2))
        scale = (
            float(np.sum(weights[:, None] * rotated_centered * generated_centered))
            / max(denominator, 1e-12)
        )
        if scale_index is not None:
            trial_x[scale_index] = np.clip(
                scale, space.lb[scale_index], space.ub[scale_index]
            )
        trial_state = decode_design_vector(trial_x, data, space)
        trial_metrics = combined_rmse(
            result.tip,
            result.wrist,
            trial_state.target_tip,
            trial_state.target_wrist,
        )
        trial_state, trial_metrics = _project_shared_target_translation(
            trial_state,
            result,
            trial_metrics,
            data,
            space,
            objective_mode,
            target_rmse_mm,
        )
        trial_score = _optimization_objective(
            trial_metrics, objective_mode, target_rmse_mm
        )
        if trial_score < best_score:
            best_state = trial_state
            best_metrics = trial_metrics
            best_score = trial_score
    return best_state, best_metrics


def _evaluate_candidate(
    y: np.ndarray,
    data: ProblemData,
    space: DesignSpace,
    region_id: int,
    stage: str,
    objective_mode: str = "balanced_rmse_target",
    target_rmse_mm: float = 30.0,
) -> _Candidate:
    """调用 fourbar 模型计算一个候选点，并把不可装配状态赋为大惩罚值。"""

    requested_y = np.clip(np.asarray(y, dtype=float).copy(), 0.0, 1.0)
    x = _to_physical(requested_y, space)
    try:
        result, metrics, state = evaluate_design_vector(
            x, data, space, check_smooth=True
        )
    except Exception:
        # 边界候选可能在解码阶段就使 B 穿过 A 或破坏显式装配条件；
        # 这类点是普通不可行样本，不应中断全局优化。
        result = None
        state = None
        metrics = {
            "combined": 1e6, "tip": 1e6, "wrist": 1e6,
            "tip_peak": 1e6, "wrist_peak": 1e6,
            "tip_high_harmonic_rms": 1e6,
            "wrist_high_harmonic_rms": 1e6,
            "wrist_plane_mismatch_deg": 1e6,
            "stutter_index": 1e6,
            "stutter_penalty_mm": 1e6,
            "tip_minimum_speed_ratio": 0.0,
            "wrist_minimum_speed_ratio": 0.0,
            "tip_maximum_adjacent_speed_ratio": 1e6,
            "wrist_maximum_adjacent_speed_ratio": 1e6,
        }
    valid = result is not None and bool(result.valid)
    if valid:
        metrics["tip_high_harmonic_rms"] = _high_harmonic_rms(result.tip)
        metrics["wrist_high_harmonic_rms"] = _high_harmonic_rms(result.wrist)
        metrics["wrist_plane_mismatch_deg"] = _plane_normal_mismatch_deg(
            result.wrist, state.target_wrist
        )
        # Strict initialized phase is part of the objective contract. Do not compute
        # or apply a candidate-specific cyclic alignment here.
    score = (
        _optimization_objective(metrics, objective_mode, target_rmse_mm)
        if valid
        else 1e6
    )
    actual_x = state.x if state is not None and state.x is not None else x
    snapshot = None
    if valid:
        angle_names = tuple(result.angle_history.keys())
        snapshot = {
            "tip": result.tip.copy(),
            "wrist": result.wrist.copy(),
            "target_tip": state.target_tip.copy(),
            "target_wrist": state.target_wrist.copy(),
            "b_curve": result.b_curve.copy(),
            "l2_values": np.asarray(state.l2_values, dtype=float).copy(),
            "l31_values": np.asarray(state.l31_values, dtype=float).copy(),
            "l32_values": np.asarray(state.l32_values, dtype=float).copy(),
            "l3_values": np.asarray(state.l3_values, dtype=float).copy(),
            "l6_values": result.l6_values.copy(),
            "l7_values": state.l7_values.copy(),
            "l8_values": np.asarray(state.l8_values, dtype=float).copy(),
            "l12_values": np.asarray(state.l12_values, dtype=float).copy(),
            "zc_values": np.asarray(state.zc_values, dtype=float).copy(),
            "angle_names": angle_names,
            "angle_history": np.column_stack(
                [result.angle_history[name] for name in angle_names]
            ),
        }
    return _Candidate(
        # CMA-ES 必须学习修复后的真实位置，否则均值会持续停留在不可行空间。
        y=_to_normalized(actual_x, space),
        x=np.asarray(actual_x, dtype=float).copy(),
        score=float(score),
        metrics={key: float(value) for key, value in metrics.items()},
        valid=valid,
        region_id=region_id,
        stage=stage,
        requested_y=requested_y,
        snapshot=snapshot,
    )


# =============================================================================
# 3. 完整优化：原始起点、区域 CMA-ES、SLSQP 和灵敏度分区
# =============================================================================


def _triangle_violation(a: float, b: float, c: float) -> float:
    """三角形装配与 5-175 deg 角度裕量的无量纲违反量。"""
    sides = np.asarray([a, b, c], dtype=float)
    if np.any(~np.isfinite(sides)) or np.any(sides <= 0.0):
        return 10.0
    largest = float(np.max(sides))
    scale = max(float(np.sum(sides)), 1e-9)
    triangle_loss = max(
        0.0,
        (2.0 * largest - float(np.sum(sides)) + 1e-8) / scale,
    )
    if triangle_loss > 0.0:
        return triangle_loss
    angles = np.asarray([
        np.arccos(np.clip((b * b + c * c - a * a) / (2.0 * b * c), -1.0, 1.0)),
        np.arccos(np.clip((a * a + c * c - b * b) / (2.0 * a * c), -1.0, 1.0)),
        np.arccos(np.clip((a * a + b * b - c * c) / (2.0 * a * b), -1.0, 1.0)),
    ])
    angle_shortfall = max(
        0.0,
        float(CLOSURE_ANGLE_MIN_RAD - np.min(angles)),
        float(np.max(angles) - CLOSURE_ANGLE_MAX_RAD),
    )
    return angle_shortfall / np.pi


def _qa2_violation(
    ab: float,
    bc: float,
    cd: float,
    da: float,
    angle_a: float,
) -> float:
    """QA2 对角线安全净距和闭环角度的连续违反量。"""
    values = np.asarray([ab, bc, cd, da, angle_a], dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values[:4] <= 0.0):
        return 10.0
    angle_shortfall = max(
        0.0,
        float(CLOSURE_ANGLE_MIN_RAD - angle_a),
        float(angle_a - CLOSURE_ANGLE_MAX_RAD),
    ) / np.pi
    bd2 = ab * ab + da * da - 2.0 * ab * da * np.cos(angle_a)
    if (not np.isfinite(bd2)) or bd2 <= 0.0:
        return 2.0
    diagonal = float(np.sqrt(bd2))
    scale = max(float(bc + cd), 1e-9)
    clearance = min(
        diagonal - abs(float(bc - cd)),
        float(bc + cd) - diagonal,
    )
    required = qa2_required_clearance_mm(float(bc), float(cd))
    clearance_shortfall = max(0.0, required - clearance) / scale
    if clearance > 0.0:
        angle_b_abd = np.arccos(np.clip(
            (ab * ab + diagonal * diagonal - da * da)
            / (2.0 * ab * diagonal),
            -1.0,
            1.0,
        ))
        angle_d_abd = np.arccos(np.clip(
            (diagonal * diagonal + da * da - ab * ab)
            / (2.0 * diagonal * da),
            -1.0,
            1.0,
        ))
        angle_b_bdc = np.arccos(np.clip(
            (bc * bc + diagonal * diagonal - cd * cd)
            / (2.0 * bc * diagonal),
            -1.0,
            1.0,
        ))
        angle_c_bdc = np.arccos(np.clip(
            (bc * bc + cd * cd - diagonal * diagonal)
            / (2.0 * bc * cd),
            -1.0,
            1.0,
        ))
        angle_d_bdc = np.arccos(np.clip(
            (diagonal * diagonal + cd * cd - bc * bc)
            / (2.0 * diagonal * cd),
            -1.0,
            1.0,
        ))
        output_angles = np.asarray([
            angle_b_abd + angle_b_bdc,
            angle_c_bdc,
            angle_d_abd + angle_d_bdc,
        ])
        angle_shortfall = max(
            angle_shortfall,
            max(
                0.0,
                float(CLOSURE_ANGLE_MIN_RAD - np.min(output_angles)),
                float(np.max(output_angles) - CLOSURE_ANGLE_MAX_RAD),
            ) / np.pi,
        )
    return angle_shortfall + clearance_shortfall


def _staged_feasibility_loss(
    x: np.ndarray,
    data: ProblemData,
    space: DesignSpace,
) -> tuple[float, str]:
    """用逐级闭环余量评价无效候选，使可行性搜索具有方向信息。

    返回值的整数部分表示尚未通过的机构区域，越小越接近完整装配；同一
    区域内的小数部分表示归一化几何违反量。完整 fourbar 可运行时返回 0。
    """
    try:
        state = decode_design_vector(x, data, space)
        p = params_from_static(state.static, data)
        l2_curve = (
            np.asarray(state.l2_values, dtype=float).reshape(-1)
            if state.l2_values is not None
            else np.full(76, p.L2, dtype=float)
        )
        l31_curve = np.asarray(state.l31_values, dtype=float).reshape(-1)
        l32_curve = (
            np.asarray(state.l32_values, dtype=float).reshape(-1)
            if state.l32_values is not None
            else np.full(l31_curve.size, p.L32, dtype=float)
        )
        l3_curve = np.asarray(state.l3_values, dtype=float).reshape(-1)
        l5_curve = np.asarray(state.l5_values, dtype=float).reshape(-1)
        l6_curve = np.asarray(state.l6_values, dtype=float).reshape(-1)
        l7_curve = np.asarray(state.l7_values, dtype=float).reshape(-1)
        l8_curve = np.asarray(state.l8_values, dtype=float).reshape(-1)
        l12_curve = np.asarray(state.l12_values, dtype=float).reshape(-1)
        zc_curve = (
            np.asarray(state.zc_values, dtype=float).reshape(-1)
            if state.zc_values is not None
            else np.full(76, p.L_CZ, dtype=float)
        )
        use_zc_extension = (
            "ZC_C0" in space.names or "ZC_Amplitude_mm" in space.names
        )
        split_zc_extension = "ZC_Amplitude_mm" in space.names
        input_radius, theta01, _theta02, _input_norm = motcurve_to_input_angles(
            state.b_curve
        )
        n_frames = len(theta01)
        theta32 = np.zeros(n_frames)
        theta33 = np.zeros(n_frames)
        static_violation = 0.0
        for frame in range(n_frames):
            static_violation = max(
                static_violation,
                _triangle_violation(
                    float(l31_curve[frame]),
                    float(l32_curve[frame]),
                    float(l3_curve[frame]),
                ),
            )
            if static_violation <= 0.0:
                _theta31, theta32[frame], theta33[frame] = t_angles(
                    float(l31_curve[frame]),
                    float(l32_curve[frame]),
                    float(l3_curve[frame]),
                )
        if static_violation > 0.0:
            return 7.0 + static_violation, "static_triangle"

        cy = np.zeros(n_frames)
        theta2 = np.zeros(n_frames)
        theta3 = np.zeros(n_frames)
        theta4 = np.zeros(n_frames)
        input_violation = 0.0
        for frame in range(n_frames):
            angle = float(theta01[frame])
            try:
                cy[frame] = triangle_third_side_non_adjacent(
                    float(l2_curve[frame]), float(input_radius[frame]), angle
                )
            except FourBarError:
                input_violation = max(input_violation, 1.0)
                continue
            input_violation = max(
                input_violation,
                _triangle_violation(
                    float(l2_curve[frame]), float(input_radius[frame]), cy[frame]
                ),
            )
        if input_violation > 0.0:
            return 6.0 + input_violation, "input_triangle"
        if split_zc_extension:
            zc_curve = np.roll(zc_curve, int(np.argmax(cy)))
        c_minimum_y_radius = float(np.max(cy))
        zone2_violation = 0.0
        for frame in range(n_frames):
            base_distance = (
                c_minimum_y_radius + zc_curve[frame]
                if use_zc_extension else cy[frame]
            )
            zone2_violation = max(
                zone2_violation,
                _triangle_violation(
                    base_distance, p.L4, float(l31_curve[frame])
                ),
            )
            if zone2_violation <= 0.0:
                theta4[frame], theta2[frame], theta3[frame] = t_angles(
                    base_distance, p.L4, float(l31_curve[frame])
                )
        if zone2_violation > 0.0:
            return 5.0 + zone2_violation, "zone_II"

        theta5 = 2.0 * np.pi - theta4 - theta33
        theta6 = np.zeros(n_frames)
        theta7 = np.zeros(n_frames)
        theta8 = np.zeros(n_frames)
        violation = 0.0
        for frame in range(n_frames):
            violation = max(
                violation,
                _qa2_violation(
                    p.L41,
                    float(l5_curve[frame]) - p.L51 - p.L52,
                    float(l6_curve[frame]) - p.L61,
                    float(l32_curve[frame]),
                    float(theta5[frame]),
                ),
            )
            if violation <= 0.0:
                theta6[frame], theta7[frame], theta8[frame] = qa2(
                    p.L41,
                    float(l5_curve[frame]) - p.L51 - p.L52,
                    float(l6_curve[frame]) - p.L61,
                    float(l32_curve[frame]),
                    float(theta5[frame]),
                )
        if violation > 0.0:
            return 4.0 + violation, "zone_III"

        theta9 = np.pi - theta7
        theta10 = np.zeros(n_frames)
        theta11 = np.zeros(n_frames)
        theta12 = np.zeros(n_frames)
        violation = 0.0
        for frame in range(n_frames):
            violation = max(
                violation,
                _qa2_violation(
                    p.L52,
                    float(l7_curve[frame]),
                    p.L10,
                    float(l6_curve[frame]),
                    float(theta9[frame]),
                ),
            )
            if violation <= 0.0:
                theta10[frame], theta11[frame], theta12[frame] = qa2(
                    p.L52,
                    float(l7_curve[frame]),
                    p.L10,
                    float(l6_curve[frame]),
                    float(theta9[frame]),
                )
        if violation > 0.0:
            return 3.0 + violation, "zone_IV"

        theta13 = np.pi - theta10
        theta14 = np.zeros(n_frames)
        theta15 = np.zeros(n_frames)
        theta16 = np.zeros(n_frames)
        violation = 0.0
        for frame in range(n_frames):
            violation = max(
                violation,
                _qa2_violation(
                    p.L51,
                    float(l8_curve[frame]),
                    p.L9,
                    float(l7_curve[frame]),
                    float(theta13[frame]),
                ),
            )
            if violation <= 0.0:
                theta14[frame], theta15[frame], theta16[frame] = qa2(
                    p.L51,
                    float(l8_curve[frame]),
                    p.L9,
                    float(l7_curve[frame]),
                    float(theta13[frame]),
                )
        if violation > 0.0:
            return 2.0 + violation, "zone_V"

        theta18 = np.deg2rad(p.theta18_deg)
        theta21 = 2.0 * np.pi - theta11 - theta16 - theta18
        theta17 = np.zeros(n_frames)
        theta20 = np.zeros(n_frames)
        theta19 = np.zeros(n_frames)
        violation = max(0.0, -float(np.min(theta21)) / np.pi)
        for frame in range(n_frames):
            violation = max(
                violation,
                _qa2_violation(
                    p.L10,
                    p.L11,
                    p.L13,
                    float(l12_curve[frame]),
                    float(theta21[frame]),
                ),
            )
            if violation <= 0.0:
                theta17[frame], theta20[frame], theta19[frame] = qa2(
                    p.L10,
                    p.L11,
                    p.L13,
                    float(l12_curve[frame]),
                    float(theta21[frame]),
                )
        if violation > 0.0:
            return 1.0 + violation, "zone_VI"

        # 后两级角度由前六个闭环共同决定。把负角违反量连续化，避免所有
        # “接近可装配”的方案都被压成同一个失败分数，给 CMA-ES 明确方向。
        theta_m = theta9 + theta6 + theta3 - theta18 - theta16
        theta_n = theta7 + theta6 + theta3 - theta17 - theta12
        alpha = np.pi - theta6 - theta3
        angle_min = min(
            float(np.min(theta5)),
            float(np.min(alpha)),
            float(np.min(theta_m)),
            float(np.min(theta_n)),
        )
        if angle_min < 0.0:
            return 0.50 + min(0.45, -angle_min / np.pi), "angle_sign"

        # 在局部二维平面中预计算 M-J-K 腕部四边形。theta02 是刚体旋转，
        # 不改变这里的长度和内积，因此该判据与三维求解完全一致。
        wrist_violation = 0.0
        wrist_rod = p.LRod if p.LRod is not None and np.isfinite(p.LRod) else p.L12
        for frame in range(n_frames):
            base_y = -(cy[frame] + zc_curve[frame]) if use_zc_extension else -cy[frame]
            e = np.array(
                [
                    -(p.L4 - p.L41) * np.sin(theta3[frame]),
                    base_y + (p.L4 - p.L41) * np.cos(theta3[frame]),
                ],
                dtype=float,
            )
            h = e + np.array(
                [
                    -np.sin(alpha[frame]) * (l5_curve[frame] - p.L51),
                    -np.cos(alpha[frame]) * (l5_curve[frame] - p.L51),
                ],
                dtype=float,
            )
            i_point = e + np.array(
                [
                    -np.sin(alpha[frame]) * l5_curve[frame],
                    -np.cos(alpha[frame]) * l5_curve[frame],
                ],
                dtype=float,
            )
            j = i_point + np.array(
                [
                    -np.sin(theta14[frame] - alpha[frame]) * l8_curve[frame],
                    np.cos(theta14[frame] - alpha[frame]) * l8_curve[frame],
                ],
                dtype=float,
            )
            k = h + np.array(
                [
                    -np.sin(theta10[frame] - alpha[frame]) * l7_curve[frame],
                    np.cos(theta10[frame] - alpha[frame]) * l7_curve[frame],
                ],
                dtype=float,
            )
            m = k + np.array(
                [
                    -np.sin(theta_m[frame]) * l12_curve[frame],
                    np.cos(theta_m[frame]) * l12_curve[frame],
                ],
                dtype=float,
            )
            mk = k - m
            mj = j - m
            mk_norm = float(np.linalg.norm(mk))
            if mk_norm <= 1e-12:
                wrist_violation = max(wrist_violation, 1.0)
                continue
            d_mj = float(np.dot(mj, mk) / mk_norm)
            mj_down = float(np.hypot(d_mj, p.L_down))
            wrist_violation = max(
                wrist_violation,
                _qa2_violation(
                    p.L14, mj_down, wrist_rod, p.L_down, np.pi / 2.0
                ),
            )
        if wrist_violation > 0.0:
            return 0.25 + min(0.24, wrist_violation), "wrist_quadrilateral"

        result, _metrics = evaluate_design_state(state, data, check_smooth=True)
        if result is None:
            return 0.10, "coordinates_or_continuity"
        return 0.0, "feasible"
    except (FourBarError, ValueError, FloatingPointError):
        return 9.0, "decode_or_geometry"


def _find_feasible_initial(
    data: ProblemData,
    space: DesignSpace,
    config: _OptimizeConfig,
    evaluation_counter: list[int] | None = None,
    recorder: _OptimizationRecorder | None = None,
) -> _Candidate:
    """只做可行性搜索，不引用历史最优解，因此不构成锚点。"""

    def evaluate(y: np.ndarray, stage: str) -> _Candidate:
        if evaluation_counter is not None:
            evaluation_counter[0] += 1
            evaluation = evaluation_counter[0]
        else:
            evaluation = len(recorder.candidate_rows) + 1 if recorder is not None else -1
        candidate = _evaluate_candidate(
            y, data, space, 0, stage,
            config.objective_mode, config.target_rmse_mm,
        )
        if recorder is not None:
            recorder.record_candidate(
                candidate,
                evaluation,
                round_index=-1,
                metadata={"purpose": "no-anchor feasible initialization"},
            )
        return candidate

    y0 = _to_normalized(space.x0, space)
    initial = evaluate(y0, "initial")
    if initial.valid:
        return initial
    directed_seed_y = y0.copy()
    directed_seed_loss, directed_seed_stage = _staged_feasibility_loss(
        initial.x, data, space
    )

    # 动态 ZC 新拓扑增加了 A-Z-D 三角形。以下种子只取当前上下界中的工程值，
    # 用来把搜索从原始 zone-II 不可装配状态引导到后级闭环，不读取任何历史解。
    if space.periodic_length_mode in {
        "l5_l8_zc_split_periodic3",
        "l5_l6_l8_zc_split_periodic3",
        "zc_l7_periodic_l3_fixed",
        "zc_l3_l7_periodic",
        "zc_l3_l7_periodic3",
        "zc_l3_l7_periodic4",
        "zc_l7_l8_periodic4_l3_fixed",
    }:
        if space.periodic_length_mode in _L5_L8_ZC_SPLIT_MODES:
            # 这些整数化工程值仅由本模式的杆长边界与六级闭环约束构造。
            # 目标位姿保持零，且该种子的原始 RMSE 很高，因此只承担无锚点
            # 可行化作用，不向性能搜索注入历史最优方向。
            seed_definitions = (
                {
                    "L2": 230.0, "L31": 95.0, "L4": 81.0,
                    "L41": 8.0, "L51": 32.0, "L52": 26.0,
                    (
                        "L6_C0"
                        if space.periodic_length_mode
                        == "l5_l6_l8_zc_split_periodic3"
                        else "L6"
                    ): 249.0,
                    "L7": 218.0, "L3": 237.0,
                    "L61": 197.0, "L5_C0": 195.0, "L8_C0": 210.0,
                    "ZC_Amplitude_mm": 18.0,
                    "ZC_ShapeCos": 0.0, "ZC_ShapeSin": 0.0,
                    "L9": 36.0, "L10": 49.0, "L11": 31.0,
                    "L12": 17.0, "L13": 45.0, "LRod": 48.0,
                    "L14": 41.0, "L15": 21.0, "L17": 21.0,
                    "H_finger": 3.0, "Lf1": 376.0, "Lf2": 319.0,
                    "L_down": 74.0, "theta18_deg": 45.0,
                    "B_Y_C0": 13.0, "B_Y_C1c": 23.0,
                    "B_Y_C1s": 0.0, "B_Y_C2c": 0.0, "B_Y_C2s": 0.0,
                    "B_Z_C1c": 0.0, "B_Z_C1s": 59.0,
                    "B_Z_C2c": 0.0, "B_Z_C2s": 0.0,
                    "B_CenterX": 199.0, "B_Z_C3c": 0.0, "B_Z_C3s": 0.0,
                },
            )
        elif space.periodic_length_mode == "zc_l7_periodic_l3_fixed":
            seed_definitions = (
            {"L2": 250.0, "L31": 150.0, "L4": 100.0, "L3": 180.0, "ZC_C0": 10.0, "LRod": 30.0},
            {"L2": 220.0, "L31": 130.0, "L4": 100.0, "L3": 180.0, "ZC_C0": 10.0, "LRod": 30.0},
            {"L2": 250.0, "L31": 150.0, "L4": 90.0, "L3": 160.0, "ZC_C0": 10.0, "LRod": 30.0},
            )
        elif space.periodic_length_mode == "zc_l7_l8_periodic4_l3_fixed":
            seed_definitions = (
                {
                    "L2": 220.0, "L31": 120.0, "L4": 80.0,
                    "L41": 20.0, "L5": 180.0, "L51": 15.0, "L52": 30.0,
                    "L6": 290.0, "L61": 200.0, "L3": 100.0,
                    "L7_C0": 270.0, "L8_C0": 260.0, "ZC_C0": 10.0,
                    "L9": 30.0, "L10": 30.0, "L11": 20.0,
                    "L12": 15.0, "L13": 15.0, "LRod": 50.0,
                    "L14": 20.0, "L15": 20.0, "L17": 20.0,
                    "L_down": 15.0, "theta18_deg": 120.0,
                },
                {
                    "L2": 200.0, "L31": 100.0, "L4": 70.0,
                    "L41": 20.0, "L5": 180.0, "L51": 15.0, "L52": 25.0,
                    "L6": 285.0, "L61": 190.0, "L3": 100.0,
                    "L7_C0": 260.0, "L8_C0": 250.0, "ZC_C0": 10.0,
                    "L9": 35.0, "L10": 30.0, "L11": 20.0,
                    "L12": 15.0, "L13": 15.0, "LRod": 45.0,
                    "L14": 20.0, "L15": 20.0, "L17": 20.0,
                    "L_down": 15.0, "theta18_deg": 100.0,
                },
            )
        else:
            # 动态 ZC 与 L3 同时存在时，原始参数首先在第 II/III 闭环失效。
            # 以下数值只由杆长边界和三角不等式构造：L3、ZC 取工程下界，
            # 再为前三级闭环留出长度裕量；不读取任何历史优化向量。
            seed_definitions = (
                {
                    "L2": 220.0, "L31": 120.0, "L4": 80.0,
                    "L41": 20.0, "L5": 180.0, "L51": 15.0, "L52": 30.0,
                    "L6": 290.0, "L61": 200.0, "L8": 260.0,
                    "L3_C0": 100.0, "L7_C0": 270.0, "ZC_C0": 10.0,
                    "L9": 30.0, "L10": 30.0, "L11": 20.0,
                    "L12": 15.0, "L13": 15.0, "LRod": 50.0,
                    "L14": 20.0, "L15": 20.0, "L17": 20.0,
                    "L_down": 15.0, "theta18_deg": 120.0,
                },
                {
                    "L2": 200.0, "L31": 100.0, "L4": 70.0,
                    "L41": 20.0, "L5": 180.0, "L51": 15.0, "L52": 25.0,
                    "L6": 285.0, "L61": 190.0, "L8": 250.0,
                    "L3_C0": 100.0, "L7_C0": 260.0, "ZC_C0": 10.0,
                    "L9": 35.0, "L10": 30.0, "L11": 20.0,
                    "L12": 15.0, "L13": 15.0, "LRod": 45.0,
                    "L14": 20.0, "L15": 20.0, "L17": 20.0,
                    "L_down": 15.0, "theta18_deg": 100.0,
                },
            )
        name_to_index = {name: index for index, name in enumerate(space.names)}
        for definition in seed_definitions:
            x_seed = space.x0.copy()
            for name, value in definition.items():
                index = name_to_index[name]
                x_seed[index] = np.clip(value, space.lb[index], space.ub[index])
            for name in space.names:
                if (
                    name.startswith(
                        ("L3_C", "L5_C", "L6_C", "L7_C", "L8_C", "ZC_C")
                    )
                    and not name.endswith("_C0")
                ):
                    x_seed[name_to_index[name]] = 0.0
            candidate = evaluate(_to_normalized(x_seed, space), "feasibility_geometry_seed")
            if candidate.valid:
                return candidate
            loss, stage = _staged_feasibility_loss(candidate.x, data, space)
            if loss < directed_seed_loss:
                directed_seed_loss = loss
                directed_seed_stage = stage
                directed_seed_y = candidate.y.copy()

    direct_b_fourier = "B_Y_C0" in space.names
    mot_center_y_index = space.names.index(
        "B_Y_C0" if direct_b_fourier else "B_CenterY"
    )
    mot_radius_c0_index = (
        None if direct_b_fourier else space.names.index("B_R_C0")
    )
    # 周期杆用 C0 表示，静态杆用其长度标量表示；两种模式都从原始参数附近
    # 搜索可行点，不读取任何历史最优解。
    l6_index = (
        space.names.index("L6") if "L6" in space.names
        else space.names.index("L6_C0")
    )
    l7_index = (
        space.names.index("L7") if "L7" in space.names
        else space.names.index("L7_C0")
    )
    l8_index = (
        space.names.index("L8") if "L8" in space.names
        else space.names.index("L8_C0")
    )
    center_y_trials = (
        0.0, -10.0, 10.0, -30.0, 30.0, -50.0, 50.0,
        float(space.x0[mot_center_y_index]),
    )
    radius_trials = (
        50.0, 40.0, 60.0, 30.0, 80.0, 100.0, 150.0,
        50.0 if direct_b_fourier else float(space.x0[mot_radius_c0_index]),
    )
    harmonic_scales = (0.0,) if space.periodic_length_mode == "zc_l7_periodic_l3_fixed" else (0.0, 0.25, 0.50, 1.0)
    if space.periodic_length_mode == "zc_l7_periodic_l3_fixed":
        center_y_trials = (0.0, -30.0, 30.0)
        radius_trials = (50.0, 80.0)
    high_order_zc_mode = space.periodic_length_mode in {
        "l5_l8_zc_split_periodic3",
        "l5_l6_l8_zc_split_periodic3",
        "l31_l8_periodic3_l7_fixed",
        "l31_l6_periodic3_l7_l8_fixed",
        "l2_l31_l6_periodic3_l7_l8_fixed",
        "l3_l5_l8_zc2_periodic_l32_fixed",
        "zc_l3_l7_periodic",
        "zc_l3_l7_periodic3",
        "zc_l3_l7_periodic4",
        "zc_l7_l8_periodic4_l3_fixed",
    }
    if high_order_zc_mode:
        # 旧的全笛卡尔枚举在首个可行点前消耗约 1.7 万次模型评估。
        # 高阶模式直接从上面的约束种子进入 CMA 可行化，只保留小型 B 圆扫描。
        center_y_trials = (0.0, -30.0, 30.0)
        radius_trials = (40.0, 60.0, 80.0)
        harmonic_scales = (0.0,)
        l6_shifts = (0.0,)
        l7_shifts = (0.0,)
        l8_shifts = (0.0,)
    else:
        l6_shifts = (0.0, 15.0, 30.0, -15.0, -30.0)
        l7_shifts = (0.0, 15.0, -15.0, 30.0, -30.0)
        l8_shifts = (0.0, 15.0, -15.0)
    for harmonic_scale in harmonic_scales:
        for radius_c0 in radius_trials:
            for center_y in center_y_trials:
                for l6_shift in l6_shifts:
                    for l7_shift in l7_shifts:
                        for l8_shift in l8_shifts:
                            x = _to_physical(directed_seed_y, space)
                            x[mot_center_y_index] = center_y
                            if direct_b_fourier:
                                b_index = {name: space.names.index(name) for name in MOT_POLAR_NAMES}
                                x[b_index["B_Y_C1c"]] = -radius_c0
                                x[b_index["B_Y_C1s"]] = 0.0
                                x[b_index["B_Z_C1c"]] = 0.0
                                x[b_index["B_Z_C1s"]] = -radius_c0
                                for name in ("B_Y_C2c", "B_Y_C2s", "B_Z_C2c", "B_Z_C2s"):
                                    x[b_index[name]] = 0.0
                            else:
                                x[mot_radius_c0_index] = radius_c0
                                x[mot_radius_c0_index + 1:mot_radius_c0_index + 5] *= harmonic_scale
                            x[l6_index] += l6_shift
                            x[l7_index] += l7_shift
                            x[l8_index] += l8_shift
                            x = np.clip(x, space.lb, space.ub)
                            candidate = evaluate(_to_normalized(x, space), "feasibility")
                            if candidate.valid:
                                return candidate

    # 原始点与规则网格均不可行时，使用分阶段几何违反量推进无锚点 CMA 搜索。
    # 可行性只由机构几何决定，因此目标位姿在本阶段严格保持原始值，避免在找到
    # 第一组可行机构之前浪费维度或随机改变目标曲线姿态。
    rng = np.random.default_rng(config.seed + 104729)
    dimension = len(space.names)
    # 与主搜索使用同一有效活动集：目标位姿由内层求解，B_Z_C3c 是相位
    # 约束派生量，H_finger/Lf1 对当前 Tip/Wrist 输出无作用。
    feasibility_indices = _outer_active_indices(space)
    feasibility_dimension = len(feasibility_indices)
    population = max(
        18, 4 + int(3.0 * np.log(max(2, feasibility_dimension)))
    )
    elite_count = max(4, population // 3)
    rank_weights = np.log(elite_count + 0.5) - np.log(
        np.arange(1, elite_count + 1)
    )
    rank_weights /= np.sum(rank_weights)
    best_loss, best_stage = directed_seed_loss, directed_seed_stage
    best_y = directed_seed_y.copy()

    for restart, initial_sigma in enumerate((0.08, 0.16, 0.30, 0.50, 0.70, 0.90)):
        mean = best_y.copy()
        if restart > 0:
            mean[feasibility_indices] = np.clip(
                best_y[feasibility_indices]
                + rng.normal(
                    0.0, 0.45 * initial_sigma, feasibility_dimension
                ),
                0.0,
                1.0,
            )
        mean[np.setdiff1d(np.arange(dimension), feasibility_indices)] = y0[
            np.setdiff1d(np.arange(dimension), feasibility_indices)
        ]
        diagonal = np.ones(dimension, dtype=float)
        sigma = float(initial_sigma)
        stale_generations = 0

        for generation in range(90):
            ranked: list[tuple[float, np.ndarray, str]] = []
            for member in range(population):
                if member == 0:
                    proposal = mean.copy()
                elif member == population - 1 and restart >= 2:
                    proposal = mean.copy()
                    proposal[feasibility_indices] = rng.random(
                        feasibility_dimension
                    )
                else:
                    proposal = mean.copy()
                    proposal[feasibility_indices] = np.clip(
                        mean[feasibility_indices]
                        + sigma
                        * diagonal[feasibility_indices]
                        * rng.normal(size=feasibility_dimension),
                        0.0,
                        1.0,
                    )
                candidate = evaluate(proposal, "feasibility_cma")
                if candidate.valid:
                    return candidate
                loss, stage = _staged_feasibility_loss(candidate.x, data, space)
                if loss <= 0.0:
                    # candidate.x 是模型投影和相位规范化后的真实物理点。部分 B
                    # 系数在首次规范化后会越过分量边界，第二次稳定投影才形成
                    # 固定点；必须立即复评该真实点，不能把修复前 proposal 丢回种群。
                    stabilized = evaluate(candidate.y, "feasibility_stabilized")
                    if stabilized.valid:
                        return stabilized
                    candidate = stabilized
                    loss, stage = _staged_feasibility_loss(
                        candidate.x, data, space
                    )
                ranked.append((float(loss), candidate.y.copy(), stage))

            ranked.sort(key=lambda row: row[0])
            generation_loss, generation_y, generation_stage = ranked[0]
            if generation_loss + 1e-12 < best_loss:
                best_loss = generation_loss
                best_stage = generation_stage
                best_y = generation_y.copy()
                stale_generations = 0
            else:
                stale_generations += 1

            elites = np.vstack([row[1] for row in ranked[:elite_count]])
            previous_mean = mean.copy()
            mean = np.sum(rank_weights[:, None] * elites, axis=0)
            centered = elites - mean
            elite_scale = np.sqrt(
                np.sum(rank_weights[:, None] * centered * centered, axis=0)
            )
            diagonal = np.clip(
                0.80 * diagonal
                + 0.20 * elite_scale / max(sigma, 1e-6),
                0.08,
                3.0,
            )
            movement = float(
                np.linalg.norm(
                    mean[feasibility_indices]
                    - previous_mean[feasibility_indices]
                )
                / np.sqrt(feasibility_dimension)
            )
            if generation_loss <= best_loss + 1e-12 and movement > 0.01:
                sigma = min(0.95, sigma * 1.03)
            else:
                sigma = max(0.015, sigma * 0.94)
            if stale_generations >= 18:
                mean[feasibility_indices] = np.clip(
                    0.65 * best_y[feasibility_indices]
                    + 0.35 * rng.random(feasibility_dimension),
                    0.0,
                    1.0,
                )
                sigma = max(sigma, 0.25)
                stale_generations = 0

    raise RuntimeError(
        "No feasible point was found from the original fourbar start; "
        f"best staged feasibility loss={best_loss:.6g} at {best_stage}."
    )


def _build_initial_feasible_pool(
    initial: _Candidate,
    data: ProblemData,
    space: DesignSpace,
    config: _OptimizeConfig,
    evaluation_counter: list[int],
    recorder: _OptimizationRecorder | None,
) -> list[_Candidate]:
    """围绕原始可行起点用 Sobol 扰动构造分散可行池，不读取历史最优解。"""

    target_count = max(1, int(config.initial_pool_size))
    if target_count <= 1:
        return [initial]
    active = _outer_active_indices(space)
    sampler = qmc.Sobol(d=len(active), scramble=True, seed=int(config.seed) + 991)
    pool = [initial]
    proposals = 0
    # 采用可行延拓而不是始终围绕第一个可行点做微扰。每个新可行点都可成为
    # 下一段路径的起点，因此能够沿连通可行域逐步覆盖远距离参数组合。
    scales = (0.08, 0.20, 0.45, 1.00)

    while (
        len(pool) < target_count
        and proposals < int(config.initial_pool_max_proposals)
    ):
        unit = sampler.random(1)[0]
        scale = scales[proposals % len(scales)]
        origin_index = proposals % len(pool)
        origin = pool[origin_index]
        requested = origin.y.copy()
        if proposals % 5 == 0:
            requested[active] = np.clip(
                origin.y[active] + (2.0 * unit - 1.0) * scale,
                0.0,
                1.0,
            )
        else:
            requested[active] = np.clip(
                origin.y[active] + scale * (unit - origin.y[active]),
                0.0,
                1.0,
            )
        attempts: list[tuple[float, _Candidate]] = []
        candidate = _evaluate_candidate(
            requested,
            data,
            space,
            0,
            "initial_sobol_pool",
            config.objective_mode,
            config.target_rmse_mm,
        )
        attempts.append((1.0, candidate))
        if not candidate.valid:
            for fraction in (0.75, 0.50, 0.25, 0.125, 0.0625):
                backtracked = origin.y + fraction * (requested - origin.y)
                trial = _evaluate_candidate(
                    backtracked,
                    data,
                    space,
                    0,
                    "initial_pool_backtrack",
                    config.objective_mode,
                    config.target_rmse_mm,
                )
                attempts.append((fraction, trial))
                if trial.valid:
                    candidate = trial
                    break
        start = evaluation_counter[0]
        evaluation_counter[0] += len(attempts)
        if recorder is not None:
            for offset, (fraction, attempt) in enumerate(attempts, start=1):
                recorder.record_candidate(
                    attempt,
                    start + offset,
                    round_index=-1,
                    metadata={
                        "initial_pool": True,
                        "proposal_index": proposals,
                        "proposal_scale": scale,
                        "proposal_fraction": fraction,
                        "origin_pool_index": origin_index,
                    },
                )
        if candidate.valid:
            distance = min(
                float(np.linalg.norm(candidate.y[active] - known.y[active]))
                for known in pool
            )
            if distance > float(config.initial_pool_min_distance):
                pool.append(candidate)
        proposals += 1
    if recorder is not None:
        recorder.record_region_event(
            "initial_feasible_pool_complete",
            {
                "requested_size": target_count,
                "actual_size": len(pool),
                "proposal_count": proposals,
                "active_dimensions": len(active),
            },
        )
    return pool


def _update_region(
    region: _Region,
    candidates: list[_Candidate],
    *,
    count_visit: bool = True,
    count_evaluations: bool = True,
) -> None:
    """把本轮样本写入区域，并更新该区域已知最优点。"""

    region.samples.extend(candidates)
    if count_visit:
        region.visits += 1
    if count_evaluations:
        region.evaluations_since_creation += len(candidates)
    valid = [candidate for candidate in candidates if candidate.valid]
    if valid:
        best = min(valid, key=lambda candidate: candidate.score)
        if best.score < region.best_score:
            region.best_score = best.score
            region.best_y = best.y.copy()
    if np.isfinite(region.best_score):
        region.prior_score = min(region.prior_score, region.best_score)


def _reseed_split_child(
    child: _Region,
    dimension: int,
    sigma0: float,
) -> None:
    """让新叶区从自己的可行中心和未衰减步长开始独立搜索。"""

    if child.best_y is not None:
        child.cma_mean = child.best_y.copy()
    child.cma_sigma = float(sigma0)
    child.cma_path_c = np.zeros(int(dimension), dtype=float)
    child.cma_path_sigma = np.zeros(int(dimension), dtype=float)
    child.cma_generation_count = 0
    child.cma_stale_generations = 0


def _consolidate_deep_region(
    best: _Candidate,
    dimension: int,
    region_id: int,
) -> _Region:
    """围绕当前最优点重建一个全设计域深层搜索区域。

    旧叶边界和协方差只描述进入深盆地前的覆盖历史。达到深层阈值后重新使用
    完整归一化边界，并让 CMA-ES 从当前最优点学习局部协方差，可避免窄叶区
    截断后续需要多个物理参数共同变化的方向。
    """

    best.region_id = int(region_id)
    region = _Region(
        region_id=int(region_id),
        lo=np.zeros(int(dimension), dtype=float),
        hi=np.ones(int(dimension), dtype=float),
        depth=0,
    )
    _update_region(
        region,
        [best],
        count_visit=False,
        count_evaluations=False,
    )
    region.visits = 1
    region.cma_mean = best.y.copy()
    region.cma_diagonal = None
    region.cma_covariance = None
    region.cma_sigma = None
    region.cma_path_c = None
    region.cma_path_sigma = None
    region.cma_generation_count = 0
    region.cma_stale_generations = 0
    return region


def _refresh_region_scores(
    regions: list[_Region],
    objective_mode: str,
    target_rmse_mm: float,
) -> _Candidate:
    """阶段目标改变时仅按已存指标重算分数，不重复调用机构模型。"""

    valid_candidates: list[_Candidate] = []
    seen: set[int] = set()
    for region in regions:
        region.best_score = math.inf
        region.best_y = None
        for candidate in region.samples:
            identity = id(candidate)
            if identity not in seen and candidate.valid:
                candidate.score = _optimization_objective(
                    candidate.metrics, objective_mode, target_rmse_mm
                )
                seen.add(identity)
                valid_candidates.append(candidate)
            if candidate.valid and candidate.score < region.best_score:
                region.best_score = float(candidate.score)
                region.best_y = candidate.y.copy()
        if np.isfinite(region.best_score):
            region.prior_score = min(region.prior_score, region.best_score)
    if not valid_candidates:
        raise RuntimeError("No valid candidate remained while refreshing stage scores.")
    return min(valid_candidates, key=lambda candidate: candidate.score)


def _evaluate_cma_sample_worker(
    payload: tuple[
        int,
        np.ndarray,
        ProblemData,
        DesignSpace,
        int,
        str,
        float,
        np.ndarray | None,
    ],
) -> tuple[_Candidate, list[tuple[int, int, float, _Candidate]]]:
    """评估一个 CMA-ES 样本及其可行性回退点，供线程或独立进程调用。"""

    (
        population_index,
        sample,
        data,
        space,
        region_id,
        objective_mode,
        target_rmse_mm,
        region_best_y,
    ) = payload
    candidate = _evaluate_candidate(
        sample,
        data,
        space,
        region_id,
        "cmaes",
        objective_mode,
        target_rmse_mm,
    )
    attempts = [(population_index, 0, 1.0, candidate)]
    if not candidate.valid and region_best_y is not None:
        for attempt_index, fraction in enumerate(
            (0.75, 0.50, 0.25, 0.125, 0.0625), start=1
        ):
            repaired_sample = region_best_y + fraction * (sample - region_best_y)
            repaired = _evaluate_candidate(
                repaired_sample,
                data,
                space,
                region_id,
                "cmaes_backtrack",
                objective_mode,
                target_rmse_mm,
            )
            attempts.append(
                (population_index, attempt_index, fraction, repaired)
            )
            if repaired.valid:
                candidate = repaired
                break
    return candidate, attempts


def _run_cma_es(
    region: _Region,
    data: ProblemData,
    space: DesignSpace,
    config: _OptimizeConfig,
    rng: np.random.Generator,
    trace: list[dict[str, Any]],
    evaluation_counter: list[int],
    recorder: _OptimizationRecorder | None,
    round_index: int,
    deadline: float,
) -> list[_Candidate]:
    """在一个分区内执行并行、分块协方差和可行性回退 CMA-ES。"""

    dimension = len(space.names)
    active = _search_active_indices(space, config.search_scope)
    partition_active = _outer_active_indices(space)
    inactive = np.setdiff1d(np.arange(dimension), active, assume_unique=True)
    covariance_blocks = _cma_blocks(
        space, config.cma_full_covariance, active
    )
    # 远距注入仍然一次只展开一个物理参数组。即使高斯主体使用完整协方差，
    # 同时随机化全部 62 个外层变量也几乎必然离开狭窄几何可行域。
    injection_blocks = _cma_blocks(space, False, active)
    population = max(6, int(config.popsize))
    elite_fraction = float(np.clip(config.cma_elite_fraction, 0.05, 1.0))
    mu = max(2, min(population, int(math.ceil(population * elite_fraction))))
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    # 优先延续该区域上一轮学到的 CMA 状态；首次访问才从历史最优或几何中心开始。
    if region.cma_mean is not None and region.cma_mean.shape == (dimension,):
        mean = region.cma_mean.copy()
    else:
        mean = region.best_y.copy() if region.best_y is not None else region.center()
    mean = np.clip(mean, region.lo, region.hi)
    if region.cma_diagonal is not None and region.cma_diagonal.shape == (dimension,):
        diagonal = np.maximum(
            region.cma_diagonal.copy(), float(config.cma_variance_floor)
        )
    else:
        diagonal = np.ones(dimension)
    if (
        region.cma_covariance is not None
        and region.cma_covariance.shape == (dimension, dimension)
    ):
        covariance = region.cma_covariance.copy()
    else:
        covariance = np.zeros((dimension, dimension), dtype=float)
        for block in covariance_blocks:
            covariance[np.ix_(block, block)] = np.eye(len(block))
        # 用初始可行池的优良样本初始化块内方向，而不是从单个点和单位方差开始。
        seed_candidates = sorted(
            (
                candidate
                for candidate in region.samples
                if candidate.valid
                and _point_in_region(candidate.y, region, partition_active)
            ),
            key=lambda candidate: candidate.score,
        )[: max(3, min(population, len(region.samples)))]
        if len(seed_candidates) >= 3:
            seed_y = np.stack([candidate.y for candidate in seed_candidates])
            for block in covariance_blocks:
                block_cov = _seed_cma_covariance(
                    seed_y,
                    block,
                    config.cma_covariance_learning_rate,
                    config.cma_variance_floor,
                )
                covariance[np.ix_(block, block)] = block_cov
    sigma = float(config.sigma0)
    if region.cma_sigma is not None and np.isfinite(region.cma_sigma):
        # 每个区域批次使用当前阶段 sigma0 作为信赖半径上限；批次内部仍允许
        # CMA 自适应扩张，下一批重新受限可防止高可行率规则持续放大到过宽区域。
        sigma = min(sigma, float(region.cma_sigma))
    path_c = (
        region.cma_path_c.copy()
        if region.cma_path_c is not None
        and region.cma_path_c.shape == (dimension,)
        else np.zeros(dimension, dtype=float)
    )
    path_sigma = (
        region.cma_path_sigma.copy()
        if region.cma_path_sigma is not None
        and region.cma_path_sigma.shape == (dimension,)
        else np.zeros(dimension, dtype=float)
    )
    generation_count = int(region.cma_generation_count)
    width = region.width()
    output: list[_Candidate] = []
    local_best = math.inf
    executor: ProcessPoolExecutor | ThreadPoolExecutor | None = None
    if int(config.workers) > 1:
        executor_class = (
            ProcessPoolExecutor
            if config.parallel_backend == "process"
            else ThreadPoolExecutor
        )
        # 同一区域的多代 CMA-ES 复用工作进程，避免 Windows 每代重复启动解释器。
        executor = executor_class(max_workers=int(config.workers))

    for generation in range(int(config.generations_per_region)):
        if (
            time.monotonic() >= deadline
            or evaluation_counter[0] >= int(config.max_evaluations)
        ):
            break
        # 每个物理参数块使用完整协方差，块间保持独立。
        mean_before = mean.copy()
        diagonal_before = diagonal.copy()
        sigma_before = float(sigma)
        z = np.zeros((population, dimension), dtype=float)
        for block in covariance_blocks:
            block_covariance = covariance[np.ix_(block, block)]
            eigenvalues, eigenvectors = np.linalg.eigh(
                0.5 * (block_covariance + block_covariance.T)
            )
            eigenvalues = np.maximum(
                eigenvalues, float(config.cma_variance_floor)
            )
            transform = eigenvectors @ np.diag(np.sqrt(eigenvalues))
            z[:, block] = rng.normal(size=(population, len(block))) @ transform.T
        samples = np.clip(mean + sigma * width * z, region.lo, region.hi)
        samples[:, inactive] = mean[inactive]
        sample_source = np.full(population, "gaussian", dtype=object)
        samples[0] = mean
        sample_source[0] = "mean"

        # 远距注入仍属于区域 CMA-ES 的候选生成：每次只展开一个物理参数块，
        # 避免在全部外层维度上同时均匀采样造成近乎必然的几何失效。
        local_block_count = min(
            population - 1,
            max(0, int(round(population * config.cma_local_block_fraction))),
        )
        global_count = min(
            population - 1 - local_block_count,
            max(0, int(round(population * config.cma_global_injection_fraction))),
        )
        boundary_count = min(
            population - 1 - local_block_count - global_count,
            max(0, int(round(population * config.cma_boundary_injection_fraction))),
        )
        local_start = population - local_block_count - global_count - boundary_count
        for offset in range(local_block_count):
            sample_index = local_start + offset
            block = injection_blocks[
                (generation + offset) % len(injection_blocks)
            ]
            samples[sample_index] = mean
            samples[sample_index, block] = np.clip(
                mean[block] + sigma * width[block] * z[sample_index, block],
                region.lo[block],
                region.hi[block],
            )
            sample_source[sample_index] = "block_local"
        global_start = local_start + local_block_count
        for offset in range(global_count):
            sample_index = global_start + offset
            block = injection_blocks[
                (generation + offset) % len(injection_blocks)
            ]
            samples[sample_index] = mean
            samples[sample_index, block] = (
                region.lo[block] + rng.random(len(block)) * width[block]
            )
            sample_source[sample_index] = "block_global"
        for offset in range(boundary_count):
            sample_index = population - boundary_count + offset
            block = injection_blocks[
                (generation + global_count + offset) % len(injection_blocks)
            ]
            samples[sample_index] = mean
            selected_count = max(1, min(4, int(math.ceil(0.25 * len(block)))))
            selected_block = rng.choice(block, size=selected_count, replace=False)
            choose_upper = rng.random(selected_count) >= 0.5
            samples[sample_index, selected_block] = np.where(
                choose_upper,
                region.hi[selected_block],
                region.lo[selected_block],
            )
            sample_source[sample_index] = "block_boundary"

        evaluation_payloads = [
            (
                population_index,
                sample,
                data,
                space,
                region.region_id,
                config.objective_mode,
                config.target_rmse_mm,
                region.best_y,
            )
            for population_index, sample in enumerate(samples)
        ]
        if executor is not None:
            evaluated = list(
                executor.map(_evaluate_cma_sample_worker, evaluation_payloads)
            )
        else:
            evaluated = [
                _evaluate_cma_sample_worker(payload)
                for payload in evaluation_payloads
            ]
        candidates = [item[0] for item in evaluated]
        recorded_attempts = [
            attempt for _candidate, attempts in evaluated for attempt in attempts
        ]
        # 同时保存 CMA-ES 的原始提议范围和实际模型测试范围。后者包含边界投影
        # 与可行性回退后的真实坐标，供全变量迭代图绘制每一代各自的上下包络。
        proposal_min = np.min(samples, axis=0)
        proposal_max = np.max(samples, axis=0)
        tested_coordinates = np.stack(
            [attempt.y for _, _, _, attempt in recorded_attempts]
        )
        tested_min = np.min(tested_coordinates, axis=0)
        tested_max = np.max(tested_coordinates, axis=0)
        evaluation_start = evaluation_counter[0]
        evaluation_counter[0] += len(recorded_attempts)
        if recorder is not None:
            for attempt_offset, (
                population_index,
                attempt_index,
                fraction,
                attempt,
            ) in enumerate(recorded_attempts):
                recorder.record_candidate(
                    attempt,
                    evaluation_start + attempt_offset + 1,
                    round_index=round_index,
                    generation=generation,
                    population_index=population_index,
                    metadata={
                        "population": population,
                        "feasibility_attempt": attempt_index,
                        "proposal_fraction": fraction,
                        "sample_source": str(sample_source[population_index]),
                        "used_for_cma_selection": attempt is candidates[population_index],
                    },
                )
        output.extend(attempt for _, _, _, attempt in recorded_attempts)
        valid = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.valid
                and _point_in_region(candidate.y, region, partition_active)
            ),
            key=lambda candidate: candidate.score,
        )
        generation_best = valid[0].score if valid else 1e6
        trace.append({
            "evaluation": evaluation_counter[0],
            "stage": "cmaes",
            "region_id": region.region_id,
            "generation": generation,
            "best_score": float(generation_best),
            "sigma": sigma_before,
            "population": population,
            "valid_count": len(valid),
            "valid_fraction": float(len(valid) / population),
            "block_local_count": int(local_block_count),
            "block_global_count": int(global_count),
            "block_boundary_count": int(boundary_count),
        })
        if not valid:
            # 已知均值可行但整代失效，说明搜索半径越过狭窄可行域，应立即收缩。
            sigma = max(
                float(config.cma_sigma_min),
                float(config.cma_sigma_no_valid_factor) * sigma,
            )
            if recorder is not None:
                recorder.record_cma_generation({
                    "round": round_index,
                    "region_id": region.region_id,
                    "generation": generation,
                    "evaluation_start": evaluation_start + 1,
                    "evaluation_end": evaluation_counter[0],
                    "population": population,
                    "valid_count": 0,
                    "generation_best": 1e6,
                    "sigma_before": sigma_before,
                    "sigma_after": float(sigma),
                    "mean_before": mean_before,
                    "mean_after": mean.copy(),
                    "diagonal_before": diagonal_before,
                    "diagonal_after": diagonal.copy(),
                    "selected_evaluations": [],
                    "region_lo": region.lo.copy(),
                    "region_hi": region.hi.copy(),
                    "proposal_min": proposal_min,
                    "proposal_max": proposal_max,
                    "tested_min": tested_min,
                    "tested_max": tested_max,
                })
            continue
        # 只有修复后真实坐标仍在本叶区的候选才可更新该叶区分布。
        selected = valid[: min(mu, len(valid))]
        selected_y = np.stack([candidate.y for candidate in selected])
        selected_weights = weights[: len(selected)].copy()
        selected_weights /= np.sum(selected_weights)
        old_best = local_best
        local_best = min(local_best, selected[0].score)
        selected_mean = np.sum(selected_y * selected_weights[:, None], axis=0)
        mean[active] = selected_mean[active]
        mean[inactive] = selected[0].y[inactive]
        mean = np.clip(mean, region.lo, region.hi)
        use_standard_cma = bool(
            config.cma_full_covariance and config.cma_use_evolution_paths
        )
        if use_standard_cma:
            # 标准 CMA-ES：在当前叶区域的归一化坐标内累积协方差路径
            # 和步长路径，并组合 rank-one 与 rank-mu 更新。
            active_width = np.maximum(width[active], 1e-12)
            normalized_steps = (
                selected_y[:, active] - mean_before[active]
            ) / max(sigma_before, 1e-12) / active_width
            active_covariance = covariance[np.ix_(active, active)]
            eigenvalues, eigenvectors = np.linalg.eigh(
                0.5 * (active_covariance + active_covariance.T)
            )
            eigenvalues = np.maximum(
                eigenvalues, float(config.cma_variance_floor)
            )
            inverse_sqrt = (
                eigenvectors
                @ np.diag(1.0 / np.sqrt(eigenvalues))
                @ eigenvectors.T
            )
            active_dimension = max(1, len(active))
            # 远距注入和边界样本并非由当前高斯分布直接产生。按标准
            # injected-solution 处理限制其 Mahalanobis 步长，防止单个
            # 注入点把全局 sigma 一代推到上限。
            whitened_steps = normalized_steps @ inverse_sqrt.T
            whitened_norms = np.linalg.norm(whitened_steps, axis=1)
            maximum_injected_norm = (
                math.sqrt(active_dimension)
                + 2.0 * active_dimension / (active_dimension + 2.0)
            )
            step_scales = np.minimum(
                1.0,
                maximum_injected_norm
                / np.maximum(whitened_norms, 1e-12),
            )
            normalized_steps = normalized_steps * step_scales[:, None]
            weighted_step = np.sum(
                selected_weights[:, None] * normalized_steps, axis=0
            )
            mu_effective = 1.0 / max(
                float(np.sum(selected_weights ** 2)), 1e-12
            )
            c_sigma = (mu_effective + 2.0) / (
                active_dimension + mu_effective + 5.0
            )
            d_sigma = (
                1.0
                + 2.0
                * max(
                    0.0,
                    math.sqrt(
                        max(mu_effective - 1.0, 0.0)
                        / (active_dimension + 1.0)
                    )
                    - 1.0,
                )
                + c_sigma
            )
            c_c = (
                4.0 + mu_effective / active_dimension
            ) / (
                active_dimension
                + 4.0
                + 2.0 * mu_effective / active_dimension
            )
            c_one = 2.0 / (
                (active_dimension + 1.3) ** 2 + mu_effective
            )
            c_mu = min(
                1.0 - c_one,
                2.0
                * (
                    mu_effective
                    - 2.0
                    + 1.0 / max(mu_effective, 1e-12)
                )
                / (
                    (active_dimension + 2.0) ** 2 + mu_effective
                ),
            )
            path_sigma[active] = (
                (1.0 - c_sigma) * path_sigma[active]
                + math.sqrt(
                    c_sigma * (2.0 - c_sigma) * mu_effective
                )
                * (inverse_sqrt @ weighted_step)
            )
            generation_count += 1
            expected_norm = math.sqrt(active_dimension) * (
                1.0
                - 1.0 / (4.0 * active_dimension)
                + 1.0 / (21.0 * active_dimension ** 2)
            )
            path_normalizer = math.sqrt(
                max(
                    1.0 - (1.0 - c_sigma) ** (2 * generation_count),
                    1e-12,
                )
            )
            h_sigma = float(
                np.linalg.norm(path_sigma[active]) / path_normalizer
                < (
                    1.4 + 2.0 / (active_dimension + 1.0)
                )
                * expected_norm
            )
            path_c[active] = (
                (1.0 - c_c) * path_c[active]
                + h_sigma
                * math.sqrt(c_c * (2.0 - c_c) * mu_effective)
                * weighted_step
            )
            rank_mu = np.einsum(
                "n,ni,nj->ij",
                selected_weights,
                normalized_steps,
                normalized_steps,
            )
            covariance_decay = (
                1.0
                - c_one
                - c_mu
                + c_one * (1.0 - h_sigma) * c_c * (2.0 - c_c)
            )
            updated = (
                covariance_decay * active_covariance
                + c_one
                * np.outer(path_c[active], path_c[active])
                + c_mu * rank_mu
            )
            updated = 0.5 * (updated + updated.T)
            eigenvalues, eigenvectors = np.linalg.eigh(updated)
            eigenvalues = np.maximum(
                eigenvalues, float(config.cma_variance_floor)
            )
            covariance[np.ix_(active, active)] = (
                eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            )
            sigma = float(
                sigma_before
                * math.exp(
                    (c_sigma / d_sigma)
                    * (
                        np.linalg.norm(path_sigma[active])
                        / max(expected_norm, 1e-12)
                        - 1.0
                    )
                )
            )
        else:
            covariance_scale = width.copy()
            if config.cma_normalize_covariance_by_sigma:
                # 样本生成时已经乘过 sigma；协方差学习必须除回 sigma。
                covariance_scale = covariance_scale * max(sigma_before, 1e-12)
            centered = (selected_y - mean) / np.maximum(covariance_scale, 1e-12)
            covariance_lr = float(
                np.clip(config.cma_covariance_learning_rate, 0.0, 1.0)
            )
            for block in covariance_blocks:
                block_centered = centered[:, block]
                block_sample_covariance = np.einsum(
                    "n,ni,nj->ij",
                    selected_weights,
                    block_centered,
                    block_centered,
                )
                previous = covariance[np.ix_(block, block)]
                updated = (
                    (1.0 - covariance_lr) * previous
                    + covariance_lr * block_sample_covariance
                )
                updated = 0.5 * (updated + updated.T)
                updated += float(config.cma_variance_floor) * np.eye(len(block))
                covariance[np.ix_(block, block)] = updated
        diagonal = np.maximum(
            np.diag(covariance), float(config.cma_variance_floor)
        )
        valid_fraction = len(valid) / population
        improved = local_best < old_best
        if improved:
            region.cma_stale_generations = 0
        else:
            region.cma_stale_generations += 1
        if use_standard_cma:
            if valid_fraction < float(config.cma_low_valid_fraction):
                sigma = min(
                    sigma,
                    float(config.cma_sigma_no_valid_factor) * sigma_before,
                )
            sigma = float(
                np.clip(
                    sigma,
                    float(config.cma_sigma_min),
                    float(config.cma_sigma_max),
                )
            )
        else:
            if valid_fraction < float(config.cma_low_valid_fraction):
                sigma = max(
                    float(config.cma_sigma_min),
                    float(config.cma_sigma_no_valid_factor) * sigma,
                )
            elif valid_fraction > float(config.cma_high_valid_fraction) and improved:
                sigma = min(
                    float(config.cma_sigma_max),
                    float(config.cma_sigma_expand_factor) * sigma,
                )
            else:
                sigma = max(
                    float(config.cma_sigma_min),
                    float(config.cma_sigma_default_factor) * sigma,
                )
        restarted = False
        if (
            config.cma_restart_mode == "per_region_generation"
            and region.cma_stale_generations
            >= int(config.cma_stagnation_generations)
        ):
            restart_candidates = sorted(
                (
                    candidate
                    for candidate in region.samples
                    if candidate.valid
                    and _point_in_region(candidate.y, region, partition_active)
                ),
                key=lambda candidate: candidate.score,
            )
            if restart_candidates:
                shortlist = restart_candidates[: max(2, len(restart_candidates) // 2)]
                anchor = region.best_y if region.best_y is not None else mean
                restart = max(
                    shortlist,
                    key=lambda candidate: float(
                        np.linalg.norm(candidate.y[active] - anchor[active])
                    ),
                )
                mean = np.clip(restart.y.copy(), region.lo, region.hi)
                sigma = _cma_restart_sigma(sigma, config)
                covariance.fill(0.0)
                for block in covariance_blocks:
                    covariance[np.ix_(block, block)] = np.eye(len(block))
                diagonal = np.maximum(
                    np.diag(covariance), float(config.cma_variance_floor)
                )
                path_c.fill(0.0)
                path_sigma.fill(0.0)
                generation_count = 0
                region.cma_stale_generations = 0
                restarted = True
        if recorder is not None:
            recorder.record_cma_generation({
                "round": round_index,
                "region_id": region.region_id,
                "generation": generation,
                "evaluation_start": evaluation_start + 1,
                "evaluation_end": evaluation_counter[0],
                "population": population,
                "valid_count": len(valid),
                "generation_best": float(generation_best),
                "sigma_before": sigma_before,
                "sigma_after": float(sigma),
                "mean_before": mean_before,
                "mean_after": mean.copy(),
                "diagonal_before": diagonal_before,
                "diagonal_after": diagonal.copy(),
                "selected_evaluations": [candidate.evaluation_id for candidate in selected],
                "region_lo": region.lo.copy(),
                "region_hi": region.hi.copy(),
                "proposal_min": proposal_min,
                "proposal_max": proposal_max,
                "tested_min": tested_min,
                "tested_max": tested_max,
                "block_covariance": not bool(config.cma_full_covariance),
                "full_covariance": bool(config.cma_full_covariance),
                "evolution_paths": bool(use_standard_cma),
                "path_c_norm": float(np.linalg.norm(path_c[active])),
                "path_sigma_norm": float(np.linalg.norm(path_sigma[active])),
                "restart": restarted,
            })
    region.cma_mean = np.clip(mean, region.lo, region.hi)
    region.cma_diagonal = np.maximum(diagonal, float(config.cma_variance_floor))
    region.cma_covariance = covariance.copy()
    region.cma_sigma = float(sigma)
    region.cma_path_c = path_c.copy()
    region.cma_path_sigma = path_sigma.copy()
    region.cma_generation_count = int(generation_count)
    if executor is not None:
        executor.shutdown(wait=True)
    return output


def _coupled_repair_tangent_proposals(
    start: _Candidate,
    region: _Region,
    space: DesignSpace,
    trust_radius: float,
) -> list[tuple[float, np.ndarray]]:
    """沿 L3/L31 的共同物理位移生成装配约束切向候选。

    当 ``L3-L31+2 = L5-L51-L52+0.01`` 活跃时，分别扰动 L3 或 L31
    会被代数修复器抵消；两者等量移动保持该等式，却能改变主三角形角度。
    这里显式给出这个可行切向，避免有限差分 SQP 把它误判为零灵敏度。
    """

    name_to_index = {name: index for index, name in enumerate(space.names)}
    l31_name = "L31_C0" if "L31_C0" in name_to_index else "L31"
    if "L3" not in name_to_index or l31_name not in name_to_index:
        return []
    indices = np.asarray(
        [name_to_index["L3"], name_to_index[l31_name]], dtype=np.int32
    )
    spans = space.ub[indices] - space.lb[indices]
    if np.any(spans <= 1e-12):
        return []

    base_y = np.asarray(start.y, dtype=float)
    lower_delta = max(
        float((region.lo[index] - base_y[index]) * span)
        for index, span in zip(indices, spans)
    )
    upper_delta = min(
        float((region.hi[index] - base_y[index]) * span)
        for index, span in zip(indices, spans)
    )
    trust_mm = max(0.25, float(trust_radius) * float(np.min(spans)))
    magnitudes = sorted(
        {
            min(trust_mm, 0.5),
            min(trust_mm, 1.0),
            0.5 * trust_mm,
            trust_mm,
        }
    )
    proposals: list[tuple[float, np.ndarray]] = []
    for signed_magnitude in (
        *(-value for value in reversed(magnitudes)),
        *magnitudes,
    ):
        delta = float(np.clip(signed_magnitude, lower_delta, upper_delta))
        if abs(delta) <= 1e-10 or any(
            abs(delta - known_delta) <= 1e-10 for known_delta, _ in proposals
        ):
            continue
        trial_y = base_y.copy()
        trial_y[indices] += delta / spans
        proposals.append((delta, np.clip(trial_y, region.lo, region.hi)))
    return proposals


def _run_repair_tangent_line_search(
    start: _Candidate,
    region: _Region,
    data: ProblemData,
    space: DesignSpace,
    config: _OptimizeConfig,
    trace: list[dict[str, Any]],
    evaluation_counter: list[int],
    recorder: _OptimizationRecorder | None,
    round_index: int,
    deadline: float,
) -> _Candidate:
    """用少量真实模型评价搜索修复投影形成的局部可行切向。"""

    best = start
    partition_active = _outer_active_indices(space)
    for delta_mm, trial_y in _coupled_repair_tangent_proposals(
        start, region, space, config.sqp_trust_radius
    ):
        if (
            time.monotonic() >= deadline
            or evaluation_counter[0] >= int(config.max_evaluations)
        ):
            break
        candidate = _evaluate_candidate(
            trial_y,
            data,
            space,
            region.region_id,
            "sqp_repair_tangent",
            config.objective_mode,
            config.target_rmse_mm,
        )
        evaluation_counter[0] += 1
        if recorder is not None:
            recorder.record_candidate(
                candidate,
                evaluation_counter[0],
                round_index=round_index,
                metadata={
                    "tangent": "equal_physical_delta_L3_L31",
                    "delta_mm": delta_mm,
                },
            )
        if (
            candidate.valid
            and _point_in_region(candidate.y, region, partition_active)
            and candidate.score < best.score
        ):
            best = candidate
        trace.append(
            {
                "evaluation": evaluation_counter[0],
                "stage": "sqp_repair_tangent",
                "region_id": region.region_id,
                "delta_mm": delta_mm,
                "best_score": float(best.score),
                "trial_score": float(candidate.score),
            }
        )
    return best


def _run_slsqp(
    start: _Candidate,
    region: _Region,
    data: ProblemData,
    space: DesignSpace,
    config: _OptimizeConfig,
    trace: list[dict[str, Any]],
    evaluation_counter: list[int],
    recorder: _OptimizationRecorder | None,
    round_index: int,
    deadline: float,
) -> _Candidate | None:
    """在归一化空间内精修局部灵敏度最高的少量活动变量。"""

    if not start.valid or config.sqp_maxiter <= 0:
        return None
    entry_start = start
    refinement_evaluation_start = int(evaluation_counter[0])
    maximum_curve_rmse = max(
        float(start.metrics.get("tip", math.inf)),
        float(start.metrics.get("wrist", math.inf)),
    )
    if maximum_curve_rmse > float(config.sqp_trigger_rmse_mm):
        skip_event = {
            "round": round_index,
            "region_id": region.region_id,
            "stage": "slsqp_skipped_high_rmse",
            "tip_rmse_mm": float(start.metrics.get("tip", math.inf)),
            "wrist_rmse_mm": float(start.metrics.get("wrist", math.inf)),
            "threshold_mm": float(config.sqp_trigger_rmse_mm),
        }
        trace.append({
            "evaluation": evaluation_counter[0],
            "best_score": float(start.score),
            **skip_event,
        })
        if recorder is not None:
            recorder.record_region_event("slsqp_skipped_high_rmse", skip_event)
        return None
    start = _run_repair_tangent_line_search(
        start,
        region,
        data,
        space,
        config,
        trace,
        evaluation_counter,
        recorder,
        round_index,
        deadline,
    )
    outer_active = _search_active_indices(space, config.search_scope)
    covariance_diagonal = (
        np.diag(region.cma_covariance)
        if region.cma_covariance is not None
        else np.ones(len(space.names))
    )
    sensitivity = np.zeros(len(space.names), dtype=float)
    tip_sensitivity = np.zeros(len(space.names), dtype=float)
    wrist_sensitivity = np.zeros(len(space.names), dtype=float)
    best = start
    probe_step = max(float(config.sqp_fd_step), 1e-5)

    # 灵敏度探针只决定活跃集；所有探针仍调用真实 fourbar 并写入完整历史。
    for index in outer_active:
        if (
            time.monotonic() >= deadline
            or evaluation_counter[0] >= int(config.max_evaluations)
        ):
            break
        plus = start.y.copy()
        minus = start.y.copy()
        plus[index] = min(region.hi[index], plus[index] + probe_step)
        minus[index] = max(region.lo[index], minus[index] - probe_step)
        plus_candidate = _evaluate_candidate(
            plus, data, space, region.region_id, "sqp_sensitivity_probe",
            config.objective_mode, config.target_rmse_mm,
        )
        minus_candidate = _evaluate_candidate(
            minus, data, space, region.region_id, "sqp_sensitivity_probe",
            config.objective_mode, config.target_rmse_mm,
        )
        evaluation_start = evaluation_counter[0]
        evaluation_counter[0] += 2
        if recorder is not None:
            recorder.record_candidate(
                plus_candidate,
                evaluation_start + 1,
                round_index=round_index,
                metadata={
                    "sqp_probe_variable": space.names[index],
                    "direction": "plus",
                },
            )
            recorder.record_candidate(
                minus_candidate,
                evaluation_start + 2,
                round_index=round_index,
                metadata={
                    "sqp_probe_variable": space.names[index],
                    "direction": "minus",
                },
            )
        denominator = max(plus[index] - minus[index], 1e-12)
        if plus_candidate.valid and minus_candidate.valid:
            sensitivity[index] = abs(
                plus_candidate.score - minus_candidate.score
            ) / denominator
            tip_sensitivity[index] = abs(
                plus_candidate.metrics["tip"] - minus_candidate.metrics["tip"]
            ) / denominator
            wrist_sensitivity[index] = abs(
                plus_candidate.metrics["wrist"] - minus_candidate.metrics["wrist"]
            ) / denominator
        elif plus_candidate.valid:
            one_sided_step = max(plus[index] - start.y[index], 1e-12)
            sensitivity[index] = (
                abs(plus_candidate.score - start.score) / one_sided_step
            )
            tip_sensitivity[index] = abs(
                plus_candidate.metrics["tip"] - start.metrics["tip"]
            ) / one_sided_step
            wrist_sensitivity[index] = abs(
                plus_candidate.metrics["wrist"] - start.metrics["wrist"]
            ) / one_sided_step
        elif minus_candidate.valid:
            one_sided_step = max(start.y[index] - minus[index], 1e-12)
            sensitivity[index] = (
                abs(minus_candidate.score - start.score) / one_sided_step
            )
            tip_sensitivity[index] = abs(
                minus_candidate.metrics["tip"] - start.metrics["tip"]
            ) / one_sided_step
            wrist_sensitivity[index] = abs(
                minus_candidate.metrics["wrist"] - start.metrics["wrist"]
            ) / one_sided_step

    covariance_scale = np.sqrt(
        np.maximum(covariance_diagonal, float(config.cma_variance_floor))
    )
    score_activity = sensitivity * covariance_scale
    tip_activity = tip_sensitivity * covariance_scale
    wrist_activity = wrist_sensitivity * covariance_scale
    combined_activity = np.sqrt(
        score_activity ** 2 + tip_activity ** 2 + wrist_activity ** 2
    )
    active_count = min(len(outer_active), max(1, int(config.sqp_active_dimensions)))
    quota = max(1, active_count // 3)
    rankings = (
        outer_active[np.argsort(score_activity[outer_active])[::-1]],
        outer_active[np.argsort(tip_activity[outer_active])[::-1]],
        outer_active[np.argsort(wrist_activity[outer_active])[::-1]],
        outer_active[np.argsort(combined_activity[outer_active])[::-1]],
    )
    selected_list: list[int] = []
    for ranking in rankings[:3]:
        for index in ranking[:quota]:
            if int(index) not in selected_list:
                selected_list.append(int(index))
    for index in rankings[3]:
        if len(selected_list) >= active_count:
            break
        if int(index) not in selected_list:
            selected_list.append(int(index))
    selected_indices = np.asarray(selected_list, dtype=np.int32)
    trust_radius = float(config.sqp_trust_radius)
    local_lb = np.maximum(region.lo[selected_indices], start.y[selected_indices] - trust_radius)
    local_ub = np.minimum(region.hi[selected_indices], start.y[selected_indices] + trust_radius)
    free = (local_ub - local_lb) > 1e-10
    selected_indices = selected_indices[free]
    local_lb = local_lb[free]
    local_ub = local_ub[free]
    if selected_indices.size == 0:
        refined = start if start.score < entry_start.score else None
        no_active_event = {
            "round": round_index,
            "region_id": region.region_id,
            "stage": "slsqp_refinement_result",
            "status": "no_active_dimension",
            "evaluation_start": refinement_evaluation_start,
            "evaluation_end": int(evaluation_counter[0]),
            "before_score": float(entry_start.score),
            "after_score": float(start.score),
            "before_tip_rmse_mm": float(entry_start.metrics["tip"]),
            "before_wrist_rmse_mm": float(entry_start.metrics["wrist"]),
            "after_tip_rmse_mm": float(start.metrics["tip"]),
            "after_wrist_rmse_mm": float(start.metrics["wrist"]),
            "accepted_improvement_mm": float(max(0.0, entry_start.score - start.score)),
        }
        trace.append({"evaluation": evaluation_counter[0], **no_active_event})
        if recorder is not None:
            recorder.record_region_event("slsqp_refinement_result", no_active_event)
        return refined
    if recorder is not None:
        recorder.record_region_event(
            "slsqp_active_set",
            {
                "round": round_index,
                "region_id": region.region_id,
                "variables": [space.names[index] for index in selected_indices],
                "indices": selected_indices,
                "sensitivity": sensitivity[selected_indices],
                "tip_sensitivity": tip_sensitivity[selected_indices],
                "wrist_sensitivity": wrist_sensitivity[selected_indices],
                "normalized_lower": local_lb,
                "normalized_upper": local_ub,
            },
        )

    def objective(active_y: np.ndarray) -> float:
        nonlocal best
        if (
            time.monotonic() >= deadline
            or evaluation_counter[0] >= int(config.max_evaluations)
        ):
            return float(best.score)
        trial_y = start.y.copy()
        trial_y[selected_indices] = np.clip(active_y, local_lb, local_ub)
        candidate = _evaluate_candidate(
            trial_y,
            data,
            space,
            region.region_id,
            "slsqp",
            config.objective_mode,
            config.target_rmse_mm,
        )
        evaluation_counter[0] += 1
        if recorder is not None:
            recorder.record_candidate(
                candidate,
                evaluation_counter[0],
                round_index=round_index,
                metadata={
                    "active_indices": selected_indices,
                    "normalized_lower": local_lb,
                    "normalized_upper": local_ub,
                },
            )
        if candidate.valid and candidate.score < best.score:
            best = candidate
        trace.append({
            "evaluation": evaluation_counter[0],
            "stage": "slsqp",
            "region_id": region.region_id,
            "best_score": float(best.score),
            "trial_score": float(candidate.score),
        })
        return candidate.score

    status = "completed"
    error_message = ""
    try:
        minimize(
            objective,
            np.clip(start.y[selected_indices], local_lb, local_ub),
            method="SLSQP",
            jac=None,
            bounds=list(zip(local_lb, local_ub)),
            options={
                "maxiter": int(config.sqp_maxiter),
                "ftol": float(config.sqp_ftol),
                # SLSQP 的 eps 是归一化空间内的绝对差分步长；避免靠近 0
                # 的变量使用近乎零的相对步长而得到噪声梯度。
                "eps": max(float(config.sqp_fd_step), 1e-5),
                "disp": False,
            },
        )
    except Exception as error:
        status = "solver_exception"
        error_message = f"{type(error).__name__}: {error}"
    refined = best if best.score < entry_start.score else None
    final_candidate = best if refined is not None else entry_start
    result_event = {
        "round": round_index,
        "region_id": region.region_id,
        "stage": "slsqp_refinement_result",
        "status": status,
        "error": error_message,
        "evaluation_start": refinement_evaluation_start,
        "evaluation_end": int(evaluation_counter[0]),
        "before_score": float(entry_start.score),
        "after_score": float(final_candidate.score),
        "before_tip_rmse_mm": float(entry_start.metrics["tip"]),
        "before_wrist_rmse_mm": float(entry_start.metrics["wrist"]),
        "after_tip_rmse_mm": float(final_candidate.metrics["tip"]),
        "after_wrist_rmse_mm": float(final_candidate.metrics["wrist"]),
        "accepted_improvement_mm": float(
            max(0.0, entry_start.score - final_candidate.score)
        ),
        "accepted": bool(refined is not None),
    }
    trace.append({"evaluation": evaluation_counter[0], **result_event})
    if recorder is not None:
        recorder.record_region_event("slsqp_refinement_result", result_event)
    return refined


def _run_pso_refinement(
    start: _Candidate,
    region: _Region,
    data: ProblemData,
    space: DesignSpace,
    config: _OptimizeConfig,
    rng: np.random.Generator,
    trace: list[dict[str, Any]],
    evaluation_counter: list[int],
    recorder: _OptimizationRecorder | None,
    round_index: int,
    deadline: float,
) -> _Candidate | None:
    """在当前 CMA 叶区内对高灵敏度变量执行真实 fourbar 粒子群精修。"""

    if not start.valid or config.pso_iterations <= 0 or config.pso_particles <= 0:
        return None
    entry_start = start
    best = start
    refinement_evaluation_start = int(evaluation_counter[0])
    outer_active = _search_active_indices(space, config.search_scope)
    if outer_active.size == 0:
        return None
    covariance_diagonal = (
        np.diag(region.cma_covariance)
        if region.cma_covariance is not None
        else np.ones(len(space.names), dtype=float)
    )
    sensitivity = np.zeros(len(space.names), dtype=float)
    tip_sensitivity = np.zeros(len(space.names), dtype=float)
    wrist_sensitivity = np.zeros(len(space.names), dtype=float)
    probe_step = max(float(config.pso_fd_step), 1e-5)

    # 用正负真实模型探针同时衡量总目标、Tip 和 Wrist，再按三者配额选活动集。
    for index in outer_active:
        if (
            time.monotonic() >= deadline
            or evaluation_counter[0] >= int(config.max_evaluations)
        ):
            break
        plus = start.y.copy()
        minus = start.y.copy()
        plus[index] = min(region.hi[index], plus[index] + probe_step)
        minus[index] = max(region.lo[index], minus[index] - probe_step)
        plus_candidate = _evaluate_candidate(
            plus, data, space, region.region_id, "pso_sensitivity_probe",
            config.objective_mode, config.target_rmse_mm,
        )
        minus_candidate = _evaluate_candidate(
            minus, data, space, region.region_id, "pso_sensitivity_probe",
            config.objective_mode, config.target_rmse_mm,
        )
        evaluation_start = int(evaluation_counter[0])
        evaluation_counter[0] += 2
        if recorder is not None:
            recorder.record_candidate(
                plus_candidate,
                evaluation_start + 1,
                round_index=round_index,
                metadata={
                    "pso_probe_variable": space.names[index],
                    "direction": "plus",
                },
            )
            recorder.record_candidate(
                minus_candidate,
                evaluation_start + 2,
                round_index=round_index,
                metadata={
                    "pso_probe_variable": space.names[index],
                    "direction": "minus",
                },
            )
        valid_probes = [
            candidate
            for candidate in (plus_candidate, minus_candidate)
            if candidate.valid
        ]
        for candidate in valid_probes:
            if candidate.score < best.score:
                best = candidate
        denominator = max(plus[index] - minus[index], 1e-12)
        if plus_candidate.valid and minus_candidate.valid:
            sensitivity[index] = abs(
                plus_candidate.score - minus_candidate.score
            ) / denominator
            tip_sensitivity[index] = abs(
                plus_candidate.metrics["tip"] - minus_candidate.metrics["tip"]
            ) / denominator
            wrist_sensitivity[index] = abs(
                plus_candidate.metrics["wrist"] - minus_candidate.metrics["wrist"]
            ) / denominator
        elif valid_probes:
            candidate = valid_probes[0]
            candidate_step = max(
                abs(float(candidate.y[index] - start.y[index])), 1e-12
            )
            sensitivity[index] = abs(candidate.score - start.score) / candidate_step
            tip_sensitivity[index] = abs(
                candidate.metrics["tip"] - start.metrics["tip"]
            ) / candidate_step
            wrist_sensitivity[index] = abs(
                candidate.metrics["wrist"] - start.metrics["wrist"]
            ) / candidate_step

    covariance_scale = np.sqrt(
        np.maximum(covariance_diagonal, float(config.cma_variance_floor))
    )
    score_activity = sensitivity * covariance_scale
    tip_activity = tip_sensitivity * covariance_scale
    wrist_activity = wrist_sensitivity * covariance_scale
    combined_activity = np.sqrt(
        score_activity ** 2 + tip_activity ** 2 + wrist_activity ** 2
    )
    active_count = min(
        len(outer_active), max(1, int(config.pso_active_dimensions))
    )
    quota = max(1, active_count // 3)
    rankings = (
        outer_active[np.argsort(score_activity[outer_active])[::-1]],
        outer_active[np.argsort(tip_activity[outer_active])[::-1]],
        outer_active[np.argsort(wrist_activity[outer_active])[::-1]],
        outer_active[np.argsort(combined_activity[outer_active])[::-1]],
    )
    selected_list: list[int] = []
    for ranking in rankings[:3]:
        for index in ranking[:quota]:
            if int(index) not in selected_list:
                selected_list.append(int(index))
    for index in rankings[3]:
        if len(selected_list) >= active_count:
            break
        if int(index) not in selected_list:
            selected_list.append(int(index))
    selected_indices = np.asarray(selected_list, dtype=np.int32)
    trust_radius = max(float(config.pso_trust_radius), 1e-5)
    local_lb = np.maximum(
        region.lo[selected_indices], best.y[selected_indices] - trust_radius
    )
    local_ub = np.minimum(
        region.hi[selected_indices], best.y[selected_indices] + trust_radius
    )
    free = (local_ub - local_lb) > 1e-10
    selected_indices = selected_indices[free]
    local_lb = local_lb[free]
    local_ub = local_ub[free]
    if selected_indices.size == 0:
        return best if best.score < entry_start.score else None

    active_names = [space.names[index] for index in selected_indices]
    if recorder is not None:
        recorder.record_region_event(
            "pso_active_set",
            {
                "round": round_index,
                "region_id": region.region_id,
                "variables": active_names,
                "indices": selected_indices,
                "sensitivity": sensitivity[selected_indices],
                "tip_sensitivity": tip_sensitivity[selected_indices],
                "wrist_sensitivity": wrist_sensitivity[selected_indices],
                "normalized_lower": local_lb,
                "normalized_upper": local_ub,
                "physical_lower": space.lb[selected_indices]
                + local_lb
                * (space.ub[selected_indices] - space.lb[selected_indices]),
                "physical_upper": space.lb[selected_indices]
                + local_ub * (space.ub[selected_indices] - space.lb[selected_indices]),
            },
        )

    particle_count = max(4, int(config.pso_particles))
    dimension = int(selected_indices.size)
    span = np.maximum(local_ub - local_lb, 1e-12)
    center = np.clip(best.y[selected_indices], local_lb, local_ub)
    positions = rng.uniform(local_lb, local_ub, size=(particle_count, dimension))
    local_count = max(1, particle_count // 2)
    positions[:local_count] = np.clip(
        center
        + rng.normal(
            0.0,
            0.25 * span,
            size=(local_count, dimension),
        ),
        local_lb,
        local_ub,
    )
    positions[0] = center
    velocity_limit = max(float(config.pso_velocity_fraction), 1e-4) * span
    velocities = rng.uniform(
        -velocity_limit,
        velocity_limit,
        size=(particle_count, dimension),
    )
    velocities[0] = 0.0
    personal_positions = positions.copy()
    personal_scores = np.full(particle_count, math.inf, dtype=float)
    global_position = center.copy()
    global_candidate = best
    stale_iterations = 0
    iteration_count = max(1, int(config.pso_iterations))

    for iteration in range(iteration_count):
        if (
            time.monotonic() >= deadline
            or evaluation_counter[0] >= int(config.max_evaluations)
            or (
                not config.continue_after_target
                and _goal_reached(global_candidate.metrics, config.target_rmse_mm)
            )
        ):
            break
        before_score = float(global_candidate.score)
        tested_positions: list[np.ndarray] = []
        for particle_index in range(particle_count):
            if (
                time.monotonic() >= deadline
                or evaluation_counter[0] >= int(config.max_evaluations)
            ):
                break
            trial_y = best.y.copy()
            trial_y[selected_indices] = np.clip(
                positions[particle_index], local_lb, local_ub
            )
            candidate = _evaluate_candidate(
                trial_y,
                data,
                space,
                region.region_id,
                "pso_refinement",
                config.objective_mode,
                config.target_rmse_mm,
            )
            evaluation_counter[0] += 1
            tested_positions.append(candidate.y[selected_indices].copy())
            if recorder is not None:
                recorder.record_candidate(
                    candidate,
                    evaluation_counter[0],
                    round_index=round_index,
                    generation=iteration,
                    population_index=particle_index,
                    metadata={
                        "active_indices": selected_indices,
                        "normalized_lower": local_lb,
                        "normalized_upper": local_ub,
                    },
                )
            if candidate.valid and candidate.score < personal_scores[particle_index]:
                personal_scores[particle_index] = candidate.score
                personal_positions[particle_index] = np.clip(
                    candidate.y[selected_indices], local_lb, local_ub
                )
            if candidate.valid and candidate.score < global_candidate.score:
                global_candidate = candidate
                global_position = np.clip(
                    candidate.y[selected_indices], local_lb, local_ub
                )
            if candidate.valid and candidate.score < best.score:
                best = candidate
        if global_candidate.score < before_score - 1e-12:
            stale_iterations = 0
        else:
            stale_iterations += 1
        tested = (
            np.stack(tested_positions)
            if tested_positions
            else positions.copy()
        )
        event = {
            "round": round_index,
            "region_id": region.region_id,
            "iteration": iteration,
            "evaluation_end": int(evaluation_counter[0]),
            "particle_count": len(tested_positions),
            "active_indices": selected_indices,
            "variables": active_names,
            "tested_normalized_min": np.min(tested, axis=0),
            "tested_normalized_max": np.max(tested, axis=0),
            "tested_physical_min": space.lb[selected_indices]
            + np.min(tested, axis=0)
            * (space.ub[selected_indices] - space.lb[selected_indices]),
            "tested_physical_max": space.lb[selected_indices]
            + np.max(tested, axis=0)
            * (space.ub[selected_indices] - space.lb[selected_indices]),
            "before_score": before_score,
            "after_score": float(global_candidate.score),
            "tip_rmse_mm": float(global_candidate.metrics["tip"]),
            "wrist_rmse_mm": float(global_candidate.metrics["wrist"]),
            "stale_iterations": stale_iterations,
        }
        trace.append({
            "evaluation": int(evaluation_counter[0]),
            "stage": "pso_iteration",
            **event,
        })
        if recorder is not None:
            recorder.record_region_event("pso_iteration", event)
        if (
            not config.continue_after_target
            and _goal_reached(global_candidate.metrics, config.target_rmse_mm)
        ):
            break

        progress = iteration / max(iteration_count - 1, 1)
        inertia = (
            float(config.pso_inertia_start)
            + progress
            * (float(config.pso_inertia_end) - float(config.pso_inertia_start))
        )
        random_cognitive = rng.random((particle_count, dimension))
        random_social = rng.random((particle_count, dimension))
        velocities = (
            inertia * velocities
            + float(config.pso_cognitive)
            * random_cognitive
            * (personal_positions - positions)
            + float(config.pso_social)
            * random_social
            * (global_position - positions)
        )
        velocities = np.clip(velocities, -velocity_limit, velocity_limit)
        proposed = positions + velocities
        reflected = (proposed < local_lb) | (proposed > local_ub)
        velocities[reflected] *= -0.5
        positions = np.clip(proposed, local_lb, local_ub)

        if stale_iterations >= max(1, int(config.pso_stall_iterations)):
            restart_count = max(
                1,
                int(round(
                    np.clip(float(config.pso_restart_fraction), 0.0, 1.0)
                    * particle_count
                )),
            )
            worst = np.argsort(personal_scores)[-restart_count:]
            positions[worst] = rng.uniform(
                local_lb, local_ub, size=(restart_count, dimension)
            )
            velocities[worst] = rng.uniform(
                -velocity_limit,
                velocity_limit,
                size=(restart_count, dimension),
            )
            personal_scores[worst] = math.inf
            personal_positions[worst] = positions[worst]
            stale_iterations = 0
            if recorder is not None:
                recorder.record_region_event(
                    "pso_partial_restart",
                    {
                        "round": round_index,
                        "region_id": region.region_id,
                        "iteration": iteration,
                        "restarted_particles": worst,
                    },
                )

    refined = best if best.score < entry_start.score else None
    final_candidate = best if refined is not None else entry_start
    result_event = {
        "round": round_index,
        "region_id": region.region_id,
        "stage": "pso_refinement_result",
        "evaluation_start": refinement_evaluation_start,
        "evaluation_end": int(evaluation_counter[0]),
        "before_score": float(entry_start.score),
        "after_score": float(final_candidate.score),
        "before_tip_rmse_mm": float(entry_start.metrics["tip"]),
        "before_wrist_rmse_mm": float(entry_start.metrics["wrist"]),
        "after_tip_rmse_mm": float(final_candidate.metrics["tip"]),
        "after_wrist_rmse_mm": float(final_candidate.metrics["wrist"]),
        "accepted_improvement_mm": float(
            max(0.0, entry_start.score - final_candidate.score)
        ),
        "accepted": bool(refined is not None),
    }
    trace.append({"evaluation": int(evaluation_counter[0]), **result_event})
    if recorder is not None:
        recorder.record_region_event("pso_refinement_result", result_event)
    return refined


def _sensitivity_split(
    region: _Region,
    data: ProblemData,
    space: DesignSpace,
    next_region_id: int,
    evaluation_counter: list[int],
    recorder: _OptimizationRecorder | None,
    round_index: int,
    config: _OptimizeConfig,
    deadline: float,
) -> tuple[list[_Region], dict[str, Any]]:
    """用局部目标变化率选择分割轴，避免只按几何最长边机械分区。"""
    if region.best_y is None:
        return [region], {"split": False, "reason": "no_valid_point"}
    if region.evaluations_since_creation < int(config.minimum_samples_before_split):
        return [region], {
            "split": False,
            "reason": "insufficient_new_samples",
            "new_samples": region.evaluations_since_creation,
            "required": int(config.minimum_samples_before_split),
        }
    width = region.width()
    eligible = np.intersect1d(
        np.where(width > float(config.sensitivity_min_region_width))[0],
        _search_active_indices(space, config.search_scope),
        assume_unique=True,
    )
    if eligible.size == 0:
        return [region], {"split": False, "reason": "region_too_narrow"}
    # 对每个仍可分割的变量做正负小扰动，估计目标函数的局部变化率。
    baseline = region.best_score
    sensitivity = np.zeros(len(space.names))
    probe_candidates: list[_Candidate] = []
    for index in eligible:
        slopes: list[float] = []
        for requested_step in (0.005, 0.02, 0.05):
            if (
                time.monotonic() >= deadline
                or evaluation_counter[0] >= int(config.max_evaluations)
            ):
                break
            step = min(
                requested_step,
                float(config.sensitivity_probe_width_fraction) * width[index],
            )
            plus = region.best_y.copy()
            minus = region.best_y.copy()
            plus[index] = min(region.hi[index], plus[index] + step)
            minus[index] = max(region.lo[index], minus[index] - step)
            plus_result = _evaluate_candidate(
                plus, data, space, region.region_id, "split_probe",
                config.objective_mode, config.target_rmse_mm,
            )
            minus_result = _evaluate_candidate(
                minus, data, space, region.region_id, "split_probe",
                config.objective_mode, config.target_rmse_mm,
            )
            probe_candidates.extend((plus_result, minus_result))
            evaluation_start = evaluation_counter[0]
            evaluation_counter[0] += 2
            if recorder is not None:
                recorder.record_candidate(
                    plus_result,
                    evaluation_start + 1,
                    round_index=round_index,
                    metadata={
                        "probe_variable_index": int(index),
                        "probe_variable": space.names[index],
                        "direction": "plus",
                        "step_normalized": float(step),
                    },
                )
                recorder.record_candidate(
                    minus_result,
                    evaluation_start + 2,
                    round_index=round_index,
                    metadata={
                        "probe_variable_index": int(index),
                        "probe_variable": space.names[index],
                        "direction": "minus",
                        "step_normalized": float(step),
                    },
                )
            denominator = max(plus[index] - minus[index], 1e-12)
            if plus_result.valid and minus_result.valid:
                slopes.append(
                    abs(plus_result.score - minus_result.score) / denominator
                )
            elif plus_result.valid:
                slopes.append(
                    abs(plus_result.score - baseline)
                    / max(plus[index] - region.best_y[index], 1e-12)
                )
            elif minus_result.valid:
                slopes.append(
                    abs(minus_result.score - baseline)
                    / max(region.best_y[index] - minus[index], 1e-12)
                )
        if slopes:
            sensitivity[index] = float(np.median(slopes))
    # “灵敏度 x 当前区间宽度”兼顾变量的重要性和剩余可探索空间。
    split_index = int(eligible[np.argmax(sensitivity[eligible] * width[eligible])])
    guard = float(np.clip(config.split_guard_fraction, 0.0, 0.49))
    lower_guard = region.lo[split_index] + guard * width[split_index]
    upper_guard = region.hi[split_index] - guard * width[split_index]
    # 切分轴由灵敏度决定，切点则必须来自真实可行样本。旧版把最优点强行
    # 截到保护边界，可能产生一个完全没有可行种子的子区域，使 CMA-ES
    # 连续整代失效。这里选取可行样本坐标的中位位置，并落到最近的真实样本。
    split_pool = [
        candidate
        for candidate in [*region.samples, *probe_candidates]
        if candidate.valid
        and lower_guard <= candidate.y[split_index] <= upper_guard
    ]
    if not split_pool:
        return [region], {
            "split": False,
            "reason": "no_interior_feasible_split_seed",
            "variable": space.names[split_index],
            "lower_guard": float(lower_guard),
            "upper_guard": float(upper_guard),
        }
    split_coordinates = np.asarray(
        [candidate.y[split_index] for candidate in split_pool], dtype=float
    )
    median_coordinate = float(np.median(split_coordinates))
    split_seed = min(
        split_pool,
        key=lambda candidate: abs(
            float(candidate.y[split_index]) - median_coordinate
        ),
    )
    split_value = float(split_seed.y[split_index])
    left_hi = region.hi.copy()
    right_lo = region.lo.copy()
    left_hi[split_index] = split_value
    right_lo[split_index] = split_value
    left = _Region(next_region_id, region.lo.copy(), left_hi, region.depth + 1)
    right = _Region(next_region_id + 1, right_lo, region.hi.copy(), region.depth + 1)
    left.prior_score = float(region.best_score)
    right.prior_score = float(region.best_score)
    inherited_mean = (
        region.cma_mean.copy()
        if region.cma_mean is not None
        else region.best_y.copy()
    )
    for child in (left, right):
        child.cma_mean = np.clip(inherited_mean, child.lo, child.hi)
        if region.cma_diagonal is not None:
            child.cma_diagonal = region.cma_diagonal.copy()
        if region.cma_covariance is not None:
            child.cma_covariance = region.cma_covariance.copy()
        if region.cma_sigma is not None:
            child.cma_sigma = min(float(region.cma_sigma), float(config.sigma0))
        if region.cma_path_c is not None:
            child.cma_path_c = region.cma_path_c.copy()
        if region.cma_path_sigma is not None:
            child.cma_path_sigma = region.cma_path_sigma.copy()
        child.cma_generation_count = int(region.cma_generation_count)
    # 灵敏度探针也是真实模型评价，应进入对应子区域，不能只用于计算导数后丢弃。
    split_tolerance = 1e-12
    for candidate in [*region.samples, *probe_candidates]:
        coordinate = float(candidate.y[split_index])
        if coordinate <= split_value + split_tolerance:
            _update_region(
                left,
                [candidate],
                count_visit=False,
                count_evaluations=False,
            )
        if coordinate >= split_value - split_tolerance:
            _update_region(
                right,
                [candidate],
                count_visit=False,
                count_evaluations=False,
            )
    # 子区域必须从各自的可行样本重新建立搜索中心。旧实现让两个子区继承同一个
    # 父均值和已经收缩的 sigma，虽然边界已经分开，数值搜索仍挤在原来的小邻域。
    if config.reseed_split_children:
        for child in (left, right):
            _reseed_split_child(child, len(space.names), config.sigma0)
    split_vector = region.best_y.copy()
    split_vector[split_index] = split_value
    return [left, right], {
        "split": True,
        "parent_region_id": region.region_id,
        "child_region_ids": [left.region_id, right.region_id],
        "variable": space.names[split_index],
        "split_point_rule": "nearest feasible sample to the interior feasible-sample median",
        "split_seed_count": len(split_pool),
        "split_value_normalized": split_value,
        "split_value_absolute": float(_to_physical(split_vector, space)[split_index]),
        "local_sensitivity": float(sensitivity[split_index]),
        "sensitivity_by_variable": sensitivity.copy(),
        "eligible_variable_indices": eligible.copy(),
        "parent_lo": region.lo.copy(),
        "parent_hi": region.hi.copy(),
        "left_lo": left.lo.copy(),
        "left_hi": left.hi.copy(),
        "right_lo": right.lo.copy(),
        "right_hi": right.hi.copy(),
    }


def _save_optimization_outputs(
    best: _Candidate,
    data: ProblemData,
    space: DesignSpace,
    output_dir: Path,
    prefix: str,
    trace: list[dict[str, Any]],
    split_events: list[dict[str, Any]],
    elapsed_seconds: float,
    config: _OptimizeConfig,
    recorder: _OptimizationRecorder | None = None,
    evaluation_count: int | None = None,
) -> dict[str, Any]:
    """把最优机构、全部变量、迭代轨迹和分区事件保存为可复现文件。"""

    checkpoint_path = (
        Path(config.start_checkpoint_path).expanduser().resolve()
        if config.start_checkpoint_path
        else None
    )
    result, metrics, state = evaluate_design_vector(best.x, data, space, check_smooth=True)
    if result is None:
        raise RuntimeError("Best optimization point is no longer valid during output generation.")
    metrics["tip_high_harmonic_rms"] = _high_harmonic_rms(result.tip)
    metrics["wrist_high_harmonic_rms"] = _high_harmonic_rms(result.wrist)
    metrics["wrist_plane_mismatch_deg"] = _plane_normal_mismatch_deg(
        result.wrist, state.target_wrist
    )
    serializable_split_events = json.loads(
        json.dumps(split_events, ensure_ascii=False, default=_json_default)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / f"{prefix}_checkpoint.npz"
    np.savez_compressed(
        checkpoint,
        x=best.x,
        static=state.static,
        b_fourier_coeff=state.b_fourier_coeff,
        l2_fourier_coeff=state.l2_fourier_coeff,
        l2_values=state.l2_values,
        l31_fourier_coeff=state.l31_fourier_coeff,
        l31_values=state.l31_values,
        l32_values=state.l32_values,
        l3_fourier_coeff=state.l3_fourier_coeff,
        l3_values=state.l3_values,
        l5_fourier_coeff=state.l5_fourier_coeff,
        l5_values=state.l5_values,
        l6_fourier_coeff=state.l6_fourier_coeff,
        l6_values=state.l6_values,
        l7_fourier_coeff=state.l7_fourier_coeff,
        l7_values=state.l7_values,
        l8_fourier_coeff=state.l8_fourier_coeff,
        l8_values=state.l8_values,
        l12_fourier_coeff=state.l12_fourier_coeff,
        l12_values=state.l12_values,
        zc_fourier_coeff=state.zc_fourier_coeff,
        zc_split_parameters=state.zc_split_parameters,
        zc_values=state.zc_values,
        zc_start_index=result.zc_start_index,
        target_pose=state.target_pose,
        target_translation_mm=state.target_pose[:3],
        target_rotation_xyz_rad=state.target_pose[3:6],
        target_scale=float(state.target_pose[6]),
        target_tip_pose=(
            state.target_tip_pose
            if state.target_tip_pose is not None else state.target_pose
        ),
        target_wrist_pose=(
            state.target_wrist_pose
            if state.target_wrist_pose is not None else state.target_pose
        ),
        target_tip_translation_mm=(
            state.target_tip_pose[:3]
            if state.target_tip_pose is not None else state.target_pose[:3]
        ),
        target_wrist_translation_mm=(
            state.target_wrist_pose[:3]
            if state.target_wrist_pose is not None else state.target_pose[:3]
        ),
        target_tip_rotation_xyz_rad=(
            state.target_tip_pose[3:6]
            if state.target_tip_pose is not None else state.target_pose[3:6]
        ),
        target_wrist_rotation_xyz_rad=(
            state.target_wrist_pose[3:6]
            if state.target_wrist_pose is not None else state.target_pose[3:6]
        ),
        b_curve=result.b_curve,
        input_radius=result.input_radius,
        theta01=result.theta01,
        theta02=result.theta02,
        tip=result.tip,
        wrist=result.wrist,
        nodes=result.nodes,
        target_tip=state.target_tip,
        target_wrist=state.target_wrist,
        raw_target_tip=data.target_tip,
        raw_target_wrist=data.target_wrist,
        metrics=json.dumps(metrics, ensure_ascii=False),
        variable_names=np.array(space.names, dtype=object),
        variable_lb=space.lb,
        variable_ub=space.ub,
        variable_schema=(
            f"activeStatic{space.names.index(MOT_POLAR_NAMES[0])}_BFourier10_"
            f"{config.periodic_length_mode}_"
            + (
                (
                    "L5L6L8_Fourier3_ZC_split3_"
                    if config.periodic_length_mode
                    == "l5_l6_l8_zc_split_periodic3"
                    else "L5L8_Fourier3_ZC_split3_"
                )
                if config.periodic_length_mode in _L5_L8_ZC_SPLIT_MODES
                else "FourierCoefficientsPerRod"
                f"{periodic_coefficient_count(config.periodic_length_mode)}_"
            )
            + f"TargetPose_{config.target_pose_mode}_design_anchor"
        ),
    )
    with (output_dir / f"{prefix}_variables.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Index", "Name", "Lower", "Value", "Upper", "BoundaryPercent"])
        for index, (name, lower, value, upper) in enumerate(
            zip(space.names, space.lb, best.x, space.ub), start=1
        ):
            percent = 100.0 * (value - lower) / max(upper - lower, 1e-12)
            writer.writerow([index, name, lower, value, upper, percent])
    with (output_dir / f"{prefix}_trace.jsonl").open("w", encoding="utf-8") as stream:
        for row in trace:
            stream.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    _, report_stage1 = _three_stage_config(
        config,
        0.0,
        maximum_curve_rmse_mm=float("inf"),
        tip_rmse_mm=float("inf"),
        wrist_rmse_mm=float(config.stage1_advance_rmse_mm) + 1.0,
        minimum_stage_rank=1,
    )
    _, report_stage2 = _three_stage_config(
        config,
        0.0,
        maximum_curve_rmse_mm=float("inf"),
        tip_rmse_mm=float(config.stage2_advance_rmse_mm) + 1.0,
        wrist_rmse_mm=float(config.stage1_advance_rmse_mm) - 1.0,
        minimum_stage_rank=2,
    )
    _, report_stage3 = _three_stage_config(
        config,
        0.0,
        maximum_curve_rmse_mm=0.0,
        tip_rmse_mm=0.0,
        wrist_rmse_mm=0.0,
        minimum_stage_rank=3,
    )
    uses_user_table_bounds = config.static_bound_mode in {
        "user_table",
        "adaptive_expanded_20260727",
        "adaptive_expanded_v2_20260727",
        "adaptive_expanded_v3_20260727",
        "adaptive_expanded_v4_20260728",
        "adaptive_expanded_v5_20260728",
        "adaptive_expanded_v6_20260728",
        "broad_all54_20260802",
    }
    uses_adaptive_expanded_bounds = config.static_bound_mode in {
        "adaptive_expanded_20260727",
        "adaptive_expanded_v2_20260727",
        "adaptive_expanded_v3_20260727",
        "adaptive_expanded_v4_20260728",
        "adaptive_expanded_v5_20260728",
        "adaptive_expanded_v6_20260728",
        "broad_all54_20260802",
    }
    uses_adaptive_expanded_v2_bounds = (
        config.static_bound_mode == "adaptive_expanded_v2_20260727"
    )
    uses_adaptive_expanded_v3_bounds = (
        config.static_bound_mode == "adaptive_expanded_v3_20260727"
    )
    uses_adaptive_expanded_v4_bounds = (
        config.static_bound_mode == "adaptive_expanded_v4_20260728"
    )
    uses_adaptive_expanded_v5_bounds = (
        config.static_bound_mode == "adaptive_expanded_v5_20260728"
    )
    uses_adaptive_expanded_v6_bounds = (
        config.static_bound_mode == "adaptive_expanded_v6_20260728"
    )
    uses_broad_all54_bounds = (
        config.static_bound_mode == "broad_all54_20260802"
    )
    summary = {
        "task": "optimize",
        "model_file": "fourbar3d_python.py",
        "optimization_file": "fourbar_optimization.py",
        "no_anchor": checkpoint_path is None,
        "no_external_anchor": checkpoint_path is None,
        "same_campaign_continuation": False,
        "design_anchor_checkpoint": (
            str(checkpoint_path) if checkpoint_path is not None else None
        ),
        "design_anchor_space_mapping": (
            config.continuation_space_mapping
            if checkpoint_path is not None
            else None
        ),
        "initial_source": (
            (
                "explicit design-coordinate anchor; no population, covariance, "
                "evolution path, partition tree or SLSQP state loaded"
            )
            if checkpoint_path is not None
            else
            "original fourbar static parameters, model-defined B curve and zero high-order "
            "Fourier coefficients; no external checkpoint, population or best vector loaded"
        ),
        "algorithm": (
            (
                "quality-gated Wrist-upstream/Tip-downstream curriculum + "
                if config.three_stage_schedule
                else "direct equal-weight Tip/Wrist RMSE search + "
            )
            + "coverage-aware local-sensitivity partition + regional full-covariance "
            "CMA-ES with physics-block global/boundary injection and feasible "
            "backtracking + direct all-independent-variable pose search + "
            + "sensitivity-ranked normalized SLSQP refinement"
        ),
        "target_pose_mode": config.target_pose_mode,
        "periodic_length_mode": config.periodic_length_mode,
        "b_curve_mode": config.b_curve_mode,
        "static_bound_mode": config.static_bound_mode,
        "optimizer_hyperparameters": _reported_optimizer_config(config),
        "three_stage_schedule": {
            "enabled": bool(config.three_stage_schedule),
            "basis": (
                "quality-gated only: wall-clock time is not a default stopping condition; "
                "the Wrist-to-Tip-to-joint curriculum is monotonic "
                "and cannot regress to an earlier variable scope"
            ),
            "partition_split_axis_rule": "maximum local sensitivity times current region width",
            "partition_split_point_rule": (
                "nearest real feasible sample to the median coordinate of interior "
                "feasible samples on the selected axis"
            ),
            "stage1": {
                "condition": (
                    f"Wrist RMSE > {config.stage1_advance_rmse_mm:g} mm"
                ),
                "name": "Wrist upstream exploration",
                "search_scope": report_stage1.search_scope,
                "active_dimensions": int(
                    len(_search_active_indices(space, report_stage1.search_scope))
                ),
                "popsize": report_stage1.popsize,
                "generations_per_region": report_stage1.generations_per_region,
                "regions_per_round": report_stage1.regions_per_round,
                "sigma0": report_stage1.sigma0,
                "exploration": report_stage1.exploration,
                "max_depth": report_stage1.max_depth,
                "split_every_rounds": report_stage1.split_every_rounds,
                "splits_per_event": report_stage1.splits_per_event,
                "cma_local_block_fraction": report_stage1.cma_local_block_fraction,
                "sqp_maxiter": report_stage1.sqp_maxiter,
                "sqp_interval_rounds": report_stage1.sqp_interval_rounds,
                "sqp_active_dimensions": report_stage1.sqp_active_dimensions,
                "sqp_trust_radius": report_stage1.sqp_trust_radius,
                "sqp_ftol": report_stage1.sqp_ftol,
                "refinement_method": "SLSQP",
                "objective_mode": report_stage1.objective_mode,
                "objective_definition": (
                    "0.70*index RMSE_wrist + 0.30*best forward-cyclic aligned "
                    "RMSE_wrist + 2.0*abs(phase offset in steps) + "
                    "0.03*RMSE_tip + 0.001*peak_wrist + "
                    "0.02*Wrist high-harmonic RMS + 0.03*Wrist plane mismatch"
                    if config.wrist_phase_guidance
                    else (
                        (
                            "p8 smooth max(RMSE_wrist,"
                            f"{float(report_stage1.objective_mode.rsplit('_w', 1)[1]) if report_stage1.objective_mode.startswith('wrist_tip_compatibility_w') else 0.40:.2f}"
                            "*RMSE_tip)"
                        )
                        if report_stage1.objective_mode.startswith("wrist_tip_compatibility")
                        else "p8 smooth max of strict initialized equal-arc Tip/Wrist RMSE"
                        if report_stage1.objective_mode == "rmse_bottleneck_p8"
                        else "strict initialized equal-arc RMSE_wrist only"
                        if report_stage1.objective_mode == "wrist_rmse_only"
                        else "index RMSE_wrist + 0.03*RMSE_tip + "
                    "0.001*peak_wrist + 0.02*Wrist high-harmonic RMS + "
                    "0.03*Wrist plane mismatch"
                    )
                ),
            },
            "stage2": {
                "condition": (
                    f"Wrist RMSE <= {config.stage1_advance_rmse_mm:g} mm and "
                    f"Tip RMSE > {config.stage2_advance_rmse_mm:g} mm"
                ),
                "name": "Tip downstream convergence",
                "search_scope": report_stage2.search_scope,
                "active_dimensions": int(
                    len(_search_active_indices(space, report_stage2.search_scope))
                ),
                "popsize": report_stage2.popsize,
                "generations_per_region": report_stage2.generations_per_region,
                "regions_per_round": report_stage2.regions_per_round,
                "sigma0": report_stage2.sigma0,
                "exploration": report_stage2.exploration,
                "max_depth": report_stage2.max_depth,
                "split_every_rounds": report_stage2.split_every_rounds,
                "splits_per_event": report_stage2.splits_per_event,
                "cma_local_block_fraction": report_stage2.cma_local_block_fraction,
                "sqp_maxiter": report_stage2.sqp_maxiter,
                "sqp_interval_rounds": report_stage2.sqp_interval_rounds,
                "sqp_active_dimensions": report_stage2.sqp_active_dimensions,
                "sqp_trust_radius": report_stage2.sqp_trust_radius,
                "sqp_ftol": report_stage2.sqp_ftol,
                "refinement_method": "SLSQP",
                "objective_mode": report_stage2.objective_mode,
                "objective_definition": (
                    "p32 smooth max(RMSE_tip,RMSE_wrist), without auxiliary penalties"
                    if report_stage2.objective_mode == "rmse_bottleneck_p32"
                    else "RMSE_tip + 25*max(0,RMSE_wrist-"
                    f"{config.stage1_advance_rmse_mm:g}) + 0.10*RMSE_wrist + "
                    "0.001*max peak + 0.02*max high-harmonic RMS"
                ),
            },
            "stage3": {
                "condition": (
                    f"Wrist RMSE <= {config.stage1_advance_rmse_mm:g} mm and "
                    f"Tip RMSE <= {config.stage2_advance_rmse_mm:g} mm"
                ),
                "name": "joint refinement",
                "search_scope": "all outer variables",
                "active_dimensions": int(
                    len(_search_active_indices(space, "all"))
                ),
                "popsize": report_stage3.popsize,
                "generations_per_region": report_stage3.generations_per_region,
                "regions_per_round": report_stage3.regions_per_round,
                "sigma0": report_stage3.sigma0,
                "exploration": report_stage3.exploration,
                "max_depth": report_stage3.max_depth,
                "split_every_rounds": report_stage3.split_every_rounds,
                "splits_per_event": report_stage3.splits_per_event,
                "cma_local_block_fraction": report_stage3.cma_local_block_fraction,
                "sqp_maxiter": report_stage3.sqp_maxiter,
                "sqp_interval_rounds": report_stage3.sqp_interval_rounds,
                "sqp_active_dimensions": report_stage3.sqp_active_dimensions,
                "sqp_trust_radius": report_stage3.sqp_trust_radius,
                "sqp_ftol": report_stage3.sqp_ftol,
                "refinement_method": "SLSQP",
                "objective_mode": report_stage3.objective_mode,
            },
        },
        "objective_mode": config.objective_mode,
        "objective_definition": (
            "p64 smooth maximum of strict initialized equal-arc Tip/Wrist RMSE + "
            "2 mm times the larger original-time-trajectory stutter index"
            if config.objective_mode == "initialized_equal_arc_bottleneck_p64_with_stutter"
            else
            "sqrt((RMSE_tip^2 + RMSE_wrist^2)/2); equal Tip/Wrist weights "
            "without one-sided or auxiliary penalties"
            if config.objective_mode == "equal_weight_rmse"
            else
            "hypot(max(0,RMSE_tip-target),max(0,RMSE_wrist-target)) + "
            "0.10*max(RMSE_tip,RMSE_wrist) + 0.05*RMSE_combined + "
            "0.001*max(peak_tip,peak_wrist) + "
            "0.02*max(Tip/Wrist high-harmonic RMS above order 2) + "
            "0.02*Wrist PCA-plane normal mismatch (deg)"
            if config.objective_mode == "balanced_rmse_target"
            else (
                "p-norm smooth maximum of RMSE_tip/RMSE_wrist + "
                "0.05*RMSE_combined + 0.001*max(peak_tip,peak_wrist) + "
                "quality-gated guidance: p8 with 0.10 high-harmonic and 0.15 Wrist "
                "PCA-plane weights above the stage-1 threshold; p32 with 0.02 and "
                "0.03 weights in convergence/refinement"
                if config.objective_mode.startswith("smooth_bottleneck_p")
                else (
                    "p-norm smooth maximum of the two strict initialized equal-arc RMSE values; "
                    "no peak, spectral or plane-angle auxiliary penalty"
                    if config.objective_mode.startswith("rmse_bottleneck_p")
                    else "RMSE_combined + 0.015*max(peak_tip,peak_wrist)"
                )
            )
        ),
        "target_rmse_mm": float(config.target_rmse_mm),
        "success_criteria": (
            f"RMSE_tip < {config.target_rmse_mm:g} mm and "
            f"RMSE_wrist < {config.target_rmse_mm:g} mm"
        ),
        "success": _goal_reached(metrics, config.target_rmse_mm),
        "variable_count": len(space.names),
        "outer_search_dimension": int(len(_outer_active_indices(space))),
        "inner_solved_variables": sorted(_inner_solved_variables(space)),
        "dependent_or_fixed_outer_variables": sorted(
            {
                "B_Z_C3c"
                if space.b_curve_mode in {"fourier_z3", "fourier_z3_c0", "fourier_xyz3"}
                else "B_Z_C2c"
            }
        ),
        "compatibility_constants_not_in_design_space": sorted(
            _OUTPUT_INACTIVE_VARIABLES
        ),
        "variables": list(space.names),
        "constraints": {
            "B_parameterization": (
                "Bx and By are direct second-order Fourier functions; Bz is a direct "
                "third-order Fourier function with an optimized constant term"
                if config.b_curve_mode == "fourier_xyz3"
                else
                "Bx is constant; By is a direct second-order Fourier function; Bz is a "
                "direct third-order Fourier function with independent C0z"
                if config.b_curve_mode == "fourier_z3_c0"
                else
                "Bx is constant; By is a direct second-order Fourier function; Bz is a "
                "direct third-order Fourier function without a constant term"
                if config.b_curve_mode == "fourier_z3"
                else "Bx is constant; By and Bz are independent direct second-order Fourier functions; Bz has no constant term"
            ),
            "B_center_bounds_mm": (
                "B_CenterX [-120,300] mm, By constant [-130,80] mm, and B_Z_C0 "
                "[-60,40] mm in broad-all54 mode"
                if uses_broad_all54_bounds
                else
                "B_CenterX follows the selected static-bound mode, By constant term follows "
                "its existing bounds, and B_Z_C0 is optimized in [-20,20] mm"
                if config.b_curve_mode in {"fourier_xyz3", "fourier_z3_c0"}
                else
                "B_CenterX is optimized in [-80,240] mm, By constant term in "
                "[-100,50] mm, and Bz constant term is fixed at 0"
                if uses_adaptive_expanded_v6_bounds
                else "B_CenterX is optimized in [-80,160] mm, By constant term in "
                "[-50,50] mm, and Bz constant term is fixed at 0"
                if uses_adaptive_expanded_v4_bounds or uses_adaptive_expanded_v5_bounds
                else "B_CenterX is optimized in [-80,120] mm, By constant term in "
                "[-50,50] mm, and Bz constant term is fixed at 0"
                if uses_adaptive_expanded_bounds
                else "B_CenterX is optimized in [-80,80] mm, By constant term "
                "in [-50,50] mm, and Bz constant term is fixed at 0"
            ),
            "B_motion_plane": (
                "B follows a three-dimensional periodic curve; its X displacement is measured "
                "relative to B_CenterX"
                if config.b_curve_mode == "fourier_xyz3"
                else "B moves in a fixed plane parallel to global YZ at x=B_CenterX"
            ),
            "B_fourier_coefficients": (
                "Bx has first/second sine-cosine terms; By has C0 plus first/second terms; "
                "Bz has C0 plus first/second/third terms"
                if config.b_curve_mode == "fourier_xyz3"
                else
                "By has C0 plus first/second sine-cosine terms; Bz has independent C0z "
                "plus first/second/third sine-cosine terms"
                if config.b_curve_mode == "fourier_z3_c0"
                else
                "By has C0 plus first/second sine-cosine terms; Bz has first/second/third "
                "sine-cosine terms and C0z=0"
                if config.b_curve_mode == "fourier_z3"
                else "By has C0 plus first/second sine-cosine terms; Bz has first/second sine-cosine terms and C0z=0"
            ),
            "B_topology": (
                "self-intersection and non-single-connected projections are allowed; only a "
                "regular time parameterization and max|Bx-B_CenterX|<=30 mm are enforced"
                if config.b_curve_mode == "fourier_xyz3"
                else "the closed B curve is regular and strictly convex"
            ),
            "B_to_input_angles": (
                "AB(t)=norm(B(t)); theta01=atan2(sqrt(B_x^2+B_z^2),-B_y); "
                "theta02=unwrap(atan2(-B_z,B_x))"
            ),
            "L1": "legacy base32 compatibility value only; excluded from the optimization and all new kinematics",
            "fixed_compatibility_parameters": (
                "H_finger=3 mm and Lf1=412 mm remain in the complete design vector "
                "for MATLAB/input replay, but are excluded from CMA-ES and local refinement: "
                "H_finger belongs only to the disabled legacy wrist-angle branch, "
                "and Lf1 belongs only to the removed V/X branch"
            ),
            "first_loop_topology": (
                "BC=L2 is fixed over the cycle; AG=L3(t) is third-order Fourier; "
                "AD=L31 is fixed and DG=L32(t) is derived as L3(t)-L31+2; "
                "L31/L32/L3(t) form a triangle; C and Z are the same node, and the "
                "A-C-D triangle drives all downstream nodes"
                if config.periodic_length_mode == _CURRENT_PERIODIC_MODE
                else "BC=L2, AG=L3 and AD=L31 are static; L5(t), L6(t) and L8(t) are "
                "third-order Fourier lengths; Z separates from C at max|AC| and "
                "the A-Z-D triangle drives all downstream nodes"
                if config.periodic_length_mode
                == "l5_l6_l8_zc_split_periodic3"
                else "BC=L2, AG=L3, AD=L31 and L6 are static; L5(t) and L8(t) are "
                "third-order Fourier lengths; Z separates from C at max|AC| and "
                "the A-Z-D triangle drives all downstream nodes"
                if config.periodic_length_mode == "l5_l8_zc_split_periodic3"
                else "AG=L3(t) is a fourth-order Fourier periodic length; BC=L2 is static; "
                "Z=C+[0,-ZC(t),0]"
                if config.periodic_length_mode == "zc_l3_l7_periodic4"
                else "AG=L3(t) is a second-order Fourier periodic length; BC=L2 is static"
                if config.periodic_length_mode == "l3_l7_periodic"
                else "BC=L2(t) is a third-order Fourier periodic length; AG=L3 is static; "
                "C and Z are coincident at every frame"
                if config.periodic_length_mode == "l2_l31_l6_periodic3_l7_l8_fixed"
                else (
                    "BC=L2 and AG=L3 are independent static variables; C and Z are "
                    "coincident at every frame, so the first loop is A-C-D"
                    if config.periodic_length_mode in {
                        "l7_l8_periodic", "l7_l8_periodic3",
                        "l31_l8_periodic3_l7_fixed",
                        "l31_l6_periodic3_l7_l8_fixed",
                        "l2_l31_l6_periodic3_l7_l8_fixed",
                    }
                    else
                    "BC=L2 and AG=L3 are static; Z=C+[0,-ZC(t),0], and the A-Z-D triangle drives D/E"
                    if config.periodic_length_mode in {
                        "zc_l7_periodic_l3_fixed",
                        "zc_l7_l8_periodic4_l3_fixed",
                    }
                    else "BC=L2 and AG=L3; both are independent static optimization variables"
                )
            ),
            "L2_bound_mm": (
                "[10,350] mm in broad-all54 mode"
                if uses_broad_all54_bounds
                else
                "third-order Fourier periodic length with the complete-cycle envelope "
                "10<=L2(t)<=300 mm"
                if config.periodic_length_mode == "l2_l31_l6_periodic3_l7_l8_fixed"
                else "[10,300] mm in adaptive-expanded-v6 mode"
                if uses_adaptive_expanded_v6_bounds
                else "[10,320] mm in adaptive-expanded-v2 mode"
                if uses_adaptive_expanded_v2_bounds
                else "[10,250] mm in user-table/adaptive-v1 mode"
            ),
            "L13": (
                "independent static design variable in [8,100] mm"
                if uses_broad_all54_bounds
                else
                "independent static design variable in [10,80] mm"
                if uses_adaptive_expanded_v5_bounds or uses_adaptive_expanded_v6_bounds
                else "independent static design variable in [10,50] mm"
                if uses_user_table_bounds
                else "independent static design variable in [5,25] mm"
            ),
            "theta18": (
                "independent static angle in [30,150] deg; converted to radians inside thetaM"
                if uses_user_table_bounds
                else "independent static angle in [1,179] deg; converted to radians inside thetaM"
            ),
            "wrist": "node L",
            "RMSE": (
                "Excel-fixed phase zero; one synchronized closed-6D Tip+Wrist polyline "
                "is divided by coupled arc length; same-index comparison only, with no "
                "nearest-point, cyclic-shift, direction-reversal or candidate-specific "
                "phase alignment"
            ),
            "theta6_upper_bound": "theta6(t) < pi rad (180 deg) for every frame",
            "optimized_static_length_minimum_mm": (
                (
                    "user table plus the documented L5/L3/Lf2 expansions; L41/L12 "
                    + (
                        "minimum 5 mm; L_down=[10,100] mm in V4; all other listed rods "
                        "retain the stated lower bounds"
                        if uses_adaptive_expanded_v4_bounds
                        else "minimum 5 mm; L_down=[10,100], L13=[10,80], "
                        "L17=[10,150] mm in V5; all other listed rods retain "
                        "the stated lower bounds"
                        if uses_adaptive_expanded_v5_bounds or uses_adaptive_expanded_v6_bounds
                        else "minimum 5 mm; all other listed rods retain the stated lower bounds"
                    )
                    if uses_adaptive_expanded_bounds
                    else "user table applied exactly: L41/L12 minimum 5 mm; ordinary "
                    "listed rods otherwise use the stated lower bounds; inactive "
                    "compatibility parameters H_finger and Lf1 stay at their original values"
                )
                if uses_user_table_bounds
                else "10 mm except L13 in [5,25] mm"
            ),
            "static_bounds": (
                "the exact user-supplied 8h table; unlisted H_finger=[0.5,15] mm and "
                "Lf1=[250,520] mm; L_CZ is replaced by periodic ZC(t) in the current mode"
                if config.periodic_length_mode in {
                    "zc_l7_periodic_l3_fixed", "zc_l3_l7_periodic",
                    "zc_l3_l7_periodic3", "zc_l3_l7_periodic4",
                    "zc_l7_l8_periodic4_l3_fixed",
                    "l5_l8_zc_split_periodic3",
                    "l5_l6_l8_zc_split_periodic3",
                    "l3_l5_l8_zc2_periodic_l32_fixed",
                }
                else (
                    "the user-supplied 8h table with evidence-triggered expansions "
                    + (
                        "L2=[10,320], "
                        if uses_adaptive_expanded_v2_bounds
                        else ""
                    )
                    + "L5=[120,300], L3=[80,350], Lf2=[200,330] mm; all other "
                    + (
                        "static bounds unchanged except L_down=[10,100] mm; "
                        if uses_adaptive_expanded_v4_bounds
                        else "static bounds unchanged except L_down=[10,100], "
                        "L13=[10,80], L17=[10,150] mm; "
                        if uses_adaptive_expanded_v5_bounds or uses_adaptive_expanded_v6_bounds
                        else "static bounds unchanged; "
                    )
                    + "the legacy L_CZ slot is excluded because C=Z"
                    if uses_adaptive_expanded_bounds
                    else "the exact user-supplied 8h table; unlisted H_finger=[0.5,15] mm and "
                    "Lf1=[250,520] mm; the legacy L_CZ slot is excluded because C=Z"
                )
                if uses_user_table_bounds
                else (
                    "union of current bounds and the shared static-geometry bounds from the "
                    "RMSE25 Combined=25.150 mm baseline; 10 mm engineering minimum retained"
                    if config.static_bound_mode == "rmse25_union"
                    else "current model bounds"
                )
            ),
            "L6": (
                "L6(t)=C0+sum(k=1..3)[Ckc*cos(2*pi*k*phi)+Cks*sin(2*pi*k*phi)]; "
                + (
                    "200<=L6(t)<=300 mm and L6(t)>L61+0.05 mm"
                    if uses_user_table_bounds
                    else "L6(t)>L61+0.05 mm and L6(t)<=420 mm"
                )
                if config.periodic_length_mode in {
                    "l6_periodic",
                    "l31_l6_periodic3_l7_l8_fixed",
                    "l2_l31_l6_periodic3_l7_l8_fixed",
                    "l5_l6_l8_zc_split_periodic3",
                }
                else "one static design variable over all 76 frames; L6>L61+0.05 mm"
            ),
            "L5": (
                "one fixed optimization variable over all 76 frames; "
                "L5-L51-L52<=L3(t)-L31+2 for every frame"
                if config.periodic_length_mode == _CURRENT_PERIODIC_MODE
                else "third-order Fourier periodic length; its full-cycle envelope "
                "satisfies L51<=L5(t)/3, L52<=L5(t)/3 and "
                "L5(t)-L51-L52<=L3-L31+2"
                if config.periodic_length_mode in _L5_L8_ZC_SPLIT_MODES
                else "one static design variable over all 76 frames"
            ),
            "L31": (
                "one fixed optimization variable over all 76 frames; it is one side "
                "of the L31/[L3(t)-L31+2]/L3(t) triangle"
                if config.periodic_length_mode == _CURRENT_PERIODIC_MODE
                else "third-order Fourier periodic length with the full-cycle envelope "
                "10<=L31(t)<=150 mm; L32(t)=L3-L31(t)+2 is recomputed at every frame"
                if config.periodic_length_mode in {
                    "l31_l8_periodic3_l7_fixed",
                    "l31_l6_periodic3_l7_l8_fixed",
                    "l2_l31_l6_periodic3_l7_l8_fixed",
                }
                else "one static design variable over all 76 frames"
            ),
            "L32": (
                "derived internally at every frame as L3(t)-L31+2 and "
                "excluded from the optimization vector"
                if config.periodic_length_mode == _CURRENT_PERIODIC_MODE
                else "follows the selected legacy model definition"
            ),
            "L3": (
                "third-order Fourier periodic length over all 76 frames"
                if config.periodic_length_mode == _CURRENT_PERIODIC_MODE
                else "fourth-order Fourier periodic length with 100<=L3(t)<=350 mm"
                if config.periodic_length_mode == "zc_l3_l7_periodic4"
                else "second-order Fourier periodic length with 100<=L3(t)<=350 mm"
                if config.periodic_length_mode in {"l3_l7_periodic", "zc_l3_l7_periodic"}
                else (
                    (
                        "one static design variable over all 76 frames; 80<=L3<=350 mm"
                        if uses_adaptive_expanded_bounds
                        else "one static design variable over all 76 frames; 100<=L3<=350 mm"
                    )
                    if uses_user_table_bounds
                    else "one static design variable over all 76 frames"
                )
            ),
            "L7": (
                "one fixed optimization variable over all 76 frames"
                if config.periodic_length_mode == _CURRENT_PERIODIC_MODE
                else "fourth-order Fourier periodic length with 200<=L7(t)<=300 mm"
                if config.periodic_length_mode in {
                    "zc_l3_l7_periodic4",
                    "zc_l7_l8_periodic4_l3_fixed",
                }
                else "third-order Fourier periodic length with 200<=L7(t)<=300 mm"
                if config.periodic_length_mode == "l7_l8_periodic3"
                else "one static design variable over all 76 frames; 200<=L7<=300 mm"
                if config.periodic_length_mode in {
                    "l31_l8_periodic3_l7_fixed",
                    "l31_l6_periodic3_l7_l8_fixed",
                    "l2_l31_l6_periodic3_l7_l8_fixed",
                    "l5_l8_zc_split_periodic3",
                    "l5_l6_l8_zc_split_periodic3",
                }
                else "second-order Fourier periodic length with 200<=L7(t)<=300 mm"
                if config.periodic_length_mode in {
                    "l7_l8_periodic", "l3_l7_periodic", "l7_periodic_l3_fixed",
                    "zc_l7_periodic_l3_fixed", "zc_l3_l7_periodic",
                }
                else (
                    (
                        "one static design variable over all 76 frames; 200<=L7<=300 mm"
                        if uses_user_table_bounds
                        else "one static design variable over all 76 frames; L7>=10 mm"
                    )
                    if config.periodic_length_mode == "l6_periodic"
                    else "second-order Fourier periodic length with min(L7(t))>=10 mm"
                )
            ),
            "L8": (
                "third-order Fourier periodic length over all 76 frames"
                if config.periodic_length_mode == _CURRENT_PERIODIC_MODE
                else "fourth-order Fourier periodic length with 200<=L8(t)<=300 mm"
                if config.periodic_length_mode == "zc_l7_l8_periodic4_l3_fixed"
                else "third-order Fourier periodic length with 200<=L8(t)<=300 mm"
                if config.periodic_length_mode in {
                    "l7_l8_periodic3", "l31_l8_periodic3_l7_fixed",
                    "l5_l8_zc_split_periodic3",
                    "l5_l6_l8_zc_split_periodic3",
                }
                else "second-order Fourier periodic length with 200<=L8(t)<=300 mm"
                if config.periodic_length_mode == "l7_l8_periodic"
                else "one static design variable over all 76 frames; 200<=L8<=300 mm"
            ),
            "ZC": (
                "separation starts at the frame of max|AC|; "
                "d(tau)=A*sin(pi*tau)^2*(1+b*cos(2*pi*tau)+c*sin(2*pi*tau)); "
                "A in [0,150] mm, hypot(b,c)<=0.98; the expanded curve is "
                "second-order Fourier, nonnegative, starts/ends at zero, and Z drives A-Z-D"
                if config.periodic_length_mode in _L5_L8_ZC_SPLIT_MODES
                else "ZC(t) uses a fourth-order Fourier series; 10<=ZC(t)<=150 mm; "
                "Z=C+[0,-ZC(t),0] and Z drives the A-Z-D/D-E chain"
                if config.periodic_length_mode in {
                    "zc_l3_l7_periodic4",
                    "zc_l7_l8_periodic4_l3_fixed",
                }
                else "ZC(t)=C0+C1c*cos(2*pi*phi)+C1s*sin(2*pi*phi)+C2c*cos(4*pi*phi)+C2s*sin(4*pi*phi); "
                "0<=ZC(t)<=150 mm; Z_y(t)=min_tau(C_y(tau))-ZC(t), and Z drives the A-Z-D/D-E chain"
                if config.periodic_length_mode in {
                    "zc_l7_periodic_l3_fixed", "zc_l3_l7_periodic",
                    "l3_l5_l8_zc2_periodic_l32_fixed",
                }
                else "removed: C and Z are the same node and no ZC variable is decoded"
            ),
            "PO_QP_ratio": (
                "PO=L15 and QP=L16=L15, therefore PO/QP=1 for every frame"
            ),
            "J_O_geometry": (
                "J-O is retained as a labelled construction/coupling line for the wrist "
                "drop geometry; it is not introduced as an additional rigid closure equation"
            ),
            "theta02_theta_wrist_order": (
                "theta_wrist is first derived from the local M/J/K geometry; theta02 then expands nodes A through N "
                "into 3D, after which theta_wrist is applied only to downstream O/P/Q/W/S/T/U nodes"
            ),
            "mechanism_angles": (
                "all closed-loop physical/process angles must be nonnegative; theta_wrist is a signed "
                "derived fold angle and may be negative; every key triangle/QA2 closure angle must "
                "remain within [5,175] deg; all angle histories must satisfy a maximum "
                "cyclic circular step of 0.45 rad, including the final-to-first frame"
            ),
            "qa2_singularity_clearance": (
                "for every QA2 loop and frame, both diagonal triangle clearances must be at least "
                "max(2 mm, 0.01*(BC+CD))"
            ),
            "wrist_plane_continuity": (
                "every RExten construction requires normalized plane sine >=0.02 to prevent "
                "near-collinear normal-vector reversal"
            ),
            "trajectory_continuity": (
                "Tip/Wrist cyclic adjacent step must not exceed max(12*median step, 0.35*trajectory span)"
            ),
            "local_refinement": (
                "SLSQP is skipped while max(Tip RMSE,Wrist RMSE) exceeds "
                f"{config.sqp_trigger_rmse_mm:g} mm; below the gate, the stage schedule "
                "sets the active sensitivity-ranked outer-variable count and trust radius"
            ),
            "fixed_parameters": (
                "Z is dynamic and is not fixed in space"
                if config.periodic_length_mode in {
                    "zc_l7_periodic_l3_fixed", "zc_l3_l7_periodic",
                    "zc_l3_l7_periodic3", "zc_l3_l7_periodic4",
                    "zc_l7_l8_periodic4_l3_fixed",
                    "l5_l8_zc_split_periodic3",
                    "l5_l6_l8_zc_split_periodic3",
                    "l3_l5_l8_zc2_periodic_l32_fixed",
                }
                else
                "C and Z are coincident; the legacy L_CZ base32 slot is ignored"
                if config.periodic_length_mode in {
                    "l7_l8_periodic", "l7_l8_periodic3",
                    "l31_l8_periodic3_l7_fixed",
                    "l31_l6_periodic3_l7_l8_fixed",
                    "l2_l31_l6_periodic3_l7_l8_fixed",
                    "l3_l7_periodic", "l7_periodic_l3_fixed",
                    _CURRENT_PERIODIC_MODE,
                }
                else "L_CZ=62 mm"
                if uses_user_table_bounds
                else "H_finger, Lf1 and L_CZ follow the selected design-space definition"
            ),
            "target_translation_mm": (
                "one common translation for Tip and Wrist: Tx [-600,200], "
                "Ty [-100,200], Tz [-350,150]"
                if config.target_pose_mode == "shared_rotation_translation6"
                and uses_broad_all54_bounds
                else "one common translation for Tip and Wrist: Tx [-400,0], "
                "Ty [-600,0], Tz [-200,50]"
                if config.target_pose_mode == "shared_rotation_translation6"
                else
                "one common translation for Tip and Wrist: Tx [-400,0], "
                "Ty [-400,0], Tz [-200,50]"
                if config.target_pose_mode in {
                    "shared_fixed_rotation_translation3",
                    "shared_negative_rz_expanded_scale7",
                }
                else "one common translation for Tip and Wrist: Tx [-400,0], "
                "Ty [-200,200], Tz [-200,50]"
                if config.target_pose_mode in {
                    "shared_fixed_rotation_translation3",
                    "shared_fixed_rotation_scale4",
                    "shared_limited_rotation_scale7",
                    "shared_positive_rz_scale7",
                }
                else "Tx [-150,150], Ty [-200,200], Tz [-150,150]"
                if config.target_pose_mode in {"full", "rigid6"}
                else (
                    (
                        "Tip: Tx [-400,0], Ty [-300,200], Tz [-200,0]; "
                        "Wrist: Tx [-200,0], Ty [-150,100], Tz [-100,0]"
                        if uses_adaptive_expanded_v3_bounds
                        else "Tip: Tx [-400,0], Ty [-200,200], Tz [-200,0]; "
                        "Wrist: Tx [-200,0], Ty [-150,100], Tz [-100,0]"
                        if uses_adaptive_expanded_v4_bounds
                        else "Tip: Tx [-400,0], Ty [-200,200], Tz [-200,50]; "
                        "Wrist: Tx [-200,0], Ty [-150,100], Tz [-100,0]"
                        if uses_adaptive_expanded_v5_bounds or uses_adaptive_expanded_v6_bounds
                        else "Tip: Tx [-400,0], Ty [-200,200], Tz [-200,0]; "
                        "Wrist: Tx [-200,0], Ty [-100,100], Tz [-100,0]"
                    )
                    if config.target_pose_mode in {
                        "decoupled_full_rotation_scale13",
                        "decoupled_fixed_wrist_rotation_scale10",
                    }
                    else "Tip: Tx [-200,0], Ty [-200,-100], Tz [-200,0]; Wrist: Tx [-200,0], Ty [-100,100], Tz [-100,0]"
                    if config.target_pose_mode == "decoupled_fixed_rz_scale8"
                    else "Tip: Tx [-200,200], Ty [-200,200], Tz [-200,0]; Wrist: Tx [-200,0], Ty [-100,100], Tz [-100,0]"
                    if config.target_pose_mode == "decoupled_tip_ry_scale8"
                    else "Tip and Wrist independently: Tx [-50,50], Ty [-200,-100], Tz [-50,50]"
                    if config.target_pose_mode == "decoupled_constrained11"
                    else
                    "Tx [-50,50], Ty [-200,-100], Tz [-50,50]"
                    if config.target_pose_mode == "constrained6"
                    else
                    "Ty [-200,-100] only; Tx=0 and Tz=0"
                    if config.target_pose_mode == "ty_ry_rz"
                    else "Ty [-50,50] only; Tx=0 and Tz=0"
                )
            ),
            "target_translation_solver": (
                "disabled; common Tx/Ty/Tz are direct partition/CMA-ES/SLSQP variables"
                if config.target_pose_mode == "shared_rotation_translation6"
                else
                "independent bounded least-squares residual-mean projection for Tip and Wrist at every feasible candidate"
                if config.target_pose_mode in {
                    "decoupled_constrained11", "decoupled_fixed_rz_scale8",
                    "decoupled_tip_ry_scale8", "decoupled_full_rotation_scale13",
                    "decoupled_fixed_wrist_rotation_scale10",
                }
                else "one bounded common residual-mean projection for the stacked Tip/Wrist "
                "residuals at every feasible mechanism candidate"
                if config.target_pose_mode in {
                    "shared_fixed_rotation_scale4",
                    "shared_limited_rotation_scale7",
                    "shared_positive_rz_scale7",
                    "shared_negative_rz_expanded_scale7",
                }
                else "bounded balanced line search between the Tip-optimal and Wrist-optimal "
                "residual means at every feasible candidate; the projected value is retained only when the complete objective decreases"
            ),
            "target_rotation_rad": (
                "one common rotation: Rx [-0.9,0.9], Ry [-1.2,0.5], "
                "Rz [-1.2,0.5] rad"
                if config.target_pose_mode == "shared_rotation_translation6"
                and uses_broad_all54_bounds
                else "one common rotation: Rx [-0.5,0.5], Ry [-0.8,0], "
                "Rz [-0.8,0] rad"
                if config.target_pose_mode == "shared_rotation_translation6"
                else
                "one common fixed rotation: Rx=-0.27, Ry=-0.10, Rz=-0.71 rad"
                if config.target_pose_mode == "shared_fixed_rotation_translation3"
                else "one common fixed rotation: Rx=-0.269241, Ry=-0.099714, "
                "Rz=-0.705977 rad"
                if config.target_pose_mode == "shared_fixed_rotation_scale4"
                else "one common rotation centred at [-0.269241,-0.099714,-0.705977] "
                "rad, with each axis limited to +/-0.15 rad"
                if config.target_pose_mode == "shared_limited_rotation_scale7"
                else "one common rotation with Rx/Ry within +/-0.15 rad of "
                "[-0.269241,-0.099714], and nonnegative Rz in [0,0.15] rad"
                if config.target_pose_mode == "shared_positive_rz_scale7"
                else "one common rotation: Rx [-0.5,0.5], Ry [-0.5,0], "
                "Rz [-0.5,0] rad"
                if config.target_pose_mode == "shared_negative_rz_expanded_scale7"
                else "Rx/Ry/Rz each in [-1,1]"
                if config.target_pose_mode in {"full", "rigid6"}
                else (
                    "Tip: Rx [-1,1], Ry [-0.5,0.8], Rz [-0.5,0.5]; "
                    "Wrist: Rx [-1,1], Ry [-0.5,1], Rz [-0.5,0.5]"
                    if config.target_pose_mode == "decoupled_full_rotation_scale13"
                    else (
                        "Tip: Rx [-1.3,1], Ry [-0.85,0.8], Rz [-0.5,0.5]; "
                        "Wrist fixed at Rx=-0.269241, Ry=-0.099714, Rz=-0.705977 rad"
                        if uses_adaptive_expanded_bounds
                        else "Tip: Rx [-1,1], Ry [-0.5,0.8], Rz [-0.5,0.5]; "
                        "Wrist fixed at Rx=-0.269241, Ry=-0.099714, Rz=-0.705977 rad"
                    )
                    if config.target_pose_mode == "decoupled_fixed_wrist_rotation_scale10"
                    else "Tip Ry [-0.5,0.8], Wrist Ry [-0.5,0.8]; both Rx=Rz=0"
                    if config.target_pose_mode == "decoupled_fixed_rz_scale8"
                    else "Tip Ry [-0.5,0.8]; Wrist Ry=0; both Rx=Rz=0"
                    if config.target_pose_mode == "decoupled_tip_ry_scale8"
                    else "Tip and Wrist independently: Ry in [-0.8,0.8], Rz in [-0.5,0.5]; both Rx=0"
                    if config.target_pose_mode == "decoupled_constrained11"
                    else
                    "Ry in [-0.8,0.8], Rz in [-0.5,0.5]; Rx=0"
                    if config.target_pose_mode == "constrained6"
                    else
                    "Ry in [-0.8,0.8], Rz in [-0.5,0.5]; Rx=0"
                    if config.target_pose_mode == "ty_ry_rz"
                    else "Ry [-0.5,0.5] only; Rx=0 and Rz=0"
                )
            ),
            "target_scale": (
                "[0.8,1.2]"
                if config.target_pose_mode in {
                    "full", "constrained6", "decoupled_constrained11",
                    "decoupled_tip_ry_scale8", "decoupled_full_rotation_scale13",
                    "decoupled_fixed_wrist_rotation_scale10",
                    "shared_fixed_rotation_scale4",
                    "shared_limited_rotation_scale7",
                    "shared_positive_rz_scale7",
                    "shared_negative_rz_expanded_scale7",
                }
                else "fixed at 1.0"
            ),
            "target_initial_adjustment": "none",
            "target_rotation_scale_solver": (
                "disabled; common Rx/Ry/Rz and Tx/Ty/Tz are selected by regional CMA-ES/SLSQP, with scale fixed at 1"
                if config.target_pose_mode == "shared_rotation_translation6"
                else
                "fixed common rotation and unit scale; only the bounded common translation is solved analytically"
                if config.target_pose_mode == "shared_fixed_rotation_translation3"
                else "fixed common rotation with bounded analytic common scale and translation"
                if config.target_pose_mode == "shared_fixed_rotation_scale4"
                else "bounded weighted Procrustes alignment for one common rotation, "
                "one common scale and one common translation"
                if config.target_pose_mode in {
                    "shared_limited_rotation_scale7",
                    "shared_positive_rz_scale7",
                    "shared_negative_rz_expanded_scale7",
                }
                else "five bounded weighted-Procrustes candidates per feasible mechanism "
                "evaluation, using Tip weights 0.20/0.35/0.50/0.65/0.80"
                if config.target_pose_mode in {"full", "rigid6"}
                else (
                    "Tip/Wrist Rx/Ry/Rz are solved by independent bounded rigid alignment; translations and shared scale are then solved in the same deterministic inner projection"
                    if config.target_pose_mode == "decoupled_full_rotation_scale13"
                    else "Tip Rx/Ry/Rz are solved by bounded rigid alignment; Wrist rotation is fixed exactly; translations and shared scale are solved by deterministic inner projection"
                    if config.target_pose_mode == "decoupled_fixed_wrist_rotation_scale10"
                    else "disabled; independent Tip/Wrist Ry are selected by regional CMA-ES/SLSQP, while Rx/Rz=0 and scale=1"
                    if config.target_pose_mode == "decoupled_fixed_rz_scale8"
                    else "disabled; Tip Ry and shared scale are selected by regional CMA-ES/SLSQP, while Wrist Ry and all Rx/Rz are fixed at 0"
                    if config.target_pose_mode == "decoupled_tip_ry_scale8"
                    else "disabled; independent Tip/Wrist Ry/Rz and shared scale are selected by regional CMA-ES/SLSQP"
                    if config.target_pose_mode == "decoupled_constrained11"
                    else "disabled; Ry/Rz/scale are selected by regional CMA-ES/SLSQP"
                    if config.target_pose_mode == "constrained6"
                    else "disabled; Ry/Rz are selected by regional CMA-ES/SLSQP"
                    if config.target_pose_mode == "ty_ry_rz"
                    else "disabled; Ry is selected by regional CMA-ES/SLSQP"
                )
            ),
        },
        "metrics": metrics,
        "objective": float(best.score),
        "elapsed_seconds": float(elapsed_seconds),
        "evaluations": int(
            evaluation_count
            if evaluation_count is not None
            else (trace[-1]["evaluation"] if trace else 1)
        ),
        "split_events": serializable_split_events,
        "checkpoint": str(checkpoint),
    }
    if recorder is not None:
        summary["optimization_data_archive"] = str(recorder.run_dir.resolve())
    _write_json(output_dir / f"{prefix}_summary.json", summary)
    if recorder is not None:
        recorder.finalize(summary, checkpoint)
    return summary


def _validate_optimization_input_contract(config: _OptimizeConfig) -> dict[str, Any]:
    """Fail closed when the machine-readable input contract differs from runtime code."""

    if not config.input_contract_path:
        raise ValueError("an optimization input contract is mandatory")
    contract_path = Path(config.input_contract_path).expanduser().resolve()
    if not contract_path.is_file():
        raise FileNotFoundError(f"optimization input contract is missing: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("document", {}).get("version") != "2.0.0":
        raise ValueError("optimization input contract version must be 2.0.0")
    if contract.get("rmse", {}).get("correspondence_mode") != "strict_initialized_equal_arc_index":
        raise ValueError("optimization input contract has the wrong RMSE correspondence mode")
    if int(contract.get("target", {}).get("sample_count", -1)) != 76:
        raise ValueError("optimization input contract must contain exactly 76 target phases")
    design = contract.get("design_space", {})
    if (
        int(design.get("stored_coordinate_count", -1)) != 61
        or int(design.get("independent_search_dimension", -1)) != 60
        or design.get("derived_stored_coordinates") != ["B_Z_C3c"]
    ):
        raise ValueError("optimization input contract has the wrong 61/60 design-space contract")
    target_ty = next(
        (row for row in design.get("variables", []) if row.get("name") == "Target_Ty_mm"),
        None,
    )
    if target_ty is None or [float(target_ty["lower"]), float(target_ty["upper"])] != [-100.0, 200.0]:
        raise ValueError("optimization input contract must freeze Target_Ty_mm to [-100,200]")

    expected_runtime = contract.get("optimizer", {}).get("default", {})
    runtime_pairs = {
        "population": int(config.popsize),
        "generations_per_region": int(config.generations_per_region),
        "regions_per_round": int(config.regions_per_round),
        "sigma0": float(config.sigma0),
        "refinement_method": config.refinement_method,
        "sqp_active_dimensions": int(config.sqp_active_dimensions),
        "objective_mode": config.objective_mode,
        "search_scope": config.search_scope,
        "target_pose_mode": config.target_pose_mode,
        "periodic_length_mode": config.periodic_length_mode,
        "b_curve_mode": config.b_curve_mode,
        "static_bound_mode": config.static_bound_mode,
    }
    for key, observed in runtime_pairs.items():
        if expected_runtime.get(key) != observed:
            raise ValueError(
                f"runtime option {key}={observed!r} differs from input contract "
                f"{expected_runtime.get(key)!r}"
            )
    expected_anchor = expected_runtime.get("start_checkpoint_path")
    if not config.start_checkpoint_path or not expected_anchor:
        raise ValueError("this input contract requires an explicit design-only anchor")
    expected_anchor_path = Path(str(expected_anchor)).expanduser()
    if not expected_anchor_path.is_absolute():
        expected_anchor_path = CURRENT_DIR / expected_anchor_path
    observed_anchor_path = Path(config.start_checkpoint_path).expanduser().resolve()
    if observed_anchor_path != expected_anchor_path.resolve():
        raise ValueError(
            f"runtime anchor {observed_anchor_path} differs from contract "
            f"{expected_anchor_path.resolve()}"
        )
    anchor_contract = contract.get("anchor", {})
    if anchor_contract.get("role") != "design_coordinates_only":
        raise ValueError("anchor role must be design_coordinates_only")
    expected_anchor_hash = str(anchor_contract.get("sha256", "")).upper()
    if not expected_anchor_hash:
        raise ValueError("anchor SHA256 is missing from the input contract")
    if _sha256_file(observed_anchor_path).upper() != expected_anchor_hash:
        raise ValueError("design anchor SHA256 differs from the input contract")

    frozen_sources = contract.get("frozen_sources_at_generation", [])
    if not frozen_sources:
        raise ValueError("optimization input contract does not contain frozen source hashes")
    for row in frozen_sources:
        source_path = CURRENT_DIR / str(row["path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"frozen input source is missing: {source_path}")
        observed_hash = _sha256_file(source_path).upper()
        expected_hash = str(row["sha256"]).upper()
        if observed_hash != expected_hash:
            raise ValueError(
                f"frozen input source SHA256 mismatch for {source_path}: "
                f"{observed_hash} != {expected_hash}"
            )

    target = contract["target"]
    target_csv = CURRENT_DIR / "input" / Path(target["derived_target_csv"]).name
    target_metadata = CURRENT_DIR / "input" / Path(target["derived_target_metadata"]).name
    config.target_initialized_csv_path = str(target_csv)
    config.target_initialization_metadata_path = str(target_metadata)
    config.target_tip_txt_path = ""
    config.target_wrist_txt_path = ""
    return contract


def _run_full_optimization(config: _OptimizeConfig, output_dir: Path) -> dict[str, Any]:
    """创建版本化数据记录器，并在失败时保留已经产生的事件数据。"""

    _validate_optimization_input_contract(config)
    if not config.target_initialized_csv_path or not config.target_initialization_metadata_path:
        raise ValueError("initialized equal-arc target CSV and metadata are mandatory")
    if config.target_tip_txt_path or config.target_wrist_txt_path:
        raise ValueError("legacy Fourier TXT targets are forbidden for this optimizer revision")
    data = load_problem_data(
        target_initialized_csv_path=(
            config.target_initialized_csv_path or None
        ),
        target_initialization_metadata_path=(
            config.target_initialization_metadata_path or None
        ),
        target_tip_txt_path=(
            config.target_tip_txt_path or None
        ),
        target_wrist_txt_path=(
            config.target_wrist_txt_path or None
        ),
    )
    space = build_design_space(
        data,
        config.minimum_static_mm,
        target_pose_mode=config.target_pose_mode,
        periodic_length_mode=config.periodic_length_mode,
        static_bound_mode=config.static_bound_mode,
        b_curve_mode=config.b_curve_mode,
    )
    recorder = (
        _OptimizationRecorder.create(output_dir, data, space, config)
        if config.record_full_history
        else None
    )
    try:
        return _run_full_optimization_impl(
            config, output_dir, data, space, recorder
        )
    except BaseException as error:
        if recorder is not None:
            recorder.fail(error)
        raise


def _run_full_optimization_impl(
    config: _OptimizeConfig,
    output_dir: Path,
    data: ProblemData,
    space: DesignSpace,
    recorder: _OptimizationRecorder | None,
) -> dict[str, Any]:
    """从原始参数或同一任务检查点启动完整优化主循环。"""

    rng = np.random.default_rng(config.seed)
    start_time = time.monotonic()
    # 正式任务不再受原来的 4 小时限制。保留显式 minutes 仅用于用户主动要求的
    # 小规模限时试验；None 或非正数均表示运行至目标、评估上限或轮次上限。
    deadline = (
        math.inf
        if config.minutes is None or float(config.minutes) <= 0.0
        else start_time + max(1.0, 60.0 * float(config.minutes))
    )
    evaluation_counter = [0]
    trace: list[dict[str, Any]] = []
    split_events: list[dict[str, Any]] = []

    checkpoint_path = (
        Path(config.start_checkpoint_path).expanduser().resolve()
        if config.start_checkpoint_path
        else None
    )
    if checkpoint_path is None:
        # 严格无锚点分支：起点仅来自原始 fourbar 参数。
        initial = _find_feasible_initial(
            data, space, config, evaluation_counter, recorder
        )
    else:
        # Anchored analysis reads only the frozen design vector. Population,
        # covariance, evolution paths, partition tree and SLSQP state are rebuilt.
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Design anchor does not exist: {checkpoint_path}"
            )
        with np.load(checkpoint_path, allow_pickle=True) as checkpoint:
            checkpoint_x, continuation_mapping = _map_checkpoint_vector_to_space(
                checkpoint,
                space,
            )
        config.continuation_space_mapping = continuation_mapping
        initial = _evaluate_candidate(
            _to_normalized(checkpoint_x, space),
            data,
            space,
            0,
            "design_anchor_initial",
            config.objective_mode,
            config.target_rmse_mm,
        )
        evaluation_counter[0] += 1
        if recorder is not None:
            recorder.record_candidate(
                initial,
                evaluation_counter[0],
                round_index=-1,
                metadata={
                    "purpose": "design-coordinate anchor initialization",
                    "checkpoint_path": str(checkpoint_path),
                    "space_mapping": continuation_mapping,
                },
            )
            recorder.record_region_event(
                "design_anchor_loaded",
                {
                    "checkpoint_path": str(checkpoint_path),
                    "space_mapping": continuation_mapping,
                    "valid": bool(initial.valid),
                    "score": float(initial.score),
                    "tip_rmse_mm": float(initial.metrics["tip"]),
                    "wrist_rmse_mm": float(initial.metrics["wrist"]),
                },
            )
        if not initial.valid:
            raise RuntimeError(
                "Design anchor is invalid under the current model: "
                f"{checkpoint_path}"
            )
    initial_pool = _build_initial_feasible_pool(
        initial, data, space, config, evaluation_counter, recorder
    )
    trace.append({
        "evaluation": evaluation_counter[0],
        "stage": initial.stage,
        "region_id": 0,
        "best_score": initial.score,
    })
    root = _Region(0, np.zeros(len(space.names)), np.ones(len(space.names)))
    _update_region(
        root,
        initial_pool,
        count_visit=False,
        count_evaluations=True,
    )
    root.visits = 1
    if recorder is not None:
        recorder.record_region_event("region_created", {
            "round": -1,
            "region_id": root.region_id,
            "depth": root.depth,
            "lo": root.lo.copy(),
            "hi": root.hi.copy(),
            "source": "root",
        })
    regions = [root]
    best = min(initial_pool, key=lambda candidate: candidate.score)
    next_region_id = 1
    last_stage_name: str | None = None
    last_objective_mode: str | None = None
    last_stage3_block_phase: str | None = None
    minimum_stage_rank = 1
    stage3_cycle_index = 0
    deep_region_consolidated = False
    outer_stale_rounds = 0

    for round_index in range(int(config.max_rounds)):
        if (
            time.monotonic() >= deadline
            or evaluation_counter[0] >= int(config.max_evaluations)
            or (
                not config.continue_after_target
                and _goal_reached(best.metrics, config.target_rmse_mm)
            )
        ):
            break
        # 与 RMSE 29.99 基准一致，停滞判定首先比较两条曲线中的较大 RMSE，
        # 只有瓶颈相同时才比较 p64 平滑目标值。
        round_start_restart_key = (
            max(
                float(best.metrics.get("tip", math.inf)),
                float(best.metrics.get("wrist", math.inf)),
            ),
            float(best.score),
        )
        elapsed_fraction = (
            0.0
            if not math.isfinite(deadline)
            else (time.monotonic() - start_time)
            / max(deadline - start_time, 1e-12)
        )
        stage_name, round_config = _three_stage_config(
            config,
            elapsed_fraction,
            max(
                float(best.metrics.get("tip", math.inf)),
                float(best.metrics.get("wrist", math.inf)),
            ),
            tip_rmse_mm=float(best.metrics.get("tip", math.inf)),
            wrist_rmse_mm=float(best.metrics.get("wrist", math.inf)),
            minimum_stage_rank=minimum_stage_rank,
        )
        minimum_stage_rank = max(
            minimum_stage_rank,
            {
                "stage1_wrist_upstream": 1,
                "stage2_tip_downstream": 2,
                "stage3_joint_refinement": 3,
                "single_stage": 3,
            }[stage_name],
        )
        stage3_block_phase: str | None = None
        if stage_name == "stage3_joint_refinement" and config.stage3_block_cycle:
            stage3_block_phase, round_config = _stage3_block_config(
                round_config,
                stage3_cycle_index,
            )
            stage3_cycle_index += 1
        stage_changed = stage_name != last_stage_name
        objective_changed = round_config.objective_mode != last_objective_mode
        block_changed = bool(
            stage3_block_phase is not None
            and stage3_block_phase != last_stage3_block_phase
        )
        if stage_changed or objective_changed:
            best = _refresh_region_scores(
                regions,
                round_config.objective_mode,
                round_config.target_rmse_mm,
            )
            if recorder is not None:
                recorder.best_score = float(best.score)
        if stage_changed or block_changed:
            # 活动域切换后，上一阶段协方差在新变量子空间中没有统计意义。
            # 以每个叶区当前最优可行点作为新均值，重新学习该阶段的协方差。
            for region in regions:
                region.cma_mean = (
                    region.best_y.copy()
                    if region.best_y is not None
                    else region.cma_mean
                )
                region.cma_diagonal = None
                region.cma_covariance = None
                region.cma_sigma = None
                region.cma_path_c = None
                region.cma_path_sigma = None
                region.cma_generation_count = 0
                region.cma_stale_generations = 0
            if recorder is not None:
                recorder.best_score = float(best.score)
            stage_event = {
                "round": round_index,
                "evaluation": evaluation_counter[0],
                "stage_name": stage_name,
                "covariance_reset_reason": (
                    "stage_changed" if stage_changed else "stage3_block_changed"
                ),
                "elapsed_fraction": float(elapsed_fraction),
                "parameters": {
                    "popsize": round_config.popsize,
                    "generations_per_region": round_config.generations_per_region,
                    "regions_per_round": round_config.regions_per_round,
                    "sigma0": round_config.sigma0,
                    "exploration": round_config.exploration,
                    "max_depth": round_config.max_depth,
                    "split_every_rounds": round_config.split_every_rounds,
                    "splits_per_event": round_config.splits_per_event,
                    "cma_local_block_fraction": (
                        round_config.cma_local_block_fraction
                    ),
                    "sqp_maxiter": round_config.sqp_maxiter,
                    "sqp_ftol": round_config.sqp_ftol,
                    "refinement_method": "SLSQP",
                    "objective_mode": round_config.objective_mode,
                    "search_scope": round_config.search_scope,
                    "active_dimensions": int(
                        len(_search_active_indices(space, round_config.search_scope))
                    ),
                },
            }
            trace.append({
                "best_score": float(best.score),
                "region_id": int(best.region_id),
                **stage_event,
            })
            if recorder is not None:
                recorder.record_region_event("optimization_stage_transition", stage_event)
            last_stage_name = stage_name
        if stage3_block_phase is not None:
            block_event = {
                "round": round_index,
                "evaluation": evaluation_counter[0],
                "stage": "stage3_block_phase",
                "block_phase": stage3_block_phase,
                "objective_mode": round_config.objective_mode,
                "search_scope": round_config.search_scope,
                "active_dimensions": int(
                    len(_search_active_indices(space, round_config.search_scope))
                ),
            }
            trace.append(block_event)
            if recorder is not None:
                recorder.record_region_event("stage3_block_phase", block_event)
        last_stage3_block_phase = stage3_block_phase
        last_objective_mode = round_config.objective_mode
        maximum_rmse = max(
            float(best.metrics.get("tip", math.inf)),
            float(best.metrics.get("wrist", math.inf)),
        )
        if (
            stage_name == "stage3_joint_refinement"
            and round_config.stage3_deep_consolidation
            and maximum_rmse <= float(round_config.stage3_deep_threshold_mm)
            and not deep_region_consolidated
        ):
            consolidated = _consolidate_deep_region(
                best,
                len(space.names),
                next_region_id,
            )
            regions = [consolidated]
            next_region_id += 1
            deep_region_consolidated = True
            event = {
                "round": round_index,
                "evaluation": evaluation_counter[0],
                "stage": "deep_region_consolidation",
                "region_id": consolidated.region_id,
                "threshold_rmse_mm": float(
                    round_config.stage3_deep_threshold_mm
                ),
                "tip_rmse_mm": float(best.metrics["tip"]),
                "wrist_rmse_mm": float(best.metrics["wrist"]),
                "lo": consolidated.lo.copy(),
                "hi": consolidated.hi.copy(),
            }
            trace.append(event)
            if recorder is not None:
                recorder.record_region_event(
                    "deep_region_consolidation",
                    event,
                )
        total_visits = 1 + sum(region.visits for region in regions)

        def priority(region: _Region) -> float:
            # 分数越低越优；探索奖励使访问次数较少的区域仍有机会被选中。
            effective_score = (
                region.best_score
                if np.isfinite(region.best_score)
                else region.prior_score
            )
            if not np.isfinite(effective_score):
                effective_score = 2.0 * max(1.0, abs(best.score))
            exploration = round_config.exploration * math.sqrt(
                math.log(total_visits + 1.0) / (region.visits + 1.0)
            )
            return effective_score - exploration * max(1.0, abs(best.score))

        def selection_key(region: _Region) -> tuple[int, float]:
            # 新叶区先得到最低评价配额，再按质量/探索优先级竞争预算。
            needs_coverage = (
                region.evaluations_since_creation
                < int(round_config.minimum_leaf_evaluations)
            )
            return (0 if needs_coverage else 1, priority(region))

        # 每轮先完成所有入选叶区的 CMA-ES，再从本轮全部候选中选择真正最优的一点
        # 执行一次局部精修。旧逻辑会把精修配额交给排序最靠前的叶区，即使后续叶区
        # 找到更好的 CMA 候选也无法获得精修，造成局部求解预算错配。
        selected = sorted(regions, key=selection_key)[
            : max(1, round_config.regions_per_round)
        ]
        if recorder is not None:
            recorder.record_region_event("round_region_selection", {
                "round": round_index,
                "selected_region_ids": [region.region_id for region in selected],
                "leaf_regions": [
                    {
                        "region_id": region.region_id,
                        "depth": region.depth,
                        "visits": region.visits,
                        "evaluations_since_creation": region.evaluations_since_creation,
                        "best_score": region.best_score,
                        "prior_score": region.prior_score,
                        "priority": priority(region),
                        "lo": region.lo.copy(),
                        "hi": region.hi.copy(),
                    }
                    for region in regions
                ],
            })
        partition_active_indices = _outer_active_indices(space)
        if round_config.refinement_method == "pso":
            refinement_enabled = (
                round_config.pso_iterations > 0
                and round_config.pso_particles > 0
            )
            refinement_interval = round_config.pso_interval_rounds
        elif round_config.refinement_method == "slsqp":
            refinement_enabled = round_config.sqp_maxiter > 0
            refinement_interval = round_config.sqp_interval_rounds
        else:
            raise ValueError(
                "refinement_method must be 'pso' or 'slsqp': "
                f"{round_config.refinement_method}"
            )
        refinement_due = (
            refinement_enabled
            and round_index % max(1, int(refinement_interval)) == 0
        )
        refinement_options: list[tuple[_Candidate, _Region]] = []
        for selected_rank, region in enumerate(selected):
            if time.monotonic() >= deadline:
                break
            candidates = _run_cma_es(
                region, data, space, round_config, rng, trace, evaluation_counter,
                recorder, round_index, deadline,
            )
            valid = [
                candidate
                for candidate in candidates
                if candidate.valid
                and _point_in_region(
                    candidate.y, region, partition_active_indices
                )
            ]
            if valid:
                cma_best = min(valid, key=lambda candidate: candidate.score)
                refinement_options.append((cma_best, region))
            # 所有样本按修复后的实际坐标归属叶区；越区点不得更新原区域。
            routed: dict[int, list[_Candidate]] = {}
            for candidate in candidates:
                target = _locate_region(
                    candidate.y, regions, partition_active_indices
                )
                if target is not None:
                    candidate.region_id = target.region_id
                    routed.setdefault(target.region_id, []).append(candidate)
            updated_selected = False
            for target in regions:
                target_candidates = routed.get(target.region_id, [])
                if not target_candidates:
                    continue
                is_selected_region = target is region
                _update_region(
                    target,
                    target_candidates,
                    count_visit=is_selected_region,
                    count_evaluations=True,
                )
                updated_selected = updated_selected or is_selected_region
            if not updated_selected:
                _update_region(
                    region,
                    [],
                    count_visit=True,
                    count_evaluations=False,
                )
            valid = [candidate for candidate in candidates if candidate.valid]
            if valid:
                local_best = min(valid, key=lambda candidate: candidate.score)
                if local_best.score < best.score:
                    best = local_best
                if (
                    not config.continue_after_target
                    and _goal_reached(best.metrics, config.target_rmse_mm)
                ):
                    break

        if (
            refinement_due
            and refinement_options
            and time.monotonic() < deadline
            and (
                config.continue_after_target
                or not _goal_reached(best.metrics, config.target_rmse_mm)
            )
        ):
            refinement_start, refinement_region = min(
                refinement_options,
                key=lambda item: item[0].score,
            )
            selection_event = {
                "round": round_index,
                "region_id": refinement_region.region_id,
                "candidate_score": float(refinement_start.score),
                "eligible_region_count": len(refinement_options),
            }
            trace.append({
                "evaluation": evaluation_counter[0],
                "stage": f"{round_config.refinement_method}_round_best_selection",
                **selection_event,
            })
            if recorder is not None:
                recorder.record_region_event(
                    f"{round_config.refinement_method}_round_best_selection",
                    selection_event,
                )
            if round_config.refinement_method == "pso":
                refined = _run_pso_refinement(
                    refinement_start,
                    refinement_region,
                    data,
                    space,
                    round_config,
                    rng,
                    trace,
                    evaluation_counter,
                    recorder,
                    round_index,
                    deadline,
                )
            else:
                refined = _run_slsqp(
                    refinement_start,
                    refinement_region,
                    data,
                    space,
                    round_config,
                    trace,
                    evaluation_counter,
                    recorder,
                    round_index,
                    deadline,
                )
            if refined is not None:
                target = _locate_region(
                    refined.y,
                    regions,
                    partition_active_indices,
                )
                if target is not None:
                    refined.region_id = target.region_id
                    _update_region(
                        target,
                        [refined],
                        count_visit=False,
                        count_evaluations=True,
                    )
                if refined.score < best.score:
                    best = refined

        if (
            not config.continue_after_target
            and _goal_reached(best.metrics, config.target_rmse_mm)
        ):
            break

        # 每隔若干轮，对当前最优叶区域执行一次基于局部灵敏度的二分。
        if (
            round_index % max(1, round_config.split_every_rounds) == 0
            and len(regions) < 2 ** round_config.max_depth
        ):
            splittable = [
                region for region in regions
                if region.depth < round_config.max_depth and region.best_y is not None
            ]
            if splittable:
                # 同一分割事件同时推进高质量区域和欠访问区域，避免区域树退化成单条深链。
                parents = sorted(splittable, key=selection_key)[
                    : max(1, round_config.splits_per_event)
                ]
                for parent in parents:
                    if time.monotonic() >= deadline or parent not in regions:
                        break
                    children, event = _sensitivity_split(
                        parent, data, space, next_region_id, evaluation_counter,
                        recorder, round_index, round_config, deadline,
                    )
                    full_event = {"round": round_index, **event}
                    split_events.append(full_event)
                    if recorder is not None:
                        recorder.record_region_event("region_split", full_event)
                    if len(children) == 2:
                        regions.remove(parent)
                        regions.extend(children)
                        child_candidates = [
                            candidate
                            for child in children
                            for candidate in child.samples
                            if candidate.valid
                        ]
                        if child_candidates:
                            child_best = min(child_candidates, key=lambda item: item.score)
                            if child_best.score < best.score:
                                best = child_best
                        next_region_id += 2

        round_end_restart_key = (
            max(
                float(best.metrics.get("tip", math.inf)),
                float(best.metrics.get("wrist", math.inf)),
            ),
            float(best.score),
        )
        bottleneck_improvement_mm = float(
            round_start_restart_key[0] - round_end_restart_key[0]
        )
        score_improvement_mm = float(
            round_start_restart_key[1] - round_end_restart_key[1]
        )
        minimum_meaningful_improvement_mm = max(
            float(config.cma_outer_min_improvement_mm),
            float(config.cma_outer_min_improvement_fraction)
            * max(1.0, abs(float(round_start_restart_key[0]))),
        )
        meaningful_improvement = (
            bottleneck_improvement_mm >= minimum_meaningful_improvement_mm
            or score_improvement_mm >= minimum_meaningful_improvement_mm
        )
        outer_stale_rounds = 0 if meaningful_improvement else outer_stale_rounds + 1
        outer_active = _outer_active_indices(space)
        boundary_distance = (
            float(np.min(np.minimum(best.y[outer_active], 1.0 - best.y[outer_active])))
            if outer_active.size
            else 1.0
        )
        boundary_saturated = boundary_distance <= float(
            config.cma_boundary_trigger_fraction
        )
        stale_round_threshold = (
            int(config.cma_boundary_stagnation_rounds)
            if boundary_saturated
            else int(config.cma_outer_stagnation_rounds)
        )
        if (
            config.cma_restart_mode == "outer_round_baseline"
            and outer_stale_rounds >= stale_round_threshold
        ):
            factor_before = float(config.cma_restart_sigma_factor)
            factor_after = min(
                float(config.cma_restart_factor_max),
                factor_before * float(config.cma_restart_growth_factor),
            )
            config.cma_restart_sigma_factor = factor_after
            restarted_region_ids: list[int] = []
            for region in regions:
                if region.best_y is None:
                    continue
                current_sigma = (
                    float(region.cma_sigma)
                    if region.cma_sigma is not None
                    else float(config.sigma0)
                )
                center_blend = (
                    float(np.clip(config.cma_boundary_restart_center_blend, 0.0, 1.0))
                    if boundary_saturated
                    else 0.0
                )
                region.cma_mean = (
                    (1.0 - center_blend) * region.best_y
                    + center_blend * region.center()
                )
                region.cma_diagonal = None
                region.cma_covariance = None
                region.cma_sigma = _cma_restart_sigma(current_sigma, config)
                region.cma_path_c = None
                region.cma_path_sigma = None
                region.cma_generation_count = 0
                region.cma_stale_generations = 0
                restarted_region_ids.append(int(region.region_id))
            restart_event = {
                "round": round_index,
                "evaluation": evaluation_counter[0],
                "stage": "outer_stagnation_restart",
                "stale_round_threshold": stale_round_threshold,
                "minimum_meaningful_improvement_mm": minimum_meaningful_improvement_mm,
                "bottleneck_improvement_mm": bottleneck_improvement_mm,
                "score_improvement_mm": score_improvement_mm,
                "boundary_distance_fraction": boundary_distance,
                "boundary_saturated": boundary_saturated,
                "restart_center_blend": center_blend,
                "restart_factor_before": factor_before,
                "restart_factor_after": factor_after,
                "restart_growth_factor": float(
                    config.cma_restart_growth_factor
                ),
                "restart_factor_max": float(config.cma_restart_factor_max),
                "restarted_region_ids": restarted_region_ids,
                "before_max_rmse_mm": float(round_start_restart_key[0]),
                "after_max_rmse_mm": float(round_end_restart_key[0]),
            }
            trace.append(restart_event)
            if recorder is not None:
                recorder.record_region_event(
                    "outer_stagnation_restart",
                    restart_event,
                )
            outer_stale_rounds = 0

    return _save_optimization_outputs(
        best,
        data,
        space,
        output_dir,
        config.prefix,
        trace,
        split_events,
        time.monotonic() - start_time,
        config,
        recorder,
        evaluation_counter[0],
    )


# =============================================================================
# 4. 固定杆区域分析
# =============================================================================
# 本节与完整优化严格分开：Mot 输入曲线和目标旋转冻结，L6 为静态杆长，
# L7 周期项折算成周期均值固定杆长；目标平移与尺度仍参与波动分析。


def _build_analysis_context(checkpoint_path: Path) -> _AnalysisContext:
    """从检查点构建 29 维固定驱动曲线、固定 L7 的区域分析问题。"""

    data = load_problem_data()
    state = load_checkpoint_state(checkpoint_path)
    lb_map = dict(zip(state.variable_names, state.lb if state.lb is not None else []))
    ub_map = dict(zip(state.variable_names, state.ub if state.ub is not None else []))

    static_by_name = dict(zip(STATIC_NAMES, state.static))
    fallback_lb, fallback_ub = static_bounds(state.static, minimum_mm=10.0)
    fallback_lb_map = dict(zip(STATIC_NAMES, fallback_lb))
    fallback_ub_map = dict(zip(STATIC_NAMES, fallback_ub))

    names = ACTIVE_STATIC_NAMES + (
        "L7",
        "Target_Tx_mm", "Target_Ty_mm", "Target_Tz_mm", "Target_Scale",
    )
    x0: list[float] = []
    lb: list[float] = []
    ub: list[float] = []
    for name in ACTIVE_STATIC_NAMES:
        x0.append(float(static_by_name[name]))
        lb.append(max(10.0, float(lb_map.get(name, fallback_lb_map[name]))))
        ub.append(float(ub_map.get(name, fallback_ub_map[name])))

    # L6 已属于静态参数；L7 的周期均值等于二阶傅里叶常数项 C0。
    l7_fixed = float(np.mean(state.l7_values))
    x0.append(l7_fixed)
    lb.append(max(10.0, float(lb_map.get("L7_C0", l7_fixed - 80.0))))
    ub.append(float(ub_map.get("L7_C0", l7_fixed + 80.0)))
    pose_indices = (0, 1, 2, 6)
    fallback_pose_lb = (-150.0, -200.0, -150.0, 0.8)
    fallback_pose_ub = (150.0, 200.0, 150.0, 1.2)
    for local_index, pose_index in enumerate(pose_indices):
        pose_name = TARGET_POSE_NAMES[pose_index]
        x0.append(float(state.target_pose[pose_index]))
        lb.append(float(lb_map.get(pose_name, fallback_pose_lb[local_index])))
        ub.append(float(ub_map.get(pose_name, fallback_pose_ub[local_index])))
    lb_array = np.asarray(lb, dtype=float)
    ub_array = np.maximum(np.asarray(ub, dtype=float), lb_array + 1e-6)
    return _AnalysisContext(
        data=data,
        source=state,
        x0=np.clip(np.asarray(x0, dtype=float), lb_array, ub_array),
        lb=lb_array,
        ub=ub_array,
        names=names,
        fixed_target_rotation=state.target_pose[3:6].copy(),
    )


def _evaluate_analysis_vector(
    x: np.ndarray,
    context: _AnalysisContext,
) -> tuple[bool, dict[str, float]]:
    """计算一个 29 维候选，并返回 Tip 与 Wrist 各坐标轴的误差。"""

    values = np.clip(np.asarray(x, dtype=float), context.lb, context.ub)
    cursor = 0
    static_map = dict(zip(STATIC_NAMES, context.source.static))
    for name in ACTIVE_STATIC_NAMES:
        static_map[name] = float(values[cursor])
        cursor += 1
    l6_fixed = float(static_map["L6"])
    l7_fixed = float(values[cursor])
    cursor += 1
    static_map["L7"] = l7_fixed

    target_pose = context.source.target_pose.copy()
    target_pose[:3] = values[cursor:cursor + 3]
    target_pose[3:6] = context.fixed_target_rotation
    target_pose[6] = float(values[cursor + 3])
    static = np.array([static_map[name] for name in STATIC_NAMES])
    mot_curve = np.asarray(context.source.mot_curve, dtype=float).copy()
    _input_radius, _theta01, _theta02, b_curve, _mot_radius = motcurve_to_b_input(
        mot_curve, float(static_map["L1"])
    )
    center = target_pose_center(context.data)
    state = DesignState(
        static=static,
        mot_curve=mot_curve,
        b_curve=b_curve,
        l6_values=np.full(mot_curve.shape[0], l6_fixed),
        l7_values=np.full(mot_curve.shape[0], l7_fixed),
        target_tip=apply_target_pose(context.data.target_tip, target_pose, center),
        target_wrist=apply_target_pose(context.data.target_wrist, target_pose, center),
        target_pose=target_pose,
        b_fourier_coeff=context.source.b_fourier_coeff.copy(),
        l6_fourier_coeff=np.array([l6_fixed, 0.0, 0.0, 0.0, 0.0]),
        l7_fourier_coeff=np.array([l7_fixed, 0.0, 0.0, 0.0, 0.0]),
    )
    result, _metrics = evaluate_design_state(
        state,
        context.data,
        check_smooth=False,
        fixed_moving_lengths=True,
    )
    if result is None:
        return False, {"objective": 1e6}
    return True, coordinate_error_metrics(
        result.tip, result.wrist, state.target_tip, state.target_wrist
    )


def _admissible_boundary(
    context: _AnalysisContext,
    index: int,
    target: float,
    threshold: float,
    steps: int,
) -> float:
    """用二分法寻找单变量仍满足目标阈值的最远边界。"""

    accepted = float(context.x0[index])
    rejected = float(target)
    trial = context.x0.copy()
    trial[index] = rejected
    valid, metrics = _evaluate_analysis_vector(trial, context)
    if valid and metrics["objective"] <= threshold:
        return rejected
    for _ in range(max(1, int(steps))):
        middle = 0.5 * (accepted + rejected)
        trial[index] = middle
        valid, metrics = _evaluate_analysis_vector(trial, context)
        if valid and metrics["objective"] <= threshold:
            accepted = middle
        else:
            rejected = middle
    return accepted


def _attach_region_analysis_to_latest(
    output_dir: Path,
    checkpoint_path: Path,
    summary_path: Path,
    sensitivity_path: Path,
    narrow_path: Path,
) -> str | None:
    """若区域分析对应最新检查点，则把结果附加到同一版本化归档。"""

    latest_path = output_dir / "optimization_data" / "latest.json"
    if not latest_path.exists():
        return None
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    if latest.get("status") != "complete":
        return None
    run_dir = Path(latest["run_dir"])
    archived_checkpoint = run_dir / "final_checkpoint.npz"
    if not archived_checkpoint.exists() or not checkpoint_path.exists():
        return None
    if _sha256_file(archived_checkpoint) != _sha256_file(checkpoint_path):
        return None
    destination = run_dir / "supplemental_region_analysis"
    destination.mkdir(exist_ok=True)
    for source in (summary_path, sensitivity_path, narrow_path):
        shutil.copy2(source, destination / source.name)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["supplemental_region_analysis"] = {
        "attached_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint_sha256": _sha256_file(checkpoint_path),
        "directory": str(destination.resolve()),
        "files": sorted(path.name for path in destination.iterdir()),
    }
    manifest["files"] = sorted(path.name for path in run_dir.iterdir())
    _write_json(manifest_path, manifest)
    return str(destination.resolve())


def _run_region_analysis(
    checkpoint_path: Path,
    output_dir: Path,
    config: _RegionConfig,
) -> dict[str, Any]:
    """执行固定杆口径的灵敏度与窄区域分析。"""
    context = _build_analysis_context(checkpoint_path)
    valid, baseline = _evaluate_analysis_vector(context.x0, context)
    if not valid:
        raise RuntimeError("The current parameter combination is invalid with fixed L6 and cycle-mean L7.")

    metric_names = tuple(baseline.keys())
    raw_rows: list[dict[str, Any]] = []
    effects: dict[str, np.ndarray] = {
        metric: np.zeros(len(context.names), dtype=float) for metric in metric_names
    }
    # 中心有限差分：逐一扰动每个变量，其他 28 个变量保持不变。
    for index, name in enumerate(context.names):
        step = max(
            1e-6,
            config.delta_fraction * (context.ub[index] - context.lb[index]),
        )
        lower_x = context.x0.copy()
        upper_x = context.x0.copy()
        lower_x[index] = max(context.lb[index], context.x0[index] - step)
        upper_x[index] = min(context.ub[index], context.x0[index] + step)
        lower_valid, lower_metrics = _evaluate_analysis_vector(lower_x, context)
        upper_valid, upper_metrics = _evaluate_analysis_vector(upper_x, context)
        denominator = max(upper_x[index] - lower_x[index], 1e-12)
        for metric in metric_names:
            if lower_valid and upper_valid:
                derivative = (upper_metrics[metric] - lower_metrics[metric]) / denominator
            elif upper_valid:
                derivative = (upper_metrics[metric] - baseline[metric]) / max(upper_x[index] - context.x0[index], 1e-12)
            elif lower_valid:
                derivative = (baseline[metric] - lower_metrics[metric]) / max(context.x0[index] - lower_x[index], 1e-12)
            else:
                derivative = math.nan
            range_effect = abs(derivative) * (context.ub[index] - context.lb[index]) if np.isfinite(derivative) else math.nan
            effects[metric][index] = 0.0 if not np.isfinite(range_effect) else range_effect
            raw_rows.append({
                "variable": name,
                "metric": metric,
                "value": float(context.x0[index]),
                "lower": float(context.lb[index]),
                "upper": float(context.ub[index]),
                "step": float(step),
                "derivative_metric_per_unit": float(derivative),
                "full_range_effect": float(range_effect),
            })

    # 对每一个误差指标单独归一化，使同一指标下所有变量灵敏度之和为 1。
    row_lookup = {(row["variable"], row["metric"]): row for row in raw_rows}
    for metric in metric_names:
        total = float(np.sum(effects[metric]))
        for index, name in enumerate(context.names):
            row_lookup[(name, metric)]["normalized_sensitivity"] = (
                float(effects[metric][index] / total) if total > 0 else 0.0
            )

    # 窄区间定义：单变量变化后，目标值不超过基准值加给定容差。
    threshold = float(baseline["objective"] + config.narrow_tolerance_mm)
    narrow_rows = []
    for index, name in enumerate(context.names):
        lower = _admissible_boundary(
            context, index, context.lb[index], threshold, config.narrow_bisection_steps
        )
        upper = _admissible_boundary(
            context, index, context.ub[index], threshold, config.narrow_bisection_steps
        )
        narrow_rows.append({
            "variable": name,
            "value": float(context.x0[index]),
            "admissible_lower": float(lower),
            "admissible_upper": float(upper),
            "admissible_width": float(upper - lower),
            "full_width": float(context.ub[index] - context.lb[index]),
            "width_fraction": float((upper - lower) / max(context.ub[index] - context.lb[index], 1e-12)),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_path = output_dir / f"{config.prefix}_sensitivity.csv"
    with sensitivity_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)
    narrow_path = output_dir / f"{config.prefix}_narrow_intervals.csv"
    with narrow_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(narrow_rows[0].keys()))
        writer.writeheader()
        writer.writerows(narrow_rows)

    excluded = {
        "removed_static": ["H_finger", "Lf1", "L_CZ"],
        "target_rotation": ["Target_Rx_rad", "Target_Ry_rad", "Target_Rz_rad"],
        "Mot_input": list(MOT_POLAR_NAMES),
        "moving_length_periodic_terms": list(L7_FOURIER_NAMES[1:]),
    }
    summary = {
        "task": "region_analysis",
        "source_checkpoint": str(checkpoint_path),
        "rule": "Mot input and target rotation are frozen; L7 is one static length; target translation and scale remain active.",
        "active_variable_count": len(context.names),
        "active_variables": list(context.names),
        "excluded_variables": excluded,
        "fixed_values": {
            "Mot_input": "unchanged from checkpoint",
            "Target_Rx_Ry_Rz_rad": context.fixed_target_rotation,
            "L6_fixed_mm": float(context.x0[ACTIVE_STATIC_NAMES.index("L6")]),
            "L7_fixed_mm": float(context.x0[len(ACTIVE_STATIC_NAMES)]),
        },
        "baseline_metrics": baseline,
        "narrow_region_definition": {
            "criterion": "one-at-a-time admissible interval",
            "threshold": "baseline objective + tolerance",
            "tolerance_mm": config.narrow_tolerance_mm,
            "objective_threshold": threshold,
            "bisection_steps": config.narrow_bisection_steps,
        },
        "sensitivity_delta_fraction_of_full_range": config.delta_fraction,
        "sensitivity_csv": str(sensitivity_path),
        "narrow_intervals_csv": str(narrow_path),
    }
    summary_path = output_dir / f"{config.prefix}_summary.json"
    _write_json(summary_path, summary)
    attached = _attach_region_analysis_to_latest(
        output_dir,
        checkpoint_path,
        summary_path,
        sensitivity_path,
        narrow_path,
    )
    if attached is not None:
        summary["optimization_data_archive_attachment"] = attached
        _write_json(summary_path, summary)
        shutil.copy2(summary_path, Path(attached) / summary_path.name)
    return summary


# =============================================================================
# 5. 可恢复的长时多起点搜索
# =============================================================================


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """先写临时文件再替换，避免长时任务中断时留下半个 JSON。"""

    _write_json(path, payload)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def _run_rmse30_campaign(config: _CampaignConfig, output_dir: Path) -> dict[str, Any]:
    """在总预算内运行多个独立无锚点阶段，并保存可恢复研究状态。

    每个阶段都重新调用 ``_run_full_optimization``，因此起点始终来自原始
    fourbar 参数。阶段最优解仅用于最终比较，不会成为后续阶段的初始点。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / "campaign_state"
    phases_dir = output_dir / "phases"
    logs_dir = state_dir / "logs"
    state_dir.mkdir(parents=True, exist_ok=True)
    phases_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    progress_path = state_dir / "progress.json"
    heartbeat_path = state_dir / "heartbeat.json"
    findings_path = state_dir / "findings.jsonl"
    iterations_path = state_dir / "iteration_log.jsonl"
    directions_path = state_dir / "directions_tried.json"
    task_spec_path = state_dir / "task_spec.md"
    if not task_spec_path.exists():
        task_spec_path.write_text(
            f"# RMSE{config.target_rmse_mm:g} 长时优化任务\n\n"
            "- 模型：当前 `fourbar3d_python.py`，Wrist 为 L 点。\n"
            f"- 目标：Excel 固定初始事件后的 76 点联合等弧长严格同索引 RMSE；Tip < {config.target_rmse_mm:g} mm "
            f"且 Wrist < {config.target_rmse_mm:g} mm。\n"
            "- 起点：每个阶段均由原始 fourbar 参数独立启动，不读取历史最优解。\n"
            "- 方法：灵敏度 LA-MCTS 风格分区、带可行性回退的区域 CMA-ES、区域 SQP。\n"
            "- 拓扑：固定 L2/L5/L6/L7/L31，三阶 L3(t)/L8(t)，L32(t)=L3(t)-L31+2，C=Z，PO=QP。\n"
            "- 位姿：Tip/Wrist 使用共同三轴平移和共同三轴旋转，尺度固定为 1。\n"
            "- 卡顿：原始等时间机构帧接受停滞、相邻速度突变和 jerk 硬检查，并加入连续惩罚。\n"
            "- 数据：保存所有 fourbar 候选、区域、进化选择、精修和最优轨迹快照。\n",
            encoding="utf-8",
        )

    profiles = (
        {
            "label": "narrow_feasible_exploitation",
            "sigma0": 0.015,
            "popsize": 32,
            "generations_per_region": 8,
            "regions_per_round": 3,
            "sqp_maxiter": 28,
            "exploration": 0.14,
            "split_every_rounds": 2,
        },
        {
            "label": "medium_backtracked_exploration",
            "sigma0": 0.040,
            "popsize": 40,
            "generations_per_region": 7,
            "regions_per_round": 3,
            "sqp_maxiter": 18,
            "exploration": 0.22,
            "split_every_rounds": 2,
        },
        {
            "label": "wide_backtracked_exploration",
            "sigma0": 0.140,
            "popsize": 48,
            "generations_per_region": 6,
            "regions_per_round": 4,
            "sqp_maxiter": 10,
            "exploration": 0.30,
            "split_every_rounds": 3,
        },
        {
            "label": "fine_regional_polishing",
            "sigma0": 0.010,
            "popsize": 28,
            "generations_per_region": 10,
            "regions_per_round": 2,
            "sqp_maxiter": 45,
            "exploration": 0.10,
            "split_every_rounds": 2,
        },
    )
    _atomic_write_json(directions_path, {
        "no_anchor_between_phases": True,
        "selection_only_not_initialization": True,
        "profiles": profiles,
    })

    previous: dict[str, Any] = {}
    if progress_path.exists():
        try:
            previous = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    completed: list[dict[str, Any]] = list(previous.get("completed_phases", []))
    completed_ids = {int(row["phase_index"]) for row in completed if "phase_index" in row}
    best_row = previous.get("best_phase")
    campaign_start = time.monotonic()
    wall_start = datetime.now(timezone.utc)
    if previous.get("started_at_utc"):
        try:
            wall_start = datetime.fromisoformat(str(previous["started_at_utc"]))
        except ValueError:
            pass
    previously_used_seconds = sum(
        float(row.get("elapsed_seconds", 0.0))
        for row in completed
        if isinstance(row, Mapping)
    )
    search_seconds = max(
        60.0,
        3600.0 * float(config.hours) - 60.0 * float(config.reserve_minutes),
    )
    deadline = campaign_start + max(0.0, search_seconds - previously_used_seconds)
    stop_heartbeat = threading.Event()

    def heartbeat_worker() -> None:
        while not stop_heartbeat.wait(60.0):
            _atomic_write_json(heartbeat_path, {
                "status": "running",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds_current_process": time.monotonic() - campaign_start,
                "completed_phase_count": len(completed),
            })

    heartbeat = threading.Thread(target=heartbeat_worker, name="rmse30-heartbeat", daemon=True)
    heartbeat.start()
    status = "budget_exhausted"
    try:
        for phase_index in range(int(config.max_phases)):
            if phase_index in completed_ids:
                continue
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds < 5.0 * 60.0:
                break
            profile = dict(profiles[phase_index % len(profiles)])
            phase_minutes = min(float(config.phase_minutes), remaining_seconds / 60.0)
            phase_name = f"phase_{phase_index + 1:02d}_{profile['label']}"
            phase_dir = phases_dir / phase_name
            phase_config = _OptimizeConfig(
                minutes=phase_minutes,
                seed=int(config.seed) + phase_index,
                max_rounds=10000,
                popsize=int(profile["popsize"]),
                generations_per_region=int(profile["generations_per_region"]),
                regions_per_round=int(profile["regions_per_round"]),
                sigma0=float(profile["sigma0"]),
                max_depth=10,
                split_every_rounds=int(profile["split_every_rounds"]),
                sqp_maxiter=int(profile["sqp_maxiter"]),
                sqp_ftol=1e-8,
                exploration=float(profile["exploration"]),
                prefix=phase_name,
                objective_mode="initialized_equal_arc_bottleneck_p64_with_stutter",
                target_rmse_mm=float(config.target_rmse_mm),
            )
            phase_started = datetime.now(timezone.utc)
            _atomic_write_json(progress_path, {
                "status": "running",
                "current_phase": phase_index,
                "current_phase_name": phase_name,
                "current_profile": profile,
                "phase_started_at_utc": phase_started.isoformat(),
                "completed_phases": completed,
                "best_phase": best_row,
                "success": bool(best_row and best_row.get("success", False)),
            })
            try:
                summary = _run_full_optimization(phase_config, phase_dir)
                metrics = summary["metrics"]
                row = {
                    "phase_index": phase_index,
                    "phase_name": phase_name,
                    "profile": profile,
                    "seed": phase_config.seed,
                    "minutes_requested": phase_minutes,
                    "started_at_utc": phase_started.isoformat(),
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "tip_rmse_mm": float(metrics["tip"]),
                    "wrist_rmse_mm": float(metrics["wrist"]),
                    "combined_rmse_mm": float(metrics["combined"]),
                    "tip_peak_mm": float(metrics["tip_peak"]),
                    "wrist_peak_mm": float(metrics["wrist_peak"]),
                    "objective": float(summary["objective"]),
                    "evaluations": int(summary["evaluations"]),
                    "elapsed_seconds": float(summary["elapsed_seconds"]),
                    "success": bool(summary["success"]),
                    "checkpoint": summary["checkpoint"],
                    "optimization_data_archive": summary.get("optimization_data_archive"),
                    "summary_path": str((phase_dir / f"{phase_name}_summary.json").resolve()),
                }
                completed.append(row)
                completed_ids.add(phase_index)
                _append_jsonl(iterations_path, row)
                if best_row is None or (
                    max(row["tip_rmse_mm"], row["wrist_rmse_mm"]),
                    row["tip_rmse_mm"] + row["wrist_rmse_mm"],
                ) < (
                    max(best_row["tip_rmse_mm"], best_row["wrist_rmse_mm"]),
                    best_row["tip_rmse_mm"] + best_row["wrist_rmse_mm"],
                ):
                    best_row = row
                    shutil.copy2(Path(row["checkpoint"]), output_dir / f"{config.prefix}_best_checkpoint.npz")
                    source_variables = phase_dir / f"{phase_name}_variables.csv"
                    if source_variables.exists():
                        shutil.copy2(source_variables, output_dir / f"{config.prefix}_best_variables.csv")
                    _append_jsonl(findings_path, {
                        "event": "new_campaign_best",
                        **row,
                    })
                if row["success"]:
                    status = "target_reached"
                    break
            except Exception as error:
                failure = {
                    "phase_index": phase_index,
                    "phase_name": phase_name,
                    "profile": profile,
                    "seed": phase_config.seed,
                    "started_at_utc": phase_started.isoformat(),
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
                completed.append(failure)
                completed_ids.add(phase_index)
                _append_jsonl(iterations_path, failure)
            _atomic_write_json(progress_path, {
                "status": "running",
                "completed_phases": completed,
                "best_phase": best_row,
                "success": bool(best_row and best_row.get("success", False)),
            })
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=5.0)

    if best_row is None:
        status = "failed_without_valid_phase"
    summary = {
        "task": "campaign",
        "status": status,
        "objective": (
            f"Tip and Wrist strict initialized equal-arc RMSE both below "
            f"{config.target_rmse_mm:g} mm"
        ),
        "target_rmse_mm": float(config.target_rmse_mm),
        "no_anchor_between_phases": True,
        "phase_results_only_used_for_selection": True,
        "budget_hours": float(config.hours),
        "reserved_postprocessing_minutes": float(config.reserve_minutes),
        "started_at_utc": wall_start.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds_current_process": time.monotonic() - campaign_start,
        "elapsed_seconds_total_phases": previously_used_seconds + sum(
            float(row.get("elapsed_seconds", 0.0))
            for row in completed
            if isinstance(row, Mapping)
            and row not in previous.get("completed_phases", [])
        ),
        "completed_phase_count": len(completed),
        "completed_phases": completed,
        "best_phase": best_row,
        "success": bool(best_row and best_row.get("success", False)),
        "best_checkpoint": (
            str((output_dir / f"{config.prefix}_best_checkpoint.npz").resolve())
            if best_row is not None
            else None
        ),
        "state_directory": str(state_dir.resolve()),
    }
    summary_path = output_dir / f"{config.prefix}_campaign_summary.json"
    _write_json(summary_path, summary)
    _atomic_write_json(progress_path, {**summary, "status": status})
    _atomic_write_json(heartbeat_path, {
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_phase_count": len(completed),
    })
    return summary


# =============================================================================
# 6. 唯一公共入口和命令行入口
# =============================================================================


def run_fourbar_optimization(
    task: str = "optimize",
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    **options: Any,
) -> dict[str, Any]:
    """唯一公共入口。

    Parameters
    ----------
    task:
        "optimize"、"campaign" 或 "region_analysis"。
    checkpoint_path:
        区域分析所用的当时参数组合检查点；完整优化不读取该文件。
    output_dir:
        默认写入 当前文件/output。
    options:
        对应 _OptimizeConfig 或 _RegionConfig 中的参数。
    """
    destination = Path(output_dir) if output_dir is not None else CURRENT_DIR / "output"
    if task == "optimize":
        allowed = _OptimizeConfig.__dataclass_fields__.keys()
        config = _OptimizeConfig(**{key: value for key, value in options.items() if key in allowed})
        return _run_full_optimization(config, destination)
    if task == "campaign":
        allowed = _CampaignConfig.__dataclass_fields__.keys()
        config = _CampaignConfig(**{key: value for key, value in options.items() if key in allowed})
        return _run_rmse30_campaign(config, destination)
    if task == "region_analysis":
        if checkpoint_path is None:
            checkpoint_path = CURRENT_DIR / "output" / "strict_feasible_from_initial_checkpoint.npz"
        allowed = _RegionConfig.__dataclass_fields__.keys()
        config = _RegionConfig(**{key: value for key, value in options.items() if key in allowed})
        return _run_region_analysis(Path(checkpoint_path), destination, config)
    raise ValueError("task must be 'optimize', 'campaign', or 'region_analysis'.")


def _main() -> None:
    """把命令行参数转换为唯一公共函数的参数。"""

    parser = argparse.ArgumentParser(description="Fourbar unified optimization and regional analysis")
    parser.add_argument(
        "--task", choices=("optimize", "campaign", "region_analysis"), default="optimize"
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--input-contract",
        dest="input_contract_path",
        default=str(
            CURRENT_DIR / "input" / "fourbar_initialized_equal_arc_optimization_input_v2.json"
        ),
        help="每次正式优化必须先校验的模型、目标、设计域和求解器输入合同。",
    )
    parser.add_argument(
        "--target-initialized-csv",
        dest="target_initialized_csv_path",
        default=str(
            CURRENT_DIR / "input" / "Length_normalized_target_initialized_equal_arc_76_mm.csv"
        ),
    )
    parser.add_argument(
        "--target-initialization-metadata",
        dest="target_initialization_metadata_path",
        default=str(
            CURRENT_DIR / "input" / "Length_normalized_target_initialized_equal_arc_76_metadata.json"
        ),
    )
    parser.add_argument(
        "--target-tip-txt",
        dest="target_tip_txt_path",
        default="",
        help="旧目标兼容入口；初始化等弧长模式下必须留空。",
    )
    parser.add_argument(
        "--target-wrist-txt",
        dest="target_wrist_txt_path",
        default="",
        help="旧目标兼容入口；初始化等弧长模式下必须留空。",
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=None,
        help=(
            "可选的临时墙钟时限；默认不限制运行时间，直到达到目标、最大评估数、"
            "最大轮次或人工停止。"
        ),
    )
    parser.add_argument(
        "--start-checkpoint",
        dest="start_checkpoint_path",
        default="",
        help="仅续跑当前正式任务自己的检查点；空值表示从原始 Fourbar 参数无锚点启动。",
    )
    parser.add_argument("--max-rounds", type=int, default=10_000)
    parser.add_argument("--max-evaluations", type=int, default=100_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--parallel-backend",
        choices=("thread", "process"),
        default="process",
        help="候选机构评估的并行后端；process 可在 Windows 上绕开 Python GIL。",
    )
    parser.add_argument("--popsize", type=int, default=192)
    parser.add_argument("--generations-per-region", type=int, default=10)
    parser.add_argument("--regions-per-round", type=int, default=12)
    parser.add_argument("--sqp-maxiter", type=int, default=600)
    parser.add_argument("--sqp-ftol", type=float, default=1e-7)
    parser.add_argument("--sqp-trigger-rmse-mm", type=float, default=140.0)
    parser.add_argument("--sqp-interval-rounds", type=int, default=1)
    parser.add_argument("--sqp-active-dimensions", type=int, default=54)
    parser.add_argument("--sqp-fd-step", type=float, default=5.0e-4)
    parser.add_argument("--sqp-trust-radius", type=float, default=0.035)
    parser.add_argument(
        "--refinement-method",
        choices=("pso", "slsqp"),
        default="slsqp",
    )
    parser.add_argument("--pso-particles", type=int, default=48)
    parser.add_argument("--pso-iterations", type=int, default=24)
    parser.add_argument("--pso-interval-rounds", type=int, default=4)
    parser.add_argument("--pso-active-dimensions", type=int, default=24)
    parser.add_argument("--pso-fd-step", type=float, default=2e-3)
    parser.add_argument("--pso-trust-radius", type=float, default=0.06)
    parser.add_argument("--pso-inertia-start", type=float, default=0.78)
    parser.add_argument("--pso-inertia-end", type=float, default=0.38)
    parser.add_argument("--pso-cognitive", type=float, default=1.55)
    parser.add_argument("--pso-social", type=float, default=1.75)
    parser.add_argument("--pso-velocity-fraction", type=float, default=0.20)
    parser.add_argument("--pso-stall-iterations", type=int, default=7)
    parser.add_argument("--pso-restart-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260839)
    parser.add_argument("--sigma0", type=float, default=0.040)
    parser.add_argument("--exploration", type=float, default=0.55)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--split-every-rounds", type=int, default=1)
    parser.add_argument("--splits-per-event", type=int, default=12)
    parser.add_argument("--cma-elite-fraction", type=float, default=0.35)
    parser.add_argument("--cma-covariance-learning-rate", type=float, default=0.18)
    parser.add_argument("--cma-variance-floor", type=float, default=1e-8)
    parser.add_argument(
        "--legacy-cma-covariance",
        dest="cma_normalize_covariance_by_sigma",
        action="store_false",
        default=True,
        help="使用旧的未除以 sigma 的协方差尺度，仅用于对照实验。",
    )
    parser.add_argument("--cma-sigma-min", type=float, default=0.00025)
    parser.add_argument("--cma-sigma-max", type=float, default=0.18)
    parser.add_argument("--cma-sigma-no-valid-factor", type=float, default=0.65)
    parser.add_argument("--cma-sigma-expand-factor", type=float, default=1.12)
    parser.add_argument("--cma-sigma-default-factor", type=float, default=1.00)
    parser.add_argument("--cma-low-valid-fraction", type=float, default=0.12)
    parser.add_argument("--cma-high-valid-fraction", type=float, default=0.45)
    parser.add_argument(
        "--cma-restart-mode",
        choices=("outer_round_baseline", "per_region_generation"),
        default="outer_round_baseline",
    )
    parser.add_argument("--cma-stagnation-generations", type=int, default=12)
    parser.add_argument("--cma-restart-sigma-factor", type=float, default=2.0)
    parser.add_argument("--cma-outer-stagnation-rounds", type=int, default=3)
    parser.add_argument("--cma-outer-min-improvement-mm", type=float, default=0.15)
    parser.add_argument("--cma-outer-min-improvement-fraction", type=float, default=0.0015)
    parser.add_argument("--cma-boundary-trigger-fraction", type=float, default=0.03)
    parser.add_argument("--cma-boundary-stagnation-rounds", type=int, default=2)
    parser.add_argument("--cma-boundary-restart-center-blend", type=float, default=0.40)
    parser.add_argument("--cma-restart-growth-factor", type=float, default=1.35)
    parser.add_argument("--cma-restart-factor-max", type=float, default=4.0)
    parser.add_argument("--cma-local-block-fraction", type=float, default=0.18)
    parser.add_argument("--cma-global-injection-fraction", type=float, default=0.15)
    parser.add_argument("--cma-boundary-injection-fraction", type=float, default=0.08)
    parser.add_argument(
        "--cma-use-evolution-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用标准 CMA-ES 协方差路径和步长路径。",
    )
    parser.add_argument("--initial-pool-size", type=int, default=1024)
    parser.add_argument("--initial-pool-max-proposals", type=int, default=24000)
    parser.add_argument("--initial-pool-min-distance", type=float, default=0.006)
    parser.add_argument("--minimum-samples-before-split", type=int, default=240)
    parser.add_argument("--minimum-leaf-evaluations", type=int, default=160)
    parser.add_argument("--sensitivity-min-region-width", type=float, default=0.018)
    parser.add_argument("--sensitivity-probe-step", type=float, default=0.020)
    parser.add_argument("--sensitivity-probe-width-fraction", type=float, default=0.35)
    parser.add_argument("--split-guard-fraction", type=float, default=0.06)
    parser.add_argument("--prefix", default="fourbar_baseline_matched_coldstart")
    parser.add_argument(
        "--objective-mode",
        choices=(
            "initialized_equal_arc_bottleneck_p64_with_stutter",
            "equal_weight_rmse",
            "balanced_rmse_target",
            "combined_peak",
            "rmse_bottleneck_p32",
            "rmse_bottleneck_p64",
            "smooth_bottleneck_p32",
        ),
        default="initialized_equal_arc_bottleneck_p64_with_stutter",
    )
    parser.add_argument("--target-rmse-mm", type=float, default=25.0)
    parser.add_argument(
        "--continue-after-target",
        action="store_true",
        help="Continue until the wall-clock deadline after first meeting the RMSE target.",
    )
    parser.add_argument(
        "--target-pose-mode",
        choices=(
            "full", "rigid6", "ty_ry", "ty_ry_rz", "constrained6",
            "decoupled_constrained11", "decoupled_fixed_rz_scale8",
            "decoupled_tip_ry_scale8", "decoupled_both_ry_scale9",
            "decoupled_both_ry_rz_scale11",
            "decoupled_full_rotation_scale13",
            "decoupled_fixed_wrist_rotation_scale10",
            "shared_fixed_rotation_translation3",
            "shared_fixed_rotation_scale4",
            "shared_limited_rotation_scale7",
            "shared_positive_rz_scale7",
            "shared_negative_rz_expanded_scale7",
            "shared_rotation_translation6",
        ),
        default="shared_rotation_translation6",
        help=(
            "目标位姿：Tip/Wrist 共用 Tx/Ty/Tz/Rx/Ry/Rz，尺度固定为 1。"
        ),
    )
    parser.add_argument(
        "--single-stage",
        dest="three_stage_schedule",
        action="store_false",
        default=False,
        help="保持单阶段参数；该选项为旧命令兼容，当前默认已经是单阶段。",
    )
    parser.add_argument(
        "--search-scope",
        choices=(
            "all",
            "wrist_upstream",
            "wrist_core",
            "tip_downstream",
            "sensitivity_wrist",
            "sensitivity_tip",
            "sensitivity_joint",
        ),
        default="all",
        help="限制 CMA-ES 当前允许扰动的物理变量子空间；不删除变量或改变边界。",
    )
    parser.add_argument("--stage1-advance-rmse-mm", type=float, default=105.0)
    parser.add_argument("--stage2-advance-rmse-mm", type=float, default=110.0)
    parser.add_argument(
        "--stage3-block-cycle",
        action="store_true",
        help="联合阶段循环优化腕部上游、Tip 下游和全部变量。",
    )
    parser.add_argument(
        "--stage3-broad-search",
        action="store_true",
        help="联合阶段继续使用调用方给定的广域种群和分区参数。",
    )
    parser.add_argument(
        "--stage3-deep-consolidation",
        action="store_true",
        help="双曲线进入深层阈值后合并叶区并集中精修。",
    )
    parser.add_argument("--stage3-deep-threshold-mm", type=float, default=35.0)
    parser.add_argument("--stage3-deep-sigma0", type=float, default=0.008)
    parser.add_argument(
        "--periodic-length-mode",
        choices=(
            "l7_periodic", "l6_periodic", "l7_l8_periodic",
            "l7_l8_periodic3", "l31_l8_periodic3_l7_fixed",
            "l31_l6_periodic3_l7_l8_fixed",
            "l2_l31_l6_periodic3_l7_l8_fixed",
            "l3_l8_periodic3_l5_fixed_l32_derived",
            "l3_l8_l2_periodic3_l32_derived",
            "l3_l8_l5_periodic3_l32_derived",
            "l3_l8_l6_periodic3_l32_derived",
            "l3_l8_l7_periodic3_l32_derived",
            "l3_l8_l12_periodic3_l5_fixed_l32_derived",
            "l3_l5_l8_periodic3_l32_derived",
            "l3_l5_l8_periodic3_l32_fixed",
            "l3_l7_periodic",
            "l7_periodic_l3_fixed",
        ),
        default=_CURRENT_PERIODIC_MODE,
        help=(
            "当前正式模式固定 L2/L5/L6/L7/L31，逐帧派生 L32(t)=L3(t)-L31+2，"
            "仅三阶 L3(t)/L8(t)，C=Z；"
            "其余选项仅保留用于读取历史结果。"
        ),
    )
    parser.add_argument(
        "--b-curve-mode",
        choices=("constant_x", "fourier_x", "fourier_z3", "fourier_z3_c0", "fourier_xyz3"),
        default="fourier_z3_c0",
        help=(
            "B 点曲线模式；当前设置保持 59.9 基准的 Bx 常数、By 二阶、"
            "Bz 三阶，并仅新增 Bz 常数项 C0z。"
        ),
    )
    parser.add_argument(
        "--static-bound-mode",
        choices=(
            "current", "rmse25_union", "user_table",
            "adaptive_expanded_20260727",
            "adaptive_expanded_v2_20260727",
            "adaptive_expanded_v3_20260727",
            "adaptive_expanded_v4_20260728",
            "adaptive_expanded_v5_20260728",
            "adaptive_expanded_v6_20260728",
            "broad_all54_20260802",
        ),
        default="broad_all54_20260802",
        help="静态几何边界：当前范围、RMSE25 并集，或用户给出的 8h 表。",
    )
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--reserve-minutes", type=float, default=30.0)
    parser.add_argument("--phase-minutes", type=float, default=27.0)
    parser.add_argument("--max-phases", type=int, default=20)
    parser.add_argument("--narrow-tolerance-mm", type=float, default=5.0)
    parser.add_argument("--narrow-bisection-steps", type=int, default=16)
    args = parser.parse_args()
    options = vars(args).copy()
    task = options.pop("task")
    checkpoint = options.pop("checkpoint")
    output_dir = options.pop("output_dir")
    result = run_fourbar_optimization(
        task=task,
        checkpoint_path=checkpoint,
        output_dir=output_dir,
        **options,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    _main()
