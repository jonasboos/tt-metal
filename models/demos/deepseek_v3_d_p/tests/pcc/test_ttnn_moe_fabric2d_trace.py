# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TtMoe with moe_transport="fabric2d": a traced forward against eager ones, bit for bit.

Eager, TtMoe runs dispatch_fabric2d, the routed expert, then a standalone combine_fabric2d. Traced, it runs the
routed expert and combine_fabric2d as one overlapped program, or with overlap_routed_expert_with_combine=False the
same standalone ops captured in the trace. Every one of them must give the eager output exactly, and so must eight
eager forwards launched back to back with no host sync between them, which is where a chip running a launch ahead
of its neighbour would corrupt a forwarded token. Accuracy against torch is test_ttnn_moe.py's job.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import moe_fabric_payload, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.pcc.test_ttnn_moe import run_model
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombine2dModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatch2dModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

_TRACE_REGION_SIZE = 16 * 1024 * 1024
_EAGER_LAUNCHES = 8
_REPLAYS = 2

# (variant, model config, dispatch buffer capacity factor): the fabric2d rows of test_ttnn_moe.py.
_CASES = [
    pytest.param("deepseek_v3_d_p", DeepSeekV3Config, 8, id="deepseek_v3"),
    pytest.param("deepseek_v3_d_p", GLM53Config, 8, id="glm"),
    pytest.param("kimi_k2_7", KimiK27Config, 5, id="kimi_k2_7"),
]


def _host(tensor):
    return [ttnn.to_torch(device_tensor) for device_tensor in ttnn.get_device_tensors(tensor)]


def _assert_same(got, want, what):
    differ = [i for i, (g, w) in enumerate(zip(got, want)) if not torch.equal(g, w)]
    assert not differ, f"{what} differs from the first eager forward on devices {differ}"


@pytest.mark.parametrize("overlap_routed_expert_with_combine", [None, False], ids=["overlap_auto", "overlap_off"])
@pytest.mark.parametrize("variant, model_cfg, dispatch_buffer_capacity_factor", _CASES, indirect=["variant"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=moe_fabric_payload(KimiK27Config, "fabric2d"), trace_region_size=_TRACE_REGION_SIZE
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.timeout(1400)
def test_moe_fabric2d_traced_matches_eager(
    variant,
    model_cfg,
    dispatch_buffer_capacity_factor,
    overlap_routed_expert_with_combine,
    config_only,
    mesh_device,
    device_params,
    num_links,
    request,
):
    # One mesh for every model: the 7168-wide token payload covers GLM's 6144 too.
    assert moe_fabric_payload(model_cfg, "fabric2d") <= moe_fabric_payload(KimiK27Config, "fabric2d")

    def measure(forward, tt_moe, tt_x):
        # A plumbing slip that left the transport or the overlap at its default would still match itself.
        assert type(tt_moe.dispatch_module) is TtDispatch2dModule, type(tt_moe.dispatch_module)
        assert type(tt_moe.combine_module) is TtCombine2dModule, type(tt_moe.combine_module)
        expected_overlap = overlap_routed_expert_with_combine is not False
        assert tt_moe.overlap_routed_expert_with_combine == expected_overlap

        # forward() frees its input, hence a clone per launch. No sync until all eight are queued.
        eager = [forward(ttnn.clone(tt_x))[0] for _ in range(_EAGER_LAUNCHES)]
        ttnn.synchronize_device(mesh_device)
        want = _host(eager[0])
        for launch, out in enumerate(eager[1:], start=1):
            _assert_same(_host(out), want, f"eager launch {launch} of {_EAGER_LAUNCHES} without a host sync")
        for out in eager:
            ttnn.deallocate(out)
        logger.info(f"{_EAGER_LAUNCHES} eager forwards without a host sync are identical")

        controller = SubDeviceTraceController(mesh_device)
        copy_tid = None
        capturing = False
        tt_moe.set_trace_controller(controller)
        try:
            # A capture cannot JIT-build kernels, so the overlapped program compiles in this pass.
            compile_out, _ = forward(ttnn.clone(tt_x))
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(compile_out)

            # forward() frees its input mid-capture and the trace reuses that address, so a one-op trace
            # refreshes it from tt_x before every replay.
            copy_tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            x = ttnn.clone(tt_x)
            ttnn.end_trace_capture(mesh_device, copy_tid, cq_id=0)

            controller.begin_capture()
            capturing = True
            result = forward(x)
            controller.end_capture()
            capturing = False

            for replay in range(_REPLAYS):
                ttnn.execute_trace(mesh_device, copy_tid, cq_id=0, blocking=False)
                controller.replay()  # ends in synchronize_device
                _assert_same(_host(result[0]), want, f"traced replay {replay}")
            logger.info(f"{_REPLAYS} traced replays match the eager forward")
            return result
        finally:
            if capturing:  # close the open segment so release() runs and the real error surfaces
                controller.end_capture()
            controller.release()
            if copy_tid is not None:
                ttnn.release_trace(mesh_device, copy_tid)
            tt_moe.set_trace_controller(None)
            tt_moe.release_sub_device_manager()

    run_model(
        variant,
        config_only,
        mesh_device,
        device_params,
        PREFILL_CHUNK_TOKENS_PER_CHIP,
        model_cfg.EMB_SIZE,
        model_cfg.MOE_INTERMEDIATE_SIZE,
        model_cfg.NUM_ROUTED_EXPERTS,
        model_cfg.NUM_EXPERTS_PER_TOKEN,
        dispatch_buffer_capacity_factor,
        False,  # run_pcc_check: test_ttnn_moe.py owns accuracy
        num_links,
        per_axis_topology(device_params["fabric_config"]),
        GateComputeMode.DEVICE_FP32,
        request,
        measure=measure,
        moe_transport="fabric2d",
        overlap_routed_expert_with_combine=overlap_routed_expert_with_combine,
    )
