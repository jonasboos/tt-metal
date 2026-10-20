# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""What dispatch_fabric2d and combine_fabric2d need from the mesh, the fabric and the model.

Shared by TtDispatch2dModule, TtCombine2dModule and TtMoe. The fabric payload they need is sized by
init_helpers.moe_fabric_payload_size, which MLAPrefillAdapter.fabric_payload_size calls before the mesh opens.
"""

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    CMB_FABRIC2D_ROUTING_INFO_BYTES,
    combine_overlap_payload_size,
    dispatch_fabric2d_packet_bytes,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

# Values of TtMoe's moe_transport: dispatch and combine (TtDispatchModule, TtCombineModule), or dispatch_fabric2d
# and combine_fabric2d (TtDispatch2dModule, TtCombine2dModule).
MOE_TRANSPORTS = ("direct", "fabric2d")
# Largest top-k dispatch_fabric2d accepts (validated in dispatch_fabric2d_device_operation.cpp).
# combine_fabric2d has no such limit.
DISPATCH_FABRIC2D_MAX_TOPK = 16


def check_fabric2d_setup(
    mesh_device, cluster_axis: int, emb_dim: int, op: str, num_experts_per_tok: int | None = None
) -> None:
    """Raise ValueError if `op` ("dispatch" or "combine", for dispatch_fabric2d or combine_fabric2d) cannot run on
    this mesh and fabric.

    Both ops send bf16 tokens one hop at a time around a ring on `cluster_axis`, and both check the ring size and
    the packet size when they launch. dispatch_fabric2d sends a whole token in one packet; combine_fabric2d splits
    one that does not fit into two halves. Whether the fabric came up with the ring wired needs a device tensor,
    so check_axis_wired_into_ring does that separately. `num_experts_per_tok` is checked for dispatch only.
    """
    if op not in ("dispatch", "combine"):
        raise ValueError(f"op must be 'dispatch' or 'combine', got {op!r}")
    if op == "dispatch" and num_experts_per_tok is not None and num_experts_per_tok > DISPATCH_FABRIC2D_MAX_TOPK:
        raise ValueError(
            f"dispatch_fabric2d supports top-k up to {DISPATCH_FABRIC2D_MAX_TOPK}, "
            f"got num_experts_per_tok={num_experts_per_tok}"
        )
    extent = mesh_device.shape[cluster_axis]
    # Even, because the ops split the traffic to the opposite chip across both directions.
    if extent < 4 or extent % 2 != 0:
        raise ValueError(
            f"fabric2d dispatch/combine need an even ring of at least 4 chips on cluster_axis {cluster_axis}, "
            f"got {extent}"
        )
    fabric_config = ttnn.get_fabric_config()
    # FABRIC_1D_RING wraps axis 0 too, but the ops build 2D fabric routes, so it does not work.
    wraps = (
        fabric_config != ttnn.FabricConfig.FABRIC_1D_RING
        and per_axis_topology(fabric_config)[cluster_axis] == ttnn.Topology.Ring
    )
    if not wraps:
        raise ValueError(
            f"fabric2d dispatch/combine need a 2D fabric that wraps cluster_axis {cluster_axis} into a ring, "
            f"got {fabric_config}"
        )
    if op == "dispatch":
        packet_bytes = dispatch_fabric2d_packet_bytes(emb_dim)
        what = f"dispatch_fabric2d sends a {emb_dim}-wide bf16 token"
    else:
        packet_bytes = combine_overlap_payload_size(emb_dim)
        what = f"combine_fabric2d sends a {emb_dim}-wide bf16 token as two halves at most"
    max_payload = ttnn.get_tt_fabric_max_payload_size_bytes()
    if max_payload < packet_bytes:
        raise ValueError(
            f"{what} plus a {CMB_FABRIC2D_ROUTING_INFO_BYTES} B routing tail in one packet ({packet_bytes} B), but "
            f"the fabric max payload is {max_payload} B. Open the mesh with "
            f"init_helpers.moe_fabric_payload_size(model_cfg, 'fabric2d'), or raise it with "
            f"create_fabric_router_config(max_payload_size=...) in device_params."
        )


def check_axis_wired_into_ring(tensor, cluster_axis: int) -> None:
    """Raise ValueError unless the fabric wired `cluster_axis` of the mesh `tensor` lives on into a ring.

    The fabric config alone does not say: auto-discovery falls back TORUS_XY -> TORUS_Y -> TORUS_X -> MESH behind a
    log warning when a wrap link is missing, and get_fabric_config() still reports the request. get_usable_topology
    keeps Ring exactly when the closing link of every line along the axis is wired, the check dispatch_fabric2d
    makes when it launches. combine_fabric2d does not check, and runs on such an axis unchecked.
    """
    if ttnn.get_usable_topology(tensor, ttnn.Topology.Ring, cluster_axis) != ttnn.Topology.Ring:
        raise ValueError(
            f"fabric2d dispatch/combine need cluster_axis {cluster_axis} wired into a ring, but the fabric "
            f"{ttnn.get_fabric_config()} came up without a wrap link on that axis; check the wrap cabling"
        )
