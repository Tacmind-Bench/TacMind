"""Read authored asset geometry for collection and benchmark scoring."""


from functools import lru_cache

from benchmark.catalog import object_asset


@lru_cache(maxsize=128)
def _asset_geometry(asset_type: str):
    """Cache local asset bounds and the authored world-bound centroid."""
    from pxr import Usd, UsdGeom

    asset_path = object_asset(asset_type)
    stage = Usd.Stage.Open(str(asset_path))
    prim = stage.GetDefaultPrim() if stage else None
    if not prim:
        raise ValueError(f"USD asset has no default prim: {asset_path}")
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    local = bounds.ComputeLocalBound(prim).ComputeAlignedRange()
    centroid = bounds.ComputeWorldBound(prim).ComputeCentroid()
    return (
        tuple(float(value) for value in local.GetMin()),
        tuple(float(value) for value in local.GetMax()),
        tuple(float(value) for value in centroid),
    )



def _collision_height(env, name: str, *, highest: bool) -> float:
    from pxr import Usd, UsdGeom, UsdPhysics

    prim = env.sim.stage.GetPrimAtPath(f"/World/envs/env_0/{name}")
    if not prim:
        raise ValueError(f"Missing support object: {name}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    heights = []
    for collider in Usd.PrimRange(prim):
        if (collider.HasAPI(UsdPhysics.CollisionAPI)
                and UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get()):
            bounds = cache.ComputeWorldBound(collider).ComputeAlignedRange()
            heights.append(float((bounds.GetMax() if highest else bounds.GetMin())[2]))
    if not heights:
        raise ValueError(f"No enabled collision geometry on {name}")
    return max(heights) if highest else min(heights)



def _object_center_local(env, task: dict, local_offset=None):
    import torch

    object_name = task["obj_of_interest"][0]
    spec = task["objects"][object_name]
    if local_offset is None:
        minimum, maximum, _ = _asset_geometry(spec["type"])
        local_offset = tuple((low + high) * 0.5 for low, high in zip(minimum, maximum))
    dtype = env.scene["object"].data.root_pos_w.dtype
    return torch.as_tensor(
        local_offset, device=env.device, dtype=dtype,
    ) * torch.as_tensor(spec["scale"], device=env.device, dtype=dtype)



def _object_bottom_offset(env, task: dict):
    """Return the scaled asset bounds' lowest world-z offset from its root."""
    from itertools import product
    import torch
    from curobo.types import Pose

    object_name = task["obj_of_interest"][0]
    spec = task["objects"][object_name]
    minimum, maximum, _ = _asset_geometry(spec["type"])
    data = env.scene["object"].data
    scale = torch.as_tensor(spec["scale"], device=env.device, dtype=data.root_pos_w.dtype)
    rotation = Pose(torch.zeros_like(data.root_pos_w), data.root_quat_w)
    if spec["type"] == "ball_class_golf":
        center = _object_center_local(env, task)
        center_offset = rotation.transform_points(center.expand_as(data.root_pos_w))
        radius = (maximum[2] - minimum[2]) * scale[2] * 0.5
        return center_offset[:, 2] - radius
    corners = torch.tensor(list(product(*zip(minimum, maximum))),
                           device=env.device, dtype=scale.dtype) * scale
    points = corners.unsqueeze(0).expand(data.root_pos_w.shape[0], -1, -1).contiguous()
    return rotation.batch_transform_points(points)[:, :, 2].amin(dim=1)

