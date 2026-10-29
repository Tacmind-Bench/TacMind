"""Shared physics composition for the legacy Panda's empty flange frame."""

from pxr import Gf, UsdPhysics


def disable_stock_franka_fingers(stage):
    """Disable inherited Panda fingers before mounting a gripper."""
    for path in ("/panda/panda_hand/panda_finger_joint1",
                 "/panda/panda_hand/panda_finger_joint2",
                 "/panda/panda_leftfinger",
                 "/panda/panda_rightfinger"):
        stage.OverridePrim(path).SetActive(False)


def merge_massless_flange(stage):
    """Retain link8 as a frame and connect the physical hand directly to link7."""
    # Reuse the fixed-joint frames from the existing Newton composition,
    # matching Isaac's multiphysics Franka without inventing flange mass.
    link8 = stage.GetPrimAtPath("/panda/panda_link8")
    link8.RemoveAPI(UsdPhysics.RigidBodyAPI)
    link8.RemoveAPI(UsdPhysics.MassAPI)
    stage.GetPrimAtPath("/panda/panda_link7/panda_joint8").SetActive(False)
    hand_joint = UsdPhysics.Joint(stage.GetPrimAtPath("/panda/panda_link8/panda_hand_joint"))
    hand_joint.GetBody0Rel().SetTargets(["/panda/panda_link7"])
    hand_joint.GetLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0.10699999))
    hand_joint.GetLocalPos1Attr().Set(Gf.Vec3f(-3.809022e-8, 3.8090203e-8, -5.9604645e-8))
    hand_joint.GetLocalRot0Attr().Set(Gf.Quatf(0.70710677, 0, 0.70710677, 0))
    hand_joint.GetLocalRot1Attr().Set(Gf.Quatf(0.6532815, -0.2705981, 0.6532814, 0.27059814))
