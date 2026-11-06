"""Built-in rule-based poses.

Coordinates are image pixels with y growing downward, so "above" means smaller y.
Distances are normalised by body scale (torso length, or shoulder width when the hips
are out of frame, e.g. sitting close to a webcam) so rules are distance invariant.
Left/right refer to the person's anatomical sides.
"""

from __future__ import annotations

from posedemo.poses.base import PoseRule, register
from posedemo.vision.geometry import angle, bell, distance, midpoint, ramp
from posedemo.vision.keypoints import KP
from posedemo.vision.types import Person

SHOULDERS = (KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER)
HIPS = (KP.LEFT_HIP, KP.RIGHT_HIP)
LEFT_ARM = (KP.LEFT_SHOULDER, KP.LEFT_ELBOW, KP.LEFT_WRIST)
RIGHT_ARM = (KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, KP.RIGHT_WRIST)
LEGS = (KP.LEFT_KNEE, KP.RIGHT_KNEE, KP.LEFT_ANKLE, KP.RIGHT_ANKLE)


# Typical torso length (shoulders → hips) relative to keypoint shoulder width.
TORSO_PER_SHOULDER_WIDTH = 1.4


def body_scale(p: Person) -> float:
    """Torso length in pixels; estimated from shoulder width if hips aren't visible."""
    mid_sh = midpoint(p.xy(KP.LEFT_SHOULDER), p.xy(KP.RIGHT_SHOULDER))
    if p.visible(*HIPS):
        mid_hip = midpoint(p.xy(KP.LEFT_HIP), p.xy(KP.RIGHT_HIP))
        return max(distance(mid_sh, mid_hip), 1e-6)
    width = distance(p.xy(KP.LEFT_SHOULDER), p.xy(KP.RIGHT_SHOULDER))
    return max(width * TORSO_PER_SHOULDER_WIDTH, 1e-6)


def hip_y(p: Person) -> float:
    """Hip height in pixels (estimated below the shoulders if hips aren't visible)."""
    if p.visible(*HIPS):
        return float(midpoint(p.xy(KP.LEFT_HIP), p.xy(KP.RIGHT_HIP))[1])
    return float(midpoint(p.xy(KP.LEFT_SHOULDER), p.xy(KP.RIGHT_SHOULDER))[1]) + body_scale(p)


def wrist_raise(p: Person, side: str) -> float:
    """How far the wrist is above the shoulder, in torso lengths (positive = above)."""
    sh = KP.LEFT_SHOULDER if side == "left" else KP.RIGHT_SHOULDER
    wr = KP.LEFT_WRIST if side == "left" else KP.RIGHT_WRIST
    return (p.xy(sh)[1] - p.xy(wr)[1]) / body_scale(p)


def hand_up_score(p: Person, side: str) -> float:
    return ramp(wrist_raise(p, side), 0.15, 0.5)


def arm_out_score(p: Person, side: str) -> float:
    """Arm extended horizontally sideways."""
    if side == "left":
        hip, sh, el, wr = KP.LEFT_HIP, KP.LEFT_SHOULDER, KP.LEFT_ELBOW, KP.LEFT_WRIST
    else:
        hip, sh, el, wr = KP.RIGHT_HIP, KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, KP.RIGHT_WRIST
    # Body axis: toward the hip if visible, otherwise straight down.
    axis = p.xy(hip) if p.visible(hip) else p.xy(sh) + [0.0, 1.0]
    abduction = angle(axis, p.xy(sh), p.xy(wr))  # ~90 when arm is horizontal
    straight = angle(p.xy(sh), p.xy(el), p.xy(wr))  # ~180 when arm is straight
    level = abs(wrist_raise(p, side))  # wrist near shoulder height
    return min(bell(abduction, 90, 35), ramp(straight, 130, 160), ramp(level, 0.45, 0.2))


def knee_angles(p: Person) -> tuple[float, float]:
    left = angle(p.xy(KP.LEFT_HIP), p.xy(KP.LEFT_KNEE), p.xy(KP.LEFT_ANKLE))
    right = angle(p.xy(KP.RIGHT_HIP), p.xy(KP.RIGHT_KNEE), p.xy(KP.RIGHT_ANKLE))
    return left, right


@register
class TPose(PoseRule):
    name = "t_pose"
    description = "Both arms stretched out horizontally"
    required = (*LEFT_ARM, *RIGHT_ARM)

    def score(self, p: Person) -> float:
        return min(arm_out_score(p, "left"), arm_out_score(p, "right"))


@register
class HandsUp(PoseRule):
    name = "hands_up"
    description = "Both hands raised above the shoulders"
    required = (*LEFT_ARM, *RIGHT_ARM)

    def score(self, p: Person) -> float:
        return min(hand_up_score(p, "left"), hand_up_score(p, "right"))


@register
class LeftHandUp(PoseRule):
    name = "left_hand_up"
    description = "Only the left hand raised"
    required = (*LEFT_ARM, *RIGHT_ARM)

    def score(self, p: Person) -> float:
        return min(hand_up_score(p, "left"), 1.0 - hand_up_score(p, "right"))


@register
class RightHandUp(PoseRule):
    name = "right_hand_up"
    description = "Only the right hand raised"
    required = (*LEFT_ARM, *RIGHT_ARM)

    def score(self, p: Person) -> float:
        return min(hand_up_score(p, "right"), 1.0 - hand_up_score(p, "left"))


@register
class ArmsCrossed(PoseRule):
    name = "arms_crossed"
    description = "Wrists crossed in front of the chest"
    required = (*LEFT_ARM, *RIGHT_ARM)

    def score(self, p: Person) -> float:
        ls, rs = p.xy(KP.LEFT_SHOULDER), p.xy(KP.RIGHT_SHOULDER)
        span = ls[0] - rs[0]
        if abs(span) < 1e-6:
            return 0.0
        # Position along the shoulder axis: 0 at right shoulder, 1 at left shoulder.
        t_left = (p.xy(KP.LEFT_WRIST)[0] - rs[0]) / span
        t_right = (p.xy(KP.RIGHT_WRIST)[0] - rs[0]) / span
        crossed = min(ramp(t_left, 0.6, 0.35), ramp(t_right, 0.4, 0.65))
        # Wrists between shoulder and hip height (in front of the chest).
        torso = body_scale(p)
        low_y = hip_y(p)
        sh_y = midpoint(ls, rs)[1]
        in_band = min(
            ramp(p.xy(w)[1], sh_y - 0.2 * torso, sh_y)
            * ramp(p.xy(w)[1], low_y, low_y - 0.2 * torso)
            for w in (KP.LEFT_WRIST, KP.RIGHT_WRIST)
        )
        return min(crossed, in_band)


@register
class Squat(PoseRule):
    name = "squat"
    description = "Knees bent with hips lowered toward knee height"
    required = (*HIPS, *LEGS, *SHOULDERS)

    def score(self, p: Person) -> float:
        lk, rk = knee_angles(p)
        bent = min(ramp(lk, 150, 110), ramp(rk, 150, 110))
        knee_y = midpoint(p.xy(KP.LEFT_KNEE), p.xy(KP.RIGHT_KNEE))[1]
        # Hip-to-knee vertical gap shrinks as you squat (in torso lengths).
        low = ramp((knee_y - hip_y(p)) / body_scale(p), 0.8, 0.35)
        return min(bent, low)


@register
class Standing(PoseRule):
    name = "standing"
    description = "Upright with straight legs and arms down"
    required = (*HIPS, *LEGS, *SHOULDERS)
    priority = 0.85  # generic; more specific poses should win ties

    def score(self, p: Person) -> float:
        lk, rk = knee_angles(p)
        straight = min(ramp(lk, 145, 165), ramp(rk, 145, 165))
        # Upright: torso roughly vertical.
        mid_sh = midpoint(p.xy(KP.LEFT_SHOULDER), p.xy(KP.RIGHT_SHOULDER))
        mid_hip = midpoint(p.xy(KP.LEFT_HIP), p.xy(KP.RIGHT_HIP))
        below = mid_hip + [0.0, 1.0]
        upright = ramp(angle(mid_sh, mid_hip, below), 140, 165)
        # Arms not doing anything notable (when visible).
        arms_down = 1.0
        for side, wr in (("left", KP.LEFT_WRIST), ("right", KP.RIGHT_WRIST)):
            if p.visible(wr):
                arms_down = min(arms_down, ramp(wrist_raise(p, side), 0.0, -0.4))
        return min(straight, upright, arms_down)
