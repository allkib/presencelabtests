"""Pose recognition. Importing this package registers the built-in rules."""

from posedemo.poses import rules as _rules  # noqa: F401  (registers built-in rules)
from posedemo.poses.base import PoseRule, all_rules, get_rule, register
from posedemo.poses.classifier import UNKNOWN, PoseClassifier
from posedemo.poses.smoothing import LabelSmoother

__all__ = [
    "UNKNOWN",
    "LabelSmoother",
    "PoseClassifier",
    "PoseRule",
    "all_rules",
    "get_rule",
    "register",
]
