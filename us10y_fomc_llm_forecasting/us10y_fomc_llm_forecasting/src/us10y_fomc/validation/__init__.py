"""Hard validation gates for data, leakage and learned model plumbing."""

from .leakage import audit_backbone_cutoff, audit_event_splits

__all__ = ["audit_backbone_cutoff", "audit_event_splits"]
