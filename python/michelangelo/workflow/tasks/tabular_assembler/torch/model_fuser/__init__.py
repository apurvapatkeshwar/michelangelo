"""Model fusion utilities for combining a native-transform model with a predictor.

``FusedModel`` composes a transform module and a predictor module into a
single ``nn.Module`` for serving. The fusion engine that traces/exports the
composed graph (``fuse.py``) lands in a follow-up change.
"""

from .fused_model import FusedModel

__all__ = ["FusedModel"]
