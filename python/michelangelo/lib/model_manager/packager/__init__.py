"""Model packager abstractions.

Provides the shared ``PackagerBase`` contract implemented by the concrete
Triton packagers:

    from michelangelo.lib.model_manager.packager import PackagerBase
    from michelangelo.lib.model_manager.packager.custom_triton import (
        CustomTritonPackager,
    )
    from michelangelo.lib.model_manager.packager.torch_triton import (
        TorchTritonPackager,
    )
"""

from michelangelo.lib.model_manager.packager.base import PackagerBase

__all__ = ["PackagerBase"]
