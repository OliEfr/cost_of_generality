# Compat shim for running the lerobot 0.5.2 multi_task_dit policy on pinned lerobot 0.4.4.
#
# 0.4.4's lerobot.utils.import_utils has `_transformers_available` but neither
# `_diffusers_available` nor `require_package` (both added upstream by 0.5.x). We provide
# them here on top of 0.4.4's own `is_package_available`. `require_package` is copied
# verbatim from the 0.5.2 checkout (utils/import_utils.py:86-95,
# /home/admin_07/project_repos/lerobot_AICchallange/lerobot @ fc6c94c).
from lerobot.utils.import_utils import is_package_available

_diffusers_available = is_package_available("diffusers")

_require_package_cache: dict[str, bool] = {}


def require_package(pkg_name: str, extra: str, import_name: str | None = None) -> None:
    """Raise an informative ImportError if a package required by an optional feature is missing."""
    cache_key = import_name or pkg_name
    if cache_key not in _require_package_cache:
        _require_package_cache[cache_key] = is_package_available(pkg_name, import_name)
    if not _require_package_cache[cache_key]:
        raise ImportError(
            f"'{pkg_name}' is required but not installed. Install it with: "
            f"pip install 'lerobot[{extra}]' (or uv pip install 'lerobot[{extra}]')"
        )
