import os
import re
import logging

logger = logging.getLogger(__name__)


def resolve_video_path(path: str) -> str:
    """
    Resolves a video or input asset path.
    If the path exists as-is, returns it.
    If not found, searches on D: drive (e.g., D:\\cctv_samples)
    and checks common naming variations.
    """
    if not path:
        return path

    # If it already exists on disk
    if os.path.exists(path):
        return os.path.abspath(path) if not os.path.isabs(path) else path

    # Normalize path separators
    norm = path.replace("\\", "/")

    # Candidate filename/path variations
    def _get_variations(p: str):
        variants = [p]
        substitutions = [
            ("test_2", "test2"),
            ("test2", "test_2"),
            ("passage1", "passage_1"),
            ("passage_1", "passage1"),
            ("pantry", "panty"),
            ("panty", "pantry"),
            ("cctv_samples/pass3.mp4", "cctv_samples/frame/pass3.mp4"),
            ("cctv_samples/pass1.mp4", "cctv_samples/frame/pass1.mp4"),
        ]
        for old, new in substitutions:
            if old in p:
                variants.append(p.replace(old, new))
        return variants

    # Check local variations first
    for v in _get_variations(norm):
        if os.path.exists(v):
            return os.path.abspath(v)

    # Check on D: drive if cctv_samples was moved to D:
    no_drive = re.sub(r'^[A-Za-z]:[\\/]', '', norm)
    d_candidates = [
        f"D:/{no_drive.lstrip('/')}",
    ]
    if not no_drive.startswith("cctv_samples"):
        d_candidates.append(f"D:/cctv_samples/{no_drive.lstrip('/')}")

    for d_base in d_candidates:
        for v in _get_variations(d_base):
            if os.path.exists(v):
                return v

    # If still not found, return original path
    return path


def resolve_output_path(path: str, default_subpath: str = "output") -> str:
    """
    Resolves an output path to prevent C: drive disk-space errors.
    If the output folder has moved to D:\\output or D:\\ is available,
    relative paths under 'output' or default paths are redirected to D:\\output.
    """
    if not path:
        path = default_subpath

    norm = path.replace("\\", "/")

    # If already explicitly targeting D:
    if re.match(r'^[Dd]:', norm):
        out_dir = os.path.dirname(norm)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        return norm

    # If already an absolute path pointing elsewhere (e.g. temp directory in tests)
    if os.path.isabs(path) and not re.match(r'^[Cc]:.*[\\/]output([\\/]|$)', path, re.IGNORECASE):
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        return path

    # If D:\output exists or D:\ exists, re-route output to D:
    if os.path.exists(r"D:\output") or os.path.exists(r"D:\\"):
        no_drive = re.sub(r'^[A-Za-z]:[\\/]', '', norm).lstrip('/')
        if no_drive.lower().startswith("output"):
            target = f"D:/{no_drive}"
        else:
            target = f"D:/output/{no_drive}"
        out_dir = os.path.dirname(target)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        return target

    # Fallback to local
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return path
