"""One dataset root with raw_data, annotated_data, and analysis folders."""
import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent
DEMO_ROOT = PACKAGE_DIR / 'synthetic_demo'


def resolve_paths():
    configured = os.environ.get('SEB_DATA_ROOT')
    root = Path(configured).expanduser().resolve() if configured else PACKAGE_DIR
    if not configured and not (root / 'annotated_data' / 'series').is_dir():
        root = DEMO_ROOT
    return root, root / 'analysis', root / 'annotated_data'


ROOT_DIR, ALIGNMENT_DIR, STATE_DIR = resolve_paths()
ANNOTATION_STATE_JSON = STATE_DIR / 'alignment_annotations.json'
