"""File system tools (stub)."""
from pathlib import Path


def read_file(filepath: str) -> dict:
    """
    Read content of a file relative to current working directory.
    Returns content and metadata.
    """
    path = Path(filepath)

    if not path.exists():
        return {"error": f"File not found: {filepath}", "content": None}

    if not path.is_file():
        return {"error": f"Path is not a file: {filepath}", "content": None}

    # safety guard — don't read binary files
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {"error": f"Cannot read binary file: {filepath}", "content": None}

    return {
        "filepath": str(path),
        "content": content,
        "lines": len(content.splitlines()),
        "size_bytes": path.stat().st_size,
        "error": None
    }


def list_files(directory: str = ".", extensions: list[str] = None) -> dict:
    """
    List all files in a directory recursively.
    Optionally filter by extensions e.g. ['.py', '.md']
    Ignores hidden dirs, __pycache__, .venv, chroma_db.
    """
    IGNORE_DIRS = {".git", "__pycache__", ".venv", "venv",
                   "chroma_db", ".mypy_cache", "node_modules", ".pytest_cache"}

    base = Path(directory)
    if not base.exists():
        return {"error": f"Directory not found: {directory}", "files": []}

    files = []
    for path in sorted(base.rglob("*")):
        # skip ignored directories
        if any(ignored in path.parts for ignored in IGNORE_DIRS):
            continue
        if not path.is_file():
            continue
        if extensions and path.suffix not in extensions:
            continue
        files.append(str(path))

    return {
        "directory": str(base),
        "file_count": len(files),
        "files": files,
        "error": None
    }