"""Git tools (stub)."""
from pathlib import Path
import git


def get_recent_commits(repo_path: str = ".", n: int = 5) -> dict:
    """
    Return the last n commit messages with author and date.
    """
    try:
        repo = git.Repo(repo_path, search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        return {"error": f"Not a git repository: {repo_path}", "commits": []}

    commits = []
    for commit in repo.iter_commits(max_count=n):
        commits.append({
            "hash": commit.hexsha[:7],
            "message": commit.message.strip(),
            "author": str(commit.author),
            "date": commit.committed_datetime.isoformat()
        })

    return {
        "repo": repo_path,
        "commits": commits,
        "error": None
    }


def get_file_diff(filepath: str, repo_path: str = ".") -> dict:
    """
    Get the unstaged diff for a specific file.
    Falls back to last commit diff if no unstaged changes.
    """
    try:
        repo = git.Repo(repo_path, search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        return {"error": f"Not a git repository: {repo_path}", "diff": None}

    path = Path(filepath)
    if not path.exists():
        return {"error": f"File not found: {filepath}", "diff": None}

    # try unstaged diff first
    diff = repo.git.diff(str(filepath))

    # fall back to diff against last commit
    if not diff:
        try:
            diff = repo.git.diff("HEAD~1", "HEAD", "--", str(filepath))
        except git.GitCommandError:
            diff = "No diff available — file may not be committed yet."

    return {
        "filepath": str(filepath),
        "diff": diff or "No changes detected.",
        "error": None
    }


def get_repo_summary(repo_path: str = ".") -> dict:
    """
    Return high-level repo info: branch, latest commit, tracked files count.
    """
    try:
        repo = git.Repo(repo_path, search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        return {"error": f"Not a git repository: {repo_path}"}

    try:
        branch = repo.active_branch.name
    except TypeError:
        branch = "detached HEAD"

    latest = repo.head.commit
    tracked = len([item for item in repo.tree().traverse() if item.type == "blob"])

    return {
        "branch": branch,
        "latest_commit": {
            "hash": latest.hexsha[:7],
            "message": latest.message.strip(),
            "author": str(latest.author),
            "date": latest.committed_datetime.isoformat()
        },
        "tracked_files": tracked,
        "error": None
    }