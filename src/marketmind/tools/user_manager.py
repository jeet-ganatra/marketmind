"""
Active user management — persists the current user across CLI sessions.

The active user is stored as a plain text file at {data_dir}/.active_user.
Defaults to "default" if no user has been set.
"""

from pathlib import Path

DEFAULT_USERNAME = "default"


def get_active_username(data_dir: Path) -> str:
    """Read the active username from disk. Returns 'default' if not set."""
    user_file = data_dir / ".active_user"
    if user_file.exists():
        username = user_file.read_text().strip()
        if username:
            return username
    return DEFAULT_USERNAME


def set_active_username(data_dir: Path, username: str) -> None:
    """Write the active username to disk."""
    data_dir.mkdir(parents=True, exist_ok=True)
    user_file = data_dir / ".active_user"
    user_file.write_text(username.strip())
