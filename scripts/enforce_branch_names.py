"""Enforce Git branch naming convention."""

import re
from subprocess import check_output
import sys


def main() -> None:
    # 1. Fetch current topology context
    branch = check_output(["git", "branch", "--show-current"]).decode().strip()

    # 2. Strict engineering pattern boundaries
    regex = r"^(feat|feature|fix|bugfix|infra|chore|docs|test|perf)\/[a-z0-9-]+$"
    protected_branches = ["main", "master", "dev", "development"]

    if branch in protected_branches:
        sys.exit(0)

    if not re.match(regex, branch):
        print(f"ERROR: Invalid branch name '{branch}'.")
        print(
            "Matches must strictly follow the blueprint:"
            " <prefix>/kebab-case-description"
        )
        print(
            "Valid prefixes:"
            " feat/, feature/, fix/, bugfix/, infra/, chore/, docs/, test/, perf/"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
