"""
Fetch GitHub Issues and PRs with comments from a public repository.
Uses the GitHub REST API (no auth required for public repos, but token increases rate limits).
"""

import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class GitHubCorpusFetcher:
    """Fetches issues, PRs, and comments from a GitHub repository."""

    BASE_URL = "https://api.github.com"

    def __init__(self, repo: str = None, token: str = None):
        self.repo = repo or config.GITHUB_REPO
        self.token = token or config.GITHUB_TOKEN
        self.session = requests.Session()
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
        self.session.headers["Accept"] = "application/vnd.github.v3+json"
        self.session.headers["User-Agent"] = "Layer10-Memory-Graph"

    def _get(self, url: str, params: dict = None) -> requests.Response:
        """Make a GET request with rate-limit handling."""
        resp = self.session.get(url, params=params)
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            reset_time = int(resp.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset_time - time.time(), 10)
            print(f"  Rate limited. Waiting {wait:.0f}s...")
            time.sleep(wait + 1)
            resp = self.session.get(url, params=params)
        resp.raise_for_status()
        return resp

    def fetch_issues(self, max_issues: int = None, state: str = "all",
                     sort: str = "comments", direction: str = "desc") -> list[dict]:
        """
        Fetch issues (including PRs) from the repo.
        Sorted by most comments by default to get richer data.
        """
        max_issues = max_issues or config.MAX_ISSUES
        url = f"{self.BASE_URL}/repos/{self.repo}/issues"
        issues = []
        page = 1
        per_page = min(100, max_issues)

        print(f"Fetching up to {max_issues} issues from {self.repo}...")
        with tqdm(total=max_issues, desc="Issues") as pbar:
            while len(issues) < max_issues:
                params = {
                    "state": state,
                    "sort": sort,
                    "direction": direction,
                    "per_page": per_page,
                    "page": page,
                }
                resp = self._get(url, params)
                batch = resp.json()
                if not batch:
                    break
                for issue in batch:
                    if len(issues) >= max_issues:
                        break
                    issues.append(self._normalize_issue(issue))
                    pbar.update(1)
                page += 1

        print(f"Fetched {len(issues)} issues.")
        return issues

    def fetch_comments(self, issues: list[dict]) -> list[dict]:
        """Fetch comments for each issue."""
        print(f"Fetching comments for {len(issues)} issues...")
        all_comments = []
        for issue in tqdm(issues, desc="Comments"):
            if issue["comment_count"] == 0:
                continue
            url = issue["comments_url"]
            try:
                resp = self._get(url, params={"per_page": 100})
                comments = resp.json()
                for c in comments:
                    all_comments.append(self._normalize_comment(c, issue["id"]))
            except Exception as e:
                print(f"  Warning: Failed to fetch comments for issue #{issue['number']}: {e}")

        print(f"Fetched {len(all_comments)} comments.")
        return all_comments

    def _normalize_issue(self, raw: dict) -> dict:
        """Normalize a GitHub issue/PR into a canonical format."""
        is_pr = "pull_request" in raw
        labels = [l["name"] for l in raw.get("labels", [])]
        assignees = [a["login"] for a in raw.get("assignees", [])]

        return {
            "id": f"github-issue-{self.repo}-{raw['number']}",
            "source": "github",
            "repo": self.repo,
            "number": raw["number"],
            "type": "pull_request" if is_pr else "issue",
            "title": raw["title"],
            "body": raw.get("body") or "",
            "state": raw["state"],
            "author": raw["user"]["login"] if raw.get("user") else "unknown",
            "labels": labels,
            "assignees": assignees,
            "created_at": raw["created_at"],
            "updated_at": raw["updated_at"],
            "closed_at": raw.get("closed_at"),
            "comment_count": raw.get("comments", 0),
            "comments_url": raw["comments_url"],
            "html_url": raw["html_url"],
            "content_hash": hashlib.sha256(
                (raw["title"] + (raw.get("body") or "")).encode()
            ).hexdigest()[:16],
        }

    def _normalize_comment(self, raw: dict, issue_id: str) -> dict:
        """Normalize a GitHub comment."""
        return {
            "id": f"github-comment-{raw['id']}",
            "issue_id": issue_id,
            "source": "github",
            "author": raw["user"]["login"] if raw.get("user") else "unknown",
            "body": raw.get("body") or "",
            "created_at": raw["created_at"],
            "updated_at": raw["updated_at"],
            "html_url": raw["html_url"],
            "content_hash": hashlib.sha256(
                (raw.get("body") or "").encode()
            ).hexdigest()[:16],
        }

    def fetch_and_save(self, output_dir: Path = None, max_issues: int = None) -> dict:
        """Fetch the full corpus and save to disk."""
        output_dir = output_dir or config.RAW_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        issues = self.fetch_issues(max_issues=max_issues)
        comments = self.fetch_comments(issues)

        corpus = {
            "metadata": {
                "repo": self.repo,
                "fetched_at": datetime.utcnow().isoformat(),
                "num_issues": len(issues),
                "num_comments": len(comments),
            },
            "issues": issues,
            "comments": comments,
        }

        output_path = output_dir / "corpus.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)

        print(f"Corpus saved to {output_path}")
        print(f"  Issues: {len(issues)}, Comments: {len(comments)}")
        return corpus


def load_corpus(path: Path = None) -> dict:
    """Load a previously fetched corpus from disk."""
    path = path or (config.RAW_DIR / "corpus.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    fetcher = GitHubCorpusFetcher()
    fetcher.fetch_and_save()
