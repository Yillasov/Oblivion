#!/bin/bash

# Simple version control integration script
# Usage: ./version_control.sh [command] [options]

# Set default values
REPO_ROOT="/Users/yessine/Oblivion"
BRANCH="main"

# Function to check if we're in a git repository
check_git_repo() {
  if ! git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Error: Not a git repository"
    echo "Initialize with: ./version_control.sh init"
    exit 1
  fi
}

# Function to show status with neuromorphic-specific focus
show_status() {
  check_git_repo
  
  echo "=== Oblivion Project Status ==="
  
  # Show current branch
  CURRENT_BRANCH=$(git -C "$REPO_ROOT" branch --show-current)
  echo "Current branch: $CURRENT_BRANCH"
  
  # Show modified neuromorphic files
  echo -e "\nModified neuromorphic files:"
  git -C "$REPO_ROOT" diff --name-only | grep -E '(neuro|neural|snn|hardware)' || echo "None"
  
  # Show all modified files count
  MODIFIED_COUNT=$(git -C "$REPO_ROOT" diff --name-only | wc -l | tr -d ' ')
  echo -e "\nTotal modified files: $MODIFIED_COUNT"
  
  # Show recent commits
  echo -e "\nRecent commits:"
  git -C "$REPO_ROOT" log --oneline -n 5
}

# Function to save changes with a message
save_changes() {
  check_git_repo
  
  if [ -z "$1" ]; then
    echo "Error: Commit message required"
    echo "Usage: ./version_control.sh save \"Your commit message\""
    exit 1
  fi
  
  # Add all changes
  git -C "$REPO_ROOT" add .
  
  # Commit with message
  git -C "$REPO_ROOT" commit -m "$1"
  
  echo "Changes saved with message: $1"
}

# Function to create a snapshot (tag)
create_snapshot() {
  check_git_repo
  
  if [ -z "$1" ]; then
    # Generate default tag name with timestamp
    TAG_NAME="snapshot-$(date +%Y%m%d-%H%M%S)"
  else
    TAG_NAME="$1"
  fi
  
  # Create tag
  git -C "$REPO_ROOT" tag -a "$TAG_NAME" -m "Snapshot: $TAG_NAME"
  
  echo "Created snapshot: $TAG_NAME"
}

# Function to switch to a different version
switch_version() {
  check_git_repo
  
  if [ -z "$1" ]; then
    echo "Error: Branch/tag/commit required"
    echo "Usage: ./version_control.sh switch [branch/tag/commit]"
    exit 1
  fi
  
  # Check if there are uncommitted changes
  if [ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]; then
    echo "Warning: You have uncommitted changes"
    read -p "Do you want to stash them? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      git -C "$REPO_ROOT" stash save "Auto-stash before switching to $1"
      echo "Changes stashed"
    fi
  fi
  
  # Switch to specified version
  git -C "$REPO_ROOT" checkout "$1"
  
  echo "Switched to: $1"
}

# Function to initialize repository
init_repo() {
  if git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Repository already initialized"
  else
    git -C "$REPO_ROOT" init
    git -C "$REPO_ROOT" add .
    git -C "$REPO_ROOT" commit -m "Initial commit for Oblivion neuromorphic project"
    echo "Repository initialized"
  fi
}

# Function to show version history
show_history() {
  check_git_repo
  
  echo "=== Oblivion Project History ==="
  
  # Show commit history with graph
  git -C "$REPO_ROOT" log --graph --oneline --decorate -n 10
}

# Function to sync with remote repository
sync_repo() {
  check_git_repo
  
  # Check if remote exists
  if ! git -C "$REPO_ROOT" remote | grep -q "origin"; then
    echo "No remote repository configured"
    echo "Add with: git remote add origin [url]"
    exit 1
  fi
  
  # Pull changes
  echo "Pulling changes from remote..."
  git -C "$REPO_ROOT" pull origin "$(git -C "$REPO_ROOT" branch --show-current)"
  
  # Push changes
  echo "Pushing changes to remote..."
  git -C "$REPO_ROOT" push origin "$(git -C "$REPO_ROOT" branch --show-current)"
  
  echo "Synchronized with remote repository"
}

# Parse command
case "$1" in
  status)
    show_status
    ;;
  save)
    save_changes "$2"
    ;;
  snapshot)
    create_snapshot "$2"
    ;;
  switch)
    switch_version "$2"
    ;;
  init)
    init_repo
    ;;
  history)
    show_history
    ;;
  sync)
    sync_repo
    ;;
  *)
    echo "Oblivion Version Control"
    echo "Usage: ./version_control.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  status              Show project status"
    echo "  save \"message\"      Save changes with commit message"
    echo "  snapshot [name]     Create a snapshot (tag)"
    echo "  switch [version]    Switch to different version"
    echo "  init                Initialize repository"
    echo "  history             Show version history"
    echo "  sync                Sync with remote repository"
    ;;
esac