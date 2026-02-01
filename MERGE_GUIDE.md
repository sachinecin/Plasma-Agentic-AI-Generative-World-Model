# Git Branch Merge Guide

This document provides step-by-step instructions for merging the 'feature' and 'development' branches into the 'main' branch.

## Prerequisites

Before starting the merge process, ensure you have:
- Git installed on your system
- Proper access rights to the repository
- All local changes committed or stashed

## Step-by-Step Merge Process

### Step 1: Switch to Main Branch

First, switch to the main branch:

```bash
git checkout main
```

### Step 2: Pull Latest Changes

Ensure your main branch is up to date with the remote repository:

```bash
git pull origin main
```

This fetches and merges any changes from the remote main branch to your local main branch.

### Step 3: Merge Feature Branch

Merge the 'feature' branch into main:

```bash
git merge feature
```

#### Handling Conflicts (if any)

If there are merge conflicts:

1. Git will notify you of conflicted files
2. Open each conflicted file and look for conflict markers:
   ```
   <<<<<<< HEAD
   (your changes in main)
   =======
   (changes from feature branch)
   >>>>>>> feature
   ```
3. Edit the file to resolve conflicts by:
   - Keeping the changes you want
   - Removing the conflict markers
   - Combining changes if needed

4. After resolving conflicts, stage the resolved files:
   ```bash
   git add <resolved-file>
   ```

5. Complete the merge:
   ```bash
   git commit -m "Merge feature branch into main"
   ```

### Step 4: Merge Development Branch

After successfully merging the feature branch, merge the 'development' branch:

```bash
git merge development
```

Follow the same conflict resolution process as described in Step 3 if conflicts arise.

### Step 5: Push Changes to Remote

After successfully merging both branches, push the updated main branch to the remote repository:

```bash
git push origin main
```

## Complete Command Sequence

Here's the complete sequence of commands for a smooth merge (assuming no conflicts):

```bash
# Switch to main branch
git checkout main

# Pull latest changes from remote
git pull origin main

# Merge feature branch
git merge feature

# If conflicts occur, resolve them and commit
# git add <resolved-files>
# git commit -m "Merge feature branch into main"

# Merge development branch
git merge development

# If conflicts occur, resolve them and commit
# git add <resolved-files>
# git commit -m "Merge development branch into main"

# Push merged changes to remote
git push origin main
```

## Advanced Options

### Fast-Forward Merge

If you want a clean linear history (only if main hasn't diverged):

```bash
git merge --ff-only feature
git merge --ff-only development
```

### No Fast-Forward Merge

To always create a merge commit (preserves branch history):

```bash
git merge --no-ff feature
git merge --no-ff development
```

### Merge with Message

Specify a custom merge commit message:

```bash
git merge feature -m "Merge feature branch: Add phantom path simulation"
git merge development -m "Merge development branch: Add adversarial auditing"
```

## Conflict Resolution Tips

1. **Use a merge tool**: Git supports various visual merge tools
   ```bash
   git mergetool
   ```

2. **Check conflict status**:
   ```bash
   git status
   ```

3. **View changes**:
   ```bash
   git diff
   ```

4. **Abort merge** (if you need to start over):
   ```bash
   git merge --abort
   ```

## Verification Commands

After merging, verify the changes:

```bash
# View commit history
git log --oneline --graph --all

# View what changed
git diff HEAD~1

# Check current branch status
git status
```

## Best Practices

1. **Always pull before merging**: Ensure your main branch is up to date
2. **Test after merging**: Run tests to ensure merged code works correctly
3. **Review changes**: Use `git log` and `git diff` to review what was merged
4. **Communicate**: Inform team members about the merge
5. **Backup**: Create a backup branch before complex merges
   ```bash
   git branch backup-main
   ```

## Troubleshooting

### Issue: "Your branch is behind 'origin/main'"
**Solution**: Pull the latest changes:
```bash
git pull origin main
```

### Issue: "fatal: refusing to merge unrelated histories"
**Solution**: Use the `--allow-unrelated-histories` flag:
```bash
git merge feature --allow-unrelated-histories
```

### Issue: Merge conflict in binary files
**Solution**: Choose one version:
```bash
# Use version from main
git checkout --ours <file>
# Or use version from feature branch
git checkout --theirs <file>
git add <file>
```

## Summary

This guide covers the complete process of merging multiple branches into main. The key steps are:

1. Switch to main: `git checkout main`
2. Pull latest: `git pull origin main`
3. Merge branches: `git merge feature` then `git merge development`
4. Resolve any conflicts
5. Push changes: `git push origin main`

Remember to always communicate with your team and test thoroughly after merging.
