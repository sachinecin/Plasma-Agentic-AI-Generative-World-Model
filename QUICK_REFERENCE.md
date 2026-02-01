# Quick Reference: Branch Merging Commands

## Basic Merge Workflow

```bash
# Step 1: Switch to main branch
git checkout main

# Step 2: Pull latest changes
git pull origin main

# Step 3: Merge feature branch
git merge feature

# Step 4: Merge development branch  
git merge development

# Step 5: Push to remote
git push origin main
```

## With Conflict Resolution

```bash
# After merge conflict
git status                    # See conflicted files
git add <resolved-file>       # Stage resolved files
git commit                    # Complete the merge
```

## Common Options

```bash
# Merge with no fast-forward (preserves branch history)
git merge feature --no-ff

# Merge with custom message
git merge feature -m "Your merge message"

# Abort merge if needed
git merge --abort
```

## Verification

```bash
# View branch structure
git log --oneline --graph --all

# Check current status
git status

# View recent changes
git diff HEAD~1
```

---

📖 **For detailed instructions**, see `MERGE_GUIDE.md`  
📊 **For execution summary**, see `MERGE_SUMMARY.md`
