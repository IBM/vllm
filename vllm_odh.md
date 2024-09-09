# Git Workflow: Syncing and Merging PRs with Squash

This guide outlines the steps to create a new branch, sync it with upstream changes, and merge pull requests (PRs) using squash commits. The process is flexible and can be applied to various repositories, not just `vllm`.

## Steps

### 1. **Create a New Branch**
Before working on new changes or syncing updates, create a new branch from `odh/vllm:main` or another repository branch.
```bash
git checkout -b <new-branch-name>
```
- Make sure you have the latest changes from the main branch before branching off.

### 2. **Add Upstream Repository**
Add the upstream repository if it’s not already set.
```bash
git remote add upstream https://github.com/<upstream-repo>.git
```
- Use the appropriate URL for the upstream repository, which could be `vllm`, or any other related repo.

### 3. **Fetch Updates from Origin and Upstream**
Sync your local repository with both `origin` and `upstream`.
```bash
git fetch origin main
git fetch upstream main
```
- **Tip:** Always ensure your local branch is up to date with the main branch of both `origin` and `upstream`.

### 4. **Pull the Latest Changes from Upstream**
Now, pull the latest changes from the upstream repository into your local branch.
```bash
git pull upstream main
```
- **Tip:** If you encounter merge conflicts, carefully resolve them before proceeding.

### 5. **Check Repository Status**
Before proceeding with PRs, check the current status of your repository.
```bash
git status
```
- This helps ensure you don’t have uncommitted changes before fetching a new PR.

### 6. **Fetch the PR from Upstream (or Another Repo)**
Fetch the PR from the upstream repository or from another source.
```bash
export PR_NUMBER=<PR-number>
git fetch upstream pull/${PR_NUMBER}/head:${PR_NUMBER}
```
- **Tip:** Ensure the PR number is correct before fetching.

### 7. **Merge the PR with Squash**
Now, merge the fetched PR using squash to combine all commits into a single commit.
```bash
git merge --squash ${PR_NUMBER}
```
- **Tip:** Use squash to maintain a clean commit history.

### 8. **Commit the Squashed Changes**
After the squash, commit the changes with a message that reflects the PR.
```bash
git commit -s -m "Squash ${PR_NUMBER}"
```
- **Tip:** The `-s` flag signs your commit, which may be required depending on your repository settings.

### 9. **(Optional) Cherry-pick the Squashed Commit**
If you need to cherry-pick the squashed commit into another branch, use:
```bash
export SQUASH_HEAD=$(git rev-parse --short HEAD)
git checkout <target-branch>
git cherry-pick $SQUASH_HEAD
```
- **Tip:** Make sure the target branch is checked out before cherry-picking.

### 10. **Push the Changes to Origin**
Once the changes are ready, push them to the origin repository.
```bash
git push origin <branch-name>
```
- **Tip:** Use `--force` if necessary (e.g., after an amended commit), but be cautious of overwriting history.

### 11. **Trigger the Build**
If the repository (like `odh/vllm:ibm-dev`) is set up with a CI/CD pipeline, pulling the changes to the target branch can trigger a build, which will be monitored via OpenShift logs.

```bash
# After merging PRs, push to the build branch
git push origin odh_vllm_release_<date>:odh_vllm_release_<date>
```

Monitor the logs in OpenShift to ensure the build completes successfully.

### 12. **Check Image Availability**
After the build is complete, the new image will be available on `quay.io`.

## Tips to Avoid Mistakes

1. **Always double-check your PR number and branch name** before executing `fetch` or `merge` commands.
2. **Resolve merge conflicts carefully** during the `pull` process to avoid overwriting changes.
3. **Use squash commits** to maintain a clean history and avoid a cluttered commit log.
4. **Cherry-pick commits carefully** to avoid unnecessary duplication of changes across branches.
5. **Monitor the CI/CD pipeline** (such as OpenShift) to ensure the build is completed without issues.
6. **Be cautious with `--force` push**, as it rewrites the history of the branch.

---

This process can be adapted to different repositories by adjusting the remote URLs and PR numbers as needed.
