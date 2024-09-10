# Git Workflow: Building the Development Image Release for VLLM

This guide outlines the steps to create a new branch, sync it with upstream changes, merge pull requests (PRs) using squash commits, and handle updates from both upstream and ODH repositories. The process is geared towards building the development image release for VLLM, ensuring that each step is clear to avoid common mistakes.
In the previous process, we used the repository `github.com/ibm/vllm:main` as the base repository, while keeping `odh/vllm:main` as a tracked repository. Now, in this tutorial, we directly use the `odh/vllm:main` repository as the base, bringing in updates from `upstream:main` and merging some necessary PRs to build the current development image.

## Repositories
- **Upstream Repository**: https://github.com/vllm-project/
- **ODH Repository (Origin)**: https://github.com/opendatahub-io/vllm
- **vllm-tgis-adapter** Repository: https://github.com/opendatahub-io/vllm-tgis-adapter

## Steps

### 1. **Create a New Branch**
Clone and create a new branch from `odh/vllm:main`.
```bash
git clone git@github.com:opendatahub-io/vllm.git
cd vllm
git checkout -b <new-branch-name>
```
- This branch will be used to prepare the new release.

### 2. **Add Upstream Repository**
Ensure you have the upstream repository set to track changes from the VLLM project.
```bash
git remote add upstream https://github.com/vllm-project/vllm.git
```

### 3. **Fetch Updates from Origin and Upstream**
Keep your local repository up-to-date with both `origin` and `upstream`.
```bash
git fetch origin main
git fetch upstream main
```

### 4. **Pull the Latest Changes from Upstream**
Update your branch with the latest changes from upstream to ensure compatibility and commit them.
```bash
git config pull.rebase false # merging without rebasing to avoid conflicts.
git pull upstream main
git commit -s --amend # To sign-off the merge commit. 
```
**Tip:** We will use this commit hash in changelog documentation.  

### 5. **Fetch the PR from Upstream**
Fetch the specific PR that needs to be merged into your branch.
```bash
export PR_NUMBER=<PR-number>
git branch -D ${PR_NUMBER} # Ensure any existing branch with the same name is deleted
git fetch upstream pull/${PR_NUMBER}/head:${PR_NUMBER}
```

### 6. **Switch to upstream:main**
Note that the PR changes will be merged into the `upstream:main` branch before being merged into your working branch.
```bash
git checkout upstream/main
```

### 7. **Merge the PR with Squash**
Combine all commits from the PR into a single commit for a cleaner history.
```bash
git merge --squash ${PR_NUMBER}
git commit -s -m "Squash ${PR_NUMBER}"
```

### 8. **Export the Squash Commit Hash**
Retrieve and store the hash of the squashed commit.
```bash
export SQUASH_HEAD=$(git rev-parse --short HEAD)
# This command saves the commit hash of the squashed changes for later use.
```

### 9. **Cherry-Pick the Squashed Commit**
Mandatory step to apply the changes to the release branch.
```bash
git checkout <target-branch>
git cherry-pick $SQUASH_HEAD
```

**Important Note**: If more PRs need to be added, repeat the loop from steps 5 to 9.

### 10. **Verify the `vllm-tgis-adapter` Version**
Before proceeding to pull changes into the `ibm-dev` branch, verify the version of `vllm-tgis-adapter` being installed in the `Dockerfile.ubi`. This is a critical step to avoid mismatches during the build.
```bash
# Open the Dockerfile in your preferred text editor
vim Dockerfile.ubi
# or use grep:
grep "vllm-tgis-adapter" Dockerfile.ubi
```
- The repository for `vllm-tgis-adapter` is located at: https://github.com/opendatahub-io/vllm-tgis-adapter

Ensure the correct version is being installed to avoid issues during the build process.

### 11. **Push the Changes to Origin**
Push the final changes to trigger the build process.
Currently, the branch `odh/vllm:ibm-dev` is configured to trigger the `fast-ibm` build.
```bash
git push --force origin <target-branch>:ibm-dev
```
- The build logs will be available in **Prow** after pushing to `odh/vllm:ibm-dev`.

### 12. **Add a Tag to the Repository**
After completing the updates and ensuring everything is functioning as expected, tag the last commit.
```bash
export LAST_COMMIT=$(git rev-parse --short HEAD)
git tag fast-ibm-$LAST_COMMIT
git push origin fast-ibm-$LAST_COMMIT
```
- This tag helps identify the build associated with the current release.

## Example Scenarios

### Example 1: Merging a PR from ODH (Origin) Instead of Upstream
To merge a pull request from the ODH repository instead of the upstream, follow the same loop starting from step 5, with a slight change in the repository used to fetch the PR, replacing `upstream` with `origin`:
```bash
git fetch origin pull/${PR_NUMBER}/head:${PR_NUMBER}
```

### Example 2: Bring Upstream Updates While Keeping Merged PR Changes
To bring updates from upstream without losing the changes already merged in your branch, follow these steps:

#### 1. **Log the Last Commit**
Before performing the update, it's a good idea to note the last commit so you can revert if something goes wrong:
```bash
git log --oneline
```
Copy the hash of the last commit for future reference.

#### 2. **Checkout `upstream:main` and Pull the Latest Changes**
```bash
git checkout upstream/main
git pull upstream main
```

#### 3. **Merge the Updates into Your Working Branch**
After pulling the upstream updates, switch back to your working branch and merge them:
```bash
git checkout <working-branch>
git merge upstream/main
```

#### 4. **Reset if Necessary**
If anything goes wrong during the merge or pull, you can use the commit hash you copied earlier to reset the branch:
```bash
git reset --hard <commit-hash>
```
If you want to bring modification from origin instead upstream, you just need to change `upstream` to `origin` in the steps **2.** and **3.** 

## Tips to Avoid Mistakes
- **Always double-check PR numbers and branch names** before fetching or merging.
- **Resolve conflicts attentively** during the pull and merge processes.
- **Monitor the CI/CD pipeline in Prow** to verify the build and deployment status.
- **Be cautious with `--force` push**, especially after amending commits or when pushing tags.
