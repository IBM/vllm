# Repo organization and building the TGIS-vLLM image

This fork attempts to remain aligned with the vLLM repo as much as possible,
while also containing a set of permanent changes to add:
- A TGIS api adapter layer (see [TGIS](github.com/IBM/text-generation-inference))
- A RedHat UBI-based Docker image delivery

Given the fast pace of vLLM development, we also provide builds that include yet-to-be-merged
PRs to vLLM by squash-merging open vLLM PRs onto an ephemeral branch on top of main that is continually
reconstructed as we make more contributions.

See a sketch of the commit graph
![vllm commit strategy](docs/source/assets/tgis-vllm-repo.png)

## Contributing changes for vLLM

To contribute vLLM-specific changes, _don't_ base them on the `main` branch in this repo.


## Contributing changes specific to the TGIS adapter or IBM delivery


## Main branch rebasing procedure

Rebasing vllm:main onto ibm:main is pretty straightforward. Assuming you have vllm-project/vllm as the
`upstream` remote and ibm/vllm as the `origin` remote, one way to do this is:
```shell
# fetch latest ibm main
git fetch origin/main

# fetch latest vllm main
git fetch upstream/main
git checkout upstream/main

# point a branch there
git checkout -b upstream-main-sync
# (or if you already had that branch)
git branch -f upstream-main-sync HEAD
git checkout upstream-main-sync

# rebase the branch onto ibm main
git rebase origin/main

# push push...

```

## Ephemeral branch building procedure

