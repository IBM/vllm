# Repo organization and building the TGIS-vLLM image

This fork attempts to remain aligned with the vLLM repo as much as possible, while also containing a set of permanent changes to add:
- A TGIS api adapter layer (see [TGIS](github.com/IBM/text-generation-inference))
- A RedHat UBI-based Docker image delivery

Given the fast pace of vLLM development, we also provide builds that include yet-to-be-merged PRs to vLLM.
