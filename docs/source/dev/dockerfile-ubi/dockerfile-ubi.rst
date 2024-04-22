Dockerfile-ubi
====================

-  Visualization of the multi-stage Dockerfile.ubi

   .. figure:: ../../assets/dev/dockerfile-ubi-dependency-graph.png
      :alt: query
      :width: 100%
      :align: center

   Made using: https://github.com/patrickhoefler/dockerfilegraph

   Commands to regenerate it:

   .. code:: bash

      dockerfilegraph -o png --legend --dpi 200 --max-label-length 50 -f Dockerfile.ubi

   or in case you want to run it directly with the docker image:
   
   .. code:: bash

      docker run \
         --rm \
         --user "$(id -u):$(id -g)" \
         --workdir /workspace \
         --volume "$(pwd)":/workspace \
         ghcr.io/patrickhoefler/dockerfilegraph:alpine \
         --output png \
         --dpi 200 \
         --max-label-length 50 \
         --legend \
         -f Dockerfile.ubi

   