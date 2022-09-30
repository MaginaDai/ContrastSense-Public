#!/bin/bash

for a in 1024
do
  for b in 0.3
  do
    name="queue_${a}/grid_queue_${a}_lambda_${b}"
    python results_analysis.py -name "${name}_HASC" "${name}_HHAR" "${name}_MotionSense" "${name}_Shoaib"
  done
done

# python results_analysis.py -name "${name}_HASC" "${name}_HHAR" "${name}_MotionSense" "${name}_Shoaib"