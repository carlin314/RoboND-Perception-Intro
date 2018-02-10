#!/usr/bin/python

# Import PCL module
import pcl

# Load Point Cloud file
cloud = pcl.load("./pcd_in/table_scene_lms400.pcd")

#Start by creating a filter object:
fil = cloud.make_statistical_outlier_filter()

# Set the number of neighboring points to analyze for any given point
fil.set_mean_k(50)

# Set threshold scale factor
# Any point with a mean distance larger than global
# (mean distance+x*std_dev) will be considered outlier
fil.set_std_dev_mul_thresh(1.0)

# Save inliers
pcl.save(fil.filter(), "./pcd_out/table_scene_lms400_inliers.pcd")

# Save outliers
fil.set_negative(True)
pcl.save(fil.filter(), "./pcd_out/table_scene_lms400_outliers.pcd")