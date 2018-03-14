Instructions regarding usage of `Pose_all_label.txt`:

* In Pascal-VOC dataset, each image may contain more than one object categories. Pascal-VOC dataset also provides bounding boxes of each object. Using this information we have cropped each object, and the number at the end of each file name specifies a cropped image. For eg. 2008_003094_3.jpg is the 3rd crop of the VOC image 2008_003094.jpg

* The pose labels are as follows: 0=North,1=West,2=North-West,3=North-East,4=South,5=East,6=South-East,7=South-West

Instructions regarding usage of files with the format `train_*.txt`:

* Pascal VOC filenames are suffixed to signify the 13 (and one original) augmentations. Please find additional details about these augmentations in the paper.
