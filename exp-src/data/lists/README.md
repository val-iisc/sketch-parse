Instructions regarding usage of `Pose_all_label.txt`:
* Augmented training data: For training our CNN, we used sketchified images. The training sketchified images after augmentation can be downloaded from [this link](https://1drv.ms/u/s!AvBNaER10ndvhb1Cl_Y2THHrbi1D9A). The corresponding ground truth can be downloaded from [this link](https://1drv.ms/u/s!AvBNaER10ndvhb1B7nS3sqvzdOSUTg). All filenames in the lists train_aeroplane_bird.txt, train_bicycle_motorbike.txt, train_bus_car.txt, train_cat_dog_sheep.txt, train_cow_horse.txt  are present in the augmented data provided in the above links.

* In Pascal-VOC dataset, each image may contain more than one object categories. Pascal-VOC dataset also provides bounding boxes of each object. Using this information we have cropped each object, and the number at the end of each file name specifies a cropped image. For eg. 2008_003094_3.jpg is the 3rd crop of the VOC image 2008_003094.jpg

* The pose labels are as follows: 0=North,1=West,2=North-West,3=North-East,4=South,5=East,6=South-East,7=South-West

Instructions regarding usage of files with the format `train_*.txt`:

* Pascal VOC filenames are suffixed to signify the 13 (and one original) augmentations. Please find additional details about these augmentations in the paper.
