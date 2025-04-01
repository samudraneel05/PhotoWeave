Given a pair of images I want to stitch them to create a panoramic scene. 

Note, one fundamental requirement is that both images need to share some common regions.

Also, the solution need to work even if the pictures have different Scaling or Angle (Perspective) or Spacial position or Capturing devices.

### What is Image Stitching

A technique of combining several overlapping images from
the similar viewpoint into a bigger one without thrashing of
information is known as an image stitching. The most
universally used methods are the Harris corner detection
method and the Scale Invariant Feature Transform (SIFT)
method.


### Image Matching 

Features matching or generally Image Matching, a part of many computer vision applications such as image registration, camera calibration and object recognition, is the task of establishing correspondences between two images of the same scene/object. 

A common approach to image matching consists of detecting a set of interest points - each associated with **image descriptors** from the individual image data. Once the features and their descriptors have been extracted from two or more images, the next step is to establish some preliminary feature matches between these images.


![Imgur](https://imgur.com/XzsXL5T.png)




### Feature Descriptor

A feature descriptor is an algorithm which takes an image and outputs locations (i.e. pixel coordinates) of significant areas in your image. An example of this is a corner detector, which outputs the locations of corners in your image but does not tell you any other information about the features detected.  The **"location"** might also include a number describing the size or scale of the feature. This is because things that look like corners when "zoomed in" may not look like corners when "zoomed out", and so specifying scale information is important. So instead of just using an (x,y) pair as a location in "image space", you might have a triple (x,y,scale) as location in "scale space".

Feature descriptors encode interesting information into a series of numbers and act as a sort of numerical “fingerprint” that can be used to differentiate one feature from another.


Ideally, this information would be invariant under image transformation, so we can find the feature again even if the image is transformed in some way. After detecting interest point we go on to compute a descriptor for every one of them. Descriptors can be categorized into two classes:

**Local Descriptor**: It is a compact representation of a point’s local neighborhood. Local descriptors try to resemble shape and appearance only in a local neighborhood around a point and thus are very suitable for representing it in terms of matching.

**Global Descriptor:** A global descriptor describes the whole image. They are generally not very robust as a change in part of the image may cause it to fail as it will affect the resulting descriptor.


#### Main Feature Descriptor Algorithms

* SIFT(Scale Invariant Feature Transform)
* SURF(Speeded Up Robust Feature)
* BRISK (Binary Robust Invariant Scalable Keypoints)
* BRIEF (Binary Robust Independent Elementary Features)
* ORB(Oriented FAST and Rotated BRIEF)


Methods like SIFT and SURF try to address the limitations of corner detection algorithms. Usually, corner detector algorithms use a fixed size kernel to detect regions of interest (corners) on images. It is easy to see that when we scale an image, this kernel might become too small or too big.

To address this limitation, methods like SIFT uses Difference of Gaussians (DoG). The idea is to apply DoG on differently scaled versions of the same image. It also uses the neighboring pixel information to find and refine key points and corresponding descriptors.


"When you have two identical images, except one is scaled differently than the other, SIFT maximizes the Difference of Gaussians (DoG) in scale and in space to find same key points independently in each image. DoG is basically the difference of the Gaussian blurring of an image with different standard deviation. Every octave, or scale, of the image is blurred with Gaussians with standard deviations of different scaling factors. The differences between adjacent Gaussian-blurred images are calculated as DoG. The process is repeated for each octave of scaled image."

![Imgur](https://imgur.com/9FOpV1N.png)

[Source](https://ai.stanford.edu/~syyeung/cvweb/tutorial2.html)


### SIFT vs Harris Corner Detection for Image Matching


1. The Harris Detector, is rotation-invariant, which means that the detector can still distinguish the corners even if the image is rotated. However, the Harris Detector cannot perform well if the image is scaled differently.

2. Due to the characteristics of SIFT, the key point descriptors are constructed to extract features by finding the precise positioning and main directions of feature points, in different scale spaces. 

The key points extracted by SIFT have scale invariance and rotation.