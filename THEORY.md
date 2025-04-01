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



#### Get Keypoints and Descriptors. 

For example SIFT is both rotation as well as scale invariant.  SIFT provides key points and keypoint descriptors where keypoint descriptor describes the keypoint at a selected scale and rotation with image gradients.
Directly find [keypoints and descriptors](https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html) in a single step with the function, `sift.detectAndCompute()`.

#### What does sift.detectAndCompute() return

```py
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray,None)

```

### In above `keypoints` will be a list of keypoints and `descriptors` is a numpy array of shape (Number of Keypoints)×128


### Why do we use keypoint descriptors?

Very good explanation from [here](https://dsp.stackexchange.com/questions/10423/why-do-we-use-keypoint-descriptors)


One important thing to understand is that after extracting the keypoints, you only obtain information about their position, and sometimes their coverage area (usually approximated by a circle or ellipse) in the image. While the information about keypoint position might sometimes be useful, it does not say much about the keypoints themselves.

Depending on the algorithm used to extract keypoint (SIFT, Harris corners, MSER), you will know some general characteristics of the extracted keypoints (e.g. they are centered around blobs, edges, prominent corners...) but you will not know how different or similar one keypoint is to the other.

Here's two simple examples where only the position and keypoint area will not help us:

- If you have an image A (of a bear on a white background), and another image B, exact copy of A but translated for a few pixels: the extracted keypoints will be the same (on the same part of that bear). Those two images should be recognized as same, or similar.

But, if the only information we have is their position, and that changed because of the translation, you can not compare the images.

- If you have an image A (let's say, of a duck this time), and another image B, exactly the same duck as in A except twice the size: the extracted keypoints will be the same (same parts of the duck). Those are also same (similar) images.

But all their sizes (areas) will be different: all the keypoints from the image B will be twice the size of those from image A.

So, **here come descriptors:** they are the way to compare the keypoints. They summarize, in vector format (of constant length) some characteristics about the keypoints. For example, it could be their intensity in the direction of their most pronounced orientation. It's assigning a numerical description to the area of the image the keypoint refers to.

Some important things for descriptors are:

- **they should be independent of keypoint position**

If the same keypoint is extracted at different positions (e.g. because of translation) the descriptor should be the same.

- **they should be robust against image transformations**

Some examples are changes of contrast (e.g. image of the same place during a sunny and cloudy day) and changes of perspective (image of a building from center-right and center-left, we would still like to recognize it as a same building).

Of course, no descriptor is completely robust against all transformations (nor against any single one if it is strong, e.g. big change in perspective).

Different descriptors are designed to be robust against different transformations which is sometimes opposed to the speed it takes to calculate them.

- **they should be scale independent**

The descriptors should take scale in to account. If the "prominent" part of the one keypoint is a vertical line of 10px (inside a circular area with radius of 8px), and the prominent part of another a vertical line of 5px (inside a circular area with radius of 4px) -- these keypoints should be assigned similar descriptors.


---


SURF (Speeded-Up Robust Features)

Even though SIFT works well it performs intensive operations which are time consuming. SURF was introduced to have all the advantages of SIFT with reduced processing time.

ORB

ORB is an efficient open source alternative to SIFT and SURF. Even though it computes less key points when compared to SIFT and SURF yet they are effective. It uses FAST and BRIEF techniques to detect the key points and compute the image descriptors respectively.