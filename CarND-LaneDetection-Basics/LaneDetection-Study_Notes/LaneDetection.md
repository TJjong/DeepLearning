#Lane Detection

###Features useful in the identification of lane lines on the road

- color
- shape
- orientation
- position in the image (The position of the camera is fixed)

### Color Selection

RGB - [Red, Green, Blue][0-dark, 255-white]

HSV/HSB - [Hue, Saturation, Brightness]

HSL - [Hue, Saturation,Lightness]

![img](https://github.com/kenshiro-o/CarND-LaneLines-P1/raw/master/docs/hsv_diagram.jpg)![![img](https://github.com/kenshiro-o/CarND-LaneLines-P1/raw/master/docs/hsl_diagram.jpg)HSL Diagram](https://github.com/kenshiro-o/CarND-LaneLines-P1/raw/master/docs/hsl_diagram.jpg)H

### Canny Edge Detection

```
edges = cv2.Canny(gray, low_threshold, high_threshold)
```

In this case, you are applying `Canny` to the image `gray` and your output will be another image called `edges`.`low_threshold` and `high_threshold` are your thresholds for edge detection.

The algorithm will first detect strong edge (strong gradient) pixels above the `high_threshold`, and reject pixels below the `low_threshold`. Next, pixels with values between the `low_threshold` and `high_threshold` will be included as long as they are connected to strong edges. The output `edges` is a binary image with white pixels tracing out the detected edges and black everywhere else. 

As far as a ratio of `low_threshold` to `high_threshold`, [John Canny himself recommended](http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html#steps) a low to high ratio of 1:2 or 1:3.

### Hough Transform

- image space: a line                      Hough space: a point

- image space: two parallel line   Hough space: two points have the same 'm'

- image space: a point                   Hough space:  a line

- image space: two points             Hough space: two intersecting lines

- image space: intersecting point of two intersecting lines  

  Hough space: two points lie on the same line

- As the above transformation is not differentiable when m = 0, we usually use polar coordinates system, where a given point in image space will correspond to a sinusoidal curve in the Hough space.

  ![img](https://cdn-images-1.medium.com/max/800/1*-x1gzNc1nIfHEN1SFjv7-w.png)



