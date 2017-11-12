# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # "Image References"

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 8 steps. 

### Pipeline

1. **Read in image**
   - read_image(image)
2. **Grayscale the image**
   - grayscale(image)
3. **Apply Gaussian smoothing**
   - gaussian_blur(image, kernel_size)
4. **Canny edge detection**
   - canny(image, low_threshold, high_threshold)
5. **Mask edge image** (ignore everything outside of region of interest)
   - region_of_interest(image, vertices)
6. **Draw line segments**
   - hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap)
7. **Combine the line image with original image**
   - weighted_img(processed_image, original_image)

####Further refinement

- **In order to draw a single line on the left and right lanes, I modified the `draw_lines()` function used in `hough_lines()` by `draw_lines_2()` ** 
- **The right side and left side of the lane can be distinguished from the slope of the lane: when `slope > 0` , it is the `right side`; when `slope < 0`, it is the `left side`. **




### 2. Identify potential shortcomings with your current pipeline

- This pipeline is restricted only to straight lanes. Therefore, it can not detect the line correctly around the corner.
- Light change, weather change, road change (when the road is uneven, for example) will also affect the performance of the pipeline greatly.
- When the slope of the line is infinite, an error will be encountered. For example, when dealing with `challange.mp4`, an error was reported, `ValueError: cannot convert float NaN to integer`.


### 3. Suggest possible improvements to your pipeline

Step by step, the pipeline can be improved in the following ways:

1. **Remove the error encountered in the current pipeline.**
2. **Dealing with corner case**
3. **Dealing with road and light change**
4. **Dealing with weather change**

Considering combine traditional `computer vision` techniques with `deep learning` techniques to obtain a robust lane detector.
