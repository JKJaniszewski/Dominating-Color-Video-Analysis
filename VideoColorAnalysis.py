import cv2
import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def rescaleFrame(frame, scale = 0.2):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation= cv.INTER_AREA)

def get_dominating_color(frame, num_clusters=3):
    # Reshape the frame to a list of pixels
    pixels = frame.reshape((-1, 3))

    # Convert to float32 for k-means clustering
    pixels = np.float32(pixels)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pixels)

    # Get the dominant color (center of the cluster with the most members)
    dominating_color = kmeans.cluster_centers_.astype(int)[np.argmax(np.bincount(kmeans.labels_))]

    return dominating_color

def display_dominating_color(frame, clusters = 3):
    dominating_color = get_dominating_color(frame, clusters)

    dominating_color_image = np.zeros((1,1,3), dtype=np.uint8)
    dominating_color_image[:,:] = dominating_color
    dominating_color_image_rescaled = rescaleFrame(dominating_color_image, 100)

    cv.namedWindow("Dominating Color", cv.WINDOW_NORMAL)
    cv.resizeWindow("Dominating Color", 100, 100)
    cv.imshow("Dominating Color", dominating_color_image_rescaled)

def get_perceived_main_color(frame, num_clusters=3):
    # Reshape the frame to a list of pixels
    pixels = frame.reshape((-1, 3))

    # Convert to float32 for k-means clustering
    pixels = np.float32(pixels)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to 8-bit values
    centers = np.uint8(centers)

    # Find the most frequent color
    unique_labels, counts = np.unique(labels, return_counts=True)
    main_color_label = unique_labels[np.argmax(counts)]
    main_color = centers[main_color_label]

    return main_color

def display_perceived_main_color(frame, clusters = 3):
    dominating_color = get_perceived_main_color(frame, clusters)

    perceived_main_color_image = np.zeros((1,1,3), dtype=np.uint8)
    perceived_main_color_image [:,:] = dominating_color
    perceived_main_color_image_rescaled = rescaleFrame(perceived_main_color_image, 100)

    cv.namedWindow("Perceived Main Color", cv.WINDOW_NORMAL)
    cv.resizeWindow("Perceived Main Color", 100, 100)
    cv.imshow("Perceived Main Color", perceived_main_color_image_rescaled)

def get_dominating_saturated_color(frame, num_clusters=3):
    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Extract the saturation channel
    saturation_channel = hsv_frame[:, :, 1]

    # Reshape the saturation channel to a list of pixels
    pixels = saturation_channel.reshape((-1, 1))

    # Convert to float32 for k-means clustering
    pixels = np.float32(pixels)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pixels)

    # Get the dominant saturation level
    dominating_saturation = kmeans.cluster_centers_.astype(int)[np.argmax(np.bincount(kmeans.labels_))]

    # Find pixels with the dominant saturation level
    dominating_pixels = (saturation_channel == dominating_saturation).astype(np.uint8) * 255

    # Find the corresponding pixels in the original frame
    dominating_color = np.array(cv2.mean(frame, mask=dominating_pixels)[:3]).astype(int)

    return dominating_color

def display_dominating_saturated_main_color(color):

    dominating_saturated_color_image = np.zeros((100,800,3), dtype=np.uint8)
    dominating_saturated_color_image [:,:] = color

    cv.namedWindow("Dominating Saturated Color", cv.WINDOW_NORMAL)
    cv.resizeWindow("Dominating Saturated Color", 800, 100)
    cv.imshow("Dominating Saturated Color", dominating_saturated_color_image)


def video_color_analysis(video_path, output_texture_path):
    capture = cv.VideoCapture(video_path)
    texture = np.zeros((100,1,3), dtype=np.uint8)
    all_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened():
        print(f"Error: Unable to open the video file at {video_path}")
        return

    frame_count = -1

    while True:
        ret, frame = capture.read()

        if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
            break


        cv.imshow('og video rescaled', frame)

        frame_count += 1

        # Skip frames and only process every 30 seconds
        if frame_count % (24) != 0:
             continue

        rescaled_frame = rescaleFrame(frame, 0.025)
        current_color = get_dominating_saturated_color(rescaled_frame, 3)
        texture = generate_and_display_live_texture(texture, current_color)
        print(str(frame_count) + ' out of ' + str(all_frame_count))
        display_dominating_saturated_main_color(current_color)
        cv.namedWindow("Live Texture", cv.WINDOW_NORMAL)
        cv.resizeWindow("Live Texture", texture.shape[1], 100)
        cv.imshow("Live Texture", texture)

    cv.imwrite(output_texture_path, texture)
    print(f"Texture saved to {output_texture_path}")

    capture.release()
    cv2.destroyAllWindows()

def generate_and_display_live_texture(texture, color):
    new_texture = np.zeros((100, 10, 3), dtype=np.uint8)
    new_texture[:100, :10] = color


    return np.hstack([texture, new_texture])

cv.destroyAllWindows()

# img = cv.imread('frame.png')
# rescaled_img = rescaleFrame(img, scale = 0.1)
#
# cv.imshow('rescaled og img', rescaled_img)
#
# display_dominating_color(rescaled_img)

if __name__ == "__main__":
    video_path = "chlopi.mp4"
    output_texture_path = "output_texture.jpg"

video_color_analysis(video_path, output_texture_path)

cv.waitKey(0)