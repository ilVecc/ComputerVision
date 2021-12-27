import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import regionprops
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic

# load the image and convert it to a floating point data type
image = io.imread("../imgs/river/river_1.jpg ").astype(float)

# apply SLIC and extract (approximately) the supplied number of segments
segments = slic(image, n_segments=200, sigma=5)
centroids = [props.centroid for props in regionprops(segments)]

plt.imshow(mark_boundaries(image, segments))
plt.axis("off")
plt.show()
