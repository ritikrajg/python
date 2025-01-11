
Running the Object and Sub-Object Detection Project in Google Colab

1. Open Google Colab
2. Create a New Notebook
3. Install Dependencies
In the first code cell, install the necessary libraries by running the following commands:
!pip install opencv-python torch torchvision torchaudio
from google.colab.patches import cv2_imshow
4. Upload Your Files
Use the following code to upload your sample_video.mp4 file:
from google.colab import files
uploaded = files.upload()
5. Copy and Paste the Script
Copy the full script from the object_subobject_detection.py file and paste it into a new code cell in your Colab notebook.

Modify the cv2.imshow() line to cv2_imshow() to avoid errors in Colab.

6. Run the Script
Execute the code cell containing the script by clicking the play button.

The script will process the uploaded video, perform object detection, and display the results within the notebook.

7. Download Output Files
After running the script, download the generated output.json file using:

from google.colab import files
files.download('output.json')
