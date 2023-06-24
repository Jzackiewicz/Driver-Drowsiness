
# Driver drowsiness detection program

It is a university project focused on making detections whether the driver is drowsy based on the camera output. There has been done two approaches towards this problem - the conventional one and one using machine learning. \
Project done fully in Python 3.9 except for two functions made in MATLAB.



## Conventional approach
  Using MediaPipe solutions program gets face and pose landmarks of a user. Particular landmarks' coordinates are used to calculate distances and ratios. Detections are made based on those ratios and thresholds set by the user or via thresholds.txt file.


### How to run and use
#### In order to run the program you need to type in bash
```bash
python Conv/main.py
```
in the right directory. 

In case of lacking packages remember to install the following:
```bash
pip install opencv-python
pip install mediapipe
pip install numpy
pip install imutils
```

If you want to run a video file instead of capturing webcam type video file directory in "..." in main.py:
```python
if __name__ == '__main__':
    video_dir = "..."
    main(video_dir)
``` 

#### To overlay the following points on the shown image, you can use keyboard:

• 1 - drawing the FaceMesh grid\
• 2 - drawing the functionality for detecting eye closure\
• 3 - drawing the functionality for determining gaze direction\
• 4 - drawing the functionality for detecting mouth closure\
• 5 - drawing the functionality for determining face direction\
• 6 - drawing the Pose grid\
• 7 - drawing the functionality for measuring the distance between hands and head\
• Backspace - removing all drawn indicators



#### To configure new detection thresholds, you can use the following buttons:

• C - configuring the detection thresholds for individual states\
• S - saving the set thresholds to a text file\
• For indicators that use more than one threshold:\
&ensp;&ensp;•U - setting the upper threshold\
&ensp;&ensp;•D - setting the lower threshold\
&ensp;&ensp;•L - setting the left threshold\
&ensp;&ensp;•R - setting the right threshold

## Machine Learning approach
In this case face landmarks obtained with MediaPipe solutions are used to create dataset which is then used to train an AI model. Model is then used to make detections in terms of drowsiness.
### How to run and use
#### In order to make detections you need to type in bash
```bash
python ML/main.py
```
in the right directory.

In case of lacking packages remember to install the following:
```bash
pip install opencv-python
pip install imutils
pip install mediapipe
pip install numpy
pip install pandas
```

If you want to run a video file instead of capturing webcam type video file directory in "..." in main.py:
```python
if __name__ == '__main__':
    models = ["model1", "model2", "model3", ...]
    video_dir = "..."

    main(video_dir, models[0])
```
#### In order to train new model you need to run following programs:
```bash
python exporting.py
python training.py
```
Additionally, remember to change CLASS_NAME according to your data.


Optionally you can run 
```bash
python data_augmentation.py
```
in the first place to expand your dataset.
## License
Used datasets:\
[• dataset 1.](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset) \
[• dataset 2.](https://www.kaggle.com/datasets/rakibuleceruet/drowsiness-prediction-dataset) \
[• dataset 3.](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) - [Research paper](https://doi.org/10.1007/978-981-33-6893-4_6)


