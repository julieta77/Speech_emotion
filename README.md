
# Welcome to my speech emotion project


![image](https://th.bing.com/th/id/OIP.Db_DMVy2vIJFXc7KHvR1rAHaFO?pid=ImgDet&rs=1) 

## Description: 

In this project we created a detection model of the three most important emotions: "Happy", "Sad" and "Angry", through voice analysis. This model has an accuracy of 82.8% and the data used for its training was extracted from the database [RAVDESS](https://zenodo.org/record/1188976#.ZFAzPNqZO3D). In addition, we have developed an application in [Streamlit](https://julieta77-appstreamlit-app-bzcyfs.streamlit.app/) so that the user can upload an audio file and get the prediction of the detected emotion.


##  To use the model, follow these commands:

```bash
!git clone https://github.com/julieta77/Speech_emotion
!cd Speech_emotion
 
```


##  After this, in a jupyter or .py file, add the following code: 

``` Python
from joblib import load 
model = load('speech_emotion.joblib')  
```


## Finally, you will be able to use the speech_emotion in your future projects.