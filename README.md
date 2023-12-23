# Sign language detection project

The aim of this project is to create a basic classifier which can detect different signs from the American Sign Language alphabet. The classifier is not at all perfect but it is a great example for the use of the Mediapipe library and the Long-Short Term Memory (LSTM) model architecture.

The project is inspired by and based on the sign language detection videos on the Computer Vision Engineer and Nicholas Renotte Youtube channels.  
Computer Vision Engineer: https://www.youtube.com/watch?v=MJCSjXepaAM
Nicholas Renotte: https://www.youtube.com/watch?v=doDUihpj6ro

The project uses the mediapipe library, which comes with a wide range of functionality and a trained model to detect keypoints and connections on hands, face and full body pose.
The project also comes with a trained model named `full_alpha_model` which was trained on 57 * 30 'frames' on captured hand keypoint coordinates per alphabetic character. As we are using coordinates of keypoints on the palm, there are no real frames, or images involved in the data. 

The model is not perfect. Model accuracy can be improoved by collecting more training data, preferably from people who are used to sign language (they might have more articulate hands than my own hands...), and running collection and inference in ambient light settings as the mediapipe keypoint detector fails in extreme light conditions.

# Use of the code

### 1. Clone the repository
Call `git clone https://github.com/xmarkx/sign_language_detection_project.git` at the selected folder path

Change the directory to the project folder
`cd sign_language_detection_project`

### 2. Install requirements
`pip install -r requirements.txt`

### 3. Run main.py
Callable arguments:

__`-d` or `--demo`__: Use the --demo to run demo
Runs the hand keypoint detection demo, no sign language detection.
`python src\main.py -d`

__`-i` or `--run_inference`__: Use the --run_inference flag to run inference
Runs the model inference with sign language detection.
`python src\main.py -i`

If we run the full pipeline, collecting (`-c`) data and training a model (`-t`) on the collected data, the inference is done on this newly trained model automatically.
Otherwise we have the choice to use the default model (full_alpha_model.keras) trained on the 26 alphabet characters by pressing `d`, or choosing a custom model by pressing `c`.


__=== Data collection and model training is needed if using any other arguments than `-d` or `-i` ===__

__`-c` or `--collect`__: Use the --collect flag to run the data collection
`python src\main.py -c`

We can define the different classification classes we want to collect. We can provide multiple action classes to collect, but the classes have to be separated with a space inbetween.
Example: Provide the actions to collect (use space inbetween if multiple actions are provided): a b c d → results in 4 classes to collect: class 'a', 'b', 'c' and 'd'.

When collecting, we can also call the __`--start_folder`__ argument.
Example: `--start_folder 10` → Defines the starting folder number to be 10 at data collection. Useful if we collected 10 samples and want to collect more without overwriting the existing 10 samples.

__`-t` or `--train`__: Use the --train flag to run the training
`python src\main.py -t`

If data collection is also done in the same run, we have the choice to train the model on the newly collected data by inputting 'c'. The model name needs to also be defined.
By inputting 'd' we can train a model on existing data found in the MP_Data folder. The model name needs to also be defined.

When training, we can also call the __`--epochs`__ argument.
`--epochs` → Defines the number of training epochs.

__`-e` or `--evaluate`__: Use the --evaluate to evaluate a specific model
`python src\main.py -e`

Evaluates the selected model on the alphabet characters. In case of new action classes are collected the Collector.actions attribute has to be updated in the code in collector.py.
