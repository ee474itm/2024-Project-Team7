# **WARNY**: AI Chatbot for Automotive Dashboard Warning Lights

## Description

Chatbot based on a large language model and image detection model that detects the warning light from uploaded images and provides causes and solutions for single and multiple warning lights on a vehicle dashboard

## Getting Started

### Dependencies

* Linux server with GPU supports CUDA

### Installing

1. Create your own virtual environment by `Conda`, `Pyenv`, or `Venv`
2. Run the command for installing python libraries

```
pip install -r requirements.txt
```


### Executing program

* You can run the sample with following command

```
python run.py
```

* Then, you can check the response and image in the terminal, and speech files would be created in `./speech`.

## Authors

[@Jaesung](https://github.com/jaesung8)

## Acknowledgments

* [Llama3](https://github.com/meta-llama/llama3)
* [YOLOv8](https://github.com/ultralytics/ultralytics/)
* [Bark](https://github.com/suno-ai/bark)
* [Langchain](https://github.com/langchain-ai/langchain)
* [Warning Light by SOCAR](https://xn--289aqc922a.com/)
* [roboflow](https://universe.roboflow.com/test-ogkz5/car-dashboard-icons)
