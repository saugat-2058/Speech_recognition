from decode_mod import Model,DecodeModel,SetLogLevel
import os

def model_load(model_path):
    if not os.path.exists(model_path):  # if model path doesnot match exit
        exit(1)
    model = Model(model_path)   
    return model

def recognize(sample_rate,model):
    return DecodeModel(model,sample_rate)

def logs(level):
    SetLogLevel(level)


    