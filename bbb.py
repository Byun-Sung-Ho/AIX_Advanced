import os
import torch
import librosa
import librosa.display
import numpy as np
import pandas as pd
from torch import cuda
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2ForSequenceClassification, Trainer, TrainingArguments, Wav2Vec2Processor, DataCollatorWithPadding, EvalPrediction
from datasets import Dataset, ClassLabel, Audio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt