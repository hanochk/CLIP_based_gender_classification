# CLIP_based_gender_classification
Based on dataset with cropped faces by Kaggle :https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset
Zero-shot : retrieved top-1 vs. best similar female or male. ACC=96.7 and with additional 1 iteration of fine tuning, of the image encoder only, ACC reaches 96.9 
Preliminary :
pip install git+https://github.com/openai/CLIP.git
