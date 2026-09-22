PROJECT: Explainable NLP-Based Detection and Credibility Assessment of Online News Content Using Linguistic Evidence

VS CODE STRUCTURE
explainable_news_credibility/
  app.py
  train_model.py
  requirements.txt
  data/
    WELFake_Dataset.csv        <-- PUT DATASET HERE
  models/                      <-- created after training
  templates/
    index.html
  static/
    style.css
    script.js

DATASET
Recommended primary dataset: WELFake_Dataset.csv.
Expected columns: title, text, label.
WELFake label convention used here: 0 = fake, 1 = real.

SETUP (Windows)
1. Open this folder in VS Code.
2. Open Terminal.
3. Create environment:
   python -m venv venv
4. Activate:
   venv\Scripts\activate
5. Install:
   pip install -r requirements.txt
6. Put WELFake_Dataset.csv inside the data folder.
7. Train:
   python train_model.py
8. Start:
   python app.py
9. Open:
   http://127.0.0.1:5000

The training script creates:
models/tfidf_vectorizer.joblib
models/news_classifier.joblib
models/model_metadata.json

The webpage accepts headline + article text and returns:
- predicted class
- credibility level
- confidence
- influential words/phrases
- basic linguistic indicators
- disclaimer that this is a text-based model assessment, not factual verification.

OPTIONAL DATASET FORMAT
If using ISOT instead, place Fake.csv and True.csv in data/ and run:
python train_model.py --dataset isot

WELFake SOURCE
https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
or
https://zenodo.org/records/4561253

NOTE
Do not describe the output as proof that an article is factually true or false. The model learns patterns from the labeled dataset.
