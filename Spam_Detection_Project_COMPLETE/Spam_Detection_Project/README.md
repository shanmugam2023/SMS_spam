# SMS Spam Detection Using Machine Learning

A beginner-level ML project that classifies SMS messages as spam or ham.

## Files
- `app.py` — Streamlit user interface and prediction
- `train_model.py` — model training and evaluation
- `data/sms_dataset.csv` — 100 sample messages (50 ham, 50 spam)
- `PROJECT_REPORT.md` — project report
- `PROJECT_PLAN.md` — project management plan
- `requirements.txt` — Python dependencies

## Run
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
Then open the local URL shown in the terminal (usually http://localhost:8501).

## GitHub upload
```bash
git init
git add .
git commit -m "Add SMS spam detection ML project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```
Replace the URL with your own GitHub repository URL.

## Method
TF-IDF converts message text into numeric features. A Multinomial Naive Bayes classifier learns from the labeled examples and predicts new messages.

## Dataset limitation
The included 100-message dataset is a small teaching sample, not a production benchmark. Accuracy can vary. Use a larger, properly licensed dataset for meaningful evaluation.
