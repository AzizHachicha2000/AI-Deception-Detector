#train.py
from data_loader import load_kaggle_dataset
from preprocessor import preprocess_text
from sklearn.model_selection import train_test_split
from Sherlock_Holmes import Sherlock_Holmes
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report,roc_curve, auc
import seaborn as sns
import joblib


df = load_kaggle_dataset(r"C:\Users\GMI\Desktop\Sherlock_Holmes\deceptive-opinion.csv")
df['clean_text'] = df['text'].apply(preprocess_text)


X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42
)


model = Sherlock_Holmes()
model.fit(X_train, y_train)


joblib.dump(model, 'Sherlock_Holmes.pkl')


y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


def plot_performance(y_true, y_pred):
    plt.figure(figsize=(12,4))
    
    
    plt.subplot(121)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion')
    
    
    plt.subplot(122)
    y_proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label='AUC = %0.2f' % auc(fpr, tpr))
    plt.plot([0,1], [0,1], 'k--')
    plt.title('Courbe ROC')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('performance.png')
    plt.show()

plot_performance(y_test, y_pred)