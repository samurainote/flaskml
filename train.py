
# import libraries
from sklearn.datasets import load_boston
from sklearn.externals import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# 
def train_model_creater():
    # load data and spilt in X, y
    boston = load_boston()
    X, y = boston.data, boston.target
    # modelin
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=3)),
        ("linear", LinearRegression())
        ])
    # fit data into model
    model.fit(X, y)
    joblib.dump(model, "polynomial_regression.pkl")

if __name__ == "__main__":
    train_model_creater()


