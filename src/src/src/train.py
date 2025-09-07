import argparse
from pathlib import Path
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score

from src.preprocess import load_data, ensure_dir

def main():
    parser = argparse.ArgumentParser(description="Train LightGBM on Medicare Part D sample")
    parser.add_argument("--input", default="data/sample/partd_spending_sample.csv")
    parser.add_argument("--target", default="Avg_Spnd_Per_Dsg_Unt_Wghtd_2022")
    parser.add_argument("--out", default="models/lgbm.pkl")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data(args.input, args.target)

    model = LGBMRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    print(f"RMSE: {rmse:.4f}  |  R²: {r2:.4f}")

    ensure_dir(Path(args.out).parent)
    joblib.dump(model, args.out)
    print(f"Saved model to {args.out}")

if __name__ == "__main__":
    main()
