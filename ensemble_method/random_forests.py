import pandas as pd
import os


# load data
def random_forests():
    file_path = os.path.dirname(__file__) + '/dataset/melb_data.csv'

    data = pd.read_csv(file_path)

    data = data.dropna()

    y = data.Price

    features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']

    X = data[features]

    from sklearn.model_selection import train_test_split

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

    from sklearn.ensemble import RandomForestRegressor

    from sklearn.metrics import mean_absolute_error

    forests_model = RandomForestRegressor(random_state=1)

    forests_model.fit(train_X, train_y)

    melb_preds = forests_model.predict(val_X)

    print(mean_absolute_error(val_y, melb_preds))


if __name__ == '__main__':
    random_forests()

