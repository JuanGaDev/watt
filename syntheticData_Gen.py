import argparse
import pandas as pd
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata
from copulas.multivariate import GaussianMultivariate
from sklearn.preprocessing import LabelEncoder

MODEL_MAP = {
    'ctgan': TVAESynthesizer,
    'copulas': GaussianMultivariate
}

def load_data(input_path):
    return pd.read_csv(input_path)

def preprocess_for_copulas(data):
    data = data.copy()
    # Imputar NaN con el valor más frecuente en cada columna
    for col in data.columns:
        if data[col].isnull().any():
            if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                data[col] = data[col].fillna(data[col].mode()[0])
            else:
                data[col] = data[col].fillna(data[col].median())
    # Codificar columnas categóricas
    for col in data.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
    return data

def train_model(model_name, data):
    if model_name == 'ctgan':
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=data)
        model = CTGANSynthesizer(metadata)
        model.fit(data)
    elif model_name == 'copulas':
        data = preprocess_for_copulas(data)
        model = GaussianMultivariate()
        model.fit(data)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")
    return model

def generate_synthetic(model, num_samples):
    return model.sample(num_samples)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='dataset_test.csv')
    parser.add_argument('--output', type=str, default='synthetic_data.csv')
    parser.add_argument('--model', type=str, choices=['ctgan', 'copulas'], default='ctgan')
    parser.add_argument('--num-samples', type=int, default=3000)
    args = parser.parse_args()

    data = load_data(args.input)
    model = train_model(args.model, data)
    synthetic = generate_synthetic(model, args.num_samples)
    synthetic.to_csv(args.output, index=False)
    print(f"Datos sintéticos guardados en {args.output}")

if __name__ == '__main__':
    main()