import dill
import pandas as pd
import datetime
import warnings

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

def filter_data(df):
    df = df.copy()
    columns_to_drop = ['device_model',
                       'client_id',
                       'visit_date',
                       'visit_time',
                       'visit_number']
    cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    return df.drop(cols_to_drop, axis=1)

def filter_columns(df):
    df = df.copy()
    columns_to_fill_unknown = ['utm_keyword', 'utm_adcontent', 'utm_campaign']
    columns_to_fill_other = ['device_brand']
    columns_to_drop_na = ['utm_source']

    for col in columns_to_fill_unknown:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
    for col in columns_to_fill_other:
         if col in df.columns:
            df[col] = df[col].fillna('other')
    for col in columns_to_drop_na:
        if col in df.columns:
            df = df.dropna(subset=[col])
    return df

def create_new_features(df):
    df = df.copy()
    android_brands = [
        'Huawei', 'Samsung', 'Xiaomi', 'Lenovo', 'Vivo', 'Meizu', 'OnePlus',
        'Realme', 'OPPO', 'itel', 'Tecno', 'Infinix', 'ZTE', 'Wiko', 'Google',
        'Micromax', 'Blackview', 'Oukitel', 'Motorola', 'HOMTOM', 'Cubot',
        'DOOGEE', 'DEXP', 'Neffos', 'Hisense', 'Umidigi', 'Leagoo', 'Ulefone',
        'HTC', 'LeEco', 'Prestigio', 'POCO', 'TCL', 'Inoi', 'Nomu', 'Vernee',
        'BLU', 'Haier', 'Coolpad', 'Gionee', 'Digma', 'Archos', 'Black Fox',
        'Lava', 'Condor', 'Honor', 'Redmi', 'InFocus', 'Sharp', 'BQ', 'Philips',
        'Nokia', 'Alcatel', 'Sony', 'Wileyfox', 'Fairphone', 'Vsmart', 'AGM',
        'UMI', 'ThL', 'Xgody', 'Mobiistar', 'Evertek', 'Vertex', 'Jiayu',
        'Leegoog', 'Smartisan', 'Chuwi', 'Walton', 'Xiaolajiao', 'myPhone'
    ]

    pc_brands = [
        'Dell', 'HP', 'Acer', 'MSI', 'Toshiba', 'Fujitsu', 'Panasonic',
        'Medion', 'Clevo', 'Vaio', 'IBM', 'Framework', 'System76', 'Gateway',
        'Digma', 'Teclast', 'Chuwi', 'Dell', 'Fujitsu', 'Medion', 'Clevo',
        'Vaio', 'IBM', 'Framework', 'System76', 'Gateway', 'Digma', 'Teclast'
    ]

    def classify_device(row):
        brand = row.get('device_brand', 'other')

        if brand in pc_brands:
            return 'PC'
        if brand == 'Apple':
            return 'iOS'
        elif brand in android_brands:
            return 'Android'
        else:
            return 'other'

    df['device_os'] = df.apply(classify_device, axis=1)
    return df

def filter_country_data(df):
    df = df.copy()
    if 'geo_country' in df.columns:
        df = df[df['geo_country'] == 'Russia']
    return df

def fill_top50(df):
    df = df.copy()
    if 'utm_keyword' in df.columns:
        top50_keyword = df['utm_keyword'].value_counts(ascending=False).head(50).index
        df['utm_keyword'] = df['utm_keyword'].apply(lambda x: x if x in top50_keyword else 'other')
    if 'device_screen_resolution' in df.columns:
        top50_resolution = df['device_screen_resolution'].value_counts().head(50).index
        df['device_screen_resolution'] = df['device_screen_resolution'].apply(lambda x: x if x in top50_resolution else 'other')
    return df

def df_merge(df_sessions, df_hits):
    df_sessions = df_sessions.copy()
    df_hits = df_hits.copy()

    unnecessary_columns_hits = ['hit_date', 'hit_time', 'hit_number', 'hit_type',
                                'hit_referer', 'hit_page_path', 'event_category',
                                'event_label', 'event_value']
    cols_to_drop_hits = [col for col in unnecessary_columns_hits if col in df_hits.columns]
    df_hits = df_hits.drop(cols_to_drop_hits, axis=1)

    if 'session_id' in df_hits.columns:
        df_hits_unique = df_hits.drop_duplicates(subset='session_id', keep='first')

        merged_df = pd.merge(df_sessions, df_hits_unique, how='left', on='session_id')

        if 'session_id' in merged_df.columns:
            merged_df = merged_df.drop('session_id', axis=1)
    else:
         print("Предупреждение: 'session_id' не найден в df_hits.")
         merged_df = df_sessions
         missing_cols = [col for col in df_hits.columns if col not in merged_df.columns and col != 'session_id']
         for col in missing_cols:
             merged_df[col] = pd.NA

    return merged_df

def create_new_feautures_after_merge(df):
    df = df.copy()
    premium_brands = ['Apple', 'Sony', 'Google']
    big_cities = ['Moscow', 'Saint Petersburg', 'Yekaterinburg', 'Krasnodar', 'Nizhny Novgorod']
    target_list = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
                   'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
                   'sub_submit_success', 'sub_car_request_submit_click']

    if 'geo_city' in df.columns:
        df['big_city'] = df['geo_city'].apply(lambda x: 1 if x in big_cities else 0)

    if 'device_brand' in df.columns:
         df['is_premium'] = df['device_brand'].apply(lambda x: 1 if x in premium_brands else 0)

    if 'event_action' in df.columns:
        # Заполняем NaN перед применением lambda, чтобы избежать ошибок
        df['event_action'] = df['event_action'].fillna('no_event')
        df['target'] = df['event_action'].apply(lambda x: 1 if x in target_list else 0)
        df = df.drop(['event_action'], axis=1)

    return df

def features_for_predict(df):
    df = df.copy()
    premium_brands = ['Apple', 'Sony', 'Google']
    big_cities = ['Moscow', 'Saint Petersburg', 'Yekaterinburg', 'Krasnodar', 'Nizhny Novgorod']

    if 'geo_city' in df.columns:
        df['big_city'] = df['geo_city'].apply(lambda x: 1 if x in big_cities else 0)

    if 'device_brand' in df.columns:
        df['is_premium'] = df['device_brand'].apply(lambda x: 1 if x in premium_brands else 0)

    return df

def main():
    print('Пайплайн по предсказанию совершения целевого действия.')

    print('Загрузка данных...')

    df_sessions = pd.read_csv('ga_sessions.csv', low_memory=False)
    df_hits = pd.read_csv('ga_hits-001.csv', low_memory=False)
    print('Данные загружены успешно.')

    pipeline_custom_preprocessing = Pipeline(steps=[
        ('filter_data', FunctionTransformer(filter_data)),
        ('filter_columns', FunctionTransformer(filter_columns)),
        ('create_new_features', FunctionTransformer(create_new_features)),
        ('filter_country_data', FunctionTransformer(filter_country_data)),
        ('fill_top50', FunctionTransformer(fill_top50)),
        # Передаем df_hits как аргумент для шага слияния
        ('merge_hits', FunctionTransformer(df_merge, kw_args={'df_hits': df_hits})),
        ('create_target', FunctionTransformer(create_new_feautures_after_merge)),
    ])

    print('Выполнение пользовательской предобработки данных...')
    processed_df = pipeline_custom_preprocessing.fit_transform(df_sessions)
    print('Пользовательская предобработка завершена.')

    y = processed_df['target']
    X_processed = processed_df.drop(['target'], axis=1)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_selector = make_column_selector(dtype_include=['int64', 'float64', 'int32', 'float32'])
    categorical_selector = make_column_selector(dtype_include=object)

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_selector),
        ('categorical', categorical_transformer, categorical_selector)
    ])

    models = {
        'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42, max_iter=100),
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, max_depth=5),
        'MLP Classifier': MLPClassifier(activation='logistic', hidden_layer_sizes=(32, 16), random_state=42, max_iter=500) # Уменьшены слои и итерации
    }

    best_score = -1.0
    best_model_name = None
    best_pipeline_config = None

    print('Обучение и оценка моделей с кросс-валидацией...')
    # Подавляем ConvergenceWarning от LogisticRegression и MLP
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    for name, model in models.items():
        cv_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        score = cross_val_score(cv_pipeline, X_processed, y, cv=3, scoring='roc_auc', n_jobs=-1)
        mean_score = score.mean()
        std_score = score.std()
        print(f'Model: {name}, ROC AUC Mean: {mean_score:.4f}, ROC AUC Std: {std_score:.4f}')

        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name
            best_pipeline_config = cv_pipeline


    warnings.filterwarnings('default', category=ConvergenceWarning)

    print(f'\nЛучшая модель: {best_model_name}, ROC AUC: {best_score:.4f}')

    print('Обучение лучшего пайплайна на всех обработанных данных...')
    best_pipeline_config.fit(X_processed, y)
    print('Обучение завершено.')

    print('Сохранение лучшего пайплайна и метаданных...')
    data_to_save = {
        'model_pipeline': best_pipeline_config,

        'metadata': {
            'name': 'Предсказание совершения пользователем целевого действия',
            'author': 'Georgii Lozovoi',
            'version': 1,
            'date': datetime.datetime.now(),
            'model_type': best_model_name,
            'roc_auc': best_score,
            'expected_columns': X_processed.columns.tolist()
        }
    }
    with open('models/ga_pipeline_final.pkl', 'wb') as f:
        dill.dump(data_to_save, f)

    print('Пайплайн успешно сохранен в файл ga_pipeline_final.pkl!')

    print('Сохранение пайплайна для обработки входных данных.')
    pipeline_custom_preprocessing_for_prediction = Pipeline(steps=[
        ('features_for_predict', FunctionTransformer(features_for_predict)),
        ('classify_device', FunctionTransformer(create_new_features))
    ])
    with open('models/custom_preprocessor.pkl', 'wb') as f:
        dill.dump(pipeline_custom_preprocessing_for_prediction, f)

    print('Работа завершена!')

if __name__ == '__main__':
    main()