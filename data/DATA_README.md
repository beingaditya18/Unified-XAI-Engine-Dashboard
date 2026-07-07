# UCI Adult Income Census Data Catalog

This catalog outlines the details, schema, and preprocessing transformations applied to the dataset in this repository.

## Data Provenance & Description

- **Source:** [UCI Machine Learning Repository - Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **Donor:** Ronny Kohavi and Barry Becker (Silicon Graphics, 1996).
- **Target:** Classify whether a person makes more than $50K a year based on census demographics.
- **Licensing:** Public Domain / Creative Commons (CC0).
- **Format:** Tabular CSV files.

## Dataset Files

1. **`adult_income.csv`**: Raw dataset extracted from the census database containing training and validation instances.
2. **`adult_test.csv`**: A holdout validation cohort split ($N = 9,633$) representing test instances used for fairness auditing and model divergence testing.
3. **`processed_test.csv`**: Preprocessed and label-encoded holdout validation subset matching the feature space fed to model architectures.

## Attribute Metadata (Schema)

The dataset contains the following attributes:

| Column Name | Data Type | Description / Values |
| --- | --- | --- |
| `age` | Integer | Continuous variable representing age. |
| `workclass` | Categorical | Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. |
| `fnlwgt` | Integer | Continuous final weight value. |
| `education` | Categorical | Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. |
| `education-num` | Integer | Continuous representation of education levels. |
| `marital-status` | Categorical | Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. |
| `occupation` | Categorical | Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. |
| `relationship` | Categorical | Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. |
| `race` | Categorical | White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. |
| `sex` | Categorical | Female, Male. |
| `capital-gain` | Integer | Continuous capital gain statistic. |
| `capital-loss` | Integer | Continuous capital loss statistic. |
| `hours-per-week` | Integer | Continuous work hours per week. |
| `native-country` | Categorical | United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands. |
| `income` | Binary (Target) | `<=50K` or `>50K`. |

## Preprocessing Steps

The preprocessing pipeline (`src/data/data_pipeline.py`) performs the following cleanups:
1. **Handling Missing Values:** Missing entries marked as `?` in categorical features (`workclass`, `occupation`, `native-country`) are filled using the mode value of their respective columns.
2. **Imputing Numerical NaNs:** Continuous features containing empty/nan blocks are imputed using the column's median value.
3. **Encoding:** Categorical labels are transformed into serialized integers via `LabelEncoder` to support standard scikit-learn models.
