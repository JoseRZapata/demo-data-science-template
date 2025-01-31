# Demo data science template

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![CI](https://github.com/JoseRZapata/demo-data-science-template/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/JoseRZapata/demo-data-science-template/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JoseRZapata/demo-data-science-template/graph/badge.svg?token=PpCcK9jKy9)](https://codecov.io/gh/JoseRZapata/demo-data-science-template)
---

This demo of a data science project is created using the template from [@JoseRZapata]'s [data science project template] which have all the necessary tools for experiment, development, testing, and deployment data science From notebooks to production.

> [!WARNING]
> 🚧 Work in progress 🚧, This is a demo project, It is only for educational purposes.

## 🗃️ Project structure

```bash
.
├── codecov.yml                         # configuration for codecov
├── .code_quality
│   ├── mypy.ini                        # mypy configuration
│   └── ruff.toml                       # ruff configuration
├── data
│   ├── 01_raw                          # raw immutable data
│   ├── 02_intermediate                 # typed data
│   ├── 03_primary                      # domain model data
│   ├── 04_feature                      # model features
│   ├── 05_model_input                  # often called 'master tables'
│   ├── 06_models                       # serialized models
│   ├── 07_model_output                 # data generated by model runs
│   ├── 08_reporting                    # reports, results, etc
│   └── README.md                       # description of the data structure
├── docs                                # documentation for your project
├── .editorconfig                       # editor configuration
├── .github                             # github configuration
│   ├── dependabot.md                   # github action to update dependencies
│   ├── pull_request_template.md        # template for pull requests
│   └── workflows                       # github actions workflows
│       ├── ci.yml                      # run continuous integration (tests, pre-commit, etc.)
│       ├── dependency_review.yml       # review dependencies
│       ├── docs.yml                    # build documentation (mkdocs)
│       └── pre-commit_autoupdate.yml   # update pre-commit hooks
├── .gitignore                          # files to ignore in git
├── Makefile                            # useful commands to setup environment, run tests, etc.
├── models                              # store final models
├── notebooks
│   ├── 1-data                          # data extraction and cleaning
│   ├── 2-exploration                   # exploratory data analysis (EDA)
│   ├── 3-analysis                      # Statistical analysis, hypothesis testing.
│   ├── 4-feat_eng                      # feature engineering (creation, selection, and transformation.)
│   ├── 5-models                        # model training, evaluation, and hyperparameter tuning.
│   ├── 6-interpretation                # model interpretation
│   ├── 7-deploy                        # model packaging, deployment strategies.
│   ├── 8-reports                       # story telling, summaries and analysis conclusions.
│   ├── notebook_template.ipynb         # template for notebooks
│   └── README.md                       # information about the notebooks
├── .pre-commit-config.yaml             # configuration for pre-commit hooks
├── pyproject.toml                      # dependencies for the python project
├── README.md                           # description of your project
├── src                                 # source code for use in this project
│   ├── libs                            # custom python scripts
│   │   ├── data_etl                    # data extraction, transformation, and loading  
│   │   ├── data_validation             # data validation  
│   │   ├── feat_cleaning               # feature engineering data cleaning
│   │   ├── feat_encoding               # feature engineering encoding
│   │   ├── feat_imputation             # feature engineering imputation    
│   │   ├── feat_new_features           # feature engineering new features
│   │   ├── feat_pipelines              # feature engineering pipelines
│   │   ├── feat_preprocess_strings     # feature engineering pre process strings
│   │   ├── feat_scaling                # feature engineering scaling data
│   │   ├── feat_selection              # feature engineering feature selection
│   │   ├── feat_strings                # feature engineering strings
│   │   ├── metrics                     # evaluation metrics
│   │   ├── model                       # model training and prediction    
│   │   ├── model_evaluation            # model evaluation
│   │   ├── model_selection             # model selection
│   │   ├── model_validation            # model validation
│   │   └── reports                     # reports
│   ├── pipelines
│   │   ├── data_etl                    # data extraction, transformation, and loading
│   │   ├── feature_engineering         # prepare data for modeling
│   │   ├── model_evaluation            # evaluate model performance
│   │   ├── model_prediction            # model predictions
│   │   └── model_train                 # train models    
├── tests                               # test code for your project
└── .vscode                             # vscode configuration
    ├── extensions.json                 # list of recommended extensions
    ├── launch.json                     # vscode launch configuration
    └── settings.json                   # vscode settings
```

## Data Science Code structure

### Orchestrated experiment

```mermaid
flowchart TD
  subgraph input [ETL]
    %%nodes
    A1[(Data web)]
    B[Process_etl]
    BB1{{Data integrity}}
    BB2{{Data Validation}}
    Dcheck[(Data Checked)]

    %%links
    A1 ==>B
    B ==> BB1 ==> BB2 ==> Dcheck[(Data Checked)]
  end

  subgraph split [Train / Test data split]
    %%nodes
    C[Split - Train /Test]
    C1[(Train)]
    C2[(Test)]
    CC{{Train / Test Validation}}

    %%links
    Dcheck ==> C
    C --> |data test|C2
    C --> |data train|C1
    C2 & C1 --> CC
  end

  subgraph train_feature [Train Feature Engineering]
    %%nodes
    D[<b>Pre - process Train</b> <br> Not needed in test <br> Ex:  Remove outliers, Duplicated, Drops]


    subgraph feature [Feature Engineering pipeline <br> for use in train and test]
      style feature fill:grey,stroke:#333,stroke-width:2px
      %%nodes
      E[<b>Initial Processing</b> <br> Ex: Casting, New columns, Replace empty values for NaN]
      F{Split <br> Data Type}
      G1[Transformation <br> Numerics <br> <s>No Drops</s>]
      G2[Transformation <br> Categoric <br> <s>No Drops</s>]
      G3[Transformation <br> Booleans <br> <s>No Drops</s>]
      G4[Transformation <br> Dates <br> <s>No Drops</s>]
      G5[Transformation <br> Strings <br> <s>No Drops</s>]
      H[<b>Final Processing</b> <br> Final Pipeline <br> ColumnTranformer <br> and last transforms]
      TRfit[Train Transformer]
      TRdb[(Transformer <br> Pipeline)]
      %%links
      E -.-> F
      F -.->|Numeric|G1
      F -.->|Categoric|G2
      F -.->|Bool|G3
      F -.->|Dates|G4
      F -.->|Strings|G5
      G1 & G2 & G3 & G4 & G5 -.-> H

      H -.-> |objeto pipeline|TRfit
      TRfit -.-> |objeto pipeline|TRdb
    end

    %%nodes
    I[<b>Post - Processing Train</b> <br> Ej: Data Balance - smote, Drop duplicates <br> Not needed in test]

    %%links
    C1 --->D
    D --> |X - data train <br> pre-processed|E
    D --> |X - data train <br> pre-processed|TRfit
    TRfit --> |X - data train <br> transformed|I
    D --> |Y - data train <br> pre-processed|I

  end

  subgraph mod[Modeling]

    %%nodes
    J[Modeling]
    Modeldb[(Train Model <br> candidate)]

    %%links
    I ---> |X - data train <br> post-processed|J
    I --> |Y - data train <br> post-processed|J
    J -.-> |Model Object| Modeldb

  end

  subgraph pred [Prediction]
    %%nodes
    TRtest[Transformation <br> X - Data test]
    Pred_test[Prediction test]
    Pred_train[Prediction train]
    Pred_db[(Predictions)]

    %%links
    C2 --> |X - data test|TRtest
    TRdb -.->TRtest
    TRtest --> |X - data test <br> transformed|Pred_test
    C2 --> |Y - data test|Pred_test
    I --> |X - data train <br> post-processed|Pred_train
    I --> |Y - data train <br> post-processed|Pred_train

    Modeldb -.-> |model|Pred_train --> Pred_db
    Modeldb -.-> |model|Pred_test --> Pred_db
  end

  subgraph eval [Evaluation]
    %%nodes
    Modelcheck{{Model validation}}
    M[Eval]
    N[(Score)]

    %%links

    I  --> |X data train <br> post-processed|Modelcheck
    I  --> |Y data train <br> post-processed|Modelcheck
    TRtest --> |X - data test <br> transformed|Modelcheck
    C2 --> |Y - data test|Modelcheck
    Modeldb -....-> |model|Modelcheck
    Pred_db --> M
    M -.->N
  end

  %%links


  Modelcheck -..->  pass{Pass ?}
  pass -.-> |no|no((Alert!))
  pass -.-> |yes|si(Execute modeling <br> with all data):::Passclass


  %% Definine link styles
  linkStyle default stroke:blue

  linkStyle 8,10,12,33,35,42,45,46 stroke:orange
  linkStyle 29,31,38,44 stroke:deepskyblue
  linkStyle 36,46 stroke:gold

  %% Styling the title subgraph
  classDef Title stroke-width:0, color:#f66,  font-weight:bold, font-size: 24px;

  class input,train_feature,feature,pred,mod,eval Title


  %% Definine node styles
  classDef Objclass fill:#329cc1;
  classDef Checkclass fill:#EC5800;
  classDef Alertclass fill:#FF0000;
  classDef Passclass fill:#00CC88;

  %% Assigning styles to nodes
  class C1,C2,Dcheck,TRdb,Modeldb,Pred_db,N Objclass;
  class BB1,BB2,CC,Modelcheck Checkclass;
  class no Alertclass;
  class si Passclass;
```

### Deployment

```mermaid
flowchart TD
  orch_exp[Orchestrated Experiment] -.-> Modelcheck
  Modelcheck{{Model validation}}:::Checkclass -.-> |si| input
  Modelcheck -.-> |no|stop((Alert! <br> Stop)):::Alertclass
  subgraph input [ETL]
   Dcheck[(Data Checked)]:::Objclass
  end
  Dcheck ==>  D[<b>Pre - processing</b>]

  subgraph train_feature [Train Feature Engineering]
    %%nodes
    D[<b>Pre - processing Train</b> <br> Not needed in test <br> Ej: Drop outliers, Duplicates, Drops]


    subgraph feature [Feature Engineering pipeline <br> for use in train and test]
      style feature fill:grey,stroke:#333,stroke-width:2px
      %%nodes
      E[<b>Initial Processing </b> <br> Ej: Casting, New columns, Replace empty values for NaN]
      F{Split <br> Data Type}
      G1[Transformation <br> Numerics <br> <s>No Drops</s>]
      G2[Transformation <br> Categoric <br> <s>No Drops</s>]
      G3[Transformation <br> Booleans <br> <s>No Drops</s>]
      G4[Transformation <br> Dates <br> <s>No Drops</s>]
      G5[Transformation <br> Strings <br> <s>No Drops</s>]
      H[<b>Processing Final</b> <br> Final Pipeline <br> ColumnTranformer <br>  and final transforms]
      TRfit[Train Transformer]
      %%links
      E -.-> F
      F -.->|Numeric|G1
      F -.->|Categoric|G2
      F -.->|Bool|G3
      F -.->|Dates|G4
      F -.->|Strings|G5
      G1 & G2 & G3 & G4 & G5 -.-> H

      H -.-> |objeto pipeline|TRfit
    end

    %%nodes
    I[<b>Post - Processing Train</b> <br> Ej: Data Balance - smote, dorp duplicates <br> Not needed in test]

    %%links

    D --> |X - data train <br> pre-processed|E
    D --> |X - data train <br> pre-processed|TRfit
    TRfit --> |X - data train <br> transformed|I
    D --> |Y - data train <br> pre-processed|I

  end

  subgraph mod[Modeling]
    J[Train]
  end

  subgraph artefacto[Artefacto de salida]
    TRfit -.-> |pipeline object|TRdb[(Transformer <br> Pipeline)]:::Objclass
    J -.-> |model object| Modeldb[(Train Model <br> Final)]:::Objclass
  end

  I --> |data post-processed|mod
  J -.->N[(Performance <br> Score)]:::Objclass

  N -.-> Scorecheck{{Performance validation <br> Score actual vs anteriores}}:::Checkclass
  Scorecheck -.->  pass{Pass ?}
  pass -.-> |no|no((Alert!)):::Alertclass
  pass -.-> |yes|si(Send Artifact to Deploy):::Passclass
  si -.-> artefacto

  linkStyle 19 stroke:deepskyblue

  classDef Objclass fill:#329cc1;
  classDef Checkclass fill:#EC5800;
  classDef Alertclass fill:#FF0000;
  classDef Passclass fill:#00CC88;
```

## Credits

This project was generated from [@JoseRZapata]'s [data science project template] template.

---
[@JoseRZapata]: https://github.com/JoseRZapata

[bandit]: https://github.com/PyCQA/bandit
[codecov]: https://codecov.io/
[Cookiecutter]:https://cookiecutter.readthedocs.io/en/stable/
[coverage.py]: https://coverage.readthedocs.io/
[Cruft]: https://cruft.github.io/cruft/
[data science project template]: https://github.com/JoseRZapata/data-science-project-template
[Data structure]: demo-data-science-template/data/README.md
[hydra]: https://hydra.cc/
[Mypy]: http://mypy-lang.org/
[Notebook template]: demo-data-science-template/notebooks/notebook_template.ipynb
[pre-commit]: https://pre-commit.com/
[Pull Request template]: demo-data-science-template/.github/pull_request_template.md
[Pytest]: https://docs.pytest.org/en/latest/
[Ruff]: https://docs.astral.sh/ruff/
[UV]: https://docs.astral.sh/uv/
