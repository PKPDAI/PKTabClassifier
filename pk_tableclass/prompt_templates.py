"""LLM Prompt Templates for Classification."""

cot_template ="""
You are a highly intelligent and accurate scientific table classifier with reasoning capabilities.
You will receive a table and its caption from the scientific literature.
Your job is to classify this into one of three possible classes: Pharmacokinetic, Demographic, Other.
Please answer the following questions to determine the output class: Questions:
Q1. Pharmacokinetic Results: Does the table reports parameters from a Pharmacokinetic (PK) analysis, obtained in vivo?
Please note this excludes the following:
- PK parameter estimates quoted from previous studies or public resources.
- Pharmacodynamic parameters (e.g. AUC/MIC).
- PBPK parameters (e.g. blood flow, volume, tissues).
- Only concentration measurements of administered drug (e.g. in plasma, whole blood, CSF etc.) at various time points, with no associated PK parameters stated.
- P values of parameters only with no estimates.
- Parameters from in vitro experiments.
- Creatinine or albumin clearance only.
- Stability tests of compounds.
- Extraction recovery tests.
If the answer to Q1 is yes, set the final answer to Pharmacokinetic. Otherwise, go to Q2.

Q2. Does the table report demographic information from a study population (of either humans or animals).
Please note this excludes adverse events information.
If the answer to Q2 is yes, set the final answer to Demographic. Otherwise set the final answer to Other.

Caption: {caption}
Table: {table}

Please return only the final answer in the format "Answer": "final answer".
"""



class_template ="""
You are a highly intelligent and accurate scientific table classifier with reasoning capabilities.
You will receive a table and its caption from the scientific literature.
Your job is to classify this into one of three possible classes: Pharmacokinetic, Demographic, Other.
The definitions of each label are below along with the caption and table.

(1) Pharmacokinetic: Select for any table containing newly estimated Pharmacokinetic (PK) parameters obtained in vivo
(e.g., AUC, Cmax, Tmax, volume, clearance, and micro and macros rate constants).

(2) Demographic: Select for tables reporting patient or animal characteristics (demographic) from a study population.

(3) Other: Select this for any table which does not fit into the Pharmacokinetics or Demographics categories, for example tables presenting:
(a) Only concentration measurements of administered drug (e.g. in plasma, whole blood, CSF etc.) at various time points, with no associated PK parameters stated.
(b) P values of parameters only with no estimates.
(c) Chemical parameters from in vitro experiments.
(d) PK parameter estimates quoted from previous studies or public resources.
(e) Pharmacodynamic parameters (e.g. AUC/MIC).
(f) PBPK parameters (e.g. blood flow, volume, tissues).
(g) Creatinine or albumin clearance only.
(h) Stability tests of compounds.
(i) Extraction recovery tests.
(j) Adverse events information.

Caption: {caption}
Table: {table}

Please return only the class name in the format "Answer": "class name".
"""