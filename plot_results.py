### ## ### ##
# Given a "results" folder path from build_model.py
# Build a MS Word compatible file with, for each currency:
# - Pie charts with class distribution for train and test sets
# ?- Bar chart with data split
# - # of features, # of records for train and test set
# - Feature importances bar graph
# - Learning curves
# - Classification reports for train and test set
# ?- Parameters table
#
# From equity results print:
# - Cumulative table with profit for each currency
# - Cumulative equity graph

### ## ### ##
import os, json
try:
   import cPickle as pickle
except:
   import pickle
from pathlib import Path
import argparse
from matplotlib import pyplot as plt
import pandas as pd
import docx
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.shared import Cm
from lib.dataset import get_class_distribution
import numpy as np

def get_report_list(dir, equity_dir=None):
    index = {}
    for file in os.listdir(dir):
        if not file.endswith('_report.json'):
            continue
        key = file.replace('_report.json', '')
        index[key] = {
            'report' : '{}/{}_report.p'.format(dir, key),
            'learning' : '{}/{}_learning_curve.png'.format(dir, key),
            'features' : '{}/{}_feature_importances.png'.format(dir, key),
            'distribution' : '{}/{}_class_dist.png'.format(dir, key),
            'parameters' : '{}/{}_parameters.json'.format(dir, key)
        }
        if equity_dir:
            index[key].update({'trading':'{}/{}_trading.png'.format(equity_dir, key)})
    return index

def main(pipeline, result_dir, out_file, equity_dir=None):
    doc = docx.Document()
    index = get_report_list(result_dir, equity_dir)
    doc.add_heading("{} Models training".format(pipeline), 0)
    for key, files in index.items():
        print("Processing {} ..".format(key))
        report = None
        with open(files['report'], 'rb') as f:
            report = pickle.load(f)
        if not report:
            print("Error opening report: {}".format(files['report']))
            continue
        if not os.path.exists(files['features']):
            #features = report['feature_importances']
            features = report.cv_report['feature_importances']
            df = pd.DataFrame.from_dict(features, orient='index', columns=['importance']).iloc[::-1] # Create dataframe and invert index (high to low)
            plt.rcdefaults()
            df.plot(
                kind='barh',
                title='{} Feature Importances'.format(key),
                figsize=(14, 20),
                ylim=(df.shape[0]+1),
                legend=False,
                fontsize=8,
                width=0.5
            )
            plt.tight_layout(pad=0.5)
        if not os.path.exists(files['distribution']):
            train_dist = get_class_distribution(report.y_train)
            test_dist = get_class_distribution(report.y_test)
            class_labels = ['SELL', 'HOLD', 'BUY']
            train_wedges = [ train_dist[0]['count'], train_dist[1]['count'], train_dist[2]['count']]
            test_wedges = [ test_dist[0]['count'], test_dist[1]['count'], test_dist[2]['count']]

            def func(pct, allvals):
                absolute = int(pct / 100. * np.sum(allvals))
                return "{:.1f}% ({:d})".format(pct, absolute)
            # Plot training set
            fig, (train_ax, test_ax) = plt.subplots(1, 2, figsize=(6, 3), subplot_kw=dict(aspect="equal"))
            fig.suptitle("Class Distribution")
            _n_wedges, _n_texts, _n_autotexts = train_ax.pie(train_wedges, autopct=lambda pct: func(pct, train_wedges), textprops=dict(color="w"))
            train_ax.legend(_n_wedges, class_labels,
                           title="Classes",
                           loc="center left",
                           bbox_to_anchor=(1, 0, 0.5, 1))
            train_ax.set_title("Train")
            # Plot test set
            _t_wedges, _t_texts, _t_autotexts = test_ax.pie(test_wedges, autopct=lambda pct: func(pct, test_wedges), textprops=dict(color="w"))
            test_ax.legend(_t_wedges, class_labels,
                      title="Classes",
                      loc="center left",
                      bbox_to_anchor=(1, 0, 0.5, 1))
            test_ax.set_title("Test")
            # Make the layout tight
            plt.tight_layout(pad=0.5)
            plt.savefig(files['distribution'])
            plt.close()
        doc.add_heading("{} Dataset".format(key), 1)
        # Dataset table
        doc.add_heading("Dataset analysis", 3)
        table = doc.add_table(rows=2, cols=4)
        table.cell(0, 0).text = ""
        table.cell(0, 1).text = "Train"
        table.cell(0, 2).text = "Test"
        table.cell(0, 3).text = "Total"
        table.cell(1, 0).text = "# of Records"
        table.cell(1, 1).text = str(report.y_train.shape[0])
        table.cell(1, 2).text = str(report.y_test.shape[0])
        table.cell(1, 3).text = str(report.y_train.shape[0] + report.y_test.shape[0])
        table.style = 'Medium Grid 3 Accent 1'
        # Class distribution graph
        doc.add_picture(files['distribution'], width=Cm(16))


        doc.add_heading("{} Model".format(key), 1)
        #doc.add_picture(files['learning'], width=docx.shared.Inches(5), height=docx.shared.Inches(7))
        doc.add_picture(files['learning'], width=Cm(16))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        # Add Parameters table
        with open(files['parameters']) as f:
            params = json.load(f)
            doc.add_heading("Classifier Parameters".format(key), 3)
            table = doc.add_table(rows=len(params) + 1, cols=2)
            table.cell(0, 0).text = "Parameter"
            table.cell(0, 1).text = "Value"
            for i, key in enumerate(params.keys()):
                table.cell(i+1, 0).text = str(key)
                table.cell(i+1, 1).text = str(params[key])
            table.style = 'Medium Grid 3 Accent 1'
        doc.add_picture(files['features'], width=Cm(16))
        if 'trading' in files and os.path.exists(files['trading']):
            doc.add_picture(files['trading'], width=Cm(16))
    doc.save(out_file)

if __name__ == '__main__':
    main(
        'XGBoost',
        'B:/Tesi-POLITO/LSTM_forecaster/results/timedspline_debug_xgboost_splines_experiment_171220_060755',
        'B:/Tesi-POLITO/LSTM_forecaster/results/result.docx',
        equity_dir='B:/Tesi-POLITO/LSTM_forecaster/equities/test2'
    )