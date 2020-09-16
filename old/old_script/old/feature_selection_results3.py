import json
import pandas as pd
import matplotlib.pyplot as plt

def main(dataset):
    resultFile = 'data/datasets/{}/feature_selection3.json'.format(dataset)
    results = {}
    with open(resultFile) as f:
        results = json.load(f)

    for _sym, runs in results.items():
        for run in runs:
            feature_set = run['feature_set']
            test_size = run['test_size']

            xls_path = 'data/datasets/{}/feature_selection/{}_test{}_{}.xlsx'.format(dataset, _sym.lower(), int(test_size*100), feature_set)
            plot_path = 'data/datasets/{}/feature_selection/{}_test{}_{}.png'.format(dataset, _sym.lower(), int(test_size*100), feature_set)

            rfecv = run["rfecv"]["feature_importances"]
            rfecv_rank = run["rfecv"]["rank"]
            rfecv_support = run["rfecv"]["support"]
            rfecv_score = run['rfecv']['score']
            rfecv_test_score = run['rfecv']['test_score']
            genetic = run["genetic"]["feature_importances"]
            genetic_support = run["genetic"]["support"]
            genetic_score = run['genetic']['score']
            genetic_test_score = run['genetic']['test_score']
            randomforest = run["randomforest"]["feature_importances"]
            randomforest_score = run['randomforest']['score']
            randomforest_test_score = run['randomforest']['test_score']

            #Build DataFrames
            feature_imp_dict = {}
            for feature in randomforest.keys(): # random forest's results have all the keys
                feature_imp_dict[feature] = [
                    randomforest[feature],
                    rfecv[feature] if feature in rfecv else 0,
                    genetic[feature] if feature in genetic else 0,
                ]
            line_count = len(feature_imp_dict.keys())


            # Save results DataFrame
            df = pd.DataFrame.from_dict(feature_imp_dict, orient='index', columns=["randomforest", "rfecv", "genetic"])
            df.sort_index(inplace=True)
            excel_writer = pd.ExcelWriter(xls_path, engine='xlsxwriter')
            df.to_excel(excel_writer, index=True, index_label='Feature', sheet_name='Sheet1')
            worksheet = excel_writer.sheets['Sheet1']
            worksheet.conditional_format('B2:B{}'.format(line_count+1), {'type': '3_color_scale'})
            worksheet.conditional_format('C2:C{}'.format(line_count+1), {'type': '3_color_scale'})
            worksheet.conditional_format('D2:D{}'.format(line_count+1), {'type': '3_color_scale'})
            excel_writer.save()
            ax = df.plot(kind='barh', figsize=(20, 20), title='{}: {} feature importances ({})'.format(dataset, _sym, feature_set))
            textstr = '\n'.join([
                'Dataset: {}'.format(dataset),
                'Feature Set: {}'.format(feature_set),
                'Train/Test: {}/{} %'.format(int((1-test_size)*100), int(test_size*100)),
                'Randomforest Score: {:.2f}% (Train), {:.2f}% (Test)'.format(randomforest_score*100, randomforest_test_score*100),
                'RFECV Score: {:.2f}% (Train), {:.2f}% (Test)'.format(rfecv_score*100, rfecv_test_score*100),
                'Genetic Score: {:.2f}% (Train), {:.2f}% (Test)'.format(genetic_score*100, genetic_test_score*100),
            ])
            ax.text(0.65, 0.10, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox={
                'boxstyle':'round',
                'facecolor':'wheat',
                'alpha':0.5
            })

            plt.savefig(plot_path)
            ax.figure.clear()
            plt.close()



if __name__ == '__main__':
    main('ohlcv_coinmetrics')
    main('ohlcv_social')
    main('resampled_ohlcv_ta')
