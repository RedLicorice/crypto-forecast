import json
import pandas as pd
import matplotlib.pyplot as plt

def main(dataset):
    resultFile = 'data/datasets/{}/feature_selection2.json'.format(dataset)
    results = {}
    with open(resultFile) as f:
        results = json.load(f)

    for _sym, result in results.items():
        rfecv_03 = result["test_size=0.3"]["rfecv"]["feature_importances"]
        rfecv_03_rank = result["test_size=0.3"]["rfecv"]["rank"]
        rfecv_03_support = result["test_size=0.3"]["rfecv"]["support"]
        genetic_03 = result["test_size=0.3"]["genetic"]["feature_importances"]
        genetic_03_support = result["test_size=0.3"]["genetic"]["support"]
        randomforest_03 = result["test_size=0.3"]["randomforest"]["feature_importances"]

        #Build DataFrames
        feature_imp_dict = {}
        for feature in randomforest_03.keys(): # random forest's results have all the keys
            feature_imp_dict[feature] = [
                randomforest_03[feature],
                rfecv_03[feature] if feature in rfecv_03 else 0,
                genetic_03[feature] if feature in genetic_03 else 0,
            ]
        line_count = len(feature_imp_dict.keys())


        # Save results DataFrame
        df = pd.DataFrame.from_dict(feature_imp_dict, orient='index', columns=["randomforest", "rfecv", "genetic"])
        df.sort_index(inplace=True)
        xls_path = 'data/datasets/{}/feature_selection/{}.xlsx'.format(dataset, _sym.lower())
        excel_writer = pd.ExcelWriter(xls_path, engine='xlsxwriter')
        df.to_excel(excel_writer, index=True, index_label='Feature', sheet_name='Sheet1')
        worksheet = excel_writer.sheets['Sheet1']
        worksheet.conditional_format('B2:B{}'.format(line_count+1), {'type': '3_color_scale'})
        worksheet.conditional_format('C2:C{}'.format(line_count+1), {'type': '3_color_scale'})
        worksheet.conditional_format('D2:D{}'.format(line_count+1), {'type': '3_color_scale'})
        excel_writer.save()
        ax = df.plot(kind='barh', figsize=(20, 20), title='{}: {} feature importances '.format(dataset, _sym))
        plt.savefig('data/datasets/{}/feature_selection/{}.png'.format(dataset, _sym.lower()))
        ax.figure.clear()



if __name__ == '__main__':
    main('ohlcv_coinmetrics')
    #main('ohlcv_social')
    #main('resampled_ohlcv_ta')
