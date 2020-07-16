import json
import pandas as pd
import matplotlib.pyplot as plt

def main(dataset):
    resultFile = 'data/datasets/{}/feature_selection.json'.format(dataset)
    results = {}
    with open(resultFile) as f:
        results = json.load(f)

    for _sym, result in results.items():
        feature_imp_05 = result["test_size=0.5"]["randomforest"]["feature_importances"]
        feature_imp_03 = result["test_size=0.3"]["randomforest"]["feature_importances"]
        feature_imp_01 = result["test_size=0.1"]["randomforest"]["feature_importances"]

        genetic_05 = result["test_size=0.5"]["genetic"]["feature_importances"]
        genetic_03 = result["test_size=0.3"]["genetic"]["feature_importances"]
        genetic_01 = result["test_size=0.1"]["genetic"]["feature_importances"]

        rfecv_rank_01 = result["test_size=0.1"]["randomforest"]["rank"]
        rfecv_rank_03 = result["test_size=0.3"]["randomforest"]["rank"]
        rfecv_rank_05 = result["test_size=0.5"]["randomforest"]["rank"]

        #Build DataFrames
        feature_imp_dict = {}
        rfecv_rank_dict = {}
        for feature in feature_imp_01.keys():
            feature_imp_dict[feature] = [feature_imp_05[feature], feature_imp_03[feature], feature_imp_01[feature]]
            rfecv_rank_dict[feature] = [rfecv_rank_05[feature], rfecv_rank_03[feature], rfecv_rank_01[feature]]
        line_count = len(feature_imp_dict.keys())
        # Save RandomForest DataFrame
        randomforest_df = pd.DataFrame.from_dict(feature_imp_dict, orient='index', columns=["test_size=0.5", "test_size=0.3", "test_size=0.1"])
        randomforest_df.sort_index(inplace=True)
        randomforest_xls_path = 'data/datasets/{}/feature_selection/{}_randomforest.xlsx'.format(dataset, _sym.lower())
        randomforest_writer = pd.ExcelWriter(randomforest_xls_path, engine='xlsxwriter')
        randomforest_df.to_excel(randomforest_writer, index=True, index_label='Feature', sheet_name='Sheet1')
        randomforest_worksheet = randomforest_writer.sheets['Sheet1']
        randomforest_worksheet.conditional_format('B2:B{}'.format(line_count+1), {'type': '3_color_scale'})
        randomforest_worksheet.conditional_format('C2:C{}'.format(line_count+1), {'type': '3_color_scale'})
        randomforest_worksheet.conditional_format('D2:D{}'.format(line_count+1), {'type': '3_color_scale'})
        randomforest_writer.save()
        ax = randomforest_df.plot(kind='barh', figsize=(20, 20))
        plt.savefig('data/datasets/{}/feature_selection/{}_randomforest_barh.png'.format(dataset, _sym.lower()))
        ax.figure.clear()
        ax = randomforest_df.plot(kind='box', figsize=(20, 20))
        plt.savefig('data/datasets/{}/feature_selection/{}_randomforest_box.png'.format(dataset, _sym.lower()))
        ax.figure.clear()
        # Save RFECV DataFrame
        rfecv_rank_df = pd.DataFrame.from_dict(rfecv_rank_dict, orient='index', columns=["test_size=0.5", "test_size=0.3", "test_size=0.1"])
        rfecv_rank_df.sort_index(inplace=True)
        rfecv_rank_xls_path = 'data/datasets/{}/feature_selection/{}_rfecv_rank.xlsx'.format(dataset, _sym.lower())
        rfecv_rank_writer = pd.ExcelWriter(rfecv_rank_xls_path, engine='xlsxwriter')
        rfecv_rank_df.to_excel(rfecv_rank_writer, index=True, index_label='Feature', sheet_name='Sheet1')
        rfecv_rank_worksheet = rfecv_rank_writer.sheets['Sheet1']
        rfecv_rank_worksheet.conditional_format('B2:B{}'.format(line_count + 1), {'type': '3_color_scale'})
        rfecv_rank_worksheet.conditional_format('C2:C{}'.format(line_count + 1), {'type': '3_color_scale'})
        rfecv_rank_worksheet.conditional_format('D2:D{}'.format(line_count + 1), {'type': '3_color_scale'})
        rfecv_rank_writer.save()




if __name__ == '__main__':
    main('ohlcv_coinmetrics')
    #main('ohlcv_social')
    #main('resampled_ohlcv_ta')
