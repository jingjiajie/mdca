from mdca import main

if __name__ == '__main__':
    main()

# data_df: pd.DataFrame = mock_hmeq_data()
#
# analyzer: MultiDimensionalAnalyzer = MultiDimensionalAnalyzer(data_df, target_column='BAD', target_value=1,
#                                                               min_coverage=0.05, mode=mode)
# data_df: pd.DataFrame = pl.read_csv('data/hmeq/hmeq_train.csv').to_pandas()
#
# analyzer: MultiDimensionalAnalyzer = MultiDimensionalAnalyzer(data_df, target_column='BAD', target_value=1,
                                                                # min_coverage=0.05, mode=mode)

# data_df = pl.read_csv('data/flights/flights.csv', encoding="utf8-lossy").to_pandas()
#
# data_df['DELAYED'] = ~(data_df['AIR_SYSTEM_DELAY'].isna() & data_df['SECURITY_DELAY'].isna() &
#                          data_df['AIRLINE_DELAY'].isna() & data_df['LATE_AIRCRAFT_DELAY'].isna() &
#                          data_df['WEATHER_DELAY'].isna())
# data_df.drop(['DEPARTURE_DELAY','ARRIVAL_DELAY','AIR_SYSTEM_DELAY', 'SECURITY_DELAY',
#               'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'],
#              axis=1, inplace=True)
#
# data_df = data_df[['YEAR','MONTH','DAY','DAY_OF_WEEK','AIRLINE','FLIGHT_NUMBER',
#                    'TAIL_NUMBER','ORIGIN_AIRPORT','DESTINATION_AIRPORT','DELAYED']]
# analyzer: MultiDimensionalAnalyzer = MultiDimensionalAnalyzer(data_df, target_column='DELAYED', target_value=1,
#                                                               min_coverage=0.05, mode=mode)

# data_df: pd.DataFrame = pd.read_csv('data/tianchi-loan/pred_2011.csv')
# data_df = data_df[data_df['term'] != 6]
# data_df['predict'] = 0
# data_df['target'] = data_df['isError']
# data_df.drop('isError', axis=1, inplace=True)
# analyzer: MultiDimensionalAnalyzer = MultiDimensionalAnalyzer(data_df, columns=['term','verificationStatus'],
#                                                               mode='error', target_column='target',
#                                                               prediction_column='predict',
#                                                               min_error_coverage=0.01)

# print('Loading data...')
# data_df: pd.DataFrame = pl.read_csv('data/recruitment/recruitmentdataset-2022-1.3.csv', encoding="utf8-lossy").to_pandas()
# print('Load data cost: %.2f seconds' % (time.time() - start))
# # data_df.drop(['decision'], axis=1, inplace=True)
# # analyzer: MultiDimensionalAnalyzer = MultiDimensionalAnalyzer(data_df, columns=['gender', 'company'],mode='distribution', min_coverage=0.01)
# analyzer: MultiDimensionalAnalyzer = MultiDimensionalAnalyzer(data_df,columns=['gender', 'company'],
#                                                               mode='distribution',
#                                                               target_column='decision', target_value=1,
#                                                               min_coverage=0.01)


# results: list[CalculatedResult] = analyzer.run()
# print("\nTotal time cost: %.2f seconds" % (time.time() - start))
# analyzer.print_results(results)
