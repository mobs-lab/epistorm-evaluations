### REQUIREMENTS
#################################################
import pandas as pd
import numpy as np
import multiprocess as mp
import datetime
import argparse
import itertools
import glob
from epiweeks import Week

import warnings
warnings.filterwarnings('ignore')


### CLASSES AND METHODS
#################################################

# functions for calculating scores of the forecasts against the surveillance data
class Forecast_Eval:
    """ Used for scoring and evaluating flu forecasting predictions from the Flusight challenge for the 
        2023-24 season
    """
    
    def __init__(self, df, obsdf, target,  start_week = False, end_week = False):
        self.df = df # pandas Dataframe: input dataframe of forecasts with all scenarios, locations, and quantiles
        self.obsdf = obsdf # pandas Dataframe: input of surveillance data of interest
        self.target = target # str: target metric of interest (case, death, hospitalization)
        self.start_week = start_week # epiweek: beginning of observations of interest
        self.end_week = end_week # epiweek: end of observations of interest
        
    def process_observations(self, data, value_col='value', t_col='date', other_ind_cols=None):
        """
    
        Parameters:
        - data: pd.DataFrame
            The input data containing observational information.
        - value_col: str
            The column name representing the value column.
        - t_col: str
            The column name representing the time column.
        - other_ind_cols: list
            A list of additional independent columns for sorting.

        Returns:
        - pd.DataFrame with additional helper functions for accessing specific columns.
        """
        # Ensure required columns are present
        if value_col not in data.columns or t_col not in data.columns:
            raise ValueError(f"DataFrame must contain '{value_col}' and '{t_col}' columns.")

        # Prepare independent columns and sort data
        ind_cols = [t_col] + (other_ind_cols if other_ind_cols else [])
        sorted_data = data.sort_values(by=ind_cols).reset_index(drop=True)

        # Define helper functions to access specific columns and set them as attributes
        sorted_data.get_value = lambda: sorted_data[value_col].to_numpy()
        sorted_data.get_t = lambda: sorted_data[t_col].to_numpy()
        sorted_data.get_x = lambda: sorted_data[ind_cols].to_numpy()
        sorted_data.get_unique_x = lambda: np.unique(np.array(sorted_data[ind_cols].to_numpy(), dtype=str), axis=0)

        return sorted_data      
            
    def get_observations(self, target_location):
        """ get_observations. Load and filter surveillance data for a certain location.
        Parameters
        ----------
        target_location : str
            location to filter surveillance data by
        """
        
        if self.target == 'hosp':
            target_obs = 'hospitalization'
        else:
            target_obs = self.target
            
        # read in observations dataframe
        
        observations = self.obsdf.copy().drop(columns= ['Unnamed: 0', 'weekly_rate'])
        observations['date'] = pd.to_datetime(observations['date'])

        #filter start - end week
        if self.start_week:
            observations = observations[(observations['date'] >= pd.to_datetime(self.start_week.startdate())) ]
            
        if self.end_week:
            observations = observations[(observations['date'] <= pd.to_datetime(self.end_week.enddate()))]
                                
        #filter location
        observations = observations[observations['location'] == target_location]

        #aggregate to weekly
        observations = observations.groupby(['location', pd.Grouper(key='date', freq='W-SAT')]).sum().reset_index()

        #transform to Observation object
        observations = self.process_observations(observations)

        return observations
    
    def process_predictions(self, data, value_col='value', quantile_col='output_type_id', type_col='output_type',
                        t_col='target_end_date', other_ind_cols=None):
        """
        
        Parameters:
        - data: pd.DataFrame
            The input data containing prediction information.
        - value_col: str
            Column label for the predictions' value.
        - quantile_col: str
            Column label for the predictions' quantile.
        - type_col: str
            Column label for the type of predictions (e.g., quantile or point).
        - t_col: str
            Column label for the timestamp of predictions.
        - other_ind_cols: list
            List of other independent variable columns (e.g., location).

        Returns:
        - pd.DataFrame with additional helper methods to access specific data arrays or filtered predictions.
        """
        # Ensure required columns are present
        if not all(col in data.columns for col in [value_col, quantile_col, t_col]):
            raise ValueError(f"DataFrame must contain '{value_col}', '{quantile_col}', and '{t_col}' columns.")
        if other_ind_cols and not all(col in data.columns for col in other_ind_cols):
            raise ValueError("DataFrame must contain all specified independent columns.")

        # Define independent columns and sort data
        ind_cols = [t_col] + (other_ind_cols if other_ind_cols else [])
        sorted_data = data.sort_values(by=ind_cols).reset_index(drop=True)

        # Set default value for type column if missing
        if type_col not in sorted_data.columns:
            sorted_data[type_col] = 'quantile'

        # Attach helper methods as attributes of the DataFrame
        sorted_data.get_t = lambda: sorted_data[t_col].to_numpy()
        sorted_data.get_x = lambda: sorted_data[ind_cols].to_numpy()
        sorted_data.get_unique_x = lambda: np.unique(np.array(sorted_data[ind_cols].to_numpy(), dtype=str), axis=0)

        return sorted_data

    def format_forecasts_all(self, dfformat):
        """ format_forecasts_all. Get forecasts into standard format to use for scoring.
        Parameters
        ----------
        dfformat : pandas DataFrame
            dataframe of forecast output used for formatting
        """
        
        pred = dfformat.copy()
        pred = pred[pred.output_type == 'quantile'] # only keep quantile predictions
        pred['target_end_date'] = pd.to_datetime(pred['target_end_date']) #make sure dates are in datetime format
        if self.start_week:
            pred = pred[(pred['target_end_date'] >= pd.to_datetime(self.start_week.startdate()))] # filter dates
        
        if self.end_week:
            pred = pred[(pred['target_end_date'] <= pd.to_datetime(self.end_week.enddate()))] # filter dates
        
        pred['output_type_id'] = pred["output_type_id"].astype(float) # make sure quantile levels are floats
        
        return pred
        
    
class Scoring(Forecast_Eval):
    """ calculate score values for probabilistic epidemic forecasts 
    find WIS, MAPE, and coverage over whole projection window as well as timestamped for every week.
    score dataframe must have 'Model' column to differentiate and calculate scores for different models
    """
    
    def __init__(self, df, obsdf, target, start_week = False, 
                 end_week = False):
        super().__init__(df, obsdf, target, start_week, end_week)
        
    def interval_score(self, observation, lower, upper, interval_range, specify_range_out=False):
        """interval_score.

        Parameters
        ----------
        observation : array_like
            Vector of observations.
        lower : array_like
            Prediction for the lower quantile.
        upper : array_like
            Prediction for the upper quantile.
        interval_range : int
            Percentage covered by the interval. For instance, if lower and upper correspond to 0.05 and 0.95
            quantiles, interval_range is 90.

        Returns
        -------
        out : dict
            Dictionary containing vectors for the interval scores, but also the dispersion, underprediction and
            overprediction.

        Raises
        ------
        ValueError:
            If the observation, the lower and upper vectors are not the same length or if interval_range is not
            between 0 and 100
        """
        if len(lower) != len(upper) or len(lower) != len(observation):
            raise ValueError("vector shape mismatch")
        if interval_range > 100 or interval_range < 0:
            raise ValueError("interval range should be between 0 and 100")

        #make sure vector operation works
        obs,l,u = np.array(observation),np.array(lower),np.array(upper)

        alpha = 1-interval_range/100 #prediction probability outside the interval
        dispersion = u - l
        underprediction = (2/alpha) * (l-obs) * (obs < l)
        overprediction = (2/alpha) * (obs-u) * (obs > u)
        score = dispersion + underprediction + overprediction
        if not specify_range_out:
            out = {'interval_score': score,
                   'dispersion': dispersion,
                   'underprediction': underprediction,
                   'overprediction': overprediction}
        else:
            out = {f'{interval_range}_interval_score': score,
                   f'{interval_range}_dispersion': dispersion,
                   f'{interval_range}_underprediction': underprediction,
                   f'{interval_range}_overprediction': overprediction}
        return out

    def timestamp_wis(self,observations, predsfilt, interval_ranges =[10,20,30,40,50,60,70,80,90,95,98]):
        """timestamp_wis.

        Parameters
        ----------
        observations : Observations object
            Specialized dateframe for the observations across time.
        predictions : Predictions object
            Specialized dateframe for the predictions (quantile and point) across time.
        interval_ranges : list of int
            Percentage covered by each interval. For instance, if interval_range is 90, this corresponds
            to the interval for the 0.05 and 0.95 quantiles.

        Returns
        -------
        df : DataFrame
            DataFrame containing the weighted interval score across time.

        Raises
        ------
        ValueError:
            If the independent columns do not match for observations and predictions.
            If the median is not calculated.
            If the point estimate is not included.

        """

        quantiles = np.array(predsfilt.sort_values(by='output_type_id').output_type_id)

        qs = []
        for q in quantiles:
            df = predsfilt[predsfilt.output_type_id==q].sort_values(by='target_end_date')
            val = np.array(df.value)
            qs.append(val)

        Q = np.array(qs) # quantiles array
        y = np.array(observations.value) # observations array

        # calculate WIS
        WIS = np.zeros(len(y))

        for i in range(len(quantiles) // 2):
            interval_range = 100*(quantiles[-i-1]-quantiles[i])
            #print(interval_range)
            alpha = 1-(quantiles[-i-1]-quantiles[i])
            IS = self.interval_score(y,Q[i],Q[-i-1],interval_range)
            WIS += IS['interval_score']*alpha/2
        WIS += 0.5*np.abs(Q[11] - y)

        WISlist = np.array(WIS) / (len(interval_ranges) + 0.5)

        df = pd.DataFrame({'Model':predsfilt.Model.unique(), 'location':predsfilt.location.unique(), 'horizon':predsfilt.horizon.unique(),
                           'reference_date':predsfilt.reference_date.unique(), 'target_end_date':predsfilt.target_end_date.unique(),
                           'wis':WISlist[0]},index=[0])

        return df
    
    def coverage(self,observation,lower,upper):
        """coverage. Output the fraction of observations within lower and upper.

        Parameters
        ----------
        observation : array_like
            Vector of observations.
        lower : array_like
            Prediction for the lower quantile.
        upper : array_like
            Prediction for the upper quantile.

        Returns
        -------
        cov : float
            Fraction of observations within the lower and upper bound.


        Raises
        ------
        ValueError:
            If the observation, the lower and upper vectors are not the same length.
        """
        if len(lower) != len(upper) or len(lower) != len(observation):
            raise ValueError(f"vector shape mismatch: lower/upper/obs {len(lower)}/{len(upper)}/{len(observation)}")

        #make sure vector operation works
        obs,l,u = np.array(observation),np.array(lower),np.array(upper)

        return np.mean(np.logical_and(obs >= l, obs <= u))

    def all_coverages_from_df(self,observations, predictions, interval_ranges=[10,20,30,40,50,60,70,80,90,95,98],
                              **kwargs):
        """all_coverages_from_df.

        Parameters
        ----------
        observations : DataFrame object
            Dateframe for the observations across time.
        predictions : DataFrame object
            Dateframe for the predictions (intervals) across time.
        interval_ranges : list of int
            Percentage covered by each interval. For instance, if interval_range is 90, this corresponds
            to the interval for the 0.05 and 0.95 quantiles.

        Returns
        -------
        out : dict
            Dictionary containing the coverage for all interval ranges.

        Raises
        ------
        ValueError:
            If the independent columns do not match for observations and predictions.
        """
        #verify that the independent variable columns (usually dates and location) matches
        if not np.array_equal(observations.get_unique_x(), predictions.get_unique_x()):
            raise ValueError("Values for the independent columns do not match")

        out = dict()
        for interval_range in interval_ranges:
            q_low,q_upp = round(0.5-interval_range/200,3),round(0.5+interval_range/200,3)
            cov = self.coverage(list(observations.value),
                           list(predictions[predictions.output_type_id ==q_low].value),
                           list(predictions[predictions.output_type_id==q_upp].value))
            out[f'{interval_range}_cov'] = cov
        return out
        
    def get_mape(self):
        """ get_mape. Calculate MAPE (mean absolute percentage error) for each date of a forecast. If 
            surveillance data point is equal to zero, the score is undefined (Nan).
        
        """
        
        predictions = self.df.copy()
        
        # get point forecast, here we say it is the median
        predictions = predictions[predictions['output_type_id'] == 0.5] # get point forecast, here we say it is the median

        mapedf = pd.DataFrame()

        # find mape for a given model and location over projection period
        for model in predictions.Model.unique():
            for target_location in predictions.location.unique():

                    if target_location in ['60','66','69', '72', '78']:
                        continue

                    observations = self.get_observations(target_location)
                    
                    pred = predictions[(predictions.location == target_location) & (predictions.Model==model)]
                    pred = self.process_predictions(pred, t_col = 'target_end_date',quantile_col='output_type_id')
                    
                    observations = observations[observations.date.isin(pred.target_end_date.unique())]

                    n = observations.shape[0]

                    realvals = list(observations.value)
                    predvals = list(pred.value)
                    
                    if len(predvals) == 0 or len(realvals) == 0: continue

                    if realvals[0] == 0:
                        n = n - 1
                        continue

                    err = abs((realvals[0]-predvals[0])/realvals[0]) # find relative error

                    if n == 0:
                        mape = None
                    else:
                        mape = err # calculate mape

                    data = {'Model': model,'Location': target_location, 'MAPE':mape}

                    # store in pandas DataFrame
                    newdf = pd.DataFrame(data, index=[1])

                    mapedf = pd.concat([mapedf, newdf])

        mapedf = mapedf.reset_index()
        mapedf = mapedf.drop(['index'], axis=1)

        return mapedf


### HANDLE INPUTS & LOAD DATA
#################################################
# all existing models
all_models = [f.split('/')[-2] for f in glob.glob('./FluSight-forecast-hub/model-output/*/')]

# Report available and used RAM
def report_memory():
    '''
    Report available and used RAM
    Prints with flush=True
    '''
    import os

    # Getting all memory using os.popen()
    total_memory, used_memory, free_memory = map(
	    int, os.popen('free -t -m').readlines()[-1].split()[1:])

    # Memory usage
    print(f'Total RAM: {total_memory}\nRAM % used: {round((used_memory/total_memory) * 100, 2)}', flush=True)

report_memory()

# Accept inputs for:
# mode - 'update', 'scratch', or 'archive' (archive is for 2021-22 and 2022-23 seasons)
# models - any number of model names in a space-separated string, or 'all'
# dates - any number of dates in YYYY-MM-DD format in a space-separated string
parser = argparse.ArgumentParser()
parser.add_argument('--mode', action='store', nargs=1, choices=['update', 'scratch', 'archive'], required=True,
                    help='Update deployment evaluations, work in scratch folder, or score 2021-23 archive data.')
parser.add_argument('--models', nargs='+', choices=all_models+['all'], required=False, default='all',
                    help='Specify any number of space-separated model names, or \'all\'.')
parser.add_argument('--dates', nargs='+', required=False, default='all',
                    help='Specify any number of space-separated dates in YYYY-MM-DD format, or \'all\'.')
args = parser.parse_args()

# mode handling and data reading
mode = args.mode[0]
print('Reading data...')
            
if mode == 'update':
    # list of files with new predictions data
    updated_forecasts = pd.read_csv('./updated_forecasts.csv')
    print(f'New forecasts:\n{str(updated_forecasts)}\n')
    
    # detect dates with new/changed surveillance numbers
    surv = pd.read_csv('./data/ground-truth/target-hospital-admissions.csv')
    surv_old = pd.read_csv('./data/ground-truth/target-hospital-admissions.csv_old')
    all_df = pd.merge(surv, surv_old, on=surv.columns.tolist(), how='left', indicator='exists')
    is_new = np.where(all_df.exists == 'both', False, True) # True if row in surv does not exist in surv_old
    new_records = surv[is_new]
    print(f'New surveillance:\n{new_records}\n')
    
    # read predictions for all models for the updated surveillance target dates
    predictionsall = pd.DataFrame()
    if not new_records.empty:
        update_target_dates = pd.unique(new_records.date)
        
        # calculate reference dates for predictions including desired target dates
        update_reference_dates = np.array([])
        date_format = '%Y-%m-%d'
        for date in update_target_dates:
            update_reference_dates = np.append(update_reference_dates, [date])
            for i in [1, 2, 3]:
                update_reference_dates = np.append(update_reference_dates, datetime.datetime.strftime(
                    datetime.datetime.strptime(date, date_format) - datetime.timedelta(weeks=1), date_format))
        
        # add predictions from all models for the updated surveillance target dates to a single dataframe
        predictionsall = pd.DataFrame()
        new_datelocs = set(_ for _ in new_records[['date','location']].itertuples(index=False, name=None))
        dateloc_lookup = {date: set() for date in update_target_dates} #lookup locations with updated surv
        for item in new_datelocs: dateloc_lookup[item[0]].add(item[1])
        
        def read_preds_csv(model, date, ext):
            '''
            Read csv predictions for new surveillance data in update mode.
            Assumes update target dates are recorded and dateloc_lookup is calculated.
            Uses file tracking coordinated with GitHub Actions and data retrieval bash script.
            '''
            try:
                predictions = pd.read_csv(f'./data/predictions/{model}/{date}-{model}{ext}', dtype={'location':object})
                predictions['Model'] = model
                predictions = predictions[predictions.target_end_date.isin(update_target_dates)] #filter for updated target dates
                predictions = pd.DataFrame([row for row in predictions.itertuples(index=False) 
                                            if row.location in dateloc_lookup[row.target_end_date]]) #filter for locations with updated surv
                return predictions
            except Exception:
                return
                
        def read_preds_pq(model, date, ext):
            '''
            Read parquet predictions for new surveillance data in update mode.
            Assumes update target dates are recorded and dateloc_lookup is calculated.
            Uses file tracking coordinated with GitHub Actions and data retrieval bash script.
            '''
            try:
                predictions = pd.read_parquet(f'./data/predictions/{model}/{date}-{model}{ext}')
                predictions['Model'] = model
                predictions = predictions[predictions.target_end_date.isin(update_target_dates)] #filter for updated target dates
                predictions = pd.DataFrame([row for row in predictions.itertuples(index=False) 
                                            if row.location in dateloc_lookup[row.target_end_date]]) #filter for locations with updated surv
                return predictions
            except Exception:
                return
                
        with mp.Pool() as pool:
            import os
            print(f'Reading predictions for updated surveillance, {len(os.sched_getaffinity(0))} cores available...', flush=True)
            
            a = [all_models, update_reference_dates, [".csv",".gz",".zip",".csv.zip",".csv.gz"]]
            arguments = list(itertools.product(*a))
            try:
                preds = pd.concat(pool.starmap(read_preds_csv, arguments))
                predictionsall = pd.concat([predictionsall, preds]).drop_duplicates().reset_index(drop=True)
            except ValueError as e:
                print(f'{e}\nIf error \"All objects passed were None\" no csv files found', flush=True)
            
            a = [all_models, update_reference_dates, ['.parquet','.pq',".gz",".zip"]]
            arguments = list(itertools.product(*a))
            try:
                preds = pd.concat(pool.starmap(read_preds_pq, arguments))
                predictionsall = pd.concat([predictionsall, preds]).drop_duplicates().reset_index(drop=True)
            except ValueError as e:
                print(f'{e}\nIf error \"All objects passed were None\" no parquet files found', flush=True)
    
    # add new/changed predictions files to the dataframe
    print('Reading updated forecasts...')
    for filename in updated_forecasts.file:
        model = filename.split('/')[2]
        date = '-'.join(filename.split('/')[-1].split('-', 3)[:3])
        # ensure baseline is present, needed for WIS ratio, so far these are only published as .csv so not checking other extensions
        try:
            predictions = pd.read_csv(f'./data/predictions/FluSight-baseline/{date}-FluSight-baseline.csv', dtype={'location':object})
            predictions['Model'] = 'FluSight-baseline'
            predictionsall = pd.concat([predictionsall, predictions]).drop_duplicates().reset_index(drop=True)
        except Exception:
            pass
        # read updated predictions files
        try:
            predictions = pd.read_csv(filename, dtype={'location':object})
            predictions['Model'] = model
            predictionsall = pd.concat([predictionsall, predictions]).drop_duplicates().reset_index(drop=True)
        except Exception:
            try:
                predictions = pd.read_parquet(filename)
                predictions['Model'] = model
                predictionsall = pd.concat([predictionsall, predictions]).drop_duplicates().reset_index(drop=True)
            except Exception:
                continue
                                          
elif mode == 'scratch':    
    # read files for specified models and dates directly from the flusight repo folder
    surv = pd.read_csv('./FluSight-forecast-hub/target-data/target-hospital-admissions.csv')
    if args.models[0] == 'all': models = all_models
    else: models = args.models
    if args.dates[0] == 'all': dates = pd.unique(surv.date)
    else: dates = args.dates
    
    # ensure baseline is present, needed for WIS ratio
    models = set(models)
    models.add('FluSight-baseline')
    models = list(models)
    
    # read files
    predictionsall = pd.DataFrame()
    
    def read_preds_csv(model, date, ext):
        '''
        Read csv predictions in scratch mode.
        Reads directly from FluSight repo.
        '''
        try:
            predictions = pd.read_csv(f'./FluSight-forecast-hub/model-output/{model}/{date}-{model}{ext}', dtype={'location':object})
            predictions['Model'] = model
            return predictions
        except Exception:
            return
            
    def read_preds_pq(model, date, ext):
        '''
        Read parquet predictions in scratch mode.
        Reads directly from FluSight repo.
        '''
        try:
            predictions = pd.read_parquet(f'./FluSight-forecast-hub/model-output/{model}/{date}-{model}{ext}')
            predictions['Model'] = model
            return predictions
        except Exception:
            return
            
    with mp.Pool() as pool:
        import os
        print(f'{len(os.sched_getaffinity(0))} cores available', flush=True)
        
        a = [models, dates, [".csv",".gz",".zip",".csv.zip",".csv.gz"]]
        arguments = list(itertools.product(*a))
        try:
            preds = pd.concat(pool.starmap(read_preds_csv, arguments))
            predictionsall = pd.concat([predictionsall, preds]).drop_duplicates().reset_index(drop=True)
        except ValueError as e:
            print(f'{e}\nIf error \"All objects passed were None\" no csv files found', flush=True)
        
        a = [models, dates, ['.parquet','.pq',".gz",".zip"]]
        arguments = list(itertools.product(*a))
        try:
            preds = pd.concat(pool.starmap(read_preds_pq, arguments))
            predictionsall = pd.concat([predictionsall, preds]).drop_duplicates().reset_index(drop=True)
        except ValueError as e:
            print(f'{e}\nIf error \"All objects passed were None\" no parquet files found', flush=True)

elif mode == 'archive':
    surv = pd.read_csv('./FluSight-forecast-hub/target-data/target-hospital-admissions.csv')

    # read files
    predictionsall = pd.DataFrame()

    def read_preds_csv(filename):
        '''
        Read csv predictions in archive mode.
        '''
        model = filename.split('/')[-2]
        try:
            predictions = pd.read_csv(filename, dtype={'location':object})
            predictions['Model'] = model
            return predictions
        except Exception:
            return
            
    with mp.Pool() as pool:
        import os
        print(f'{len(os.sched_getaffinity(0))} cores available', flush=True)
        
        predsarchive = [f for f in glob.glob('./data/predictions-archive-2021-2023/*/*.csv')] #all files in archive are .csv
        
        try:
            preds = pd.concat(pool.map(read_preds_csv, predsarchive))
            predictionsall = pd.concat([predictionsall, preds]).drop_duplicates().reset_index(drop=True)
        except Exception as e:
            print(e, flush=True)
            
    # make format match new data
    predictionsall.dropna(inplace=True)
    predictionsall['horizon'] = predictionsall.target.str.split(' ', n=1, expand=True)[0].astype(int) - 1
    predictionsall['target'] = 'wk inc flu hosp'
    predictionsall.rename(columns={'forecast_date':'reference_date', 'type':'output_type', 'quantile':'output_type_id'}, inplace=True)
    predictionsall['reference_date'] = pd.to_datetime(predictionsall.reference_date, format='%Y-%m-%d') + datetime.timedelta(days=5)
    predictionsall['reference_date'] = predictionsall.reference_date.astype(str)
    
    
print('Data reading completed.')


### CALCULATE SCORES
#################################################

### Instantiate Forecast_Eval Class and Format Data for Scoring
# format forecasts in order to calculate scores
surv['Unnamed: 0'] = 0 # needed for Forecast_Eval methods
test = Forecast_Eval(df=pd.DataFrame(), obsdf=surv, target='hosp')
predsall = test.format_forecasts_all(dfformat = predictionsall)
predsall = predsall[predsall.target=='wk inc flu hosp']
del predictionsall
print(f'Predictions to score:\n{predsall}')

### WIS
# calculate WIS for all forecasts
print('Calculating WIS...')

def batch_wis(model, date, loc, horizon, verbose=False):
    '''
    Calculate WIS for given model, date, location, and horizon.
    Assumes Scoring class is defined and formatted predsall dataframe is available.
    
    Arguments:
      model   - model name as string
      date    - reference date as formatted in predsall
      loc     - location code as string
      horizon - horizon as int
      verbose - if True prints message if data exist and scoring is completed
    
    Returns:
      formatted dataframe of scores
    '''
    # filter by horizon, model and submission date
    pred = predsall[(predsall.horizon==horizon) & (predsall.Model == model) & \
                    (predsall.reference_date == date) & (predsall.location==loc)]

    test = Scoring(df=pred, obsdf=surv, target='hosp')
    predss = test.process_predictions(pred, t_col = 'target_end_date', quantile_col = 'output_type_id')

    if len(predss) == 0: return

    obs = test.get_observations(loc)
    obs = obs[obs.date==pred.target_end_date.unique()[0]]

    if len(obs) == 0: return

    out = test.timestamp_wis(obs, predss)
    
    if verbose: print(f'WIS completed {model} {date} location {loc} horizon {horizon}', flush=True)

    return out

dfwis = pd.DataFrame()
with mp.Pool() as pool:
    import os
    print(f'{len(os.sched_getaffinity(0))} cores available', flush=True)
    report_memory()
    arguments = set(_ for _ in predsall[['Model','reference_date','location','horizon']].itertuples(index=False, name=None))
    scores = pool.starmap(batch_wis, arguments)
    dfwis = pd.concat(scores)

# save to csv
print('Saving WIS to file...')
if mode == 'update':
    old_df = pd.read_csv('./evaluations/WIS.csv', parse_dates=['target_end_date'])
    
    # filter out duplicate scores
    all_df = pd.merge(dfwis, old_df, on=dfwis.columns.tolist(), how='left', indicator='exists')
    is_new = np.where(all_df.exists == 'both', False, True) # True if row in dfwis does not exist in old_df
    new_df = dfwis[is_new]
    
    # filter out scores which are being replaced with new scores
    trunc_old_df = old_df.iloc[:,:5]
    trunc_new_df = new_df.iloc[:,:5]
    all_df = pd.merge(trunc_old_df, trunc_new_df, on=trunc_new_df.columns.tolist(), how='left', indicator='exists')
    retain_rows = np.where(all_df.exists == 'both', False, True) # True if row in old_df should be retained (is not updated)
    old_df = old_df[retain_rows]

    # save the updated scores
    dfwis = pd.concat([old_df, new_df])
    dfwis.to_csv('./evaluations/WIS.csv', index=False, mode='w')
    
elif mode == 'scratch':
    dfwis.to_csv('./scratch/WIS.csv', index=False, mode='w')
    
elif mode == 'archive':
    # save only archive scores to separate file
    dfwis.to_csv('./evaluations/archive-2021-2023/WIS.csv', index=False, mode='w')
    
    # insert archive scores into main evals file
    old_df = pd.read_csv('./evaluations/WIS.csv', parse_dates=['target_end_date'])
    all_df = pd.concat([old_df, dfwis]).drop_duplicates().reset_index(drop=True)
    all_df.to_csv('./evaluations/WIS.csv', index=False, mode='w')


### WIS Ratio
# compute wis ratio, comparing the Flusight models' forecast scores to the Flusight baseline model
# divide flusight models by flusight baseline WIS scores at each location, week, horizon, location
print('Calculating WIS ratio...')
baseline = dfwis[dfwis.Model == 'FluSight-baseline'] 
baseline = baseline.rename(columns={'wis':'wis_baseline', 'Model':'baseline'})
dfwis_test = dfwis[dfwis.Model != 'FluSight-baseline']

dfwis_ratio = pd.merge(dfwis_test, baseline, how='inner',
                       on = ['location', 'target_end_date', 'horizon', 'reference_date'])

# calculate wis ratio
dfwis_ratio['wis_ratio'] = dfwis_ratio['wis']/dfwis_ratio['wis_baseline']

# save to csv
print('Saving WIS ratio to file...')
if mode == 'update':
    old_df = pd.read_csv('./evaluations/WIS_ratio.csv', parse_dates=['target_end_date'])
    
    # filter out duplicate scores
    all_df = pd.merge(dfwis_ratio, old_df, on=dfwis_ratio.columns.tolist(), how='left', indicator='exists')
    is_new = np.where(all_df.exists == 'both', False, True) # True if row in dfwis_ratio does not exist in old_df
    new_df = dfwis_ratio[is_new]
    
    # filter out scores which are being replaced with new scores
    trunc_old_df = old_df.iloc[:,:5]
    trunc_new_df = new_df.iloc[:,:5]
    all_df = pd.merge(trunc_old_df, trunc_new_df, on=trunc_new_df.columns.tolist(), how='left', indicator='exists')
    retain_rows = np.where(all_df.exists == 'both', False, True) # True if row in old_df should be retained (is not updated)
    old_df = old_df[retain_rows]

    # save the updated scores
    dfwis_ratio = pd.concat([old_df, new_df])
    dfwis_ratio.to_csv('./evaluations/WIS_ratio.csv', index=False, mode='w')
    
elif mode == 'scratch':
    dfwis_ratio.to_csv('./scratch/WIS_ratio.csv', index=False, mode='w')

elif mode == 'archive':
    # save only archive scores to separate file
    dfwis_ratio.to_csv('./evaluations/archive-2021-2023/WIS_ratio.csv', index=False, mode='w')
    
    # insert archive scores into main evals file
    old_df = pd.read_csv('./evaluations/WIS_ratio.csv', parse_dates=['target_end_date'])
    all_df = pd.concat([old_df, dfwis_ratio]).drop_duplicates().reset_index(drop=True)
    all_df.to_csv('./evaluations/WIS_ratio.csv', index=False, mode='w')
    
    
### Coverage
# calculate coverage for all forecasts
print('Calculating coverage...')

def batch_coverage(model, date, loc, horizon, verbose=False):
    '''
    Calculate coverage for given model, date, location, and horizon.
    Assumes Scoring class is defined and formatted predsall dataframe is available.
    
    Arguments:
      model   - model name as string
      date    - reference date as formatted in predsall
      loc     - location code as string
      horizon - horizon as int
      verbose - if True prints message if data exist and scoring is completed
    
    Returns:
      formatted dataframe of scores
    '''
    # filter by model and submission date, only look at horizon 0-3
    pred = predsall[(predsall.Model == model)& (predsall.reference_date == date) &\
                    (predsall.horizon == horizon) & (predsall.location == loc)]

    if len(pred) == 0: return

    test = Scoring(df=pred, obsdf=surv, target='hosp')
    predss = test.process_predictions(pred, t_col = 'target_end_date', quantile_col = 'output_type_id')

    obs = test.get_observations(loc)
    obs = test.process_observations(obs[obs.date.isin(pred.target_end_date.unique())])

    if len(obs) == 0: return
    try:
        out = test.all_coverages_from_df(obs, predss)
    except Exception as e:
        print(f'{e} encountered in {model} {date} location {loc} horizon {horizon}')
        return

    out['horizon'] = horizon
    out['Model'] = model
    out['reference_date'] = date
    out['location'] = loc

    if verbose: print(f'Coverage completed {model} {date} location {loc} horizon {horizon}', flush=True)
    
    return pd.DataFrame(out,index=[0])

dfcoverage = pd.DataFrame()
with mp.Pool() as pool:
    import os
    print(f'{len(os.sched_getaffinity(0))} cores available', flush=True)
    report_memory()
    arguments = set(_ for _ in predsall[['Model','reference_date','location','horizon']].itertuples(index=False, name=None))
    scores = pool.starmap(batch_coverage, arguments)
    dfcoverage = pd.concat(scores)
dfcoverage = dfcoverage.reset_index().drop(columns='index')

# save to csv
print('Saving coverage to file...')
if mode == 'update':
    old_df = pd.read_csv('./evaluations/coverage.csv')
    
    # filter out duplicate scores
    all_df = pd.merge(dfcoverage, old_df, on=dfcoverage.columns.tolist(), how='left', indicator='exists')
    is_new = np.where(all_df.exists == 'both', False, True) # True if row in dfcoverage does not exist in old_df
    new_df = dfcoverage[is_new]
    
    # filter out scores which are being replaced with new scores
    trunc_old_df = old_df.iloc[:,11:]
    trunc_new_df = new_df.iloc[:,11:]
    all_df = pd.merge(trunc_old_df, trunc_new_df, on=trunc_new_df.columns.tolist(), how='left', indicator='exists')
    retain_rows = np.where(all_df.exists == 'both', False, True) # True if row in old_df should be retained (is not updated)
    old_df = old_df[retain_rows]

    # save the updated scores
    dfcoverage = pd.concat([old_df, new_df])
    dfcoverage.to_csv('./evaluations/coverage.csv', index=False, mode='w')
    
elif mode == 'scratch':
    dfcoverage.to_csv('./scratch/coverage.csv', index=False, mode='w')

elif mode == 'archive':
    # save only archive scores to separate file
    dfcoverage.to_csv('./evaluations/archive-2021-2023/coverage.csv', index=False, mode='w')

    # insert archive scores into main evals file
    old_df = pd.read_csv('./evaluations/coverage.csv')
    all_df = pd.concat([old_df, dfcoverage]).drop_duplicates().reset_index(drop=True)
    all_df.to_csv('./evaluations/coverage.csv', index=False, mode='w')
    

### MAPE
# calculate MAPE for all forecasts
print('Calculating MAPE...')

def batch_mape(model, date, horizon, verbose=False):
    '''
    Calculate MAPE for given model, date, and horizon.
    Assumes Scoring class is defined and formatted predsall dataframe is available.
    
    Arguments:
      model   - model name as string
      date    - reference date as formatted in predsall
      horizon - horizon as int
      verbose - if True prints message if data exist and scoring is completed
    
    Returns:
      formatted dataframe of scores
    '''
    # filter by horizon, model and submission date
    pred = predsall[(predsall.horizon==horizon) & (predsall.Model == model) & \
                    (predsall.reference_date == date)]
    
    if len(pred)==0: return
    
    # calculate mape for each week
    test = Scoring(df=pred, obsdf=surv, target='hosp')

    out = test.get_mape()
    
    out['horizon'] = horizon
    out['reference_date'] = date
    
    if verbose: print(f'MAPE completed {model} {date} horizon {horizon}', flush=True)
    
    return out

dfmape = pd.DataFrame()
with mp.Pool() as pool:
    import os
    print(f'{len(os.sched_getaffinity(0))} cores available', flush=True)
    report_memory()
    arguments = set(_ for _ in predsall[['Model','reference_date','horizon']].itertuples(index=False, name=None))
    scores = pool.starmap(batch_mape, arguments)
    dfmape = pd.concat(scores)         

# save to csv
print('Saving MAPE to file...')
if mode == 'update':
    old_df = pd.read_csv('./evaluations/MAPE.csv')
    
    # filter out duplicate scores
    all_df = pd.merge(dfmape, old_df, on=dfmape.columns.tolist(), how='left', indicator='exists')
    is_new = np.where(all_df.exists == 'both', False, True) # True if row in dfmape does not exist in old_df
    new_df = dfmape[is_new]
    
    # filter out scores which are being replaced with new scores
    trunc_old_df = old_df[['Model','Location','horizon','reference_date']]
    trunc_new_df = new_df[['Model','Location','horizon','reference_date']]
    all_df = pd.merge(trunc_old_df, trunc_new_df, on=trunc_new_df.columns.tolist(), how='left', indicator='exists')
    retain_rows = np.where(all_df.exists == 'both', False, True) # True if row in old_df should be retained (is not updated)
    old_df = old_df[retain_rows]

    # save the updated scores
    dfmape = pd.concat([old_df, new_df])
    dfmape.to_csv('./evaluations/MAPE.csv', index=False, mode='w')
    
elif mode == 'scratch':
    dfmape.to_csv('./scratch/MAPE.csv', index=False, mode='w')

elif mode == 'archive':
    # save only archive scores to separate file
    dfmape.to_csv('./evaluations/archive-2021-2023/MAPE.csv', index=False, mode='w')

    # insert archive scores into main evals file
    old_df = pd.read_csv('./evaluations/MAPE.csv')
    all_df = pd.concat([old_df, dfmape]).drop_duplicates().reset_index(drop=True)
    all_df.to_csv('./evaluations/MAPE.csv', index=False, mode='w')











