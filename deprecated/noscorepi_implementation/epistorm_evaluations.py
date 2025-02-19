### REQUIREMENTS
#################################################
import pandas as pd
import numpy as np
import datetime
import sys
from epiweeks import Week
from os.path import exists

import warnings
warnings.filterwarnings('ignore')


### DOWNLOAD DATA
#################################################

### Functions to download flusight model predictions and surveillance data
def pull_flusight_predictions(model,date):
    """pull_flusight_predictions. Load predictions of the model saved by the Flusight Forecast hub

    Parameters
    ----------
    model : str
        Model name on the
    dates : list or string
        List of potential dates in the iso format, e.g., 'yyyy-mm-dd', for the submission.
    """
    predictions = None
    
    url = f"https://raw.githubusercontent.com/cdcepi/Flusight-forecast-hub/main/model-output/{model}/{date}-{model}"
    for ext in [".csv",".gz",".zip",".csv.zip",".csv.gz"]:
        try:
            predictions = pd.read_csv(url+ext,dtype={'location':str},parse_dates=['target_end_date'])
        except:
            pass
    if predictions is None:
        print(f"Data for model {model} and date {date} unavailable")
    return predictions


def pull_surveillance_data():
    """pull_surveillance_data. Load hospitalization admissions surveillance data
    """
    
    url = f"https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main/target-data/target-hospital-admissions.csv"
    return pd.read_csv(url, dtype={'location':str})


### Download and save all available surveillance data
surv = pull_surveillance_data()
surv.to_parquet(f"./dat/target-hospital-admissions.pq", index=False)
surv = pd.read_parquet(f"./dat/target-hospital-admissions.pq")
surv['Unnamed: 0'] = 0

# predictions will be downloaded after input handling


### HANDLE INPUTS & DOWNLOAD PREDICTIONS
#################################################
# all existing models as of Dec 2024
all_models = ['CADPH-FluCAT_Ensemble', 'CEPH-Rtrend_fluH',  'CMU-TimeSeries', 'CU-ensemble', 'FluSight-baseline',
          'FluSight-ensemble','FluSight-equal_cat', 'FluSight-lop_norm', 'GH-model', 'GT-FluFNP', 'ISU_NiemiLab-ENS', 
          'ISU_NiemiLab-NLH','ISU_NiemiLab-SIR', 'LUcompUncertLab-chimera', 'LosAlamos_NAU-CModel_Flu', 
          'MIGHTE-Nsemble','MOBS-GLEAM_FLUH', 'NIH-Flu_ARIMA', 'PSI-PROF', 'SGroup-RandomForest', 'SigSci-CREG', 
          'SigSci-TSENS','Stevens-GBR', 'UGA_flucast-Copycat', 'UGA_flucast-INFLAenza', 'UGA_flucast-OKeeffe', 
          'UGuelph-CompositeCurve', 'UGuelphensemble-GRYPHON', 'UM-DeepOutbreak', 'UMass-flusion', 'UMass-trends_ensemble',
          'UNC_IDD-InfluPaint', 'UVAFluX-Ensemble', 'VTSanghani-Ensemble', 'cfa-flumech', 'cfarenewal-cfaepimlight', 
          'fjordhest-ensemble', 'NU_UCSD-GLEAM_AI_FLUH', 'PSI-PROF_beta', 'JHU_CSSE-CSSE_Ensemble', 'FluSight-national_cat',
          'FluSight-ens_q_cat', 'FluSight-baseline_cat', 'FluSight-base_seasonal', 'Gatech-ensemble_point', 'Gatech-ensemble_prob',
          'ISU_NiemiLab-GPE', 'JHUAPL-DMD', 'MDPredict-SIRS', 'MIGHTE-Joint', 'Metaculus-cp', 'NEU_ISI-AdaptiveEnsemble',
          'NEU_ISI-FluBcast', 'OHT_JHU-nbxd', 'SigSci-BECAM', 'Stevens-ILIForecast', 'UGA_CEID-Walk', 'UGA_flucast-Scenariocast',
          'UI_CompEpi-EpiGen', 'UMass-AR2', 'VTSanghani-PRIME']

# Accept inputs for:
# mode - 'most_recent' or 'recalculate_all'
# output_directory - 'evaluations' or 'scratch'
# models - any number of model names in a space-separated string
n_in = len(sys.argv)
if n_in == 1: # no arguments input
    mode = 'most_recent'
    output_directory = 'evaluations'
    models = all_models
elif n_in == 3: # mode and output specified
    mode = sys.argv[1]
    output_directory = sys.argv[2]
    models = all_models
elif n_in >= 4: # mode, output, and model(s) specified
    mode = sys.argv[1]
    output_directory = sys.argv[2]
    models = sys.argv[2:]
else:
    for arg in sys.argv: print(arg)
    raise ValueError(f'incorrect number of arguments received: {n_in}')

# mode
if mode == 'most_recent': surv = surv[surv.date == surv.date.max()]
elif mode != 'recalculate_all': raise ValueError(f'improper mode input: {mode}')

# output_directory and write_mode
write_mode = 'w' # determine whether to append to or overwrite files
if output_directory == 'evaluations': 
    output_directory = './evaluations/'
    if mode == 'most_recent': write_mode = 'a'
elif output_directory == 'scratch': output_directory = './scratch/'
else: raise ValueError(f'improper output_directory input: {output_directory}'))

# model name validation is handled by pull_flusight_predictions()

# selecting all target dates that exist in the surveillance file
dates = pd.unique(surv.date)

# download and save forecasts for specified submission week (date) and for specified models from flusight github
for model in models:
    for date in dates:
        if exists(f'./dat/{model}_{date}.pq'): continue
        else:
            try:
                predictions = pull_flusight_predictions(model,date)
                predictions.to_parquet(f'./dat/{model}_{date}.pq', index=False)
            except Exception as e: print(e)


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
            raise ValueError("vector shape mismatch")

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
                    
                    if len(predvals) == 0:
                        continue

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


### CALCULATE SCORES
#################################################

### Instantiate Forecast_Eval Class and Format Data for Scoring
# put all forecasts into one dataframe
predictionsall = pd.DataFrame()
for model in models:
    for date in dates:
        try:
            predictions = pd.read_parquet(f'./dat/{model}_{date}.pq')
            predictions['Model'] = model
            predictionsall = pd.concat([predictionsall, predictions])
        except Exception as e:
            print(e)
            
# format forecasts in order to calculate scores
# input start and end weeks for the period of interest
start_week = Week.fromdate(datetime.datetime.strptime(surv.date.min(), '%Y-%m-%d'))
end_week = Week.fromdate(datetime.datetime.strptime(surv.date.max(), '%Y-%m-%d'))
test = Forecast_Eval(df=pd.DataFrame(), obsdf=surv, target='hosp', 
                            start_week = start_week, end_week = end_week)
predsall = test.format_forecasts_all(dfformat = predictionsall)

### WIS
# calculate WIS for all forecasts
dfwis = pd.DataFrame()
for horizon in [0, 1, 2, 3]:
    for model in models:
        for date in dates: 
            for loc in predsall.location.unique():
                start_week = Week.fromdate(pd.to_datetime(date)) # week of submission date
                end_week = start_week + 3 # target end date of last horizon

                # filter by horizon, model and submission date
                pred = predsall[(predsall.horizon==horizon) & (predsall.Model == model) & \
                                (predsall.reference_date == date) & (predsall.location==loc)]

                test = Scoring(df=pred, obsdf=surv, target='hosp')
                predss = test.process_predictions(pred, t_col = 'target_end_date', quantile_col = 'output_type_id')

                if len(predss) == 0: continue

                obs = test.get_observations(loc)
                obs = obs[obs.date==pred.target_end_date.unique()[0]]
                
                out = test.timestamp_wis(obs, predss)

                dfwis = pd.concat([dfwis, out])

# save to csv
if write_mode == 'a': # if appending, ensure there are unseen rows to append
    old_df = pd.read_csv('./evaluations/WIS.csv')
    all_df = pd.merge(dfwis, old_df, on=dfwis.columns.tolist(), how='left', indicator='exists')
    is_new = np.where(all_df.exists == 'both', False, True) # True if row in dfwis does not exist in old_df
    dfwis = dfwis[is_new]
    del old_df
    del all_df
    if not dfwis.empty: dfwis.to_csv('./evaluations/WIS.csv', index=False, mode=write_mode)
else: dfwis.to_csv('./evaluations/WIS.csv', index=False, mode=write_mode)


### WIS Ratio
# compute wis ratio, comparing the Flusight models' forecast scores to the Flusight baseline model
# divide flusight models by flusight baseline WIS scores at each location, week, horizon, location
dfwis = pd.read_csv('./evaluations/WIS.csv')
baseline = dfwis[dfwis.Model == 'FluSight-baseline'] 
baseline = baseline.rename(columns={'wis':'wis_baseline', 'Model':'baseline'})
dfwis_test = dfwis[dfwis.Model != 'FluSight-baseline']

dfwis_ratio = pd.merge(dfwis_test, baseline, how='inner',
                       on = ['location', 'target_end_date', 'horizon', 'reference_date'])

# calculate wis ratio
dfwis_ratio['wis_ratio'] = dfwis_ratio['wis']/dfwis_ratio['wis_baseline']

# save to csv
if write_mode == 'a': # if appending, ensure there are unseen rows to append
    old_df = pd.read_csv('./evaluations/WIS_ratio.csv')
    all_df = pd.merge(dfwis_ratio, old_df, on=dfwis_ratio.columns.tolist(), how='left', indicator='exists')
    is_new = np.where(all_df.exists == 'both', False, True) # True if row in dfwis_ratio does not exist in old_df
    dfwis_ratio = dfwis_ratio[is_new]
    del old_df
    del all_df
    if not dfwis_ratio.empty: dfwis_ratio.to_csv('./evaluations/WIS_ratio.csv', index=False, mode=write_mode)
else: dfwis_ratio.to_csv('./evaluations/WIS_ratio.csv', index=False, mode=write_mode)


### Coverage
# calculate coverage for all forecasts
dfcoverage = pd.DataFrame()
for date in dates:
    for model in models:
        for loc in predsall.location.unique():
            for horizon in [0,1,2,3]:
                start_week = Week.fromdate(pd.to_datetime(date)) # week of submission date
                end_week = start_week + 3 # target end date of last horizon

                # filter by model and submission date, only look at horizon 0-3
                pred = predsall[(predsall.Model == model)& (predsall.reference_date == date) &\
                                (predsall.horizon == horizon) & (predsall.location == loc)]

                if len(pred) == 0: continue

                test = Scoring(df=pred, obsdf=surv, target='hosp')
                predss = test.process_predictions(pred, t_col = 'target_end_date', quantile_col = 'output_type_id')

                obs = test.get_observations(loc)
                obs = obs[obs.date.isin(pred.target_end_date.unique())]

                out = test.all_coverages_from_df(obs, predss)

                out['horizon'] = horizon
                out['Model'] = model
                out['reference_date'] = date
                out['location'] = loc

                dfcoverage = pd.concat([dfcoverage, pd.DataFrame(out,index=[0])])
dfcoverage = dfcoverage.reset_index().drop(columns='index')

# save to csv
if write_mode == 'a': # if appending, ensure there are unseen rows to append
    old_df = pd.read_csv('./evaluations/coverage.csv')
    all_df = pd.merge(dfcoverage, old_df, on=dfcoverage.columns.tolist(), how='left', indicator='exists')
    is_new = np.where(all_df.exists == 'both', False, True) # true if row in dfcoverage does not exist in old_df
    dfcoverage = dfcoverage[is_new]
    del old_df
    del all_df
    if not dfcoverage.empty: dfcoverage.to_csv('./evaluations/coverage.csv', index=False, mode=write_mode)
else: dfcoverage.to_csv('./evaluations/coverage.csv', index=False, mode=write_mode)


### MAPE
# calculate MAPE for all forecasts
dfmape = pd.DataFrame()
for horizon in [0, 1, 2,3]:
    for model in models:
        for date in dates: 
            start_week = Week.fromdate(pd.to_datetime(date)) # week of submission date
            end_week = start_week + 3 # target end date of last horizon
            
            # filter by horizon, model and submission date
            pred = predsall[(predsall.horizon==horizon) & (predsall.Model == model) & \
                            (predsall.reference_date == date)]
            if len(pred)==0:
                continue
            
            # calculate mape for each week
            test = Scoring(df=pred, obsdf=surv, target='hosp',
                            start_week = start_week, end_week = end_week)

            out = test.get_mape()
            
            out['horizon'] = horizon
            out['reference_date'] = date
            
            dfmape = pd.concat([dfmape, out])
            
# save to csv
if write_mode == 'a': # if appending, ensure there are unseen rows to append
    old_df = pd.read_csv('./evaluations/MAPE.csv')
    all_df = pd.merge(dfmape, old_df, on=dfmape.columns.tolist(), how='left', indicator='exists')
    is_new = np.where(all_df.exists == 'both', False, True) # true if row in dfmape does not exist in old_df
    dfmape = dfmape[is_new]
    del old_df
    del all_df
    if not dfmape.empty: dfmape.to_csv('./evaluations/MAPE.csv', index=False, mode=write_mode)
else: dfmape.to_csv('./evaluations/MAPE.csv', index=False, mode=write_mode)













