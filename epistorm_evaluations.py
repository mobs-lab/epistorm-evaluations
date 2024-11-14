### REQUIREMENTS
#################################################
import pandas as pd
import numpy as np
import datetime
from scorepi import *
import scorepi as sp
from epiweeks import Week


### DOWNLOAD DATA
#################################################

# functions to download flusight model predictions and surveillance data
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


# download and save surveillance data
surv = pull_surveillance_data()
surv.to_parquet(f"./dat/target-hospital-admissions.pq", index=False)


### NOTE: dates and models will need to be changed for automation
# download and save forecasts for specified submission week (date) and for specified models from flusight github
#selecting all target dates that exist in the surveillance file
dates = [ '2024-04-06', '2024-04-13', '2024-04-20', '2024-04-27']
#dates = pd.unique(surv.date)

#selecting just models used in the dashboard for now
#will need to expand eventually whether we keep the parquet implementation or pull files from the flusight repo as a submodule
#models = ['CEPH-Rtrend_fluH', 'FluSight-baseline', 'FluSight-ensemble', 'MIGHTE-Nsemble', 'MOBS-GLEAM_FLUH', 'NU_UCSD-GLEAM_AI_FLUH']
models = ['MOBS-GLEAM_FLUH']

for model in models:
    for date in dates:
        try:
            predictions = pull_flusight_predictions(model,date)

            predictions.to_parquet(f'./dat/{model}_{date}.pq', index=False)
        except Exception as e:
            print(e)


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
        observations = sp.Observations(observations)

        return observations
    
    
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
    uses scorepi package to calculate the scores  (https://github.com/gstonge/scorepi/tree/main)
    score dataframe must have 'Model' column to differentiate and calculate scores for different models
    """
    
    def __init__(self, df, obsdf, target, start_week = False, 
                 end_week = False):
        super().__init__(df, obsdf, target, start_week, end_week)
        
    def get_all_average_scores(self, models):
        """ get_all_average_scores. Calculate all score in scorepi package that average over the full forecast
        time series. 
        
        Parameters
        ----------
        models: list
            list of models that the scores will be calculated for, each element is a string corresponding to
            a forecast model's name from the Model column of the forecast dataframe
        """
        
        pred1 = self.df.copy() # dataframe that will be scored
        loclist = list(pred1.location.unique()) 
        
        allscore = {}
        for model in models:
            allscore[model] = {}
            for target_location in loclist:
                if target_location == '72':
                    continue
                #print(target_location)
                
                observations = self.get_observations(target_location) # get surveillance data for target location 

                # filter by model and location
                pred = pred1[(pred1.Model==model) & (pred1['location']==target_location) ] 
                # make into Predictions object
                pred = Predictions(pred, t_col = 'target_end_date', quantile_col = 'output_type_id')
                observations = Observations(observations[observations.date<=pred.target_end_date.max()])
                #calculate scores
                d,_ = score_utils.all_scores_from_df(observations, pred, mismatched_allowed=False) 

                # save in dictionary
                allscore[model][target_location] = d
            
        
        return allscore
    
    def organize_average_scores(self, want_scores, models):
        """ organize_average_scores. save average scores of interest into a pandas dataframe
        
        Parameters
        ----------
        want_scores: list
            list of scores you want to save in the dataframe
            wis is 'wis_mean', and all coverages are '10_cov', '20_cov', ... '95_cov' etc.
        models: list
            list of models that the scores will be calculated for, each element is a string correspongding to
            a forecast model's name from the Model column of the forecast dataframe. 
            used for get_all_average_scores function call.
        """
        
        average_scores = pd.DataFrame()
        
        allscore = self.get_all_average_scores(models) #calculate all average scores
        
        for model in allscore.keys():
            scoresmod = allscore[model]
            for loc in scoresmod.keys():
                    
                scoresloc = scoresmod[loc]

                scoredict = {'Model': model ,'location': loc}
                for score in want_scores: # only save scores input into want_scores
                    scoredict[score] = scoresloc[score]

                average_scores = pd.concat([average_scores, pd.DataFrame(scoredict, index=[0])])

        average_scores = average_scores.reset_index() 
        average_scores = average_scores.drop(columns=['index'])
        
        return average_scores
    
    def get_all_timestamped_scores(self, models):
        """ get_all_timestamped_scores. Calculate all scores in scorepi package for each week of the full forecast
        time series. 
        
        Parameters
        ----------
        models: list
            list of models that the scores will be calculated for, each element is a string corresponding to
            a forecast model's name from the Model column of the forecast dataframe
        """
        
        pred = self.df.copy() # dataframe used for scoring
        loclist = list(pred.location.unique())
        
        allscore = {}
        
        for model in models:
            allscore[model] = {}
            for target_location in loclist:
                    
                observations = self.get_observations(target_location) # get surveillance data for target location
                
                try:
                    predss = pred[pred['location'] == target_location] #filter by location
                    # format forecasts into Predictions scorepi objec
                    predss = Predictions(predss, t_col = 'target_end_date', quantile_col = 'output_type_id')
                    
                    if len(predss)==0:
                        continue
                    
                    allscore[model][target_location] = {}
                    # loop over all time points in the predictions
                    for t in predss.target_end_date.unique():
                        prednew = predss[predss.target_end_date == t]
                        obsnew = observations[observations.date == t]

                        obsnew = Observations(obsnew)
                        prednew = Predictions(prednew, t_col = 'target_end_date', quantile_col = 'output_type_id')

                        # calculate scores
                        d = score_utils.all_timestamped_scores_from_df(obsnew, prednew)

                        allscore[model][target_location][t] = d
                except Exception as e:
                    print(e)
        
        return allscore
    
    
    def organize_timestamped_scores(self, want_scores, models):
        """ organize_timestamped_scores. save timestamped scores of interest into a pandas dataframe
        
        Parameters
        ----------
        want_scores: list
            list of scores you want to save in the dataframe
            wis is 'wis'
        models: list
            list of models that the scores will be calculated for, each element is a string correspongding to
            a forecast model's name from the Model column of the forecast dataframe. 
            used for get_all_timestamped_scores function call.
        """
        
        time_scores = pd.DataFrame()
        
        # calculate all scores evaluated for each time point
        allscore = self.get_all_timestamped_scores(models=models)
        
        for model in allscore.keys():
            scoremod = allscore[model]
        
            for loc in scoremod.keys():
                    
                scoresloc = scoremod[loc]

                for t in scoresloc.keys():
                    tdf = scoresloc[t]

                    scoredict = {'Model':model ,'location':loc, 'target_end_date':t}
                    for score in want_scores:
                        scoredict[score] = tdf[score]

                    # save scores in want_scores into a dataframe
                    time_scores = pd.concat([time_scores, pd.DataFrame(scoredict, index=[0])])

        time_scores = time_scores.reset_index() 
        time_scores = time_scores.drop(columns=['index'])
        
        return time_scores
    
    
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
                    pred = Predictions(pred, t_col = 'target_end_date',quantile_col='output_type_id')
                    
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
test = Forecast_Eval(df=pd.DataFrame(), obsdf=surv, target='hosp', 
                            start_week = Week(2023,40), end_week = Week(2024, 17))
predsall = test.format_forecasts_all( dfformat = predictionsall)


### WIS

# calculate WIS for all forecasts
dfwis = pd.DataFrame()
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
            
            # calculate wis for each week
            test = Scoring(df=pred, obsdf=surv, target='hosp',
                            start_week = start_week, end_week = end_week)

            out = test.organize_timestamped_scores(want_scores = ['wis'], models = [model])
            
            out['horizon'] = horizon
            out['reference_date'] = date
            
            dfwis = pd.concat([dfwis, out])

# save to csv
dfwis.to_csv('./evaluations/WIS.csv', index=False)#, mode='a')


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
dfwis_ratio.to_csv('./evaluations/WIS_ratio.csv', index=False)#, mode='a')


### Coverage

dfcoverage = pd.DataFrame()
for date in dates:
    for model in models:
         
        start_week = Week.fromdate(pd.to_datetime(date)) # week of submission date
        end_week = start_week + 3 # target end date of last horizon

        # filter by model and submission date, only look at horizon 0-3
        pred = predsall[(predsall.Model == model) & \
                        (predsall.reference_date == date) & (predsall.horizon >=0)]
        if len(pred)==0:
            continue

        # calculate wis for each week
        test = Scoring(df=pred, obsdf=surv, target='hosp',  
                        start_week = start_week, end_week = end_week)

        out = test.organize_average_scores(want_scores=['10_cov', '20_cov', '30_cov', '40_cov', '50_cov',
            '60_cov', '70_cov', '80_cov', '90_cov', '95_cov', '98_cov'], models = [model])

        out['horizon'] = horizon
        out['reference_date'] = date

        dfcoverage = pd.concat([dfcoverage, out])

# save to csv
dfcoverage.to_csv('./evaluations/coverage.csv', index=False)#, mode='a')


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

dfmape.to_csv('./evaluations/MAPE.csv', index=False)#, mode='a')













