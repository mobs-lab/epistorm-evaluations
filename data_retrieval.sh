# Model Names Listed Here
team_names=('CADPH-FluCAT_Ensemble' 'CEPH-Rtrend_fluH'  'CMU-TimeSeries' 'CU-ensemble' 'FluSight-baseline'
          'FluSight-ensemble' 'FluSight-equal_cat' 'FluSight-lop_norm' 'GH-model' 'GT-FluFNP' 'ISU_NiemiLab-ENS' 
          'ISU_NiemiLab-NLH' 'ISU_NiemiLab-SIR' 'LUcompUncertLab-chimera' 'LosAlamos_NAU-CModel_Flu' 
          'MIGHTE-Nsemble' 'MOBS-GLEAM_FLUH' 'NIH-Flu_ARIMA' 'PSI-PROF' 'SGroup-RandomForest' 'SigSci-CREG' 
          'SigSci-TSENS' 'Stevens-GBR' 'UGA_flucast-Copycat' 'UGA_flucast-INFLAenza' 'UGA_flucast-OKeeffe' 
          'UGuelph-CompositeCurve' 'UGuelphensemble-GRYPHON' 'UM-DeepOutbreak' 'UMass-flusion' 'UMass-trends_ensemble'
          'UNC_IDD-InfluPaint' 'UVAFluX-Ensemble' 'VTSanghani-Ensemble' 'cfa-flumech' 'cfarenewal-cfaepimlight' 
          'fjordhest-ensemble' 'NU_UCSD-GLEAM_AI_FLUH' 'PSI-PROF_beta' 'JHU_CSSE-CSSE_Ensemble' 'FluSight-national_cat'
          'FluSight-ens_q_cat' 'FluSight-baseline_cat' 'FluSight-base_seasonal' 'Gatech-ensemble_point' 'Gatech-ensemble_prob'
          'ISU_NiemiLab-GPE' 'JHUAPL-DMD' 'MDPredict-SIRS' 'MIGHTE-Joint' 'Metaculus-cp' 'NEU_ISI-AdaptiveEnsemble'
          'NEU_ISI-FluBcast' 'OHT_JHU-nbxd' 'SigSci-BECAM' 'Stevens-ILIForecast' 'UGA_CEID-Walk' 'UGA_flucast-Scenariocast'
          'UI_CompEpi-EpiGen' 'UMass-AR2' 'VTSanghani-PRIME')

NEW_PREDICTION_DATA_COPIED=false
NEW_SURVEILLANCE_DATA_COPIED=false

PREDICTION_DATA_SOURCE_LOCATION='FluSight-forecast-hub/model-output'
PREDICTION_DATA_TARGET_LOCATION='data/predictions'

SURVEILLANCE_DATA_SOURCE_LOCATION='FluSight-forecast-hub/target-data'
SURVEILLANCE_DATA_TARGET_LOCATION='data/ground-truth'
SURVEILLANCE_DATA_FILE_NAME='target-hospital-admissions.csv'

PREDICTION_UPDATE_TRACKING_FILE='updated_forecasts.csv'

#region Check if new model predictions are available and copy them over if yes
for team in "${team_names[@]}"; do
  echo "Checking for new files from $team..."

# Make sure each model has a subdirectory
  #if [ -d "$PREDICTION_DATA_SOURCE_LOCATION/$team" ]; then
  mkdir -p "$PREDICTION_DATA_TARGET_LOCATION/$team"

  # Init file update tracking
  printf "file" > "$PREDICTION_UPDATE_TRACKING_FILE"
  
  # Iterate through all the models on CDC's source
  for file in "$PREDICTION_DATA_SOURCE_LOCATION/$team"/*; do
    filename=$(basename "$file")

    # Check if the file exists in the target directory
    # If not, it is new, so we copy it over
    if [ ! -f "$PREDICTION_DATA_TARGET_LOCATION/$team/$filename" ]; then
      cp "$file" "$PREDICTION_DATA_TARGET_LOCATION/$team/"
      echo "Copied $filename to $PREDICTION_DATA_TARGET_LOCATION/$team/"
      printf "\n$PREDICTION_DATA_TARGET_LOCATION/$team/$filename" >> "$PREDICTION_UPDATE_TRACKING_FILE"
      NEW_PREDICTION_DATA_COPIED=true
    else
      # Check if the file has been updated and if so, copy it
      if ! cmp -s "$file" "$PREDICTION_DATA_TARGET_LOCATION/$team/$filename"; then
        echo "Detected new version of $filename, copying..."
        rm "$PREDICTION_DATA_TARGET_LOCATION/$team/$filename" # Remove the old file
        cp "$file" "$PREDICTION_DATA_TARGET_LOCATION/$team/" # Copy the new file over
        echo "Copied $filename into $PREDICTION_DATA_TARGET_LOCATION/$team/."
        printf "\n$PREDICTION_DATA_TARGET_LOCATION/$team/$filename" >> "$PREDICTION_UPDATE_TRACKING_FILE"
        NEW_PREDICTION_DATA_COPIED=true
      fi
    fi
  done
  #else
  #  echo "Error: team subdirectory does not exist. Please make sure subdirectories are set up."
  #fi
  echo
done
#endregion

#region Check if new Surveillance data is available, and copy it over if yes
# Duplicate: Check if target directory is set up for ground truth
if [ ! -d "$SURVEILLANCE_DATA_TARGET_LOCATION" ]; then
  mkdir -p "$SURVEILLANCE_DATA_TARGET_LOCATION"
fi

# NOTE: This set up should only run during initialization of project
if [ ! -f "$SURVEILLANCE_DATA_TARGET_LOCATION/$SURVEILLANCE_DATA_FILE_NAME" ]; then
  echo "Missing required surveillance data file, copying newest one over..."
  cp "$SURVEILLANCE_DATA_SOURCE_LOCATION/$SURVEILLANCE_DATA_FILE_NAME" "$SURVEILLANCE_DATA_TARGET_LOCATION/$SURVEILLANCE_DATA_FILE_NAME"
  echo "Copied target-hospital-admissions.csv to $SURVEILLANCE_DATA_TARGET_LOCATION"
  NEW_SURVEILLANCE_DATA_COPIED=true
else
  if ! cmp -s "$SURVEILLANCE_DATA_SOURCE_LOCATION/$SURVEILLANCE_DATA_FILE_NAME" "$SURVEILLANCE_DATA_TARGET_LOCATION/$SURVEILLANCE_DATA_FILE_NAME"; then
    echo "Detected new version of $SURVEILLANCE_DATA_FILE_NAME in source, copying into compare area..."
    cp --remove-destination "$SURVEILLANCE_DATA_TARGET_LOCATION/$SURVEILLANCE_DATA_FILE_NAME"{,_old} # Preserve old file for comparison
    rm "$SURVEILLANCE_DATA_TARGET_LOCATION/$SURVEILLANCE_DATA_FILE_NAME" # Remove the old file
    cp "$SURVEILLANCE_DATA_SOURCE_LOCATION/$SURVEILLANCE_DATA_FILE_NAME" "$SURVEILLANCE_DATA_TARGET_LOCATION/$SURVEILLANCE_DATA_FILE_NAME" # Copy the new file over
    echo "Copied $SURVEILLANCE_DATA_FILE_NAME into our $SURVEILLANCE_DATA_TARGET_LOCATION, awaiting cleanup of NA rows..."
    NEW_SURVEILLANCE_DATA_COPIED=true
  fi
fi
#endregion

# Export the environment variables to be used by the CI/CD Pipeline
{
    echo "NEW_PREDICTION_DATA_COPIED=$NEW_PREDICTION_DATA_COPIED"
    echo "NEW_SURVEILLANCE_DATA_COPIED=$NEW_SURVEILLANCE_DATA_COPIED"
} >> $GITHUB_ENV
