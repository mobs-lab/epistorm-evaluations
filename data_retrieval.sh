# Model Names Listed Here
team_names=($(ls -d FluSight-forecast-hub/model-output/*/ | xargs -n 1 basename))

NEW_PREDICTION_DATA_COPIED=false
NEW_SURVEILLANCE_DATA_COPIED=false

PREDICTION_DATA_SOURCE_LOCATION='FluSight-forecast-hub/model-output'
PREDICTION_DATA_TARGET_LOCATION='data/predictions'

SURVEILLANCE_DATA_SOURCE_LOCATION='FluSight-forecast-hub/target-data'
SURVEILLANCE_DATA_TARGET_LOCATION='data/ground-truth'
SURVEILLANCE_DATA_FILE_NAME='target-hospital-admissions.csv'

PREDICTION_UPDATE_TRACKING_FILE='updated_forecasts.csv'
PREDICTION_UPDATE_LOG_FILE='updated_forecasts_log.csv'

#region Check if new model predictions are available and copy them over if yes
printf "file" > "$PREDICTION_UPDATE_TRACKING_FILE" # Init file update tracking
printf 'date +"%Y-%m-%d %H:%M:%S"' >> PREDICTION_UPDATE_LOG_FILE
for team in "${team_names[@]}"; do
  echo "Checking for new files from $team..."

  # Make sure each model has a subdirectory
  mkdir -p "$PREDICTION_DATA_TARGET_LOCATION/$team"
  
  # Iterate through all the models on CDC's source
  for file in "$PREDICTION_DATA_SOURCE_LOCATION/$team"/*; do
    filename=$(basename "$file")

    # Check if the file exists in the target directory
    # If not, it is new, so we copy it over
    if [ ! -f "$PREDICTION_DATA_TARGET_LOCATION/$team/$filename" ]; then
      cp "$file" "$PREDICTION_DATA_TARGET_LOCATION/$team/"
      echo "Copied $filename to $PREDICTION_DATA_TARGET_LOCATION/$team/"
      printf "\n$PREDICTION_DATA_TARGET_LOCATION/$team/$filename" >> "$PREDICTION_UPDATE_TRACKING_FILE"
      printf "\n$PREDICTION_DATA_TARGET_LOCATION/$team/$filename" >> "$PREDICTION_UPDATE_LOG_FILE"
      NEW_PREDICTION_DATA_COPIED=true
    else
      # Check if the file has been updated and if so, copy it
      if ! cmp -s "$file" "$PREDICTION_DATA_TARGET_LOCATION/$team/$filename"; then
        echo "Detected new version of $filename, copying..."
        rm "$PREDICTION_DATA_TARGET_LOCATION/$team/$filename" # Remove the old file
        cp "$file" "$PREDICTION_DATA_TARGET_LOCATION/$team/" # Copy the new file over
        echo "Copied $filename into $PREDICTION_DATA_TARGET_LOCATION/$team/."
        printf "\n$PREDICTION_DATA_TARGET_LOCATION/$team/$filename" >> "$PREDICTION_UPDATE_TRACKING_FILE"
        printf "\n$PREDICTION_DATA_TARGET_LOCATION/$team/$filename" >> "$PREDICTION_UPDATE_LOG_FILE"
        NEW_PREDICTION_DATA_COPIED=true
      fi
    fi
  done
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
  echo "Checking for new surveillance data..."
  cp --remove-destination "$SURVEILLANCE_DATA_TARGET_LOCATION/$SURVEILLANCE_DATA_FILE_NAME"{,_old} # Preserve old file for comparison
  if ! cmp -s "$SURVEILLANCE_DATA_SOURCE_LOCATION/$SURVEILLANCE_DATA_FILE_NAME" "$SURVEILLANCE_DATA_TARGET_LOCATION/$SURVEILLANCE_DATA_FILE_NAME"; then
    echo "Detected new version of $SURVEILLANCE_DATA_FILE_NAME in source, copying and preserving old file for comparison..."
    rm "$SURVEILLANCE_DATA_TARGET_LOCATION/$SURVEILLANCE_DATA_FILE_NAME" # Remove the old file
    cp "$SURVEILLANCE_DATA_SOURCE_LOCATION/$SURVEILLANCE_DATA_FILE_NAME" "$SURVEILLANCE_DATA_TARGET_LOCATION/$SURVEILLANCE_DATA_FILE_NAME" # Copy the new file over
    echo "Copied $SURVEILLANCE_DATA_FILE_NAME into our $SURVEILLANCE_DATA_TARGET_LOCATION."
    NEW_SURVEILLANCE_DATA_COPIED=true
  fi
fi
#endregion

# Export the environment variables to be used by the CI/CD Pipeline
{
    echo "NEW_PREDICTION_DATA_COPIED=$NEW_PREDICTION_DATA_COPIED"
    echo "NEW_SURVEILLANCE_DATA_COPIED=$NEW_SURVEILLANCE_DATA_COPIED"
} >> $GITHUB_ENV
