#!/bin/bash

# exit on error
set -e

# default values
COLLECTION="cyber_arxiv"
QUERY="(cybersecurity) OR (cyber security)"
DAYS_PER_BATCH=14
END_DATE=$(date +%Y-%m-%d)
START_DATE=$(date -d "2 months ago" +%Y-%m-%d)

# Function to check if first date is less than or equal to second date
date_le() {
    [ "$(date -d "$1" +%s)" -le "$(date -d "$2" +%s)" ]
}

# Parse command line arguments
while getopts "s:e:q:c:d:" opt; do
    case $opt in
        s) START_DATE=$OPTARG ;;
        e) END_DATE=$OPTARG ;;
        q) QUERY=$OPTARG ;;
        c) COLLECTION=$OPTARG ;;
        d) DAYS_PER_BATCH=$OPTARG ;;
        ?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
    esac
done

# initialize current date as start date
current_date=$START_DATE

echo "starting historical ingestion from $START_DATE to $END_DATE"
echo "using query: $QUERY"
echo "collection name: $COLLECTION"

# process each period
while date_le "$current_date" "$END_DATE"; do
    # calculate batch end date 
    batch_end_date=$(date -d "$current_date + $DAYS_PER_BATCH days" +%Y-%m-%d)
    
    # if batch_end_date is beyond END_DATE, use END_DATE
    if ! date_le "$batch_end_date" "$END_DATE"; then
        batch_end_date=$END_DATE
    fi
    
    echo "processing period: $current_date to $batch_end_date"
    
    # run ingestion for this period
    poetry run python3 arxiv_pipeline.py ingest \
        --content_query="$QUERY" \
        --days_back=$DAYS_PER_BATCH \
        --end_date="$batch_end_date" \
        --collection_name="$COLLECTION"
    
    # add delay to be nice to ArXiv API
    echo "waiting 30 seconds before next batch..."
    sleep 30

    if [ "$current_date" = "$batch_end_date" ]; then
        echo "date not advancing, breaking loop"
        break
    fi
    
    # move to next period
    current_date=$batch_end_date
done
