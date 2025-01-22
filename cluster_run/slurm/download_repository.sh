#!/bin/bash

while getopts u:r:b:t:p:h opt; do
    case $opt in
            u) USER_URL=$OPTARG ;;
            r) REPO_NAME=$OPTARG ;;
            b) BRANCH_NAME=$OPTARG ;;
            t) ACCESS_TOKEN=$OPTARG ;;
            p) PROJ_PATH=$OPTARG ;;
            h)
                echo "--- Help Message ----"
                echo "-u USER URL (e.g. github.com/achtibat/)"
                echo "-r REPOSITORY NAME (e.g. zennit-crp)"
                echo "-b BRANCH NAME (e.g. zennit-crp)"
                echo "-t ACCESS TOKEN must be set in gitlab/github"
                echo "-p (optional) Path into which git is cloning the repository"
                exit;;
            *)
                echo 'Error in command line parsing' >&2
                exit 1
    esac
done


shift "$(( OPTIND - 1 ))"

if [ -z "$USER_URL" ] || [ -z "$REPO_NAME" ] || [ -z "$ACCESS_TOKEN" ]; then
    echo 'Missing -u or -r or -t' >&2
    exit 1
fi

if [[ $USER_URL != */ ]]; then
    echo "Append '/' to provided URL"
    USER_URL="$USER_URL/"
fi

if ! [ -z "$PROJ_PATH" ]; then
    echo "cd to $PROJ_PATH"
    cd $PROJ_PATH || exit 1
    export PYTHONPATH=$PYTHONPATH:$PROJ_PATH
fi

echo "Cloning branch $BRANCH_NAME from repository $USER_URL$REPO_NAME.git with access token $ACCESS_TOKEN"

git clone https://gpu_cluster:$ACCESS_TOKEN@$USER_URL$REPO_NAME.git $REPO_NAME
cd $REPO_NAME
git remote set-url --push origin https://gpu_cluster:$ACCESS_TOKEN@$USER_URL$REPO_NAME.git
git checkout $BRANCH_NAME
git pull

export PYTHONPATH=$PYTHONPATH:${PWD}
