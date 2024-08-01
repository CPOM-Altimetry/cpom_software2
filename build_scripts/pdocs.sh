#!/bin/sh

# Get the current Git branch
current_branch=$(git rev-parse --abbrev-ref HEAD)

# Check if the current branch is master
if [ "$current_branch" = "main" ]; then
echo "on $current_branch branch"
pdoc -o docs src/cpom --no-include-undocumented --mermaid --logo "https://www.homepages.ucl.ac
.uk/~ucasamu/cpom_logo.gif" --docformat google 
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "pdoc failed"
    exit 1
fi
git add docs

else
    echo "Not on main branch, skipping pdocs build."
fi

exit 0

