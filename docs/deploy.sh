#!/bin/bash

# Deploy built docs to gh-pages branch on github.
# Modified from https://gist.github.com/domenic/ec8b0fc8ab45f39403dd

set -e # exit with nonzero exit code if anything fails

dir=`mktemp -d`  # create a temporary directory

# copy output
cp -r _build/html/* $dir/.

# go to the temp directory and create a *new* Git repo
cd $dir
git init

touch .nojekyll  # bypass jekyll on GitHub Pages

# The first and only commit to this new Git repo contains all the
# files present with the commit message "Deploy to GitHub Pages".
git add .
git commit -m "Deploy to GitHub Pages"

# Force push from the current repo's master branch to the remote
# repo's gh-pages branch. (All previous history on the gh-pages branch
# will be lost, since we are overwriting it.)
git push -f git@github.com:kbarbary/nestle.git master:gh-pages
