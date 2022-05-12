#########
Using git
#########

If you are going to be editing the FETCH3 code, you should work from your own fork
so that you can easily merge changes from the FETCH3 repository as it is updated.

Some instructions for merging upstream changes to your own branch:

https://www.atlassian.com/git/tutorials/git-forks-and-upstreams

Note that the default branch of fetch3 is called `develop`, so from your branch that
you want to merge changes into, you will run::

    git fetch upstream
    git merge upstream/develop
