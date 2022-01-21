# Branching standards & conventions

## Quick legend

| Branch | Instance | Description, instructions, notes |
| ------ | -------- | -------------------------------- |
| stable | Stable | Accepts merges from testing ONLY |
| testing | Test | Working branch, Accepts merges from Features/Bugs branches ONLY |
| feature_* <br/> bug_* | Features/Bugs | Always branch off HEAD of Test |

## Main branches

The main repository will always hold two evergreen branches:

- ```stable```
- ```testing```

The main working branch should be considered ```origin/testing``` and will be the main branch where the source code of HEAD always reflects a state with the latest delivered development changes for the next release. As a developer, you will be branching and merging from ```testing```.

Consider origin/stable to always represent the latest code deployed to production. During day to day development, the stable branch will not be interacted with.

When the source code in the ```testing``` branch is stable and has been deployed, all of the changes will be merged into stable and tagged with a release number. How this is done in detail will be discussed later.

## Supporting Branches

Supporting branches are used to aid parallel development between team members, ease tracking of features, and to assist in quickly fixing live production problems. Unlike the main branches, these branches always have a limited life time, since they will be removed eventually.

The different types of branches we may use are:

- Feature branches
- Bug branches

Each of these branches have a specific purpose and are bound to strict rules as to which branches may be their originating branch and which branches must be their merge targets. Each branch and its usage is explained below.

#### Feature/Bug Branches

Feature/Bug branches are used when developing a new feature or enhancement which has the potential of a development lifespan longer than a single deployment. When starting development, the deployment in which this feature will be released may not be known. No matter when the feature/bug branch will be finished, it will always be merged back into the ```testing``` branch.

```<ticket number>``` represents the GitHub/JIRA ticket number to which Project Management will be tracked.

Must branch from: ```testing```
Must merge back into: ```testing```
Branch naming convention: ```feature_<ticket number>``` or ```bug_<ticket number>```

#### Working with a feature/bug branch

Create a branch off HEAD of ```testing``` and work on the ticket. Once the development is complete with unit tests, raise a pull request with review to merge with the ```testing``` branch.
