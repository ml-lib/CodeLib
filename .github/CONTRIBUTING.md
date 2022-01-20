# Contributing to the product

Contributions to this project are welcome. Please follow the procedures outlined below.

### Raising issues/tickets
Please use the below templates when you want to:
- Propose a new [feature](../../../issues/new?template=feature_request.md)
- Report a [bug](../../../issues/new?template=bug_report.md)
- Raise an [issue](../../..//issues)
- Raise an [eda request](../../../issues/new?template=eda_request.md)
- Open a new [user story](../../../issues/new?template=user_story.md)

Please ensure that the tickets have appropriate [labels](../../../labels) and [milestones](../../../milestones).

### Making a contribution by a team member
- Create a new branch from `testing` and name it according to the ticket type and number. Example: `feature_58` or `bug_59`. Ensure to follow [branching guidelines](../docs/Branch.md).
- Ensure to follow [SemVer](https://semver.org/) for versioning.
- Ensure the [directory structure](../docs/Directory_structure.md) is followed.
- Ensure that python codes follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) standards.
- Ensure that codes are properly documented in accordance to [PEP 257](https://www.python.org/dev/peps/pep-0257/) standards.
- Ensure function annotations are followed in accordance to [PEP 484](https://www.python.org/dev/peps/pep-0484/)
standards.
- Ensure all python modules have a [Pylint](https://www.pylint.org/)
rating of 10/10.
- Ensure to profile your modules and move any bottlenecks to a low latency system/module.
- Ensure to add unit tests with corner cases and achieve 100% code coverage.
- Limit the use of third party libraries. If you do have to use them, ensure that it exists in production environment.
- Follow [pull request](PULL_REQUEST_TEMPLATE.md) guidelines.

### Guidelines
1. Be respectful.  All contributions to the product are appreciated and we ask that you respect one another.
2. Be responsible. You are responsible for your pull request submission.
3. Give credit. Any submissions or contributions built on other work (including and not limited to research papers, open source projects, and public code) must be cited with original Author names or attached with information about the original source or work. People should be credited for the work they have done.

Return to [Home](/README.md)
