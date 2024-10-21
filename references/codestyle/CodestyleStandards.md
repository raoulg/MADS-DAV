# Code style standards

Table of contents

- [0. Overview of topics](#0-overview-of-topics)
- [1. Two cultures](#1-two-cultures)
- [2. The pros and cons of Pythons flexibility](#2-the-pros-and-cons-of-pythons-flexibility)
- [3. Development stages](#3-development-stages)
  - [Testing a concept](#testing-a-concept)
  - [Proof of concept](#proof-of-concept)
  - [Product](#product)
  - [Deployment](#deployment)

## 0. Overview of topics

For every standard, we have a possible classifier. The classifiers are:

- 🐌 : at this stage, this standard might slow you down
- 💡 : at this stage, this standard is probably a good idea
- 🏅 : at this stage, this standard is a must

The topics are ordered by the stage where they are most useful.
The topics cover different subjects; some are more a preference for one library over another, some are about the way you organize your code, and other are about additional tooling like dependency management and linting.

| Topic                                                                            | Testing a Concept | Proof of Concept | Product | Deployment |
| -------------------------------------------------------------------------------- | ----------------- | ---------------- | ------- | ---------- |
| [never hardcode](never_hardcode.md)                                   | 💡                 | 🏅                | 🏅       | 🏅          |
| [Prefer pathlib.Path over os.path](pathlib.md)                                   | 💡                 | 🏅                | 🏅       | 🏅          |
| [Prefer loguru over print](loguru.md)                                            | 💡                 | 🏅                | 🏅       | 🏅          |
| [Use pydantic for all settings](pydantic.md)                                     | 💡                 | 🏅                | 🏅       | 🏅          |
| [pyproject.toml for dependencies](03_dependencies_management.md) | 💡                 | 🏅                | 🏅       | 🏅      |
| [Use cookiecutters](cookiecutter.md)                                             | 💡                 | 🏅                | 🏅       | 🏅          |
| [Git](01_git_basics.md)                                                   | 💡                 | 🏅                | 🏅       | 🏅          |
| [Use formatters and linting](linting.md)                                         | 💡                 | 🏅                | 🏅       | 🏅          |
| [Use typehinting](typehinting.md)                                                | 🐌                 | 💡                | 🏅       | 🏅          |
| [Add a README](add_a_readme.md)                                     | 🐌                 | 💡                | 🏅       | 🏅          |
| [Encapsulation, SRP](encapsulation.md)                                                                    | 🐌                 | 💡                | 🏅       | 🏅          |
| [Open-Closed Principle](open_closed.md)                                                            | 🐌                 | 💡                | 🏅       | 🏅          |
| Makefiles or shell scripts                                     | 🐌                 | 💡                | 🏅       | 🏅          |
| [Abstract classes (ABC, Protocol)](typehinting.md)                               | 🐌                 | 🐌                | 💡       | 🏅          |
| Write tests (pytest)                                                             | 🐌                 | 🐌                | 💡       | 🏅          |

## 1. Two cultures

The book "Machine Learning Engineering" describes two types of cultures when structuring a machine learning team:

- One culture says that a machine learning team has to be composed of data analysts who collaborate closely with software engineers. In such a culture, a software engineer doesn’t need to have deep expertise in machine learning, but has to understand the vocabulary of their fellow data analysts.

- According to other culture, all engineers in a machine learning team must have a combination of machine learning and software engineering skills.

There are pros and cons in each culture. The proponents of the former say that each team member must be the best in what they do. A data analyst must be an expert in many machine learning techniques and have a deep understanding of the theory to come up with an effective solution to most problems, fast and with minimal effort. Similarly, a software engineer must have a deep understanding of various computing frameworks and be capable
of writing efficient and maintainable code.

The proponents of the latter say that scientists are hard to integrate with software engineering teams. Scientists care more about how accurate their solution is and often come up with solutions that are impractical and cannot be effectively executed in the production environment. Also, because scientists don’t usually write efficient, well-structured code, the
latter has to be rewritten into production code by a software engineer; depending on the project, that can turn out to be a daunting task.

Because one of the goals of this course is make sure you are aligned with the current work practice, we created these guidelines, and hope to find a balance between on the one side the depth of datascience, and on the other side the robustness of software engineering to be able to build solid code where your teammates can build on.

## 2. The pros and cons of Pythons flexibility

While it is often appreciated that Python gives programmers a lot of freedom regarding coding style, this can be a huge factor that slows down cooperation between programmers, making it harder to understand, debug and extend code.

You might understand your code perfectly now, and you might be used to working like this for a long time. And while others might occasionally complain about your code, it is still working, right? So why change?

The problem is that you are not the only one that will be working on your code. A lot of code will need to be understood by multiple people. And even if you are working on your own, you probably have the experience of returning to an old codebase and wondering what it was that you were doing.

In general, there will always be exceptions to the rule. The rule of thumb is:

> follow the coding standards, unless there is a good reason not to.

E.g. there are reasons why `pdm` is a better environment manager than `pip`, but some could environments dont work well with `pdm`, so in that case falling back to `pip` could be a good idea.

## 3. Development stages

There is something like "using the right tool for the right problem". At some stages of development, some standards might even slow you down.

That is why we have defined four stages of development. Each stage has its own standards. The stages we identify are:

1. Testing a concept
1. Proof of concept
1. Product
1. Deployment

### Testing a concept

This is the stage where you are still figuring out how to solve a problem. You are not sure if the solution will work, or if it is even possible to solve the problem. You are still in the process of learning. This is typically done in a notebook or a single script file, and you will work on your own or maybe with one other person.

### Proof of concept

This is the stage where your prototyping has showed that the solution is possible. You are now trying to figure out how to implement the solution in a more robust way. You are still learning and exploring, but you are now working in a more structured way. This might be done in a single notebook, but you could also be setting up a small project with multiple .py files.

### Product

This is the stage where you will distribute the product to other people that were not part of writing the code. Maybe you want someone to review your code, or to test your product.

### Deployment

This is the stage where the code will be used in production. It is now important that the code is robust and well tested. You will be working in a team, and you will be working on a codebase that is already quite large.
