# Code of Conduct

Solstice is written entirely by frontier AI models. People direct the work, review it and decide
what is merged; they do not write it. This document sets out that rule and how everyone working
on the project is expected to behave.

## 1. Everything is written by frontier AI models

All of the project's written work must be produced by a frontier AI model. This includes:

- **Code**: source, shaders, tests, scripts, build and CI configuration.
- **Pull requests**: their commits, titles and descriptions.
- **Issues**: reports, feature requests and their descriptions.
- **Comments and reviews**: on issues, pull requests and commits, and comments in the code itself.
- **Documentation**: the README, this file and everything under `docs/`.
- **Assets made from code**: scene descriptions, species, grass and weather files, and procedural
  content.

No human-written code is accepted, whether a whole change or a single line typed into a model's
work.

A **frontier AI model** is a current, leading general-purpose model from a major AI lab, at the
time the work is done, used directly or through an agent such as Claude Code.

## 2. What people do

People take part as directors and reviewers. They may:

- Describe what they want, in their own words, as instructions to a model.
- Share screenshots, measurements, references and other material for a model to work from.
- Review a model's work, ask for changes, and approve, merge or close pull requests.
- Report a problem to a model, which then writes the issue.

They may not write or edit code, pull request text, issues or comments themselves, nor paste in
code from elsewhere as their own contribution.

Third-party material that the project uses, such as scanned models, textures and libraries, keeps
its own authorship and licence, and is brought in by a model, with its source and licence stated.

## 3. Disclosure

- Every pull request names the model, or models, that wrote it.
- Commits carry a `Co-Authored-By` line naming the model.
- Issues and comments made through an agent say so where the tool allows it.

## 4. Quality

Work by a model is held to the same standard as any work: it must build, pass the tests and lints,
and be checked by running it, with measurements where performance or image quality is at stake.
Changes that cannot be verified are not merged.

## 5. Conduct

Everyone involved, people and models, is expected to:

- Be respectful, constructive and honest, including about what has and has not been verified.
- Credit sources, and respect the licences of everything the project uses.
- Stay focused on making the project better.

Harassment, personal attacks, discrimination and deliberate misrepresentation are not accepted.

## 6. Enforcement

Pull requests, issues and comments that break this code are closed or removed. The maintainer,
[@SamG-Coder](https://github.com/SamG-Coder), decides on any question about it, and may block
anyone who repeatedly breaks it.
