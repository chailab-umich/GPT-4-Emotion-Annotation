# README

## Environment

To setup dev environment, run setup_env.sh from the top-level directory. Note that variables starting with `AZURE_OPENAI` in the setup_env.sh and several global variables in gpt_utils should be updated with your own version before running.

## Examples

`gpt_utils` supports

1.  classification and generation for GoEmotion, Semeval and ISEAR;
2.  generation and regression for EmoBank

using OpenAI Azure API. Prompts used for different domains and different tasks can be found (and customized) in gpt_utils.py. Example code that uses this library can be found in gpt_annotation_samples.ipynb.

