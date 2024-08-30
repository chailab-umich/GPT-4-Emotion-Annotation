import pandas as pd
import os
import time
from openai import AzureOpenAI
import krippendorff
import numpy as np

SEED = 42
# Current quota is 110K TPM, i.e., ~1800 tokens per second.
# If we generously count one request to be using 200 tokens, that means
# 9 requests per second. 5 QPS should be pretty safe.
PER_SAMPLE_DELAY_SECONDS = 0.2
RETRY_DELAY_SECONDS = 2

DEPLOYMENT_NAME_4 = 'gpt-4-sandy'
DEPLOYMENT_NAME_35 = 'gpt-35-turbo-sandy'

FAILED_ANNOTATION = 'FAILED_ANNOTATION_REJECTED'

FAILED_ANNOTATION_OOR = 'FAILED_ANNOTATION_OUT_OF_RETRY'

_SHOULD_NEVER_HAPPEN = 'THIS SHOULD NEVER HAPPEN'


def GetOpenAIClient(deployment_name):
    if deployment_name == DEPLOYMENT_NAME_4:
        return AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_4"),
                           api_key=os.getenv("AZURE_OPENAI_KEY_4"),
                           api_version=os.getenv("AZURE_OPENAI_API_VERSION_4"))
    if deployment_name == DEPLOYMENT_NAME_35:
        return AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_35"),
            api_key=os.getenv("AZURE_OPENAI_KEY_35"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION_35"))
    raise Exception("unknown deployment name")


def GetEmotionList(domain):
    if domain == 'go':
        with open('references/goemotions/data/emotions.txt', 'r') as f:
            all_emotions = f.read().splitlines()
    elif domain == 'semeval':
        all_emotions = [
            'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love',
            'optimism', 'pessimism', 'sadness', 'surprise', 'trust'
        ]
    elif domain == 'isear':
        all_emotions = [
            'joy', 'guilt', 'anger', 'disgust', 'fear', 'sadness', 'shame'
        ]
    else:
        raise Exception(f"unsupported domain {domain}")
    return all_emotions


SYSTEM_PROMPT_CANDIDATES_GOEMOTION = ' '.join([
    "You are an emotionally-intelligent and empathetic agent.",
    "You will be given a piece of text, and your task is to identify all possible",
    "emotions expressed by the writer of the text.",
    "You are only allowed to make selections from the following emotions: {}.",
    "neutral means no emotion is clearly expressed.",
    "Reply with only the list of emotions, seperated by comma."
]).format(', '.join(GetEmotionList('go')))

SYSTEM_PROMPT_CLASSIFY_GOEMOTION = ' '.join([
    "You are an emotionally-intelligent and empathetic agent.",
    "You will be given a piece of text, and your task is to identify all the emotions expressed",
    "by the writer of the text.",
    "You are only allowed to make selections from the following emotions, and don't use any other words: {}.",
    "Only select those ones for which you are reasonably confident that they are expressed in the text."
    "If no emotion is clearly expressed, select 'neutral'.",
    "Reply with only the list of emotions, seperated by comma."
]).format(', '.join(GetEmotionList('go')))

SYSTEM_PROMPT_CLASSIFY_SEMEVAL = ' '.join([
    "You are an emotionally-intelligent and empathetic agent.",
    "You will be given a piece of text, and your task is to identify all the emotions expressed",
    "by the writer of the text.",
    "You are only allowed to make selections from the following emotions, and don't use any other words: {}.",
    "Only select those ones for which you are reasonably confident that they are expressed in the text."
    "If no emotion is clearly expressed, reply with 'neutral'.",
    "Only provide the list of emotions, seperated by comma."
]).format(', '.join(GetEmotionList('semeval')))

SYSTEM_PROMPT_CLASSIFY_ISEAR = ' '.join([
    "You are an emotionally-intelligent and empathetic agent.",
    "You will be given a piece of text, and your task is to identify all the emotions expressed",
    "by the writer of the text.",
    "You are only allowed to make one selection from the following emotions, and don't use any other words: {}.",
]).format(', '.join(GetEmotionList('isear')))

SYSTEM_PROMPT_GENERATE_GOEMOTION = ' '.join([
    "You are an emotionally-intelligent and empathetic agent.",
    "You will be given a piece of text, and your task is to identify the emotions expressed",
    "by the writer of the text.",
    "Reply with only the emotion descriptors (words or phrases), separated by comma."
    "If no emotion is clearly expressed, reply with 'neutral'.",
])

SYSTEM_PROMPT_GENERATE_SEMEVAL = ' '.join([
    "You are an emotionally-intelligent and empathetic agent.",
    "You will be given a piece of text, and your task is to identify the emotions expressed",
    "by the writer of the text.",
    "Reply with only the emotion descriptors (words or phrases), separated by comma."
    "If no emotion is clearly expressed, reply with 'neutral'.",
])

SYSTEM_PROMPT_GENERATE_ISEAR = ' '.join([
    "You are an emotionally-intelligent and empathetic agent.",
    "You will be given a piece of text, and your task is to identify the emotions expressed",
    "by the writer of the text. Reply with only one noun."
])

SYSTEM_PROMPT_GENERATE_EMOBANK = ' '.join([
    "You are an emotionally-intelligent and empathetic agent.",
    "You will be given a piece of text, and your task is to identify the emotions expressed",
    "by the writer of the text.",
    "Reply with only the emotion descriptors (words or phrases), separated by comma."
    "If no emotion is clearly expressed, reply with 'neutral'.",
])

SYSTEM_PROMPT_REGRESSION_EMOBANK = ' '.join([
    "You are an emotionally-intelligent and empathetic agent.",
    "You will be given a piece of text, and your task is to rate how positive",
    "emotions expressed by the writer of the text is.",
    "Specifically, reply a number of emotional valence: from 1 to 5, how positive is the emotion? 1-very negative, 3-neutral, 5-very positive.",
    # "The second is emotional arousal: From 1 to 5, how excited is the emotion? 1-very calm, 3-normal, 5-very excited."
    #  "If no emotion is clearly expressed, reply with '3,3'.",
    # "Reply with two numbers, separated by a comma.",
    "Only return one number from 1 to 5."
])

USER_PROMPT_RETRY_INSTRUCTION = "Please only pick from the given options separated by comma."

USER_PROMPT_RETRY_INSTRUCTION_NUM = "Please pick all possible emotions from the given list, separated by comma."

USER_PROMPT_RETRY_INSTRUCTION_EMOBANK = "Please only return one number from 1 to 5."

USER_PROMPT_RETRY_INSTRUCTION_ISEAR = "Please only return one noun that describes the emotion."

ALLOWED_DOMAIN_TASK = {
    "semeval": ['classification', 'generation'],
    "goemotions": ['classification', 'generation'],
    "isear": ['classification', 'generation'],
    "emobank": ['generation', 'regression']
}


def get_dataset(filepath):
    return pd.read_csv(filepath, encoding='utf-8')


def GetDataframeFromDomain(domain):
    return get_dataset(f"data_clean/{domain}_500.csv")


def SanitizeGoEmotionsDataFrame(df):
    all_emotions = GetEmotionList('go')
    results = []
    labels = list(df['label'])
    for label in labels:
        emos = label.split(',')
        results.append([all_emotions[int(l)] for l in emos])

    human = [','.join(result) for result in results]
    df['human_label'] = human
    # df = df[['id', 'text', 'human_label']]
    return df


def SanitizeSemevalDataFrame(df):
    all_emotions = GetEmotionList('semeval')

    results = []
    for i, row in df.iterrows():
        tmp_result = []
        for emo in all_emotions:
            if row[emo] == 1:
                tmp_result.append(emo)
        results.append(','.join(tmp_result))

    df['human_label'] = results
    df.rename(columns={'Tweet': 'text', 'ID': 'id'}, inplace=True)
    df = df[['id', 'text', 'human_label']]
    return df


def SanitizeISEARDataFrame(df):
    df.rename(columns={'label': 'human_label'}, inplace=True)
    df = df[['id', 'text', 'human_label']]
    return df


def GetGptAnnotationWithRetries(df,
                                deployment_name,
                                system_prompt,
                                user_prompt_retry_instruction,
                                validate_func,
                                sanitize_func,
                                max_retries=2,
                                verbose=False):
    """Get GPT annotations for each row in `df`.

    Args:
        df (pd.DataFrame): `df` should have a column 'text'.
        deployment_name (str): either gpt-35-sandy or gpt-4-sandy.
        system_prompt (str): System prompt that will be provided to Gpt.
        user_prompt_retry_instruction (str): User prompt that will be given to Gpt if validation fails.
        validate_func (function(str) -> bool): A function that validates if the `anno` is good 
        sanitize_func (function(str) -> str): A function that sanitizes `anno` 
        max_retries (int, optional): Maximun num of retries allowed if the labels provided by Gpt cannot be validated. Defaults to 2.
        verbose (bool, optional): If true, verbose info will be printed out on console for debugging purposes. Defaults to False.

    Returns:
        List[str] or pd.DataFrame: A list of sanitized labels. If retries are exhausted, `FAILED_ANNOTATION_OOR` will be provided. If OpenAI refuses to provide completion, return
        `FAILED_ANNOTATION_REJECTED`.
    """
    openai_client = GetOpenAIClient(deployment_name)
    gpt_anno = []
    for index, row in df.iterrows():
        messages = [{
            "role": "system",
            "content": system_prompt
        }, {
            "role": "user",
            "content": row['text']
        }]
        for i in range(max_retries + 1):
            completion = None
            try:
                time.sleep(PER_SAMPLE_DELAY_SECONDS)
                completion = openai_client.chat.completions.create(
                    model=deployment_name, messages=messages, seed=SEED)
            except Exception as e:
                completion = None
                gpt_anno.append(FAILED_ANNOTATION)
                if verbose:
                    print(
                        f"\033[91mSample#{index}: Text is \'{row['text']}\'\n\tAttempt#{i+1} Failed  with {FAILED_ANNOTATION}, error {e}"
                    )
                break
            raw = completion.choices[0].message.content
            sanitized = sanitize_func(raw)
            pass_validation = validate_func(sanitized)

            if verbose:
                sample_prefix = f"\033[92mSample#{index}: Text is \'{row['text']}\'\n" if i == 0 else "\033[93m"
                state_prefix = '\033[92m' if pass_validation else '\033[91m'
                state_suffix = 'PASSED' if pass_validation else 'FAILED'
                print(
                    f"{sample_prefix}\tAttempt#{i+1}: raw ",
                    f"{state_prefix}<{raw}> sanitized to <{sanitized}> ({state_suffix})"
                )
            if pass_validation:
                gpt_anno.append(sanitized)
                break
            elif i == max_retries:
                # We fail the last attempt
                gpt_anno.append(FAILED_ANNOTATION_OOR)
                break
            else:
                if user_prompt_retry_instruction == _SHOULD_NEVER_HAPPEN:
                    raise Exception("Retry not expected")
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": user_prompt_retry_instruction
                })
                # Retrying..
                time.sleep(RETRY_DELAY_SECONDS)
    assert len(gpt_anno) == df.shape[0]
    return gpt_anno


def GetGptAnnotationWithRetriesForISEARClassification(df,
                                                      deployment_name,
                                                      max_retries=2,
                                                      verbose=False):

    def sanitize_func(raw):
        return raw.strip().strip('.').lower()

    def validate_func(anno):
        return sanitize_func(anno) in GetEmotionList('isear')

    return GetGptAnnotationWithRetries(df,
                                       deployment_name,
                                       SYSTEM_PROMPT_CLASSIFY_ISEAR,
                                       USER_PROMPT_RETRY_INSTRUCTION_ISEAR,
                                       validate_func,
                                       sanitize_func,
                                       max_retries=max_retries,
                                       verbose=verbose)


def GetGptAnnotationWithRetriesForISEARGenerate(df,
                                                deployment_name,
                                                max_retries=2,
                                                verbose=False):

    def sanitize_func(raw):
        return raw.strip().strip('.').lower()

    def validate_func(anno):
        return True

    return GetGptAnnotationWithRetries(df,
                                       deployment_name,
                                       SYSTEM_PROMPT_GENERATE_ISEAR,
                                       _SHOULD_NEVER_HAPPEN,
                                       validate_func,
                                       sanitize_func,
                                       max_retries=max_retries,
                                       verbose=verbose)


def GetGptAnnotationWithRetriesForGoEmotionsClassification(
        df, deployment_name, max_retries=2, verbose=False):

    def sanitize_func(raw):
        return ','.join(sorted([emo.strip().lower()
                                for emo in raw.split(',')]))

    def validate_func(anno):
        return all([
            emo in GetEmotionList('go')
            for emo in sanitize_func(anno).split(',')
        ])

    return GetGptAnnotationWithRetries(df,
                                       deployment_name,
                                       SYSTEM_PROMPT_CLASSIFY_GOEMOTION,
                                       USER_PROMPT_RETRY_INSTRUCTION,
                                       validate_func,
                                       sanitize_func,
                                       max_retries=max_retries,
                                       verbose=verbose)


def GetGptAnnotationWithRetriesForGoEmotionsCandidates(df,
                                                       deployment_name,
                                                       max_retries=2,
                                                       verbose=False):

    def sanitize_func(raw):
        return ','.join(sorted([emo.strip().lower()
                                for emo in raw.split(',')]))

    def validate_func(anno):
        sanitized_anno = sanitize_func(anno).split(',')
        return all([emo in GetEmotionList('go') for emo in sanitized_anno])

    return GetGptAnnotationWithRetries(df,
                                       deployment_name,
                                       SYSTEM_PROMPT_CANDIDATES_GOEMOTION,
                                       USER_PROMPT_RETRY_INSTRUCTION_NUM,
                                       validate_func,
                                       sanitize_func,
                                       max_retries=max_retries,
                                       verbose=verbose)


def GetGptAnnotationWithRetriesForGoEmotionsGenerate(df,
                                                     deployment_name,
                                                     max_retries=2,
                                                     verbose=False):

    def sanitize_func(raw):
        return ','.join(sorted([emo.strip().lower()
                                for emo in raw.split(',')]))

    def validate_func(anno):
        return True

    return GetGptAnnotationWithRetries(df,
                                       deployment_name,
                                       SYSTEM_PROMPT_GENERATE_GOEMOTION,
                                       _SHOULD_NEVER_HAPPEN,
                                       validate_func,
                                       sanitize_func,
                                       max_retries=max_retries,
                                       verbose=verbose)


def GetGptAnnotationWithRetriesForSemevalClassification(
        df, deployment_name, max_retries=2, verbose=False):

    def sanitize_func(raw):
        return ','.join(sorted([emo.strip().lower()
                                for emo in raw.split(',')]))

    def validate_func(anno):
        return all([
            emo in GetEmotionList('semeval') or emo == 'neutral'
            for emo in sanitize_func(anno).split(',')
        ])

    return GetGptAnnotationWithRetries(df,
                                       deployment_name,
                                       SYSTEM_PROMPT_CLASSIFY_SEMEVAL,
                                       USER_PROMPT_RETRY_INSTRUCTION,
                                       validate_func,
                                       sanitize_func,
                                       max_retries=max_retries,
                                       verbose=verbose)


def GetGptAnnotationWithRetriesForSemevalGenerate(df,
                                                  deployment_name,
                                                  max_retries=2,
                                                  verbose=False):

    def sanitize_func(raw):
        return ','.join(sorted([emo.strip().lower()
                                for emo in raw.split(',')]))

    def validate_func(anno):
        return True

    return GetGptAnnotationWithRetries(df,
                                       deployment_name,
                                       SYSTEM_PROMPT_GENERATE_SEMEVAL,
                                       _SHOULD_NEVER_HAPPEN,
                                       validate_func,
                                       sanitize_func,
                                       max_retries=max_retries,
                                       verbose=verbose)


def GetGptAnnotationWithRetriesForEmoBankRegression(df,
                                                    deployment_name,
                                                    max_retries=2,
                                                    verbose=False):

    def sanitize_func(raw):
        return raw.strip()

    def validate_func(anno):
        return sanitize_func(anno) in ['1', '2', '3', '4', '5']

    return GetGptAnnotationWithRetries(df,
                                       deployment_name,
                                       SYSTEM_PROMPT_REGRESSION_EMOBANK,
                                       USER_PROMPT_RETRY_INSTRUCTION_EMOBANK,
                                       validate_func,
                                       sanitize_func,
                                       max_retries=max_retries,
                                       verbose=verbose)


def GetGptAnnotationWithRetriesForEmoBankGenerate(df,
                                                  deployment_name,
                                                  max_retries=2,
                                                  verbose=False):

    def sanitize_func(raw):
        return raw.strip().strip('.').lower()

    def validate_func(anno):
        return True

    return GetGptAnnotationWithRetries(df,
                                       deployment_name,
                                       SYSTEM_PROMPT_GENERATE_EMOBANK,
                                       _SHOULD_NEVER_HAPPEN,
                                       validate_func,
                                       sanitize_func,
                                       max_retries=max_retries,
                                       verbose=verbose)


def GetGptAnnotation(domain,
                     task,
                     deployment_name=DEPLOYMENT_NAME_4,
                     max_retries=5,
                     verbose=False):
    assert domain in ['semeval', 'goemotions', 'emobank', 'isear']
    assert task in ['classification', 'generation', 'regression']
    raw_df = GetDataframeFromDomain(domain)
    if domain == 'isear':
        sanitized_df = SanitizeISEARDataFrame(raw_df)
        if task == 'classification':
            return GetGptAnnotationWithRetriesForISEARClassification(
                sanitized_df,
                deployment_name,
                max_retries=max_retries,
                verbose=verbose)
        elif task == 'generation':
            return GetGptAnnotationWithRetriesForISEARGenerate(
                sanitized_df,
                deployment_name,
                max_retries=max_retries,
                verbose=verbose)
        else:
            raise Exception("Unimplemented")
    elif domain == 'semeval':
        sanitized_df = SanitizeSemevalDataFrame(raw_df)
        if task == 'classification':
            return GetGptAnnotationWithRetriesForSemevalClassification(
                sanitized_df,
                deployment_name,
                max_retries=max_retries,
                verbose=verbose)
        elif task == 'generation':
            return GetGptAnnotationWithRetriesForSemevalGenerate(
                sanitized_df,
                deployment_name,
                max_retries=max_retries,
                verbose=verbose)
        else:
            raise Exception("Unimplemented")
    elif domain == 'goemotions':
        sanitized_df = SanitizeGoEmotionsDataFrame(raw_df)
        if task == 'classification':
            return GetGptAnnotationWithRetriesForGoEmotionsClassification(
                sanitized_df,
                deployment_name,
                max_retries=max_retries,
                verbose=verbose)
        elif task == 'generation':
            return GetGptAnnotationWithRetriesForGoEmotionsGenerate(
                sanitized_df,
                deployment_name,
                max_retries=max_retries,
                verbose=verbose)
        else:
            raise Exception("Unimplemented")
    elif domain == 'emobank':
        if task == 'regression':
            return GetGptAnnotationWithRetriesForEmoBankRegression(
                raw_df,
                deployment_name,
                max_retries=max_retries,
                verbose=verbose)
        elif task == 'generation':
            return GetGptAnnotationWithRetriesForEmoBankGenerate(
                raw_df,
                deployment_name,
                max_retries=max_retries,
                verbose=verbose)
        else:
            raise Exception("Unimplemented")
