import openai
import argparse
import pickle
from tqdm import tqdm

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from api_key import api_key


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_response(text, max_tokens=256):
    response = openai.Completion.create(
        # model="text-curie-001",
        model="text-davinci-003",
        prompt=text,
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=5,
        # logit_bias={"17821":100, "25101":100, "198":100}
    )
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='atomic_samples.txt')
    parser.add_argument('--max_tokens', type=int, default=256)
    args = parser.parse_args()

    print(args)

    openai.api_key = api_key

    folder = "data/country_prediction/"

    with open(folder + args.file, 'r') as f:
        data = f.readlines()
    data = [d.replace('\n', '') for d in data]

    results = [None] * len(data)
    for i, sample in tqdm(enumerate(data)):
        results[i] = get_response(sample, max_tokens=args.max_tokens)
        with open("gpt3/country_prediction/{}.pkl".format(args.file[:-4]), 'wb') as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    main()
